from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import FoundationConfig
from app.agents import HermesConversationBackend, HermesTeamConversationBackend
from app.gateway.conversation_worker import ConversationQueue, ConversationWorker
from app.gateway.repository_analysis_worker import (
    RepositoryAnalysisQueue,
    RepositoryAnalysisWorker,
)
from app.gateway.outbound_worker import ActivityWorker, OutboundWorker
from app.gateway.adapters.telegram import TelegramAdapter, TelegramClient
from app.gateway.config import GatewaySettings
from app.gateway.core import (
    AccessPolicy,
    DialogueRouter,
    GatewayApplication,
    GatewayRunner,
    GatewayWorkerStopped,
    GovernedAgentBackend,
    GovernedTeamConversationBackend,
    LocalGitRepositoryValidator,
)
from app.gateway.core.application import should_hydrate_inbound
from app.orchestrator import RunStateMachine
from app.pipeline import PipelineCoordinator, PipelineWorker
from app.services.context import ContextPolicy, ContextService
from app.services.attachments import AttachmentPolicy, TelegramAttachmentService
from app.services.budget import BudgetManager, BudgetPolicy
from app.services.logging.audit import AuditLogger
from app.services.logging.gateway import GatewayLog, GatewayLogPolicy
from app.services.hermes import HermesRunner, HermesSettings
from app.services.process_tree import validate_process_tree_support
from app.services.repository import SafeRepositoryReader, SafeRepositoryToolLayer
from app.services.sandbox import DockerSandbox
from app.services.toolchains import ToolchainService
from app.services.verification import VerificationRunner
from app.storage import ArtifactStore, StateStore


@dataclass(frozen=True)
class TelegramRuntime:
    settings: GatewaySettings
    client: TelegramClient
    adapter: TelegramAdapter
    runner: GatewayRunner
    pipeline_worker: PipelineWorker
    conversation_worker: ConversationWorker
    repository_analysis_worker: RepositoryAnalysisWorker
    hermes_settings: HermesSettings
    hermes_runner: HermesRunner
    outbound_worker: OutboundWorker
    activity_worker: ActivityWorker
    sandbox: DockerSandbox


def build_telegram_runtime(
    root: Path,
    backend=None,
    *,
    team_backend=None,
    role_runner=None,
    verifier=None,
) -> TelegramRuntime:
    foundation = FoundationConfig.load(root)
    validate_process_tree_support()
    settings = GatewaySettings.load(root)
    settings.validate()

    store = StateStore(foundation.database)
    logger = AuditLogger(foundation.artifacts, store)
    machine = RunStateMachine(
        store, event_sink=logger, approval_phrase=foundation.approval_phrase
    )
    context = ContextService(
        store, policy=ContextPolicy.load(root / "config" / "limits.json")
    )
    artifacts = ArtifactStore(foundation.artifacts)
    budget = BudgetManager(BudgetPolicy.load(root / "config" / "limits.json"), store)
    gateway_log = GatewayLog(
        root / "logs" / "gateway.log",
        policy=GatewayLogPolicy(
            max_bytes=settings.logging.gateway_max_bytes,
            backup_count=settings.logging.gateway_backup_count,
        ),
    )
    sandbox = DockerSandbox(event_sink=gateway_log.write)
    toolchains = ToolchainService.load(
        root / "config" / "toolchains.json", sandbox=sandbox
    )
    gateway_log.event(
        "RUNTIME_READY",
        "게이트웨이 런타임 구성을 완료했습니다.",
        log_max_bytes=settings.logging.gateway_max_bytes,
        log_backup_count=settings.logging.gateway_backup_count,
        budget_calibration=budget.policy.calibration_mode,
    )
    client = TelegramClient(settings.telegram.token)
    attachment_service = TelegramAttachmentService(
        client,
        root / "data" / "attachments",
        AttachmentPolicy(
            max_bytes=settings.attachments.max_bytes,
            max_text_characters=settings.attachments.max_text_characters,
        ),
    )
    policy = AccessPolicy(
        allowed_users=settings.telegram.allowed_users,
        allowed_group_conversations=settings.telegram.allowed_chats,
        private_only=settings.telegram.private_only,
    )

    def accept_inbound_before_attachment(message) -> bool:
        return should_hydrate_inbound(
            store, policy, message,
            max_processing_attempts=settings.max_processing_attempts,
        )

    adapter = TelegramAdapter(
        client,
        poll_timeout_seconds=settings.telegram.poll_timeout_seconds,
        max_message_characters=settings.telegram.max_message_characters,
        attachment_processor=attachment_service.hydrate_message,
        inbound_filter=accept_inbound_before_attachment,
    )
    conversation_queue = ConversationQueue(store)
    repository_analysis_queue = RepositoryAnalysisQueue(store)
    hermes_settings = HermesSettings.load(root)
    hermes_runner = HermesRunner(hermes_settings, artifacts)
    coordinator = PipelineCoordinator(
        store,
        machine,
        foundation,
        artifacts,
        logger,
        budget,
        role_runner or hermes_runner,
        verifier or VerificationRunner(sandbox=sandbox),
        sandbox=sandbox,
        toolchains=toolchains,
    )
    activity_worker = ActivityWorker(
        lambda channel, conversation_id: (
            client.send_chat_action(conversation_id)
            if channel == adapter.name
            else None
        ),
        error_sink=gateway_log.write,
    )
    pipeline_worker = PipelineWorker(
        store,
        machine,
        coordinator,
        logger,
        activity_notifier=activity_worker.notify,
    )
    governed_backend = GovernedAgentBackend(
        backend or HermesConversationBackend(foundation, hermes_runner, sandbox=sandbox), budget
    )
    governed_team_backend = GovernedTeamConversationBackend(
        team_backend or HermesTeamConversationBackend(foundation, hermes_runner), budget
    )
    repository_reader = SafeRepositoryReader(sandbox=sandbox)
    router = DialogueRouter(
        store=store,
        state_machine=machine,
        context=context,
        artifacts=artifacts,
        logger=logger,
        backend=governed_backend,
        repository_validator=LocalGitRepositoryValidator(sandbox=sandbox),
        repository_reader=repository_reader,
        repository_tools=SafeRepositoryToolLayer(
            redactor=repository_reader.redactor, sandbox=sandbox
        ),
        team_backend=governed_team_backend,
        repository_approval_phrase=foundation.repository_approval_phrase,
        repository_approval_ttl_hours=foundation.repository_approval_ttl_hours,
        pending_project_request_ttl_hours=foundation.pending_project_request_ttl_hours,
        pipeline_scheduler=pipeline_worker,
        conversation_scheduler=conversation_queue,
        role_names={
            role_id.value: role.display_name
            for role_id, role in foundation.roles.items()
        },
        max_auto_agent_replies=settings.conversation.max_auto_agent_replies,
        group_parallel_workers=settings.conversation.group_parallel_workers,
        budget=budget,
        execution_preflight=toolchains,
        repository_analysis_scheduler=repository_analysis_queue,
        repository_analysis_limits={
            "max_files_per_batch": settings.repository_analysis.max_files_per_batch,
            "max_file_bytes": settings.repository_analysis.max_file_bytes,
            "max_batch_bytes": settings.repository_analysis.max_batch_bytes,
            "max_batches": settings.repository_analysis.max_batches,
            "max_read_bytes": settings.repository_analysis.max_read_bytes,
        },
    )
    application = GatewayApplication(
        store,
        router,
        policy,
        max_processing_attempts=settings.max_processing_attempts,
        message_preparer=adapter.prepare,
        diagnostic_sink=gateway_log.write,
        conversation_scheduler=conversation_queue,
    )
    conversation_worker = ConversationWorker(
        store,
        router,
        message_preparer=adapter.prepare,
        error_sink=gateway_log.write,
        activity_notifier=activity_worker.notify,
        worker_count=settings.conversation.worker_count,
        poll_seconds=settings.conversation.worker_poll_seconds,
        lease_seconds=settings.conversation.lease_seconds,
        max_job_attempts=settings.conversation.max_job_attempts,
    )
    repository_analysis_worker = RepositoryAnalysisWorker(
        store,
        repository_reader,
        governed_team_backend,
        context,
        worker_count=settings.repository_analysis.worker_count,
        poll_seconds=settings.repository_analysis.worker_poll_seconds,
        lease_seconds=settings.repository_analysis.lease_seconds,
        max_files_per_batch=settings.repository_analysis.max_files_per_batch,
        max_file_bytes=settings.repository_analysis.max_file_bytes,
        max_batch_bytes=settings.repository_analysis.max_batch_bytes,
        max_batches=settings.repository_analysis.max_batches,
        max_query_rounds=settings.repository_analysis.max_query_rounds,
        max_read_bytes=settings.repository_analysis.max_read_bytes,
        max_no_progress=settings.repository_analysis.max_no_progress,
        max_elapsed_seconds=settings.repository_analysis.max_elapsed_seconds,
        error_sink=gateway_log.write,
    )
    gateway_runner = GatewayRunner(
        adapter,
        application,
        error_sink=gateway_log.write,
        health_check=lambda: _require_workers_alive(
            conversation_worker, repository_analysis_worker, pipeline_worker, outbound_worker, activity_worker
        ),
        connection_retry_limit=settings.connection_retry_limit,
        retry_max_seconds=settings.retry_max_seconds,
    )
    outbound_worker = OutboundWorker(
        gateway_runner.flush_outbound,
        error_sink=gateway_log.write,
    )
    conversation_worker.set_outbound_notifier(outbound_worker.notify)
    repository_analysis_worker.set_outbound_notifier(outbound_worker.notify)
    pipeline_worker.set_outbound_notifier(outbound_worker.notify)
    coordinator.set_outbound_notifier(outbound_worker.notify)
    router.set_progress_notifier(outbound_worker.notify)

    def notify_budget_warnings(warnings, run_id: str) -> None:
        targets = store.conversation_targets_for_run(run_id)
        queued = store.queue_budget_warning_delivery(
            run_id,
            tuple(
                {
                    "scope": warning.scope,
                    "stage_id": warning.stage_id,
                    "threshold_percent": warning.threshold_percent,
                    "text": BudgetManager.render_warning(warning),
                }
                for warning in warnings
            ),
            targets,
        )
        if queued:
            outbound_worker.notify()

    budget.set_warning_sink(notify_budget_warnings)
    budget.retry_pending_warnings()
    return TelegramRuntime(
        settings=settings,
        client=client,
        adapter=adapter,
        runner=gateway_runner,
        pipeline_worker=pipeline_worker,
        conversation_worker=conversation_worker,
        repository_analysis_worker=repository_analysis_worker,
        hermes_settings=hermes_settings,
        hermes_runner=hermes_runner,
        outbound_worker=outbound_worker,
        activity_worker=activity_worker,
        sandbox=sandbox,
    )


def _require_workers_alive(
    conversation_worker: ConversationWorker,
    repository_analysis_worker: RepositoryAnalysisWorker,
    pipeline_worker: PipelineWorker,
    outbound_worker: OutboundWorker,
    activity_worker: ActivityWorker,
) -> None:
    stopped = []
    if not conversation_worker.is_alive():
        stopped.append("conversation")
    if not repository_analysis_worker.is_alive():
        stopped.append("repository-analysis")
    if not pipeline_worker.is_alive():
        stopped.append("pipeline")
    if not outbound_worker.is_alive():
        stopped.append("outbound")
    if not activity_worker.is_alive():
        stopped.append("activity")
    if stopped:
        raise GatewayWorkerStopped(
            f"필수 백그라운드 워커가 종료됐습니다: {', '.join(stopped)}"
        )
