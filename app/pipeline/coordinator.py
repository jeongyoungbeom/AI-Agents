from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol
from dataclasses import replace
import hashlib
import json

from app.agents.parsing import (
    InvalidAgentResponse,
    ReviewSummary,
    RoleSummary,
    parse_review_summary,
    parse_role_summary,
)
from app.agents.prompts import (
    development_prompt,
    development_rework_prompt,
    improvement_prompt,
    review_prompt,
)
from app.config import FoundationConfig
from app.contracts import AgentHandoff, RoleId, RunPhase, StageContract, TokenUsage
from app.orchestrator import RunStateMachine
from app.services.budget import BudgetExceeded, BudgetManager, conservative_prompt_tokens
from app.services.git import (
    GitRepository,
    GitRepositoryCancelled,
    GitRepositoryError,
    IsolatedGitWorktree,
)
from app.services.hermes import HermesCancelled, HermesExecutionError, HermesResult
from app.services.logging.audit import AuditLogger
from app.services.logging.redaction import SecretRedactor
from app.services.repository import (
    RepositoryAccessError,
    RepositoryCancelled,
    inspect_repository_identity,
)
from app.services.sandbox import DockerSandbox
from app.services.toolchains import ToolchainEnvironment, ToolchainService
from app.services.verification import (
    VerificationCancelled,
    VerificationResult,
    VerificationRunner,
    VerificationFailureKind,
)
from app.storage import ArtifactStore, StateStore
from app.storage.sqlite_store import StoreError
from app.services.git.repository import GitSnapshot


class RoleRunner(Protocol):
    def run(
        self,
        run_id: str,
        stage_id: str,
        role_id: RoleId,
        repository: Path,
        prompt: str,
        *,
        allow_writes: bool,
        max_output_tokens: int | None = None,
        cancelled: Callable[[], bool] | None = None,
        heartbeat: Callable[[], None] | None = None,
    ) -> HermesResult: ...


class PipelineCancelled(RuntimeError):
    pass


class PipelineNeedsAttention(RuntimeError):
    pass


class PipelineSteeringPending(PipelineNeedsAttention):
    """수신한 중간 메시지의 해석/권한 확인을 기다리는 안전 경계."""


class PipelineSteeringChanged(RuntimeError):
    """파일·비용은 보존하고 이전 목표의 후속 처리를 다시 판단한다."""


class PipelineUserInputRequired(PipelineNeedsAttention):
    """A role cannot safely continue until the task owner answers a question."""

    def __init__(
        self, stage_id: str, role_id: RoleId, questions: tuple[str, ...]
    ) -> None:
        self.stage_id = stage_id
        self.role_id = role_id
        self.questions = questions
        super().__init__("사용자 결정이 필요합니다: " + " / ".join(questions))


class PipelineCoordinator:
    def __init__(
        self,
        store: StateStore,
        machine: RunStateMachine,
        foundation: FoundationConfig,
        artifacts: ArtifactStore,
        logger: AuditLogger,
        budget: BudgetManager,
        runner: RoleRunner,
        verifier: VerificationRunner,
        *,
        sandbox: DockerSandbox | None = None,
        toolchains: ToolchainService | None = None,
        on_outbound: Callable[[], None] | None = None,
    ):
        self.store = store
        self.machine = machine
        self.foundation = foundation
        self.artifacts = artifacts
        self.logger = logger
        self.budget = budget
        self.runner = runner
        self.verifier = verifier
        self.sandbox = sandbox or DockerSandbox()
        self.toolchains = toolchains
        self.on_outbound = on_outbound or (lambda: None)
        self.redactor = SecretRedactor()
        self._execution_environments: dict[str, ToolchainEnvironment | None] = {}

    def set_outbound_notifier(self, notifier: Callable[[], None]) -> None:
        self.on_outbound = notifier

    def execute(
        self,
        run_id: str,
        channel: str,
        conversation_id: str,
        lease_owner: str,
        *,
        cancelled: Callable[[], bool],
        heartbeat: Callable[[], None],
    ) -> None:
        state = self.store.load_run(run_id)
        if state.phase == RunPhase.COMPLETED:
            self._complete_notice(state, channel, conversation_id, lease_owner)
            return
        self._validate_approval(state, cancelled=cancelled)
        session = self.store.load_conversation_session(channel, conversation_id)
        if session is None or not self.store.repository_is_approved(
            channel,
            conversation_id,
            session.user_id,
            state.repository,
            state.repository_identity,
        ):
            raise PipelineNeedsAttention(
                "프로젝트 사용 승인이 없거나 만료되어 다시 승인이 필요합니다."
            )
        plan_record = self.store.load_plan_revision(run_id, state.plan_revision)
        if (
            plan_record is None
            or plan_record["status"] != "APPROVED"
            or plan_record["plan_hash"] != state.approved_plan_hash
        ):
            raise PipelineNeedsAttention("승인된 계획 원본을 확인할 수 없습니다.")
        raw_stages = plan_record["plan"].get("stages", [])
        try:
            contracts = tuple(StageContract.from_dict(item) for item in raw_stages)
        except (KeyError, TypeError, ValueError) as exc:
            raise PipelineNeedsAttention(
                "승인된 계획에 안전한 저장소 상대 수정 범위가 없습니다. 계획을 다시 확인해 주세요."
            ) from exc
        if len(contracts) != state.stage_count:
            raise PipelineNeedsAttention("승인된 계획의 단계 수가 현재 상태와 다릅니다.")
        for contract in contracts:
            for command in contract.verification_commands:
                self.verifier.policy.prepare(command)

        if state.phase == RunPhase.WAITING_APPROVAL:
            state = self.machine.transition(
                state, RunPhase.DEVELOPING, message="승인된 개발 파이프라인을 시작합니다."
            )
        if state.phase not in {RunPhase.DEVELOPING, RunPhase.COMPLETED}:
            raise PipelineNeedsAttention(f"재개할 수 없는 실행 상태입니다: {state.phase.value}")

        lock_acquired = self.store.acquire_repository_lock(
            state.repository, run_id, lease_owner
        )
        if not lock_acquired:
            raise PipelineNeedsAttention("같은 저장소에서 다른 작업이 실행 중입니다.")

        def pulse() -> None:
            heartbeat()
            self.store.renew_repository_lock(state.repository, run_id, lease_owner)

        completed = False
        worktree = None
        try:
            pulse()
            worktree, repository = self._open_workspace(state, contracts, lease_owner, cancelled)
            source_repository = worktree.source
            source_snapshot = worktree.source_snapshot
            self.artifacts.write_json(
                run_id,
                Path("worktree.json"),
                {"status": "opened", **worktree.record.to_dict()},
            )
            environment = self._preflight_toolchain(
                repository, contracts, run_id, cancelled
            )
            while True:
                self._check_cancel(cancelled)
                pulse()
                state = self.store.load_run(run_id)
                checkpoint = self.store.pipeline_workspace(run_id)
                if state.phase != RunPhase.COMPLETED:
                    contract = contracts[state.stage_index]
                    self.artifacts.save_contract(contract)
                    try:
                        self._steering_boundary(repository)
                        if checkpoint['status'] in {'stage_verified', 'applying'}:
                            self._finish_verified_stage(
                                state, repository, source_repository, source_snapshot,
                                contracts, pulse,
                            )
                        else:
                            self._run_stage(
                                state, contract, repository, channel, conversation_id,
                                cancelled, pulse, environment, source_repository,
                                source_snapshot, contracts,
                            )
                    except PipelineSteeringChanged:
                        current = self.store.load_run(run_id)
                        if current.phase != RunPhase.DEVELOPING:
                            current = self.machine.pause(current, '중간 지시 반영을 위해 현재 변경을 보존했습니다.')
                            self.machine.transition(current, RunPhase.DEVELOPING, message='새 지시로 동일 단계에서 계속합니다.')
                        continue
                state = self.store.load_run(run_id)
                if state.phase == RunPhase.COMPLETED:
                    completed = True
                    cleanup_error = worktree.cleanup()
                    worktree.release_execution_guard()
                    worktree._remove_empty_root()
                    self.artifacts.write_json(
                        run_id,
                        Path("worktree.json"),
                        {
                            "status": "cleaned" if cleanup_error is None else "cleanup_failed",
                            "cleanup_error": cleanup_error or "",
                            **worktree.record.to_dict(),
                        },
                    )
                    if cleanup_error:
                        self.logger.emit(
                            run_id,
                            "WORKTREE_CLEANUP_FAILED",
                            "원본 반영은 완료됐지만 gateway-owned 임시 worktree 정리에 실패했습니다.",
                            status="COMPLETED",
                            data={"worktree_path": str(worktree.path)},
                        )
                    self._complete_notice(state, channel, conversation_id, lease_owner)
                    self.logger.write_summary(state)
                    return
                if state.phase != RunPhase.DEVELOPING:
                    raise PipelineNeedsAttention(
                        f"단계 완료 후 상태가 올바르지 않습니다: {state.phase.value}"
                    )
        except (
            HermesCancelled,
            VerificationCancelled,
            GitRepositoryCancelled,
            RepositoryCancelled,
        ) as exc:
            raise PipelineCancelled(str(exc)) from exc
        finally:
            if not completed and worktree is not None:
                self._record_worktree_recovery(run_id, worktree)
            if worktree is not None:
                worktree.release_execution_guard()
            self.store.release_repository_lock(state.repository, run_id, lease_owner)

    def _steering_boundary(self, repository):
        self._check_cancel(repository.cancelled or (lambda: False))
        run_id = repository.operation_id
        with self.store.transaction():
            inputs = self.store.execution_inputs(run_id)
            if any(item['status'] in {'RECEIVED', 'WAITING_APPROVAL'} for item in inputs):
                raise PipelineSteeringPending('STEERING_PENDING')
            ready = self.store.apply_execution_inputs(run_id, repository.pipeline_owner)
            if not ready:
                return
            current = repository.snapshot()
            self._save_workspace(repository, status='ready', validated_sha='', operation='', pending_commit=None,
                                 candidate_sha=current.head, dirty_fingerprint=self._dirty_fingerprint(repository) if current.status else '')
            job = self.store.pipeline_job(run_id)
            for item in ready:
                self.store.append_message(run_id, item['user_id'], 'execution_directive', item['text'],
                    {'input_id': item['input_id'], 'source_message_id': item['message_id'], 'intent': item['intent'], 'source': 'user'})
                self.store.queue_outbound(job['channel'], job['conversation_id'],
                    f"중간 지시 {item['input_id']}를 안전 지점에서 반영했습니다. 현재 변경과 작업 공간을 보존합니다.", reply_to=item['message_id'])
        raise PipelineSteeringChanged()

    def _test_first(self, state, contract, repository, cancelled, heartbeat, environment):
        """새 구현 없이 승인된 검증만 실행한다. 이전 완료 조건을 임의로 변경하지 않는다."""
        self._steering_boundary(repository)
        results = self._run_verification_safely(state.run_id, contract, repository, cancelled, heartbeat,
                                               operation='steering-test-first', environment=environment)
        self._steering_boundary(repository)
        self.artifacts.write_json(state.run_id, Path('stages') / contract.stage_id / 'steering-tests.json',
                                  {'commands': [item.to_dict() for item in results], 'code_repair_attempted': False})
        current = repository.snapshot()
        self._save_workspace(repository, candidate_sha=current.head,
                             dirty_fingerprint=self._dirty_fingerprint(repository) if current.status else '')
        detail = ' / '.join(f'{item.command}: {item.failure_kind}, exit={item.return_code}' for item in results)
        raise PipelineUserInputRequired(contract.stage_id, RoleId.DEVELOPMENT, (
            '테스트부터 실행했습니다. ' + detail + '. 기존 변경은 보존했습니다. 제외할 기능과 현재 변경의 충돌을 확인하고, 승인 범위에서 다음 방향을 답해 주세요.',
        ))

    def _complete_notice(self, state, channel, conversation_id, lease_owner) -> None:
        # COMPLETED + completed checkpoint were committed with the final stage.
        # Queue recovery consumes that fact without reopening execution resources.
        with self.store.transaction():
            self.store.assert_pipeline_owner(state.run_id, lease_owner)
            checkpoint = self.store.pipeline_workspace(state.run_id)
            if not (checkpoint and checkpoint['status'] == 'completed'
                    and checkpoint['stage_index'] == state.stage_index
                    and checkpoint['plan_hash'] == state.approved_plan_hash
                    and checkpoint['repository_identity'] == state.repository_identity
                    and checkpoint['candidate_sha'] == checkpoint['validated_sha']
                    and not checkpoint['operation'] and not checkpoint['dirty_fingerprint']):
                raise PipelineNeedsAttention('확정 완료 checkpoint를 확인할 수 없습니다.')
            if not checkpoint.get('completion_notified'):
                self._set_role_for_run(state.run_id, channel, conversation_id, RoleId.DEVELOPMENT.value)
                self._notify_for_run(state.run_id, channel, conversation_id, '모든 단계를 완료했습니다.')
                checkpoint['completion_notified'] = True
                self.store.save_pipeline_workspace(state.run_id, lease_owner, checkpoint)

    @staticmethod
    def _dirty_fingerprint(repository: GitRepository) -> str:
        repository.assert_safe_worktree_paths()
        snapshot = repository.snapshot()
        values = [snapshot.head, snapshot.branch, snapshot.status]
        for name in repository.worktree_files():
            path = repository.path / name
            digest = hashlib.sha256()
            if path.is_symlink():
                value = str(path.readlink())
            elif path.is_file():
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(block)
                value = digest.hexdigest()
            elif not path.exists():
                value = 'deleted'
            else:
                raise PipelineNeedsAttention('미커밋 변경의 파일 유형을 확인할 수 없습니다.')
            values.append([name, value])
        return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode('utf-8')).hexdigest()

    def _save_workspace(self, repository: GitRepository, **updates) -> dict:
        checkpoint = self.store.pipeline_workspace(repository.operation_id)
        checkpoint.update(updates)
        self.store.save_pipeline_workspace(
            repository.operation_id, repository.pipeline_owner, checkpoint,
        )
        self.store.after_commit(lambda: self.artifacts.write_json(
            repository.operation_id, Path('pipeline-checkpoint.json'), checkpoint,
        ))
        return checkpoint

    def _open_workspace(self, state, contracts, lease_owner, cancelled):
        worktree = None
        try:
            checkpoint = self.store.pipeline_workspace(state.run_id)
            if checkpoint is None:
                if state.stage_index != 0 or state.phase == RunPhase.COMPLETED:
                    raise PipelineNeedsAttention('저장된 작업 공간 checkpoint가 없습니다. 기존 복구 기록을 확인해 주세요.')
                worktree = IsolatedGitWorktree(
                    Path(state.repository), self.sandbox, run_id=state.run_id, cancelled=cancelled,
                )
                worktree.acquire_execution_guard()
                # Keep dirty-source evidence even when begin fails before a checkpoint exists.
                try:
                    _source, snapshot = worktree.begin()
                    checkpoint = {
                        'run_id': state.run_id, 'plan_hash': state.approved_plan_hash,
                        'repository_identity': state.repository_identity,
                        'stage_index': 0, 'stage_base_sha': snapshot.head,
                        'candidate_sha': snapshot.head, 'validated_sha': snapshot.head,
                        'status': 'ready', 'operation': '', 'dirty_fingerprint': '',
                        'worktree': worktree.record.to_dict(),
                    }
                    self.store.save_pipeline_workspace(state.run_id, lease_owner, checkpoint)
                    repository = worktree.create(snapshot)
                except BaseException:
                    self._record_worktree_recovery(state.run_id, worktree)
                    raise
            else:
                if (checkpoint['run_id'] != state.run_id
                        or checkpoint['plan_hash'] != state.approved_plan_hash
                        or checkpoint['repository_identity'] != state.repository_identity
                        or checkpoint['stage_index'] != state.stage_index
                        or checkpoint['worktree']['source_snapshot']['head'] != state.repository_head_sha):
                    raise PipelineNeedsAttention('stage/승인/원본과 checkpoint가 일치하지 않습니다. pipeline-checkpoint.json을 확인해 주세요.')
                worktree = IsolatedGitWorktree(
                    Path(state.repository), self.sandbox, run_id=state.run_id,
                    cancelled=cancelled, record=checkpoint['worktree'],
                )
                worktree.acquire_execution_guard()
                if worktree.path.exists():
                    repository = worktree.attach()
                else:
                    if checkpoint['dirty_fingerprint'] or checkpoint['operation']:
                        raise PipelineNeedsAttention(
                            '작업 공간이 없어 미커밋/진행 중 변경을 확인할 수 없습니다. '
                            f"보존 위치: {worktree.path}; candidate: {checkpoint['candidate_sha']}",
                        )
                    # A fresh owned checkout starts at the durable candidate, never original HEAD.
                    worktree.release_execution_guard()
                    worktree = IsolatedGitWorktree(
                        Path(state.repository), self.sandbox, run_id=state.run_id, cancelled=cancelled,
                    )
                    worktree.acquire_execution_guard()
                    saved = checkpoint['worktree']['source_snapshot']
                    worktree.source_snapshot = GitSnapshot(saved['head'], saved['branch'], saved['status'], tuple(saved['untracked_files']))
                    repository = worktree.create(worktree.source_snapshot, candidate_sha=checkpoint['candidate_sha'])
                    checkpoint['worktree'] = worktree.record.to_dict()
                    self.store.save_pipeline_workspace(state.run_id, lease_owner, checkpoint)
                    self.logger.emit(state.run_id, 'WORKSPACE_REBUILT', '보존된 candidate에서 작업 공간을 복구했습니다.', data={'candidate_sha': checkpoint['candidate_sha'], 'worktree_path': str(worktree.path)})
            repository.pipeline_owner = lease_owner
            snapshot = repository.preflight(require_clean=False, require_branch=False, check_path_escapes=True)
            if snapshot.branch:
                raise PipelineNeedsAttention('저장된 격리 worktree의 branch가 변경되었습니다.')
            invocation = checkpoint.get('invocation')
            if invocation and invocation['status'] in {'RUNNING', 'UNKNOWN'}:
                self._unknown_invocation(repository, invocation)
                raise PipelineNeedsAttention(
                    '중단된 모델 호출의 결과·사용량이 미확인입니다. 새 모델 실행 없이 변경과 예약을 보존합니다. '
                    f'작업 공간: {repository.path}',
                )
            # Older checkpoints have no invocation ID. Preserve orphaned agent
            # reservations, but concurrent conversation calls are not pipeline
            # invocations and must not become unknown development usage.
            if not invocation and self.store.reserved_token_total(state.run_id, category='agent'):
                if not self.store.has_budget_anomaly(state.run_id):
                    self.logger.emit(state.run_id, 'MODEL_USAGE_UNKNOWN',
                                     '중단된 개발 호출의 결과와 실제 사용량을 확인하지 못해 예약과 작업 공간을 보존했습니다.',
                                     stage_id=contracts[state.stage_index].stage_id, status='NEEDS_ATTENTION')
                raise PipelineNeedsAttention(
                    '중단된 모델 호출의 결과·사용량이 미확인입니다. 새 모델 실행 없이 변경과 예약을 보존합니다. '
                    f'작업 공간: {repository.path}',
                )
            scope = tuple(item for contract in contracts[:state.stage_index + 1] for item in contract.scope)
            repository.assert_commit_scope(state.repository_head_sha, snapshot.head, scope)
            repository.assert_commit_scope(checkpoint['stage_base_sha'], snapshot.head, contracts[state.stage_index].scope)
            if snapshot.head != checkpoint['candidate_sha']:
                pending = checkpoint.get('pending_commit')
                if not (pending and repository.matches_pending_commit(pending['before'], snapshot.head, pending['message'])):
                    raise PipelineNeedsAttention('worktree HEAD와 저장된 candidate가 다릅니다. 변경을 보존하고 중단합니다.')
                repository.retain_candidate(state.run_id, snapshot.head)
                checkpoint = self._save_workspace(repository, candidate_sha=snapshot.head, pending_commit=None,
                                                  operation='', dirty_fingerprint='')
            if snapshot.status:
                repository.assert_scope(contracts[state.stage_index].scope)
                fingerprint = self._dirty_fingerprint(repository)
                if fingerprint != checkpoint['dirty_fingerprint']:
                    if checkpoint['operation'] == 'writing':
                        self._save_workspace(repository, dirty_fingerprint=fingerprint, operation='')
                        raise PipelineNeedsAttention(
                            '프로세스 종료 전 미커밋 변경을 보존했습니다. 확인 후 재개해 주세요. '
                            f'작업 공간: {repository.path}',
                        )
                    raise PipelineNeedsAttention('저장 이후 미커밋 파일 내용이 변경되었습니다. 복구 위치를 확인해 주세요: ' + str(repository.path))
            elif checkpoint['dirty_fingerprint']:
                raise PipelineNeedsAttention('저장된 미커밋 변경이 현재 작업 공간에 없습니다. 자동으로 계속하지 않습니다.')
            self.artifacts.write_json(state.run_id, Path('pipeline-checkpoint.json'), checkpoint)
            return worktree, repository

        except BaseException:
            if worktree is not None:
                worktree.release_execution_guard()
            raise

    def _finish_verified_stage(self, state, repository, source_repository, source_snapshot, contracts, heartbeat):
        self._steering_boundary(repository)
        if state.stage_index + 1 == state.stage_count:
            self._validate_application_permissions(self.store.load_run(state.run_id))
            # 외부 Git 반영 전 intent는 별도 commit한다. 반영 중 프로세스 종료를 복구할 수 있어야 한다.
            self._save_workspace(repository, status='applying')
        with self.store.transaction():
            heartbeat()
            self._steering_boundary(repository)
            self._check_cancel(repository.cancelled or (lambda: False))
            checkpoint = self.store.pipeline_workspace(state.run_id)
            snapshot = repository.snapshot()
            if snapshot.head != checkpoint['validated_sha'] or snapshot.status:
                raise PipelineNeedsAttention('검증된 candidate와 현재 작업 공간이 일치하지 않습니다.')
            if state.stage_index + 1 == state.stage_count:
                current_state = self.store.load_run(state.run_id)
                self._validate_approval(current_state, check_head=False)
                self._validate_application_permissions(current_state)
                repository.assert_commit_scope(
                    source_snapshot.head, snapshot.head,
                    tuple(item for contract in contracts for item in contract.scope),
                )
                self._save_workspace(repository, status='applying')
                self._check_cancel(repository.cancelled or (lambda: False))
                current = source_repository.snapshot()
                if current.head == checkpoint['validated_sha'] and current.branch == source_snapshot.branch and not current.status:
                    self.artifacts.write_json(state.run_id, Path('source-application.json'), {
                        'applied': True, 'recovered': True, 'candidate_sha': current.head, 'applied_sha': current.head,
                    })
                else:
                    self._apply_final_candidate(state.run_id, source_repository, source_snapshot, checkpoint['validated_sha'], contracts)
                verification_path = self.artifacts.path_for(
                    state.run_id, Path('stages') / contracts[state.stage_index].stage_id / 'verification.json',
                )
                if verification_path.is_file():
                    verification = json.loads(verification_path.read_text(encoding='utf-8'))
                    verification['source_applied_sha'] = checkpoint['validated_sha']
                    self.artifacts.write_json(state.run_id, verification_path.relative_to(self.artifacts.root / state.run_id), verification)
            heartbeat()
            with self.store.transaction():
                self.store.assert_pipeline_owner(state.run_id, repository.pipeline_owner)
                current_state = self.store.load_run(state.run_id)
                if current_state.phase == RunPhase.DEVELOPING:
                    # The gateway queues a paused run as DEVELOPING; a verified checkpoint
                    # restores its completed verification boundary before advancing the stage.
                    current_state = self.store.save_run(replace(current_state, phase=RunPhase.VERIFYING, resume_phase=None))
                state = self.machine.complete_stage(current_state)
                self._save_workspace(
                    repository, stage_index=state.stage_index, stage_base_sha=snapshot.head,
                    status='completed' if state.phase == RunPhase.COMPLETED else 'ready',
                    candidate_sha=snapshot.head, operation='', pending_commit=None, dirty_fingerprint='',
                )

    def _validate_application_permissions(self, state):
        job = self.store.pipeline_job(state.run_id)
        session = self.store.load_conversation_session(job['channel'], job['conversation_id'])
        if session is None or not self.store.repository_is_approved(
            job['channel'], job['conversation_id'], session.user_id, state.repository, state.repository_identity,
        ):
            raise PipelineNeedsAttention('원본 반영 직전 프로젝트 사용 승인이 없거나 만료되었습니다.')
        plan = self.store.load_plan_revision(state.run_id, state.plan_revision)
        if not (plan and plan['status'] == 'APPROVED' and plan['plan_hash'] == state.approved_plan_hash):
            raise PipelineNeedsAttention('원본 반영 직전 승인된 계획이 변경되었습니다.')

    def _run_stage(
        self,
        state,
        contract: StageContract,
        repository: GitRepository,
        channel: str,
        conversation_id: str,
        cancelled: Callable[[], bool],
        heartbeat: Callable[[], None],
        environment: ToolchainEnvironment | None,
        source_repository: GitRepository,
        source_snapshot,
        contracts: tuple[StageContract, ...],
    ) -> None:
        stage_number = state.stage_index + 1
        stage_id = contract.stage_id
        applied = [item for item in self.store.execution_inputs(state.run_id) if item['status'] == 'APPLIED']
        directions = [item for item in applied if item['intent']['action'] in {'redirect', 'test_first'}]
        if directions and directions[-1]['intent']['action'] == 'test_first':
            answers = self.store.answered_execution_questions(state.run_id, stage_id)
            if not answers or answers[-1]['answered_at'] < directions[-1]['updated_at']:
                return self._test_first(state, contract, repository, cancelled, heartbeat, environment)
        before_development = repository.preflight(
            require_clean=False, require_branch=False, check_path_escapes=True
        )
        checkpoint = self.store.pipeline_workspace(state.run_id)
        base = replace(before_development, head=checkpoint['stage_base_sha'], status='', untracked_files=())
        self._set_role_for_run(state.run_id, channel, conversation_id, RoleId.DEVELOPMENT.value)
        development_name = self._role_name(RoleId.DEVELOPMENT)
        review_name = self._role_name(RoleId.REVIEW)
        improvement_name = self._role_name(RoleId.IMPROVEMENT)
        development = (self._acknowledged_development(state, contract, checkpoint)
                       if not directions and not before_development.status else None)
        if development is None:
            self._notify_for_run(state.run_id,
                channel,
                conversation_id,
                f"{self._subject(development_name)} {stage_number}단계 개발을 시작합니다.",
            )
            self.logger.emit(
                state.run_id,
                "AGENT_STARTED",
                f"{self._subject(development_name)} 단계 구현을 시작했습니다.",
                stage_id=stage_id,
                role_id=RoleId.DEVELOPMENT.value,
            )
            development = self._call_agent(
                state.run_id,
                stage_id,
                RoleId.DEVELOPMENT,
                repository,
                development_prompt(self._instructions(RoleId.DEVELOPMENT), contract),
                allow_writes=True,
                cancelled=cancelled,
                heartbeat=heartbeat,
            )
        self._assert_write_position(
            state.run_id,
            contract,
            repository,
            before_development,
            RoleId.DEVELOPMENT,
            "development",
        )
        development_summary = parse_role_summary(development.text)
        self._require_no_user_input(
            state.run_id,
            stage_id,
            RoleId.DEVELOPMENT,
            development_summary,
            repository=repository,
            before=before_development,
            on_dirty=lambda: self._record_recovery(
                state.run_id,
                stage_id,
                repository,
                base,
                RoleId.DEVELOPMENT,
                "development-user-question",
                "사용자 질문 전에 파일 변경이 남았습니다.",
            ),
        )
        development_sha = self._commit_scoped_changes(
            state.run_id,
            contract,
            repository,
            base,
            RoleId.DEVELOPMENT,
            "development",
            f"[AI-Agents] {state.run_id} {stage_id} development",
        )
        changed_files = repository.changed_files(base.head, development_sha)
        self.artifacts.write_json(
            state.run_id,
            Path("stages") / stage_id / "development-result.json",
            {
                "summary": development_summary.summary,
                "needs_user_input": list(development_summary.needs_user_input),
                "base_sha": base.head,
                "candidate_sha": development_sha,
                "changed_files": list(changed_files),
                "usage": development.usage.to_dict(),
            },
        )
        development_handoff = AgentHandoff(
            contract=contract,
            from_role=RoleId.DEVELOPMENT,
            to_role=RoleId.REVIEW,
            summary=development_summary.summary,
            base_sha=base.head,
            candidate_sha=development_sha,
            changed_files=changed_files,
            usage=development.usage,
        )
        self.artifacts.save_handoff(development_handoff)

        state = self.machine.transition(
            self.store.load_run(state.run_id),
            RunPhase.REVIEWING,
            message=f"{development_name}가 완료하여 {review_name}에게 독립 리뷰를 요청했습니다.",
        )
        self._set_role_for_run(state.run_id, channel, conversation_id, RoleId.REVIEW.value)
        self._notify_for_run(state.run_id,
            channel,
            conversation_id,
            f"{development_name} 개발 완료 → {self._subject(review_name)} 리뷰를 시작합니다.",
        )
        review, review_result = self._review_candidate(
            state.run_id,
            contract,
            repository,
            development_handoff,
            "review-result.json",
            cancelled,
            heartbeat,
        )
        candidate_sha = development_sha
        design_rework_sha: str | None = None
        improvement_sha: str | None = None
        improvement_usage = TokenUsage()
        verification_retry_sha: str | None = None
        closure_reviewed = False
        requires_closure_review = False

        design_findings = tuple(
            finding
            for finding in review_result.findings
            if finding.category == "design"
        )
        if design_findings:
            if not self.budget.can_retry(state.run_id, stage_id, "design_rework"):
                raise PipelineNeedsAttention(
                    "설계 문제가 발견됐지만 빌더 재작업 허용 횟수가 없습니다."
                )
            self.budget.record_retry(
                state.run_id,
                stage_id,
                "design_rework",
                f"센티널이 design finding {len(design_findings)}건을 보고했습니다.",
            )
            design_handoff = AgentHandoff(
                contract=contract,
                from_role=RoleId.REVIEW,
                to_role=RoleId.DEVELOPMENT,
                summary=review_result.summary,
                base_sha=base.head,
                candidate_sha=candidate_sha,
                changed_files=changed_files,
                findings=design_findings,
                usage=review.usage,
            )
            self.artifacts.save_handoff(design_handoff)
            self.machine.transition(
                self.store.load_run(state.run_id),
                RunPhase.DEVELOPING,
                message=f"{review_name}의 설계 지적을 {development_name}에게 되돌렸습니다.",
            )
            self._set_role_for_run(state.run_id,
                channel, conversation_id, RoleId.DEVELOPMENT.value
            )
            self._notify_for_run(state.run_id,
                channel,
                conversation_id,
                f"{review_name} 설계 문제 {len(design_findings)}건 → "
                f"{self._subject(development_name)} 재작업을 시작합니다.",
            )
            before_rework = repository.snapshot()
            rework = self._call_agent(
                state.run_id,
                stage_id,
                RoleId.DEVELOPMENT,
                repository,
                development_rework_prompt(
                    self._instructions(RoleId.DEVELOPMENT),
                    contract,
                    design_findings,
                ),
                allow_writes=True,
                cancelled=cancelled,
                heartbeat=heartbeat,
            )
            self._assert_write_position(
                state.run_id,
                contract,
                repository,
                before_rework,
                RoleId.DEVELOPMENT,
                "design-rework",
            )
            rework_summary = parse_role_summary(rework.text)
            self._require_no_user_input(
                state.run_id,
                stage_id,
                RoleId.DEVELOPMENT,
                rework_summary,
                repository=repository,
                before=before_rework,
                on_dirty=lambda: self._record_recovery(
                    state.run_id,
                    stage_id,
                    repository,
                    before_rework,
                    RoleId.DEVELOPMENT,
                    "design-rework-user-question",
                    "사용자 질문 전에 파일 변경이 남았습니다.",
                ),
            )
            candidate_sha = self._commit_scoped_changes(
                state.run_id,
                contract,
                repository,
                before_rework,
                RoleId.DEVELOPMENT,
                "design-rework",
                f"[AI-Agents] {state.run_id} {stage_id} design-rework",
            )
            design_rework_sha = candidate_sha
            rework_changed_files = repository.changed_files(
                before_rework.head, candidate_sha
            )
            changed_files = repository.changed_files(base.head, candidate_sha)
            self.artifacts.write_json(
                state.run_id,
                Path("stages") / stage_id / "development-rework-result.json",
                {
                    "summary": rework_summary.summary,
                    "needs_user_input": list(rework_summary.needs_user_input),
                    "base_sha": before_rework.head,
                    "candidate_sha": candidate_sha,
                    "changed_files": list(rework_changed_files),
                    "addressed_findings": [
                        finding.to_dict() for finding in design_findings
                    ],
                    "usage": rework.usage.to_dict(),
                },
            )
            rework_handoff = AgentHandoff(
                contract=contract,
                from_role=RoleId.DEVELOPMENT,
                to_role=RoleId.REVIEW,
                summary=rework_summary.summary,
                base_sha=base.head,
                candidate_sha=candidate_sha,
                changed_files=changed_files,
                usage=rework.usage,
            )
            self.artifacts.save_handoff(
                rework_handoff, suffix="after-design-rework"
            )
            self.machine.transition(
                self.store.load_run(state.run_id),
                RunPhase.REVIEWING,
                message=f"{development_name}의 설계 재작업을 {review_name}가 다시 리뷰합니다.",
            )
            self._set_role_for_run(state.run_id,
                channel, conversation_id, RoleId.REVIEW.value
            )
            self._notify_for_run(state.run_id,
                channel,
                conversation_id,
                f"{development_name} 재작업 완료 → "
                f"{self._subject(review_name)} 재리뷰를 시작합니다.",
            )
            review, review_result = self._review_candidate(
                state.run_id,
                contract,
                repository,
                rework_handoff,
                "review-after-design-rework.json",
                cancelled,
                heartbeat,
            )
            if any(
                finding.category == "design"
                for finding in review_result.findings
            ):
                raise PipelineNeedsAttention(
                    "빌더 재작업 후에도 설계 문제가 남아 자동 진행을 중단했습니다."
                )

        implementation_findings = tuple(
            finding
            for finding in review_result.findings
            if finding.category == "implementation"
        )
        verification_review = ReviewSummary(
            summary=review_result.summary,
            findings=implementation_findings,
            needs_user_input=review_result.needs_user_input,
        )
        if implementation_findings:
            review_handoff = AgentHandoff(
                contract=contract,
                from_role=RoleId.REVIEW,
                to_role=RoleId.IMPROVEMENT,
                summary=review_result.summary,
                base_sha=base.head,
                candidate_sha=candidate_sha,
                changed_files=repository.changed_files(base.head, candidate_sha),
                findings=implementation_findings,
                usage=review.usage,
            )
            self.artifacts.save_handoff(review_handoff)
            self.machine.transition(
                self.store.load_run(state.run_id),
                RunPhase.IMPROVING,
                message=f"{review_name}의 구현 지적을 {improvement_name}에게 전달했습니다.",
            )
            self._set_role_for_run(state.run_id,
                channel, conversation_id, RoleId.IMPROVEMENT.value
            )
            self._notify_for_run(state.run_id,
                channel,
                conversation_id,
                f"{review_name} 구현 문제 {len(implementation_findings)}건 → "
                f"{self._subject(improvement_name)} 수정을 시작합니다.",
            )
            improvement_sha, improvement_usage = self._improve(
                state.run_id,
                contract,
                verification_review,
                repository,
                cancelled,
                heartbeat,
            )
            candidate_sha = improvement_sha

            requires_closure_review = any(
                finding.severity in {"critical", "high"}
                for finding in implementation_findings
            )
            self.machine.transition(
                self.store.load_run(state.run_id),
                RunPhase.VERIFYING,
                message=f"{improvement_name}의 보완이 완료되어 결정론적 검증을 시작합니다.",
            )
            self._notify_for_run(state.run_id,
                channel,
                conversation_id,
                f"{improvement_name} 보완 완료 → 검증을 시작합니다.",
            )
        else:
            self.machine.transition(
                self.store.load_run(state.run_id),
                RunPhase.VERIFYING,
                message=f"{review_name}가 문제없음을 확인하여 보완 없이 검증합니다.",
            )
            self._notify_for_run(state.run_id,
                channel,
                conversation_id,
                f"{review_name} 리뷰 결과 문제 없음 → {improvement_name} 보완 없이 먼저 검증합니다.",
            )

        verification, verification_retry_sha, verification_retry_usage = self._verify(
            state.run_id,
            contract,
            verification_review,
            repository,
            cancelled,
            heartbeat,
            channel,
            conversation_id,
            improvement_name,
            environment,
        )
        if verification_retry_usage is not None:
            improvement_usage = verification_retry_usage
        candidate_sha = repository.snapshot().head

        if requires_closure_review:
            self._write_verification_artifact(
                state.run_id,
                stage_id,
                verification,
                development_sha=development_sha,
                design_rework_sha=design_rework_sha,
                improvement_sha=improvement_sha,
                verification_retry_sha=verification_retry_sha,
                final_sha=candidate_sha,
                closure_review_required=True,
                closure_reviewed=False,
                closure_review_passed=None,
            )
            if not self.budget.can_retry(state.run_id, stage_id, "closure_review"):
                raise PipelineNeedsAttention(
                    "중요 구현 문제의 마감 재리뷰 허용 횟수가 없습니다."
                )
            self.budget.record_retry(
                state.run_id,
                stage_id,
                "closure_review",
                "검증된 critical/high 보완 결과의 마감 재리뷰",
            )
            closure_handoff = AgentHandoff(
                contract=contract,
                from_role=RoleId.IMPROVEMENT,
                to_role=RoleId.REVIEW,
                summary=(
                    f"{improvement_name}가 중요 구현 finding을 보완했고 "
                    "결정론적 검증까지 통과했습니다."
                ),
                base_sha=base.head,
                candidate_sha=candidate_sha,
                changed_files=repository.changed_files(base.head, candidate_sha),
                findings=implementation_findings,
                usage=improvement_usage,
                verification={
                    "passed": True,
                    "commands": [result.to_dict() for result in verification],
                },
            )
            self.artifacts.save_handoff(closure_handoff)
            self.machine.transition(
                self.store.load_run(state.run_id),
                RunPhase.REVIEWING,
                message=f"검증된 {improvement_name}의 중요 보완을 {review_name}가 마감 리뷰합니다.",
            )
            self._set_role_for_run(state.run_id,
                channel, conversation_id, RoleId.REVIEW.value
            )
            self._notify_for_run(state.run_id,
                channel,
                conversation_id,
                f"검증 통과 → {self._subject(review_name)} 최종 SHA를 마감 리뷰합니다.",
            )
            _closure, closure_result = self._review_candidate(
                state.run_id,
                contract,
                repository,
                closure_handoff,
                "review-closure-result.json",
                cancelled,
                heartbeat,
            )
            closure_reviewed = True
            if closure_result.findings:
                self._write_verification_artifact(
                    state.run_id,
                    stage_id,
                    verification,
                    development_sha=development_sha,
                    design_rework_sha=design_rework_sha,
                    improvement_sha=improvement_sha,
                    verification_retry_sha=verification_retry_sha,
                    final_sha=candidate_sha,
                    closure_review_required=True,
                    closure_reviewed=True,
                    closure_review_passed=False,
                )
                raise PipelineNeedsAttention(
                    "마감 재리뷰에서 문제가 남아 자동 진행을 중단했습니다."
                )
            self.machine.transition(
                self.store.load_run(state.run_id),
                RunPhase.VERIFYING,
                message="검증된 최종 SHA가 마감 재리뷰까지 통과했습니다.",
            )

        final_sha = repository.snapshot().head
        heartbeat()
        self._steering_boundary(repository)
        self._save_workspace(repository, candidate_sha=final_sha, validated_sha=final_sha,
                             status='stage_verified', operation='', dirty_fingerprint='')
        self._write_verification_artifact(
            state.run_id,
            stage_id,
            verification,
            development_sha=development_sha,
            design_rework_sha=design_rework_sha,
            improvement_sha=improvement_sha,
            verification_retry_sha=verification_retry_sha,
            final_sha=final_sha,
            closure_review_required=requires_closure_review,
            closure_reviewed=closure_reviewed,
            closure_review_passed=True if closure_reviewed else None,
            source_applied_sha=None,
        )
        self._finish_verified_stage(state, repository, source_repository, source_snapshot, contracts, heartbeat)
        state = self.store.load_run(state.run_id)
        self._set_role_for_run(state.run_id, channel, conversation_id, RoleId.DEVELOPMENT.value)
        if state.phase == RunPhase.COMPLETED:
            self._set_mode_for_run(state.run_id, channel, conversation_id, "free_chat")
        self._notify_for_run(state.run_id, channel, conversation_id, f"{stage_number}단계를 완료했습니다.")
        self.logger.write_summary(state)

    def _apply_final_candidate(
        self,
        run_id: str,
        source_repository: GitRepository,
        source_snapshot,
        candidate_sha: str,
        contracts: tuple[StageContract, ...],
    ) -> str:
        scope = tuple(
            dict.fromkeys(
                item for contract in contracts for item in contract.scope
            )
        )
        try:
            job = self.store.pipeline_job(run_id)
            if job and job['status'] in {'CANCEL_REQUESTED', 'CANCELLED', 'PAUSE_REQUESTED'}:
                raise PipelineCancelled('원본 반영 전 사용자 중지 요청을 확인했습니다.')
            if any(item['status'] in {'RECEIVED', 'READY', 'WAITING_APPROVAL'} for item in self.store.execution_inputs(run_id)):
                raise PipelineSteeringPending('STEERING_PENDING')
            applied_sha = source_repository.apply_candidate(
                source_snapshot, candidate_sha, scope
            )
            if applied_sha != candidate_sha:
                raise GitRepositoryError('원본 반영 SHA가 검증된 candidate와 다릅니다.')
        except GitRepositoryError as exc:
            try:
                current = source_repository.snapshot()
                current_value = {
                    "head": current.head,
                    "branch": current.branch,
                    "status": current.status,
                }
            except GitRepositoryError:
                current_value = {"unavailable": True}
            self.artifacts.write_json(
                run_id,
                Path("source-application.json"),
                {
                    "applied": False,
                    "candidate_sha": candidate_sha,
                    "source_snapshot": {
                        "head": source_snapshot.head,
                        "branch": source_snapshot.branch,
                        "status": source_snapshot.status,
                        "untracked_files": list(source_snapshot.untracked_files),
                    },
                    "source_current": current_value,
                    "reason": self.redactor.text(str(exc)),
                    "guidance": [
                        "원본 checkout은 자동으로 변경하거나 되돌리지 않았습니다.",
                        "원본의 HEAD, branch, git status를 확인한 뒤 임시 worktree candidate를 직접 검토하세요.",
                        "원본을 시작 snapshot으로 되돌릴 의도가 명확할 때만 candidate commit을 별도 Git 명령으로 적용하세요.",
                    ],
                },
            )
            self.logger.emit(
                run_id,
                "SOURCE_APPLICATION_BLOCKED",
                "원본 저장소 상태가 달라져 검증된 candidate를 자동 반영하지 않았습니다.",
                status="NEEDS_ATTENTION",
                data={"candidate_sha": candidate_sha},
            )
            raise PipelineNeedsAttention(
                "원본 저장소가 변경되었거나 candidate 범위를 다시 검증하지 못해 자동 반영하지 않았습니다. "
                "source-application.json과 worktree-recovery.json을 확인해 주세요."
            ) from exc
        self.artifacts.write_json(
            run_id,
            Path("source-application.json"),
            {
                "applied": True,
                "candidate_sha": candidate_sha,
                "applied_sha": applied_sha,
                "scope": list(scope),
            },
        )
        return applied_sha

    def _record_worktree_recovery(
        self, run_id: str, worktree: IsolatedGitWorktree
    ) -> None:
        if worktree.source_snapshot is None:
            return
        if worktree.repository is None:
            if self.store.pipeline_workspace(run_id) is not None:
                # An existing checkpoint owns this partial creation/recovery location.
                # Never remove it or replace its recovery evidence on a failed resume.
                return
            self.artifacts.write_json(
                run_id,
                Path("worktree.json"),
                {
                    "status": "source_blocked",
                    "guidance": [
                        "원본 checkout에 기존 변경이 있거나 시작 상태를 검증하지 못해 worktree를 만들지 않았습니다.",
                        "원본의 git status를 직접 확인한 뒤, 기존 변경을 보존한 상태에서 새 작업을 요청하세요.",
                    ],
                    **worktree.record.to_dict(),
                },
            )
            worktree.cleanup()
            return
        worktree.repository.cancelled = None
        try:
            checkpoint = self.store.pipeline_workspace(run_id)
            if checkpoint and checkpoint['operation'] == 'writing':
                current = worktree.repository.snapshot()
                if current.head == checkpoint['candidate_sha'] and not current.branch:
                    fingerprint = self._dirty_fingerprint(worktree.repository) if current.status else ''
                    invocation = checkpoint.get('invocation')
                    unresolved = invocation and invocation['status'] in {'RUNNING', 'UNKNOWN'}
                    self._save_workspace(worktree.repository, dirty_fingerprint=fingerprint,
                                         operation=checkpoint['operation'] if unresolved else '')
            bundle = worktree.repository.recovery_bundle(worktree.source_snapshot)
        except (GitRepositoryError, StoreError, AttributeError):
            return
        patch_relative = Path("worktree-recovery.patch")
        if bundle.patch:
            self.artifacts.write_text(run_id, patch_relative, bundle.patch)
        self.artifacts.write_json(
            run_id,
            Path("worktree-recovery.json"),
            {
                "preserved": True,
                "worktree": worktree.record.to_dict(),
                "recovery": bundle.to_dict(),
                "patch": str(patch_relative) if bundle.patch else "",
                "guidance": [
                    "원본 checkout은 변경하지 않았습니다.",
                    "gateway-owned 임시 worktree 경로에서 git status와 candidate commit을 먼저 확인하세요.",
                    "추적 파일 변경은 worktree-recovery.patch에 보관했습니다. untracked 파일은 worktree 경로에서 직접 확인하세요.",
                    "보존한 worktree는 사용자 확인 전 자동 삭제하지 않습니다.",
                ],
            },
        )

    def _assert_write_position(
        self,
        run_id: str,
        contract: StageContract,
        repository: GitRepository,
        before,
        role_id: RoleId,
        operation: str,
    ) -> None:
        try:
            repository.assert_safe_worktree_paths()
            repository.assert_position(before, allow_worktree_changes=True)
        except GitRepositoryError as exc:
            self._record_recovery(
                run_id,
                contract.stage_id,
                repository,
                before,
                role_id,
                operation,
                str(exc),
            )
            raise PipelineNeedsAttention(
                "에이전트가 Git 브랜치 또는 HEAD를 변경해 자동 커밋을 중단했습니다. "
                f"{self._recovery_reference(run_id, contract.stage_id)}을 확인해 주세요."
            ) from exc

    def _assert_read_only_position(
        self,
        run_id: str,
        stage_id: str,
        repository: GitRepository,
        before,
        role_id: RoleId,
        operation: str,
    ) -> None:
        try:
            repository.assert_position(before, allow_worktree_changes=False)
        except GitRepositoryError as exc:
            self._record_recovery(
                run_id,
                stage_id,
                repository,
                before,
                role_id,
                operation,
                str(exc),
            )
            raise PipelineNeedsAttention(
                "읽기 전용 단계가 저장소를 변경해 자동 진행을 중단했습니다. "
                f"{self._recovery_reference(run_id, stage_id)}을 확인해 주세요."
            ) from exc

    def _commit_scoped_changes(
        self,
        run_id: str,
        contract: StageContract,
        repository: GitRepository,
        before,
        role_id: RoleId,
        operation: str,
        message: str,
    ) -> str:
        self._steering_boundary(repository)
        current = repository.snapshot()
        # Git commit은 SQLite rollback으로 되돌아가지 않으므로 pending intent를 먼저 영속화한다.
        self._save_workspace(repository, operation='committing', pending_commit={
            'before': current.head, 'message': message,
        })
        with self.store.transaction():
            try:
                self._steering_boundary(repository)
                current = repository.snapshot()
                self._save_workspace(repository, operation='committing', pending_commit={
                    'before': current.head, 'message': message,
                })
                candidate = repository.commit_scope(message, contract.scope)
                repository.retain_candidate(run_id, candidate)
                self._save_workspace(repository, candidate_sha=candidate, operation='',
                                     pending_commit=None, dirty_fingerprint='')
                return candidate
            except GitRepositoryError as exc:
                self._record_recovery(
                    run_id,
                    contract.stage_id,
                    repository,
                    before,
                    role_id,
                    operation,
                    str(exc),
                )
                raise PipelineNeedsAttention(
                    f"{exc} {self._recovery_reference(run_id, contract.stage_id)}을 확인한 뒤 직접 판단해 주세요."
                ) from exc

    def _record_recovery(
        self,
        run_id: str,
        stage_id: str,
        repository: GitRepository,
        before,
        role_id: RoleId | None,
        operation: str,
        reason: str,
    ) -> None:
        """Preserve evidence and stop; never erase a possibly user-owned edit."""
        try:
            bundle = repository.recovery_bundle(before)
        except GitRepositoryError:
            return
        if not bundle.has_changes:
            return
        relative = Path("stages") / stage_id
        patch_path = relative / "recovery.patch"
        if bundle.patch:
            self.artifacts.write_text(run_id, patch_path, bundle.patch)
        self.artifacts.write_json(
            run_id,
            relative / "recovery.json",
            {
                "operation": operation,
                "role_id": role_id.value if role_id is not None else "verification",
                "reason": self.redactor.text(reason),
                "repository": str(repository.path),
                "recovery": bundle.to_dict(),
                "patch": str(patch_path) if bundle.patch else "",
                "guidance": [
                    "자동 되돌리기는 하지 않았습니다. 기존 사용자 변경과 구분할 수 없기 때문입니다.",
                    "저장소에서 git status --short와 Git diff를 먼저 확인하세요.",
                    "추적 파일의 diff는 recovery.patch에 보관했습니다. Git이 추적하지 않는 새 파일은 원래 저장소에 남아 있으므로 직접 검토하세요.",
                    "base_head와 현재 HEAD가 같을 때만, 변경을 버리기로 결정한 뒤 Git의 restore 기능으로 추적 파일을 되돌리세요.",
                ],
            },
        )
        self.logger.emit(
            run_id,
            "SAFETY_RECOVERY_RECORDED",
            "안전 중단 전 변경 증거와 복구 안내를 기록했습니다.",
            stage_id=stage_id,
            role_id=role_id.value if role_id is not None else "",
            status="NEEDS_ATTENTION",
            data={
                "operation": operation,
                "changed_files": list(bundle.changed_files),
                "base_head": bundle.base_head,
                "current_head": bundle.current_head,
            },
        )

    def _write_verification_artifact(
        self,
        run_id: str,
        stage_id: str,
        verification: tuple[VerificationResult, ...],
        *,
        development_sha: str,
        design_rework_sha: str | None,
        improvement_sha: str | None,
        verification_retry_sha: str | None,
        final_sha: str,
        closure_review_required: bool,
        closure_reviewed: bool,
        closure_review_passed: bool | None,
        source_applied_sha: str | None = None,
    ) -> None:
        self.artifacts.write_json(
            run_id,
            Path("stages") / stage_id / "verification.json",
            {
                "passed": True,
                "development_sha": development_sha,
                "design_rework_sha": design_rework_sha,
                "improvement_sha": improvement_sha,
                "verification_retry_sha": verification_retry_sha,
                "closure_review_required": closure_review_required,
                "closure_reviewed": closure_reviewed,
                "closure_review_passed": closure_review_passed,
                "final_sha": final_sha,
                "source_applied_sha": source_applied_sha,
                "commands": [result.to_dict() for result in verification],
            },
        )

    def _review_candidate(
        self,
        run_id: str,
        contract: StageContract,
        repository: GitRepository,
        handoff: AgentHandoff,
        result_name: str,
        cancelled: Callable[[], bool],
        heartbeat: Callable[[], None],
    ) -> tuple[HermesResult, ReviewSummary]:
        baseline = repository.snapshot()
        bundle = repository.review_bundle(handoff.base_sha, handoff.candidate_sha)
        self.artifacts.write_json(run_id, Path('stages') / contract.stage_id / (result_name + '-bundle.json'), bundle)
        result, summary = self._call_and_parse_review(
            run_id,
            contract.stage_id,
            repository,
            review_prompt(
                self._instructions(RoleId.REVIEW),
                handoff,
                repository.diff_stat(handoff.base_sha, handoff.candidate_sha),
            ),
            cancelled,
            heartbeat,
            review_bundle=bundle,
        )
        self._assert_read_only_position(
            run_id,
            contract.stage_id,
            repository,
            baseline,
            RoleId.REVIEW,
            "review",
        )
        if repository.snapshot().head != handoff.candidate_sha:
            raise PipelineNeedsAttention('리뷰 중 candidate가 변경되어 결과를 승인에 사용하지 않습니다.')
        self._require_no_user_input(
            run_id, contract.stage_id, RoleId.REVIEW, summary
        )
        self.artifacts.write_json(
            run_id,
            Path("stages") / contract.stage_id / result_name,
            {
                "summary": summary.summary,
                "findings": [finding.to_dict() for finding in summary.findings],
                "needs_user_input": list(summary.needs_user_input),
                "base_sha": handoff.base_sha,
                "candidate_sha": handoff.candidate_sha,
                "patch_sha256": bundle['patch_sha256'],
                "usage": result.usage.to_dict(),
            },
        )
        return result, summary

    def _improve(
        self,
        run_id: str,
        contract: StageContract,
        review: ReviewSummary,
        repository: GitRepository,
        cancelled: Callable[[], bool],
        heartbeat: Callable[[], None],
        *,
        verification_failure: str = "",
    ) -> tuple[str, TokenUsage]:
        before = repository.snapshot()
        result = self._call_agent(
            run_id,
            contract.stage_id,
            RoleId.IMPROVEMENT,
            repository,
            improvement_prompt(
                self._instructions(RoleId.IMPROVEMENT),
                contract,
                review.findings,
                verification_failure=verification_failure,
            ),
            allow_writes=True,
            cancelled=cancelled,
            heartbeat=heartbeat,
        )
        self._assert_write_position(
            run_id,
            contract,
            repository,
            before,
            RoleId.IMPROVEMENT,
            "verification-retry" if verification_failure else "improvement",
        )
        summary = parse_role_summary(result.text)
        self._require_no_user_input(
            run_id,
            contract.stage_id,
            RoleId.IMPROVEMENT,
            summary,
            repository=repository,
            before=before,
            on_dirty=lambda: self._record_recovery(
                run_id,
                contract.stage_id,
                repository,
                before,
                RoleId.IMPROVEMENT,
                "verification-retry-user-question"
                if verification_failure
                else "improvement-user-question",
                "사용자 질문 전에 파일 변경이 남았습니다.",
            ),
        )
        candidate_sha = self._commit_scoped_changes(
            run_id,
            contract,
            repository,
            before,
            RoleId.IMPROVEMENT,
            "verification-retry" if verification_failure else "improvement",
            f"[AI-Agents] {run_id} {contract.stage_id} improvement",
        )
        result_name = (
            "improvement-retry-result.json"
            if verification_failure
            else "improvement-result.json"
        )
        self.artifacts.write_json(
            run_id,
            Path("stages") / contract.stage_id / result_name,
            {
                "summary": summary.summary,
                "needs_user_input": list(summary.needs_user_input),
                "base_sha": before.head,
                "candidate_sha": candidate_sha,
                "usage": result.usage.to_dict(),
                "verification_retry": bool(verification_failure),
                "changed": candidate_sha != before.head,
            },
        )
        if candidate_sha == before.head:
            raise PipelineNeedsAttention(
                "피니셔가 전달된 구현 문제에 대해 실제 코드 변경을 만들지 않았습니다."
            )
        return candidate_sha, result.usage

    def _verify(
        self,
        run_id: str,
        contract: StageContract,
        review: ReviewSummary,
        repository: GitRepository,
        cancelled: Callable[[], bool],
        heartbeat: Callable[[], None],
        channel: str,
        conversation_id: str,
        improvement_name: str,
        environment: ToolchainEnvironment | None,
    ) -> tuple[
        tuple[VerificationResult, ...],
        str | None,
        TokenUsage | None,
    ]:
        results = self._run_verification_safely(
            run_id,
            contract,
            repository,
            cancelled,
            heartbeat,
            operation="verification",
            environment=environment,
        )
        if results and all(result.passed for result in results):
            return results, None, None
        failure = self._verification_failure_text(results)
        if any(result.failure_kind in {VerificationFailureKind.TOOL_UNAVAILABLE,
                                       VerificationFailureKind.DEPENDENCY_UNAVAILABLE,
                                       VerificationFailureKind.ENVIRONMENT_UNAVAILABLE,
                                       VerificationFailureKind.TIMEOUT} for result in results):
            self.artifacts.write_json(run_id, Path('stages') / contract.stage_id / 'verification.json',
                {'passed': False, 'commands': [item.to_dict() for item in results],
                 'code_repair_attempted': False})
            raise PipelineNeedsAttention('검증 환경·시간 한도를 확인해야 합니다. 코드 보완은 실행하지 않습니다. ' + failure)
        if not self.budget.can_retry(run_id, contract.stage_id, "verification_failure"):
            self.artifacts.write_json(
                run_id,
                Path("stages") / contract.stage_id / "verification.json",
                {"passed": False, "commands": [item.to_dict() for item in results]},
            )
            raise PipelineNeedsAttention("검증이 실패했고 허용된 보완 재시도를 모두 사용했습니다.")
        self.budget.record_retry(
            run_id, contract.stage_id, "verification_failure", failure[:1000]
        )
        self.machine.transition(
            self.store.load_run(run_id),
            RunPhase.IMPROVING,
            message="검증 실패를 보완 담당에게 다시 전달했습니다.",
        )
        self._set_role_for_run(run_id,
            channel, conversation_id, RoleId.IMPROVEMENT.value
        )
        self._notify_for_run(run_id,
            channel,
            conversation_id,
            f"검증 실패 → {self._subject(improvement_name)} 보완 재시도를 시작합니다.",
        )
        retry_sha, retry_usage = self._improve(
            run_id,
            contract,
            review,
            repository,
            cancelled,
            heartbeat,
            verification_failure=failure,
        )
        self.machine.transition(
            self.store.load_run(run_id),
            RunPhase.VERIFYING,
            message="보완 재시도 후 검증을 다시 시작합니다.",
        )
        self._notify_for_run(run_id,
            channel,
            conversation_id,
            f"{improvement_name} 보완 재시도 완료 → 검증을 다시 시작합니다.",
        )
        retried = self._run_verification_safely(
            run_id,
            contract,
            repository,
            cancelled,
            heartbeat,
            operation="verification-retry",
            environment=environment,
        )
        if not retried or not all(result.passed for result in retried):
            self.artifacts.write_json(
                run_id,
                Path("stages") / contract.stage_id / "verification.json",
                {
                    "passed": False,
                    "verification_retry_sha": retry_sha,
                    "commands": [item.to_dict() for item in retried],
                },
            )
            raise PipelineNeedsAttention("보완 재시도 후에도 검증이 실패했습니다.")
        return retried, retry_sha, retry_usage

    def _run_verification_safely(
        self,
        run_id: str,
        contract: StageContract,
        repository: GitRepository,
        cancelled: Callable[[], bool],
        heartbeat: Callable[[], None],
        *,
        operation: str,
        environment: ToolchainEnvironment | None,
    ) -> tuple[VerificationResult, ...]:
        self._steering_boundary(repository)
        before = repository.snapshot()
        try:
            results = self.verifier.run_all(
                repository,
                contract.verification_commands,
                cancelled=cancelled,
                heartbeat=heartbeat,
                operation_id=run_id,
                environment=environment,
            )
            checkpoint = self.store.pipeline_workspace(run_id)
            evidence = dict(checkpoint.get('execution_evidence', {}))
            changed_files = set(repository.changed_files(checkpoint['stage_base_sha'], before.head))
            if before.status:
                changed_files.update(repository.worktree_files())
            evidence[contract.stage_id] = {
                'candidate_sha': before.head,
                'changed_files': sorted(changed_files),
                'uncommitted_changes': bool(before.status),
                'required_commands': list(contract.verification_commands),
                'commands': [{'command': item.command, 'return_code': item.return_code,
                              'started': item.started,
                              'failure_kind': item.failure_kind, 'passed': item.passed} for item in results],
                'unperformed_commands': [item.command for item in results if not item.started]
                    + list(contract.verification_commands[len(results):]),
            }
            self._save_workspace(repository, execution_evidence=evidence)
            self._steering_boundary(repository)
            return results
        except GitRepositoryError as exc:
            self._record_recovery(
                run_id,
                contract.stage_id,
                repository,
                before,
                None,
                operation,
                str(exc),
            )
            raise PipelineNeedsAttention(
                "검증 명령이 작업 트리를 변경해 자동 진행을 중단했습니다. "
                f"{self._recovery_reference(run_id, contract.stage_id)}을 확인해 주세요."
            ) from exc

    def _call_and_parse_review(
        self,
        run_id: str,
        stage_id: str,
        repository: GitRepository,
        prompt: str,
        cancelled: Callable[[], bool],
        heartbeat: Callable[[], None],
        *, review_bundle: dict | None = None,
    ) -> tuple[HermesResult, ReviewSummary]:
        while True:
            result = self._call_agent(
                run_id,
                stage_id,
                RoleId.REVIEW,
                repository,
                prompt,
                allow_writes=False,
                cancelled=cancelled,
                heartbeat=heartbeat,
                review_bundle=review_bundle,
            )
            try:
                return result, parse_review_summary(result.text)
            except InvalidAgentResponse as exc:
                if not self.budget.can_retry(run_id, stage_id, "invalid_response"):
                    raise PipelineNeedsAttention("리뷰 응답 형식을 해석하지 못했습니다.") from exc
                self.budget.record_retry(
                    run_id, stage_id, "invalid_response", str(exc)
                )
                prompt += (
                    "\n\n직전 응답 형식이 올바르지 않았다. 설명을 덧붙이지 말고 "
                    "요구된 JSON 객체 하나만 다시 출력하라."
                )

    def _call_agent(
        self,
        run_id: str,
        stage_id: str,
        role_id: RoleId,
        repository: GitRepository,
        prompt: str,
        *,
        allow_writes: bool,
        cancelled: Callable[[], bool],
        heartbeat: Callable[[], None],
        review_bundle: dict | None = None,
    ) -> HermesResult:
        self._steering_boundary(repository)
        directives = [item for item in self.store.execution_inputs(run_id) if item['status'] == 'APPLIED']
        if directives:
            prompt += '\n\n사용자 중간 지시(수신 순서, 승인된 scope 안에서만 반영; 기존 변경은 자동으로 되돌리지 않는다):\n' + json.dumps(
                [{key: item[key] for key in ('input_id', 'text', 'intent')} for item in directives], ensure_ascii=False)
        prior_answers = self.store.answered_execution_questions(run_id, stage_id)
        if prior_answers:
            answers = "\n".join(
                "- 질문: "
                + " / ".join(item["questions"])
                + "\n  사용자 답변: "
                + item["answer"]
                for item in prior_answers
            )
            prompt += (
                "\n\n이 단계에서 이전에 멈춘 뒤 사용자가 남긴 확정 답변이다. "
                "이 답변을 현재 단계 범위 안에서만 반영하라.\n"
                f"{answers}\n"
            )
        input_estimate = (
            conservative_prompt_tokens(prompt)
            + self.budget.policy.provider_input_overhead_tokens
        )
        output_cap = 2048
        estimate_invocation = getattr(self.runner, 'pipeline_invocation_estimate', None)
        invocation_estimate = (estimate_invocation(role_id, input_estimate, output_cap)
                               if callable(estimate_invocation) else input_estimate + output_cap)
        while True:
            self._steering_boundary(repository)
            repository.assert_safe_worktree_paths()
            baseline = repository.snapshot()
            self._check_cancel(cancelled)
            heartbeat()
            with self.budget.transaction():
                reservation = self.budget.reserve(
                    run_id, stage_id, role_id.value, 'agent', invocation_estimate,
                )
                call_id = 'pipeline:' + reservation.reservation_id
                invocation = {
                    'call_id': call_id, 'reservation_id': reservation.reservation_id,
                    'stage_id': stage_id, 'role_id': role_id.value, 'status': 'RUNNING',
                }
                self.store.create_model_call(call_id, call_id, run_id, stage_id,
                                             role_id.value, 'agent', reservation.reservation_id, '')
                self._save_workspace(repository, invocation=invocation,
                                     operation='writing' if allow_writes else 'reading')
            try:
                workspace_options = {}
                invocation_turns = getattr(self.runner, 'pipeline_invocation_turns', None)
                if callable(invocation_turns):
                    workspace_options['max_turns'] = invocation_turns(role_id)
                if getattr(self.runner, 'supports_workspace_policy', False):
                    state = self.store.load_run(run_id)
                    plan = self.store.load_plan_revision(run_id, state.plan_revision)
                    contract = StageContract.from_dict(plan['plan']['stages'][state.stage_index])
                    workspace_options.update({
                        'write_scope': contract.scope if allow_writes else (),
                        'review_bundle': review_bundle,
                        'execution_environment': self._execution_environments.get(run_id),
                    })
                result = self.runner.run(
                    run_id,
                    stage_id,
                    role_id,
                    repository.path,
                    prompt,
                    allow_writes=allow_writes,
                    max_output_tokens=output_cap,
                    cancelled=cancelled,
                    heartbeat=heartbeat,
                    **workspace_options,
                )
                self._settle_invocation(repository, invocation, reservation, result.usage,
                                        'COMPLETED', {'text': self.redactor.text(result.text),
                                                      'elapsed_seconds': result.elapsed_seconds,
                                                      'execution_input_ids': [item['input_id'] for item in directives],
                                                      'answered_question_ids': [item['question_id'] for item in prior_answers]})
                heartbeat()
                if allow_writes:
                    repository.assert_position(baseline, allow_worktree_changes=True)
                    current = repository.snapshot()
                    self._save_workspace(repository, operation='', dirty_fingerprint=
                        self._dirty_fingerprint(repository) if current.status else '')
                if not allow_writes:
                    self._assert_read_only_position(
                        run_id,
                        stage_id,
                        repository,
                        baseline,
                        role_id,
                        "read-only-agent",
                    )
                if not allow_writes:
                    self._save_workspace(repository, operation='')
                self._steering_boundary(repository)
                return result
            except HermesExecutionError as exc:
                usage_known = (not exc.usage.estimated and
                               (exc.usage_reported or bool(exc.usage.total_tokens)))
                if usage_known:
                    self._settle_invocation(repository, invocation, reservation, exc.usage,
                                            'CANCELLED' if isinstance(exc, HermesCancelled) else 'FAILED',
                                            {'error': self.redactor.text(str(exc))})
                elif exc.category == 'startup':
                    with self.budget.transaction():
                        self.store.assert_pipeline_owner(run_id, repository.pipeline_owner)
                        self.budget.release_reservation(reservation)
                        self.store.finish_model_call(call_id, 'NOT_STARTED', {})
                        self._save_workspace(repository, operation='', invocation=None)
                else:
                    self._unknown_invocation(repository, invocation)
                if isinstance(exc, HermesCancelled):
                    raise
                if not allow_writes:
                    self._assert_read_only_position(
                        run_id,
                        stage_id,
                        repository,
                        baseline,
                        role_id,
                        "read-only-agent",
                    )
                else:
                    self._record_recovery(
                        run_id,
                        stage_id,
                        repository,
                        baseline,
                        role_id,
                        "agent-execution",
                        str(exc),
                    )
                if not usage_known and exc.category != 'startup':
                    raise PipelineNeedsAttention(
                        '모델 호출의 결과·사용량이 미확인입니다. 변경과 예약을 보존하고 새 호출을 차단합니다.'
                    ) from exc
                if allow_writes or not self.budget.can_retry(
                    run_id, stage_id, "technical_error"
                ):
                    raise PipelineNeedsAttention(str(exc)) from exc
                self.budget.record_retry(
                    run_id,
                    stage_id,
                    "technical_error",
                    self.redactor.text(str(exc))[:1000],
                )
            except BaseException:
                current = self.store.pipeline_workspace(run_id).get('invocation')
                if current and current['call_id'] == call_id and current['status'] == 'RUNNING':
                    self._unknown_invocation(repository, current)
                raise

    def _acknowledged_development(self, state, contract, checkpoint):
        """비용 확인과 exact candidate에 연결된 완료 Builder 결과만 재사용한다."""
        event = next((item for item in reversed(self.store.budget_control_events(state.run_id))
                      if item['event_type'] == 'BUDGET_OVERRUN_ACKNOWLEDGED'
                      and item['data'].get('call_id')
                      and item['data'].get('role_id') == RoleId.DEVELOPMENT.value
                      and item['data'].get('stage_id') == contract.stage_id
                      and item['data'].get('plan_hash') == state.approved_plan_hash
                      and item['data'].get('candidate_sha') == checkpoint['candidate_sha']), None)
        if event is None:
            return None
        call = self.store.model_call(event['data']['call_id'])
        if (not call or call['run_id'] != state.run_id or call['status'] != 'COMPLETED'
                or call['stage_id'] != contract.stage_id or call['role_id'] != RoleId.DEVELOPMENT.value):
            raise PipelineNeedsAttention('초과 확인에 연결된 Builder 완료 결과가 일치하지 않습니다.')
        result = json.loads(call['result_json'])
        usage = TokenUsage.from_dict(result['usage'])
        if usage.estimated or usage.total_tokens != event['data']['reported_tokens']:
            raise PipelineNeedsAttention('보존된 Builder 비용 근거가 초과 확인과 일치하지 않습니다.')
        # 호출 저장 완료는 질문 해결이나 단계 구현 완료를 뜻하지 않는다.
        if parse_role_summary(result['text']).needs_user_input:
            return None
        inputs = [item['input_id'] for item in self.store.execution_inputs(state.run_id)
                  if item['status'] == 'APPLIED']
        answers = [item['question_id'] for item in
                   self.store.answered_execution_questions(state.run_id, contract.stage_id)]
        if (inputs != result.get('execution_input_ids', [])
                or answers != result.get('answered_question_ids', [])):
            return None
        self.logger.emit(state.run_id, 'DEVELOPMENT_RESULT_REUSED',
                         '비용을 확인한 완료 Builder 결과를 같은 candidate에서 재사용합니다.',
                         stage_id=contract.stage_id, role_id=RoleId.DEVELOPMENT.value,
                         data={'call_id': call['call_id'], 'candidate_sha': checkpoint['candidate_sha'],
                               'acknowledgment_event_id': event['event_id']})
        return HermesResult(result['text'], usage, float(result['elapsed_seconds']))

    def _settle_invocation(self, repository, invocation, reservation, usage, status, result) -> None:
        # Result, usage settlement and checkpoint form one durable boundary.
        with self.budget.transaction():
            self.store.assert_pipeline_owner(repository.operation_id, repository.pipeline_owner)
            self.budget.record_usage(repository.operation_id, invocation['stage_id'],
                                     invocation['role_id'], 'agent', usage, reservation=reservation)
            self.store.finish_model_call(invocation['call_id'], status, {**result, 'usage': usage.to_dict()})
            self._save_workspace(repository, invocation={**invocation, 'status': status})

    def _unknown_invocation(self, repository, invocation) -> None:
        with self.budget.transaction():
            self.store.assert_pipeline_owner(repository.operation_id, repository.pipeline_owner)
            if invocation['status'] == 'RUNNING':
                self.store.finish_model_call(invocation['call_id'], 'UNKNOWN', {})
                self.logger.emit(
                    repository.operation_id, 'MODEL_USAGE_UNKNOWN',
                    '모델 호출의 결과·실제 사용량이 미확인이라 예약과 작업 공간을 보존했습니다.',
                    stage_id=invocation['stage_id'], role_id=invocation['role_id'],
                    status='NEEDS_ATTENTION', data={'call_id': invocation['call_id'],
                                                   'reservation_id': invocation['reservation_id']},
                )
            self._save_workspace(repository, invocation={**invocation, 'status': 'UNKNOWN'})

    def _instructions(self, role_id: RoleId) -> str:
        role = self.foundation.roles[role_id]
        instructions = role.instructions.read_text(encoding="utf-8")
        if not role.display_name.strip():
            return instructions
        return f"당신의 표시 이름은 '{role.display_name.strip()}'이다.\n\n{instructions}"

    def _role_name(self, role_id: RoleId) -> str:
        configured = self.foundation.roles[role_id].display_name.strip()
        if configured:
            return configured
        return {
            RoleId.DEVELOPMENT: "개발 담당",
            RoleId.REVIEW: "리뷰 담당",
            RoleId.IMPROVEMENT: "보완 담당",
        }[role_id]

    @staticmethod
    def _subject(name: str) -> str:
        last = name[-1]
        if "가" <= last <= "힣":
            particle = "이" if (ord(last) - ord("가")) % 28 else "가"
        else:
            particle = "이(가)"
        return f"{name}{particle}"

    def _validate_approval(
        self, state, *, cancelled: Callable[[], bool] | None = None, check_head: bool = True
    ) -> None:
        if not state.repository or not state.repository_approved:
            raise PipelineNeedsAttention("승인된 저장소가 없습니다.")
        try:
            repository = inspect_repository_identity(
                Path(state.repository),
                sandbox=self.sandbox,
                operation_id=state.run_id,
                component="pipeline-approval",
                cancelled=cancelled,
            )
        except RepositoryAccessError as exc:
            raise PipelineNeedsAttention(
                f"승인된 저장소를 다시 확인할 수 없습니다: {exc}"
            ) from exc
        if repository.identity_hash != state.repository_identity:
            raise PipelineNeedsAttention(
                "승인 후 같은 경로의 저장소가 바뀌었습니다. 프로젝트를 다시 승인해야 합니다."
            )
        checkpoint = self.store.pipeline_workspace(state.run_id)
        application_recovery = bool(
            checkpoint and checkpoint['status'] in {'applying', 'completed'}
            and repository.head_sha == checkpoint['validated_sha']
            and checkpoint['plan_hash'] == state.approved_plan_hash
            and checkpoint['repository_identity'] == repository.identity_hash
        )
        if check_head and repository.head_sha != state.repository_head_sha and not application_recovery:
            raise PipelineNeedsAttention(
                "계획 승인 후 Git HEAD가 바뀌었습니다. 현재 코드 기준으로 다시 계획해야 합니다. "
                + (f"보존 위치: {checkpoint['worktree']['worktree_path']}; candidate: {checkpoint['candidate_sha']}" if checkpoint else '')
            )
        if not state.approval_granted:
            raise PipelineNeedsAttention("개발 실행 승인이 없습니다.")
        if (
            state.plan_revision < 1
            or state.approved_plan_revision != state.plan_revision
            or state.approved_plan_hash != state.plan_hash
        ):
            raise PipelineNeedsAttention("현재 계획 버전과 승인 버전이 일치하지 않습니다.")

    @staticmethod
    def _check_cancel(cancelled: Callable[[], bool]) -> None:
        if cancelled():
            raise PipelineCancelled("사용자가 작업을 중지했습니다.")

    def _require_no_user_input(
        self,
        run_id: str,
        stage_id: str,
        role_id: RoleId,
        value: RoleSummary | ReviewSummary,
        *,
        repository: GitRepository | None = None,
        before=None,
        on_dirty: Callable[[], None] | None = None,
    ) -> None:
        if value.needs_user_input:
            if repository is not None and before is not None:
                repository.assert_position(before, allow_worktree_changes=True)
                checkpoint = self.store.pipeline_workspace(run_id)
                plan = self.store.load_plan_revision(run_id, self.store.load_run(run_id).plan_revision)
                contract = StageContract.from_dict(plan['plan']['stages'][checkpoint['stage_index']])
                repository.assert_scope(contract.scope)
                current = repository.snapshot()
                self._save_workspace(repository, operation='', dirty_fingerprint=
                                     self._dirty_fingerprint(repository) if current.status else '')
            raise PipelineUserInputRequired(
                stage_id, role_id, value.needs_user_input
            )

    @staticmethod
    def _verification_failure_text(results) -> str:
        if not results:
            return "검증 결과가 비어 있습니다."
        failed = next((result for result in results if not result.passed), results[-1])
        detail = failed.stderr.strip() or failed.stdout.strip() or "출력 없음"
        execution = f"종료 코드: {failed.return_code}" if failed.started else "시작 실패(종료 코드 없음)"
        return (
            f"명령: {failed.command}\n{execution}\n"
            f"분류: {failed.failure_kind}\n{detail[-6000:]}"
        )

    def _preflight_toolchain(
        self,
        repository: GitRepository,
        contracts: tuple[StageContract, ...],
        run_id: str,
        cancelled: Callable[[], bool],
    ) -> ToolchainEnvironment | None:
        if self.toolchains is None:
            self._execution_environments[run_id] = None
            return None
        commands = tuple(
            command
            for contract in contracts
            for command in contract.verification_commands
        )
        environment = self.toolchains.preflight(
            repository,
            commands,
            operation_id=run_id,
            cancelled=cancelled,
        )
        self.artifacts.write_json(
            run_id, Path("toolchain-preflight.json"), environment.to_dict()
        )
        self._execution_environments[run_id] = environment
        return environment

    @staticmethod
    def _recovery_reference(run_id: str, stage_id: str) -> str:
        return f"artifacts\\{run_id}\\stages\\{stage_id}\\recovery.json"

    def _notify(self, channel: str, conversation_id: str, text: str) -> None:
        self.store.queue_outbound(channel, conversation_id, text)
        def wake_outbound():
            try:
                self.on_outbound()
            except Exception:
                # 발신함에 저장됐으므로 즉시 전송이 실패해도 다음 폴링에서 복구된다.
                return
        self.store.after_commit(wake_outbound)

    def _notify_for_run(self, run_id, channel, conversation_id, text):
        with self.store.transaction():
            if (self.store.load_conversation(channel, conversation_id) or {}).get('active_task_id') == run_id:
                self._notify(channel, conversation_id, text)

    def _set_role_for_run(self, run_id, channel, conversation_id, role):
        with self.store.transaction():
            if (self.store.load_conversation(channel, conversation_id) or {}).get('active_task_id') == run_id:
                self.store.set_conversation_role(channel, conversation_id, role)

    def _set_mode_for_run(self, run_id, channel, conversation_id, mode):
        with self.store.transaction():
            if (self.store.load_conversation(channel, conversation_id) or {}).get('active_task_id') == run_id:
                self.store.set_conversation_mode(channel, conversation_id, mode)
