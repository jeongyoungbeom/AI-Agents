from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol

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
)
from app.storage import ArtifactStore, StateStore


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
        if state.phase != RunPhase.DEVELOPING:
            raise PipelineNeedsAttention(f"재개할 수 없는 실행 상태입니다: {state.phase.value}")

        worktree = IsolatedGitWorktree(
            Path(state.repository), self.sandbox, run_id=run_id, cancelled=cancelled
        )
        lock_acquired = self.store.acquire_repository_lock(
            state.repository, run_id, lease_owner
        )
        if not lock_acquired:
            raise PipelineNeedsAttention("같은 저장소에서 다른 작업이 실행 중입니다.")

        def pulse() -> None:
            heartbeat()
            self.store.renew_repository_lock(state.repository, run_id, lease_owner)

        completed = False
        try:
            source_repository, source_snapshot = worktree.begin()
            self.artifacts.write_json(
                run_id,
                Path("worktree.json"),
                {"status": "source_checked", **worktree.record.to_dict()},
            )
            repository = worktree.create(source_snapshot)
            self.artifacts.write_json(
                run_id,
                Path("worktree.json"),
                {"status": "created", **worktree.record.to_dict()},
            )
            environment = self._preflight_toolchain(
                repository, contracts, run_id, cancelled
            )
            while True:
                self._check_cancel(cancelled)
                state = self.store.load_run(run_id)
                contract = contracts[state.stage_index]
                self.artifacts.save_contract(contract)
                self._run_stage(
                    state,
                    contract,
                    repository,
                    channel,
                    conversation_id,
                    cancelled,
                    pulse,
                    environment,
                    source_repository,
                    source_snapshot,
                    contracts,
                )
                state = self.store.load_run(run_id)
                if state.phase == RunPhase.COMPLETED:
                    completed = True
                    cleanup_error = worktree.cleanup()
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
                    self.store.set_conversation_role(
                        channel, conversation_id, RoleId.DEVELOPMENT.value
                    )
                    self._notify(channel, conversation_id, "모든 단계를 완료했습니다.")
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
            if not completed:
                self._record_worktree_recovery(run_id, worktree)
            self.store.release_repository_lock(state.repository, run_id, lease_owner)

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
        base = repository.preflight(
            require_branch=False, check_path_escapes=True
        )
        self.store.set_conversation_role(channel, conversation_id, RoleId.DEVELOPMENT.value)
        development_name = self._role_name(RoleId.DEVELOPMENT)
        review_name = self._role_name(RoleId.REVIEW)
        improvement_name = self._role_name(RoleId.IMPROVEMENT)
        self._notify(
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
            base,
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
            before=base,
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
        self.store.set_conversation_role(channel, conversation_id, RoleId.REVIEW.value)
        self._notify(
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
            self.store.set_conversation_role(
                channel, conversation_id, RoleId.DEVELOPMENT.value
            )
            self._notify(
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
            self.store.set_conversation_role(
                channel, conversation_id, RoleId.REVIEW.value
            )
            self._notify(
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
            self.store.set_conversation_role(
                channel, conversation_id, RoleId.IMPROVEMENT.value
            )
            self._notify(
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
            self._notify(
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
            self._notify(
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
            self.store.set_conversation_role(
                channel, conversation_id, RoleId.REVIEW.value
            )
            self._notify(
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
        source_applied_sha: str | None = None
        if state.stage_index + 1 == state.stage_count:
            source_applied_sha = self._apply_final_candidate(
                state.run_id,
                source_repository,
                source_snapshot,
                final_sha,
                contracts,
            )
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
            source_applied_sha=source_applied_sha,
        )
        state = self.machine.complete_stage(self.store.load_run(state.run_id))
        self.store.set_conversation_role(channel, conversation_id, RoleId.DEVELOPMENT.value)
        if state.phase == RunPhase.COMPLETED:
            self.store.set_conversation_mode(channel, conversation_id, "free_chat")
        self._notify(channel, conversation_id, f"{stage_number}단계를 완료했습니다.")
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
            applied_sha = source_repository.apply_candidate(
                source_snapshot, candidate_sha, scope
            )
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
        try:
            bundle = worktree.repository.recovery_bundle(worktree.source_snapshot)
        except GitRepositoryError:
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
        try:
            return repository.commit_scope(message, contract.scope)
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
        )
        self._assert_read_only_position(
            run_id,
            contract.stage_id,
            repository,
            baseline,
            RoleId.REVIEW,
            "review",
        )
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
        self.store.set_conversation_role(
            channel, conversation_id, RoleId.IMPROVEMENT.value
        )
        self._notify(
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
        self._notify(
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
        before = repository.snapshot()
        try:
            return self.verifier.run_all(
                repository,
                contract.verification_commands,
                cancelled=cancelled,
                heartbeat=heartbeat,
                operation_id=run_id,
                environment=environment,
            )
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
    ) -> HermesResult:
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
        while True:
            repository.assert_safe_worktree_paths()
            baseline = repository.snapshot()
            self._check_cancel(cancelled)
            reservation = self.budget.reserve(
                run_id,
                stage_id,
                role_id.value,
                "agent",
                input_estimate + output_cap,
            )
            try:
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
                )
                self.budget.record_usage(
                    run_id,
                    stage_id,
                    role_id.value,
                    "agent",
                    result.usage,
                    reservation=reservation,
                )
                if not allow_writes:
                    self._assert_read_only_position(
                        run_id,
                        stage_id,
                        repository,
                        baseline,
                        role_id,
                        "read-only-agent",
                    )
                return result
            except HermesCancelled:
                self.budget.release_reservation(reservation)
                raise
            except HermesExecutionError as exc:
                if exc.usage.total_tokens:
                    self.budget.record_usage(
                        run_id,
                        stage_id,
                        role_id.value,
                        "agent",
                        exc.usage,
                        reservation=reservation,
                    )
                else:
                    self.budget.release_reservation(reservation)
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
                self.budget.release_reservation(reservation)
                raise

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
        self, state, *, cancelled: Callable[[], bool] | None = None
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
        if repository.head_sha != state.repository_head_sha:
            raise PipelineNeedsAttention(
                "계획 승인 후 Git HEAD가 바뀌었습니다. 현재 코드 기준으로 다시 계획해야 합니다."
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
                try:
                    repository.assert_position(before, allow_worktree_changes=False)
                except GitRepositoryError as exc:
                    if on_dirty is not None:
                        on_dirty()
                    raise PipelineNeedsAttention(
                        "에이전트가 사용자 질문 전에 미커밋 파일 변경을 남겼습니다. "
                        "안전하게 재개할 수 없어 질문 답변 대기로 전환하지 않았습니다."
                    ) from exc
            raise PipelineUserInputRequired(
                stage_id, role_id, value.needs_user_input
            )

    @staticmethod
    def _verification_failure_text(results) -> str:
        if not results:
            return "검증 결과가 비어 있습니다."
        failed = next((result for result in results if not result.passed), results[-1])
        detail = failed.stderr.strip() or failed.stdout.strip() or "출력 없음"
        return (
            f"명령: {failed.command}\n종료 코드: {failed.return_code}\n"
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
        return environment

    @staticmethod
    def _recovery_reference(run_id: str, stage_id: str) -> str:
        return f"artifacts\\{run_id}\\stages\\{stage_id}\\recovery.json"

    def _notify(self, channel: str, conversation_id: str, text: str) -> None:
        self.store.queue_outbound(channel, conversation_id, text)
        try:
            self.on_outbound()
        except Exception:
            # 발신함에 저장됐으므로 즉시 전송이 실패해도 다음 폴링에서 복구된다.
            return
