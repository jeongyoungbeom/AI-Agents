from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Protocol

from app.contracts import RunPhase, RunState
from app.contracts.models import utc_now
from app.storage import StateStore


class EventSink(Protocol):
    def emit(
        self,
        run_id: str,
        event_type: str,
        message: str,
        *,
        stage_id: str = "",
        role_id: str = "",
        status: str = "",
        data: dict | None = None,
    ) -> None: ...


class InvalidTransition(RuntimeError):
    pass


class ApprovalRequired(InvalidTransition):
    pass


ALLOWED_TRANSITIONS: dict[RunPhase, frozenset[RunPhase]] = {
    RunPhase.DISCUSSING: frozenset(
        {RunPhase.WAITING_APPROVAL, RunPhase.CANCELLED, RunPhase.FAILED}
    ),
    RunPhase.WAITING_APPROVAL: frozenset(
        {RunPhase.DEVELOPING, RunPhase.DISCUSSING, RunPhase.CANCELLED, RunPhase.FAILED}
    ),
    RunPhase.DEVELOPING: frozenset(
        {RunPhase.REVIEWING, RunPhase.PAUSED, RunPhase.FAILED, RunPhase.CANCELLED}
    ),
    RunPhase.REVIEWING: frozenset(
        {
            RunPhase.DEVELOPING,
            RunPhase.IMPROVING,
            RunPhase.VERIFYING,
            RunPhase.PAUSED,
            RunPhase.FAILED,
            RunPhase.CANCELLED,
        }
    ),
    RunPhase.IMPROVING: frozenset(
        {
            RunPhase.REVIEWING,
            RunPhase.VERIFYING,
            RunPhase.PAUSED,
            RunPhase.FAILED,
            RunPhase.CANCELLED,
        }
    ),
    RunPhase.VERIFYING: frozenset(
        {
            RunPhase.STAGE_COMPLETED,
            RunPhase.IMPROVING,
            RunPhase.REVIEWING,
            RunPhase.PAUSED,
            RunPhase.FAILED,
            RunPhase.CANCELLED,
        }
    ),
    RunPhase.STAGE_COMPLETED: frozenset(
        {
            RunPhase.DEVELOPING,
            RunPhase.COMPLETED,
            RunPhase.PAUSED,
            RunPhase.FAILED,
            RunPhase.CANCELLED,
        }
    ),
    RunPhase.PAUSED: frozenset(
        {
            RunPhase.DISCUSSING,
            RunPhase.WAITING_APPROVAL,
            RunPhase.DEVELOPING,
            RunPhase.REVIEWING,
            RunPhase.IMPROVING,
            RunPhase.VERIFYING,
            RunPhase.STAGE_COMPLETED,
            RunPhase.CANCELLED,
            RunPhase.FAILED,
        }
    ),
    RunPhase.FAILED: frozenset(),
    RunPhase.CANCELLED: frozenset(),
    RunPhase.COMPLETED: frozenset(),
}


ACTIVE_PHASES = {
    RunPhase.DEVELOPING,
    RunPhase.REVIEWING,
    RunPhase.IMPROVING,
    RunPhase.VERIFYING,
}


class RunStateMachine:
    def __init__(
        self,
        store: StateStore,
        event_sink: EventSink | None = None,
        approval_phrase: str = "개발 시작해",
    ):
        self.store = store
        self.event_sink = event_sink
        self.approval_phrase = approval_phrase.strip()
        if not self.approval_phrase:
            raise ValueError("approval phrase cannot be blank")

    def create_run(
        self,
        run_id: str,
        *,
        repository: str = "",
        repository_identity: str = "",
        repository_head_sha: str = "",
        repository_approved: bool = False,
        objective: str = "",
        stage_count: int = 1,
    ) -> RunState:
        state = RunState(
            run_id=run_id,
            repository=repository,
            repository_identity=repository_identity,
            repository_head_sha=repository_head_sha,
            repository_approved=repository_approved,
            objective=objective,
            stage_count=stage_count,
        )
        self.store.create_run(state)
        self._emit(state, "RUN_CREATED", "작업 대화를 시작했습니다.")
        return state

    def register_plan(self, state: RunState, plan: dict) -> RunState:
        stages = plan.get("stages")
        if not isinstance(stages, list) or not stages:
            raise ValueError("plan must contain at least one stage")
        canonical = json.dumps(
            plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        plan_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        revision = state.plan_revision + 1
        with self.store.transaction():
            self.store.save_plan_revision(state.run_id, revision, plan_hash, plan)
            saved = self.store.save_run(
                replace(
                    state,
                    stage_count=len(stages),
                    stage_index=0,
                    plan_revision=revision,
                    plan_hash=plan_hash,
                    approval_granted=False,
                    approved_plan_revision=0,
                    approved_plan_hash="",
                    approved_at="",
                    approved_by="",
                )
            )
        return saved

    def request_approval(self, state: RunState) -> RunState:
        if not state.repository or not state.repository_approved:
            raise ApprovalRequired("repository must be explicitly approved first")
        if state.plan_revision < 1 or not state.plan_hash:
            raise ApprovalRequired("a versioned plan is required before approval")
        return self.transition(
            state,
            RunPhase.WAITING_APPROVAL,
            message="개발 계획이 준비되어 명시적 승인을 기다립니다.",
        )

    def approve(self, state: RunState, phrase: str, approved_by: str) -> RunState:
        if state.phase != RunPhase.WAITING_APPROVAL:
            raise InvalidTransition("approval is only accepted while waiting for approval")
        if phrase.strip() != self.approval_phrase:
            raise ApprovalRequired(
                f"exact approval phrase required: {self.approval_phrase}"
            )
        if not approved_by.strip():
            raise ApprovalRequired("approved_by is required")
        if state.plan_revision < 1 or not state.plan_hash:
            raise ApprovalRequired("there is no current plan to approve")
        with self.store.transaction():
            self.store.approve_plan_revision(
                state.run_id, state.plan_revision, state.plan_hash
            )
            approved = replace(
                state,
                approval_granted=True,
                approved_plan_revision=state.plan_revision,
                approved_plan_hash=state.plan_hash,
                approved_at=utc_now(),
                approved_by=approved_by.strip(),
            )
            approved = self.store.save_run(approved)
        self._emit(
            approved,
            "APPROVAL_GRANTED",
            "사용자가 개발 실행을 명시적으로 승인했습니다.",
            data={"approved_by": approved.approved_by},
        )
        return approved

    def transition(
        self,
        state: RunState,
        target: RunPhase,
        *,
        message: str = "",
        last_error: str = "",
    ) -> RunState:
        if target not in ALLOWED_TRANSITIONS[state.phase]:
            raise InvalidTransition(f"invalid transition: {state.phase.value} -> {target.value}")
        if target == RunPhase.DEVELOPING:
            if not state.approval_granted:
                raise ApprovalRequired("development cannot start without explicit approval")
            if not state.repository or not state.repository_approved:
                raise ApprovalRequired("the selected repository has not been approved")
            if state.plan_revision < 1 or not state.plan_hash:
                raise ApprovalRequired("a versioned plan must be approved")
            if (
                state.approved_plan_revision != state.plan_revision
                or state.approved_plan_hash != state.plan_hash
            ):
                raise ApprovalRequired("the current plan revision has not been approved")
        next_state = replace(
            state,
            phase=target,
            resume_phase=None if state.phase == RunPhase.PAUSED else state.resume_phase,
            last_error=last_error,
        )
        next_state = self.store.save_run(next_state)
        self._emit(
            next_state,
            "PHASE_CHANGED",
            message or f"상태 변경: {state.phase.value} -> {target.value}",
            status=target.value,
            data={"from": state.phase.value, "to": target.value},
        )
        return next_state

    def pause(self, state: RunState, reason: str) -> RunState:
        if state.phase not in ACTIVE_PHASES and state.phase != RunPhase.STAGE_COMPLETED:
            raise InvalidTransition(f"cannot pause from {state.phase.value}")
        paused = replace(state, phase=RunPhase.PAUSED, resume_phase=state.phase)
        paused = self.store.save_run(paused)
        self._emit(
            paused,
            "RUN_PAUSED",
            reason,
            status=RunPhase.PAUSED.value,
            data={"resume_phase": state.phase.value},
        )
        return paused

    def resume(self, state: RunState) -> RunState:
        if state.phase != RunPhase.PAUSED or state.resume_phase is None:
            raise InvalidTransition("run is not resumable")
        target = state.resume_phase
        resumed = replace(state, phase=target, resume_phase=None)
        resumed = self.store.save_run(resumed)
        self._emit(
            resumed,
            "RUN_RESUMED",
            f"체크포인트에서 {target.value} 상태로 재개했습니다.",
            status=target.value,
        )
        return resumed

    def complete_stage(self, state: RunState) -> RunState:
        completed = self.transition(
            state,
            RunPhase.STAGE_COMPLETED,
            message=f"{state.stage_index + 1}단계를 완료했습니다.",
        )
        if completed.stage_index + 1 >= completed.stage_count:
            return self.transition(
                completed,
                RunPhase.COMPLETED,
                message="모든 단계를 완료했습니다.",
            )
        next_state = replace(completed, stage_index=completed.stage_index + 1)
        next_state = self.store.save_run(next_state)
        return self.transition(
            next_state,
            RunPhase.DEVELOPING,
            message=f"{next_state.stage_index + 1}단계 개발을 시작합니다.",
        )

    def fail(self, state: RunState, error: str) -> RunState:
        return self.transition(
            state,
            RunPhase.FAILED,
            message=f"작업을 중단했습니다: {error}",
            last_error=error,
        )

    def _emit(
        self,
        state: RunState,
        event_type: str,
        message: str,
        *,
        status: str = "",
        data: dict | None = None,
    ) -> None:
        if self.event_sink is None:
            return
        self.event_sink.emit(
            state.run_id,
            event_type,
            message,
            stage_id=f"stage-{state.stage_index + 1:03d}",
            status=status or state.phase.value,
            data=data,
        )
