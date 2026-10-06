from __future__ import annotations

import json
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from app.contracts import RunPhase, TokenUsage
from app.contracts.models import utc_now
from app.storage import StateStore


class BudgetExceeded(RuntimeError):
    pass


def conservative_prompt_tokens(text: str) -> int:
    """Return a tokenizer-independent upper bound for UTF-8 prompt text."""
    return max(1, len(text.encode("utf-8")))


def optional_positive(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive or null")
    return parsed


@dataclass(frozen=True)
class BudgetPolicy:
    calibration_mode: bool = True
    conversation_tokens: int | None = None
    per_stage_tokens: int | None = None
    whole_task_tokens: int | None = None
    completion_reserve_tokens: int | None = None
    provider_input_overhead_tokens: int = 0
    warning_threshold_percent: int = 80
    retries: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "conversation_tokens",
            "per_stage_tokens",
            "whole_task_tokens",
            "completion_reserve_tokens",
        ):
            value = getattr(self, field_name)
            if value is not None and value <= 0:
                raise ValueError(f"{field_name} must be positive or null")
        for category, count in self.retries.items():
            if not category.strip() or int(count) < 0:
                raise ValueError("retry categories must be named and non-negative")
        if not 1 <= self.warning_threshold_percent <= 99:
            raise ValueError("warning_threshold_percent must be between 1 and 99")
        if (
            self.whole_task_tokens is not None
            and self.completion_reserve_tokens is not None
            and self.completion_reserve_tokens >= self.whole_task_tokens
        ):
            raise ValueError("completion reserve must be smaller than whole task limit")
        if self.provider_input_overhead_tokens < 0:
            raise ValueError("provider_input_overhead_tokens cannot be negative")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BudgetPolicy":
        limits = dict(value.get("token_budget", {}))
        return cls(
            calibration_mode=bool(value.get("calibration_mode", True)),
            conversation_tokens=optional_positive(
                limits.get("conversation"), "conversation_tokens"
            ),
            per_stage_tokens=optional_positive(
                limits.get("per_stage"), "per_stage_tokens"
            ),
            whole_task_tokens=optional_positive(
                limits.get("whole_task"), "whole_task_tokens"
            ),
            completion_reserve_tokens=optional_positive(
                limits.get("completion_reserve"), "completion_reserve_tokens"
            ),
            provider_input_overhead_tokens=int(
                limits.get("provider_input_overhead", 0)
            ),
            warning_threshold_percent=int(
                limits.get("warning_threshold_percent", 80)
            ),
            retries={
                str(category): int(count)
                for category, count in dict(value.get("retry", {})).items()
            },
        )

    @classmethod
    def load(cls, path: Path) -> "BudgetPolicy":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    reason: str
    task_used: int
    stage_used: int
    projected_task_total: int
    projected_stage_total: int


@dataclass(frozen=True)
class BudgetWarning:
    """One deduplicated threshold alert that can be delivered to the owner."""

    scope: str
    used_tokens: int
    limit_tokens: int
    percent: int
    stage_id: str = ""
    threshold_percent: int = 0


@dataclass(frozen=True)
class BudgetReservation:
    """A durable invocation estimate held before a model call starts."""

    reservation_id: str = ""
    reserved_tokens: int = 0

    @property
    def active(self) -> bool:
        return bool(self.reservation_id)


@dataclass(frozen=True)
class BudgetSnapshot:
    run_id: str
    total_tokens: int
    actual_tokens: int
    estimated_tokens: int
    conversation_tokens: int
    current_stage_id: str = ""
    current_stage_tokens: int = 0


class BudgetManager:
    @contextmanager
    def transaction(self):
        # Gateway 제어의 바깥 transaction과 같은 순서로 획득한다.
        with self.store.transaction(), self._lock:
            yield

    def __init__(
        self,
        policy: BudgetPolicy,
        store: StateStore,
        *,
        warning_sink: Callable[[tuple[BudgetWarning, ...], str], None] | None = None,
    ):
        self.policy = policy
        self.store = store
        self.warning_sink = warning_sink
        self._lock = threading.RLock()

    def set_warning_sink(
        self, warning_sink: Callable[[tuple[BudgetWarning, ...], str], None] | None
    ) -> None:
        self.warning_sink = warning_sink

    def can_spend(
        self,
        run_id: str,
        stage_id: str,
        estimated_tokens: int,
        *,
        category: str = "agent",
        use_completion_reserve: bool = False,
    ) -> BudgetDecision:
        if estimated_tokens < 0:
            raise ValueError("estimated_tokens cannot be negative")
        if not use_completion_reserve and self.store.has_budget_anomaly(run_id):
            used = self.store.usage_total(run_id)
            stage = self.store.usage_total(run_id, stage_id)
            events = self.store.budget_anomalies(run_id)
            if any(event['event_type'] == 'MODEL_USAGE_UNKNOWN' for event in events):
                reason = '이전 호출의 실제 사용량이 미확인입니다. 비용 확인 전 새 일반 호출을 차단합니다.'
            else:
                overrun = next(event['data'] for event in reversed(events)
                               if event['event_type'] == 'BUDGET_RESERVATION_EXCEEDED')
                reason = '이전 호출의 예약 초과가 확인되어 새 일반 호출을 차단합니다.'
                if isinstance(overrun.get('reported_tokens'), int) and isinstance(overrun.get('reserved_tokens'), int):
                    reason += (f" 실제 보고 {overrun['reported_tokens']:,}토큰 / "
                               f"추정 예약 {overrun['reserved_tokens']:,}토큰.")
                reason += ' 정산과 예산 검토 후 재개 여부를 확인해야 합니다.'
            reason += '\n' + self.render_status(run_id, stage_id=stage_id)
            return BudgetDecision(False, reason,
                                  used, stage, used + estimated_tokens, stage + estimated_tokens)
        task_used = self.store.budget_usage_total(run_id) + self.store.reserved_token_total(run_id)
        stage_used = self.store.budget_usage_total(run_id, stage_id) + self.store.reserved_token_total(
            run_id, stage_id=stage_id
        )
        projected_task = task_used + estimated_tokens
        projected_stage = stage_used + estimated_tokens
        reserve = 0 if use_completion_reserve else self.policy.completion_reserve_tokens or 0

        if category == "conversation" and self.policy.conversation_tokens is not None:
            conversation_used = self._category_total(
                run_id, "conversation"
            ) + self.store.reserved_token_total(
                run_id, category="conversation"
            )
            if conversation_used + estimated_tokens > self.policy.conversation_tokens:
                return BudgetDecision(
                    False,
                    "conversation token limit would be exceeded",
                    task_used,
                    stage_used,
                    projected_task,
                    projected_stage,
                )
        if (
            self.policy.per_stage_tokens is not None
            and projected_stage > self.policy.per_stage_tokens
        ):
            return BudgetDecision(
                False,
                "stage token limit would be exceeded",
                task_used,
                stage_used,
                projected_task,
                projected_stage,
            )
        if (
            self.policy.whole_task_tokens is not None
            and projected_task + reserve > self.policy.whole_task_tokens
        ):
            return BudgetDecision(
                False,
                "task token limit or completion reserve would be exceeded",
                task_used,
                stage_used,
                projected_task,
                projected_stage,
            )
        return BudgetDecision(
            True,
            "calibration mode" if self.policy.calibration_mode else "within budget",
            task_used,
            stage_used,
            projected_task,
            projected_stage,
        )

    def reserve(
        self,
        run_id: str,
        stage_id: str,
        role_id: str,
        category: str,
        worst_case_tokens: int,
        *,
        use_completion_reserve: bool = False,
        completion_allowance_tokens: int = 0,
    ) -> BudgetReservation:
        """Atomically reserve the estimated invocation cost before a model.

        Reservations are deliberately excluded from the durable *actual usage*
        ledger. They only close the race between parallel preflight checks; a
        completed call settles into real usage and a cancelled call releases its
        allowance.
        """
        if worst_case_tokens < 1 or completion_allowance_tokens < 0:
            raise ValueError("worst_case_tokens must be positive")
        with self.transaction():
            decision = self.can_spend(
                run_id,
                stage_id,
                worst_case_tokens + completion_allowance_tokens,
                category=category,
                use_completion_reserve=use_completion_reserve,
            )
            if not decision.allowed:
                raise BudgetExceeded(decision.reason)
            reservation = BudgetReservation(uuid.uuid4().hex, worst_case_tokens)
            self.store.create_token_reservation(
                reservation.reservation_id,
                run_id,
                stage_id,
                role_id,
                category,
                reservation.reserved_tokens,
            )
        return reservation

    def release_reservation(self, reservation: BudgetReservation) -> None:
        if not reservation.active:
            return
        with self.transaction():
            self.store.release_token_reservation(reservation.reservation_id)

    def record_usage(
        self,
        run_id: str,
        stage_id: str,
        role_id: str,
        category: str,
        usage: TokenUsage,
        *,
        reservation: BudgetReservation | None = None,
    ) -> tuple[BudgetWarning, ...]:
        reservation_exceeded = bool(
            reservation is not None
            and reservation.active
            and usage.total_tokens > reservation.reserved_tokens
        )
        with self.transaction():
            if reservation is not None and reservation.active:
                if not self.store.settle_token_reservation(reservation.reservation_id):
                    return ()
            self.store.add_usage(run_id, stage_id, role_id, category, usage)
            if reservation_exceeded:
                self.store.append_event(run_id, utc_now(), "BUDGET_RESERVATION_EXCEEDED",
                    "실제 사용량이 실행 전 추정 예약을 넘었습니다. 결과를 보존하고 후속 실행을 차단합니다.",
                    data={"stage_id": stage_id, "role_id": role_id,
                     "reservation_id": reservation.reservation_id,
                     "reserved_tokens": reservation.reserved_tokens,
                     "reported_tokens": usage.total_tokens})
            state = self.store.load_run(run_id)
            self.store.save_run(
                replace(state, total_tokens=self.store.usage_total(run_id))
            )
            warnings = self._claim_warnings(run_id, stage_id)
        self._deliver_pending_warnings(run_id)
        return warnings

    def recover_interrupted_calls(self, run_id: str, stage_id: str, request_key: str = "") -> None:
        """만료된 worker의 호출만 복구한다. 미확인 비용은 0으로 확정하지 않는다."""
        with self.transaction():
            for call in self.store.running_model_calls(run_id, stage_id, request_key):
                reservation = BudgetReservation(call["reservation_id"], call["reserved_tokens"])
                self.record_usage(run_id, stage_id, call["role_id"], call["category"],
                                  TokenUsage(total_tokens=call["reserved_tokens"], estimated=True),
                                  reservation=reservation)
                self.store.append_event(run_id, utc_now(), "MODEL_USAGE_UNKNOWN",
                                        "중단된 모델 호출의 실제 비용을 확인하지 못했습니다.",
                                        data={"call_id": call["call_id"], "stage_id": stage_id,
                                         "source": "interrupted_invocation_estimate"})
                self.store.finish_model_call(call["call_id"], "INTERRUPTED", {})

    def retry_pending_warnings(self) -> None:
        """Retry warnings that were claimed but never made it into the outbox."""
        for run_id in self.store.pending_budget_warning_run_ids():
            self._deliver_pending_warnings(run_id)

    def _deliver_pending_warnings(self, run_id: str) -> None:
        if self.warning_sink is None:
            return
        warnings = self._pending_warnings(run_id)
        if not warnings:
            return
        try:
            self.warning_sink(warnings, run_id)
        except Exception:
            # Warning claims intentionally remain PENDING until the sink has
            # atomically created durable outbound rows. A later usage event or
            # gateway restart will retry them.
            return

    def _pending_warnings(self, run_id: str) -> tuple[BudgetWarning, ...]:
        warnings: list[BudgetWarning] = []
        for claim in self.store.pending_budget_warning_claims(run_id):
            warning = self._warning_for_claim(run_id, claim)
            if warning is not None:
                warnings.append(warning)
        return tuple(warnings)

    def _warning_for_claim(
        self, run_id: str, claim: dict[str, Any]
    ) -> BudgetWarning | None:
        scope = str(claim["scope"])
        stage_id = str(claim["stage_id"])
        threshold = int(claim["threshold_percent"])
        if scope == "conversation":
            limit = self.policy.conversation_tokens
            used = self.store.usage_total_by_category(run_id, "conversation")
        elif scope == "stage":
            limit = self.policy.per_stage_tokens
            used = self.store.usage_total(run_id, stage_id)
        elif scope == "task":
            limit = self._task_usable_limit() if self.policy.whole_task_tokens else None
            used = self.store.usage_total(run_id)
        else:
            return None
        if limit is None or limit < 1:
            return None
        percent = (used * 100) // limit
        return BudgetWarning(
            scope,
            used,
            limit,
            min(percent, 100),
            stage_id,
            threshold,
        )

    @staticmethod
    def render_warning(warning: BudgetWarning) -> str:
        labels = {
            "conversation": "대화",
            "stage": "현재 단계",
            "task": "전체 작업",
        }
        label = labels[warning.scope]
        return (
            f"⚠️ 토큰 한도 경고: {label} 누적 {warning.used_tokens:,} / "
            f"{warning.limit_tokens:,} ({warning.percent}%) 사용 중입니다.\n"
            "필요하면 '사용량' 또는 '상태'로 초기화 후 예산과 누적 내역을 확인해 주세요."
        )

    def snapshot(self, run_id: str, *, stage_id: str = "") -> BudgetSnapshot:
        totals = self.store.usage_breakdown(run_id)
        return BudgetSnapshot(
            run_id=run_id,
            total_tokens=totals["total_tokens"],
            actual_tokens=totals["actual_tokens"],
            estimated_tokens=totals["estimated_tokens"],
            conversation_tokens=self._category_total(run_id, "conversation"),
            current_stage_id=stage_id,
            current_stage_tokens=self.store.budget_usage_total(run_id, stage_id)
            if stage_id
            else 0,
        )

    def render_status(self, run_id: str, *, stage_id: str = "") -> str:
        """A compact Korean summary for the Telegram status command."""
        snapshot = self.snapshot(run_id, stage_id=stage_id)
        source = (
            f"실제 {snapshot.actual_tokens:,} / 추정 {snapshot.estimated_tokens:,}"
            if snapshot.estimated_tokens
            else "실제 집계"
        )
        lines = [f"토큰 사용: {snapshot.total_tokens:,} ({source})"]
        budget_used = self.store.budget_usage_total(run_id)
        if self.store.budget_usage_cursor(run_id):
            lines.append(f"초기화 후 예산 사용: {budget_used:,} (누적 사용 기록 보존)")
        if self.policy.conversation_tokens is None:
            lines.append(f"대화 한도: 측정 중 ({snapshot.conversation_tokens:,} 사용)")
        else:
            lines.append(
                self._render_limit("대화 한도", snapshot.conversation_tokens,
                                   self.policy.conversation_tokens)
            )
        if stage_id:
            if self.policy.per_stage_tokens is None:
                lines.append(f"현재 단계: 측정 중 ({snapshot.current_stage_tokens:,} 사용)")
            else:
                lines.append(
                    self._render_limit(
                        "현재 단계", snapshot.current_stage_tokens,
                        self.policy.per_stage_tokens,
                    )
                )
        if self.policy.whole_task_tokens is None:
            lines.append("전체 작업 한도: 측정 중")
        else:
            usable = self._task_usable_limit()
            reserve = self.policy.completion_reserve_tokens or 0
            suffix = f", 완료 여유 {reserve:,}" if reserve else ""
            lines.append(
                self._render_limit("전체 작업 한도", budget_used, usable)
                + suffix
            )
        for event in self.store.budget_anomalies(run_id):
            if event['event_type'] == 'MODEL_USAGE_UNKNOWN':
                lines.append(f"미확인 비용 이벤트 {event['event_id']}: 비용 확인 전 초기화·재개 차단")
            else:
                data = event['data']
                lines.append(f"예약 초과 이벤트 {event['event_id']}: 실제 {data.get('reported_tokens')} / "
                             f"예약 {data.get('reserved_tokens')}; 확인 명령: "
                             f"/budget_ack {event['event_id']} {data.get('reported_tokens')}")
        return "\n".join(lines)

    def _budget_control_state(self, run_id: str, channel: str, conversation_id: str,
                              user_id: str, request_id: str):
        binding = self.store.load_conversation(channel, conversation_id)
        if (not request_id.strip() or not binding or binding['user_id'] != user_id
                or binding['active_task_id'] != run_id):
            raise BudgetExceeded('현재 작업 소유자의 명시적 요청만 예산을 변경할 수 있습니다.')
        state = self.store.load_run(run_id)
        job = self.store.pipeline_job(run_id)
        checkpoint = self.store.pipeline_workspace(run_id)
        if (state.phase != RunPhase.PAUSED or not state.approval_granted
                or not job or job['status'] != 'NEEDS_ATTENTION' or job['lease_owner']
                or job['channel'] != channel or job['conversation_id'] != conversation_id
                or not checkpoint or checkpoint['plan_hash'] != state.approved_plan_hash
                or checkpoint['stage_index'] != state.stage_index):
            raise BudgetExceeded('같은 승인 계획과 checkpoint에서 중단이 확정된 작업만 변경할 수 있습니다.')
        invocation = checkpoint.get('invocation') or {}
        if (invocation.get('status') in {'RUNNING', 'UNKNOWN'}
                or self.store.has_active_model_calls(run_id)
                or self.store.reserved_token_total(run_id)):
            raise BudgetExceeded('활성 호출·미확인 호출·예약을 보존합니다. 예산 변경을 차단합니다.')
        if any(event['event_type'] == 'MODEL_USAGE_UNKNOWN'
               for event in self.store.budget_anomalies(run_id)):
            raise BudgetExceeded('이전 호출의 실제 사용량이 미확인입니다. 초기화·초과 확인을 차단합니다.')
        return state, checkpoint

    def reset_budget(self, run_id: str, *, channel: str, conversation_id: str,
                     user_id: str, request_id: str) -> None:
        """원장을 보존하고 현재 작업의 새 예산 계산 기준만 확정한다."""
        with self.transaction():
            state, checkpoint = self._budget_control_state(
                run_id, channel, conversation_id, user_id, request_id)
            if any(event['event_type'] == 'BUDGET_BASELINE_RESET'
                   and event['data'].get('request_id') == request_id
                   and event['data'].get('channel') == channel
                   and event['data'].get('conversation_id') == conversation_id
                   and event['data'].get('user_id') == user_id
                   for event in self.store.budget_control_events(run_id)):
                return
            cursor = self.store.latest_usage_id(run_id)
            if cursor == self.store.budget_usage_cursor(run_id):
                return
            self.store.append_event(run_id, utc_now(), 'BUDGET_BASELINE_RESET',
                '사용자 요청으로 예산 계산 기준을 초기화했습니다. 원 사용량과 초과 차단은 보존합니다.',
                data={'usage_id': cursor, 'user_id': user_id, 'request_id': request_id,
                      'channel': channel, 'conversation_id': conversation_id,
                      'total_tokens': self.store.usage_total(run_id), 'plan_hash': state.plan_hash,
                      'stage_index': state.stage_index, 'candidate_sha': checkpoint['candidate_sha']})

    def acknowledge_overrun(self, run_id: str, event_id: int, reported_tokens: int,
                            *, channel: str, conversation_id: str, user_id: str,
                            request_id: str) -> None:
        """특정 reported 초과만 확인한다. 새 이벤트·unknown은 계속 차단한다."""
        with self.transaction():
            state, checkpoint = self._budget_control_state(
                run_id, channel, conversation_id, user_id, request_id)
            events = self.store.budget_control_events(run_id)
            event = next((item for item in events if item['event_id'] == event_id
                          and item['event_type'] == 'BUDGET_RESERVATION_EXCEEDED'), None)
            data = event['data'] if event else {}
            if (type(reported_tokens) is not int or type(data.get('reported_tokens')) is not int
                    or type(data.get('reserved_tokens')) is not int
                    or reported_tokens != data['reported_tokens']
                    or reported_tokens <= data['reserved_tokens']):
                raise BudgetExceeded('현재 작업의 초과 이벤트 ID와 실제 보고 토큰을 정확히 확인해야 합니다.')
            if any(item['event_type'] == 'BUDGET_OVERRUN_ACKNOWLEDGED'
                   and item['data']['event_id'] == event_id for item in events):
                return
            invocation = checkpoint.get('invocation') or {}
            call = self.store.model_call(invocation.get('call_id', ''))
            reservation = self.store.token_reservation(call['reservation_id']) if call else None
            result = json.loads(call['result_json']) if call else {}
            reusable_call_id = ''
            if (call and call['status'] == 'COMPLETED' and reservation
                    and reservation['status'] == 'SETTLED'
                    and reservation['reserved_tokens'] == data['reserved_tokens']
                    and call['stage_id'] == data.get('stage_id')
                    and call['role_id'] == data.get('role_id')
                    and result.get('usage', {}).get('total_tokens') == reported_tokens
                    and not result.get('usage', {}).get('estimated', True)
                    and data.get('reservation_id', call['reservation_id']) == call['reservation_id']):
                reusable_call_id = call['call_id']
            self.store.append_event(run_id, utc_now(), 'BUDGET_OVERRUN_ACKNOWLEDGED',
                '사용자가 보고된 예약 초과 비용을 확인했습니다. 수동 재개와 기존 검사는 별도로 필요합니다.',
                data={'event_id': event_id, 'reported_tokens': reported_tokens,
                      'reserved_tokens': data['reserved_tokens'], 'user_id': user_id,
                      'request_id': request_id, 'channel': channel, 'conversation_id': conversation_id,
                      'plan_hash': state.plan_hash, 'candidate_sha': checkpoint['candidate_sha'],
                      'call_id': reusable_call_id, 'stage_id': data.get('stage_id'),
                      'role_id': data.get('role_id')})

    def can_retry(self, run_id: str, stage_id: str, category: str) -> bool:
        limit = self.policy.retries.get(category, 0)
        return self.store.retry_count(run_id, stage_id, category) < limit

    def record_retry(
        self, run_id: str, stage_id: str, category: str, reason: str
    ) -> int:
        with self.transaction():
            if not self.can_retry(run_id, stage_id, category):
                raise RuntimeError(f"retry limit reached for {category}")
            return self.store.add_retry(run_id, stage_id, category, reason)

    def _category_total(self, run_id: str, category: str) -> int:
        return self.store.budget_usage_total(run_id, category=category)

    def _claim_warnings(
        self, run_id: str, stage_id: str
    ) -> tuple[BudgetWarning, ...]:
        threshold = self.policy.warning_threshold_percent
        candidates: list[tuple[str, str, int, int]] = []
        conversation_limit = self.policy.conversation_tokens
        if conversation_limit is not None:
            candidates.append(
                ("conversation", "", self.store.usage_total_by_category(run_id, "conversation"), conversation_limit)
            )
        if self.policy.per_stage_tokens is not None:
            candidates.append(
                ("stage", stage_id, self.store.usage_total(run_id, stage_id), self.policy.per_stage_tokens)
            )
        if self.policy.whole_task_tokens is not None:
            candidates.append(
                ("task", "", self.store.usage_total(run_id), self._task_usable_limit())
            )

        warnings: list[BudgetWarning] = []
        for scope, warning_stage, used, limit in candidates:
            percent = (used * 100) // limit
            if percent < threshold:
                continue
            if self.store.claim_budget_warning(
                run_id, scope, warning_stage, threshold
            ):
                warnings.append(
                    BudgetWarning(
                        scope,
                        used,
                        limit,
                        min(percent, 100),
                        warning_stage,
                        threshold,
                    )
                )
        return tuple(warnings)

    def _task_usable_limit(self) -> int:
        if self.policy.whole_task_tokens is None:
            raise RuntimeError("whole task limit is not configured")
        return self.policy.whole_task_tokens - (self.policy.completion_reserve_tokens or 0)

    @staticmethod
    def _render_limit(label: str, used: int, limit: int) -> str:
        percent = (used * 100) // limit if limit else 0
        return f"{label}: {used:,} / {limit:,} ({percent}%)"
