from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from app.contracts import RoleId, TokenUsage
from app.services.budget import (
    BudgetExceeded,
    BudgetManager,
    BudgetReservation,
    conservative_prompt_tokens,
)
from app.services.context import ContextBundle
from app.services.git import GitRepositoryError
from app.services.hermes import HermesCancelled, HermesExecutionError
from app.services.logging.redaction import SecretRedactor

from .models import (
    AgentReply,
    IncomingMessage,
    TeamConversationBatchResult,
    TeamConversationRequest,
)
from .errors import InvalidAgentResponse
from .ports import AgentConversationBackend, TeamConversationBackend


class GovernedAgentBackend:
    """모든 대화형 모델 호출에 예산 기록과 제한된 기술 재시도를 강제한다."""

    def __init__(
        self,
        delegate: AgentConversationBackend,
        budget: BudgetManager,
        *,
        response_reserve_tokens: int = 1024,
    ):
        if response_reserve_tokens < 1:
            raise ValueError("response reserve must be positive")
        self.delegate = delegate
        self.budget = budget
        self.response_reserve_tokens = response_reserve_tokens
        self.redactor = SecretRedactor()
        self.supports_cancellation = True

    def respond(
        self,
        state,
        context: ContextBundle,
        message: IncomingMessage,
        *,
        cancelled=None,
    ) -> AgentReply:
        stage_id = f"stage-{state.stage_index + 1:03d}"
        input_estimate = self._input_upper_bound(state, context, message)
        while True:
            reservation = self._reserve(state, stage_id, input_estimate)
            try:
                reply = self._invoke_delegate(
                    state, context, message, cancelled=cancelled
                )
                break
            except HermesCancelled:
                self.budget.release_reservation(reservation)
                raise
            except GitRepositoryError:
                # 읽기 전용 역할의 변경 감지는 안전 위반이므로 자동 재실행하지 않는다.
                self.budget.release_reservation(reservation)
                raise
            except Exception as exc:
                self._record_execution_failure_usage(
                    state, stage_id, exc, reservation
                )
                if not self.budget.can_retry(
                    state.run_id, stage_id, "technical_error"
                ):
                    raise
                self._require_budget(state, stage_id, input_estimate)
                self.budget.record_retry(
                    state.run_id,
                    stage_id,
                    "technical_error",
                    self.redactor.text(type(exc).__name__),
                )

        usage = reply.usage
        if usage.total_tokens == 0:
            usage = TokenUsage(
                input_tokens=input_estimate,
                output_tokens=max(1, (len(reply.text) + 3) // 4),
                estimated=True,
            )
        self.budget.record_usage(
            state.run_id,
            stage_id,
            RoleId.DEVELOPMENT.value,
            "conversation",
            usage,
            reservation=reservation,
        )
        return reply

    def _reserve(self, state, stage_id: str, input_estimate: int) -> BudgetReservation:
        return self.budget.reserve(
            state.run_id,
            stage_id,
            RoleId.DEVELOPMENT.value,
            "conversation",
            input_estimate + self.response_reserve_tokens,
        )

    def _input_upper_bound(
        self, state, context: ContextBundle, message: IncomingMessage
    ) -> int:
        estimator = getattr(self.delegate, "prompt_token_upper_bound", None)
        if callable(estimator):
            tokens = int(estimator(state, context, message))
        else:
            tokens = conservative_prompt_tokens(
                f"{context.to_dict()}\n{message.text}"
            )
        if tokens < 1:
            raise ValueError("model prompt token upper bound must be positive")
        return tokens + self.budget.policy.provider_input_overhead_tokens

    def _invoke_delegate(
        self, state, context: ContextBundle, message: IncomingMessage, *, cancelled=None
    ) -> AgentReply:
        kwargs = {}
        if getattr(self.delegate, "supports_output_token_limit", False):
            kwargs["max_output_tokens"] = self.response_reserve_tokens
        if getattr(self.delegate, "supports_cancellation", False):
            kwargs["cancelled"] = cancelled
        return self.delegate.respond(state, context, message, **kwargs)

    def _require_budget(self, state, stage_id: str, input_estimate: int) -> None:
        decision = self.budget.can_spend(
            state.run_id,
            stage_id,
            input_estimate + self.response_reserve_tokens,
            category="conversation",
        )
        if not decision.allowed:
            raise BudgetExceeded(decision.reason)

    def _record_execution_failure_usage(
        self,
        state,
        stage_id: str,
        error: Exception,
        reservation: BudgetReservation,
    ) -> None:
        if isinstance(error, HermesExecutionError) and error.usage.total_tokens:
            self.budget.record_usage(
                state.run_id,
                stage_id,
                RoleId.DEVELOPMENT.value,
                "conversation",
                error.usage,
                reservation=reservation,
            )
        else:
            self.budget.release_reservation(reservation)


class GovernedTeamConversationBackend:
    """자유 대화의 역할별 예산, 사용량, 형식/기술 재시도를 강제한다."""

    def __init__(
        self,
        delegate: TeamConversationBackend,
        budget: BudgetManager,
        *,
        response_reserve_tokens: int = 1024,
    ):
        if response_reserve_tokens < 1:
            raise ValueError("response reserve must be positive")
        self.delegate = delegate
        self.budget = budget
        self.response_reserve_tokens = response_reserve_tokens
        self.redactor = SecretRedactor()
        self.supports_cancellation = True

    def preflight(
        self,
        state,
        context: ContextBundle,
        message: IncomingMessage,
        reply_count: int,
    ) -> None:
        if reply_count < 1:
            raise ValueError("reply_count must be positive")
        per_reply = self._input_upper_bound(
            state, context, message, RoleId.DEVELOPMENT, (), "", None, 1
        )
        projected = (per_reply + self.response_reserve_tokens) * reply_count
        decision = self.budget.can_spend(
            state.run_id,
            self._stage_id(message),
            projected,
            category="conversation",
        )
        if not decision.allowed:
            raise BudgetExceeded(decision.reason)

    def respond_as(
        self,
        state,
        context: ContextBundle,
        message: IncomingMessage,
        role_id: RoleId,
        *,
        caller_role: RoleId | None = None,
        call_purpose: str = "",
        turn_messages: tuple[dict, ...] = (),
        call_index: int = 1,
        cancelled=None,
    ) -> AgentReply:
        stage_id = self._stage_id(message)
        budget_category = (
            "repository_analysis"
            if call_purpose.startswith("repository_analysis_")
            else "conversation"
        )
        use_completion_reserve = call_purpose == "repository_analysis_synthesis"
        input_estimate = self._input_upper_bound(
            state,
            context,
            message,
            role_id,
            turn_messages,
            call_purpose,
            caller_role,
            call_index,
        )
        while True:
            reservation = self._reserve(
                state,
                stage_id,
                role_id,
                input_estimate,
                use_completion_reserve=use_completion_reserve,
                category=budget_category,
            )
            try:
                kwargs = {
                    "caller_role": caller_role,
                    "call_purpose": call_purpose,
                    "turn_messages": turn_messages,
                    "call_index": call_index,
                }
                if getattr(self.delegate, "supports_output_token_limit", False):
                    kwargs["max_output_tokens"] = self.response_reserve_tokens
                if getattr(self.delegate, "supports_cancellation", False):
                    kwargs["cancelled"] = cancelled
                reply = self.delegate.respond_as(
                    state,
                    context,
                    message,
                    role_id,
                    **kwargs,
                )
                break
            except HermesCancelled:
                self.budget.release_reservation(reservation)
                raise
            except InvalidAgentResponse as exc:
                self._record_failure_usage(
                    state, stage_id, role_id, exc, reservation, category=budget_category
                )
                if not self.budget.can_retry(state.run_id, stage_id, "invalid_response"):
                    raise
                self._require_budget(
                    state,
                    stage_id,
                    input_estimate,
                    use_completion_reserve=use_completion_reserve,
                    category=budget_category,
                )
                self.budget.record_retry(
                    state.run_id,
                    stage_id,
                    "invalid_response",
                    self.redactor.text(type(exc).__name__),
                )
                call_index += 1
            except GitRepositoryError:
                self.budget.release_reservation(reservation)
                raise
            except Exception as exc:
                self._record_failure_usage(
                    state, stage_id, role_id, exc, reservation, category=budget_category
                )
                if not self.budget.can_retry(state.run_id, stage_id, "technical_error"):
                    raise
                self._require_budget(
                    state,
                    stage_id,
                    input_estimate,
                    use_completion_reserve=use_completion_reserve,
                    category=budget_category,
                )
                self.budget.record_retry(
                    state.run_id,
                    stage_id,
                    "technical_error",
                    self.redactor.text(type(exc).__name__),
                )
                call_index += 1

        return self._record_success(
            state,
            stage_id,
            role_id,
            reply,
            input_estimate,
            reservation,
            category=budget_category,
        )

    def respond_batch(
        self,
        state,
        message: IncomingMessage,
        requests: tuple[TeamConversationRequest, ...],
        *,
        max_workers: int,
        max_model_calls: int,
        cancelled=None,
    ) -> TeamConversationBatchResult:
        """모델 실행만 병렬화하고 예산 원장은 호출 스레드에서 순서대로 기록한다."""
        if not requests:
            return TeamConversationBatchResult((), 0)
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        if max_model_calls < len(requests):
            raise ValueError("max_model_calls cannot be smaller than request count")
        stage_id = self._stage_id(message)
        outcomes: list[AgentReply | Exception | None] = [None] * len(requests)
        runnable: list[tuple[int, TeamConversationRequest, int, BudgetReservation]] = []
        for index, request in enumerate(requests):
            input_estimate = self._input_upper_bound(
                state,
                request.context,
                message,
                request.role_id,
                request.turn_messages,
                request.call_purpose,
                request.caller_role,
                request.call_index,
            )
            try:
                reservation = self._reserve(
                    state, stage_id, request.role_id, input_estimate
                )
            except BudgetExceeded as exc:
                outcomes[index] = exc
            else:
                runnable.append((index, request, input_estimate, reservation))

        if runnable:
            with ThreadPoolExecutor(
                max_workers=min(max_workers, len(runnable)),
                thread_name_prefix="ai-agents-governed-chat",
            ) as executor:
                futures = [
                    executor.submit(
                        self._invoke, state, message, request, cancelled=cancelled
                    )
                    for _index, request, _estimate, _reservation in runnable
                ]
                raw = [future.result() for future in futures]

            # 세 초기 호출이 이미 소비한 사용량을 모두 먼저 반영해야, 어느 역할의
            # 재시도도 아직 기록하지 않은 다른 역할의 사용량을 우회하지 않는다.
            for (index, request, input_estimate, reservation), outcome in zip(
                runnable, raw, strict=True
            ):
                outcomes[index] = self._record_initial_outcome(
                    state,
                    stage_id,
                    request,
                    input_estimate,
                    outcome,
                    reservation,
                )
            model_calls = len(runnable)
            next_call_index = max(request.call_index for request in requests) + 1
            for index, request, input_estimate, _reservation in runnable:
                outcome = outcomes[index]
                if not isinstance(outcome, Exception):
                    continue
                resolved, used_calls, next_call_index = self._resolve_batch_outcome(
                    state,
                    message,
                    stage_id,
                    request,
                    input_estimate,
                    outcome,
                    remaining_model_calls=max_model_calls - model_calls,
                    next_call_index=next_call_index,
                    cancelled=cancelled,
                )
                outcomes[index] = resolved
                model_calls += used_calls
        else:
            model_calls = 0
        return TeamConversationBatchResult(
            tuple(
                outcome
                if outcome is not None
                else RuntimeError("parallel conversation result is missing")
                for outcome in outcomes
            ),
            model_calls,
        )

    def _record_initial_outcome(
        self,
        state,
        stage_id: str,
        request: TeamConversationRequest,
        input_estimate: int,
        outcome: AgentReply | Exception,
        reservation: BudgetReservation,
    ) -> AgentReply | Exception:
        if isinstance(outcome, AgentReply):
            return self._record_success(
                state, stage_id, request.role_id, outcome, input_estimate, reservation
            )
        self._record_failure_usage(
            state, stage_id, request.role_id, outcome, reservation
        )
        return outcome

    def _resolve_batch_outcome(
        self,
        state,
        message: IncomingMessage,
        stage_id: str,
        request: TeamConversationRequest,
        input_estimate: int,
        outcome: AgentReply | Exception,
        *,
        remaining_model_calls: int,
        next_call_index: int,
        cancelled=None,
    ) -> tuple[AgentReply | Exception, int, int]:
        current = outcome
        used_calls = 0
        while isinstance(current, Exception):
            if isinstance(current, HermesCancelled):
                return current, used_calls, next_call_index
            if isinstance(current, InvalidAgentResponse):
                category = "invalid_response"
            elif isinstance(current, GitRepositoryError):
                return current, used_calls, next_call_index
            else:
                category = "technical_error"
            if not self.budget.can_retry(state.run_id, stage_id, category):
                return current, used_calls, next_call_index
            if used_calls >= remaining_model_calls:
                return current, used_calls, next_call_index
            try:
                reservation = self._reserve(
                    state, stage_id, request.role_id, input_estimate
                )
            except BudgetExceeded as exc:
                return exc, used_calls, next_call_index
            try:
                self.budget.record_retry(
                    state.run_id,
                    stage_id,
                    category,
                    self.redactor.text(type(current).__name__),
                )
            except RuntimeError:
                self.budget.release_reservation(reservation)
                return current, used_calls, next_call_index
            retry_request = replace(request, call_index=next_call_index)
            next_call_index += 1
            current = self._invoke(
                state, message, retry_request, cancelled=cancelled
            )
            used_calls += 1
            if isinstance(current, AgentReply):
                current = self._record_success(
                    state,
                    stage_id,
                    request.role_id,
                    current,
                    input_estimate,
                    reservation,
                )
            else:
                self._record_failure_usage(
                    state, stage_id, request.role_id, current, reservation
                )
        return current, used_calls, next_call_index

    def _record_failure_usage(
        self,
        state,
        stage_id: str,
        role_id: RoleId,
        error: Exception,
        reservation: BudgetReservation,
        *,
        category: str = "conversation",
    ) -> None:
        if isinstance(error, InvalidAgentResponse) and error.usage.total_tokens:
            self.budget.record_usage(
                state.run_id,
                stage_id,
                role_id.value,
                category,
                error.usage,
                reservation=reservation,
            )
        elif isinstance(error, HermesExecutionError) and error.usage.total_tokens:
            self.budget.record_usage(
                state.run_id,
                stage_id,
                role_id.value,
                category,
                error.usage,
                reservation=reservation,
            )
        else:
            self.budget.release_reservation(reservation)

    def _reserve(
        self,
        state,
        stage_id: str,
        role_id: RoleId,
        input_estimate: int,
        *,
        use_completion_reserve: bool = False,
        category: str = "conversation",
    ) -> BudgetReservation:
        return self.budget.reserve(
            state.run_id,
            stage_id,
            role_id.value,
            category,
            input_estimate + self.response_reserve_tokens,
            use_completion_reserve=use_completion_reserve,
        )

    def _require_budget(
        self,
        state,
        stage_id: str,
        input_estimate: int,
        *,
        use_completion_reserve: bool = False,
        category: str = "conversation",
    ) -> None:
        decision = self.budget.can_spend(
            state.run_id,
            stage_id,
            input_estimate + self.response_reserve_tokens,
            category=category,
            use_completion_reserve=use_completion_reserve,
        )
        if not decision.allowed:
            raise BudgetExceeded(decision.reason)

    def _invoke(
        self,
        state,
        message: IncomingMessage,
        request: TeamConversationRequest,
        *,
        cancelled=None,
    ) -> AgentReply | Exception:
        try:
            kwargs = {
                "caller_role": request.caller_role,
                "call_purpose": request.call_purpose,
                "turn_messages": request.turn_messages,
                "call_index": request.call_index,
            }
            if getattr(self.delegate, "supports_output_token_limit", False):
                kwargs["max_output_tokens"] = self.response_reserve_tokens
            if getattr(self.delegate, "supports_cancellation", False):
                kwargs["cancelled"] = cancelled
            return self.delegate.respond_as(
                state,
                request.context,
                message,
                request.role_id,
                **kwargs,
            )
        except Exception as exc:
            return exc

    def _record_success(
        self,
        state,
        stage_id: str,
        role_id: RoleId,
        reply: AgentReply,
        input_estimate: int,
        reservation: BudgetReservation,
        *,
        category: str = "conversation",
    ) -> AgentReply:
        usage = reply.usage
        if usage.total_tokens == 0:
            usage = TokenUsage(
                input_tokens=input_estimate,
                output_tokens=max(1, (len(reply.text) + 3) // 4),
                estimated=True,
            )
        self.budget.record_usage(
            state.run_id,
            stage_id,
            role_id.value,
            category,
            usage,
            reservation=reservation,
        )
        return replace(reply, usage=usage)

    def _input_upper_bound(
        self,
        state,
        context: ContextBundle,
        message: IncomingMessage,
        role_id: RoleId,
        turn_messages: tuple[dict, ...],
        call_purpose: str,
        caller_role: RoleId | None,
        call_index: int,
    ) -> int:
        estimator = getattr(self.delegate, "prompt_token_upper_bound", None)
        if callable(estimator):
            tokens = int(
                estimator(
                    state,
                    context,
                    message,
                    role_id,
                    caller_role=caller_role,
                    call_purpose=call_purpose,
                    turn_messages=turn_messages,
                    call_index=call_index,
                )
            )
        else:
            tokens = conservative_prompt_tokens(
                f"{context.to_dict()}\n{message.text}\n{call_purpose}\n{turn_messages}"
            )
        if tokens < 1:
            raise ValueError("model prompt token upper bound must be positive")
        return tokens + self.budget.policy.provider_input_overhead_tokens

    @staticmethod
    def _stage_id(message: IncomingMessage) -> str:
        digest = hashlib.sha256(
            message.external_message_id.encode("utf-8")
        ).hexdigest()[:12]
        return f"chat-{digest}"
