from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from app.contracts import RoleId, TokenUsage
from app.contracts.models import utc_now
from app.services.budget import BudgetExceeded, BudgetManager, conservative_prompt_tokens
from app.services.context import ContextBundle
from app.services.git import GitRepositoryError
from app.services.hermes import HermesCancelled, HermesExecutionError
from app.services.logging.redaction import SecretRedactor
from app.storage import StateStore

from .errors import InvalidAgentResponse
from .models import AgentReply, TeamConversationBatchResult, TeamConversationRequest


class _GovernedCalls:
    """실행 시도·사용량·결과를 함께 확정하는 대화 호출 경계."""

    supports_cancellation = True
    reserves_before_execution = True
    planning = False

    def __init__(self, delegate, budget: BudgetManager, *, response_reserve_tokens: int = 1024):
        if response_reserve_tokens < 1:
            raise ValueError("response reserve must be positive")
        self.delegate = delegate
        self.budget = budget
        self.response_reserve_tokens = response_reserve_tokens
        self.redactor = SecretRedactor()

    @staticmethod
    def _stage_id(message) -> str:
        digest = hashlib.sha256(message.external_message_id.encode("utf-8")).hexdigest()[:12]
        return f"chat-{digest}"

    @staticmethod
    def request_key(message):
        return StateStore.model_request_key(message.channel, message.conversation_id, message.external_message_id)

    def _stage(self, state, message):
        return f"stage-{state.stage_index + 1:03d}" if self.planning else self._stage_id(message)

    @staticmethod
    def _category(request):
        return "repository_analysis" if request.call_purpose.startswith("repository_analysis_") else "conversation"

    @staticmethod
    def _completion(request):
        return request.call_purpose in {"repository_analysis_synthesis", "agent_loop_final",
                                        "consultation_return", "consultation_final"}

    def _input_upper_bound(self, state, context, message, role_id, turn_messages=(),
                           call_purpose="", caller_role=None, call_index=1):
        estimator = getattr(self.delegate, "prompt_token_upper_bound", None)
        if callable(estimator):
            args = (state, context, message) if self.planning else (state, context, message, role_id)
            kwargs = {} if self.planning else dict(caller_role=caller_role, call_purpose=call_purpose,
                                                   turn_messages=turn_messages, call_index=call_index)
            tokens = int(estimator(*args, **kwargs))
        else:
            tokens = conservative_prompt_tokens(f"{context.to_dict()}\n{message.text}\n{call_purpose}\n{turn_messages}")
        if tokens < 1:
            raise ValueError("model prompt token upper bound must be positive")
        return tokens + self.budget.policy.provider_input_overhead_tokens

    def _estimate(self, state, message, request):
        tokens = self._input_upper_bound(state, request.context, message, request.role_id,
                                        request.turn_messages, request.call_purpose,
                                        request.caller_role, request.call_index)
        estimator = getattr(self.delegate, "invocation_token_estimate", None)
        total = int(estimator(tokens, self.response_reserve_tokens)) if callable(estimator) else tokens + self.response_reserve_tokens
        if total < tokens:
            raise ValueError("invocation estimate cannot be smaller than its input")
        return tokens, total

    def _logical_id(self, state, message, request):
        # worker의 재전송 제약은 원래 모델 입력을 바꾸지 않는다.
        message = replace(message, metadata={key: value for key, value in message.metadata.items()
                                            if key != "model_cache_replay"})
        fingerprint = getattr(self.delegate, "call_input_fingerprint", None)
        kwargs = dict(caller_role=request.caller_role, call_purpose=request.call_purpose,
                      turn_messages=request.turn_messages, call_index=request.call_index)
        actual_prompt = (fingerprint(state, request.context, message) if self.planning
                         else fingerprint(state, request.context, message, request.role_id, **kwargs)) if callable(fingerprint) else ""
        payload = [state.run_id, message.to_dict(), request.role_id.value,
                   request.context.to_dict(), request.caller_role, request.call_purpose,
                   request.turn_messages, request.call_index, self.response_reserve_tokens, actual_prompt]
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def _prepare(self, state, message, request, logical_id, attempt):
        store = self.budget.store
        call_id = f"{logical_id}:{attempt}"
        with self.budget.transaction():
            completed = store.completed_model_call(logical_id)
            if completed:
                result = AgentReply.from_dict(json.loads(completed["result_json"]))
                return replace(result, metadata={**result.metadata, "cached_call": True})
            if message.metadata.get("model_cache_replay"):
                return HermesExecutionError("현재 입력과 일치하는 완료 결과가 없어 재전송을 중단했습니다.",
                                            category="call_state", retryable=False)
            if store.model_call(call_id):
                return HermesExecutionError("같은 호출이 실행 중이거나 이미 중단됐습니다. 저장된 상태를 확인해야 합니다.",
                                            category="call_state", retryable=False)
            input_tokens, estimated_total = self._estimate(state, message, request)
            try:
                reservation = self.budget.reserve(state.run_id, self._stage(state, message),
                    request.role_id.value, self._category(request), estimated_total,
                    use_completion_reserve=self._completion(request),
                    completion_allowance_tokens=estimated_total if request.reserve_final_answer else 0)
            except BudgetExceeded as exc:
                return exc
            store.create_model_call(call_id, logical_id, state.run_id, self._stage(state, message),
                                    request.role_id.value, self._category(request), reservation.reservation_id,
                                    self.request_key(message))
            if not self.planning:
                store.append_event(state.run_id, utc_now(), "CHAT_MODEL_INPUT_SAVED",
                    "완료 결과 복구용 원래 호출 입력을 보존했습니다.", role_id=request.role_id.value,
                    data={"logical_id": logical_id, "request_key": self.request_key(message),
                          "context": request.context.to_dict(), "caller_role": request.caller_role,
                          "call_purpose": request.call_purpose, "turn_messages": request.turn_messages,
                          "call_index": request.call_index})
        return call_id, reservation, input_tokens

    def _invoke(self, state, message, request, *, cancelled=None):
        try:
            if cancelled is not None and cancelled():
                raise HermesCancelled("모델 시작 전에 취소했습니다.", category="startup")
            kwargs = {} if self.planning else dict(caller_role=request.caller_role,
                call_purpose=request.call_purpose, turn_messages=request.turn_messages, call_index=request.call_index)
            if getattr(self.delegate, "supports_output_token_limit", False):
                kwargs["max_output_tokens"] = self.response_reserve_tokens
            if getattr(self.delegate, "supports_cancellation", False):
                kwargs["cancelled"] = cancelled
            if self.planning:
                return self.delegate.respond(state, request.context, message, **kwargs)
            return self.delegate.respond_as(state, request.context, message, request.role_id, **kwargs)
        except Exception as exc:
            return exc

    def _finish(self, state, message, request, prepared, outcome):
        call_id, reservation, input_tokens = prepared
        stage_id = self._stage(state, message)
        usage = getattr(outcome, "usage", TokenUsage())
        unknown = False
        if isinstance(outcome, AgentReply):
            if not usage.total_tokens:
                usage = TokenUsage(input_tokens=input_tokens,
                    output_tokens=max(1, (len(outcome.text) + 3) // 4), estimated=True)
        elif getattr(outcome, "category", "") == "startup" or (
            isinstance(outcome, GitRepositoryError) and not getattr(outcome, "invocation_started", False)
        ):
            usage = TokenUsage()
        elif usage.estimated or not usage.total_tokens:
            unknown = True
            usage = TokenUsage(total_tokens=reservation.reserved_tokens, estimated=True)
        overage = usage.total_tokens > reservation.reserved_tokens
        if isinstance(outcome, AgentReply):
            outcome = replace(outcome, usage=usage, metadata={**outcome.metadata,
                "call_id": call_id, "usage_unit": "hermes_invocation", "budget_stop": overage})
        with self.budget.transaction():
            if usage.total_tokens:
                self.budget.record_usage(state.run_id, stage_id, request.role_id.value,
                                         self._category(request), usage, reservation=reservation)
            else:
                self.budget.release_reservation(reservation)
            if unknown:
                self.budget.store.append_event(state.run_id, utc_now(), "MODEL_USAGE_UNKNOWN",
                    "호출 실패·취소의 실제 사용량을 확인하지 못해 예약 추정량을 기록했습니다.",
                    stage_id=stage_id, role_id=request.role_id.value, data={"call_id": call_id})
            self.budget.store.finish_model_call(call_id,
                "COMPLETED" if isinstance(outcome, AgentReply) else "FAILED",
                outcome.to_dict() if isinstance(outcome, AgentReply) else
                {"error_type": type(outcome).__name__, "usage": usage.to_dict(),
                 "error": self.redactor.text(str(outcome))[:500]})
        return outcome

    def _retry_category(self, outcome):
        if isinstance(outcome, (HermesCancelled, GitRepositoryError, BudgetExceeded)):
            return None
        if isinstance(outcome, HermesExecutionError) and not outcome.retryable:
            return None
        return "invalid_response" if isinstance(outcome, InvalidAgentResponse) else "technical_error"

    def _resolve(self, state, message, request, logical_id, outcome, remaining, next_index, *, cancelled=None):
        calls = 0
        while isinstance(outcome, Exception) and calls < remaining:
            category = self._retry_category(outcome)
            stage_id = self._stage(state, message)
            if category is None or not self.budget.can_retry(state.run_id, stage_id, category):
                break
            retry = replace(request, call_index=next_index)
            prepared = self._prepare(state, message, retry, logical_id, next_index)
            if not isinstance(prepared, tuple):
                outcome = prepared
                break
            try:
                self.budget.record_retry(state.run_id, stage_id, category, type(outcome).__name__)
            except RuntimeError:
                outcome = self._finish(state, message, retry, prepared,
                    HermesCancelled("다른 호출이 재시도 여유를 사용했습니다.", category="startup"))
                break
            next_index += 1
            calls += 1
            outcome = self._finish(state, message, retry, prepared,
                                   self._invoke(state, message, retry, cancelled=cancelled))
        return outcome, calls, next_index


class GovernedAgentBackend(_GovernedCalls):
    """계획 실행도 시작 프롬프트가 아닌 Hermes 전체 invocation으로 정산한다."""
    planning = True

    def respond(self, state, context, message, *, cancelled=None):
        request = TeamConversationRequest(RoleId.DEVELOPMENT, context)
        logical_id = self._logical_id(state, message, request)
        prepared = self._prepare(state, message, request, logical_id, 1)
        outcome = (self._finish(state, message, request, prepared,
                    self._invoke(state, message, request, cancelled=cancelled))
                   if isinstance(prepared, tuple) else prepared)
        outcome, _, _ = self._resolve(state, message, request, logical_id, outcome,
            sum(self.budget.policy.retries.values()), 2, cancelled=cancelled)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class GovernedTeamConversationBackend(_GovernedCalls):
    """조사·협업·종합을 같은 결과/비용 확정 경계로 실행한다."""

    def replay_saved_requests(self, state, message):
        """새 문맥이나 모델 호출 없이 같은 요청의 원래 입력과 완료 결과를 반환한다."""
        store = self.budget.store
        calls = store.model_calls_for_request(self.request_key(message))
        inputs = {}
        for event in store.list_events(state.run_id):
            if (event["event_type"] == "CHAT_MODEL_INPUT_SAVED"
                    and event["data"]["request_key"] == self.request_key(message)):
                inputs.setdefault(event["data"]["logical_id"], event["data"])
        results = []
        for call in calls:
            if call["status"] != "COMPLETED":
                continue
            saved = inputs.get(call["logical_id"])
            if call["run_id"] != state.run_id or saved is None:
                raise RuntimeError("같은 작업의 원래 호출 입력을 확인할 수 없습니다.")
            context = dict(saved["context"])
            for field in ("decisions", "memories", "recent_messages", "evidence"):
                context[field] = tuple(context[field])
            request = TeamConversationRequest(RoleId(call["role_id"]), ContextBundle(**context),
                RoleId(saved["caller_role"]) if saved["caller_role"] else None,
                saved["call_purpose"], tuple(saved["turn_messages"]), saved["call_index"])
            if self._logical_id(state, message, request) != call["logical_id"]:
                raise RuntimeError("원래 입력과 일치하지 않아 완료 결과 재전송을 중단했습니다.")
            results.append((request, AgentReply.from_dict(json.loads(call["result_json"]))))
        return tuple(results)

    def preflight(self, state, context, message, reply_count):
        if reply_count < 1:
            raise ValueError("reply_count must be positive")
        request = TeamConversationRequest(RoleId.DEVELOPMENT, context)
        _, estimated = self._estimate(state, message, request)
        decision = self.budget.can_spend(state.run_id, self._stage_id(message),
                                        estimated * reply_count, category="conversation")
        if not decision.allowed:
            raise BudgetExceeded(decision.reason)

    def respond_as(self, state, context, message, role_id, *, caller_role=None,
                   call_purpose="", turn_messages=(), call_index=1, cancelled=None):
        result = self.respond_batch(state, message,
            (TeamConversationRequest(role_id, context, caller_role, call_purpose, turn_messages, call_index),),
            max_workers=1, max_model_calls=1 + sum(self.budget.policy.retries.values()), cancelled=cancelled)
        outcome = result.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def respond_batch(self, state, message, requests, *, max_workers, max_model_calls, cancelled=None):
        if not requests:
            return TeamConversationBatchResult((), 0)
        if max_workers < 1 or max_model_calls < len(requests):
            raise ValueError("invalid worker or model call limit")
        outcomes = [None] * len(requests)
        runnable = []
        for index, request in enumerate(requests):
            logical_id = self._logical_id(state, message, request)
            try:
                prepared = self._prepare(state, message, request, logical_id, request.call_index)
            except Exception as exc:
                prepared = exc
            if isinstance(prepared, tuple):
                runnable.append((index, request, logical_id, prepared))
            else:
                outcomes[index] = prepared
        if runnable:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(runnable)),
                                    thread_name_prefix="ai-agents-governed-chat") as executor:
                futures = [executor.submit(self._invoke, state, message, item[1], cancelled=cancelled)
                           for item in runnable]
                raw = [future.result() for future in futures]
            # 처음 예약한 병렬 호출의 비용을 모두 확정한 뒤 재시도를 판단한다.
            for (index, request, _, prepared), outcome in zip(runnable, raw, strict=True):
                outcomes[index] = self._finish(state, message, request, prepared, outcome)
        model_calls = len(runnable)
        next_index = max(request.call_index for request in requests) + 1
        for index, request, logical_id, _ in runnable:
            outcome, calls, next_index = self._resolve(state, message, request, logical_id,
                outcomes[index], max_model_calls - model_calls, next_index, cancelled=cancelled)
            outcomes[index] = outcome
            model_calls += calls
        return TeamConversationBatchResult(tuple(outcomes), model_calls)
