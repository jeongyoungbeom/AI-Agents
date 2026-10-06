from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import replace

from app.contracts.outcomes import RequestOutcome, RequestResult
from app.gateway.core.conversation import ConversationCancelled, DialogueRouter
from app.gateway.core.models import IncomingMessage, OutgoingMessage
from app.services.logging.redaction import SecretRedactor
from app.storage import StateStore


class ConversationQueue:
    """메신저 수신과 긴 대화 처리를 분리하는 SQLite 큐 어댑터."""

    def __init__(self, store: StateStore):
        self.store = store

    def enqueue(self, message: IncomingMessage) -> dict:
        return self.store.enqueue_conversation_job(
            message.channel,
            message.conversation_id,
            message.user_id,
            message.external_message_id,
            message.to_dict(),
        )

    def cancel(self, channel: str, conversation_id: str) -> str | None:
        return self.store.request_conversation_cancel(channel, conversation_id)

    def status(self, channel: str, conversation_id: str) -> str | None:
        return self.store.conversation_job_status(channel, conversation_id)

    def summary(self, channel: str, conversation_id: str) -> dict | None:
        return self.store.conversation_job_summary(channel, conversation_id)

    def resume(self, channel: str, conversation_id: str) -> dict | None:
        return self.store.requeue_conversation_job(channel, conversation_id)


class ConversationWorker:
    """대화 큐를 대화방별 순서대로 처리하는 백그라운드 워커."""

    def __init__(
        self,
        store: StateStore,
        router: DialogueRouter,
        *,
        message_preparer: Callable[
            [OutgoingMessage], tuple[OutgoingMessage, ...]
        ] | None = None,
        error_sink: Callable[[str], None] | None = None,
        on_outbound: Callable[[], None] | None = None,
        activity_notifier: Callable[[str, str], None] | None = None,
        worker_count: int = 1,
        poll_seconds: float = 0.2,
        lease_seconds: int = 120,
        max_job_attempts: int = 2,
    ):
        if (
            not 1 <= worker_count <= 8
            or poll_seconds <= 0
            or lease_seconds < 1
            or max_job_attempts < 1
        ):
            raise ValueError("conversation worker settings are invalid")
        self.store = store
        self.router = router
        self.message_preparer = message_preparer or (lambda message: (message,))
        self.error_sink = error_sink or (lambda _message: None)
        self.on_outbound = on_outbound or (lambda: None)
        self.activity_notifier = activity_notifier or (
            lambda _channel, _conversation_id: None
        )
        self.worker_count = worker_count
        self.poll_seconds = poll_seconds
        self.lease_seconds = lease_seconds
        self.max_job_attempts = max_job_attempts
        self.instance_id = uuid.uuid4().hex
        self.redactor = SecretRedactor()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def set_outbound_notifier(self, notifier: Callable[[], None]) -> None:
        self.on_outbound = notifier

    def start(self) -> None:
        if any(thread.is_alive() for thread in self._threads):
            return
        self._stop.clear()
        self._recover_stale()
        self._threads = [
            threading.Thread(
                target=self.run_forever,
                name=f"ai-agents-conversation-{index + 1}",
                daemon=True,
            )
            for index in range(self.worker_count)
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        for thread in self._threads:
            thread.join(timeout)

    def is_alive(self) -> bool:
        return bool(self._threads) and all(thread.is_alive() for thread in self._threads)

    def run_once(self) -> bool:
        job = self.store.claim_next_conversation_job(
            self.instance_id,
            lease_seconds=self.lease_seconds,
            max_attempts=self.max_job_attempts,
        )
        if job is None:
            return False
        job_id = int(job["job_id"])
        message = IncomingMessage.from_dict(dict(job["message"]))
        self.error_sink(
            f"대화 작업 시작 job={job_id} channel={message.channel} "
            f"conversation={message.conversation_id} attempt={job['attempts']}"
        )
        heartbeat_stop = threading.Event()
        lease_lost = threading.Event()
        self._notify_activity(message.channel, message.conversation_id)
        heartbeat_thread = threading.Thread(
            target=self._heartbeat,
            args=(
                job_id,
                message.channel,
                message.conversation_id,
                heartbeat_stop,
                lease_lost,
            ),
            name=f"conversation-heartbeat-{job_id}",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            def cancelled() -> bool:
                return (
                    self._stop.is_set()
                    or lease_lost.is_set()
                    or self.store.conversation_job_cancel_requested(job_id)
                )

            if self.store.conversation_job_cancel_requested(job_id):
                self.router.cancel_conversation_progress(message)
                self.store.finish_conversation_job(
                    job_id, self.instance_id, "CANCELLED"
                )
                return True
            if bool(job.get("replay_cached")):
                cached = self.store.conversation_responses(
                    message.channel,
                    message.conversation_id,
                    message.external_message_id,
                )
                if cached:
                    outgoing = tuple(
                        OutgoingMessage(
                            message.channel,
                            message.conversation_id,
                            text,
                            reply_to=message.external_message_id,
                        )
                        for text in cached
                    )
                    result = self.store.conversation_result(
                        message.channel, message.conversation_id, message.external_message_id,
                    ) or RequestResult(RequestOutcome.FAILED, "OUTCOME_UNKNOWN", None, None)
                    if result.reason == "RUNNING":
                        result = replace(result, outcome=RequestOutcome.PARTIAL, reason="INTERRUPTED")
                else:
                    message = replace(message, metadata={**message.metadata, "model_cache_replay": True})
                    routed = self.router.route_result(message, cancelled=cancelled)
                    outgoing, result = routed.messages, routed.result
            else:
                route_result = getattr(self.router, "route_result", None)
                if callable(route_result):
                    routed = route_result(message, cancelled=cancelled)
                    outgoing, result = routed.messages, routed.result
                else:
                    outgoing = self.router.route(message, cancelled=cancelled)
                    result = RequestResult(RequestOutcome.SUCCESS)
            if lease_lost.is_set():
                raise RuntimeError("conversation job lease was lost")
            prepared = tuple(
                part
                for item in outgoing
                for part in self.message_preparer(item)
            )
            target_run = message.metadata.get('gateway_run_id')
            binding = self.store.load_conversation(message.channel, message.conversation_id)
            if target_run and (not binding or binding['run_id'] != target_run):
                prepared = ()
                result = RequestResult(RequestOutcome.CANCELLED, 'SUPERSEDED_TASK')
            if self.store.conversation_job_cancel_requested(job_id):
                self.router.cancel_conversation_progress(message)
                prepared = ()
                status = "CANCELLED"
            else:
                status = "COMPLETED"
            with self.store.transaction():
                self.store.finish_conversation_job(
                    job_id, self.instance_id, status,
                    [self._outbound_dict(item) for item in prepared], result=result,
                )
                status = self.store.conversation_job(job_id)["status"]
                result = self.store.conversation_result(
                    message.channel, message.conversation_id, message.external_message_id,
                )
            if status == "CANCELLED":
                self.router.cancel_conversation_progress(message)
            self.router.complete_conversation_progress(message)
            self.error_sink(
                f"대화 작업 종료 job={job_id} status={status} "
                f"attempt={job['attempts']} outcome={result.outcome.value} responses={len(prepared)}"
            )
        except ConversationCancelled as exc:
            self.router.cancel_conversation_progress(message)
            self.store.finish_conversation_job(
                job_id, self.instance_id, "CANCELLED"
            )
            self.router.complete_conversation_progress(message)
            self.error_sink(
                f"대화 작업 취소 job={job_id} channel={message.channel} "
                f"conversation={message.conversation_id} attempt={job['attempts']} reason="
                f"{self.redactor.text(str(exc))[:300]}"
            )
        except Exception as exc:
            safe_error = self.redactor.text(f"{type(exc).__name__}: {exc}")[:1000]
            self.router.fail_conversation_progress(
                message,
                "대화 처리 중 문제가 발생해 작업을 중단했습니다. 로그를 확인해 주세요.",
            )
            notice = OutgoingMessage(
                message.channel,
                message.conversation_id,
                "대화 처리 중 문제가 발생해 멈췄습니다. 로그를 확인해 주세요.",
                reply_to=message.external_message_id,
            )
            try:
                previous = self.store.conversation_result(
                    message.channel, message.conversation_id, message.external_message_id,
                ) or RequestResult(RequestOutcome.FAILED)
                self.store.finish_conversation_job(
                    job_id,
                    self.instance_id,
                    "NEEDS_ATTENTION",
                    [self._outbound_dict(notice)],
                    safe_error,
                    result=replace(previous, outcome=RequestOutcome.FAILED, reason="PROCESSING_FAILED"),
                )
            except Exception as finish_exc:
                self.error_sink(
                    "대화 작업 실패 상태 저장도 실패했습니다. "
                    f"job={job_id} error={type(finish_exc).__name__}:"
                    f"{self.redactor.text(str(finish_exc))[:300]}"
                )
            finally:
                self.error_sink(
                    f"대화 작업 실패 job={job_id} channel={message.channel} "
                    f"conversation={message.conversation_id} attempt={job['attempts']} "
                    f"error={safe_error}"
                )
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=2)
        self._notify_outbound()
        return True

    def run_forever(self) -> None:
        while not self._stop.is_set():
            if not self.run_once():
                self._recover_stale()
                self._stop.wait(self.poll_seconds)

    def _recover_stale(self) -> None:
        recovered = self.store.recover_stale_conversation_jobs()
        for job in recovered:
            channel = str(job["channel"])
            conversation_id = str(job["conversation_id"])
            external_message_id = str(job["external_message_id"])
            input_id = job['message'].get('metadata', {}).get('gateway_execution_input')
            entry = self.store.execution_input(input_id) if input_id else None
            if entry and entry['status'] != 'RECEIVED' and self.store.execution_input_is_current(entry):
                self.store.requeue_conversation_job(channel, conversation_id, job_id=int(job['job_id']))
            if getattr(self.router, "budget", None) is not None:
                from app.gateway.core.governed_backend import GovernedTeamConversationBackend
                message = IncomingMessage.from_dict(job["message"])
                request_key = GovernedTeamConversationBackend.request_key(message)
                for run_id, stage_id in self.store.interrupted_call_scopes(
                    request_key
                ):
                    self.router.budget.recover_interrupted_calls(run_id, stage_id, request_key)
            cached = self.store.conversation_responses(
                channel, conversation_id, external_message_id
            )
            with self.store.transaction():
                progress_outbound_id = self.store.progress_outbound_for_reply(
                    channel, conversation_id, external_message_id
                )
                if progress_outbound_id is not None:
                    self.store.queue_outbound(
                        channel,
                        conversation_id,
                        "[이전 실행 · 중단됨]\n\n"
                        "⚠️ 대화 처리 프로세스가 중간에 종료됐습니다.\n"
                        + (
                            "저장된 응답은 '재개'로 같은 작업에서 전송할 수 있습니다."
                            if cached
                            else "모델 호출 결과를 확인할 수 없습니다. '상태'로 사유를 확인해 주세요."
                        ),
                        reply_to=external_message_id,
                        delivery_mode="edit",
                        target_outbound_id=progress_outbound_id,
                        coalesce_key=f"progress:{progress_outbound_id}",
                    )
        if recovered:
            self.error_sink(f"중단된 대화 작업 {len(recovered)}개를 확인 필요 상태로 복구했습니다.")
            self._notify_outbound()

    def _notify_outbound(self) -> None:
        try:
            self.on_outbound()
        except Exception as exc:
            self.error_sink(
                "대화 응답 즉시 전송 실패; 발신함에서 재시도합니다. "
                f"error={type(exc).__name__}:{self.redactor.text(str(exc))[:300]}"
            )

    def _heartbeat(
        self,
        job_id: int,
        channel: str,
        conversation_id: str,
        stopped: threading.Event,
        lease_lost: threading.Event,
    ) -> None:
        lease_interval = max(1.0, self.lease_seconds / 3)
        activity_interval = min(4.0, lease_interval)
        next_lease = time.monotonic() + lease_interval
        while not stopped.wait(activity_interval):
            self._notify_activity(channel, conversation_id)
            self.router.refresh_conversation_progress(channel, conversation_id)
            if time.monotonic() < next_lease:
                continue
            try:
                self.store.heartbeat_conversation_job(
                    job_id,
                    self.instance_id,
                    lease_seconds=self.lease_seconds,
                )
            except Exception:
                lease_lost.set()
                return
            next_lease = time.monotonic() + lease_interval

    def _notify_activity(self, channel: str, conversation_id: str) -> None:
        try:
            self.activity_notifier(channel, conversation_id)
        except Exception:
            # 입력 상태는 편의 기능이다. 실패해도 실제 답변과 큐 처리는 계속한다.
            return

    @staticmethod
    def _outbound_dict(message: OutgoingMessage) -> dict[str, str]:
        return {
            "channel": message.channel,
            "conversation_id": message.conversation_id,
            "text": message.text,
            "reply_to": message.reply_to,
        }
