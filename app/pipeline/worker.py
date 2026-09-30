from __future__ import annotations

import threading
import uuid
from collections.abc import Callable

from app.contracts import RunPhase
from app.orchestrator import RunStateMachine
from app.services.budget import BudgetExceeded
from app.services.git import GitRepositoryError
from app.services.logging.audit import AuditLogger
from app.services.logging.redaction import SecretRedactor
from app.storage import StateStore
from app.services.verification import UnsafeVerificationCommand

from .coordinator import (
    PipelineCancelled,
    PipelineCoordinator,
    PipelineNeedsAttention,
    PipelineUserInputRequired,
)


TERMINAL_PHASES = {RunPhase.COMPLETED, RunPhase.FAILED, RunPhase.CANCELLED}


class PipelineWorker:
    """승인된 작업을 SQLite 큐에서 하나씩 실행하는 백그라운드 워커."""

    def __init__(
        self,
        store: StateStore,
        machine: RunStateMachine,
        coordinator: PipelineCoordinator,
        logger: AuditLogger,
        *,
        poll_seconds: float = 0.5,
        lease_seconds: int = 120,
        on_outbound: Callable[[], None] | None = None,
        activity_notifier: Callable[[str, str], None] | None = None,
        activity_interval_seconds: float = 4.0,
    ):
        if poll_seconds <= 0 or lease_seconds < 1 or activity_interval_seconds <= 0:
            raise ValueError("pipeline worker timing must be positive")
        self.store = store
        self.machine = machine
        self.coordinator = coordinator
        self.logger = logger
        self.poll_seconds = poll_seconds
        self.lease_seconds = lease_seconds
        self.on_outbound = on_outbound or (lambda: None)
        self.activity_notifier = activity_notifier or (
            lambda _channel, _conversation_id: None
        )
        self.activity_interval_seconds = activity_interval_seconds
        self.instance_id = uuid.uuid4().hex
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.redactor = SecretRedactor()

    def enqueue(self, run_id: str, channel: str, conversation_id: str) -> str:
        return str(
            self.store.enqueue_pipeline_job(run_id, channel, conversation_id)["status"]
        )

    def cancel(self, run_id: str) -> str | None:
        return self.store.request_pipeline_cancel(run_id)

    def pause(self, run_id: str) -> str | None:
        return self.store.request_pipeline_pause(run_id)

    def resume(self, run_id: str) -> str | None:
        return self.store.requeue_pipeline_job(run_id)

    def status(self, run_id: str) -> str | None:
        job = self.store.pipeline_job(run_id)
        return str(job["status"]) if job else None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._recover_stale()
        self._thread = threading.Thread(
            target=self.run_forever,
            name="ai-agents-pipeline",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_outbound_notifier(self, notifier: Callable[[], None]) -> None:
        self.on_outbound = notifier

    def run_once(self) -> bool:
        job = self.store.claim_next_pipeline_job(
            self.instance_id, lease_seconds=self.lease_seconds
        )
        if job is None:
            return False
        run_id = str(job["run_id"])
        channel = str(job["channel"])
        conversation_id = str(job["conversation_id"])

        def cancelled() -> bool:
            return (
                self._stop.is_set()
                or self.store.pipeline_cancel_requested(run_id)
                or self.store.pipeline_pause_requested(run_id)
            )

        def heartbeat() -> None:
            self.store.heartbeat_pipeline_job(
                run_id, self.instance_id, lease_seconds=self.lease_seconds
            )

        activity_stop = threading.Event()
        activity_thread = threading.Thread(
            target=self._activity_loop,
            args=(channel, conversation_id, activity_stop),
            name=f"pipeline-activity-{run_id}",
            daemon=True,
        )
        activity_thread.start()

        try:
            self.coordinator.execute(
                run_id,
                channel,
                conversation_id,
                self.instance_id,
                cancelled=cancelled,
                heartbeat=heartbeat,
            )
        except PipelineCancelled as exc:
            if self.store.pipeline_pause_requested(run_id):
                message = "사용자가 중지를 요청했습니다."
                self._pause_state(run_id, message)
                self.store.finish_pipeline_job(
                    run_id, self.instance_id, "NEEDS_ATTENTION", message
                )
                self._queue_outbound(
                    channel,
                    conversation_id,
                    "작업을 안전한 지점에서 일시 중지했습니다. 계속하려면 '재개'라고 말해 주세요.",
                )
                return True
            self._cancel_state(run_id, str(exc))
            self.store.finish_pipeline_job(
                run_id, self.instance_id, "CANCELLED", self.redactor.text(str(exc))
            )
            self._queue_outbound(channel, conversation_id, "작업을 안전하게 중지했습니다.")
            self.store.set_conversation_mode(channel, conversation_id, "free_chat")
        except PipelineUserInputRequired as exc:
            questions = tuple(self.redactor.text(item) for item in exc.questions)
            self.store.open_execution_question(
                run_id, exc.stage_id, exc.role_id.value, questions
            )
            message = self.redactor.text(str(exc))
            self._pause_state(run_id, message)
            self.store.finish_pipeline_job(
                run_id, self.instance_id, "NEEDS_ATTENTION", message
            )
            self._queue_outbound(
                channel,
                conversation_id,
                "작업을 멈추고 확인을 기다립니다.\n"
                f"[{exc.role_id.value}] "
                + "\n".join(f"- {question}" for question in questions)
                + "\n\n답변을 보내면 안전한 단계 시작 지점부터 다시 진행합니다.",
            )
        except (
            PipelineNeedsAttention,
            GitRepositoryError,
            BudgetExceeded,
            UnsafeVerificationCommand,
        ) as exc:
            message = self.redactor.text(str(exc))
            self._pause_state(run_id, message)
            self.store.finish_pipeline_job(
                run_id, self.instance_id, "NEEDS_ATTENTION", message
            )
            self._queue_outbound(
                channel, conversation_id, f"작업을 멈췄습니다. 확인이 필요합니다: {message}"
            )
        except Exception as exc:
            message = self.redactor.text(f"{type(exc).__name__}: {exc}")
            self._fail_state(run_id, message)
            self.store.finish_pipeline_job(run_id, self.instance_id, "FAILED", message)
            self._queue_outbound(
                channel, conversation_id, f"작업 중 오류가 발생해 중단했습니다: {message[:1000]}"
            )
            self.store.set_conversation_mode(channel, conversation_id, "free_chat")
        else:
            self.store.finish_pipeline_job(run_id, self.instance_id, "COMPLETED")
        finally:
            activity_stop.set()
            activity_thread.join(timeout=2)
        return True

    def run_forever(self) -> None:
        while not self._stop.is_set():
            if not self.run_once():
                self._recover_stale()
                self._stop.wait(self.poll_seconds)

    def _recover_stale(self) -> None:
        for job in self.store.recover_stale_pipeline_jobs():
            run_id = str(job["run_id"])
            message = "이전 실행이 저장소 변경 도중 종료되어 자동 재실행하지 않습니다."
            self._pause_state(run_id, message)
            self._queue_outbound(
                str(job["channel"]), str(job["conversation_id"]), message
            )

    def _queue_outbound(self, channel: str, conversation_id: str, text: str) -> None:
        self.store.queue_outbound(channel, conversation_id, text)
        self._notify_outbound()

    def _notify_outbound(self) -> None:
        try:
            self.on_outbound()
        except Exception:
            # 진행 알림은 발신함에 남아 있으므로 다음 폴링에서 다시 전송된다.
            return

    def _notify_activity(self, channel: str, conversation_id: str) -> None:
        try:
            self.activity_notifier(channel, conversation_id)
        except Exception:
            # 입력 상태는 편의 기능이며 작업 성공 여부에 영향을 주지 않는다.
            return

    def _activity_loop(
        self,
        channel: str,
        conversation_id: str,
        stopped: threading.Event,
    ) -> None:
        self._notify_activity(channel, conversation_id)
        while not stopped.wait(self.activity_interval_seconds):
            self._notify_activity(channel, conversation_id)

    def _cancel_state(self, run_id: str, reason: str) -> None:
        state = self.store.load_run(run_id)
        if state.phase not in TERMINAL_PHASES:
            self.machine.transition(state, RunPhase.CANCELLED, message=reason)
        self.logger.write_summary(self.store.load_run(run_id))

    def _pause_state(self, run_id: str, reason: str) -> None:
        state = self.store.load_run(run_id)
        if state.phase in {
            RunPhase.DEVELOPING,
            RunPhase.REVIEWING,
            RunPhase.IMPROVING,
            RunPhase.VERIFYING,
            RunPhase.STAGE_COMPLETED,
        }:
            self.machine.pause(state, reason)
        elif state.phase not in TERMINAL_PHASES and state.phase != RunPhase.PAUSED:
            self.logger.emit(
                run_id,
                "RUN_NEEDS_ATTENTION",
                reason,
                stage_id=f"stage-{state.stage_index + 1:03d}",
                status="NEEDS_ATTENTION",
            )
        self.logger.write_summary(self.store.load_run(run_id))

    def _fail_state(self, run_id: str, reason: str) -> None:
        state = self.store.load_run(run_id)
        if state.phase not in TERMINAL_PHASES:
            self.machine.fail(state, reason)
        self.logger.write_summary(self.store.load_run(run_id))
