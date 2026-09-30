from __future__ import annotations

import threading
from collections.abc import Callable

from app.services.logging.redaction import SecretRedactor


class OutboundWorker:
    """작업 스레드와 Telegram 네트워크 전송을 분리하는 발신 전용 워커."""

    def __init__(
        self,
        sender: Callable[[], int],
        *,
        error_sink: Callable[[str], None] | None = None,
        poll_seconds: float = 0.5,
    ):
        if poll_seconds <= 0:
            raise ValueError("outbound worker polling must be positive")
        self.sender = sender
        self.error_sink = error_sink or (lambda _message: None)
        self.poll_seconds = poll_seconds
        self.redactor = SecretRedactor()
        self._stop = threading.Event()
        self._wakeup = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.is_alive():
            return
        self._stop.clear()
        self._wakeup.set()
        self._thread = threading.Thread(
            target=self.run_forever,
            name="ai-agents-outbound",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wakeup.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def notify(self) -> None:
        """네트워크 I/O 없이 새 발신 항목이 있음을 알린다."""
        self._wakeup.set()

    def run_forever(self) -> None:
        while not self._stop.is_set():
            self._wakeup.wait(self.poll_seconds)
            self._wakeup.clear()
            if self._stop.is_set():
                break
            try:
                self.sender()
            except Exception as exc:
                self.error_sink(
                    "발신 워커 오류 "
                    f"error={type(exc).__name__}:{self.redactor.text(str(exc))[:300]}"
                )


class ActivityWorker:
    """입력 상태 네트워크 호출을 작업·임대 스레드 밖에서 처리한다."""

    def __init__(
        self,
        sender: Callable[[str, str], None],
        *,
        error_sink: Callable[[str], None] | None = None,
    ):
        self.sender = sender
        self.error_sink = error_sink or (lambda _message: None)
        self.redactor = SecretRedactor()
        self._stop = threading.Event()
        self._wakeup = threading.Event()
        self._lock = threading.Lock()
        self._pending: set[tuple[str, str]] = set()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self.run_forever,
            name="ai-agents-activity",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wakeup.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def notify(self, channel: str, conversation_id: str) -> None:
        with self._lock:
            self._pending.add((channel, conversation_id))
        self._wakeup.set()

    def run_forever(self) -> None:
        while not self._stop.is_set():
            self._wakeup.wait()
            self._wakeup.clear()
            if self._stop.is_set():
                break
            with self._lock:
                pending = tuple(sorted(self._pending))
                self._pending.clear()
            for channel, conversation_id in pending:
                try:
                    self.sender(channel, conversation_id)
                except Exception as exc:
                    self.error_sink(
                        "활동 상태 전송 오류 "
                        f"channel={channel} conversation={conversation_id} "
                        f"error={type(exc).__name__}:"
                        f"{self.redactor.text(str(exc))[:300]}"
                    )
