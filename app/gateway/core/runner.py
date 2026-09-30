from __future__ import annotations

import time
import threading
import uuid
from collections.abc import Callable

from app.services.logging.redaction import SecretRedactor

from .application import GatewayApplication
from .models import OutgoingMessage
from .ports import ChatAdapter


class GatewayWorkerStopped(RuntimeError):
    """필수 백그라운드 워커가 예기치 않게 종료됐을 때 발생한다."""


class GatewayRunner:
    """어댑터 폴링과 공통 애플리케이션을 연결한다."""

    def __init__(
        self,
        adapter: ChatAdapter,
        application: GatewayApplication,
        *,
        error_sink: Callable[[str], None] | None = None,
        health_check: Callable[[], None] | None = None,
        connection_retry_limit: int = 8,
        retry_max_seconds: float = 60.0,
    ):
        self.adapter = adapter
        self.application = application
        self.error_sink = error_sink or (lambda message: print(message, flush=True))
        self.health_check = health_check or (lambda: None)
        self.redactor = SecretRedactor()
        self.instance_id = uuid.uuid4().hex
        self.connection_retry_limit = connection_retry_limit
        self.retry_max_seconds = retry_max_seconds
        self._delivery_lock = threading.Lock()
        if connection_retry_limit < 1 or retry_max_seconds <= 0:
            raise ValueError("gateway retry limits must be positive")

    def run_once(self) -> int:
        self.health_check()
        delivered = self.flush_outbound()
        cursor = self.application.store.load_gateway_cursor(self.adapter.name)
        batch = self.adapter.poll(cursor)
        processing_failed = False
        for incoming in batch.messages:
            try:
                outgoing_messages = self.application.handle(incoming)
            except Exception as exc:
                processing_failed = True
                self.error_sink(
                    "메시지 처리 실패 "
                    f"channel={incoming.channel} conversation={incoming.conversation_id} "
                    f"message={incoming.external_message_id} "
                    f"error={type(exc).__name__}:{self.redactor.text(str(exc))[:300]}"
                )
                try:
                    self.adapter.send(
                        OutgoingMessage(
                            channel=incoming.channel,
                            conversation_id=incoming.conversation_id,
                            text="메시지 처리 중 오류가 발생했습니다. 한 번 재시도한 뒤 계속 실패하면 중단합니다.",
                            reply_to=incoming.external_message_id,
                        )
                    )
                except Exception:
                    self.error_sink("오류 안내 메시지도 전송하지 못했습니다.")
                continue
            if outgoing_messages:
                delivered += self.flush_outbound()
        if not processing_failed and batch.cursor is not None:
            self.application.store.save_gateway_cursor(self.adapter.name, batch.cursor)
        return delivered

    def flush_outbound(self) -> int:
        """백그라운드 워커와 폴링 루프가 발신함을 안전하게 함께 비운다."""
        with self._delivery_lock:
            return self._deliver_pending_unlocked()

    def _deliver_pending_unlocked(self) -> int:
        delivered = 0
        while True:
            item = self.application.store.claim_next_outbound(
                self.adapter.name,
                self.instance_id,
                max_attempts=self.application.max_processing_attempts,
            )
            if item is None:
                break
            delivery_mode = str(item.get("delivery_mode", "send"))
            target_message_id = ""
            if delivery_mode == "edit":
                target = self.application.store.outbound_record(
                    int(item.get("target_outbound_id", 0))
                )
                if target is not None and target["external_ids"]:
                    target_message_id = str(target["external_ids"][0])
                if not target_message_id:
                    self.application.store.mark_outbound_uncertain(
                        int(item["outbound_id"]),
                        "progress target message id is unknown",
                        lease_owner=self.instance_id,
                    )
                    self.error_sink(
                        f"진행 카드 수정 보류 outbox={item['outbound_id']}: 원본 메시지 ID 불명"
                    )
                    continue
            outgoing = OutgoingMessage(
                channel=item["channel"],
                conversation_id=item["conversation_id"],
                text=item["text"],
                reply_to=item["reply_to"],
                metadata={
                    "gateway_outbox_id": item["outbound_id"],
                    "delivery_mode": delivery_mode,
                    "target_message_id": target_message_id,
                },
            )
            try:
                self._deliver(outgoing)
                delivered += 1
            except Exception as exc:
                self.error_sink(
                    "응답 전송 실패 "
                    f"channel={self.adapter.name} conversation={item['conversation_id']} "
                    f"outbox={item['outbound_id']} attempt={item['attempts'] + 1} "
                    f"error={type(exc).__name__}:{self.redactor.text(str(exc))[:300]}"
                )
                break
        return delivered

    def _deliver(self, message: OutgoingMessage) -> None:
        outbound_id = int(message.metadata["gateway_outbox_id"])
        external_ids: list[str] = []
        try:
            parts = self.adapter.prepare(message)
            if message.metadata.get("delivery_mode") == "edit" and len(parts) != 1:
                raise ValueError("editable progress text exceeds channel message limit")
            for part in parts:
                external_ids.extend(self.adapter.send(part))
        except Exception as exc:
            if external_ids:
                self.application.store.mark_outbound_uncertain(
                    outbound_id, self.redactor.text(str(exc)), tuple(external_ids),
                    lease_owner=self.instance_id,
                )
            else:
                self.application.store.fail_outbound(
                    outbound_id,
                    self.redactor.text(str(exc)),
                    lease_owner=self.instance_id,
                    max_attempts=self.application.max_processing_attempts,
                )
            raise
        self.application.store.complete_outbound(
            outbound_id, tuple(external_ids), lease_owner=self.instance_id
        )

    def run_forever(
        self,
        *,
        retry_delay_seconds: float = 3.0,
        stop_requested: Callable[[], bool] | None = None,
    ) -> None:
        recovered = self.application.store.recover_processing_inbound(self.adapter.name)
        if recovered:
            self.error_sink(f"재시작으로 중단된 메시지 {recovered}개를 재시도 대상으로 복구했습니다.")
        consecutive_failures = 0
        while True:
            if stop_requested is not None and stop_requested():
                self.error_sink("운영 중지 요청을 받아 게이트웨이를 안전하게 종료합니다.")
                return
            try:
                self.run_once()
                consecutive_failures = 0
            except KeyboardInterrupt:
                raise
            except GatewayWorkerStopped as exc:
                self.error_sink(
                    "게이트웨이 워커 오류 "
                    f"error={type(exc).__name__}:{self.redactor.text(str(exc))[:300]}"
                )
                raise
            except Exception as exc:
                consecutive_failures += 1
                retry_after = getattr(exc, "retry_after", None)
                delay = min(
                    self.retry_max_seconds,
                    float(retry_after)
                    if retry_after is not None
                    else retry_delay_seconds * (2 ** (consecutive_failures - 1)),
                )
                self.error_sink(
                    "게이트웨이 연결 오류 "
                    f"channel={self.adapter.name} failure={consecutive_failures}/"
                    f"{self.connection_retry_limit} retry_after={delay:g}s "
                    f"error={type(exc).__name__}:{self.redactor.text(str(exc))[:300]}"
                )
                if consecutive_failures >= self.connection_retry_limit:
                    raise RuntimeError("게이트웨이 연결 재시도 한도를 초과했습니다.") from exc
                remaining = delay
                while remaining > 0:
                    if stop_requested is not None and stop_requested():
                        self.error_sink("운영 중지 요청을 받아 게이트웨이를 안전하게 종료합니다.")
                        return
                    sleep_for = min(0.5, remaining)
                    time.sleep(sleep_for)
                    remaining -= sleep_for
