from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from app.services.logging.redaction import SecretRedactor
from app.storage import StateStore

from .conversation import DialogueRouter
from .models import IncomingMessage, OutgoingMessage
from .ports import ConversationScheduler
from .security import AccessPolicy


def should_hydrate_inbound(
    store: StateStore, policy: AccessPolicy, message: IncomingMessage,
    *, max_processing_attempts: int,
) -> bool:
    """Reject unauthorized and already claimed updates before any file download."""
    if not policy.allows(message):
        return False
    receipt = store.inbound_receipt(
        message.channel, f"{message.conversation_id}:{message.external_message_id}"
    )
    return receipt is None or (
        receipt["status"] == "FAILED"
        and receipt["attempts"] < max_processing_attempts
    )


class GatewayApplication:
    """권한 확인과 메시지 중복 방지를 담당하는 채널 독립 진입점."""

    def __init__(
        self,
        store: StateStore,
        router: DialogueRouter,
        access_policy: AccessPolicy,
        *,
        max_processing_attempts: int = 2,
        redactor: SecretRedactor | None = None,
        message_preparer: Callable[
            [OutgoingMessage], tuple[OutgoingMessage, ...]
        ] | None = None,
        diagnostic_sink: Callable[[str], None] | None = None,
        conversation_scheduler: ConversationScheduler | None = None,
    ):
        self.store = store
        self.router = router
        self.access_policy = access_policy
        self.max_processing_attempts = max_processing_attempts
        self.redactor = redactor or SecretRedactor()
        self.message_preparer = message_preparer or (lambda message: (message,))
        self.diagnostic_sink = diagnostic_sink or (lambda _message: None)
        self.conversation_scheduler = conversation_scheduler

    def handle(self, message: IncomingMessage) -> tuple[OutgoingMessage, ...]:
        if not self.access_policy.allows(message):
            self.diagnostic_sink(
                "접근 거부 "
                f"channel={message.channel} conversation={message.conversation_id} "
                f"user={message.user_id} message={message.external_message_id}"
            )
            return ()
        receipt_id = self._receipt_id(message)
        claimed = self.store.claim_inbound(
            message.channel,
            receipt_id,
            max_attempts=self.max_processing_attempts,
        )
        if not claimed:
            return ()
        try:
            if (
                self.conversation_scheduler is not None
                and not self.router.is_immediate_control(message)
            ):
                with self.store.transaction():
                    job = self.conversation_scheduler.enqueue(message)
                    self.store.complete_inbound_with_outbound(
                        message.channel, receipt_id, []
                    )
                self.diagnostic_sink(
                    "대화 작업 등록 "
                    f"job={job.get('job_id', '?')} channel={message.channel} "
                    f"conversation={message.conversation_id}"
                )
                return ()
            is_new_control = getattr(
                self.router, "is_new_control", lambda _message: False
            )
            if (
                self.conversation_scheduler is not None
                and (
                    self.router.is_stop_control(message)
                    or is_new_control(message)
                )
            ):
                self.conversation_scheduler.cancel(
                    message.channel, message.conversation_id
                )
            immediate_check = getattr(
                self.router, "is_immediate_control", lambda _message: False
            )
            immediate = bool(immediate_check(message))
            if immediate:
                # 상태/중지/새 작업은 모델을 호출하지 않으므로 응답까지 한 번에 확정한다.
                with self.store.transaction():
                    outgoing = self.router.route(message)
                    prepared = tuple(
                        part
                        for item in outgoing
                        for part in self.message_preparer(item)
                    )
                    outbound_ids = self.store.complete_inbound_with_outbound(
                        message.channel,
                        receipt_id,
                        [
                            {
                                "channel": item.channel,
                                "conversation_id": item.conversation_id,
                                "text": item.text,
                                "reply_to": item.reply_to,
                            }
                            for item in prepared
                        ],
                    )
            else:
                # 모델 호출은 수십 초 걸릴 수 있으므로 SQLite 쓰기 트랜잭션 밖에서 수행한다.
                outgoing = self.router.route(message)
                prepared = tuple(
                    part
                    for item in outgoing
                    for part in self.message_preparer(item)
                )
                with self.store.transaction():
                    outbound_ids = self.store.complete_inbound_with_outbound(
                        message.channel,
                        receipt_id,
                        [
                            {
                                "channel": item.channel,
                                "conversation_id": item.conversation_id,
                                "text": item.text,
                                "reply_to": item.reply_to,
                            }
                            for item in prepared
                        ],
                    )
        except Exception as exc:
            self.store.fail_inbound(
                message.channel,
                receipt_id,
                self.redactor.text(str(exc)),
            )
            raise
        queued: list[OutgoingMessage] = []
        for item, outbound_id in zip(prepared, outbound_ids, strict=True):
            metadata = dict(item.metadata)
            metadata["gateway_outbox_id"] = outbound_id
            queued.append(replace(item, metadata=metadata))
        return tuple(queued)

    @staticmethod
    def _receipt_id(message: IncomingMessage) -> str:
        return f"{message.conversation_id}:{message.external_message_id}"
