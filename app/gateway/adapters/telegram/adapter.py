from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Protocol

from app.gateway.core.models import (
    Attachment,
    IncomingMessage,
    OutgoingMessage,
    PollBatch,
)


class TelegramClientPort(Protocol):
    def get_me(self) -> dict[str, Any]: ...

    def get_updates(
        self, offset: int | None, *, timeout_seconds: int
    ) -> list[dict[str, Any]]: ...

    def send_message(self, chat_id: str, text: str) -> str: ...

    def edit_message_text(
        self, chat_id: str, message_id: str, text: str
    ) -> str: ...

    def get_file_path(self, file_id: str) -> str: ...

    def download_file(self, file_path: str, *, max_bytes: int) -> bytes: ...


class TelegramAdapter:
    name = "telegram"

    def __init__(
        self,
        client: TelegramClientPort,
        *,
        poll_timeout_seconds: int = 25,
        max_message_characters: int = 3900,
        attachment_processor: Callable[[IncomingMessage], IncomingMessage] | None = None,
        inbound_filter: Callable[[IncomingMessage], bool] | None = None,
    ):
        if not 1 <= poll_timeout_seconds <= 50:
            raise ValueError("Telegram poll timeout must be between 1 and 50 seconds")
        if not 100 <= max_message_characters <= 4096:
            raise ValueError("Telegram message size must be between 100 and 4096")
        self.client = client
        self.poll_timeout_seconds = poll_timeout_seconds
        self.max_message_characters = max_message_characters
        self.attachment_processor = attachment_processor
        self.inbound_filter = inbound_filter

    def check(self, *, online: bool = True) -> str:
        if not online:
            return "텔레그램 로컬 설정이 준비되었습니다."
        me = self.client.get_me()
        username = str(me.get("username", "")).strip() or "이름 없음"
        return f"텔레그램 봇 @{username} 연결을 확인했습니다."

    def poll(self, cursor: str | None) -> PollBatch:
        offset = int(cursor) if cursor else None
        updates = self.client.get_updates(
            offset, timeout_seconds=self.poll_timeout_seconds
        )
        messages: list[IncomingMessage] = []
        next_offset = offset
        for update in updates:
            try:
                update_id = int(update["update_id"])
            except (KeyError, TypeError, ValueError):
                continue
            next_offset = max(next_offset or 0, update_id + 1)
            normalized = self.normalize_update(update)
            if normalized is not None:
                if self.inbound_filter is not None and not self.inbound_filter(normalized):
                    continue
                if self.attachment_processor is not None:
                    normalized = self.attachment_processor(normalized)
                messages.append(normalized)
        return PollBatch(
            messages=tuple(messages),
            cursor=str(next_offset) if next_offset is not None else cursor,
        )

    def prepare(self, message: OutgoingMessage) -> tuple[OutgoingMessage, ...]:
        if message.channel != self.name:
            raise ValueError(
                f"Telegram adapter cannot send channel {message.channel}"
            )
        parts = self.split_text(message.text, self.max_message_characters)
        return tuple(
            replace(
                message,
                text=part,
                metadata={
                    **message.metadata,
                    "part_index": index,
                    "part_count": len(parts),
                },
            )
            for index, part in enumerate(parts, start=1)
        )

    def send(self, message: OutgoingMessage) -> tuple[str, ...]:
        if message.channel != self.name:
            raise ValueError(
                f"Telegram adapter cannot send channel {message.channel}"
            )
        if len(message.text) > self.max_message_characters:
            raise ValueError("전송 전 prepare()로 메시지를 분할해야 합니다.")
        delivery_mode = str(message.metadata.get("delivery_mode", "send"))
        if delivery_mode == "edit":
            target_message_id = str(
                message.metadata.get("target_message_id", "")
            ).strip()
            if not target_message_id:
                raise ValueError("Telegram edit requires a target message id")
            return (
                self.client.edit_message_text(
                    message.conversation_id,
                    target_message_id,
                    message.text,
                ),
            )
        if delivery_mode != "send":
            raise ValueError(f"unsupported Telegram delivery mode: {delivery_mode}")
        return (self.client.send_message(message.conversation_id, message.text),)

    @staticmethod
    def normalize_update(update: dict[str, Any]) -> IncomingMessage | None:
        raw_message = update.get("message")
        if not isinstance(raw_message, dict):
            return None
        sender = raw_message.get("from")
        chat = raw_message.get("chat")
        if not isinstance(sender, dict) or not isinstance(chat, dict):
            return None
        text = raw_message.get("text") or raw_message.get("caption") or ""
        attachments = TelegramAdapter._attachments(raw_message)
        if not str(text).strip() and not attachments:
            return None
        update_id = str(update.get("update_id", "")).strip()
        if not update_id:
            return None
        display_name = " ".join(
            part
            for part in (
                str(sender.get("first_name", "")).strip(),
                str(sender.get("last_name", "")).strip(),
            )
            if part
        )
        return IncomingMessage(
            channel="telegram",
            conversation_id=str(chat.get("id", "")),
            user_id=str(sender.get("id", "")),
            external_message_id=f"update-{update_id}",
            text=str(text),
            is_private=str(chat.get("type", "")) == "private",
            user_display_name=display_name,
            attachments=attachments,
            metadata={"telegram_message_id": str(raw_message.get("message_id", ""))},
        )

    @staticmethod
    def _attachments(message: dict[str, Any]) -> tuple[Attachment, ...]:
        attachments: list[Attachment] = []
        document = message.get("document")
        if isinstance(document, dict) and document.get("file_id"):
            attachments.append(
                Attachment(
                    kind="document",
                    external_id=str(document["file_id"]),
                    name=str(document.get("file_name", "")),
                    size=int(document["file_size"])
                    if document.get("file_size") is not None
                    else None,
                    metadata={
                        "mime_type": str(document.get("mime_type", "")),
                        "file_unique_id": str(document.get("file_unique_id", "")),
                    },
                )
            )
        photos = message.get("photo")
        if isinstance(photos, list) and photos:
            photo = photos[-1]
            if isinstance(photo, dict) and photo.get("file_id"):
                attachments.append(
                Attachment(
                    kind="photo",
                    external_id=str(photo["file_id"]),
                        size=int(photo["file_size"])
                    if photo.get("file_size") is not None
                    else None,
                    metadata={
                        "file_unique_id": str(photo.get("file_unique_id", "")),
                    },
                    )
                )
        voice = message.get("voice")
        if isinstance(voice, dict) and voice.get("file_id"):
            attachments.append(
                Attachment(
                    kind="voice",
                    external_id=str(voice["file_id"]),
                    size=int(voice["file_size"])
                    if voice.get("file_size") is not None
                    else None,
                )
            )
        return tuple(attachments)

    @staticmethod
    def split_text(text: str, limit: int = 3900) -> tuple[str, ...]:
        if not text:
            return ()
        parts: list[str] = []
        remaining = text
        while len(remaining) > limit:
            split_at = remaining.rfind("\n", 0, limit + 1)
            if split_at < limit // 2:
                split_at = remaining.rfind(" ", 0, limit + 1)
            if split_at < limit // 2:
                split_at = limit
            part = remaining[:split_at].rstrip()
            if not part:
                part = remaining[:limit]
                split_at = limit
            parts.append(part)
            remaining = remaining[split_at:].lstrip()
        if remaining:
            parts.append(remaining)
        return tuple(parts)
