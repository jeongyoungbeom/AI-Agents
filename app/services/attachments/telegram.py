from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from app.gateway.core.models import Attachment, IncomingMessage


class TelegramFileClient(Protocol):
    def get_file_path(self, file_id: str) -> str: ...

    def download_file(self, file_path: str, *, max_bytes: int) -> bytes: ...


@dataclass(frozen=True)
class AttachmentPolicy:
    max_bytes: int = 8 * 1024 * 1024
    max_text_characters: int = 24_000

    def __post_init__(self) -> None:
        if self.max_bytes < 1_024:
            raise ValueError("attachment max_bytes must be at least 1024")
        if self.max_text_characters < 1_000:
            raise ValueError("attachment max_text_characters must be at least 1000")


_TEXT_EXTENSIONS = frozenset(
    {
        ".c", ".cc", ".cpp", ".cs", ".css", ".csv", ".env.example",
        ".go", ".h", ".hpp", ".html", ".ini", ".java", ".js", ".json",
        ".jsx", ".kt", ".kts", ".log", ".md", ".mjs", ".py", ".rb",
        ".rs", ".sh", ".sql", ".toml", ".ts", ".tsx", ".txt", ".xml",
        ".yaml", ".yml",
    }
)
_IMAGE_SIGNATURES = (
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"GIF87a",
    b"GIF89a",
)
_BLOCKED_BINARY_PREFIXES = (b"MZ", b"\x7fELF", b"PK\x03\x04", b"\xd0\xcf\x11\xe0")
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._ -]+")


class TelegramAttachmentService:
    """Download files into gateway-owned storage without ever executing them.

    The service only extracts UTF-8 text from a narrow allowlist. Images are
    retained after signature checks and can be passed as an image attachment to
    an image-capable Hermes call. Archives, executables, office binaries and
    unsupported media remain rejected before any model sees their bytes.
    """

    def __init__(
        self,
        client: TelegramFileClient,
        storage_root: Path,
        policy: AttachmentPolicy,
    ) -> None:
        self.client = client
        self.storage_root = storage_root.resolve()
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.policy = policy

    def hydrate_message(self, message: IncomingMessage) -> IncomingMessage:
        if not message.attachments:
            return message
        return replace(
            message,
            attachments=tuple(self.hydrate(attachment) for attachment in message.attachments),
        )

    def hydrate(self, attachment: Attachment) -> Attachment:
        if attachment.kind not in {"document", "photo"}:
            return self._rejected(attachment, "지원하지 않는 첨부파일 종류입니다.")
        if attachment.size is not None and attachment.size > self.policy.max_bytes:
            return self._rejected(attachment, "파일 크기가 허용 한도를 초과했습니다.")
        try:
            file_path = self.client.get_file_path(attachment.external_id)
            content = self.client.download_file(file_path, max_bytes=self.policy.max_bytes)
        except Exception:
            # Telegram API details must not be mirrored to the owner or logs;
            # the original file id also remains out of the model context.
            return self._rejected(attachment, "Telegram에서 파일을 내려받지 못했습니다.")
        if len(content) > self.policy.max_bytes:
            return self._rejected(attachment, "파일 크기가 허용 한도를 초과했습니다.")
        if not content:
            return self._rejected(attachment, "빈 파일은 처리할 수 없습니다.")
        try:
            if attachment.kind == "document":
                return self._hydrate_document(attachment, content)
            return self._hydrate_image(attachment, content)
        except (OSError, ValueError, UnicodeError):
            return self._rejected(attachment, "첨부파일을 안전하게 저장하지 못했습니다.")

    def _hydrate_document(self, attachment: Attachment, content: bytes) -> Attachment:
        filename = self._safe_document_name(attachment.name)
        suffix = Path(filename).suffix.lower()
        if suffix not in _TEXT_EXTENSIONS and not filename.lower().endswith(".env.example"):
            return self._rejected(attachment, "허용되지 않은 문서 형식입니다.")
        if content.startswith(_BLOCKED_BINARY_PREFIXES) or b"\x00" in content:
            return self._rejected(attachment, "실행 파일 또는 바이너리 형식은 처리하지 않습니다.")
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return self._rejected(attachment, "UTF-8 텍스트 문서만 처리할 수 있습니다.")
        stored_path, digest = self._store(content, filename, attachment.external_id)
        preview = text[: self.policy.max_text_characters]
        return replace(
            attachment,
            name=filename,
            size=len(content),
            metadata={
                "attachment_origin": "telegram_gateway",
                "status": "accepted",
                "content_kind": "text",
                "content_sha256": digest,
                "stored_path": str(stored_path),
                "text_preview": preview,
                "truncated": len(text) > len(preview),
                "scan": "extension, size, binary-signature, utf8",
            },
        )

    def _hydrate_image(self, attachment: Attachment, content: bytes) -> Attachment:
        is_webp = len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP"
        if not content.startswith(_IMAGE_SIGNATURES) and not is_webp:
            return self._rejected(attachment, "사진 파일 서명을 확인할 수 없습니다.")
        filename = self._safe_image_name(attachment.name, content)
        stored_path, digest = self._store(content, filename, attachment.external_id)
        return replace(
            attachment,
            name=filename,
            size=len(content),
            metadata={
                "attachment_origin": "telegram_gateway",
                "status": "accepted",
                "content_kind": "image",
                "content_sha256": digest,
                "stored_path": str(stored_path),
                "scan": "size, image-signature",
            },
        )

    def _store(self, content: bytes, name: str, external_id: str) -> tuple[Path, str]:
        digest = hashlib.sha256(content).hexdigest()
        source = hashlib.sha256(external_id.encode("utf-8")).hexdigest()[:16]
        directory = self.storage_root / source
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{digest[:16]}-{name}"
        temporary = target.with_suffix(target.suffix + ".tmp")
        if not target.exists():
            temporary.write_bytes(content)
            os.replace(temporary, target)
        return target.resolve(), digest

    @staticmethod
    def _safe_document_name(name: str) -> str:
        raw = Path(name or "attachment.txt").name.strip()
        cleaned = _SAFE_FILENAME.sub("_", raw).strip(" .")
        return cleaned[:120] or "attachment.txt"

    @staticmethod
    def _safe_image_name(name: str, content: bytes) -> str:
        suffix = ".jpg"
        if content.startswith(b"\x89PNG"):
            suffix = ".png"
        elif content.startswith((b"GIF87a", b"GIF89a")):
            suffix = ".gif"
        elif len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
            suffix = ".webp"
        raw = TelegramAttachmentService._safe_document_name(name or f"photo{suffix}")
        return raw if Path(raw).suffix.lower() == suffix else f"{Path(raw).stem}{suffix}"

    @staticmethod
    def _rejected(attachment: Attachment, reason: str) -> Attachment:
        return replace(
            attachment,
            metadata={
                "attachment_origin": "telegram_gateway",
                "status": "rejected",
                "reason": reason,
            },
        )


def accepted_image_path(message: IncomingMessage, storage_root: Path) -> Path | None:
    """Return one immutable, gateway-owned image path for a Hermes call.

    Metadata alone is never trusted: the file must still be below the
    attachment store and match the digest recorded when it was downloaded.
    """
    root = storage_root.resolve()
    for attachment in message.attachments:
        metadata = attachment.metadata
        if (
            metadata.get("attachment_origin") != "telegram_gateway"
            or metadata.get("status") != "accepted"
            or metadata.get("content_kind") != "image"
        ):
            continue
        raw_path = metadata.get("stored_path")
        digest = metadata.get("content_sha256")
        if not isinstance(raw_path, str) or not isinstance(digest, str):
            continue
        try:
            candidate = Path(raw_path).resolve(strict=True)
            candidate.relative_to(root)
            if not candidate.is_file():
                continue
            with candidate.open("rb") as handle:
                actual = hashlib.file_digest(handle, "sha256").hexdigest()
        except (OSError, ValueError):
            continue
        if actual == digest:
            return candidate
    return None
