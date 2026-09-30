from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import PurePosixPath
from typing import Any


class TelegramAPIError(RuntimeError):
    def __init__(self, message: str, *, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class TelegramClient:
    """봇 토큰이 오류 메시지에 노출되지 않는 작은 Telegram Bot API 클라이언트."""

    def __init__(self, token: str, *, request_timeout_seconds: int = 20):
        if not token.strip():
            raise ValueError("Telegram bot token is required")
        self._token = token.strip()
        self.request_timeout_seconds = request_timeout_seconds

    def get_me(self) -> dict[str, Any]:
        result = self._call("getMe", {})
        if not isinstance(result, dict):
            raise TelegramAPIError("Telegram getMe 응답 형식이 올바르지 않습니다.")
        return result

    def delete_webhook(self) -> None:
        self._call("deleteWebhook", {"drop_pending_updates": "false"})

    def get_updates(
        self, offset: int | None, *, timeout_seconds: int
    ) -> list[dict[str, Any]]:
        payload: dict[str, str] = {
            "timeout": str(timeout_seconds),
            "allowed_updates": json.dumps(["message"], ensure_ascii=True),
        }
        if offset is not None:
            payload["offset"] = str(offset)
        result = self._call(
            "getUpdates",
            payload,
            timeout_seconds=max(self.request_timeout_seconds, timeout_seconds + 5),
        )
        if not isinstance(result, list):
            raise TelegramAPIError("Telegram getUpdates 응답 형식이 올바르지 않습니다.")
        return [item for item in result if isinstance(item, dict)]

    def send_message(self, chat_id: str, text: str) -> str:
        result = self._call(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": text,
                "disable_web_page_preview": "true",
            },
        )
        if not isinstance(result, dict):
            raise TelegramAPIError("Telegram sendMessage 응답 형식이 올바르지 않습니다.")
        return str(result.get("message_id", ""))

    def edit_message_text(self, chat_id: str, message_id: str, text: str) -> str:
        result = self._call(
            "editMessageText",
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": text,
                "disable_web_page_preview": "true",
            },
        )
        if not isinstance(result, dict):
            raise TelegramAPIError(
                "Telegram editMessageText 응답 형식이 올바르지 않습니다."
            )
        return str(result.get("message_id", message_id))

    def send_chat_action(self, chat_id: str, action: str = "typing") -> None:
        result = self._call(
            "sendChatAction",
            {"chat_id": chat_id, "action": action},
        )
        if result is not True:
            raise TelegramAPIError(
                "Telegram sendChatAction 응답 형식이 올바르지 않습니다."
            )

    def get_file_path(self, file_id: str) -> str:
        """Resolve a Telegram file id without exposing it to conversation code."""
        result = self._call("getFile", {"file_id": file_id})
        if not isinstance(result, dict):
            raise TelegramAPIError("Telegram getFile 응답 형식이 올바르지 않습니다.")
        file_path = str(result.get("file_path", "")).strip()
        path = PurePosixPath(file_path)
        if not file_path or path.is_absolute() or ".." in path.parts:
            raise TelegramAPIError("Telegram getFile 경로가 올바르지 않습니다.")
        return file_path

    def download_file(self, file_path: str, *, max_bytes: int) -> bytes:
        """Fetch a bounded attachment through Telegram's file endpoint.

        The file path comes only from ``getFile``.  It is checked again here so
        callers cannot turn the bot token into an arbitrary URL fetcher.
        """
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        path = PurePosixPath(file_path)
        if not file_path or path.is_absolute() or ".." in path.parts:
            raise TelegramAPIError("Telegram 파일 경로가 올바르지 않습니다.")
        encoded_path = urllib.parse.quote(file_path, safe="/")
        request = urllib.request.Request(
            f"https://api.telegram.org/file/bot{self._token}/{encoded_path}",
            method="GET",
        )
        chunks: list[bytes] = []
        received = 0
        try:
            with urllib.request.urlopen(
                request, timeout=self.request_timeout_seconds
            ) as response:
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > max_bytes:
                        raise TelegramAPIError("Telegram 파일 크기가 허용 한도를 초과했습니다.")
                    chunks.append(chunk)
        except TelegramAPIError:
            raise
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            raise TelegramAPIError("Telegram 파일 다운로드에 실패했습니다.") from exc
        return b"".join(chunks)

    def _call(
        self,
        method: str,
        payload: dict[str, str],
        *,
        timeout_seconds: int | None = None,
    ) -> Any:
        data = urllib.parse.urlencode(payload).encode("utf-8")
        request = urllib.request.Request(
            f"https://api.telegram.org/bot{self._token}/{method}",
            data=data,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout_seconds or self.request_timeout_seconds,
            ) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise TelegramAPIError(
                f"Telegram {method} 요청에 실패했습니다."
            ) from exc
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise TelegramAPIError(
                f"Telegram {method} 응답이 JSON이 아닙니다."
            ) from exc
        if not isinstance(decoded, dict) or not decoded.get("ok"):
            description = ""
            retry_after = None
            if isinstance(decoded, dict):
                description = str(decoded.get("description", ""))[:300]
                parameters = decoded.get("parameters")
                if isinstance(parameters, dict):
                    raw_retry_after = parameters.get("retry_after")
                    try:
                        retry_after = int(raw_retry_after)
                    except (TypeError, ValueError):
                        retry_after = None
            raise TelegramAPIError(
                f"Telegram {method} 오류: {description or '알 수 없는 오류'}",
                retry_after=retry_after,
            )
        return decoded.get("result")
