from __future__ import annotations

import os
import json
import threading
from dataclasses import dataclass
from pathlib import Path

from app.contracts.models import utc_now
from app.services.logging.redaction import SecretRedactor


@dataclass(frozen=True)
class GatewayLogPolicy:
    max_bytes: int = 5 * 1024 * 1024
    backup_count: int = 14

    def __post_init__(self) -> None:
        if self.max_bytes < 1024:
            raise ValueError("gateway log max_bytes must be at least 1024")
        if not 1 <= self.backup_count <= 90:
            raise ValueError("gateway log backup_count must be between 1 and 90")


class GatewayLog:
    """실행 ID가 생기기 전의 연결 오류를 남기는 전역 로그."""

    def __init__(
        self,
        path: Path,
        redactor: SecretRedactor | None = None,
        *,
        policy: GatewayLogPolicy | None = None,
    ):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.redactor = redactor or SecretRedactor()
        self.policy = policy or GatewayLogPolicy()
        self._lock = threading.RLock()

    def write(self, message: str) -> None:
        self.event("GATEWAY_DIAGNOSTIC", message)

    def event(self, event_type: str, message: str, **data) -> None:
        safe_message = self.redactor.text(message.strip())
        record = {
            "timestamp": utc_now(),
            "event_type": event_type,
            "message": safe_message,
            "data": self.redactor.value(data),
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock:
            self._rotate_if_needed(len(line.encode("utf-8")))
            with self.path.open("a", encoding="utf-8", newline="") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        print(safe_message, flush=True)

    def _rotate_if_needed(self, incoming_bytes: int) -> None:
        try:
            if not self.path.exists() or self.path.stat().st_size + incoming_bytes <= self.policy.max_bytes:
                return
            oldest = self.path.with_name(f"{self.path.name}.{self.policy.backup_count}")
            oldest.unlink(missing_ok=True)
            for index in range(self.policy.backup_count - 1, 0, -1):
                source = self.path.with_name(f"{self.path.name}.{index}")
                if source.exists():
                    source.replace(self.path.with_name(f"{self.path.name}.{index + 1}"))
            self.path.replace(self.path.with_name(f"{self.path.name}.1"))
        except OSError:
            # Gateway diagnostics must not take down the message loop because a
            # backup volume is temporarily unavailable.
            return
