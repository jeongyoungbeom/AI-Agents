from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from app.contracts import RunState
from app.contracts.models import utc_now
from app.services.logging.redaction import SecretRedactor
from app.storage import StateStore


class AuditLogger:
    """Writes every event to SQLite, JSONL, and a compact timeline immediately."""

    def __init__(
        self,
        artifacts_root: Path,
        store: StateStore,
        redactor: SecretRedactor | None = None,
    ):
        self.artifacts_root = artifacts_root.resolve()
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        self.store = store
        self.redactor = redactor or SecretRedactor()
        self._lock = threading.RLock()

    def run_dir(self, run_id: str) -> Path:
        path = (self.artifacts_root / run_id).resolve()
        try:
            path.relative_to(self.artifacts_root)
        except ValueError as exc:
            raise ValueError("run_id escapes artifacts root") from exc
        path.mkdir(parents=True, exist_ok=True)
        return path

    def emit(
        self,
        run_id: str,
        event_type: str,
        message: str,
        *,
        stage_id: str = "",
        role_id: str = "",
        status: str = "",
        data: dict[str, Any] | None = None,
    ) -> None:
        timestamp = utc_now()
        safe_message = self.redactor.text(message)
        safe_data = self.redactor.value(data or {})
        event = {
            "timestamp": timestamp,
            "run_id": run_id,
            "stage_id": stage_id,
            "role_id": role_id,
            "event_type": event_type,
            "status": status,
            "message": safe_message,
            "data": safe_data,
        }
        run_dir = self.run_dir(run_id)
        self.store.append_event(
            run_id,
            timestamp,
            event_type,
            safe_message,
            stage_id=stage_id,
            role_id=role_id,
            status=status,
            data=safe_data,
        )

        def persist_files() -> None:
            with self._lock:
                self._append_and_sync(
                    run_dir / "events.jsonl",
                    json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n",
                )
                labels = [timestamp, stage_id, role_id, event_type, status]
                prefix = "][".join(label for label in labels if label)
                self._append_and_sync(
                    run_dir / "timeline.log", f"[{prefix}] {safe_message}\n"
                )

        self.store.after_commit(persist_files)

    def write_summary(self, state: RunState) -> Path:
        events = self.store.list_events(state.run_id)
        lines = [
            f"# 실행 요약: {state.run_id}",
            "",
            f"- 상태: {state.phase.value}",
            f"- 프로젝트: {state.repository or '(미지정)'}",
            f"- 작업: {state.objective or '(미지정)'}",
            f"- 단계: {state.stage_index + 1}/{state.stage_count}",
            f"- 사용 토큰: {state.total_tokens}",
            f"- 승인 여부: {'예' if state.approval_granted else '아니오'}",
            f"- 최근 오류: {self.redactor.text(state.last_error) if state.last_error else '(없음)'}",
            "",
            "## 최근 기록",
            "",
        ]
        for event in events[-30:]:
            role = f"/{event['role_id']}" if event["role_id"] else ""
            stage = f"/{event['stage_id']}" if event["stage_id"] else ""
            lines.append(
                f"- {event['timestamp']} [{event['event_type']}{stage}{role}] "
                f"{event['message']}"
            )
        path = self.run_dir(state.run_id) / "summary.md"
        temporary = path.with_suffix(".md.tmp")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary.replace(path)
        return path

    @staticmethod
    def _append_and_sync(path: Path, text: str) -> None:
        with path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
