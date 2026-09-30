from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from app.contracts import AgentHandoff, StageContract
from app.services.logging.redaction import SecretRedactor


class ArtifactStore:
    """Human-readable files stored outside the user's selected project repository."""

    def __init__(self, root: Path, redactor: SecretRedactor | None = None):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.redactor = redactor or SecretRedactor()
        self._lock = threading.RLock()

    def save_contract(self, contract: StageContract) -> Path:
        return self.write_json(
            contract.run_id,
            Path("stages") / contract.stage_id / "contract.json",
            contract.to_dict(),
        )

    def save_plan_revision(
        self,
        run_id: str,
        revision: int,
        plan_hash: str,
        contracts: tuple[StageContract, ...],
    ) -> Path:
        if revision < 1:
            raise ValueError("plan revision must be positive")
        relative = Path("plans") / f"revision-{revision:04d}" / "plan.json"
        value = {
            "run_id": run_id,
            "revision": revision,
            "plan_hash": plan_hash,
            "stages": [contract.to_dict() for contract in contracts],
        }
        content = json.dumps(
            self.redactor.value(value), ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n"
        path = self._resolve(run_id, relative)
        with self._lock:
            if path.exists():
                if path.read_text(encoding="utf-8") != content:
                    raise RuntimeError("immutable plan revision already exists with different data")
                return path
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(path)
        return path

    def save_handoff(self, handoff: AgentHandoff, *, suffix: str = "") -> Path:
        if suffix:
            self._validate_id(suffix, "handoff suffix")
        qualifier = f"-{suffix}" if suffix else ""
        filename = (
            f"handoff-{handoff.from_role.value}-to-{handoff.to_role.value}"
            f"{qualifier}.json"
        )
        return self.write_json(
            handoff.contract.run_id,
            Path("stages") / handoff.contract.stage_id / filename,
            handoff.to_dict(),
        )

    def append_role_log(
        self, run_id: str, stage_id: str, role_id: str, message: str
    ) -> Path:
        self._validate_id(stage_id, "stage_id")
        self._validate_id(role_id, "role_id")
        path = self._resolve(
            run_id, Path("stages") / stage_id / f"{role_id}.log"
        )
        safe_message = self.redactor.text(message.rstrip()) + "\n"
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", newline="") as handle:
                handle.write(safe_message)
                handle.flush()
                os.fsync(handle.fileno())
        return path

    def write_json(self, run_id: str, relative: Path, value: Any) -> Path:
        safe_value = self.redactor.value(value)
        return self.write_text(
            run_id,
            relative,
            json.dumps(safe_value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            redact=False,
        )

    def write_text(
        self, run_id: str, relative: Path, value: str, *, redact: bool = True
    ) -> Path:
        path = self._resolve(run_id, relative)
        content = self.redactor.text(value) if redact else value
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(path)
        return path

    def path_for(self, run_id: str, relative: Path) -> Path:
        """Return a validated artifact path for a tool that writes its own output."""
        path = self._resolve(run_id, relative)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _resolve(self, run_id: str, relative: Path) -> Path:
        self._validate_id(run_id, "run_id")
        if relative.is_absolute():
            raise ValueError("artifact path must be relative")
        run_root = (self.root / run_id).resolve()
        path = (run_root / relative).resolve()
        try:
            path.relative_to(run_root)
        except ValueError as exc:
            raise ValueError("artifact path escapes run directory") from exc
        return path

    @staticmethod
    def _validate_id(value: str, field_name: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", value):
            raise ValueError(f"invalid {field_name}: {value}")
