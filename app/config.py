from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.contracts import RoleId


@dataclass(frozen=True)
class RoleConfig:
    role_id: RoleId
    display_name: str
    model: str
    instructions: Path
    conversation_instructions: Path


@dataclass(frozen=True)
class FoundationConfig:
    root: Path
    database: Path
    artifacts: Path
    approval_phrase: str
    repository_approval_phrase: str
    roles: dict[RoleId, RoleConfig]
    repository_approval_ttl_hours: int = 720
    pending_project_request_ttl_hours: int = 24

    @classmethod
    def load(cls, root: Path) -> "FoundationConfig":
        root = root.resolve()
        app_data = _read_json(root / "config" / "app.json")
        role_data = _read_json(root / "config" / "roles.json")
        storage = dict(app_data.get("storage", {}))
        approval = dict(app_data.get("approval", {}))
        phrase = str(approval.get("required_phrase", "")).strip()
        if not phrase:
            raise ValueError("approval.required_phrase cannot be blank")
        repository_phrase = str(
            approval.get("repository_required_phrase", "이 프로젝트 사용 승인해")
        ).strip()
        if not repository_phrase:
            raise ValueError("approval.repository_required_phrase cannot be blank")
        repository_ttl = int(approval.get("repository_approval_ttl_hours", 720))
        if not 1 <= repository_ttl <= 8760:
            raise ValueError(
                "approval.repository_approval_ttl_hours must be between 1 and 8760"
            )
        pending_request_ttl = int(
            approval.get("pending_project_request_ttl_hours", 24)
        )
        if not 1 <= pending_request_ttl <= 168:
            raise ValueError(
                "approval.pending_project_request_ttl_hours must be between 1 and 168"
            )

        roles: dict[RoleId, RoleConfig] = {}
        for raw in role_data.get("roles", []):
            role_id = RoleId(str(raw["role_id"]))
            instructions = (root / str(raw["instructions"])).resolve()
            conversation_instructions = (
                root / str(raw["conversation_instructions"])
            ).resolve()
            for path, field_name in (
                (instructions, "instructions"),
                (conversation_instructions, "conversation_instructions"),
            ):
                try:
                    path.relative_to(root)
                except ValueError as exc:
                    raise ValueError(
                        f"{field_name} escape root for {role_id.value}"
                    ) from exc
                if not path.is_file():
                    raise ValueError(
                        f"{field_name} file not found for {role_id.value}: {path}"
                    )
            roles[role_id] = RoleConfig(
                role_id=role_id,
                display_name=str(raw.get("display_name", "")),
                model=str(raw.get("model", "")),
                instructions=instructions,
                conversation_instructions=conversation_instructions,
            )
        if set(roles) != set(RoleId):
            raise ValueError("roles.json must define development, review, and improvement")

        database = _under_root(root, str(storage.get("database", "data/state.db")))
        artifacts = _under_root(root, str(storage.get("artifacts", "artifacts")))
        return cls(
            root=root,
            database=database,
            artifacts=artifacts,
            approval_phrase=phrase,
            repository_approval_phrase=repository_phrase,
            roles=roles,
            repository_approval_ttl_hours=repository_ttl,
            pending_project_request_ttl_hours=pending_request_ttl,
        )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON configuration: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be an object: {path}")
    return value


def _under_root(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"configured path escapes root: {relative}") from exc
    return candidate
