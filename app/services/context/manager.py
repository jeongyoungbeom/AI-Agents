from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.services.logging import SecretRedactor
from app.storage import StateStore


@dataclass(frozen=True)
class ContextPolicy:
    recent_messages: int = 12
    recent_decisions: int = 4
    max_characters: int = 24_000
    decision_summary_characters: int = 6_000

    def __post_init__(self) -> None:
        if self.recent_messages < 1 or self.recent_decisions < 1:
            raise ValueError("context recent limits must be positive")
        if self.max_characters < 100:
            raise ValueError("context max_characters must be at least 100")
        if not 100 <= self.decision_summary_characters <= self.max_characters:
            raise ValueError("context decision summary limit is invalid")

    @classmethod
    def load(cls, path: Path) -> "ContextPolicy":
        raw = json.loads(path.read_text(encoding="utf-8"))
        values = dict(raw.get("context", {}))
        return cls(
            recent_messages=int(values.get("recent_messages", 12)),
            recent_decisions=int(values.get("recent_decisions", 4)),
            max_characters=int(values.get("max_characters", 24_000)),
            decision_summary_characters=int(
                values.get("decision_summary_characters", 6_000)
            ),
        )


@dataclass(frozen=True)
class ContextBundle:
    decisions: tuple[dict[str, Any], ...]
    memories: tuple[dict[str, Any], ...]
    recent_messages: tuple[dict[str, Any], ...]
    truncated: bool
    characters: int
    memory_revision: int = 0
    repository_context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisions": list(self.decisions),
            "memories": list(self.memories),
            "recent_messages": list(self.recent_messages),
            "truncated": self.truncated,
            "characters": self.characters,
            "memory_revision": self.memory_revision,
            "repository_context": self.repository_context or {},
        }


class ContextService:
    """Stores sanitized chat and builds small, role-neutral context bundles."""

    def __init__(
        self,
        store: StateStore,
        redactor: SecretRedactor | None = None,
        *,
        policy: ContextPolicy | None = None,
    ):
        self.store = store
        self.redactor = redactor or SecretRedactor()
        self.policy = policy or ContextPolicy()

    def add_message(
        self,
        run_id: str,
        sender: str,
        content: str,
        *,
        kind: str = "message",
        data: dict[str, Any] | None = None,
    ) -> None:
        if not sender.strip() or not content.strip():
            raise ValueError("sender and content are required")
        self.store.append_message(
            run_id,
            sender.strip(),
            kind,
            self.redactor.text(content.strip()),
            self.redactor.value(data or {}),
        )

    def add_decision(
        self,
        run_id: str,
        content: str,
        *,
        source: str = "user",
        data: dict[str, Any] | None = None,
    ) -> None:
        sanitized = self.redactor.text(content.strip())
        self.add_message(run_id, source, sanitized, kind="decision", data=data)
        self.store.append_compacted_memory(
            "run",
            run_id,
            "compaction",
            sanitized,
            max_characters=self.policy.decision_summary_characters,
        )

    def save_memory(
        self,
        scope: str,
        scope_key: str,
        content: str,
        *,
        role_id: str = "shared",
        source_kind: str = "agent",
        source_ref: str = "",
    ) -> int:
        if not scope.strip() or not scope_key.strip() or not role_id.strip():
            raise ValueError("memory scope, key, and role are required")
        sanitized = self.redactor.text(content.strip())
        if not sanitized:
            raise ValueError("memory content is required")
        return self.store.save_memory(
            scope.strip(), scope_key.strip(), role_id.strip(), sanitized,
            source_kind=source_kind, source_ref=source_ref,
        )

    def user_memories(self, user_id: str) -> list[dict[str, Any]]:
        return self.store.list_memories((("user", user_id.strip()),))

    def delete_user_memories(self, user_id: str) -> int:
        return self.store.delete_memory("user", user_id.strip())

    def project_memories(
        self, repository_identity: str, repository_path: str = ""
    ) -> list[dict[str, Any]]:
        keys = tuple(("project", key) for key in sorted({
            repository_identity.strip(), repository_path.strip()
        }) if key)
        return self.store.list_memories(keys)

    def delete_project_memories(
        self, repository_identity: str, repository_path: str = ""
    ) -> int:
        keys = {repository_identity.strip(), repository_path.strip()}
        return sum(self.store.delete_memory("project", key) for key in keys if key)

    def replace_memory(self, fact_id: int, scope: str, scope_key: str, content: str) -> int:
        sanitized = self.redactor.text(content.strip())
        if not sanitized:
            raise ValueError("memory content is required")
        return self.store.replace_memory_fact(
            fact_id, scope, scope_key, sanitized, source_kind="user"
        )

    def build(
        self,
        run_id: str,
        *,
        recent_limit: int | None = None,
        max_characters: int | None = None,
        conversation_key: str = "",
        user_id: str = "",
        role_id: str = "",
        repository: str = "",
        repository_identity: str = "",
        repository_context: dict[str, Any] | None = None,
        source_messages: Sequence[dict[str, Any]] | None = None,
        exclude_untrusted_repository_messages: bool = False,
    ) -> ContextBundle:
        recent_limit = (
            self.policy.recent_messages if recent_limit is None else recent_limit
        )
        max_characters = (
            self.policy.max_characters if max_characters is None else max_characters
        )
        if recent_limit < 1 or max_characters < 100:
            raise ValueError("context limits are too small")
        messages = (
            list(source_messages)
            if source_messages is not None
            else self.store.list_messages(run_id)
        )
        if exclude_untrusted_repository_messages:
            messages = [
                message
                for message in messages
                if not message.get("data", {}).get("untrusted_repository_data", False)
            ]
        scoped_identity = repository_identity.strip()
        excluded_for_repository = 0
        if scoped_identity:
            before_scope_filter = len(messages)
            messages = [
                message
                for message in messages
                if str(message.get("data", {}).get("repository_identity", "")).strip()
                == scoped_identity
            ]
            excluded_for_repository = before_scope_filter - len(messages)
        all_decisions = [message for message in messages if message["kind"] == "decision"]
        ordinary = [message for message in messages if message["kind"] != "decision"]
        candidates = ordinary[-recent_limit:]
        memory_keys = [] if scoped_identity else [("run", run_id)]
        if conversation_key.strip():
            memory_keys.append(("conversation", conversation_key.strip()))
        if user_id.strip():
            memory_keys.append(("user", user_id.strip()))
        if repository.strip():
            memory_keys.append(("project", repository.strip()))
        all_memories = self.store.list_memories(
            tuple(memory_keys), role_id=role_id.strip()
        )
        decisions: list[dict[str, Any]] = []
        memories: list[dict[str, Any]] = []
        selected: list[dict[str, Any]] = []
        sanitized_repository_context = self.redactor.value(repository_context or {})
        repository_size = (
            len(
                json.dumps(
                    sanitized_repository_context,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            if sanitized_repository_context
            else 0
        )
        if repository_size > max_characters:
            sanitized_repository_context = {
                "status": "omitted_context_limit",
                "truncated": True,
            }
            repository_size = len(str(sanitized_repository_context))
        used = repository_size
        truncated = excluded_for_repository > 0 or len(candidates) < len(ordinary)

        # Decisions have priority, but even they must obey the hard context cap.
        for message in reversed(all_decisions[-self.policy.recent_decisions :]):
            size = len(message["content"])
            if used + size > max_characters:
                truncated = True
                continue
            decisions.append(message)
            used += size
        decisions.reverse()

        for memory in all_memories:
            size = len(memory["content"])
            if used + size > max_characters:
                truncated = True
                continue
            memories.append(memory)
            used += size

        for message in reversed(candidates):
            size = len(message["content"])
            if used + size > max_characters:
                truncated = True
                continue
            selected.append(message)
            used += size
        selected.reverse()
        return ContextBundle(
            decisions=tuple(decisions),
            memories=tuple(memories),
            recent_messages=tuple(selected),
            truncated=truncated,
            characters=used,
            memory_revision=max(
                (int(memory["revision"]) for memory in memories), default=0
            ),
            repository_context=sanitized_repository_context,
        )
