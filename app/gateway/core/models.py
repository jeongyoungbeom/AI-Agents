from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from app.contracts import RoleId, TokenUsage
from app.contracts.outcomes import RequestResult
from app.services.context import ContextBundle
from app.services.repository import RepositoryToolRequest
from app.services.message_intent import TaskIntent


def _required(value: str, field_name: str, maximum: int = 512) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} is required")
    if len(value) > maximum:
        raise ValueError(f"{field_name} is too long")
    return value


class ConversationMode(str, Enum):
    FREE_CHAT = "free_chat"
    PLANNING = "planning"
    EXECUTING = "executing"


class MemoryScope(str, Enum):
    CONVERSATION = "conversation"
    USER = "user"
    PROJECT = "project"
    RUN = "run"


@dataclass(frozen=True)
class Attachment:
    kind: str
    external_id: str
    name: str = ""
    size: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required(self.kind, "attachment.kind", 40)
        _required(self.external_id, "attachment.external_id")
        if self.size is not None and self.size < 0:
            raise ValueError("attachment size cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "external_id": self.external_id,
            "name": self.name,
            "size": self.size,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Attachment":
        return cls(
            kind=str(value["kind"]),
            external_id=str(value["external_id"]),
            name=str(value.get("name", "")),
            size=int(value["size"]) if value.get("size") is not None else None,
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class IncomingMessage:
    channel: str
    conversation_id: str
    user_id: str
    external_message_id: str
    text: str
    is_private: bool = True
    user_display_name: str = ""
    attachments: tuple[Attachment, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        channel = _required(self.channel, "channel", 40).lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", channel):
            raise ValueError("channel contains unsupported characters")
        object.__setattr__(self, "channel", channel)
        _required(self.conversation_id, "conversation_id")
        _required(self.user_id, "user_id")
        _required(self.external_message_id, "external_message_id")
        if not self.text.strip() and not self.attachments:
            raise ValueError("message requires text or an attachment")

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "external_message_id": self.external_message_id,
            "text": self.text,
            "is_private": self.is_private,
            "user_display_name": self.user_display_name,
            "attachments": [item.to_dict() for item in self.attachments],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "IncomingMessage":
        return cls(
            channel=str(value["channel"]),
            conversation_id=str(value["conversation_id"]),
            user_id=str(value["user_id"]),
            external_message_id=str(value["external_message_id"]),
            text=str(value.get("text", "")),
            is_private=bool(value.get("is_private", True)),
            user_display_name=str(value.get("user_display_name", "")),
            attachments=tuple(
                Attachment.from_dict(dict(item))
                for item in value.get("attachments", [])
            ),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class OutgoingMessage:
    channel: str
    conversation_id: str
    text: str
    reply_to: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required(self.channel, "channel", 40)
        _required(self.conversation_id, "conversation_id")
        if not self.text.strip():
            raise ValueError("outgoing text is required")


@dataclass(frozen=True)
class PollBatch:
    messages: tuple[IncomingMessage, ...] = ()
    cursor: str | None = None


@dataclass(frozen=True)
class ConversationResult:
    messages: tuple[OutgoingMessage, ...]
    result: RequestResult


@dataclass(frozen=True)
class ProposedStage:
    objective: str
    scope: tuple[str, ...]
    acceptance_criteria: tuple[str, ...]
    verification_commands: tuple[str, ...]
    non_goals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.objective.strip():
            raise ValueError("stage objective is required")
        if not self.scope or not self.acceptance_criteria or not self.verification_commands:
            raise ValueError("stage scope, acceptance criteria, and verification are required")


@dataclass(frozen=True)
class AgentCallRequest:
    """자유 대화에서 담당 권한을 넘기지 않고 다른 역할을 호출한다."""

    from_role: RoleId
    to_role: RoleId
    purpose: str
    mode: str = "independent"

    def __post_init__(self) -> None:
        if self.from_role == self.to_role:
            raise ValueError("an agent cannot call itself")
        _required(self.purpose, "agent_call.purpose", 2000)
        if self.mode not in {"independent", "discussion"}:
            raise ValueError("agent_call.mode must be independent or discussion")


@dataclass(frozen=True)
class MemoryUpdate:
    scope: MemoryScope
    content: str
    role_id: RoleId | None = None

    def __post_init__(self) -> None:
        _required(self.content, "memory_update.content", 12000)


@dataclass(frozen=True)
class RoleSelection:
    roles: tuple[RoleId, ...]
    explicit: bool = False
    group_call: bool = False

    def __post_init__(self) -> None:
        if not self.roles:
            raise ValueError("at least one conversation role is required")
        if len(set(self.roles)) != len(self.roles):
            raise ValueError("conversation roles cannot be duplicated")


@dataclass(frozen=True)
class AgentReply:
    text: str
    stages: tuple[ProposedStage, ...] = ()
    decisions: tuple[str, ...] = ()
    calls: tuple[AgentCallRequest, ...] = ()
    memory_updates: tuple[MemoryUpdate, ...] = ()
    usage: TokenUsage = field(default_factory=TokenUsage)
    metadata: dict[str, Any] = field(default_factory=dict)
    repository_tools: tuple[RepositoryToolRequest, ...] = ()
    task_intent: TaskIntent = TaskIntent.ANSWER
    write_forbidden: bool = False
    question_purpose: str = ""
    needs_user_input: tuple[str, ...] = ()
    execution_intent: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        internal_repository_lookup = (
            bool(self.repository_tools)
            and not self.calls
            and not self.memory_updates
        )
        if not self.text.strip() and not internal_repository_lookup:
            raise ValueError("agent reply text is required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AgentReply":
        return cls(
            text=value["text"],
            stages=tuple(ProposedStage(**{key: tuple(item) if isinstance(item, list) else item
                                         for key, item in stage.items()})
                         for stage in value.get("stages", [])),
            decisions=tuple(value.get("decisions", [])),
            calls=tuple(AgentCallRequest(RoleId(call["from_role"]), RoleId(call["to_role"]), call["purpose"], call.get("mode", "independent"))
                        for call in value.get("calls", [])),
            memory_updates=tuple(MemoryUpdate(MemoryScope(item["scope"]), item["content"],
                                             RoleId(item["role_id"]) if item["role_id"] else None)
                                 for item in value.get("memory_updates", [])),
            usage=TokenUsage(**value["usage"]), metadata=value.get("metadata", {}),
            repository_tools=tuple(RepositoryToolRequest.from_dict(item)
                                   for item in value.get("repository_tools", [])),
            task_intent=TaskIntent(value.get("task_intent", "answer")),
            write_forbidden=value.get("write_forbidden", False),
            question_purpose=value.get("question_purpose", ""),
            needs_user_input=tuple(value.get("needs_user_input", [])),
            execution_intent=value.get("execution_intent"),
        )


@dataclass(frozen=True)
class TeamConversationRequest:
    """한 역할의 모델 호출에 필요한 불변 입력 묶음."""

    role_id: RoleId
    context: ContextBundle
    caller_role: RoleId | None = None
    call_purpose: str = ""
    turn_messages: tuple[dict, ...] = ()
    call_index: int = 1
    reserve_final_answer: bool = False

    def __post_init__(self) -> None:
        if self.caller_role == self.role_id:
            raise ValueError("an agent cannot call itself")
        if self.call_index < 1:
            raise ValueError("call_index must be positive")


@dataclass(frozen=True)
class TeamConversationBatchResult:
    """역할별 결과와 실제로 실행한 모델 호출 수를 함께 반환한다."""

    outcomes: tuple[AgentReply | Exception, ...]
    model_calls: int

    def __post_init__(self) -> None:
        if self.model_calls < 0:
            raise ValueError("model_calls cannot be negative")
