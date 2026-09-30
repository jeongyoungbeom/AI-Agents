from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from app.contracts import RoleId, RunState
from app.services.context import ContextBundle

from .models import (
    AgentReply,
    IncomingMessage,
    OutgoingMessage,
    PollBatch,
    TeamConversationBatchResult,
    TeamConversationRequest,
)

if TYPE_CHECKING:
    from app.services.repository import RepositoryAnalysisRequest

if TYPE_CHECKING:
    from .repository import RepositoryInfo


class ChatAdapter(Protocol):
    """새 메신저를 추가할 때 구현하는 최소 인터페이스."""

    name: str

    def check(self, *, online: bool = True) -> str: ...

    def poll(self, cursor: str | None) -> PollBatch: ...

    def prepare(self, message: OutgoingMessage) -> tuple[OutgoingMessage, ...]: ...

    def send(self, message: OutgoingMessage) -> tuple[str, ...]: ...


class AgentConversationBackend(Protocol):
    """설계 대화를 담당하는 모델 백엔드 교체 지점."""

    def respond(
        self,
        state: RunState,
        context: ContextBundle,
        message: IncomingMessage,
        *,
        max_output_tokens: int | None = None,
    ) -> AgentReply: ...


class TeamConversationBackend(Protocol):
    """역할별 자유 대화와 상담 호출을 담당하는 교체 지점."""

    def preflight(
        self,
        state: RunState,
        context: ContextBundle,
        message: IncomingMessage,
        reply_count: int,
    ) -> None: ...

    def respond_as(
        self,
        state: RunState,
        context: ContextBundle,
        message: IncomingMessage,
        role_id: RoleId,
        *,
        caller_role: RoleId | None = None,
        call_purpose: str = "",
        turn_messages: tuple[dict, ...] = (),
        call_index: int = 1,
        max_output_tokens: int | None = None,
    ) -> AgentReply: ...


class BatchTeamConversationBackend(TeamConversationBackend, Protocol):
    """첫 그룹 응답을 안전하게 병렬 실행할 수 있는 선택적 확장 인터페이스."""

    def respond_batch(
        self,
        state: RunState,
        message: IncomingMessage,
        requests: tuple[TeamConversationRequest, ...],
        *,
        max_workers: int,
        max_model_calls: int,
    ) -> TeamConversationBatchResult: ...


class ConversationScheduler(Protocol):
    """수신과 긴 AI 처리를 분리하는 영속 대화 큐."""

    def enqueue(self, message: IncomingMessage) -> dict: ...

    def cancel(self, channel: str, conversation_id: str) -> str | None: ...

    def status(self, channel: str, conversation_id: str) -> str | None: ...

    def summary(self, channel: str, conversation_id: str) -> dict | None: ...

    def resume(self, channel: str, conversation_id: str) -> dict | None: ...


class RepositoryAnalysisScheduler(Protocol):
    """재개 가능한 저장소 분석 전용 큐이다. 대화 역할 호출 한도와 분리한다."""

    def enqueue(self, request: "RepositoryAnalysisRequest") -> tuple[dict, bool]: ...

    def request_stop(self, channel: str, conversation_id: str) -> dict | None: ...

    def supersede(self, channel: str, conversation_id: str) -> dict | None: ...

    def resume(self, channel: str, conversation_id: str) -> dict | None: ...

    def summary(self, channel: str, conversation_id: str) -> dict | None: ...


class RepositoryValidator(Protocol):
    def validate(
        self,
        raw_path: str,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> "RepositoryInfo": ...


class PipelineScheduler(Protocol):
    def enqueue(self, run_id: str, channel: str, conversation_id: str) -> str: ...

    def cancel(self, run_id: str) -> str | None: ...

    def pause(self, run_id: str) -> str | None: ...

    def resume(self, run_id: str) -> str | None: ...

    def status(self, run_id: str) -> str | None: ...
