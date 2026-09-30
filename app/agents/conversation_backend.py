from __future__ import annotations

from pathlib import Path
from collections.abc import Callable

from app.config import FoundationConfig
from app.contracts import RoleId
from app.gateway.core.models import AgentReply, IncomingMessage
from app.services.context import ContextBundle
from app.services.budget import conservative_prompt_tokens
from app.services.attachments import accepted_image_path
from app.services.git import GitRepository, GitRepositoryCancelled
from app.services.hermes import HermesCancelled, HermesRunner
from app.services.sandbox import DockerSandbox

from .parsing import InvalidAgentResponse, parse_conversation_reply
from .prompts import planning_prompt


class HermesConversationBackend:
    """승인 전 개발 담당과의 설계 대화를 Hermes에 연결한다."""

    def __init__(
        self,
        foundation: FoundationConfig,
        runner: HermesRunner,
        *,
        sandbox: DockerSandbox | None = None,
    ):
        self.foundation = foundation
        self.runner = runner
        self.sandbox = sandbox or DockerSandbox()
        self.supports_output_token_limit = True
        self.supports_cancellation = True

    def respond(
        self,
        state,
        context: ContextBundle,
        message: IncomingMessage,
        *,
        max_output_tokens: int | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> AgentReply:
        if not state.repository:
            return AgentReply("먼저 사용할 Git 프로젝트를 선택해 주세요.")
        repository = GitRepository(
            Path(state.repository),
            self.sandbox,
            operation_id=state.run_id,
            component="conversation-planning-git",
            cancelled=cancelled,
        )
        try:
            baseline = repository.snapshot()
            planning = self.runner.settings.planning
            role = self.foundation.roles[RoleId.DEVELOPMENT]
            try:
                result = self.runner.run(
                    state.run_id,
                    f"planning-r{state.plan_revision + 1:03d}",
                    RoleId.DEVELOPMENT,
                    repository.path,
                    self._prompt(state, context),
                    allow_writes=False,
                    model=planning.model,
                    reasoning=planning.reasoning,
                    max_output_tokens=max_output_tokens,
                    image_path=accepted_image_path(
                        message, self.foundation.root / "data" / "attachments"
                    ),
                    cancelled=cancelled,
                )
            finally:
                repository.assert_position(baseline, allow_worktree_changes=False)
        except GitRepositoryCancelled as exc:
            # A Git snapshot/check made while a queued conversation is being
            # cancelled must not turn the job into NEEDS_ATTENTION.
            raise HermesCancelled("사용자가 Git 작업을 중지했습니다.") from exc
        try:
            reply = parse_conversation_reply(result.text)
        except InvalidAgentResponse:
            name = role.display_name.strip() or "설계 담당"
            return AgentReply(
                f"{name} 응답 형식이 올바르지 않아 계획을 확정하지 않았습니다. "
                "같은 요청을 한 번 더 말해 주세요.",
                usage=result.usage,
            )
        return AgentReply(
            text=reply.text,
            stages=reply.stages,
            decisions=reply.decisions,
            usage=result.usage,
        )

    def prompt_token_upper_bound(self, state, context: ContextBundle, message: IncomingMessage) -> int:
        """Count the complete local prompt before the governed backend reserves it."""
        return conservative_prompt_tokens(self._prompt(state, context))

    def _prompt(self, state, context: ContextBundle) -> str:
        role = self.foundation.roles[RoleId.DEVELOPMENT]
        instructions = role.instructions.read_text(encoding="utf-8")
        if role.display_name.strip():
            instructions = f"당신의 표시 이름은 '{role.display_name.strip()}'이다.\n\n{instructions}"
        return planning_prompt(instructions, state.objective, context)
