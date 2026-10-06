from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import replace

from app.config import FoundationConfig
from app.contracts import RoleId
from app.gateway.core.models import AgentReply, IncomingMessage
from app.services.context import ContextBundle
from app.services.budget import conservative_prompt_tokens
from app.services.attachments import accepted_image_path
from app.services.hermes import HermesRunner

from .parsing import InvalidAgentResponse, parse_team_conversation_reply
from .prompts import team_conversation_prompt


class HermesTeamConversationBackend:
    """범용 도구 없이 게이트웨이 중개 조회만 쓰는 Hermes 자유 대화를 실행한다."""

    def __init__(self, foundation: FoundationConfig, runner: HermesRunner):
        self.foundation = foundation
        self.runner = runner
        self.runtime_directory = foundation.root / "data" / "conversation-runtime"
        self.supports_output_token_limit = runner.settings.provider != "openai-codex"
        self.supports_cancellation = True

    def respond_as(
        self,
        state,
        context: ContextBundle,
        message: IncomingMessage,
        role_id: RoleId,
        *,
        caller_role: RoleId | None = None,
        call_purpose: str = "",
        turn_messages: tuple[dict, ...] = (),
        call_index: int = 1,
        max_output_tokens: int | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> AgentReply:
        settings = self.runner.settings.conversation
        self.runtime_directory.mkdir(parents=True, exist_ok=True)
        prompt = self._prompt(
            context,
            message,
            role_id,
            caller_role=caller_role,
            call_purpose=call_purpose,
            turn_messages=turn_messages,
        )
        message_key = hashlib.sha256(
            message.external_message_id.encode("utf-8")
        ).hexdigest()[:12]
        result = self.runner.run(
            state.run_id,
            f"chat-{message_key}-{call_index:02d}",
            role_id,
            self.runtime_directory,
            prompt,
            allow_writes=False,
            model=settings.model,
            reasoning=settings.reasoning,
            no_tools=True,
            max_turns=1,
            max_output_tokens=max_output_tokens,
            image_path=accepted_image_path(
                message, self.foundation.root / "data" / "attachments"
            ),
            ignore_rules=True,
            cancelled=cancelled,
        )
        try:
            reply = parse_team_conversation_reply(result.text, role_id)
        except InvalidAgentResponse as exc:
            raise InvalidAgentResponse(str(exc), usage=result.usage) from exc
        return replace(
            reply,
            usage=result.usage,
            metadata={
                "model": settings.model,
                "reasoning": settings.reasoning,
                "elapsed_seconds": round(result.elapsed_seconds, 3),
                "toolsets": "none",
            },
        )

    def prompt_token_upper_bound(
        self,
        state,
        context: ContextBundle,
        message: IncomingMessage,
        role_id: RoleId,
        *,
        caller_role: RoleId | None = None,
        call_purpose: str = "",
        turn_messages: tuple[dict, ...] = (),
        call_index: int = 1,
    ) -> int:
        """Count the complete local prompt before the governed backend reserves it."""
        return conservative_prompt_tokens(
            self._prompt(
                context,
                message,
                role_id,
                caller_role=caller_role,
                call_purpose=call_purpose,
                turn_messages=turn_messages,
            )
        )

    def _prompt(
        self,
        context: ContextBundle,
        message: IncomingMessage,
        role_id: RoleId,
        *,
        caller_role: RoleId | None,
        call_purpose: str,
        turn_messages: tuple[dict, ...],
    ) -> str:
        role = self.foundation.roles[role_id]
        instructions = role.conversation_instructions.read_text(encoding="utf-8")
        return team_conversation_prompt(
            instructions,
            role.display_name.strip() or role_id.value,
            role_id,
            context,
            message.text,
            caller_role=caller_role,
            call_purpose=call_purpose,
            turn_messages=turn_messages,
            user_intent=message.metadata.get("user_intent"),
        )

    def invocation_token_estimate(self, input_tokens: int, output_tokens: int) -> int:
        return input_tokens + output_tokens  # no_tools=True, max_turns=1

    def call_input_fingerprint(self, state, context, message, role_id, **kwargs) -> str:
        kwargs.pop("call_index", None)
        payload = f"{self.runner.settings!r}\n{self._prompt(context, message, role_id, **kwargs)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
