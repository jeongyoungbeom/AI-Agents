from __future__ import annotations

from dataclasses import dataclass, field

from .models import IncomingMessage


@dataclass(frozen=True)
class AccessPolicy:
    """기본값은 모두 거부하며, 개인 채팅도 사용자 허용 목록이 필요하다."""

    allowed_users: frozenset[str] = field(default_factory=frozenset)
    allowed_group_conversations: frozenset[str] = field(default_factory=frozenset)
    private_only: bool = True

    def allows(self, message: IncomingMessage) -> bool:
        if message.user_id not in self.allowed_users:
            return False
        if message.is_private:
            return True
        if self.private_only:
            return False
        return message.conversation_id in self.allowed_group_conversations
