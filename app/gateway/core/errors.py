from __future__ import annotations

from app.contracts import TokenUsage


class InvalidAgentResponse(ValueError):
    def __init__(self, message: str, *, usage: TokenUsage | None = None):
        super().__init__(message)
        self.usage = usage or TokenUsage()
