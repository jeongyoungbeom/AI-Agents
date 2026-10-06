from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class RequestOutcome(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING_USER = "waiting_user"

    @property
    def label(self) -> str:
        return {
            self.SUCCESS: "완료", self.PARTIAL: "부분 완료", self.FAILED: "실패",
            self.CANCELLED: "취소됨", self.WAITING_USER: "사용자 대기",
        }[self]


@dataclass(frozen=True)
class RequestResult:
    outcome: RequestOutcome
    reason: str = ""
    attempts: int | None = 0
    successes: int | None = 0
    provider_requests: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, RequestOutcome):
            raise ValueError("invalid request outcome")
        if (self.attempts is None) != (self.successes is None):
            raise ValueError("request call counts must both be known or unknown")
        if self.attempts is not None and not 0 <= self.successes <= self.attempts:
            raise ValueError("invalid request call counts")
        if self.provider_requests is not None and self.provider_requests < 0:
            raise ValueError("invalid provider request count")

    def to_dict(self) -> dict:
        return {**asdict(self), "outcome": self.outcome.value}

    @classmethod
    def from_dict(cls, value: dict) -> RequestResult:
        return cls(
            RequestOutcome(value["outcome"]), str(value.get("reason", "")),
            value.get("attempts", 0), value.get("successes", 0),
            value.get("provider_requests"),
        )

    @property
    def calls_text(self) -> str:
        provider = "미확인" if self.provider_requests is None else str(self.provider_requests)
        counts = ("앱 호출 시도·성공 미확인" if self.attempts is None
                  else f"앱 호출 시도 {self.attempts}회 · 성공 {self.successes}회")
        return f"{counts} · provider 요청 {provider}"

    @classmethod
    def for_analysis(cls, status: str, reason: str = "", **counts) -> RequestResult:
        outcome = {
            "COMPLETED": RequestOutcome.SUCCESS,
            "PARTIAL_COMPLETED": RequestOutcome.PARTIAL,
            "CANCELLED": RequestOutcome.CANCELLED,
            "SUPERSEDED": RequestOutcome.CANCELLED,
            "PAUSED": RequestOutcome.CANCELLED,
            "NEEDS_ATTENTION": (
                RequestOutcome.WAITING_USER
                if reason in {"SNAPSHOT_UNAVAILABLE", "REPOSITORY_IDENTITY_CHANGED"}
                else RequestOutcome.FAILED
            ),
        }.get(status, RequestOutcome.PARTIAL)
        return cls(outcome, reason, **counts)
