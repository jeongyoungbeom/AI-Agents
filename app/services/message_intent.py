from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from app.contracts import RoleId


class TaskIntent(str, Enum):
    ANSWER = "answer"
    PLAN_DEVELOPMENT = "plan_development"
    ANALYZE_REPOSITORY = "analyze_repository"


@dataclass(frozen=True)
class UserIntent:
    """현재 사용자 원문에서 만든 요청 계약. 모델의 해석은 승인이 아니다."""

    addressed_roles: tuple[RoleId, ...]
    allowed_delegate_roles: tuple[RoleId, ...] = ()
    task_intent: TaskIntent = TaskIntent.ANSWER
    write_forbidden: bool = False
    explicit: bool = False
    group_call: bool = False
    question_purpose: str = ""
    read_forbidden: bool = False

    @property
    def roles(self) -> tuple[RoleId, ...]:
        return self.addressed_roles

    def to_dict(self) -> dict:
        return {
            "addressed_roles": [role.value for role in self.addressed_roles],
            "allowed_delegate_roles": [role.value for role in self.allowed_delegate_roles],
            "task_intent": self.task_intent.value,
            "write_forbidden": self.write_forbidden,
            "explicit": self.explicit,
            "group_call": self.group_call,
            "question_purpose": self.question_purpose,
            "read_forbidden": self.read_forbidden,
        }


_BROAD_SCOPE = re.compile(
    r"전체\s*(?:코드|소스|저장소|레포|프로젝트|모듈|구조|테스트|감사)|"
    r"(?:프로젝트|저장소|레포)\s*전반|모든\s*(?:코드|소스|파일|모듈|테스트)|"
    r"(?:아키텍처|구조)\s*전체|(?:complete|full)\s+(?:repository|repo|codebase|audit|analysis|감사)|"
    r"(?:repository|repo|codebase)[\s-]*(?:wide|audit)", re.I,
)
_READ_ACTION = re.compile(r"(?:분석|검토|확인|읽|살펴|조사|파악|점검|감사)")
_PURPOSE_READ = re.compile(r"읽고")
_NEGATED_WRITE = re.compile(
    r"(?:수정|변경|개발|구현|작성|추가|삭제)(?:은|는|도)?\s*"
    r"(?:하지\s*(?:마|말|않)|안\s*해|금지)|(?:계획|설명|답변)만"
)
_NEGATED_READ = re.compile(
    r"(?:분석|검토|확인|읽기|조사|감사)(?:는|를|은|을)?\s*"
    r"(?:하지\s*(?:마|말|않)|안\s*해|금지)|읽지\s*(?:마|말|않)"
)
_READ_EXPLANATION = re.compile(
    r"(?:계획|방법|이유|전략|준비).*?(?:설명|알려)|"
    r"(?:제안|시작|실행)하면.*?(?:어떤|무엇|뭐|어떻게|생겨|생기)"
)
_QUESTION = re.compile(
    r"^(?:왜|무엇|뭐|어떻게|언제|어떤)\b|"
    r"(?:해도\s*되는지|할지\s*판단|라고\s*말하면|라는\s*말|이란|란\s*뭐)|"
    r"(?:왜|무엇|뭐|어떻게|언제|어떤).*?(?:말해|설명|알려|궁금)|"
    r"(?:들어가는지|되는지|하는지).*?(?:말해|설명|알려)"
)


def forbids_writes(text: str) -> bool:
    return bool(_NEGATED_WRITE.search(text))


def forbids_repository_reads(text: str) -> bool:
    return bool(_NEGATED_READ.search(text) or (
        (re.search(r"(?:계획|설명|답변)만", text) or _READ_EXPLANATION.search(text))
        and not _PURPOSE_READ.search(text)
    ))


def is_work_request(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip().casefold())
    if forbids_writes(normalized) or _QUESTION.search(normalized):
        return False
    if re.search(r"(?:계획|방법|이유|전략).*(?:작성|설명|알려)", normalized):
        return False
    if normalized.endswith("개발") or "작업으로 진행" in normalized:
        return True
    return bool(re.search(
        r"(?:개발|구현|수정|고쳐|추가|만들|설계|작성|잡아|바꿔|적용)"
        r"\s*(?:해|해주세요|해\s*줘|하자|하고\s*싶어|하고\s*싶어요|해\s*보고\s*싶어)",
        normalized,
    ) or re.search(r"(?:고쳐|만들어|잡아|바꿔)\s*(?:줘|주세요|주라|보자)?$", normalized))


def is_long_repository_analysis_request(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip()).casefold()
    if not _BROAD_SCOPE.search(normalized) or is_work_request(normalized):
        return False
    # Explicit reading before a purpose question is still a reading request.
    question = _QUESTION.search(normalized)
    purpose_read = _PURPOSE_READ.search(normalized)
    if question and (purpose_read is None or purpose_read.start() >= question.start()):
        return False
    if forbids_repository_reads(normalized):
        return False
    return bool(_READ_ACTION.search(normalized))


def is_full_repository_audit_request(text: str) -> bool:
    return is_long_repository_analysis_request(text) and bool(re.search(
        r"(?:전체\s*(?:코드|소스|파일|감사)|모든\s*(?:코드|소스|파일)|full\s+(?:codebase|repository|repo|audit|감사))",
        text, re.I,
    ))


def explicit_task_intent(text: str) -> TaskIntent:
    if is_work_request(text):
        return TaskIntent.PLAN_DEVELOPMENT
    if is_long_repository_analysis_request(text):
        return TaskIntent.ANALYZE_REPOSITORY
    return TaskIntent.ANSWER
