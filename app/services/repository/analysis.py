from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from enum import Enum


class RepositoryAnalysisStatus(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    STOP_REQUESTED = "STOP_REQUESTED"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    PARTIAL_COMPLETED = "PARTIAL_COMPLETED"
    NEEDS_ATTENTION = "NEEDS_ATTENTION"
    SUPERSEDED = "SUPERSEDED"
    CANCELLED = "CANCELLED"

    @property
    def active(self) -> bool:
        return self in {
            RepositoryAnalysisStatus.QUEUED,
            RepositoryAnalysisStatus.PROCESSING,
            RepositoryAnalysisStatus.STOP_REQUESTED,
            RepositoryAnalysisStatus.PAUSED,
        }


class RepositoryAnalysisPhase(str, Enum):
    STRUCTURE = "STRUCTURE"
    TESTS = "TESTS"
    CORE = "CORE"
    RISKS = "RISKS"
    SYNTHESIS = "SYNTHESIS"


class RepositoryAnalysisStopReason(str, Enum):
    COMPLETED = "COMPLETED"
    BUDGET_LIMIT = "BUDGET_LIMIT"
    TIME_LIMIT = "TIME_LIMIT"
    QUERY_LIMIT = "QUERY_LIMIT"
    BYTE_LIMIT = "BYTE_LIMIT"
    NO_PROGRESS = "NO_PROGRESS"
    BATCH_LIMIT = "BATCH_LIMIT"
    CONTEXT_LIMIT = "CONTEXT_LIMIT"
    FILE_SIZE_LIMIT = "FILE_SIZE_LIMIT"
    BATCH_SIZE_LIMIT = "BATCH_SIZE_LIMIT"
    SELECTION_LIMIT = "SELECTION_LIMIT"
    CATEGORY_NOT_SELECTED = "CATEGORY_NOT_SELECTED"
    USER_STOPPED = "USER_STOPPED"
    SNAPSHOT_UNAVAILABLE = "SNAPSHOT_UNAVAILABLE"
    REPOSITORY_IDENTITY_CHANGED = "REPOSITORY_IDENTITY_CHANGED"
    MODEL_OUTCOME_UNKNOWN = "MODEL_OUTCOME_UNKNOWN"
    SUPERSEDED = "SUPERSEDED"


_LONG_ANALYSIS_PATTERNS = (
    r"전체\s*(?:코드|소스|저장소|레포|프로젝트|모듈|구조|테스트)",
    r"(?:프로젝트|저장소|레포)\s*전반",
    r"모든\s*(?:코드|소스|파일|모듈|테스트)",
    r"전체\s*테스트\s*전략",
    r"(?:아키텍처|구조)\s*전체",
    r"(?:complete|full)\s+(?:repository|repo|codebase|audit|analysis)",
    r"(?:repository|repo|codebase)[\s-]*(?:wide|audit)",
)
_LONG_ANALYSIS_REQUEST = re.compile("|".join(_LONG_ANALYSIS_PATTERNS), re.IGNORECASE)


def is_long_repository_analysis_request(text: str) -> bool:
    """Return true only for broad repository requests, not ordinary code questions."""
    normalized = re.sub(r"\s+", " ", text.strip()).casefold()
    if not normalized or not _LONG_ANALYSIS_REQUEST.search(normalized):
        return False
    # Mentioning an analysis in a question is not a request to run one.
    if re.search(r"(?:왜|무엇|뭐|어떻게|언제|어떤).*?(?:말해|설명|알려|궁금)|(?:이란|란|라는|인지|하는지|되는지)\b", normalized):
        return False
    if re.search(r"(?:수정|고쳐|개발|구현|추가|작성|만들|삭제|바꿔|적용)\s*(?:해|해줘|해주세요|하자|하고\s*싶)", normalized):
        return False
    if re.search(r"(?:하지\s*마|말아|하지\s*않|안\s*해)", normalized):
        return False
    return bool(re.search(r"(?:분석|검토|확인|읽어|살펴|조사|파악|점검|전략|감사)(?:해|해줘|해주세요|하자|해\s*줘|해\s*주세요|해봐|해\s*봐)?", normalized))


def is_full_repository_audit_request(text: str) -> bool:
    """A whole-code request promises every eligible source, not an adaptive sample."""
    return is_long_repository_analysis_request(text) and bool(re.search(
        r"(?:전체\s*(?:코드|소스|파일)|모든\s*(?:코드|소스|파일)|full\s+(?:codebase|repository|repo|audit))",
        text, re.IGNORECASE,
    ))


@dataclass(frozen=True)
class RepositoryAnalysisRequest:
    analysis_id: str
    channel: str
    conversation_id: str
    user_id: str
    source_message_id: str
    role_id: str
    request_text: str
    repository_path: str
    repository_identity: str
    commit_sha: str
    branch: str

    @classmethod
    def create(
        cls,
        *,
        channel: str,
        conversation_id: str,
        user_id: str,
        source_message_id: str,
        role_id: str,
        request_text: str,
        repository_path: str,
        repository_identity: str,
        commit_sha: str,
        branch: str,
    ) -> "RepositoryAnalysisRequest":
        return cls(
            analysis_id=uuid.uuid4().hex,
            channel=channel,
            conversation_id=conversation_id,
            user_id=user_id,
            source_message_id=source_message_id,
            role_id=role_id,
            request_text=request_text.strip(),
            repository_path=repository_path.strip(),
            repository_identity=repository_identity.strip(),
            commit_sha=commit_sha.strip(),
            branch=branch.strip(),
        )

    def __post_init__(self) -> None:
        required = {
            "analysis_id": self.analysis_id,
            "channel": self.channel,
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "source_message_id": self.source_message_id,
            "role_id": self.role_id,
            "request_text": self.request_text,
            "repository_path": self.repository_path,
            "repository_identity": self.repository_identity,
            "commit_sha": self.commit_sha,
        }
        if any(not value.strip() for value in required.values()):
            raise ValueError("repository analysis request requires complete identity and scope")
        if not re.fullmatch(r"[0-9a-f]{40,64}", self.repository_identity):
            raise ValueError("repository analysis identity must be a Git object hash")
        if not re.fullmatch(r"[0-9a-f]{40,64}", self.commit_sha):
            raise ValueError("repository analysis commit must be a Git object hash")
