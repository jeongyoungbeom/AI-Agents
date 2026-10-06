from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from enum import Enum

from app.services.message_intent import (
    is_full_repository_audit_request,
    is_long_repository_analysis_request,
)


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
