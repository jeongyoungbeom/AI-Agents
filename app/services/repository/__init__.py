"""승인된 Git 스냅샷을 안전하게 조회하는 읽기 전용 서비스."""

from .reader import (
    RepositoryAccessError,
    RepositoryCancelled,
    RepositoryIdentity,
    RepositoryIdentityChanged,
    RepositoryPinnedSnapshotUnavailable,
    RepositoryReadContext,
    RepositorySnapshotEntry,
    RepositorySnapshotManifest,
    RepositoryTimeout,
    SafeRepositoryReader,
    inspect_repository_identity,
)
from .tools import (
    RepositorySnapshotChanged,
    RepositoryToolBatch,
    RepositoryToolError,
    RepositoryToolRequest,
    RepositoryToolResult,
    SafeRepositoryToolLayer,
)
from .analysis import (
    RepositoryAnalysisPhase,
    RepositoryAnalysisRequest,
    RepositoryAnalysisStatus,
    RepositoryAnalysisStopReason,
    is_full_repository_audit_request,
    is_long_repository_analysis_request,
)
from .analysis_plan import build_repository_analysis_plan, classify_analysis_path

__all__ = [
    "RepositoryAccessError",
    "RepositoryCancelled",
    "RepositoryIdentity",
    "RepositoryIdentityChanged",
    "RepositoryPinnedSnapshotUnavailable",
    "RepositoryReadContext",
    "RepositorySnapshotEntry",
    "RepositorySnapshotManifest",
    "RepositorySnapshotChanged",
    "RepositoryToolBatch",
    "RepositoryToolError",
    "RepositoryToolRequest",
    "RepositoryToolResult",
    "RepositoryTimeout",
    "SafeRepositoryReader",
    "SafeRepositoryToolLayer",
    "RepositoryAnalysisPhase",
    "RepositoryAnalysisRequest",
    "RepositoryAnalysisStatus",
    "RepositoryAnalysisStopReason",
    "inspect_repository_identity",
    "is_long_repository_analysis_request",
    "is_full_repository_audit_request",
    "build_repository_analysis_plan",
    "classify_analysis_path",
]
