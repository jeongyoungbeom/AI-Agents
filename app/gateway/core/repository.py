from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.services.repository import (
    RepositoryAccessError,
    RepositoryCancelled,
    inspect_repository_identity,
)
from app.services.sandbox import DockerSandbox


class RepositorySelectionError(ValueError):
    pass


@dataclass(frozen=True)
class RepositoryInfo:
    path: Path
    branch: str = ""
    identity_hash: str = ""
    head_sha: str = ""


class LocalGitRepositoryValidator:
    """프로젝트를 수정하지 않고 Git 작업 트리인지 확인한다."""

    def __init__(
        self,
        *,
        allow_network_paths: bool = False,
        sandbox: DockerSandbox | None = None,
    ):
        self.allow_network_paths = allow_network_paths
        self.sandbox = sandbox or DockerSandbox()

    def validate(
        self,
        raw_path: str,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> RepositoryInfo:
        cleaned = raw_path.strip().strip('"').strip("'")
        if cleaned.startswith("\\\\") and not self.allow_network_paths:
            raise RepositorySelectionError("네트워크 경로는 기본 정책에서 허용하지 않습니다.")
        candidate = Path(cleaned)
        if not candidate.is_absolute():
            raise RepositorySelectionError("프로젝트 경로는 절대경로여야 합니다.")
        if not candidate.is_dir():
            raise RepositorySelectionError("해당 프로젝트 폴더를 찾을 수 없습니다.")
        candidate = candidate.resolve(strict=True)
        root = self._find_root(candidate)
        try:
            identity = inspect_repository_identity(
                root,
                sandbox=self.sandbox,
                component="repository-validation",
                cancelled=cancelled,
            )
        except RepositoryCancelled:
            raise
        except RepositoryAccessError as exc:
            raise RepositorySelectionError(str(exc)) from exc
        return RepositoryInfo(
            path=root,
            branch=identity.branch,
            identity_hash=identity.identity_hash,
            head_sha=identity.head_sha,
        )

    @staticmethod
    def _find_root(candidate: Path) -> Path:
        """Find a possible worktree root without executing repository Git on host."""
        current = candidate
        while True:
            if (current / ".git").exists():
                return current
            if current.parent == current:
                break
            current = current.parent
        raise RepositorySelectionError("선택한 폴더는 Git 작업 트리가 아닙니다.")
