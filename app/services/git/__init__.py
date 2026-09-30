"""승인된 저장소만 다루는 로컬 Git 안전 어댑터."""

from .repository import (
    GitRecoveryBundle,
    GitRepository,
    GitRepositoryCancelled,
    GitRepositoryError,
    GitRepositoryTimeout,
    GitSnapshot,
)
from .worktree import GitWorktreeError, IsolatedGitWorktree, IsolatedWorktreeRecord

__all__ = [
    "GitRecoveryBundle",
    "GitRepository",
    "GitRepositoryCancelled",
    "GitRepositoryError",
    "GitRepositoryTimeout",
    "GitSnapshot",
    "GitWorktreeError",
    "IsolatedGitWorktree",
    "IsolatedWorktreeRecord",
]
