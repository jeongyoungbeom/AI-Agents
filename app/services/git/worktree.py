from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.services.sandbox import DockerGitMetadataMount, DockerSandbox

from .repository import GitRepository, GitRepositoryError, GitSnapshot


_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_GITDIR_PREFIX = "gitdir: "


class GitWorktreeError(GitRepositoryError):
    pass


@dataclass(frozen=True)
class IsolatedWorktreeRecord:
    source_path: str
    source_snapshot: GitSnapshot
    temp_root: str
    worktree_path: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "source_snapshot": {
                "head": self.source_snapshot.head,
                "branch": self.source_snapshot.branch,
                "status": self.source_snapshot.status,
                "untracked_files": list(self.source_snapshot.untracked_files),
            },
            "temp_root": self.temp_root,
            "worktree_path": self.worktree_path,
        }


class IsolatedGitWorktree:
    """A gateway-owned, detached worktree for one approved pipeline run.

    Git stores host-absolute worktree metadata. The host lifecycle only creates
    that metadata without checking out repository files. The fixed Docker Git
    boundary performs checkout, reads, staging, commits, and verification. A
    synthetic pointer lets that boundary resolve the linked worktree's common
    Git directory without showing it to model processes.
    """

    def __init__(
        self,
        source_path: Path,
        sandbox: DockerSandbox,
        *,
        run_id: str,
        cancelled=None,
    ) -> None:
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("run_id is invalid for an isolated worktree")
        source_path = source_path.resolve(strict=True)
        if not source_path.is_dir():
            raise GitWorktreeError("승인된 원본 저장소 폴더를 찾을 수 없습니다.")
        git_entry = source_path / ".git"
        if git_entry.is_symlink() or not git_entry.is_dir():
            raise GitWorktreeError(
                "원본 저장소의 .git metadata가 일반 디렉터리가 아니어서 격리 worktree를 만들지 않습니다."
            )
        self.source = GitRepository(
            source_path,
            sandbox,
            operation_id=run_id,
            component="source-git",
            cancelled=cancelled,
        )
        self.run_id = run_id
        self._root = Path(tempfile.mkdtemp(prefix=f"ai-agents-worktree-{run_id}-"))
        self.path = self._root / "worktree"
        self._pointer_file = self._root / "container-gitdir"
        self.repository: GitRepository | None = None
        self.source_snapshot: GitSnapshot | None = None

    def begin(self) -> tuple[GitRepository, GitSnapshot]:
        """Capture and validate the clean original before any worktree exists."""
        before = self.source.snapshot()
        self.source_snapshot = before
        validated = self.source.preflight()
        if before != validated:
            raise GitWorktreeError("원본 저장소가 시작 점검 도중 변경되어 worktree를 만들지 않습니다.")
        self.source_snapshot = validated
        return self.source, validated

    def create(self, source_snapshot: GitSnapshot) -> GitRepository:
        if self.source_snapshot != source_snapshot:
            raise GitWorktreeError("원본 저장소 snapshot이 worktree 생성 전 변경되었습니다.")
        if self.repository is not None:
            return self.repository
        self._run_lifecycle(
            "worktree",
            "add",
            "--no-checkout",
            "--detach",
            "--force",
            str(self.path),
            source_snapshot.head,
        )
        try:
            metadata = self._linked_metadata()
            repository = GitRepository(
                self.path,
                self.source.sandbox,
                operation_id=self.run_id,
                component="worktree-git",
                cancelled=self.source.cancelled,
                git_metadata=metadata,
            )
            repository.checkout_detached(source_snapshot.head)
            isolated_snapshot = repository.preflight(
                require_branch=False, check_path_escapes=True
            )
            if isolated_snapshot.head != source_snapshot.head or isolated_snapshot.status:
                raise GitWorktreeError("생성한 임시 worktree의 시작 상태가 올바르지 않습니다.")
        except BaseException:
            # A partially created worktree is gateway-owned and has no model
            # changes. Best-effort removal is safe here; a failure is reported
            # by the original exception and the owned directory remains.
            self.cleanup()
            raise
        self.repository = repository
        return repository

    @property
    def record(self) -> IsolatedWorktreeRecord:
        if self.source_snapshot is None:
            raise GitWorktreeError("원본 snapshot 없이 worktree 기록을 만들 수 없습니다.")
        return IsolatedWorktreeRecord(
            source_path=str(self.source.path),
            source_snapshot=self.source_snapshot,
            temp_root=str(self._root),
            worktree_path=str(self.path),
        )

    def cleanup(self) -> str | None:
        """Remove only this successfully completed, gateway-owned worktree."""
        if self.repository is None:
            if self.path.exists():
                try:
                    self._run_lifecycle("worktree", "remove", "--force", str(self.path))
                except GitWorktreeError as exc:
                    return str(exc)
            self._remove_empty_root()
            return None
        try:
            self._run_lifecycle("worktree", "remove", "--force", str(self.path))
        except GitWorktreeError as exc:
            return str(exc)
        if self.path.exists():
            return "Git worktree remove 뒤에도 gateway-owned 작업 경로가 남아 있습니다."
        self._remove_empty_root()
        return None

    def _linked_metadata(self) -> DockerGitMetadataMount:
        if self.path.is_symlink() or not self.path.is_dir():
            raise GitWorktreeError("생성한 worktree 경로가 안전한 디렉터리가 아닙니다.")
        if self.path.parent.resolve() != self._root.resolve():
            raise GitWorktreeError("생성한 worktree가 gateway-owned temp root 밖에 있습니다.")
        pointer = self.path / ".git"
        if pointer.is_symlink() or not pointer.is_file():
            raise GitWorktreeError("생성한 worktree의 .git pointer가 안전하지 않습니다.")
        try:
            value = pointer.read_text(encoding="utf-8")
        except OSError as exc:
            raise GitWorktreeError("생성한 worktree의 .git pointer를 읽지 못했습니다.") from exc
        if not value.startswith(_GITDIR_PREFIX):
            raise GitWorktreeError("생성한 worktree의 .git pointer 형식이 올바르지 않습니다.")
        raw_directory = value[len(_GITDIR_PREFIX) :].strip()
        if not raw_directory:
            raise GitWorktreeError("생성한 worktree의 Git directory가 비어 있습니다.")
        git_directory = Path(raw_directory)
        if not git_directory.is_absolute():
            git_directory = pointer.parent / git_directory
        try:
            git_directory = git_directory.resolve(strict=True)
            common_directory = (self.source.path / ".git").resolve(strict=True)
            relative = git_directory.relative_to(common_directory).as_posix()
        except (OSError, ValueError) as exc:
            raise GitWorktreeError(
                "생성한 worktree의 Git metadata가 원본 저장소 밖을 가리켜 중단했습니다."
            ) from exc
        parts = relative.split("/")
        if (
            len(parts) != 2
            or parts[0] != "worktrees"
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", parts[1])
        ):
            raise GitWorktreeError("생성한 worktree의 Git metadata 경로가 안전하지 않습니다.")
        self._pointer_file.write_text(
            f"gitdir: /ai-agents-gitdir/{relative}\n", encoding="utf-8"
        )
        return DockerGitMetadataMount(
            common_directory=common_directory,
            git_dir_relative=relative,
            pointer_file=self._pointer_file,
        )

    def _run_lifecycle(self, *arguments: str) -> None:
        """Run one fixed, non-hook Git worktree lifecycle command on the host."""
        try:
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.source.path),
                    "-c",
                    "core.hooksPath=/dev/null",
                    "worktree",
                    *arguments[1:],
                ]
                if arguments and arguments[0] == "worktree"
                else ["git", "-C", str(self.source.path), *arguments],
                text=True,
                encoding="utf-8",
                errors="replace",
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GitWorktreeError("임시 Git worktree 수명주기 명령을 실행하지 못했습니다.") from exc
        if result.returncode != 0:
            detail = (result.stderr.strip() or result.stdout.strip())[:1200]
            raise GitWorktreeError(
                f"임시 Git worktree 수명주기 명령이 실패했습니다: {detail}"
            )

    def _remove_empty_root(self) -> None:
        for path in (self._pointer_file,):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                return
        try:
            self._root.rmdir()
        except OSError:
            return
