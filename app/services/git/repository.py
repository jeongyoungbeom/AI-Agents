from __future__ import annotations

import os
import re
import base64
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.contracts import normalize_stage_scope
from app.services.sandbox import (
    DockerGitMetadataMount,
    DockerSandbox,
    DockerSandboxCancelled,
    DockerSandboxError,
    DockerSandboxTimeout,
)


class GitRepositoryError(RuntimeError):
    pass


class GitRepositoryCancelled(GitRepositoryError):
    pass


class GitRepositoryTimeout(GitRepositoryError):
    pass


@dataclass(frozen=True)
class GitSnapshot:
    head: str
    branch: str
    status: str
    untracked_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class GitRecoveryBundle:
    """Non-destructive evidence for a stopped stage with local changes."""

    base_head: str
    current_head: str
    branch: str
    status: str
    changed_files: tuple[str, ...]
    patch: str

    @property
    def has_changes(self) -> bool:
        return bool(self.changed_files) or self.base_head != self.current_head

    def to_dict(self) -> dict:
        return {
            "base_head": self.base_head,
            "current_head": self.current_head,
            "branch": self.branch,
            "status": self.status,
            "changed_files": list(self.changed_files),
            "patch_available": bool(self.patch),
        }


class GitRepository:
    """브랜치를 바꾸거나 원격에 쓰지 않는 로컬 Git 어댑터."""

    def __init__(
        self,
        path: Path,
        sandbox: DockerSandbox | None = None,
        *,
        operation_id: str | None = None,
        component: str = "git",
        cancelled: Callable[[], bool] | None = None,
        git_metadata: DockerGitMetadataMount | None = None,
    ):
        self.path = path.resolve(strict=True)
        self.sandbox = sandbox or DockerSandbox()
        self.operation_id = operation_id
        self.component = component
        self.cancelled = cancelled
        self.git_metadata = git_metadata

    def preflight(
        self,
        *,
        require_clean: bool = True,
        require_branch: bool = True,
        check_path_escapes: bool = False,
    ) -> GitSnapshot:
        top = self._git("rev-parse", "--show-toplevel")
        if top != "/workspace":
            raise GitRepositoryError(
                f"승인된 경로가 Git 최상위 폴더가 아닙니다: {self.path}"
            )
        snapshot = self.snapshot()
        if require_branch and not snapshot.branch:
            raise GitRepositoryError("detached HEAD에서는 실행하지 않습니다.")
        if require_clean and snapshot.status:
            raise GitRepositoryError("작업 트리에 기존 변경이 있어 실행을 중단했습니다.")
        if check_path_escapes:
            self.assert_safe_worktree_paths()
        if not self._git("config", "user.name", allow_empty=True):
            raise GitRepositoryError("Git user.name 설정이 필요합니다.")
        if not self._git("config", "user.email", allow_empty=True):
            raise GitRepositoryError("Git user.email 설정이 필요합니다.")
        return snapshot

    def snapshot(self) -> GitSnapshot:
        status = self._git("status", "--porcelain=v1", "--untracked-files=all", allow_empty=True)
        return GitSnapshot(
            head=self._git("rev-parse", "HEAD"),
            branch=self._git("branch", "--show-current", allow_empty=True),
            status=status,
            untracked_files=tuple(
                line[3:] for line in status.splitlines() if line.startswith("?? ")
            ),
        )

    def checkout_detached(self, revision: str) -> None:
        """Materialize one immutable revision inside the Git sandbox.

        Linked worktrees are created without a host checkout, so any
        repository-configured checkout filters run only behind this boundary.
        """
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise GitRepositoryError("격리 worktree checkout revision이 올바르지 않습니다.")
        self._run("checkout", "--detach", "--force", "--no-progress", revision)

    def retain_candidate(self, run_id: str, revision: str) -> None:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,79}', run_id):
            raise GitRepositoryError('candidate run ID가 올바르지 않습니다.')
        if not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise GitRepositoryError('candidate SHA가 올바르지 않습니다.')
        self._run('update-ref', f'refs/ai-agents/{run_id}/candidate', revision)

    def matches_pending_commit(self, before: str, candidate: str, message: str) -> bool:
        return (self._git('rev-list', '--parents', '-n', '1', candidate) == f'{candidate} {before}'
                and self._git('show', '-s', '--format=%s', candidate) == message)

    def assert_position(self, before: GitSnapshot, *, allow_worktree_changes: bool) -> GitSnapshot:
        after = self.snapshot()
        if after.head != before.head or after.branch != before.branch:
            raise GitRepositoryError(
                "에이전트가 Git 브랜치 또는 HEAD를 변경해 자동 처리를 중단했습니다."
            )
        if not allow_worktree_changes and after.status != before.status:
            raise GitRepositoryError(
                "읽기 전용 에이전트가 작업 트리를 변경해 자동 처리를 중단했습니다."
            )
        return after

    def commit_scope(self, message: str, scope: tuple[str, ...]) -> str:
        """Commit only current changes explicitly authorized by a stage scope.

        The preflight requirement means this worktree was clean before the
        stage began.  We still validate every current path before staging, so
        an out-of-scope write is never silently included in an AI commit.
        """
        self.assert_safe_worktree_paths()
        snapshot = self.snapshot()
        changed = self.assert_scope(scope)
        if not changed:
            return snapshot.head
        self._run("add", "--", *changed)
        self._run(
            "-c",
            "commit.gpgSign=false",
            "commit",
            "--no-verify",
            "-m",
            message,
        )
        committed = self.snapshot()
        if committed.status:
            raise GitRepositoryError(
                "범위가 확인되지 않은 작업 트리 변경이 남아 단계 완료를 중단했습니다."
            )
        return committed.head

    def assert_scope(self, scope: tuple[str, ...]) -> tuple[str, ...]:
        self.assert_safe_worktree_paths()
        allowed = tuple(normalize_stage_scope(item) for item in scope)
        changed = self.worktree_files()
        disallowed = tuple(
            path for path in changed if not self._path_is_in_scope(path, allowed)
        )
        if disallowed:
            rendered = ", ".join(disallowed[:12])
            if len(disallowed) > 12:
                rendered += f" 외 {len(disallowed) - 12}개"
            raise GitRepositoryError(
                "승인된 단계 범위를 벗어난 파일 변경이 감지되어 커밋하지 않았습니다: "
                f"{rendered} (허용 범위: {', '.join(allowed)})"
            )
        return changed

    def worktree_files(self) -> tuple[str, ...]:
        """Return all Git-visible changed paths, including both rename sides."""
        output = self._run(
            "status", "--porcelain=v1", "-z", "--untracked-files=all"
        ).stdout
        records = output.split("\0")
        paths: list[str] = []
        index = 0
        while index < len(records):
            record = records[index]
            index += 1
            if not record:
                continue
            if len(record) < 4:
                raise GitRepositoryError("Git 상태 출력을 해석하지 못했습니다.")
            code, path = record[:2], record[3:]
            paths.append(path)
            if "R" in code or "C" in code:
                if index >= len(records) or not records[index]:
                    raise GitRepositoryError("Git 이름 변경 상태 출력을 해석하지 못했습니다.")
                paths.append(records[index])
                index += 1
        return tuple(dict.fromkeys(paths))

    def recovery_bundle(self, base: GitSnapshot) -> GitRecoveryBundle:
        """Capture a reviewable patch without modifying the user's worktree."""
        after = self.snapshot()
        patch = self._run(
            "diff", "--binary", "--no-ext-diff", base.head, "--"
        ).stdout
        return GitRecoveryBundle(
            base_head=base.head,
            current_head=after.head,
            branch=after.branch,
            status=after.status,
            changed_files=self.worktree_files(),
            patch=patch,
        )

    def changed_files(self, base_sha: str, candidate_sha: str) -> tuple[str, ...]:
        if base_sha == candidate_sha:
            return ()
        output = self._git("diff", "--name-only", f"{base_sha}..{candidate_sha}", allow_empty=True)
        return tuple(line.strip() for line in output.splitlines() if line.strip())

    def diff_stat(self, base_sha: str, candidate_sha: str) -> str:
        if base_sha == candidate_sha:
            return "(변경 없음)"
        return self._git("diff", "--stat", f"{base_sha}..{candidate_sha}", allow_empty=True)

    def review_bundle(self, base_sha: str, candidate_sha: str) -> dict:
        if any(not re.fullmatch(r'[0-9a-f]{40}', sha) for sha in (base_sha, candidate_sha)):
            raise GitRepositoryError('리뷰 revision이 유효한 SHA가 아닙니다.')
        baseline = self.snapshot()
        if baseline.head != candidate_sha or baseline.status:
            raise GitRepositoryError('리뷰 candidate와 현재 작업 공간이 일치하지 않습니다.')
        patch_bytes = self._run('diff', '--binary', '--no-ext-diff', '--no-textconv',
                                f'{base_sha}..{candidate_sha}', binary_output=True).stdout
        self.assert_position(baseline, allow_worktree_changes=False)
        import hashlib
        return {'base_sha': base_sha, 'candidate_sha': candidate_sha,
                'patch': patch_bytes.decode('utf-8', errors='replace'),
                'patch_base64': base64.b64encode(patch_bytes).decode('ascii'),
                'patch_sha256': hashlib.sha256(patch_bytes).hexdigest(),
                'changed_files': list(self.changed_files(base_sha, candidate_sha)),
                'untrusted_repository_data': True}

    def assert_safe_worktree_paths(self) -> None:
        """Reject symlinks that could take a writable role outside its checkout."""
        root = self.path.resolve(strict=True)
        for parent, directories, files in os.walk(root, followlinks=False):
            parent_path = Path(parent)
            directories[:] = [name for name in directories if name != ".git"]
            for name in [*directories, *files]:
                candidate = parent_path / name
                if not candidate.is_symlink():
                    continue
                try:
                    candidate.resolve(strict=False).relative_to(root)
                except ValueError as exc:
                    relative = candidate.relative_to(root).as_posix()
                    raise GitRepositoryError(
                        "저장소 밖을 가리키는 symlink가 있어 쓰기 작업을 중단했습니다: "
                        f"{relative}"
                    ) from exc

    def assert_commit_scope(
        self, base_sha: str, candidate_sha: str, scope: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Verify a candidate range is linear and remains within its scope."""
        if base_sha == candidate_sha:
            return ()
        merge_base = self._git("merge-base", base_sha, candidate_sha)
        if merge_base != base_sha:
            raise GitRepositoryError(
                "반영 후보 commit이 시작 snapshot에서 선형으로 이어지지 않아 중단했습니다."
            )
        allowed = tuple(normalize_stage_scope(item) for item in scope)
        changed = self.changed_files(base_sha, candidate_sha)
        disallowed = tuple(
            path for path in changed if not self._path_is_in_scope(path, allowed)
        )
        if disallowed:
            rendered = ", ".join(disallowed[:12])
            if len(disallowed) > 12:
                rendered += f" 외 {len(disallowed) - 12}개"
            raise GitRepositoryError(
                "반영 후보에 승인 범위 밖 변경이 있습니다: "
                f"{rendered} (허용 범위: {', '.join(allowed)})"
            )
        return changed

    def apply_candidate(
        self,
        base: GitSnapshot,
        candidate_sha: str,
        scope: tuple[str, ...],
    ) -> str:
        """Apply a verified linear candidate only to an unchanged clean source.

        The source checkout must still be exactly the snapshot from which the
        isolated worktree started. With that invariant, fast-forwarding the
        verified range preserves its exact SHA for crash recovery. A failed command leaves
        Git's evidence in place instead of attempting an unsafe rollback.
        """
        current = self.snapshot()
        if current != base:
            raise GitRepositoryError(
                "원본 저장소의 HEAD, 브랜치 또는 작업 트리가 실행 중 바뀌어 자동 반영하지 않았습니다."
            )
        self.assert_commit_scope(base.head, candidate_sha, scope)
        if base.head == candidate_sha:
            return current.head
        self._run(
            "-c",
            "commit.gpgSign=false",
            "merge",
            "--ff-only",
            "--no-edit",
            candidate_sha,
        )
        applied = self.snapshot()
        if applied.head != candidate_sha or applied.branch != base.branch or applied.status:
            raise GitRepositoryError(
                "원본 반영 뒤 확인되지 않은 작업 트리 변경이 남아 자동 처리를 중단했습니다."
            )
        return applied.head

    @staticmethod
    def _path_is_in_scope(path: str, scope: tuple[str, ...]) -> bool:
        normalized_path = normalize_stage_scope(path)
        if normalized_path.endswith("/"):
            # Git's status output refers to files, but fail closed if a future
            # Git version ever yields a directory-like path here.
            return False
        return any(
            normalized_path.startswith(item)
            if item.endswith("/")
            else normalized_path == item
            for item in scope
        )

    def _git(self, *arguments: str, allow_empty: bool = False) -> str:
        result = self._run(*arguments)
        output = result.stdout.strip()
        if not output and not allow_empty:
            raise GitRepositoryError(f"Git 결과가 비어 있습니다: {' '.join(arguments)}")
        return output

    def _run(self, *arguments: str, binary_output: bool = False):
        git_prefix = ["git"]
        if self.git_metadata is not None:
            git_prefix.extend(
                (
                    f"--git-dir=/ai-agents-gitdir/{self.git_metadata.git_dir_relative}",
                    "--work-tree=/workspace",
                )
            )
        command = [*git_prefix, '-c', 'core.hooksPath=/dev/null', '-c', 'core.fsmonitor=false',
                   '-c', 'core.useBuiltinFSMonitor=false', '-c', 'core.pager=cat',
                   '-c', f"core.autocrlf={'true' if os.name == 'nt' else 'false'}", *arguments]
        if binary_output:
            # Keep CRLF, trailing whitespace and non-UTF8 bytes across the
            # Docker CLI's text transport. Only fixed Git argv reaches this bridge.
            command = ['python3', '-c',
                'import base64,subprocess,sys; r=subprocess.run(sys.argv[1:],stdout=subprocess.PIPE,stderr=subprocess.PIPE); '
                'sys.stdout.buffer.write(base64.b64encode(r.stdout)); sys.stderr.buffer.write(r.stderr); sys.exit(r.returncode)',
                *command]
        try:
            result = self.sandbox.run(
                self.path,
                command,
                writable_workspace=not binary_output,
                timeout=120,
                cancelled=self.cancelled,
                operation_id=self.operation_id,
                component=self.component,
                git_metadata=self.git_metadata,
                writable_git_metadata=self.git_metadata is not None and not binary_output,
            )
        except DockerSandboxCancelled as exc:
            raise GitRepositoryCancelled("사용자가 Git 작업을 중지했습니다.") from exc
        except DockerSandboxTimeout as exc:
            raise GitRepositoryTimeout("격리 컨테이너의 Git 작업 시간이 초과되었습니다.") from exc
        except DockerSandboxError as exc:
            raise GitRepositoryError("격리 컨테이너에서 Git 명령을 실행하지 못했습니다.") from exc
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise GitRepositoryError(f"Git 명령 실패: {detail[:1200]}")
        if binary_output:
            return subprocess.CompletedProcess(result.args, result.returncode,
                base64.b64decode(result.stdout.strip(), validate=True), result.stderr)
        return result
