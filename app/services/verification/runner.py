from __future__ import annotations

import re
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Callable

from app.services.git import GitRepository
from app.services.process_tree import ProcessTree, isolated_process_options
from app.services.sandbox import (
    DockerSandbox, DockerSandboxCancelled, DockerSandboxError, DockerSandboxTimeout,
)

from .policy import SafeVerificationPolicy


class VerificationCancelled(RuntimeError):
    pass


class VerificationFailureKind:
    PASSED = "passed"
    TOOL_UNAVAILABLE = "tool_unavailable"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    ENVIRONMENT_UNAVAILABLE = "environment_unavailable"
    TIMEOUT = "timeout"
    TEST_FAILURE = "test_failure"


@dataclass(frozen=True)
class VerificationResult:
    command: str
    return_code: int | None
    stdout: str
    stderr: str
    elapsed_seconds: float
    failure_kind: str = VerificationFailureKind.PASSED
    started: bool = True

    @property
    def passed(self) -> bool:
        return self.started and self.return_code == 0

    def to_dict(self) -> dict:
        value = asdict(self)
        value["passed"] = self.passed
        value["failure_kind"] = (
            VerificationFailureKind.PASSED if self.passed else self.failure_kind
        )
        return value


class VerificationRunner:
    def __init__(
        self,
        policy: SafeVerificationPolicy | None = None,
        *,
        sandbox: DockerSandbox | None = None,
        timeout_seconds: float = 900,
        poll_seconds: float = 0.5,
    ):
        self.policy = policy or SafeVerificationPolicy()
        self.sandbox = sandbox or DockerSandbox()
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds

    def run_all(
        self,
        repository: GitRepository,
        commands: tuple[str, ...],
        *,
        cancelled: Callable[[], bool] | None = None,
        heartbeat: Callable[[], None] | None = None,
        operation_id: str | None = None,
        environment: object | None = None,
    ) -> tuple[VerificationResult, ...]:
        baseline = repository.snapshot()
        results: list[VerificationResult] = []
        execution_sandbox = (
            environment.sandbox_for(self.sandbox)
            if environment is not None
            else self.sandbox
        )
        cache_mounts = tuple(getattr(environment, "cache_mounts", ()))
        for command in commands:
            if cancelled is not None and cancelled():
                raise VerificationCancelled("사용자가 검증을 중지했습니다.")
            prepared = self.policy.prepare(command)
            started = time.monotonic()
            try:
                # A verification command runs repository code (tests, Gradle
                # tasks, npm scripts).  It must never run on the gateway host.
                process = execution_sandbox.popen(
                    repository.path,
                    list(prepared.argv),
                    writable_workspace=True,
                    operation_id=operation_id,
                    component="verification",
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    **isolated_process_options(),
                    additional_mounts=cache_mounts,
                    git_metadata=getattr(repository, "git_metadata", None),
                    writable_git_metadata=False,
                )
            except DockerSandboxError as exc:
                if isinstance(exc, DockerSandboxCancelled):
                    raise VerificationCancelled("사용자가 검증을 중지했습니다.") from exc
                results.append(
                    VerificationResult(
                        prepared.display, None, "",
                        f"검증 환경을 시작하지 못했습니다: {exc}",
                        time.monotonic() - started,
                        VerificationFailureKind.TIMEOUT if isinstance(exc, DockerSandboxTimeout)
                        else VerificationFailureKind.ENVIRONMENT_UNAVAILABLE,
                        started=False,
                    )
                )
                repository.assert_position(baseline, allow_worktree_changes=False)
                return tuple(results)
            try:
                process_tree = ProcessTree(process)
            except BaseException:
                execution_sandbox.cleanup_process(
                    process, reason="process-tree-initialization-failed"
                )
                raise
            timed_out = False
            cleanup_reason = "completed"
            try:
                while True:
                    try:
                        stdout, stderr = process.communicate(timeout=self.poll_seconds)
                        break
                    except subprocess.TimeoutExpired:
                        if heartbeat is not None:
                            heartbeat()
                        if cancelled is not None and cancelled():
                            cleanup_reason = "cancelled"
                            execution_sandbox.note_process_termination(
                                process, reason=cleanup_reason
                            )
                            process_tree.terminate()
                            raise VerificationCancelled("사용자가 검증을 중지했습니다.")
                        if time.monotonic() - started >= self.timeout_seconds:
                            cleanup_reason = "timeout"
                            execution_sandbox.note_process_termination(
                                process, reason=cleanup_reason
                            )
                            process_tree.terminate()
                            stdout, stderr = process.communicate()
                            timed_out = True
                            break
            finally:
                process_tree.close()
                execution_sandbox.cleanup_process(process, reason=cleanup_reason)
            if timed_out:
                results.append(
                    VerificationResult(
                        prepared.display,
                        124,
                        stdout[-20000:],
                        (stderr + "\n검증 시간 제한 초과").strip()[-20000:],
                        time.monotonic() - started,
                        VerificationFailureKind.TIMEOUT,
                    )
                )
                repository.assert_position(baseline, allow_worktree_changes=False)
                return tuple(results)
            results.append(
                VerificationResult(
                    prepared.display,
                    int(process.returncode),
                    stdout[-20000:],
                    stderr[-20000:],
                    time.monotonic() - started,
                    self._failure_kind(stdout, stderr, int(process.returncode)),
                )
            )
            repository.assert_position(baseline, allow_worktree_changes=False)
            if process.returncode != 0:
                break
        return tuple(results)

    @staticmethod
    def _failure_kind(stdout: str, stderr: str, return_code: int) -> str:
        if return_code == 0:
            return VerificationFailureKind.PASSED
        # An uncaught terminal exception takes precedence over earlier handled
        # warnings and diagnostic words embedded in assertion messages.
        diagnostics = stderr.strip() or stdout.strip()
        exceptions = re.findall(
            r"^\s*(?:E\s+)?((?:[A-Za-z_][\w.]*)?(?:Error|Exception)):\s*(.*)$",
            diagnostics, re.MULTILINE,
        )
        if exceptions:
            name, message = exceptions[-1]
            exception_type = name.rsplit('.', 1)[-1]
            if exception_type == "ModuleNotFoundError" or (
                exception_type == "ImportError" and re.match(r"No module named\b", message)
            ):
                return VerificationFailureKind.DEPENDENCY_UNAVAILABLE
            if name == "Error" and re.match(r"Cannot find (?:module|package)\b", message):
                return VerificationFailureKind.DEPENDENCY_UNAVAILABLE
            return VerificationFailureKind.TEST_FAILURE
        if return_code == 127:
            return VerificationFailureKind.TOOL_UNAVAILABLE
        # Recognize tool/build diagnostics, not arbitrary log substrings.
        if re.search(
            r"(?im)^\s*(?:exec:|(?:/bin/)?(?:ba)?sh:)\s*.*"
            r"(?:command not found|executable file not found|not recognized as an internal|: not found)(?:\s|$)",
            stderr,
        ):
            return VerificationFailureKind.TOOL_UNAVAILABLE
        if re.search(
            r"(?im)^\s*(?:>\s*|\[ERROR\]\s*)"
            r"(?:Could not resolve\b|No cached version\b|.*\boffline mode\b)",
            diagnostics,
        ):
            return VerificationFailureKind.DEPENDENCY_UNAVAILABLE
        return VerificationFailureKind.TEST_FAILURE
