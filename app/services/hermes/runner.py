from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.contracts import RoleId, TokenUsage
from app.services.logging.redaction import SecretRedactor
from app.services.process_tree import ProcessTree, isolated_process_options
from app.storage import ArtifactStore

from .config import HermesModelSettings, HermesSettings


class HermesExecutionError(RuntimeError):
    def __init__(self, message: str, *, usage: TokenUsage | None = None):
        super().__init__(message)
        self.usage = usage or TokenUsage()


class HermesCancelled(HermesExecutionError):
    pass


@dataclass(frozen=True)
class HermesResult:
    text: str
    usage: TokenUsage
    elapsed_seconds: float


class HermesRunner:
    """Hermes를 shell 없이 실행하고 취소·시간 제한·역할별 쓰기 권한을 적용한다."""

    def __init__(
        self,
        settings: HermesSettings,
        artifacts: ArtifactStore,
        *,
        redactor: SecretRedactor | None = None,
    ):
        self.settings = settings
        self.artifacts = artifacts
        self.redactor = redactor or SecretRedactor()
        self._stop_requested = threading.Event()
        self._active_lock = threading.RLock()
        self._active_processes: dict[subprocess.Popen[str], ProcessTree] = {}

    def request_stop(self) -> None:
        """새 실행을 막고 현재 Hermes 프로세스들이 즉시 종료를 시작하게 한다."""
        self._stop_requested.set()
        with self._active_lock:
            active = tuple(self._active_processes.values())
        for process_tree in active:
            process_tree.terminate()

    def active_process_count(self) -> int:
        with self._active_lock:
            return len(self._active_processes)

    def run(
        self,
        run_id: str,
        stage_id: str,
        role_id: RoleId,
        repository: Path,
        prompt: str,
        *,
        allow_writes: bool,
        model: str | None = None,
        reasoning: str | None = None,
        toolsets: str | None = None,
        max_turns: int | None = None,
        max_output_tokens: int | None = None,
        image_path: Path | None = None,
        ignore_rules: bool = False,
        cancelled: Callable[[], bool] | None = None,
        heartbeat: Callable[[], None] | None = None,
    ) -> HermesResult:
        if self._stop_requested.is_set():
            raise HermesCancelled("게이트웨이 종료로 Hermes 실행을 중지했습니다.")
        if max_output_tokens is not None and max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        self.settings.validate_installation()
        repository = repository.resolve(strict=True)
        if image_path is not None:
            image_path = image_path.resolve(strict=True)
            if not image_path.is_file():
                raise ValueError("image_path must be a file")
        role = self.settings.roles[role_id]
        invocation = HermesModelSettings(
            model=model or role.model,
            reasoning=reasoning or role.reasoning,
        )
        prompt_path = self.artifacts.write_text(
            run_id,
            Path("stages") / stage_id / f"{role_id.value}-prompt.txt",
            prompt,
        )
        usage_path = self.artifacts.path_for(
            run_id, Path("stages") / stage_id / f"{role_id.value}-usage.json"
        )
        # A fresh placeholder prevents a failed retry from reading a previous
        # attempt's report. Hermes replaces it only after its one-shot finalizer.
        self.artifacts.write_text(
            run_id,
            Path("stages") / stage_id / f"{role_id.value}-usage.json",
            "{}\n",
            redact=False,
        )
        command = [
            str(self.settings.executable),
            "--profile",
            role.profile,
            "--usage-file",
            str(usage_path),
            "chat",
            "--query-file",
            str(prompt_path),
            "--quiet",
            "--oneshot",
            "--in",
            str(repository),
            "--no-restore-cwd",
            "--provider",
            self.settings.provider,
            "--model",
            invocation.model,
            "--reasoning",
            invocation.reasoning,
            "--toolsets",
            toolsets if toolsets is not None else role.toolsets,
            "--max-turns",
            str(max_turns if max_turns is not None else role.max_turns),
            "--source",
            "tool",
        ]
        if max_output_tokens is not None:
            command.extend(("--max-output-tokens", str(max_output_tokens)))
        if image_path is not None:
            command.extend(("--image", str(image_path)))
        if ignore_rules:
            command.append("--ignore-rules")
        if allow_writes:
            # Hermes starts in the approved repository and is not permitted to
            # resume a session's old CWD. Checkpoints provide an additional
            # local recovery trail; the coordinator still verifies Git scope
            # before it stages anything.
            command.extend(("--checkpoints", "--yolo"))
        env = os.environ.copy()
        # Environment variables are lower precedence in spirit but can still
        # bypass a profile's terminal settings in Hermes.  The gateway owns
        # this boundary, so clear all inherited TERMINAL_* overrides first.
        for key in tuple(env):
            if key.startswith("TERMINAL_"):
                env.pop(key, None)
        env["HERMES_HOME"] = str(self.settings.home)
        env["HERMES_TERMINAL_SECURITY_MODE"] = (
            "auto" if allow_writes else "approval-required"
        )
        started = time.monotonic()
        try:
            process = subprocess.Popen(
                command,
                cwd=repository,
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **isolated_process_options(),
            )
        except OSError as exc:
            raise HermesExecutionError("Hermes 프로세스를 시작하지 못했습니다.") from exc
        process_tree = ProcessTree(process)
        with self._active_lock:
            self._active_processes[process] = process_tree
        stdout = ""
        stderr = ""
        try:
            while True:
                try:
                    stdout, stderr = process.communicate(
                        timeout=self.settings.poll_seconds
                    )
                    break
                except subprocess.TimeoutExpired:
                    if heartbeat is not None:
                        heartbeat()
                    if self._stop_requested.is_set():
                        process_tree.terminate()
                        raise HermesCancelled(
                            "게이트웨이 종료로 Hermes 실행을 중지했습니다."
                        )
                    if cancelled is not None and cancelled():
                        process_tree.terminate()
                        raise HermesCancelled("사용자가 Hermes 실행을 중지했습니다.")
                    if time.monotonic() - started >= self.settings.timeout_seconds:
                        process_tree.terminate()
                        raise HermesExecutionError("Hermes 실행 시간 제한을 초과했습니다.")
        except BaseException:
            if process.poll() is None:
                process_tree.terminate()
            raise
        finally:
            process_tree.close()
            with self._active_lock:
                self._active_processes.pop(process, None)
        if self._stop_requested.is_set():
            raise HermesCancelled("게이트웨이 종료로 Hermes 실행을 중지했습니다.")
        elapsed = time.monotonic() - started
        safe_stdout = self.redactor.text(stdout.strip())
        safe_stderr = self.redactor.text(stderr.strip())
        self.artifacts.append_role_log(
            run_id,
            stage_id,
            role_id.value,
            f"[stdout]\n{safe_stdout}\n[stderr]\n{safe_stderr}",
        )
        usage = self._usage_from_report(usage_path)
        if usage is None:
            usage = TokenUsage(
                input_tokens=max(1, (len(prompt) + 3) // 4),
                output_tokens=max(1, (len(safe_stdout) + 3) // 4),
                estimated=True,
            )
            usage_source = "estimated"
        else:
            usage_source = "actual"
        self.artifacts.append_role_log(
            run_id,
            stage_id,
            role_id.value,
            "[usage] "
            f"source={usage_source} input={usage.input_tokens} "
            f"output={usage.output_tokens} total={usage.total_tokens}",
        )
        if process.returncode != 0:
            detail = safe_stderr[-1200:] or safe_stdout[-1200:] or "출력 없음"
            raise HermesExecutionError(
                f"Hermes {role_id.value} 실행 실패(code={process.returncode}): {detail}",
                usage=usage,
            )
        if not safe_stdout:
            raise HermesExecutionError(
                f"Hermes {role_id.value} 응답이 비어 있습니다.", usage=usage
            )
        return HermesResult(safe_stdout, usage, elapsed)

    @staticmethod
    def _usage_from_report(path: Path) -> TokenUsage | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(value, dict):
            return None
        try:
            input_tokens = HermesRunner._usage_number(value.get("input_tokens"))
            output_tokens = HermesRunner._usage_number(value.get("output_tokens"))
            total_tokens = HermesRunner._usage_number(value.get("total_tokens"))
        except ValueError:
            return None
        if input_tokens is None and output_tokens is None and total_tokens is None:
            return None
        input_tokens = input_tokens or 0
        output_tokens = output_tokens or 0
        total_tokens = total_tokens or 0
        if total_tokens < input_tokens + output_tokens:
            return None
        if total_tokens == 0:
            return None
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated=False,
        )

    @staticmethod
    def _usage_number(value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("boolean is not a token count")
        if isinstance(value, int):
            number = value
        elif isinstance(value, str) and value.strip().isdecimal():
            number = int(value)
        else:
            raise ValueError("invalid token count")
        if number < 0:
            raise ValueError("negative token count")
        return number
