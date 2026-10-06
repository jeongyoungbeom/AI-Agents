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
from .workspace import ModelWorkspace, ModelWorkspaceError


class HermesExecutionError(RuntimeError):
    def __init__(self, message: str, *, usage: TokenUsage | None = None,
                 category: str = "execution", retryable: bool = False,
                 diagnostic_ref: str = ""):
        super().__init__(message)
        self.usage = usage or TokenUsage()
        self.usage_reported = usage is not None
        self.category = category
        self.retryable = retryable
        self.diagnostic_ref = diagnostic_ref


class HermesCancelled(HermesExecutionError):
    pass


@dataclass(frozen=True)
class HermesResult:
    text: str
    usage: TokenUsage
    elapsed_seconds: float


class HermesRunner:
    """Hermes를 shell 없이 실행하고 취소·시간 제한·역할별 쓰기 권한을 적용한다."""

    supports_workspace_policy = True

    def pipeline_invocation_turns(self, role_id: RoleId) -> int:
        # Match the existing bounded planning invocation (D007); a pipeline
        # call must not reserve one turn while silently executing 120/160.
        return min(8, self.settings.roles[role_id].max_turns)

    def pipeline_invocation_estimate(self, role_id: RoleId, input_tokens: int, output_tokens: int) -> int:
        return self.pipeline_invocation_turns(role_id) * (input_tokens + output_tokens)

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

    def run(self, run_id, stage_id, role_id, repository, prompt, *, allow_writes,
            write_scope=None, review_bundle=None, execution_environment=None, **kwargs):
        if (kwargs.get('no_tools', False) or (not self.settings.docker_required
                and write_scope is None and review_bundle is None and execution_environment is None)):
            return self._run(run_id, stage_id, role_id, repository, prompt,
                             allow_writes=allow_writes, **kwargs)
        if not self.settings.docker_required:
            raise HermesExecutionError('도구 실행에는 필수 Docker 격리가 필요합니다.', category='startup')
        if allow_writes and not write_scope:
            raise HermesExecutionError('쓰기 도구에는 명시적인 stage scope가 필요합니다.', category='startup')
        image = getattr(execution_environment, 'image', self.settings.isolation.image)
        try:
            workspace = ModelWorkspace(
                Path(repository), self.artifacts.path_for(run_id, Path('stages') / stage_id / 'model-workspaces'),
                scope=write_scope if allow_writes else (), review=review_bundle, image=image,
                cache_mounts=getattr(execution_environment, 'cache_mounts', ()),
            )
        except (ModelWorkspaceError, OSError, ValueError) as exc:
            raise HermesExecutionError(str(exc), category='startup') from exc
        result = None
        execution_error = None
        self.artifacts.write_json(run_id, Path('stages') / stage_id / f'{role_id.value}-workspace.json', {
            'snapshot_path': str(workspace.path), 'policy_path': str(workspace.policy),
            'write_scope': list(workspace.scope), 'changed_files': None, 'image': image,
            'status': 'RUNNING',
            'review_base_sha': (review_bundle or {}).get('base_sha'),
            'review_candidate_sha': (review_bundle or {}).get('candidate_sha'),
        })
        try:
            prompt += ('\n\n게이트웨이가 제공한 작업 공간은 /workspace의 파일 snapshot이다. '
                       '원본 Git metadata는 제공하지 않는다. Git 명령으로 기준점을 추측하지 말고 '
                       '계약의 SHA와 리뷰 manifest/diff를 사용하라. 파일·터미널 도구에서 '
                       '/workspace 경로를 사용한다. 쓰기는 명시된 scope에만 허용된다. '
                       '외부 호스트 도구·스킬·브라우저는 이번 호출의 실행 권한에 포함되지 않는다.\n')
            result = self._run(run_id, stage_id, role_id, workspace.path, prompt,
                               allow_writes=allow_writes, workspace_policy=workspace.policy, **kwargs)
            return result
        except HermesExecutionError as exc:
            execution_error = exc
            raise
        finally:
            try:
                changed = workspace.preserve_changes()
                self.artifacts.write_json(run_id, Path('stages') / stage_id / f'{role_id.value}-workspace.json', {
                    'snapshot_path': str(workspace.path), 'policy_path': str(workspace.policy),
                    'write_scope': list(workspace.scope), 'changed_files': list(changed), 'image': image,
                    'status': 'FILES_PRESERVED',
                    'review_base_sha': (review_bundle or {}).get('base_sha'),
                    'review_candidate_sha': (review_bundle or {}).get('candidate_sha'),
                })
            except ModelWorkspaceError as exc:
                usage = result.usage if result else (
                    execution_error.usage if execution_error and execution_error.usage_reported else None)
                raise HermesExecutionError(str(exc), usage=usage, category='workspace_integrity') from exc

    def _run(
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
        no_tools: bool = False,
        max_turns: int | None = None,
        max_output_tokens: int | None = None,
        image_path: Path | None = None,
        ignore_rules: bool = False,
        cancelled: Callable[[], bool] | None = None,
        heartbeat: Callable[[], None] | None = None,
        workspace_policy: Path | None = None,
    ) -> HermesResult:
        if self._stop_requested.is_set():
            raise HermesCancelled("게이트웨이 종료로 Hermes 실행을 중지했습니다.", category="startup")
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
        result_path = self.artifacts.path_for(
            run_id, Path("stages") / stage_id / f"{role_id.value}-result.json"
        )
        # A fresh placeholder prevents a failed retry from reading a previous
        # attempt's report. Hermes replaces it only after its one-shot finalizer.
        self.artifacts.write_text(
            run_id,
            Path("stages") / stage_id / f"{role_id.value}-usage.json",
            "{}\n",
            redact=False,
        )
        self.artifacts.write_text(
            run_id, Path("stages") / stage_id / f"{role_id.value}-result.json",
            "{}\n", redact=False,
        )
        command = [
            str(self.settings.executable),
            "--profile",
            role.profile,
            "--usage-file",
            str(usage_path),
            "chat",
            "--result-file",
            str(result_path),
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
            "--max-turns",
            str(max_turns if max_turns is not None else role.max_turns),
            "--source",
            "tool",
        ]
        if no_tools:
            command.append("--no-tools")
        else:
            command.extend(("--toolsets", 'terminal,file' if workspace_policy is not None
                            else toolsets if toolsets is not None else role.toolsets))
        if max_output_tokens is not None and self.settings.provider != "openai-codex":
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
        env.pop('AI_AGENTS_WORKSPACE_POLICY', None)
        if workspace_policy is not None:
            policy = json.loads(workspace_policy.read_text(encoding='utf-8'))
            env['AI_AGENTS_WORKSPACE_POLICY'] = str(workspace_policy)
            env['TERMINAL_DOCKER_IMAGE'] = policy['image']
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
            raise HermesExecutionError("Hermes 프로세스를 시작하지 못했습니다.",
                                       category="startup", retryable=True) from exc
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
                        raise HermesExecutionError("Hermes 실행 시간 제한을 초과했습니다.", category="timeout")
        except BaseException as exc:
            if process.poll() is None:
                process_tree.terminate()
            if isinstance(exc, HermesExecutionError):
                reported = self._usage_from_report(usage_path)
                if reported is not None:
                    exc.usage = reported
                    exc.usage_reported = True
            raise
        finally:
            process_tree.close()
            with self._active_lock:
                self._active_processes.pop(process, None)
        if self._stop_requested.is_set():
            raise HermesCancelled("게이트웨이 종료로 Hermes 실행을 중지했습니다.",
                                  usage=self._usage_from_report(usage_path))
        elapsed = time.monotonic() - started
        safe_stdout = self.redactor.text(stdout.strip())
        safe_stderr = self.redactor.text(stderr.strip())
        self.artifacts.append_role_log(
            run_id,
            stage_id,
            role_id.value,
            f"[stdout]\n{safe_stdout}\n[stderr]\n{safe_stderr}",
        )
        diagnostic_ref = f"{run_id}/stages/{stage_id}/{role_id.value}.log"
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
        try:
            report = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report = None
        if not isinstance(report, dict) or report.get("status") not in {"succeeded", "failed"}:
            detail = "\n".join(part for part in (safe_stdout, safe_stderr)
                               if part and not part.startswith("session_id:"))[-1200:] or "결과 파일 없음"
            category = self._failure_category(detail) if process.returncode else "invalid_result"
            raise HermesExecutionError(
                f"Hermes {role_id.value} {category}(code={process.returncode}): {detail} [진단: {diagnostic_ref}]",
                usage=usage, category=category, diagnostic_ref=diagnostic_ref,
            )
        internal_stream_retries = report.get("internal_stream_retries")
        if (not isinstance(internal_stream_retries, int)
                or isinstance(internal_stream_retries, bool)
                or internal_stream_retries < 0):
            internal_stream_retries = None
        self.artifacts.append_role_log(
            run_id, stage_id, role_id.value,
            f"[runtime] internal_stream_retries={internal_stream_retries if internal_stream_retries is not None else 'unknown'}",
        )
        payload = report.get("text")
        if not isinstance(payload, str):
            raise HermesExecutionError(f"Hermes 결과 text 형식이 잘못됐습니다. [진단: {diagnostic_ref}]",
                                       usage=usage, category="invalid_result", diagnostic_ref=diagnostic_ref)
        if process.returncode != 0 or report["status"] == "failed":
            detail = self.redactor.text(str(report.get("error") or report.get("failure_reason") or payload))[-1200:]
            category = self._failure_category(detail)
            raise HermesExecutionError(
                f"Hermes {role_id.value} {category}(code={process.returncode}): {detail or '원인 미확인'} [진단: {diagnostic_ref}]",
                usage=usage, category=category, diagnostic_ref=diagnostic_ref,
            )
        if not payload.strip():
            raise HermesExecutionError(
                f"Hermes {role_id.value} 응답이 비어 있습니다. [진단: {diagnostic_ref}]", usage=usage,
                category="invalid_result", diagnostic_ref=diagnostic_ref,
            )
        return HermesResult(self.redactor.text(payload), usage, elapsed)

    @staticmethod
    def _failure_category(detail: str) -> str:
        lowered = detail.casefold()
        if any(term in lowered for term in ("unauthorized", "authentication", "invalid api key", "401")):
            return "authentication"
        if any(term in lowered for term in ("timeout", "timed out", "no sse events", "stalled stream")):
            return "timeout"
        return "provider"

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
