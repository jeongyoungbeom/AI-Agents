from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from app.services.process_tree import ProcessTree, isolated_process_options


# This is the image verified during E-5. A digest cannot be replaced by a tag
# push after the image has been approved.
DEFAULT_SECURE_DOCKER_IMAGE = (
    "nikolaik/python-nodejs@sha256:"
    "8f958bdc1b4a422bfafd97cab4f69836401f616ae985d4b57a53d254f5bcb038"
)

MANAGED_LABEL = "io.ai-agents.managed"
INSTALLATION_LABEL = "io.ai-agents.installation"
OPERATION_LABEL = "io.ai-agents.operation"
COMPONENT_LABEL = "io.ai-agents.component"
GATEWAY_INSTANCE_LABEL = "io.ai-agents.gateway-instance"
GATEWAY_PID_LABEL = "io.ai-agents.gateway-pid"
GATEWAY_STARTED_LABEL = "io.ai-agents.gateway-started"
_SAFE_OWNER_PART = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}$")
_SAFE_VOLUME_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
_SAFE_CONTAINER_PATH = re.compile(r"^/(?:[a-zA-Z0-9_.-]+/)*[a-zA-Z0-9_.-]+$")
_PROCESS_OWNER_ATTR = "_ai_agents_docker_ownership"
_PROCESS_INSTANCE_ID = uuid.uuid4().hex
_INSTALLATION_ROOT = Path(__file__).resolve().parents[3]
_INSTALLATION_ID = hashlib.sha256(
    str(_INSTALLATION_ROOT).casefold().encode("utf-8")
).hexdigest()[:24]


def _current_process_started_at() -> float:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).create_time())
    except Exception:
        # Startup cleanup treats an unverified owner as live, so a missing
        # optional dependency never broadens deletion authority.
        return 0.0


class DockerSandboxError(RuntimeError):
    pass


class DockerSandboxCancelled(DockerSandboxError):
    pass


class DockerSandboxTimeout(DockerSandboxError):
    pass


@dataclass(frozen=True)
class DockerVolumeMount:
    """A profile-owned named volume mounted inside an isolated container."""

    name: str
    target: str
    read_only: bool = True

    def __post_init__(self) -> None:
        if not _SAFE_VOLUME_NAME.fullmatch(self.name):
            raise ValueError("Docker volume name is invalid")
        if not _SAFE_CONTAINER_PATH.fullmatch(self.target):
            raise ValueError("Docker volume target is invalid")
        if self.target == "/workspace" or self.target.startswith("/workspace/"):
            raise ValueError("Docker volume target cannot overlap the repository mount")
        if not self.read_only:
            raise ValueError("Toolchain cache volumes must be read-only")

    def argument(self) -> str:
        return f"type=volume,src={self.name},dst={self.target},readonly"


@dataclass(frozen=True)
class DockerGitMetadataMount:
    """Gateway-created metadata bridge for an isolated Git worktree.

    A linked worktree's ``.git`` file contains a host path, which is not
    meaningful inside a Linux container.  This mount is deliberately not a
    general additional bind mount: it exposes exactly one checked common Git
    directory and replaces the worktree's pointer with a gateway-created
    container-local pointer.
    """

    common_directory: Path
    git_dir_relative: str
    pointer_file: Path

    def __post_init__(self) -> None:
        common_directory = self.common_directory.resolve(strict=True)
        pointer_file = self.pointer_file.resolve(strict=True)
        if not common_directory.is_dir():
            raise ValueError("Git common directory must be a directory")
        if not pointer_file.is_file():
            raise ValueError("Git metadata pointer must be a file")
        if any(
            unsafe in str(path)
            for path in (common_directory, pointer_file)
            for unsafe in (",", "\r", "\n")
        ):
            raise ValueError("Git metadata mount path is unsafe")
        parts = self.git_dir_relative.split("/")
        if (
            not self.git_dir_relative
            or any(not part or part in {".", ".."} for part in parts)
            or any(not _SAFE_OWNER_PART.fullmatch(part) for part in parts)
        ):
            raise ValueError("Git metadata relative directory is invalid")
        expected = f"gitdir: /ai-agents-gitdir/{self.git_dir_relative}\n"
        if pointer_file.read_text(encoding="utf-8") != expected:
            raise ValueError("Git metadata pointer does not match its Git directory")
        object.__setattr__(self, "common_directory", common_directory)
        object.__setattr__(self, "pointer_file", pointer_file)

    def arguments(self, *, writable_common_directory: bool) -> tuple[str, ...]:
        common = f"type=bind,src={self.common_directory},dst=/ai-agents-gitdir"
        if not writable_common_directory:
            common += ",readonly"
        return (
            common,
            f"type=bind,src={self.pointer_file},dst=/workspace/.git,readonly",
        )


@dataclass(frozen=True)
class DockerContainerOwnership:
    """The one Docker resource a sandbox invocation is allowed to remove."""

    operation_id: str
    component: str
    gateway_instance_id: str
    gateway_process_id: int
    gateway_process_started_at: float
    installation_id: str
    container_name: str
    cidfile: Path

    @property
    def labels(self) -> dict[str, str]:
        return {
            MANAGED_LABEL: "true",
            INSTALLATION_LABEL: self.installation_id,
            OPERATION_LABEL: self.operation_id,
            COMPONENT_LABEL: self.component,
            GATEWAY_INSTANCE_LABEL: self.gateway_instance_id,
            GATEWAY_PID_LABEL: str(self.gateway_process_id),
            GATEWAY_STARTED_LABEL: f"{self.gateway_process_started_at:.6f}",
        }

    def matches(self, labels: dict[str, str]) -> bool:
        return all(labels.get(key) == value for key, value in self.labels.items())


@dataclass
class DockerSandbox:
    """Run fixed argv in an air-gapped repository container with ownership.

    `--rm` is only the normal-exit path. Callers that use ``popen`` must call
    ``cleanup_process`` in ``finally`` after terminating the Docker CLI tree,
    because an interrupted CLI can leave a Created or Running container.
    """

    image: str = DEFAULT_SECURE_DOCKER_IMAGE
    timeout_seconds: int = 900
    gateway_instance_id: str = _PROCESS_INSTANCE_ID
    gateway_process_id: int = os.getpid()
    gateway_process_started_at: float = _current_process_started_at()
    installation_id: str = _INSTALLATION_ID
    event_sink: Callable[[str], None] | None = None
    cidfile_directory: Path | None = None

    def __post_init__(self) -> None:
        if "@sha256:" not in self.image:
            raise ValueError("Docker sandbox image must be pinned by digest")
        if self.timeout_seconds < 1:
            raise ValueError("Docker sandbox timeout must be positive")
        self._validate_owner_part(self.gateway_instance_id, "gateway instance")
        self._validate_owner_part(self.installation_id, "installation")
        if self.gateway_process_id < 1 or self.gateway_process_started_at < 0:
            raise ValueError("Docker gateway process identity is invalid")
        if self.cidfile_directory is not None:
            self.cidfile_directory = self.cidfile_directory.resolve()

    def command(
        self,
        repository: Path,
        argv: Sequence[str],
        *,
        writable_workspace: bool,
        interactive: bool = False,
        ownership: DockerContainerOwnership | None = None,
        additional_mounts: Sequence[DockerVolumeMount] = (),
        git_metadata: DockerGitMetadataMount | None = None,
        writable_git_metadata: bool = False,
    ) -> list[str]:
        root = repository.resolve(strict=True)
        if not root.is_dir():
            raise DockerSandboxError("승인된 저장소 폴더를 찾을 수 없습니다.")
        if any(unsafe in str(root) for unsafe in (",", "\r", "\n")):
            raise DockerSandboxError("승인된 저장소 경로에 허용되지 않은 문자가 있습니다.")
        if not argv or any(not isinstance(item, str) or not item for item in argv):
            raise DockerSandboxError("컨테이너 실행 명령이 올바르지 않습니다.")
        if writable_git_metadata and git_metadata is None:
            raise DockerSandboxError("Git 메타데이터 mount가 없는 쓰기 요청은 허용되지 않습니다.")
        mount = f"type=bind,src={root},dst=/workspace"
        if not writable_workspace:
            mount += ",readonly"
        command = [
            "docker",
            "run",
            "--rm",
            "--init",
            "--network=none",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=256",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=1g",
            "--tmpfs",
            "/root:rw,nosuid,nodev,size=64m",
            "--mount",
            mount,
            "--workdir",
            "/workspace",
        ]
        for mount_spec in additional_mounts:
            if not isinstance(mount_spec, DockerVolumeMount):
                raise DockerSandboxError("허용되지 않은 추가 Docker mount입니다.")
            command.extend(("--mount", mount_spec.argument()))
        if git_metadata is not None:
            if not isinstance(git_metadata, DockerGitMetadataMount):
                raise DockerSandboxError("허용되지 않은 Git 메타데이터 mount입니다.")
            for mount_spec in git_metadata.arguments(
                writable_common_directory=writable_git_metadata
            ):
                command.extend(("--mount", mount_spec))
        if ownership is not None:
            command.extend(("--name", ownership.container_name))
            command.extend(("--cidfile", str(ownership.cidfile)))
            for key, value in ownership.labels.items():
                command.extend(("--label", f"{key}={value}"))
        if interactive:
            command.insert(3, "--interactive")
        return [*command, self.image, *argv]

    def prepare(
        self,
        repository: Path,
        argv: Sequence[str],
        *,
        writable_workspace: bool,
        interactive: bool = False,
        operation_id: str | None = None,
        component: str = "sandbox",
        additional_mounts: Sequence[DockerVolumeMount] = (),
        git_metadata: DockerGitMetadataMount | None = None,
        writable_git_metadata: bool = False,
    ) -> tuple[list[str], DockerContainerOwnership]:
        ownership = self._new_ownership(operation_id, component)
        return (
            self.command(
                repository,
                argv,
                writable_workspace=writable_workspace,
                interactive=interactive,
                ownership=ownership,
                additional_mounts=additional_mounts,
                git_metadata=git_metadata,
                writable_git_metadata=writable_git_metadata,
            ),
            ownership,
        )

    def popen(
        self,
        repository: Path,
        argv: Sequence[str],
        *,
        writable_workspace: bool,
        interactive: bool = False,
        operation_id: str | None = None,
        component: str = "sandbox",
        additional_mounts: Sequence[DockerVolumeMount] = (),
        git_metadata: DockerGitMetadataMount | None = None,
        writable_git_metadata: bool = False,
        **kwargs: Any,
    ) -> subprocess.Popen[Any]:
        command, ownership = self.prepare(
            repository,
            argv,
            writable_workspace=writable_workspace,
            interactive=interactive,
            operation_id=operation_id,
            component=component,
            additional_mounts=additional_mounts,
            git_metadata=git_metadata,
            writable_git_metadata=writable_git_metadata,
        )
        try:
            process = subprocess.Popen(command, **kwargs)
        except OSError as exc:
            self.cleanup(ownership, reason="docker-cli-start-failed")
            raise DockerSandboxError(
                "Docker 격리 컨테이너를 시작하지 못했습니다. Docker Desktop 상태를 확인해 주세요."
            ) from exc
        setattr(process, _PROCESS_OWNER_ATTR, ownership)
        self._emit("Docker 컨테이너 실행을 시작했습니다.", ownership, event="DOCKER_CONTAINER_START")
        return process

    def run(
        self,
        repository: Path,
        argv: Sequence[str],
        *,
        writable_workspace: bool,
        timeout: float | None = None,
        cancelled: Callable[[], bool] | None = None,
        operation_id: str | None = None,
        component: str = "sandbox",
        additional_mounts: Sequence[DockerVolumeMount] = (),
        git_metadata: DockerGitMetadataMount | None = None,
        writable_git_metadata: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        limit = timeout if timeout is not None else self.timeout_seconds
        if limit <= 0:
            raise ValueError("Docker sandbox timeout must be positive")
        process = self.popen(
            repository,
            argv,
            writable_workspace=writable_workspace,
            operation_id=operation_id,
            component=component,
            additional_mounts=additional_mounts,
            git_metadata=git_metadata,
            writable_git_metadata=writable_git_metadata,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **isolated_process_options(),
        )
        try:
            process_tree = ProcessTree(process)
        except BaseException:
            self.cleanup_process(process, reason="process-tree-initialization-failed")
            raise
        started = time.monotonic()
        stdout = ""
        stderr = ""
        reason = "completed"
        try:
            while True:
                try:
                    stdout, stderr = process.communicate(timeout=min(0.25, limit))
                    break
                except subprocess.TimeoutExpired:
                    if cancelled is not None and cancelled():
                        reason = "cancelled"
                        self.note_process_termination(process, reason=reason)
                        process_tree.terminate()
                        raise DockerSandboxCancelled("사용자가 Docker 실행을 중지했습니다.")
                    if time.monotonic() - started >= limit:
                        reason = "timeout"
                        self.note_process_termination(process, reason=reason)
                        process_tree.terminate()
                        raise DockerSandboxTimeout("Docker 격리 컨테이너 실행 시간이 초과되었습니다.")
        except BaseException:
            if process.poll() is None:
                self.note_process_termination(process, reason=reason)
                process_tree.terminate()
            raise
        finally:
            process_tree.close()
            self.cleanup_process(process, reason=reason)
        return subprocess.CompletedProcess(list(process.args), int(process.returncode), stdout, stderr)

    def with_image(self, image: str) -> "DockerSandbox":
        """Keep the ownership and cleanup identity while selecting a pinned profile image."""
        return replace(self, image=image)

    def cleanup_process(self, process: object, *, reason: str) -> bool:
        ownership = self._ownership_for(process)
        if ownership is None:
            return False
        try:
            return self.cleanup(ownership, reason=reason)
        finally:
            try:
                delattr(process, _PROCESS_OWNER_ATTR)
            except AttributeError:
                pass

    def note_process_termination(self, process: object, *, reason: str) -> None:
        """Record a requested Docker CLI-tree stop without exposing command input."""
        ownership = self._ownership_for(process)
        if ownership is not None:
            self._emit(
                "Docker CLI 프로세스 트리 종료를 요청했습니다.",
                ownership,
                event="DOCKER_CLI_TERMINATE_REQUESTED",
                reason=reason,
            )

    def cleanup(self, ownership: DockerContainerOwnership, *, reason: str) -> bool:
        """Idempotently remove only a container that exactly matches labels."""
        removed = False
        errors: list[str] = []
        try:
            for candidate in self._cleanup_candidates(ownership):
                labels = self._container_labels(candidate)
                if labels is None:
                    continue
                if not ownership.matches(labels):
                    self._emit(
                        "소유권이 일치하지 않아 Docker 컨테이너 정리를 건너뛰었습니다.",
                        ownership,
                        event="DOCKER_CONTAINER_CLEANUP_SKIPPED",
                        container=candidate,
                    )
                    continue
                result = self._docker(["rm", "-f", candidate], timeout=8)
                if result.returncode == 0:
                    removed = True
                    self._emit(
                        "소유 Docker 컨테이너를 정리했습니다.",
                        ownership,
                        event="DOCKER_CONTAINER_CLEANED",
                        container=candidate,
                        reason=reason,
                    )
                else:
                    errors.append(self._docker_detail(result))
        except DockerSandboxError as exc:
            errors.append(str(exc))
        finally:
            try:
                ownership.cidfile.unlink(missing_ok=True)
            except OSError as exc:
                errors.append(f"cidfile 삭제 실패: {type(exc).__name__}")
        if errors:
            self._emit(
                "Docker 컨테이너 정리에 실패했습니다.",
                ownership,
                event="DOCKER_CONTAINER_CLEANUP_FAILED",
                reason=reason,
                error="; ".join(errors)[:500],
            )
        return removed

    def cleanup_stale(
        self, *, active_gateway_instances: Iterable[str] = ()
    ) -> tuple[str, ...]:
        """Remove only stale containers for this installation; never use prune."""
        active = set(active_gateway_instances)
        active.add(self.gateway_instance_id)
        listed = self._docker(
            [
                "ps", "--all", "--filter", f"label={MANAGED_LABEL}=true",
                "--format", "{{.ID}}",
            ],
            timeout=8,
        )
        if listed.returncode != 0:
            raise DockerSandboxError(
                "AI-Agents Docker 컨테이너 목록을 확인하지 못했습니다: "
                + self._docker_detail(listed)
            )
        removed: list[str] = []
        for candidate in (item.strip() for item in listed.stdout.splitlines() if item.strip()):
            labels = self._container_labels(candidate)
            if labels is None:
                continue
            if (
                labels.get(MANAGED_LABEL) != "true"
                or labels.get(INSTALLATION_LABEL) != self.installation_id
                or not labels.get(OPERATION_LABEL)
                or not labels.get(COMPONENT_LABEL)
                or not labels.get(GATEWAY_INSTANCE_LABEL)
                or not labels.get(GATEWAY_PID_LABEL)
                or not labels.get(GATEWAY_STARTED_LABEL)
                or labels[GATEWAY_INSTANCE_LABEL] in active
                or self._gateway_owner_is_alive(labels)
            ):
                continue
            result = self._docker(["rm", "-f", candidate], timeout=8)
            if result.returncode == 0:
                removed.append(candidate)
                self._emit(
                    "이전 Gateway 실행의 stale Docker 컨테이너를 정리했습니다.",
                    None,
                    event="DOCKER_STALE_CONTAINER_CLEANED",
                    container=candidate,
                    operation_id=labels[OPERATION_LABEL],
                    component=labels[COMPONENT_LABEL],
                    gateway_instance=labels[GATEWAY_INSTANCE_LABEL],
                )
            else:
                self._emit(
                    "stale Docker 컨테이너 정리에 실패했습니다.",
                    None,
                    event="DOCKER_STALE_CONTAINER_CLEANUP_FAILED",
                    container=candidate,
                    operation_id=labels[OPERATION_LABEL],
                    component=labels[COMPONENT_LABEL],
                    gateway_instance=labels[GATEWAY_INSTANCE_LABEL],
                    error=self._docker_detail(result),
                )
        return tuple(removed)

    def _new_ownership(self, operation_id: str | None, component: str) -> DockerContainerOwnership:
        operation = operation_id or f"sandbox-{uuid.uuid4().hex}"
        self._validate_owner_part(operation, "operation")
        self._validate_owner_part(component, "component")
        suffix = uuid.uuid4().hex[:10]
        name = "ai-agents-" + "-".join((
            self.installation_id[:8], self._name_part(component, 12),
            self._name_part(operation, 16), suffix,
        ))
        return DockerContainerOwnership(
            operation_id=operation,
            component=component,
            gateway_instance_id=self.gateway_instance_id,
            gateway_process_id=self.gateway_process_id,
            gateway_process_started_at=self.gateway_process_started_at,
            installation_id=self.installation_id,
            container_name=name,
            cidfile=self._new_cidfile(name),
        )

    def _new_cidfile(self, name: str) -> Path:
        directory = self.cidfile_directory or (_INSTALLATION_ROOT / "data" / "docker-cids")
        directory.mkdir(parents=True, exist_ok=True)
        descriptor, raw_path = tempfile.mkstemp(
            prefix=f"{name[:40]}-", suffix=".cid", dir=str(directory)
        )
        os.close(descriptor)
        path = Path(raw_path)
        # Docker owns creation and rejects a cidfile that already exists.
        path.unlink(missing_ok=True)
        return path

    @staticmethod
    def _validate_owner_part(value: str, name: str) -> None:
        if not isinstance(value, str) or not _SAFE_OWNER_PART.fullmatch(value):
            raise ValueError(f"Docker {name} identifier is invalid")

    @staticmethod
    def _name_part(value: str, maximum: int) -> str:
        rendered = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip(".-")
        return (rendered or "unknown")[:maximum]

    @staticmethod
    def _ownership_for(process: object) -> DockerContainerOwnership | None:
        value = getattr(process, _PROCESS_OWNER_ATTR, None)
        return value if isinstance(value, DockerContainerOwnership) else None

    def _cleanup_candidates(self, ownership: DockerContainerOwnership) -> tuple[str, ...]:
        candidates = [ownership.container_name]
        try:
            raw = ownership.cidfile.read_text(encoding="utf-8").strip()
        except OSError:
            raw = ""
        if raw:
            candidates.insert(0, raw.splitlines()[0].strip())
        return tuple(dict.fromkeys(item for item in candidates if item))

    def _container_labels(self, container: str) -> dict[str, str] | None:
        result = self._docker(["inspect", "--format", "{{json .Config.Labels}}", container], timeout=5)
        if result.returncode != 0:
            return None
        try:
            value = json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            return None
        if not isinstance(value, dict):
            return None
        return {str(key): str(item) for key, item in value.items()}

    @staticmethod
    def _gateway_owner_is_alive(labels: dict[str, str]) -> bool:
        """Fail closed: unverified ownership is never safe to delete."""
        try:
            import psutil
        except ImportError:
            return True
        try:
            process_id = int(labels[GATEWAY_PID_LABEL])
            started_at = float(labels[GATEWAY_STARTED_LABEL])
            if process_id < 1 or not psutil.pid_exists(process_id):
                return False
            process = psutil.Process(process_id)
            return abs(float(process.create_time()) - started_at) < 0.01
        except psutil.NoSuchProcess:
            return False
        except Exception:
            return True

    @staticmethod
    def _docker(arguments: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["docker", *arguments], text=True, encoding="utf-8", errors="replace",
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=timeout, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DockerSandboxError("Docker Desktop 상태를 확인할 수 없습니다.") from exc

    @staticmethod
    def _docker_detail(result: subprocess.CompletedProcess[str]) -> str:
        return (result.stderr.strip() or result.stdout.strip() or "출력 없음")[:500]

    def _emit(
        self,
        message: str,
        ownership: DockerContainerOwnership | None,
        *,
        event: str,
        **extra: str,
    ) -> None:
        if self.event_sink is None:
            return
        details = {
            "event": event,
            "operation_id": ownership.operation_id if ownership else extra.pop("operation_id", ""),
            "component": ownership.component if ownership else extra.pop("component", ""),
            "gateway_instance": ownership.gateway_instance_id if ownership else extra.pop("gateway_instance", self.gateway_instance_id),
            "container": extra.pop("container", ownership.container_name if ownership else ""),
            **extra,
        }
        self.event_sink(message + " " + " ".join(f"{key}={value}" for key, value in details.items() if value))
