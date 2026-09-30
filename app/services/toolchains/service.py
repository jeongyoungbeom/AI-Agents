from __future__ import annotations

import json
import re
import shlex
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from app.services.sandbox import (
    DockerSandbox,
    DockerSandboxCancelled,
    DockerSandboxError,
    DockerSandboxTimeout,
    DockerVolumeMount,
)
from app.services.git.repository import GitRepository
from app.services.verification import PreparedVerificationCommand, SafeVerificationPolicy


_PROFILE_NAME = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
_VERSION = re.compile(r"(?<!\d)(\d+(?:\.\d+){0,3})(?!\d)")
_CACHE_MARKER = re.compile(r"^/[a-zA-Z0-9_.-]+(?:/[a-zA-Z0-9_.-]+)*$")
_PINNED_IMAGE = re.compile(r"^[a-z0-9][a-z0-9./:_-]*@sha256:[0-9a-f]{64}$")
_STACK_BY_EXECUTABLE = {
    "gradle": "gradle", "gradlew": "gradle", "gradlew.bat": "gradle", "java": "gradle",
    "node": "node", "npm": "node", "npx": "node", "pnpm": "node", "yarn": "node",
    "python": "python", "python3": "python", "py": "python", "pytest": "python",
}
_DEPENDENCY_KEYS = frozenset({"dependencies", "devDependencies", "optionalDependencies", "peerDependencies"})


class ToolchainPreflightError(RuntimeError):
    """A safe execution environment could not be proved before approval."""

    def __init__(self, category: str, detail: str) -> None:
        self.category = category
        self.detail = detail
        super().__init__(f"[{category}] {detail}")


@dataclass(frozen=True)
class _CachePolicy:
    mode: str
    marker: str = ""
    volume: DockerVolumeMount | None = None

    @property
    def mounts(self) -> tuple[DockerVolumeMount, ...]:
        return (self.volume,) if self.volume is not None else ()


@dataclass(frozen=True)
class _Profile:
    profile_id: str
    stacks: frozenset[str]
    image: str
    executables: dict[str, str]
    cache: _CachePolicy
    default: bool = False


@dataclass(frozen=True)
class ToolchainEnvironment:
    """The exact profile that preflight proved and verification must reuse."""

    profile_id: str
    image: str
    cache_mounts: tuple[DockerVolumeMount, ...]
    stacks: tuple[str, ...]

    def sandbox_for(self, base: DockerSandbox) -> DockerSandbox:
        return base.with_image(self.image)

    def to_dict(self) -> dict:
        return {
            "profile": self.profile_id,
            "image": self.image,
            "stacks": list(self.stacks),
            "cache_mounts": [
                {"name": item.name, "target": item.target, "read_only": item.read_only}
                for item in self.cache_mounts
            ],
        }


@dataclass(frozen=True)
class _RepositorySnapshot:
    repository: GitRepository
    head: str
    entries: dict[str, str]
    service: "ToolchainService"
    operation_id: str | None
    cancelled: Callable[[], bool] | None

    def has(self, path: str) -> bool:
        return path in self.entries

    def mode(self, path: str) -> str | None:
        if path in self.entries:
            return self.entries[path]
        return self.service._snapshot_path_mode(
            self.repository, self.head, path, self.operation_id, self.cancelled
        )

    def read_optional(self, path: str, *, maximum_bytes: int) -> str | None:
        if self.mode(path) != "100644" and self.mode(path) != "100755":
            return None
        return self.service._snapshot_read(
            self.repository,
            self.head,
            path,
            maximum_bytes,
            self.operation_id,
            self.cancelled,
        )


class ToolchainService:
    """Detect a repository stack and prove the selected container can run it.

    Repository metadata is untrusted input. It can select only a profile name
    defined by the gateway-owned catalog; it cannot provide Docker syntax,
    image names, cache locations, or executable argv.
    """

    def __init__(self, profiles: dict[str, _Profile], *, sandbox: DockerSandbox | None = None, policy: SafeVerificationPolicy | None = None) -> None:
        if not profiles:
            raise ValueError("At least one toolchain profile is required")
        self._profiles = dict(profiles)
        self._sandbox = sandbox or DockerSandbox()
        self._policy = policy or SafeVerificationPolicy()

    @classmethod
    def load(cls, path: Path, *, sandbox: DockerSandbox | None = None, policy: SafeVerificationPolicy | None = None) -> "ToolchainService":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Toolchain catalog could not be read: {path}") from exc
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise ValueError("Toolchain catalog schema_version must be 1")
        values = raw.get("profiles")
        if not isinstance(values, dict):
            raise ValueError("Toolchain catalog profiles must be an object")
        return cls({identifier: cls._parse_profile(identifier, value) for identifier, value in values.items()}, sandbox=sandbox, policy=policy)

    @staticmethod
    def _parse_profile(identifier: object, raw: object) -> _Profile:
        if not isinstance(identifier, str) or not _PROFILE_NAME.fullmatch(identifier):
            raise ValueError("Toolchain profile identifier is invalid")
        if not isinstance(raw, dict):
            raise ValueError(f"Toolchain profile {identifier} must be an object")
        if set(raw) - {"stacks", "image", "executables", "cache", "default"}:
            raise ValueError(f"Toolchain profile {identifier} contains unsupported settings")
        stacks, image, executables, default = raw.get("stacks"), raw.get("image"), raw.get("executables"), raw.get("default", False)
        if (not isinstance(stacks, list) or not stacks or len(set(stacks)) != len(stacks) or not all(isinstance(item, str) and item in {"gradle", "node", "python"} for item in stacks) or not isinstance(image, str) or not _PINNED_IMAGE.fullmatch(image) or not isinstance(executables, dict) or not executables or not isinstance(default, bool)):
            raise ValueError(f"Toolchain profile {identifier} is incomplete or unsafe")
        normalized: dict[str, str] = {}
        for executable, version in executables.items():
            if (not isinstance(executable, str) or executable.casefold() not in _STACK_BY_EXECUTABLE or not isinstance(version, str) or not _VERSION.fullmatch(version)):
                raise ValueError(f"Toolchain profile {identifier} executable requirement is invalid")
            normalized[executable.casefold()] = version
        return _Profile(identifier, frozenset(stacks), image, normalized, ToolchainService._parse_cache(identifier, raw.get("cache", {"mode": "none"})), default)

    @staticmethod
    def _parse_cache(identifier: str, raw: object) -> _CachePolicy:
        if not isinstance(raw, dict) or set(raw) - {"mode", "marker", "volume", "target"}:
            raise ValueError(f"Toolchain profile {identifier} cache policy is invalid")
        mode = raw.get("mode")
        if mode == "none":
            if set(raw) != {"mode"}:
                raise ValueError(f"Toolchain profile {identifier} cache policy is invalid")
            return _CachePolicy("none")
        marker = raw.get("marker")
        if not isinstance(marker, str) or not _CACHE_MARKER.fullmatch(marker):
            raise ValueError(f"Toolchain profile {identifier} cache marker is invalid")
        if mode == "image" and set(raw) == {"mode", "marker"}:
            return _CachePolicy("image", marker=marker)
        if mode == "volume" and set(raw) == {"mode", "marker", "volume", "target"}:
            volume, target = raw["volume"], raw["target"]
            if not isinstance(volume, str) or not isinstance(target, str):
                raise ValueError(f"Toolchain profile {identifier} cache volume is invalid")
            return _CachePolicy("volume", marker=marker, volume=DockerVolumeMount(volume, target))
        raise ValueError(f"Toolchain profile {identifier} cache policy is invalid")

    def preflight(self, repository: GitRepository, commands: Iterable[str], *, operation_id: str | None = None, cancelled: Callable[[], bool] | None = None) -> ToolchainEnvironment:
        snapshot = self._snapshot(repository, operation_id=operation_id, cancelled=cancelled)
        prepared = tuple(self._policy.prepare(command) for command in commands)
        if not prepared:
            raise ToolchainPreflightError("command", "검증 명령이 없어 실행 환경을 결정할 수 없습니다.")
        stacks = self._detected_stacks(snapshot)
        override = self._read_override(snapshot)
        command_stacks = self._command_stacks(prepared)
        profile = self._select_profile(stacks, command_stacks, override)
        self._validate_wrapper(snapshot, prepared, profile)
        environment = ToolchainEnvironment(profile.profile_id, profile.image, profile.cache.mounts, tuple(sorted(stacks or command_stacks)))
        self._assert_snapshot_current(snapshot)
        self._probe(profile, repository, prepared, operation_id=operation_id, cancelled=cancelled)
        if self._requires_dependency_cache(snapshot, stacks, prepared):
            self._probe_cache(profile, repository, environment, operation_id=operation_id, cancelled=cancelled)
        return environment

    def _snapshot(self, repository: GitRepository, *, operation_id: str | None, cancelled: Callable[[], bool] | None) -> _RepositorySnapshot:
        head_result = self._snapshot_git(repository, ["rev-parse", "HEAD"], operation_id, cancelled)
        head = head_result.stdout.strip()
        if not re.fullmatch(r"[0-9a-f]{40}", head):
            raise ToolchainPreflightError("snapshot", "저장소 HEAD를 고정할 수 없습니다.")
        tree = self._snapshot_git(repository, ["ls-tree", "-z", head], operation_id, cancelled)
        entries: dict[str, str] = {}
        for record in tree.stdout.split("\0"):
            if not record:
                continue
            try:
                metadata, name = record.split("\t", 1)
                mode, object_type, _object_id = metadata.split(" ", 2)
            except ValueError as exc:
                raise ToolchainPreflightError("snapshot", "저장소 root tree를 해석할 수 없습니다.") from exc
            if object_type in {"blob", "tree"} and "/" not in name:
                entries[name] = mode
        return _RepositorySnapshot(repository, head, entries, self, operation_id, cancelled)

    def _snapshot_path_mode(self, repository: GitRepository, head: str, path: str, operation_id: str | None, cancelled: Callable[[], bool] | None) -> str | None:
        result = self._snapshot_git(repository, ["ls-tree", "-z", head, "--", path], operation_id, cancelled)
        record = next((item for item in result.stdout.split("\0") if item), "")
        if not record:
            return None
        try:
            metadata, rendered_path = record.split("\t", 1)
            mode, object_type, _object_id = metadata.split(" ", 2)
        except ValueError as exc:
            raise ToolchainPreflightError("snapshot", "저장소 파일 메타데이터를 해석할 수 없습니다.") from exc
        return mode if object_type == "blob" and rendered_path == path else None

    def _snapshot_read(self, repository: GitRepository, head: str, path: str, maximum_bytes: int, operation_id: str | None, cancelled: Callable[[], bool] | None) -> str:
        result = self._snapshot_git(repository, ["show", "--no-textconv", f"{head}:{path}"], operation_id, cancelled)
        value = result.stdout
        if len(value.encode("utf-8")) > maximum_bytes:
            raise ToolchainPreflightError("snapshot", f"{path} 파일이 {maximum_bytes} byte를 초과합니다.")
        return value

    def _assert_snapshot_current(self, snapshot: _RepositorySnapshot) -> None:
        result = self._snapshot_git(snapshot.repository, ["rev-parse", "HEAD"], snapshot.operation_id, snapshot.cancelled)
        if result.stdout.strip() != snapshot.head:
            raise ToolchainPreflightError("snapshot", "실행 환경 확인 중 저장소 HEAD가 바뀌었습니다. 새 snapshot으로 다시 계획을 확인해 주세요.")

    def _snapshot_git(self, repository: GitRepository, arguments: list[str], operation_id: str | None, cancelled: Callable[[], bool] | None):
        git_prefix = ["git"]
        if repository.git_metadata is not None:
            git_prefix.extend(
                (
                    f"--git-dir=/ai-agents-gitdir/{repository.git_metadata.git_dir_relative}",
                    "--work-tree=/workspace",
                )
            )
        try:
            result = self._sandbox.run(repository.path, [*git_prefix, "-c", "core.pager=cat", *arguments], writable_workspace=False, timeout=60, cancelled=cancelled, operation_id=operation_id, component="toolchain-snapshot", git_metadata=repository.git_metadata, writable_git_metadata=False)
        except DockerSandboxCancelled:
            raise
        except DockerSandboxTimeout as exc:
            raise ToolchainPreflightError("snapshot", "고정 저장소 snapshot 확인 시간이 초과되었습니다.") from exc
        except DockerSandboxError as exc:
            raise ToolchainPreflightError("snapshot", "고정 저장소 snapshot 확인용 Docker 컨테이너를 시작하지 못했습니다.") from exc
        if result.returncode != 0:
            detail = (result.stderr.strip() or result.stdout.strip() or "출력 없음")[-800:]
            raise ToolchainPreflightError("snapshot", f"고정 저장소 snapshot을 읽지 못했습니다: {detail}")
        return result

    def _detected_stacks(self, snapshot: _RepositorySnapshot) -> set[str]:
        names = {item.casefold() for item in snapshot.entries}
        stacks: set[str] = set()
        if names & {"build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts", "gradlew", "gradlew.bat"}:
            stacks.add("gradle")
        if names & {"package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock"}:
            stacks.add("node")
        if names & {"pyproject.toml", "requirements.txt", "requirements-dev.txt", "setup.py", "setup.cfg", "tox.ini"}:
            stacks.add("python")
        return stacks

    def _read_override(self, snapshot: _RepositorySnapshot) -> str | None:
        raw_text = snapshot.read_optional(".ai-agents/toolchain.json", maximum_bytes=4096)
        if raw_text is None:
            return None
        try:
            raw = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ToolchainPreflightError("override", "toolchain override JSON을 읽을 수 없습니다.") from exc
        if not isinstance(raw, dict) or set(raw) != {"schema_version", "profile"} or raw.get("schema_version") != 1:
            raise ToolchainPreflightError("override", "override는 schema_version과 profile만 지정할 수 있습니다.")
        profile = raw.get("profile")
        if not isinstance(profile, str) or not _PROFILE_NAME.fullmatch(profile):
            raise ToolchainPreflightError("override", "override profile 이름이 올바르지 않습니다.")
        return profile

    def _command_stacks(self, commands: tuple[PreparedVerificationCommand, ...]) -> set[str]:
        stacks: set[str] = set()
        for command in commands:
            executable = Path(command.argv[0].replace("\\", "/")).name.casefold()
            if executable == "py":
                raise ToolchainPreflightError("wrapper", "Windows Python launcher 'py'는 Linux 컨테이너에서 실행할 수 없습니다. python 또는 python3을 사용해 주세요.")
            stack = _STACK_BY_EXECUTABLE.get(executable)
            if stack is None:
                raise ToolchainPreflightError("tool", f"지원하지 않는 검증 도구입니다: {executable}")
            stacks.add(stack)
        return stacks

    def _select_profile(self, stacks: set[str], command_stacks: set[str], override: str | None) -> _Profile:
        effective = stacks or command_stacks
        if len(effective) > 1 and override is None:
            raise ToolchainPreflightError("selection", "여러 기술 스택이 감지됐습니다. .ai-agents/toolchain.json에서 승인된 profile을 명시해 주세요.")
        if override is not None:
            profile = self._profiles.get(override)
            if profile is None:
                raise ToolchainPreflightError("override", f"정의되지 않은 toolchain profile입니다: {override}")
            if not effective.issubset(profile.stacks):
                raise ToolchainPreflightError("override", f"profile '{override}'이(가) 감지된 stack 또는 검증 명령과 호환되지 않습니다.")
            return profile
        candidates = [profile for profile in self._profiles.values() if effective.issubset(profile.stacks)]
        defaults = [profile for profile in candidates if profile.default]
        if len(defaults) == 1:
            return defaults[0]
        if len(candidates) != 1:
            raise ToolchainPreflightError("selection", "감지된 stack에 대응하는 단일 toolchain profile을 찾지 못했습니다.")
        return candidates[0]

    def _validate_wrapper(self, snapshot: _RepositorySnapshot, commands: tuple[PreparedVerificationCommand, ...], profile: _Profile) -> None:
        executable_names = {Path(command.argv[0].replace("\\", "/")).name.casefold() for command in commands}
        if "gradlew.bat" in executable_names:
            raise ToolchainPreflightError("wrapper", "gradlew.bat는 Linux 컨테이너에서 실행할 수 없습니다. gradlew를 제공하거나 검증 명령을 gradle로 바꿔 주세요.")
        if "gradlew" in executable_names:
            if snapshot.mode("gradlew") not in {"100644", "100755"}:
                suffix = "gradlew.bat만 있습니다" if snapshot.mode("gradlew.bat") in {"100644", "100755"} else "gradlew가 없습니다"
                raise ToolchainPreflightError("wrapper", f"Linux Gradle wrapper를 사용할 수 없습니다: {suffix}.")
            if "java" not in profile.executables:
                raise ToolchainPreflightError("tool", "선택된 profile에 Gradle wrapper용 Java가 없습니다.")

    def _probe(self, profile: _Profile, repository: GitRepository, commands: tuple[PreparedVerificationCommand, ...], *, operation_id: str | None, cancelled: Callable[[], bool] | None) -> None:
        sandbox = self._sandbox.with_image(profile.image)
        for executable in self._required_executables(commands):
            if executable == "gradlew":
                argv, expected, label = ["./gradlew", "--version"], profile.executables["gradle"], "Gradle wrapper"
            else:
                expected = profile.executables.get(executable)
                if expected is None:
                    raise ToolchainPreflightError("tool", f"선택된 profile에 필요한 실행 파일이 없습니다: {executable}")
                argv, label = [executable, "--version"], executable
            try:
                result = sandbox.run(repository.path, argv, writable_workspace=False, timeout=120, cancelled=cancelled, operation_id=operation_id, component="toolchain-preflight", additional_mounts=profile.cache.mounts, git_metadata=repository.git_metadata, writable_git_metadata=False)
            except DockerSandboxCancelled:
                raise
            except DockerSandboxTimeout as exc:
                raise ToolchainPreflightError("tool", f"{label} 버전 확인 시간이 초과되었습니다.") from exc
            except DockerSandboxError as exc:
                raise ToolchainPreflightError("tool", f"{label} 확인용 Docker 컨테이너를 시작하지 못했습니다.") from exc
            detail = (result.stdout + "\n" + result.stderr).strip()
            if result.returncode != 0:
                category = "dependency-cache" if self._looks_like_dependency_failure(detail) else "tool"
                raise ToolchainPreflightError(category, f"{label}을(를) 실행할 수 없습니다: {detail[-800:] or '출력 없음'}")
            actual = self._version(detail)
            if actual is None or not self._at_least(actual, expected):
                raise ToolchainPreflightError("version", f"{label} 버전이 부족합니다. 필요: {expected}, 확인 결과: {actual or '알 수 없음'}")

    @staticmethod
    def _required_executables(commands: tuple[PreparedVerificationCommand, ...]) -> tuple[str, ...]:
        values: list[str] = []
        for command in commands:
            executable = Path(command.argv[0].replace("\\", "/")).name.casefold()
            values.extend(("java", "gradlew") if executable == "gradlew" else (("java", "gradle") if executable == "gradle" else (executable,)))
        return tuple(dict.fromkeys(values))

    def _probe_cache(self, profile: _Profile, repository: GitRepository, environment: ToolchainEnvironment, *, operation_id: str | None, cancelled: Callable[[], bool] | None) -> None:
        if profile.cache.mode == "none":
            raise ToolchainPreflightError("dependency-cache", "이 프로젝트는 외부 dependency가 필요하지만 선택된 profile에는 사전 구축 image 또는 승인된 읽기 전용 cache가 없습니다.")
        try:
            result = environment.sandbox_for(self._sandbox).run(repository.path, ["/bin/sh", "-c", f"test -r {shlex.quote(profile.cache.marker)}"], writable_workspace=False, timeout=30, cancelled=cancelled, operation_id=operation_id, component="toolchain-cache-preflight", additional_mounts=environment.cache_mounts, git_metadata=repository.git_metadata, writable_git_metadata=False)
        except DockerSandboxCancelled:
            raise
        except (DockerSandboxError, DockerSandboxTimeout) as exc:
            raise ToolchainPreflightError("dependency-cache", "승인된 dependency cache를 확인할 수 없습니다.") from exc
        if result.returncode != 0:
            raise ToolchainPreflightError("dependency-cache", "승인된 dependency cache가 준비되지 않았습니다.")

    def _requires_dependency_cache(self, snapshot: _RepositorySnapshot, stacks: set[str], commands: tuple[PreparedVerificationCommand, ...]) -> bool:
        if "gradle" in stacks:
            names = {Path(command.argv[0].replace("\\", "/")).name.casefold() for command in commands}
            return "gradlew" in names or self._gradle_has_dependencies(snapshot)
        if "node" in stacks:
            return self._package_has_dependencies(snapshot)
        if "python" in stacks:
            return self._python_has_dependencies(snapshot)
        return False

    def _package_has_dependencies(self, snapshot: _RepositorySnapshot) -> bool:
        value = snapshot.read_optional("package.json", maximum_bytes=65536)
        if value is None:
            return False
        try:
            raw = json.loads(value)
        except json.JSONDecodeError:
            return True
        return isinstance(raw, dict) and any(isinstance(raw.get(key), dict) and raw[key] for key in _DEPENDENCY_KEYS)

    def _python_has_dependencies(self, snapshot: _RepositorySnapshot) -> bool:
        for name in ("requirements.txt", "requirements-dev.txt"):
            value = snapshot.read_optional(name, maximum_bytes=65536)
            if value is not None and any(line.strip() and not line.lstrip().startswith("#") for line in value.splitlines()):
                return True
        value = snapshot.read_optional("pyproject.toml", maximum_bytes=65536)
        if value is not None:
            return "dependencies" in value
        return False

    def _gradle_has_dependencies(self, snapshot: _RepositorySnapshot) -> bool:
        for name in ("build.gradle", "build.gradle.kts"):
            value = snapshot.read_optional(name, maximum_bytes=65536)
            if value is not None and "dependencies" in value:
                return True
        return False

    @staticmethod
    def _version(value: str) -> tuple[int, ...] | None:
        match = _VERSION.search(value)
        return tuple(int(item) for item in match.group(1).split(".")) if match else None

    @staticmethod
    def _at_least(actual: tuple[int, ...], minimum: str) -> bool:
        required = tuple(int(item) for item in minimum.split("."))
        width = max(len(actual), len(required))
        return actual + (0,) * (width - len(actual)) >= required + (0,) * (width - len(required))

    @staticmethod
    def _looks_like_dependency_failure(value: str) -> bool:
        lowered = value.casefold()
        return any(item in lowered for item in ("offline mode", "no cached version", "could not resolve", "cannot find module", "no module named", "dependency"))
