from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.contracts import RoleId
from app.services.sandbox import DEFAULT_SECURE_DOCKER_IMAGE


SUPPORTED_REASONING_LEVELS = {
    "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"
}


@dataclass(frozen=True)
class HermesModelSettings:
    model: str
    reasoning: str

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("Hermes model cannot be blank")
        if self.reasoning not in SUPPORTED_REASONING_LEVELS:
            raise ValueError(f"unsupported Hermes reasoning level: {self.reasoning}")


@dataclass(frozen=True)
class HermesRoleSettings:
    profile: str
    model: str
    reasoning: str
    toolsets: str
    max_turns: int

    def __post_init__(self) -> None:
        HermesModelSettings(self.model, self.reasoning)
        if not self.profile.strip() or not self.toolsets.strip():
            raise ValueError("Hermes profile and toolsets cannot be blank")
        if self.max_turns < 1:
            raise ValueError("Hermes max_turns must be positive")


@dataclass(frozen=True)
class DockerIsolationSettings:
    image: str = DEFAULT_SECURE_DOCKER_IMAGE
    network: bool = False
    profile_asset_mounts: bool = False

    def __post_init__(self) -> None:
        if "@sha256:" not in self.image:
            raise ValueError("Docker 격리 이미지는 변경 불가능한 digest로 고정해야 합니다.")
        if self.network:
            raise ValueError("필수 Docker 격리는 네트워크를 허용할 수 없습니다.")
        if self.profile_asset_mounts:
            raise ValueError("필수 Docker 격리는 프로필 캐시·스킬 마운트를 허용할 수 없습니다.")


@dataclass(frozen=True)
class HermesSettings:
    root: Path
    executable: Path
    home: Path
    provider: str
    timeout_seconds: float
    poll_seconds: float
    conversation: HermesModelSettings
    planning: HermesModelSettings
    roles: dict[RoleId, HermesRoleSettings]
    docker_required: bool = False
    isolation: DockerIsolationSettings = DockerIsolationSettings()

    @classmethod
    def load(cls, root: Path) -> "HermesSettings":
        root = root.resolve()
        path = root / "config" / "agents.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"Hermes 설정 파일이 없습니다: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Hermes 설정 JSON이 올바르지 않습니다: {path}") from exc
        if not isinstance(raw, dict):
            raise ValueError("Hermes 설정 최상위 값은 객체여야 합니다.")
        roles_raw = raw.get("roles")
        if not isinstance(roles_raw, dict):
            raise ValueError("Hermes roles 설정이 필요합니다.")
        conversation = cls._model_settings(raw, "conversation")
        planning = cls._model_settings(raw, "planning")
        roles: dict[RoleId, HermesRoleSettings] = {}
        for role_id in RoleId:
            value = roles_raw.get(role_id.value)
            if not isinstance(value, dict):
                raise ValueError(f"Hermes 역할 설정이 없습니다: {role_id.value}")
            roles[role_id] = HermesRoleSettings(
                profile=str(value.get("profile", "")),
                model=str(value.get("model", "")),
                reasoning=str(value.get("reasoning", "xhigh")),
                toolsets=str(value.get("toolsets", "")),
                max_turns=int(value.get("max_turns", 120)),
            )
        executable = cls._under_root(root, str(raw.get("hermes_executable", "")))
        home = cls._under_root(root, str(raw.get("hermes_home", "")))
        provider = str(raw.get("provider", "openai-codex")).strip()
        isolation = raw.get("execution_isolation", {})
        if not isinstance(isolation, dict):
            raise ValueError("execution_isolation 설정은 객체여야 합니다.")
        isolation_backend = str(isolation.get("backend", "local")).strip().casefold()
        docker_required = bool(isolation.get("required", False))
        if isolation_backend not in {"local", "docker"}:
            raise ValueError("execution_isolation.backend는 local 또는 docker여야 합니다.")
        if docker_required and isolation_backend != "docker":
            raise ValueError("필수 실행 격리는 docker backend여야 합니다.")
        docker_isolation = DockerIsolationSettings(
            image=str(isolation.get("image", DEFAULT_SECURE_DOCKER_IMAGE)).strip(),
            network=bool(isolation.get("network", False)),
            profile_asset_mounts=bool(isolation.get("profile_asset_mounts", False)),
        )
        timeout_seconds = float(raw.get("timeout_seconds", 3600))
        poll_seconds = float(raw.get("poll_seconds", 0.5))
        if not provider or timeout_seconds <= 0 or poll_seconds <= 0:
            raise ValueError("Hermes provider와 시간 제한 설정을 확인해 주세요.")
        return cls(
            root=root,
            executable=executable,
            home=home,
            provider=provider,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
            conversation=conversation,
            planning=planning,
            roles=roles,
            docker_required=docker_required,
            isolation=docker_isolation,
        )

    @staticmethod
    def _model_settings(raw: dict, name: str) -> HermesModelSettings:
        value = raw.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"Hermes {name} 모델 설정이 필요합니다.")
        return HermesModelSettings(
            model=str(value.get("model", "")),
            reasoning=str(value.get("reasoning", "xhigh")),
        )

    def validate_installation(self) -> None:
        if not self.executable.is_file():
            raise RuntimeError(f"Hermes 실행 파일이 없습니다: {self.executable}")
        if not self.home.is_dir():
            raise RuntimeError(f"Hermes 홈 폴더가 없습니다: {self.home}")
        auth = self.home / "auth.json"
        if not auth.is_file() or auth.stat().st_size < 2:
            raise RuntimeError("ChatGPT OAuth 로그인이 필요합니다. finish-auth.ps1을 실행해 주세요.")
        for role in self.roles.values():
            profile = self.home / "profiles" / role.profile
            if not profile.is_dir():
                raise RuntimeError(f"Hermes 프로필이 없습니다: {role.profile}")
        if self.docker_required:
            self._validate_docker_engine()
            self._validate_secure_profiles()

    def _validate_secure_profiles(self) -> None:
        required = {
            "backend": "docker",
            "docker_image": self.isolation.image,
            "docker_mount_cwd_to_workspace": "true",
            "docker_mount_profile_assets": "false",
            "container_persistent": "false",
            "docker_persist_across_processes": "false",
            "docker_network": "false",
            "docker_forward_env": "[]",
            "docker_volumes": "[]",
            "docker_extra_args": "[]",
            "docker_run_as_host_user": "false",
            "docker_snap_compat": "false",
        }
        for role in self.roles.values():
            profile_path = self.home / "profiles" / role.profile / "config.yaml"
            configured = self._terminal_yaml_values(profile_path)
            mismatches = [
                key for key, expected in required.items()
                if configured.get(key) != expected
            ]
            if mismatches:
                raise RuntimeError(
                    f"Hermes 프로필 {role.profile}의 필수 Docker 보안 설정이 올바르지 않습니다: "
                    + ", ".join(mismatches)
                )

    @staticmethod
    def _terminal_yaml_values(path: Path) -> dict[str, str]:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise RuntimeError(f"Hermes 프로필 설정을 읽지 못했습니다: {path}") from exc
        values: dict[str, str] = {}
        in_terminal = False
        for line in lines:
            if re.fullmatch(r"terminal:\s*", line):
                in_terminal = True
                continue
            if in_terminal and line and not line[0].isspace():
                break
            if not in_terminal:
                continue
            match = re.fullmatch(r"\s{2}([A-Za-z0-9_]+):\s*(.*?)\s*(?:#.*)?", line)
            if not match:
                continue
            key, raw = match.groups()
            value = raw.strip().strip("\"").strip("'")
            values[key] = value.casefold() if value.casefold() in {"true", "false"} else value
        return values

    @staticmethod
    def _validate_docker_engine() -> None:
        try:
            result = subprocess.run(
                ["docker", "version", "--format", "{{.Server.Version}}"],
                text=True,
                encoding="utf-8",
                errors="replace",
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                "Docker 격리 실행을 위해 Docker Desktop이 실행 중이어야 합니다."
            ) from exc
        if result.returncode != 0 or not result.stdout.strip():
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                "Docker 격리 실행을 위해 Docker Desktop이 실행 중이어야 합니다"
                + (f": {detail[:400]}" if detail else ".")
            )

    @staticmethod
    def _under_root(root: Path, relative: str) -> Path:
        if not relative.strip():
            raise ValueError("Hermes 경로 설정이 비어 있습니다.")
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Hermes 경로가 시스템 루트를 벗어납니다: {relative}") from exc
        return candidate
