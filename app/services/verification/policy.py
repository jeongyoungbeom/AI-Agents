from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path


class UnsafeVerificationCommand(ValueError):
    pass


@dataclass(frozen=True)
class PreparedVerificationCommand:
    argv: tuple[str, ...]
    display: str


class SafeVerificationPolicy:
    """검증 명령을 셸 없이 실행할 수 있는 인자 목록으로 제한한다."""

    _DIRECT_CODE_FLAGS = frozenset({"-c", "--command", "-e", "--eval", "--exec"})
    _PACKAGE_MUTATION_COMMANDS = frozenset(
        {
            "add",
            "ci",
            "i",
            "init",
            "install",
            "login",
            "logout",
            "publish",
            "remove",
            "rm",
            "uninstall",
            "update",
        }
    )

    DEFAULT_TOOLS = frozenset(
        {
            "cargo",
            "dotnet",
            "go",
            "gradle",
            "gradlew",
            "gradlew.bat",
            "java",
            "mvn",
            "mvnw",
            "mvnw.cmd",
            "node",
            "npm",
            "npm.cmd",
            "npx",
            "npx.cmd",
            "pnpm",
            "pnpm.cmd",
            "py",
            "pytest",
            "python",
            "python3",
            "yarn",
            "yarn.cmd",
        }
    )

    def __init__(self, allowed_tools: frozenset[str] | None = None):
        self.allowed_tools = allowed_tools or self.DEFAULT_TOOLS

    def prepare(self, command: str) -> PreparedVerificationCommand:
        value = command.strip()
        if not value:
            raise UnsafeVerificationCommand("검증 명령이 비어 있습니다.")
        if re.search(r"[\r\n\0;&|<>`]", value) or "$(" in value:
            raise UnsafeVerificationCommand("셸 연결·리다이렉션 문법은 허용하지 않습니다.")
        try:
            raw_argv = shlex.split(value, posix=False)
        except ValueError as exc:
            raise UnsafeVerificationCommand("검증 명령의 따옴표가 올바르지 않습니다.") from exc
        argv = tuple(item.strip('"') for item in raw_argv)
        if not argv:
            raise UnsafeVerificationCommand("검증 명령이 비어 있습니다.")
        executable = Path(argv[0].replace("\\", "/")).name.casefold()
        if executable not in {tool.casefold() for tool in self.allowed_tools}:
            raise UnsafeVerificationCommand(
                f"허용 목록에 없는 검증 도구입니다: {executable}"
            )
        if argv[0] == "gradlew":
            # Docker does not add /workspace to PATH.  Keep the wrapper argv
            # consistent with the workspace-relative probe used by preflight.
            argv = ("./gradlew", *argv[1:])
        lowered = tuple(item.casefold() for item in argv)
        if executable == "git" or "--hard" in lowered or "--force" in lowered:
            raise UnsafeVerificationCommand("파괴적일 수 있는 검증 명령은 허용하지 않습니다.")
        if any(item in self._DIRECT_CODE_FLAGS for item in lowered[1:]):
            raise UnsafeVerificationCommand(
                "검증 명령에 직접 코드 실행 인자를 사용할 수 없습니다."
            )
        if executable in {"python", "python3", "py"}:
            for index, item in enumerate(lowered[:-1]):
                if item == "-m" and lowered[index + 1] in {"pip", "ensurepip"}:
                    raise UnsafeVerificationCommand(
                        "패키지 설치·변경 모듈은 검증 명령으로 사용할 수 없습니다."
                    )
        if executable in {
            "npm",
            "npm.cmd",
            "pnpm",
            "pnpm.cmd",
            "yarn",
            "yarn.cmd",
        }:
            subcommands = [item for item in lowered[1:] if not item.startswith("-")]
            if subcommands and subcommands[0] in self._PACKAGE_MUTATION_COMMANDS:
                raise UnsafeVerificationCommand(
                    "패키지 설치·배포·계정 변경 명령은 검증으로 사용할 수 없습니다."
                )
        if executable in {"npx", "npx.cmd"} and any(
            item in {"--package", "-p"} for item in lowered[1:]
        ):
            raise UnsafeVerificationCommand(
                "npx의 패키지 주입 옵션은 검증 명령으로 사용할 수 없습니다."
            )
        return PreparedVerificationCommand(argv=argv, display=value)
