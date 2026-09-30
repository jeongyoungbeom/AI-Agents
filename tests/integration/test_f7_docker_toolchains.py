from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

from app.services.git import GitRepository
from app.services.sandbox import DockerSandbox
from app.services.sandbox.docker import (
    INSTALLATION_LABEL,
    MANAGED_LABEL,
    OPERATION_LABEL,
)
from app.services.toolchains import ToolchainService
from app.services.verification import VerificationRunner
from tests.pipeline.support import git, temporary_directory


AI_ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(
    os.environ.get("AI_AGENTS_F7_DOCKER") == "1",
    "F-7 Docker fixture는 scripts\\f7-validation.ps1 -Docker에서만 실행합니다.",
)
class F7DockerToolchainTests(unittest.TestCase):
    """고정 Docker profile로 실제 무의존성 fixture를 실행한다."""

    @staticmethod
    def _repository(root: Path, name: str, files: dict[str, str]) -> Path:
        repository = root / name
        repository.mkdir()
        git(repository, "init", "--quiet")
        git(repository, "config", "user.name", "AI Agents F7 Fixture")
        git(repository, "config", "user.email", "f7@example.invalid")
        for relative_path, content in files.items():
            target = repository / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        git(repository, "add", ".")
        git(repository, "commit", "--quiet", "-m", "F-7 fixture")
        return repository

    def _assert_no_owned_containers(
        self, sandbox: DockerSandbox, operation_id: str
    ) -> None:
        result = subprocess.run(
            [
                "docker",
                "ps",
                "--all",
                "--filter",
                f"label={MANAGED_LABEL}=true",
                "--filter",
                f"label={INSTALLATION_LABEL}={sandbox.installation_id}",
                "--filter",
                f"label={OPERATION_LABEL}={operation_id}",
                "--format",
                "{{.ID}}",
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            self.fail(f"F-7 Docker container lookup failed: {result.stderr.strip()}")
        self.assertEqual((), tuple(line for line in result.stdout.splitlines() if line))

    def _verify(
        self,
        repository_path: Path,
        commands: tuple[str, ...],
        expected_profile: str,
        operation_id: str,
    ) -> None:
        sandbox = DockerSandbox()
        repository = GitRepository(
            repository_path, sandbox, operation_id=operation_id
        )
        baseline = repository.preflight()
        toolchains = ToolchainService.load(
            AI_ROOT / "config" / "toolchains.json", sandbox=sandbox
        )
        environment = toolchains.preflight(
            repository, commands, operation_id=operation_id
        )
        results = VerificationRunner(
            sandbox=sandbox, timeout_seconds=120, poll_seconds=0.1
        ).run_all(
            repository,
            commands,
            operation_id=operation_id,
            environment=environment,
        )

        self.assertEqual(expected_profile, environment.profile_id)
        self.assertTrue(results and all(result.passed for result in results))
        self.assertEqual(baseline, repository.snapshot())
        self._assert_no_owned_containers(sandbox, operation_id)

    def test_python_and_node_fixture_runs_through_the_pinned_profile(self):
        with temporary_directory() as directory:
            repository = self._repository(
                Path(directory),
                "python-node",
                {
                    ".ai-agents/toolchain.json": (
                        '{"schema_version": 1, "profile": "python-nodejs"}\n'
                    ),
                    "package.json": '{"name":"f7-node-python","private":true}\n',
                    "pyproject.toml": "[project]\nname = 'f7-node-python'\n",
                    "verify-node.js": (
                        "const value = 40 + 2;\n"
                        "if (value !== 42) process.exit(1);\n"
                    ),
                    "verify-python.py": "assert 40 + 2 == 42\n",
                },
            )

            self._verify(
                repository,
                ("node verify-node.js", "python verify-python.py"),
                "python-nodejs",
                "RUN-F7-NODE-PYTHON",
            )

    def test_kotlin_gradle_fixture_runs_without_network_or_dependencies(self):
        with temporary_directory() as directory:
            repository = self._repository(
                Path(directory),
                "kotlin-gradle",
                {
                    "settings.gradle.kts": 'rootProject.name = "f7-kotlin-fixture"\n',
                    "build.gradle.kts": (
                        "tasks.register(\"f7Fixture\") {\n"
                        "    doLast {\n"
                        "        check(project.name == \"f7-kotlin-fixture\")\n"
                        "        println(\"f7-kotlin-gradle-ok\")\n"
                        "    }\n"
                        "}\n"
                    ),
                },
            )

            self._verify(
                repository,
                ("gradle f7Fixture --no-daemon",),
                "gradle-jdk21",
                "RUN-F7-KOTLIN-GRADLE",
            )


if __name__ == "__main__":
    unittest.main()
