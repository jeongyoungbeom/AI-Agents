from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from app.services.git import GitRepository
from app.services.toolchains import ToolchainPreflightError, ToolchainService


IMAGE = "example.invalid/toolchain@sha256:" + "a" * 64
AI_ROOT = Path(__file__).resolve().parents[2]


class ProbeSandbox:
    def __init__(self) -> None:
        self.images: list[str] = []
        self.calls: list[tuple[tuple[str, ...], bool, tuple[object, ...]]] = []

    def with_image(self, image: str) -> "ProbeSandbox":
        self.images.append(image)
        return self

    def run(self, _repository, argv, *, writable_workspace, additional_mounts=(), **_kwargs):
        values = tuple(argv)
        self.calls.append((values, writable_workspace, tuple(additional_mounts)))
        if values[0] == "git":
            repository = Path(_repository)
            arguments = values[3:]
            if arguments[:2] == ("rev-parse", "HEAD"):
                return subprocess.CompletedProcess(values, 0, "a" * 40 + "\n", "")
            if arguments[:2] == ("ls-tree", "-z"):
                requested = arguments[-1] if "--" in arguments else ""
                if requested:
                    path = repository / requested
                    if not path.is_file():
                        return subprocess.CompletedProcess(values, 0, "", "")
                    return subprocess.CompletedProcess(
                        values, 0, f"100644 blob {'b' * 40}\t{requested}\0", ""
                    )
                entries = []
                for path in repository.iterdir():
                    object_type = "tree" if path.is_dir() else "blob"
                    entries.append(f"100644 {object_type} {'b' * 40}\t{path.name}")
                return subprocess.CompletedProcess(values, 0, "\0".join(entries) + "\0", "")
            if arguments[:2] == ("show", "--no-textconv"):
                path = arguments[-1].split(":", 1)[1]
                return subprocess.CompletedProcess(values, 0, (repository / path).read_text(encoding="utf-8"), "")
        output = {
            "python": "Python 3.12.3",
            "python3": "Python 3.12.3",
            "node": "v22.12.0",
            "npm": "10.9.0",
            "npx": "10.9.0",
            "java": "openjdk version \"21.0.6\"",
            "gradle": "Gradle 8.14.5",
            "./gradlew": "Gradle 8.14.5",
        }.get(values[0], "")
        return subprocess.CompletedProcess(values, 0, output, "")


def write_catalog(root: Path, *, cache_mode: str = "none") -> Path:
    cache = {"mode": cache_mode}
    if cache_mode == "image":
        cache["marker"] = "/opt/ai-agents-cache/gradle/READY"
    path = root / "toolchains.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profiles": {
                    "python-nodejs": {
                        "stacks": ["python", "node"],
                        "image": IMAGE,
                        "executables": {"python": "3.11", "node": "20", "npm": "9", "npx": "9"},
                        "cache": {"mode": "none"},
                    },
                    "gradle-jdk21": {
                        "stacks": ["gradle"],
                        "image": IMAGE,
                        "executables": {"gradle": "8", "java": "21"},
                        "cache": cache,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


class ToolchainServiceTests(unittest.TestCase):
    def build(self, root: Path, *, cache_mode: str = "none") -> tuple[ToolchainService, ProbeSandbox]:
        sandbox = ProbeSandbox()
        return ToolchainService.load(write_catalog(root, cache_mode=cache_mode), sandbox=sandbox), sandbox

    @staticmethod
    def probes(sandbox: ProbeSandbox):
        return [call for call in sandbox.calls if call[0][0] != "git"]

    @staticmethod
    def preflight(
        service: ToolchainService,
        sandbox: ProbeSandbox,
        root: Path,
        commands: tuple[str, ...],
    ):
        return service.preflight(GitRepository(root, sandbox), commands)

    def test_python_manifest_selects_pinned_python_node_profile_and_read_only_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
            service, sandbox = self.build(root)

            environment = self.preflight(service, sandbox, root, ("python -m unittest",))

        self.assertEqual("python-nodejs", environment.profile_id)
        self.assertEqual(IMAGE, environment.image)
        self.assertEqual([("python", "--version")], [call[0] for call in self.probes(sandbox)])
        self.assertTrue(all(not call[1] for call in sandbox.calls))

    def test_node_manifest_without_dependencies_uses_the_same_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "package.json").write_text('{"scripts":{"test":"node test.js"}}', encoding="utf-8")
            service, sandbox = self.build(root)

            environment = self.preflight(service, sandbox, root, ("npm test",))

        self.assertEqual("python-nodejs", environment.profile_id)
        self.assertEqual(("npm", "--version"), self.probes(sandbox)[0][0])

    def test_windows_only_gradle_wrapper_is_blocked_before_container_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "build.gradle.kts").write_text("tasks.register(\"test\") {}", encoding="utf-8")
            (root / "gradlew.bat").write_text("@echo off", encoding="utf-8")
            service, sandbox = self.build(root)

            with self.assertRaisesRegex(ToolchainPreflightError, "gradlew.bat"):
                self.preflight(service, sandbox, root, ("gradlew.bat test",))

        self.assertEqual([], self.probes(sandbox))

    def test_gradle_wrapper_with_dependencies_requires_a_prepared_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "build.gradle.kts").write_text("dependencies { }", encoding="utf-8")
            (root / "gradlew").write_text("#!/bin/sh\n", encoding="utf-8")
            service, sandbox = self.build(root)

            with self.assertRaisesRegex(ToolchainPreflightError, "dependency-cache"):
                self.preflight(service, sandbox, root, ("gradlew test",))

        self.assertEqual([("java", "--version"), ("./gradlew", "--version")], [call[0] for call in self.probes(sandbox)])

    def test_explicit_override_cannot_inject_image_or_mount_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "package.json").write_text("{}", encoding="utf-8")
            override = root / ".ai-agents"
            override.mkdir()
            (override / "toolchain.json").write_text(
                json.dumps({"schema_version": 1, "profile": "python-nodejs", "image": "evil@sha256:" + "b" * 64}),
                encoding="utf-8",
            )
            service, sandbox = self.build(root)

            with self.assertRaisesRegex(ToolchainPreflightError, "profile만"):
                self.preflight(service, sandbox, root, ("npm test",))

    def test_multiple_detected_stacks_need_an_explicit_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "package.json").write_text("{}", encoding="utf-8")
            (root / "pyproject.toml").write_text("[project]\nname='fixture'", encoding="utf-8")
            service, sandbox = self.build(root)

            with self.assertRaisesRegex(ToolchainPreflightError, "여러 기술 스택"):
                self.preflight(service, sandbox, root, ("python -m unittest",))

    def test_prebuilt_cache_profile_is_checked_without_a_writable_mount(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "build.gradle.kts").write_text("dependencies { }", encoding="utf-8")
            (root / "gradlew").write_text("#!/bin/sh\n", encoding="utf-8")
            override = root / ".ai-agents"
            override.mkdir()
            (override / "toolchain.json").write_text(
                json.dumps({"schema_version": 1, "profile": "gradle-jdk21"}), encoding="utf-8"
            )
            service, sandbox = self.build(root, cache_mode="image")

            environment = self.preflight(service, sandbox, root, ("gradlew test",))

        self.assertEqual("gradle-jdk21", environment.profile_id)
        self.assertEqual(("/bin/sh", "-c", "test -r /opt/ai-agents-cache/gradle/READY"), self.probes(sandbox)[-1][0])
        self.assertFalse(self.probes(sandbox)[-1][1])

    def test_gateway_catalog_uses_the_declared_default_gradle_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "build.gradle.kts").write_text("tasks.register(\"test\") {}", encoding="utf-8")
            sandbox = ProbeSandbox()
            service = ToolchainService.load(AI_ROOT / "config" / "toolchains.json", sandbox=sandbox)

            environment = self.preflight(service, sandbox, root, ("gradle test",))

        self.assertEqual("gradle-jdk21", environment.profile_id)
        self.assertTrue(environment.image.startswith("gradle@sha256:"))


if __name__ == "__main__":
    unittest.main()
