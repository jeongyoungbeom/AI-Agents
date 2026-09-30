from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.toolchains import ToolchainEnvironment
from app.services.verification import (
    VerificationFailureKind,
    VerificationRunner,
)


class _Repository:
    path = Path(".")

    def snapshot(self):
        return object()

    def assert_position(self, _baseline, *, allow_worktree_changes):
        self.asserted = allow_worktree_changes


class _Process:
    args = ["docker", "run"]
    returncode = 1

    def communicate(self, *, timeout):
        return "", "ModuleNotFoundError: No module named 'fixture'"

    def poll(self):
        return self.returncode


class _ProcessTree:
    def __init__(self, _process):
        pass

    def terminate(self):
        pass

    def close(self):
        pass


class _SelectedSandbox:
    def __init__(self) -> None:
        self.calls = []

    def popen(self, _repository, argv, **kwargs):
        self.calls.append((tuple(argv), kwargs))
        return _Process()

    def cleanup_process(self, _process, *, reason):
        self.cleanup_reason = reason

    def note_process_termination(self, _process, *, reason):
        self.termination_reason = reason


class _BaseSandbox:
    def __init__(self, selected: _SelectedSandbox) -> None:
        self.selected = selected
        self.image = ""

    def with_image(self, image):
        self.image = image
        return self.selected


class VerificationToolchainTests(unittest.TestCase):
    def test_verification_reuses_preflight_image_and_classifies_dependency_failure(self):
        selected = _SelectedSandbox()
        base = _BaseSandbox(selected)
        environment = ToolchainEnvironment(
            "python-nodejs", "example.invalid/toolchain@sha256:" + "c" * 64, (), ("python",)
        )
        repository = _Repository()

        with patch("app.services.verification.runner.ProcessTree", _ProcessTree):
            result = VerificationRunner(sandbox=base, poll_seconds=0.01).run_all(
                repository,
                ("python -m unittest",),
                environment=environment,
            )

        self.assertEqual(environment.image, base.image)
        self.assertEqual(("python", "-m", "unittest"), selected.calls[0][0])
        self.assertEqual((), selected.calls[0][1]["additional_mounts"])
        self.assertEqual(VerificationFailureKind.DEPENDENCY_UNAVAILABLE, result[0].failure_kind)
        self.assertEqual(
            VerificationFailureKind.DEPENDENCY_UNAVAILABLE,
            result[0].to_dict()["failure_kind"],
        )

    def test_gradle_wrapper_command_uses_the_workspace_relative_wrapper(self):
        selected = _SelectedSandbox()
        base = _BaseSandbox(selected)
        environment = ToolchainEnvironment(
            "gradle-jdk21", "example.invalid/toolchain@sha256:" + "d" * 64, (), ("gradle",)
        )

        with patch("app.services.verification.runner.ProcessTree", _ProcessTree):
            VerificationRunner(sandbox=base, poll_seconds=0.01).run_all(
                _Repository(),
                ("gradlew test",),
                environment=environment,
            )

        self.assertEqual(("./gradlew", "test"), selected.calls[0][0])


if __name__ == "__main__":
    unittest.main()
