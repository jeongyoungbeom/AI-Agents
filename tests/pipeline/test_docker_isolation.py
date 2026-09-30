from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.git import GitRepository, GitRepositoryCancelled, IsolatedGitWorktree
from app.services.hermes import HermesSettings
from app.services.sandbox import (
    DEFAULT_SECURE_DOCKER_IMAGE,
    DockerContainerOwnership,
    DockerGitMetadataMount,
    DockerSandbox,
    DockerSandboxCancelled,
    DockerSandboxTimeout,
    DockerVolumeMount,
)
from app.services.sandbox.docker import (
    COMPONENT_LABEL,
    GATEWAY_INSTANCE_LABEL,
    GATEWAY_PID_LABEL,
    GATEWAY_STARTED_LABEL,
    INSTALLATION_LABEL,
    MANAGED_LABEL,
    OPERATION_LABEL,
)
from app.services.verification import VerificationRunner
from tests.pipeline.support import create_repository, temporary_directory


class DockerIsolationTests(unittest.TestCase):
    class _Process:
        def __init__(self, outcome: str):
            self.args = ["docker", "run"]
            self.outcome = outcome
            self.returncode = None

        def communicate(self, *, timeout):
            if self.outcome == "waiting":
                raise subprocess.TimeoutExpired(self.args, timeout)
            self.returncode = 0 if self.outcome == "success" else 23
            return ("ok", "" if self.outcome == "success" else "failed")

        def poll(self):
            return self.returncode

        def terminate(self):
            if self.returncode is None:
                self.returncode = -15

        def kill(self):
            if self.returncode is None:
                self.returncode = -9

        def wait(self, timeout=None):
            if self.returncode is None:
                self.returncode = -15
            return self.returncode

    class _ProcessTree:
        def __init__(self, process):
            self.process = process

        def terminate(self):
            self.process.terminate()

        def close(self):
            self.process.terminate()

    @staticmethod
    def _ownership(root: Path, operation_id: str = "RUN-LIFECYCLE-1"):
        return DockerContainerOwnership(
            operation_id=operation_id,
            component="verification",
            gateway_instance_id="gateway-test-1",
            gateway_process_id=1,
            gateway_process_started_at=1.0,
            installation_id="installation-test-1",
            container_name=f"ai-agents-{operation_id.lower()}",
            cidfile=root / f"{operation_id}.cid",
        )

    def test_sandbox_command_is_ephemeral_air_gapped_and_repo_scoped(self):
        with temporary_directory() as directory:
            repository = Path(directory) / "repository"
            repository.mkdir()
            command = DockerSandbox().command(
                repository, ["git", "status"], writable_workspace=True
            )

        self.assertEqual("docker", command[0])
        self.assertIn("--rm", command)
        self.assertIn("--network=none", command)
        self.assertIn("--read-only", command)
        self.assertIn("--cap-drop=ALL", command)
        self.assertIn("--security-opt=no-new-privileges", command)
        self.assertIn(DEFAULT_SECURE_DOCKER_IMAGE, command)
        mount = command[command.index("--mount") + 1]
        self.assertIn(f"src={repository.resolve()}", mount)
        self.assertIn("dst=/workspace", mount)
        self.assertNotIn("-v", command)

    def test_toolchain_cache_volume_is_named_read_only_and_never_replaces_repository_mount(self):
        with temporary_directory() as directory:
            repository = Path(directory) / "repository"
            repository.mkdir()
            command = DockerSandbox().command(
                repository,
                ["python", "--version"],
                writable_workspace=False,
                additional_mounts=(
                    DockerVolumeMount("ai-agents-gradle-cache-v1", "/opt/gradle-cache"),
                ),
            )

        mounts = [command[index + 1] for index, value in enumerate(command) if value == "--mount"]
        self.assertIn(f"type=bind,src={repository.resolve()},dst=/workspace,readonly", mounts)
        self.assertIn(
            "type=volume,src=ai-agents-gradle-cache-v1,dst=/opt/gradle-cache,readonly",
            mounts,
        )

    def test_linked_worktree_metadata_mount_is_gateway_fixed_and_read_only_for_code(self):
        with temporary_directory() as directory:
            root = Path(directory)
            repository = root / "worktree"
            repository.mkdir()
            (repository / ".git").write_text("gitdir: host-only\n", encoding="utf-8")
            common = root / "source-git"
            (common / "worktrees" / "run-1").mkdir(parents=True)
            pointer = root / "container-gitdir"
            pointer.write_text(
                "gitdir: /ai-agents-gitdir/worktrees/run-1\n", encoding="utf-8"
            )
            metadata = DockerGitMetadataMount(common, "worktrees/run-1", pointer)
            command = DockerSandbox().command(
                repository,
                ["git", "status"],
                writable_workspace=True,
                git_metadata=metadata,
                writable_git_metadata=False,
            )

        mounts = [command[index + 1] for index, value in enumerate(command) if value == "--mount"]
        self.assertIn(f"type=bind,src={common.resolve()},dst=/ai-agents-gitdir,readonly", mounts)
        self.assertIn(
            f"type=bind,src={pointer.resolve()},dst=/workspace/.git,readonly", mounts
        )

    def test_owned_launch_has_unique_name_labels_and_cidfile(self):
        repository = Path(__file__).resolve().parents[2]
        sandbox = DockerSandbox()
        with patch.object(sandbox, "_new_cidfile", return_value=Path("test.cid")):
            command, ownership = sandbox.prepare(
                repository,
                ["git", "status"],
                writable_workspace=False,
                operation_id="RUN-OWNERSHIP-1",
                component="repository-reader",
            )

        self.assertIn("--name", command)
        self.assertIn(ownership.container_name, command)
        self.assertIn("--cidfile", command)
        self.assertIn(str(ownership.cidfile), command)
        self.assertLessEqual(len(ownership.container_name), 63)
        for key, value in ownership.labels.items():
            self.assertIn(f"{key}={value}", command)
        self.assertEqual("true", ownership.labels[MANAGED_LABEL])
        self.assertEqual("RUN-OWNERSHIP-1", ownership.labels[OPERATION_LABEL])
        self.assertEqual("repository-reader", ownership.labels[COMPONENT_LABEL])
        self.assertTrue(ownership.labels[GATEWAY_INSTANCE_LABEL])

    def test_created_container_cleanup_is_exact_and_idempotent(self):
        root = Path(__file__).resolve().parent
        sandbox = DockerSandbox()
        ownership = self._ownership(root, "RUN-CREATED-1")
        removed = subprocess.CompletedProcess(
            ["docker", "rm", "-f", "created-container-id"], 0, "", ""
        )
        with patch.object(
            sandbox, "_cleanup_candidates", return_value=("created-container-id",)
        ), patch.object(
            sandbox,
            "_container_labels",
            side_effect=(ownership.labels, None),
        ), patch.object(sandbox, "_docker", return_value=removed) as docker:
            self.assertTrue(sandbox.cleanup(ownership, reason="cancelled"))
            self.assertFalse(sandbox.cleanup(ownership, reason="cancelled"))

        docker.assert_called_once_with(
            ["rm", "-f", "created-container-id"], timeout=8
        )
        self.assertFalse(ownership.cidfile.exists())

    def test_run_cleans_owned_container_on_success_failure_timeout_and_cancel(self):
        """Every run outcome reaches the same ownership-scoped cleanup boundary."""
        root = Path(__file__).resolve().parent
        repository = Path(__file__).resolve().parents[2]
        sandbox = DockerSandbox()

        def invoke(outcome, *, timeout=None, cancelled=None):
            ownership = self._ownership(root, f"RUN-{outcome.upper()}-1")
            process = self._Process(outcome)
            setattr(process, "_ai_agents_docker_ownership", ownership)
            with patch.object(sandbox, "popen", return_value=process), patch(
                "app.services.sandbox.docker.ProcessTree", self._ProcessTree
            ), patch.object(sandbox, "cleanup", return_value=True) as cleanup:
                if outcome == "waiting" and cancelled is not None:
                    with self.assertRaises(DockerSandboxCancelled):
                        sandbox.run(
                            repository,
                            ["git", "status"],
                            writable_workspace=False,
                            timeout=timeout,
                            cancelled=cancelled,
                        )
                elif outcome == "waiting":
                    with patch(
                        "app.services.sandbox.docker.time.monotonic",
                        side_effect=(0.0, timeout + 1.0),
                    ), self.assertRaises(DockerSandboxTimeout):
                        sandbox.run(
                            repository,
                            ["git", "status"],
                            writable_workspace=False,
                            timeout=timeout,
                        )
                else:
                    completed = sandbox.run(
                        repository,
                        ["git", "status"],
                        writable_workspace=False,
                        timeout=timeout,
                    )
                    self.assertEqual(0 if outcome == "success" else 23, completed.returncode)
            return cleanup.call_args.kwargs["reason"]

        self.assertEqual("completed", invoke("success", timeout=1))
        self.assertEqual("completed", invoke("failure", timeout=1))
        self.assertEqual("timeout", invoke("waiting", timeout=0.01))
        self.assertEqual("cancelled", invoke("waiting", timeout=1, cancelled=lambda: True))

    def test_parallel_cleanup_cannot_remove_another_operations_container(self):
        root = Path(__file__).resolve().parent
        sandbox = DockerSandbox()
        first = self._ownership(root, "RUN-PARALLEL-A")
        second = self._ownership(root, "RUN-PARALLEL-B")
        removed = subprocess.CompletedProcess(["docker", "rm"], 0, "", "")
        with patch.object(
            sandbox, "_cleanup_candidates", return_value=("container-a",)
        ), patch.object(
            sandbox,
            "_container_labels",
            side_effect=lambda container: {
                "container-a": first.labels,
                "container-b": second.labels,
            }.get(container),
        ), patch.object(sandbox, "_docker", return_value=removed) as docker:
            self.assertTrue(sandbox.cleanup(first, reason="cancelled"))

        docker.assert_called_once_with(["rm", "-f", "container-a"], timeout=8)

    def test_git_cancellation_callback_reaches_the_docker_boundary(self):
        class CancellingSandbox:
            def __init__(self):
                self.callback = None

            def run(self, *_args, cancelled=None, **_kwargs):
                self.callback = cancelled
                raise DockerSandboxCancelled("cancelled")

        repository = Path(__file__).resolve().parents[2]
        sandbox = CancellingSandbox()
        callback = lambda: True
        git = GitRepository(repository, sandbox, cancelled=callback)
        with self.assertRaises(GitRepositoryCancelled):
            git.snapshot()

        self.assertIs(callback, sandbox.callback)

    def test_startup_cleanup_skips_live_and_unrelated_containers(self):
        sandbox = DockerSandbox()
        stale = {
            MANAGED_LABEL: "true",
            INSTALLATION_LABEL: sandbox.installation_id,
            OPERATION_LABEL: "RUN-STALE-1",
            COMPONENT_LABEL: "verification",
            GATEWAY_INSTANCE_LABEL: "previous-instance",
            GATEWAY_PID_LABEL: "999999999",
            GATEWAY_STARTED_LABEL: "0",
        }
        live = {
            MANAGED_LABEL: "true",
            INSTALLATION_LABEL: sandbox.installation_id,
            OPERATION_LABEL: "RUN-LIVE-1",
            COMPONENT_LABEL: "verification",
            GATEWAY_INSTANCE_LABEL: sandbox.gateway_instance_id,
            GATEWAY_PID_LABEL: str(sandbox.gateway_process_id),
            GATEWAY_STARTED_LABEL: f"{sandbox.gateway_process_started_at:.6f}",
        }
        unrelated = {**stale, INSTALLATION_LABEL: "other-installation"}

        def docker(arguments, *, timeout):
            if arguments[0] == "ps":
                return subprocess.CompletedProcess(arguments, 0, "stale\nlive\nother\n", "")
            if arguments == ["rm", "-f", "stale"]:
                return subprocess.CompletedProcess(arguments, 0, "", "")
            self.fail(f"unexpected Docker command: {arguments}")

        with patch.object(sandbox, "_docker", side_effect=docker), patch.object(
            sandbox,
            "_container_labels",
            side_effect=lambda container: {
                "stale": stale,
                "live": live,
                "other": unrelated,
            }[container],
        ), patch.object(sandbox, "_gateway_owner_is_alive", return_value=False):
            self.assertEqual(("stale",), sandbox.cleanup_stale())

    @unittest.skipUnless(shutil.which("docker"), "Docker CLI is required")
    def test_git_commit_and_verification_run_through_the_container_boundary(self):
        with temporary_directory() as directory:
            root = Path(directory)
            repository_path = create_repository(root)
            repository = GitRepository(repository_path)
            repository.preflight()
            (repository_path / "feature.txt").write_text("fixed\n", encoding="utf-8")
            commit = repository.commit_scope("test: isolated commit", ("feature.txt",))

            result = VerificationRunner(timeout_seconds=30).run_all(
                repository, ("python verify.py",)
            )

        self.assertRegex(commit, r"^[0-9a-f]{40}$")
        self.assertEqual(1, len(result))
        self.assertTrue(result[0].passed)

    @unittest.skipUnless(shutil.which("docker"), "Docker CLI is required")
    def test_linked_worktree_commit_and_verification_keep_source_unchanged_until_apply(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source_path = create_repository(root)
            worktree = IsolatedGitWorktree(
                source_path, DockerSandbox(), run_id="RUN-WORKTREE-SMOKE-1"
            )
            source, baseline = worktree.begin()
            repository = worktree.create(baseline)
            (repository.path / "feature.txt").write_text("fixed\n", encoding="utf-8")
            candidate = repository.commit_scope(
                "test: isolated worktree commit", ("feature.txt",)
            )

            result = VerificationRunner(timeout_seconds=30).run_all(
                repository, ("python verify.py",)
            )

            self.assertFalse((source_path / "feature.txt").exists())
            self.assertEqual(1, len(result))
            self.assertTrue(result[0].passed)
            applied = source.apply_candidate(baseline, candidate, ("feature.txt",))
            self.assertRegex(applied, r"^[0-9a-f]{40}$")
            self.assertEqual(
                "fixed\n", (source_path / "feature.txt").read_text(encoding="utf-8")
            )
            self.assertIsNone(worktree.cleanup())

    def test_required_profiles_fail_closed_when_a_security_value_is_missing(self):
        root = Path(__file__).resolve().parents[2]
        settings = HermesSettings.load(root)
        with patch.object(HermesSettings, "_terminal_yaml_values", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "필수 Docker 보안 설정"):
                settings._validate_secure_profiles()


if __name__ == "__main__":
    unittest.main()
