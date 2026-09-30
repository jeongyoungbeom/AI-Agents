from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import unittest
from pathlib import Path

from app.contracts import RoleId, RunPhase
from app.services.git import IsolatedGitWorktree
from app.services.toolchains import ToolchainService

from tests.pipeline.support import (
    AI_ROOT,
    FakeRoleRunner,
    HostGitSandbox,
    build_pipeline,
    create_repository,
    git,
    temporary_directory,
)


class _CapturingRunner(FakeRoleRunner):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.repositories: list[Path] = []

    def run(self, *args, **kwargs):
        self.repositories.append(Path(args[3]).resolve())
        return super().run(*args, **kwargs)


class IsolatedWorktreePipelineTests(unittest.TestCase):
    @staticmethod
    def _artifact(root: Path, run_id: str, name: str) -> dict:
        return json.loads(
            (root / "artifacts" / run_id / name).read_text(encoding="utf-8")
        )

    def test_success_uses_a_temp_worktree_and_applies_only_after_verification(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            initial_head = git(source, "rev-parse", "HEAD")
            runner = _CapturingRunner(
                review_responses=[[]], development_outputs=["fixed"]
            )
            store, worker, run_id = build_pipeline(
                root, source, runner, sandbox=HostGitSandbox()
            )

            self.assertTrue(worker.run_once())

            self.assertEqual(RunPhase.COMPLETED, store.load_run(run_id).phase)
            self.assertNotEqual(initial_head, git(source, "rev-parse", "HEAD"))
            self.assertEqual("fixed", (source / "feature.txt").read_text().strip())
            self.assertEqual("", git(source, "status", "--porcelain=v1"))
            self.assertTrue(runner.repositories)
            self.assertTrue(all(path != source for path in runner.repositories))
            worktree = self._artifact(root, run_id, "worktree.json")
            self.assertEqual("cleaned", worktree["status"])
            self.assertFalse(Path(worktree["worktree_path"]).exists())
            application = self._artifact(root, run_id, "source-application.json")
            self.assertTrue(application["applied"])

    def test_dirty_source_is_recorded_and_never_passed_to_a_role(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            (source / "README.md").write_text("user edit\n", encoding="utf-8")
            (source / "user-note.txt").write_text("untracked\n", encoding="utf-8")
            before_status = git(source, "status", "--porcelain=v1")
            runner = _CapturingRunner(review_responses=[[]])
            store, worker, run_id = build_pipeline(
                root, source, runner, sandbox=HostGitSandbox()
            )

            self.assertTrue(worker.run_once())

            self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
            self.assertEqual(before_status, git(source, "status", "--porcelain=v1"))
            self.assertFalse(runner.calls)
            worktree = self._artifact(root, run_id, "worktree.json")
            self.assertEqual("source_blocked", worktree["status"])
            self.assertIn("user-note.txt", worktree["source_snapshot"]["status"])
            self.assertIn("user-note.txt", worktree["source_snapshot"]["untracked_files"])

    def test_external_symlink_is_blocked_before_a_writable_role_runs(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            outside = root / "outside.txt"
            outside.write_text("keep\n", encoding="utf-8")
            escape = source / "escape.txt"
            try:
                escape.symlink_to(outside)
            except OSError as exc:
                self.skipTest(f"symlink creation is unavailable: {exc}")
            if not escape.is_symlink():
                self.skipTest("Git fixture filesystem does not preserve symlinks")
            git(source, "add", "escape.txt")
            git(source, "commit", "-m", "add external symlink")
            initial_head = git(source, "rev-parse", "HEAD")
            runner = _CapturingRunner(review_responses=[[]])
            store, worker, run_id = build_pipeline(
                root, source, runner, sandbox=HostGitSandbox()
            )

            self.assertTrue(worker.run_once())

            self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
            self.assertEqual(initial_head, git(source, "rev-parse", "HEAD"))
            self.assertEqual("keep\n", outside.read_text(encoding="utf-8"))
            self.assertFalse(runner.calls)

    def test_source_head_change_blocks_application_and_preserves_candidate(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)

            class ConcurrentWriter(_CapturingRunner):
                def __init__(self):
                    super().__init__(review_responses=[[]], development_outputs=["fixed"])
                    self.changed_source = False

                def run(self, *args, **kwargs):
                    result = super().run(*args, **kwargs)
                    if args[2] == RoleId.DEVELOPMENT and not self.changed_source:
                        self.changed_source = True
                        (source / "concurrent.txt").write_text(
                            "external\n", encoding="utf-8"
                        )
                        git(source, "add", "concurrent.txt")
                        git(source, "commit", "-m", "external change")
                    return result

            runner = ConcurrentWriter()
            store, worker, run_id = build_pipeline(
                root, source, runner, sandbox=HostGitSandbox()
            )

            self.assertTrue(worker.run_once())

            self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
            self.assertTrue((source / "concurrent.txt").is_file())
            self.assertFalse((source / "feature.txt").exists())
            application = self._artifact(root, run_id, "source-application.json")
            self.assertFalse(application["applied"])
            recovery = self._artifact(root, run_id, "worktree-recovery.json")
            self.assertTrue(recovery["preserved"])
            self.assertTrue(Path(recovery["worktree"]["worktree_path"]).is_dir())

    def test_scope_violation_never_changes_the_source_and_keeps_evidence(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            initial_head = git(source, "rev-parse", "HEAD")
            runner = _CapturingRunner(development_out_of_scope_write=True)
            store, worker, run_id = build_pipeline(
                root, source, runner, sandbox=HostGitSandbox()
            )

            self.assertTrue(worker.run_once())

            self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
            self.assertEqual(initial_head, git(source, "rev-parse", "HEAD"))
            self.assertFalse((source / "feature.txt").exists())
            self.assertFalse((source / "outside-stage-scope.txt").exists())
            recovery = self._artifact(root, run_id, "worktree-recovery.json")
            self.assertIn(
                "outside-stage-scope.txt", recovery["recovery"]["changed_files"]
            )

    def test_cancellation_preserves_worktree_changes_without_touching_source(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)

            class CancellingRunner(_CapturingRunner):
                def __init__(self):
                    super().__init__(review_responses=[[]], development_outputs=["fixed"])
                    self.cancel = lambda: None
                    self.cancelled = False

                def run(self, *args, **kwargs):
                    result = super().run(*args, **kwargs)
                    if args[2] == RoleId.DEVELOPMENT and not self.cancelled:
                        self.cancelled = True
                        self.cancel()
                    return result

            runner = CancellingRunner()
            store, worker, run_id = build_pipeline(
                root, source, runner, sandbox=HostGitSandbox()
            )
            runner.cancel = lambda: worker.cancel(run_id)

            self.assertTrue(worker.run_once())

            self.assertEqual(RunPhase.CANCELLED, store.load_run(run_id).phase)
            self.assertFalse((source / "feature.txt").exists())
            recovery = self._artifact(root, run_id, "worktree-recovery.json")
            self.assertTrue(recovery["preserved"])
            self.assertTrue(Path(recovery["worktree"]["worktree_path"]).is_dir())

    def test_verification_failure_keeps_source_unchanged_and_candidate_recoverable(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root, verifier_passes=False)
            initial_head = git(source, "rev-parse", "HEAD")
            runner = _CapturingRunner(review_responses=[[]], development_outputs=["fixed"])
            store, worker, run_id = build_pipeline(
                root, source, runner, sandbox=HostGitSandbox()
            )

            self.assertTrue(worker.run_once())

            self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
            self.assertEqual(initial_head, git(source, "rev-parse", "HEAD"))
            self.assertFalse((source / "feature.txt").exists())
            recovery = self._artifact(root, run_id, "worktree-recovery.json")
            self.assertTrue(recovery["recovery"]["patch_available"])

    def test_toolchain_preflight_reads_linked_worktree_through_metadata_bridge(self):
        class CapturingSandbox(HostGitSandbox):
            def __init__(self) -> None:
                self.calls: list[tuple[tuple[str, ...], dict]] = []

            def run(self, repository, argv, **kwargs):
                self.calls.append((tuple(argv), kwargs))
                return super().run(repository, argv, **kwargs)

        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            sandbox = CapturingSandbox()
            worktree = IsolatedGitWorktree(source, sandbox, run_id="RUN-TOOLCHAIN-1")
            _source, baseline = worktree.begin()
            repository = worktree.create(baseline)

            environment = ToolchainService.load(
                AI_ROOT / "config" / "toolchains.json", sandbox=sandbox
            ).preflight(repository, ("python verify.py",), operation_id="RUN-TOOLCHAIN-1")

            snapshot_calls = [
                (argv, kwargs)
                for argv, kwargs in sandbox.calls
                if argv[0] == "git" and kwargs.get("component") == "toolchain-snapshot"
            ]
            self.assertEqual("python-nodejs", environment.profile_id)
            self.assertTrue(snapshot_calls)
            self.assertTrue(
                all(
                    kwargs["git_metadata"] is repository.git_metadata
                    and not kwargs["writable_git_metadata"]
                    and "--git-dir=/ai-agents-gitdir/" in " ".join(argv)
                    for argv, kwargs in snapshot_calls
                )
            )
            self.assertIsNone(worktree.cleanup())

    def test_worktree_creation_never_runs_a_host_checkout_filter(self):
        class NoCheckoutSandbox(HostGitSandbox):
            def __init__(self) -> None:
                self.calls: list[tuple[tuple[str, ...], dict]] = []

            def run(self, repository, argv, **kwargs):
                self.calls.append((tuple(argv), kwargs))
                if "checkout" in argv:
                    return subprocess.CompletedProcess(argv, 0, "", "")
                if "status" in argv:
                    return subprocess.CompletedProcess(argv, 0, "", "")
                return super().run(repository, argv, **kwargs)

        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            marker = root / "host-filter-ran.txt"
            filter_script = root / "smudge.py"
            filter_script.write_text(
                "import sys\nfrom pathlib import Path\n"
                "Path(sys.argv[1]).write_text('ran', encoding='utf-8')\n"
                "sys.stdout.write(sys.stdin.read())\n",
                encoding="utf-8",
            )
            filter_command = (
                subprocess.list2cmdline([sys.executable, str(filter_script), str(marker)])
                if os.name == "nt"
                else shlex.join([sys.executable, str(filter_script), str(marker)])
            )
            (source / ".gitattributes").write_text(
                "filtered.txt filter=host-filter\n", encoding="utf-8"
            )
            (source / "filtered.txt").write_text("tracked\n", encoding="utf-8")
            git(source, "config", "filter.host-filter.smudge", filter_command)
            git(source, "add", ".gitattributes", "filtered.txt")
            git(source, "commit", "-m", "configure checkout filter")
            self.assertFalse(marker.exists())

            sandbox = NoCheckoutSandbox()
            worktree = IsolatedGitWorktree(source, sandbox, run_id="RUN-NO-HOST-FILTER-1")
            _source, baseline = worktree.begin()
            repository = worktree.create(baseline)

            checkout_calls = [
                (argv, kwargs) for argv, kwargs in sandbox.calls if "checkout" in argv
            ]
            self.assertFalse(marker.exists())
            self.assertEqual(1, len(checkout_calls))
            self.assertIs(checkout_calls[0][1]["git_metadata"], repository.git_metadata)
            self.assertTrue(checkout_calls[0][1]["writable_git_metadata"])
            self.assertIsNone(worktree.cleanup())


if __name__ == "__main__":
    unittest.main()
