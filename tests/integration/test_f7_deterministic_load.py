from __future__ import annotations

from contextlib import closing
import json
import sqlite3
import unittest
from pathlib import Path

from tests.pipeline.support import (
    FakeRoleRunner,
    HostGitSandbox,
    build_pipeline,
    create_repository,
    temporary_directory,
)


class F7DeterministicLoadTests(unittest.TestCase):
    """유료 모델·Docker 없이 pipeline의 durable 경로를 반복 검증한다."""

    def test_one_hundred_pipeline_runs_leave_no_worktree_lock_or_queue_residue(self):
        with temporary_directory() as directory:
            root = Path(directory)
            run_ids = [f"RUN-F7-LOAD-{number:03d}" for number in range(100)]
            worktree_paths: list[Path] = []

            for run_id in run_ids:
                repository_root = root / "repositories" / run_id
                repository_root.mkdir(parents=True)
                repository = create_repository(repository_root)
                store, worker, _ = build_pipeline(
                    root,
                    repository,
                    FakeRoleRunner(),
                    sandbox=HostGitSandbox(),
                    run_id=run_id,
                )
                self.assertTrue(worker.run_once())
                self.assertEqual("COMPLETED", store.pipeline_job(run_id)["status"])

                record = json.loads(
                    (root / "artifacts" / run_id / "worktree.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual("cleaned", record["status"])
                worktree_paths.append(Path(record["worktree_path"]))
                self.assertTrue((root / "artifacts" / run_id / "events.jsonl").is_file())
                self.assertTrue((root / "artifacts" / run_id / "timeline.log").is_file())

            self.assertIsNone(store.claim_next_pipeline_job("f7-load-check"))
            self.assertTrue(all(not path.exists() for path in worktree_paths))
            self.assertEqual((), tuple(root.rglob("*.cid")))

            with closing(sqlite3.connect(root / "state.db")) as connection:
                self.assertEqual(
                    100,
                    connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
                )
                self.assertEqual(
                    100,
                    connection.execute(
                        "SELECT COUNT(*) FROM pipeline_jobs WHERE status = 'COMPLETED'"
                    ).fetchone()[0],
                )
                self.assertEqual(
                    0,
                    connection.execute(
                        "SELECT COUNT(*) FROM repository_execution_locks"
                    ).fetchone()[0],
                )
                self.assertEqual(
                    100,
                    connection.execute(
                        "SELECT COUNT(DISTINCT run_id) FROM events"
                    ).fetchone()[0],
                )



if __name__ == "__main__":
    unittest.main()
