from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from app.contracts import RunPhase
from app.pipeline.coordinator import PipelineSteeringPending
from tests.pipeline.support import (
    FakeRoleRunner, build_pipeline, create_repository, git, temporary_directory,
)


class S11ReservationRecoveryTests(unittest.TestCase):
    def test_pending_conversation_reservation_does_not_become_unknown_pipeline_call(self):
        self._resume_with_reservation('conversation')

    def test_orphaned_agent_reservation_still_blocks_before_another_call(self):
        self._resume_with_reservation('agent')

    def _resume_with_reservation(self, category):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            before = git(source, 'rev-parse', 'HEAD')
            runner = FakeRoleRunner(review_responses=[[]], development_outputs=['fixed'])
            store, worker, run_id = build_pipeline(root, source, runner)
            # An input arriving during workspace setup can pause before any
            # pipeline invocation exists, while its conversation call runs.
            with patch.object(worker.coordinator, '_steering_boundary',
                              side_effect=PipelineSteeringPending('STEERING_PENDING')):
                worker.run_once()
            checkpoint = store.pipeline_workspace(run_id)
            self.assertFalse(checkpoint.get('invocation'))
            self.assertEqual([], runner.calls)
            store.create_token_reservation('concurrent-input', run_id, 'chat-input',
                                           'development', category, 1000)
            worker.run_once()
            self.assertEqual(1000, store.reserved_token_total(run_id, category=category))
            if category == 'conversation':
                self.assertEqual(RunPhase.COMPLETED, store.load_run(run_id).phase,
                                 store.pipeline_job(run_id)['last_error'])
                self.assertFalse(store.has_budget_anomaly(run_id))
                self.assertEqual(2, len(runner.calls))
                self.assertEqual('fixed\n', (source / 'feature.txt').read_text(encoding='utf-8'))
            else:
                self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
                self.assertTrue(store.has_budget_anomaly(run_id))
                self.assertEqual([], runner.calls)
                self.assertEqual(before, git(source, 'rev-parse', 'HEAD'))
                self.assertEqual(checkpoint, store.pipeline_workspace(run_id))


if __name__ == '__main__':
    unittest.main()
