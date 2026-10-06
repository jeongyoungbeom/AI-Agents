from __future__ import annotations

import json
import sqlite3
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from app.contracts import RoleId, RunPhase, TokenUsage
from app.gateway.core import AccessPolicy, IncomingMessage
from app.services.budget import BudgetExceeded, BudgetManager, BudgetPolicy
from app.services.hermes import HermesResult
from tests.gateway.support import build_application
from tests.pipeline.support import FakeRoleRunner, build_pipeline, create_repository, git, temporary_directory
from tests.pipeline.test_s08_resume import reopen_worker, resume


class OverrunRunner(FakeRoleRunner):
    def __init__(self):
        super().__init__(review_responses=[[]], development_outputs=['fixed'])

    def run(self, *args, **kwargs):
        result = super().run(*args, **kwargs)
        if len(self.calls) == 1:
            return HermesResult(result.text, TokenUsage(total_tokens=100000), result.elapsed_seconds)
        return result


class S11BudgetResumeTests(unittest.TestCase):
    def paused(self, root, *, stages=1):
        source = create_repository(root)
        runner = OverrunRunner()
        store, worker, run_id = build_pipeline(root, source, runner, stage_count=stages)
        self.assertTrue(worker.run_once())
        self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase,
                         store.pipeline_job(run_id)['last_error'])
        self.assertEqual([(RoleId.DEVELOPMENT, True)], runner.calls)
        self.assertEqual('', git(Path(store.pipeline_workspace(run_id)['worktree']['worktree_path']), 'status', '--porcelain'))
        manager = worker.coordinator.budget
        actor = dict(channel='telegram', conversation_id='chat-1', user_id='test-user', request_id='control-1')
        event = next(item for item in store.budget_control_events(run_id)
                     if item['event_type'] == 'BUDGET_RESERVATION_EXCEEDED')
        return store, worker, run_id, source, runner, manager, actor, event

    def application(self, root, worker, manager):
        # Production constructs gateway, worker and budget with one StateStore.
        with patch('tests.gateway.support.StateStore', return_value=worker.store):
            _, app = build_application(root, None, pipeline_scheduler=worker, budget=manager)
        app.access_policy = AccessPolicy(allowed_users=frozenset({'test-user', 'other'}))
        return app

    def message(self, number, text, **kwargs):
        return IncomingMessage('telegram', 'chat-1', kwargs.pop('user_id', 'test-user'), str(number), text, **kwargs)

    def test_gateway_reset_ack_manual_resume_preserves_candidate_and_skips_completed_builder(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store, worker, run_id, source, runner, manager, actor, event = self.paused(root)
            checkpoint = store.pipeline_workspace(run_id)
            original_sha = git(source, 'rev-parse', 'HEAD')
            original_call = store.model_call(checkpoint['invocation']['call_id'])
            app = self.application(root, worker, manager)
            denied = app.handle(self.message(1, '/resume'))
            self.assertEqual('BUDGET_ANOMALY', denied[0].metadata['request_result']['reason'])
            self.assertEqual('NEEDS_ATTENTION', worker.status(run_id))
            reset = app.handle(self.message(2, '/budget_reset'))
            self.assertIn('누적', reset[0].text)
            self.assertEqual(0, store.budget_usage_total(run_id))
            self.assertEqual(100000, store.usage_total(run_id))
            self.assertTrue(store.has_budget_anomaly(run_id))
            self.assertEqual(checkpoint, store.pipeline_workspace(run_id))
            ack = app.handle(self.message(3, f"/budget_ack {event['event_id']} 100000"))
            self.assertEqual('success', ack[0].metadata['request_result']['outcome'])
            self.assertFalse(store.has_budget_anomaly(run_id))
            self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
            self.assertEqual(original_sha, git(source, 'rev-parse', 'HEAD'))
            queued = app.handle(self.message(4, '/resume'))
            self.assertIn('QUEUED', queued[0].text)
            self.assertTrue(worker.run_once())
            self.assertEqual(RunPhase.COMPLETED, store.load_run(run_id).phase,
                             store.pipeline_job(run_id)['last_error'])
            self.assertEqual([(RoleId.DEVELOPMENT, True), (RoleId.REVIEW, False)], runner.calls)
            self.assertEqual(checkpoint['candidate_sha'], git(source, 'rev-parse', 'HEAD'))
            self.assertEqual(100120, store.usage_total(run_id))
            self.assertEqual(120, store.budget_usage_total(run_id))
            self.assertEqual(original_call, store.model_call(original_call['call_id']))
            self.assertEqual(checkpoint['candidate_sha'], store.pipeline_workspace(run_id)['validated_sha'])

    def test_legacy_overrun_and_restart_reuse_same_stage_then_next_stage_executes_once(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store, worker, run_id, source, runner, manager, actor, event = self.paused(root, stages=2)
            # The retained event538 format predates reservation_id in the event.
            with closing(sqlite3.connect(store.path)) as connection, connection:
                data = {k: v for k, v in event['data'].items() if k != 'reservation_id'}
                connection.execute('UPDATE events SET data_json=? WHERE event_id=?',
                                   (json.dumps(data), event['event_id']))
            manager.reset_budget(run_id, **actor)
            manager.acknowledge_overrun(run_id, event['event_id'], 100000, **actor)
            candidate = store.pipeline_workspace(run_id)['candidate_sha']
            restarted_runner = FakeRoleRunner(review_responses=[[]], development_outputs=['fixed'])
            reopened, restarted = reopen_worker(root, restarted_runner)
            self.assertEqual(0, reopened.budget_usage_total(run_id))
            self.assertFalse(reopened.has_budget_anomaly(run_id))
            self.assertEqual('QUEUED', resume(reopened, restarted, run_id))
            restarted.run_once()
            self.assertEqual(RunPhase.COMPLETED, reopened.load_run(run_id).phase,
                             reopened.pipeline_job(run_id)['last_error'])
            self.assertEqual([(RoleId.REVIEW, False), (RoleId.DEVELOPMENT, True),
                              (RoleId.REVIEW, False)], restarted_runner.calls)
            self.assertEqual(100360, reopened.usage_total(run_id))
            reused = [e for e in reopened.list_events(run_id) if e['event_type'] == 'DEVELOPMENT_RESULT_REUSED']
            self.assertEqual([candidate], [e['data']['candidate_sha'] for e in reused])

    def test_wrong_owner_binding_report_or_event_cannot_change_budget(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store, worker, run_id, source, runner, manager, actor, event = self.paused(root)
            before = store.budget_control_events(run_id)
            for altered in ({**actor, 'user_id': 'other'}, {**actor, 'conversation_id': 'wrong'},
                            {**actor, 'request_id': ''}):
                with self.subTest(actor=altered), self.assertRaises(BudgetExceeded):
                    manager.reset_budget(run_id, **altered)
            for event_id, tokens in ((event['event_id'], 99999), (event['event_id'] + 100, 100000)):
                with self.subTest(event_id=event_id), self.assertRaises(BudgetExceeded):
                    manager.acknowledge_overrun(run_id, event_id, tokens, **actor)
            app = self.application(root, worker, manager)
            reply = app.handle(self.message(1, '/budget_reset', user_id='other'))
            self.assertEqual('ACCESS_DENIED', reply[0].metadata['request_result']['reason'])
            self.assertEqual(before, store.budget_control_events(run_id))
            self.assertTrue(store.has_budget_anomaly(run_id))

    def test_unknown_cost_active_reservation_and_running_call_block_both_controls(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store, worker, run_id, source, runner, manager, actor, event = self.paused(root)
            store.create_token_reservation('active', run_id, 'chat', 'development', 'conversation', 500)
            for action in ('reset', 'ack'):
                with self.subTest(action=action), self.assertRaises(BudgetExceeded):
                    (manager.reset_budget(run_id, **actor) if action == 'reset' else
                     manager.acknowledge_overrun(run_id, event['event_id'], 100000, **actor))
            store.release_token_reservation('active')
            store.create_model_call('unfinished', 'unfinished', run_id, 'chat', 'development',
                                    'conversation', 'active', '')
            with self.assertRaises(BudgetExceeded):
                manager.reset_budget(run_id, **actor)
            store.finish_model_call('unfinished', 'UNKNOWN', {})
            store.append_event(run_id, '2026-10-04', 'MODEL_USAGE_UNKNOWN', 'unknown remains')
            manager.record_usage(run_id, 'chat', 'development', 'conversation', TokenUsage(total_tokens=500, estimated=True))
            with self.assertRaises(BudgetExceeded):
                manager.reset_budget(run_id, **actor)
            with self.assertRaises(BudgetExceeded):
                manager.acknowledge_overrun(run_id, event['event_id'], 100000, **actor)
            self.assertEqual(500, store.usage_breakdown(run_id)['estimated_tokens'])
            self.assertEqual(0, store.budget_usage_cursor(run_id))
            self.assertTrue(store.has_budget_anomaly(run_id))

    def test_duplicate_controls_preserve_all_records_and_new_overrun_reblocks(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store, worker, run_id, source, runner, manager, actor, event = self.paused(root)
            manager.reset_budget(run_id, **actor)
            manager.acknowledge_overrun(run_id, event['event_id'], 100000, **actor)
            before = store.budget_control_events(run_id)
            manager.reset_budget(run_id, **{**actor, 'request_id': 'again'})
            manager.acknowledge_overrun(run_id, event['event_id'], 100000, **{**actor, 'request_id': 'again'})
            self.assertEqual(before, store.budget_control_events(run_id))
            reservation = manager.reserve(run_id, 'stage-001', 'review', 'agent', 50)
            manager.record_usage(run_id, 'stage-001', 'review', 'agent', TokenUsage(total_tokens=60), reservation=reservation)
            manager.reset_budget(run_id, **actor)
            self.assertEqual(60, store.budget_usage_total(run_id))
            self.assertTrue(store.has_budget_anomaly(run_id))
            self.assertFalse(manager.can_spend(run_id, 'stage-001', 1).allowed)
            self.assertEqual(100060, store.usage_total(run_id))

    def test_new_epoch_checks_all_numeric_limits_and_keeps_completion_reserve(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store, worker, run_id, source, runner, manager, actor, event = self.paused(root)
            manager.reset_budget(run_id, **actor)
            manager.acknowledge_overrun(run_id, event['event_id'], 100000, **actor)
            manager.policy = BudgetPolicy(conversation_tokens=100, per_stage_tokens=160,
                                          whole_task_tokens=220, completion_reserve_tokens=20)
            manager.record_usage(run_id, 'stage-001', 'development', 'conversation', TokenUsage(total_tokens=90))
            self.assertFalse(manager.can_spend(run_id, 'stage-001', 11, category='conversation').allowed)
            self.assertFalse(manager.can_spend(run_id, 'stage-001', 71).allowed)
            self.assertFalse(manager.can_spend(run_id, 'stage-002', 111).allowed)
            self.assertTrue(manager.can_spend(run_id, 'stage-002', 130, use_completion_reserve=True).allowed)
            self.assertIn('100,090', manager.render_status(run_id))
            self.assertIn('초기화 후 예산 사용: 90', manager.render_status(run_id))
            reservation = manager.reserve(run_id, 'stage-002', 'review', 'agent', 100)
            self.assertFalse(manager.can_spend(run_id, 'stage-002', 11).allowed)
            self.assertEqual(100, store.reserved_token_total(run_id))
            manager.release_reservation(reservation)

    def test_transaction_rollback_and_readiness_checks_leave_no_control_marker(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store, worker, run_id, source, runner, manager, actor, event = self.paused(root)
            before = store.budget_control_events(run_id)
            with self.assertRaises(RuntimeError), store.transaction():
                manager.reset_budget(run_id, **actor)
                manager.acknowledge_overrun(run_id, event['event_id'], 100000, **actor)
                raise RuntimeError('rollback')
            self.assertEqual(before, store.budget_control_events(run_id))
            store.set_conversation_task('telegram', 'chat-1', None)
            with self.assertRaises(BudgetExceeded):
                manager.reset_budget(run_id, **actor)
            store.set_conversation_task('telegram', 'chat-1', run_id)
            store.save_run(replace(store.load_run(run_id), phase=RunPhase.DEVELOPING))
            with self.assertRaises(BudgetExceeded):
                manager.reset_budget(run_id, **actor)

    def test_parallel_reset_and_reservation_are_serialized_across_store_instances(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store, worker, run_id, source, runner, manager, actor, event = self.paused(root)
            manager.acknowledge_overrun(run_id, event['event_id'], 100000, **actor)
            peer = BudgetManager(manager.policy, type(store)(store.path))
            barrier = threading.Barrier(2)
            def reset_budget():
                barrier.wait(timeout=10)
                try:
                    manager.reset_budget(run_id, **actor)
                    return 'reset'
                except BudgetExceeded:
                    return 'reserved-first'
            def reserve_tokens():
                barrier.wait(timeout=10)
                return peer.reserve(run_id, 'stage-001', 'review', 'agent', 1)
            with ThreadPoolExecutor(max_workers=2) as pool:
                resetting = pool.submit(reset_budget)
                reserving = pool.submit(reserve_tokens)
                outcome = resetting.result(timeout=15)
                reservation = reserving.result(timeout=15)
            self.assertEqual(100000, store.usage_total(run_id))
            self.assertEqual(1, store.reserved_token_total(run_id))
            self.assertEqual(0 if outcome == 'reset' else 100000, store.budget_usage_total(run_id))
            self.assertEqual(store.budget_usage_total(run_id) + 1,
                             manager.can_spend(run_id, 'stage-001', 1).task_used)
            manager.release_reservation(reservation)

    def test_changed_completed_usage_blocks_reuse_without_new_builder_or_settlement(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store, worker, run_id, source, runner, manager, actor, event = self.paused(root)
            checkpoint = store.pipeline_workspace(run_id)
            source_sha = git(source, 'rev-parse', 'HEAD')
            manager.reset_budget(run_id, **actor)
            manager.acknowledge_overrun(run_id, event['event_id'], 100000, **actor)
            call = store.model_call(checkpoint['invocation']['call_id'])
            result = json.loads(call['result_json'])
            result['usage']['total_tokens'] = 123
            with closing(sqlite3.connect(store.path)) as connection, connection:
                connection.execute('UPDATE model_calls SET result_json=? WHERE call_id=?',
                                   (json.dumps(result), call['call_id']))
            self.assertEqual('QUEUED', resume(store, worker, run_id))
            worker.run_once()
            self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
            self.assertEqual([(RoleId.DEVELOPMENT, True)], runner.calls)
            self.assertEqual(100000, store.usage_total(run_id))
            self.assertEqual(source_sha, git(source, 'rev-parse', 'HEAD'))
            self.assertEqual(checkpoint['candidate_sha'], store.pipeline_workspace(run_id)['candidate_sha'])


if __name__ == '__main__':
    unittest.main()
