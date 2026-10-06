from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from app.contracts import RoleId, RunPhase, TokenUsage
from app.services.git.repository import GitRepository
from app.services.hermes import HermesCancelled, HermesExecutionError, HermesResult
from tests.pipeline.support import (
    AI_ROOT, FakeRoleRunner, HostGitSandbox, build_pipeline, create_repository, git,
    temporary_directory,
)
from tests.pipeline.test_s08_resume import crash_child, expire_job, reopen_worker, resume


def crash_invocation(root, role, boundary, temp_root):
    """Abrupt exits exercise SQLite durability without finally/exception cleanup."""
    tempfile.tempdir = temp_root
    import uuid
    def mkdtemp(suffix=None, prefix=None, dir=None):
        path = Path(dir or temp_root) / ((prefix or 'tmp-') + uuid.uuid4().hex + (suffix or ''))
        path.mkdir(parents=True)
        return str(path)
    tempfile.mkdtemp = mkdtemp
    selected = RoleId(role)
    class Runner(FakeRoleRunner):
        def run(self, *args, **kwargs):
            if args[2] == selected and boundary == 'running':
                if selected != RoleId.REVIEW:
                    (Path(args[3]) / 'feature.txt').write_text('partial\n', encoding='utf-8')
                os._exit(73)
            return super().run(*args, **kwargs)
    store, worker = reopen_worker(Path(root), Runner())
    if boundary in {'before_settle', 'inside_settle', 'after_settle'}:
        settle = worker.coordinator._settle_invocation
        def interrupted(repository, invocation, *args, **kwargs):
            if invocation['role_id'] == role:
                if boundary == 'before_settle':
                    os._exit(73)
                if boundary == 'inside_settle':
                    original = store.add_usage
                    def add_usage(*values, **options):
                        original(*values, **options)
                        os._exit(73)
                    store.add_usage = add_usage
                settle(repository, invocation, *args, **kwargs)
                os._exit(73)
            return settle(repository, invocation, *args, **kwargs)
        worker.coordinator._settle_invocation = interrupted
    if boundary == 'before_notice':
        worker.coordinator._complete_notice = lambda *args, **kwargs: os._exit(73)
    worker.run_once()
    raise AssertionError('requested crash boundary not reached')


def child(root, role, boundary):
    command = ('from tests.pipeline.test_s08_review_resolution import crash_invocation; '
               'import sys; crash_invocation(*sys.argv[1:])')
    result = subprocess.run(
        [sys.executable, '-B', '-X', 'utf8', '-c', command, str(root), role.value,
         boundary, tempfile.gettempdir()], cwd=AI_ROOT, capture_output=True, text=True,
        encoding='utf-8', timeout=60, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
    )
    if result.returncode != 73:
        raise AssertionError(result.stdout + result.stderr)


class S08ReviewResolutionTests(unittest.TestCase):
    def _unknown_crash(self, role, boundary='running'):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            before = git(source, 'rev-parse', 'HEAD')
            store, _, run_id = build_pipeline(root, source, FakeRoleRunner())
            child(root, role, boundary)
            cp = store.pipeline_workspace(run_id)
            invocation = cp['invocation']
            self.assertEqual(role.value, invocation['role_id'])
            self.assertEqual('RUNNING', invocation['status'])
            reservation = store.reserved_token_total(run_id)
            cost = store.usage_total(run_id)
            self.assertGreater(reservation, 0)
            expire_job(store, run_id)
            runner = FakeRoleRunner()
            store, worker = reopen_worker(root, runner)
            worker._recover_stale()
            for _ in range(2):
                resume(store, worker, run_id)
                worker.run_once()
                self.assertEqual([], runner.calls)
                self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
                self.assertEqual(reservation, store.reserved_token_total(run_id))
                self.assertEqual(cost, store.usage_total(run_id))
                self.assertEqual('UNKNOWN', store.model_call(invocation['call_id'])['status'])
            self.assertEqual(1, sum(e['event_type'] == 'MODEL_USAGE_UNKNOWN' for e in store.list_events(run_id)))
            self.assertEqual(before, git(source, 'rev-parse', 'HEAD'))
            self.assertTrue(Path(cp['worktree']['worktree_path']).exists())

    def test_development_process_exit_blocks_repeated_resume(self):
        self._unknown_crash(RoleId.DEVELOPMENT)

    def test_review_process_exit_blocks_repeated_resume(self):
        self._unknown_crash(RoleId.REVIEW)

    def test_improvement_process_exit_blocks_repeated_resume(self):
        self._unknown_crash(RoleId.IMPROVEMENT)

    def test_returned_result_before_durable_usage_still_blocks(self):
        self._unknown_crash(RoleId.REVIEW, 'before_settle')

    def test_exit_inside_usage_transaction_rolls_back_result_and_cost(self):
        self._unknown_crash(RoleId.REVIEW, 'inside_settle')

    def test_exit_after_atomic_result_and_usage_does_not_charge_that_call_twice(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            store, _, run_id = build_pipeline(root, source, FakeRoleRunner())
            child(root, RoleId.REVIEW, 'after_settle')
            call = store.pipeline_workspace(run_id)['invocation']
            self.assertEqual('COMPLETED', call['status'])
            self.assertEqual(0, store.reserved_token_total(run_id))
            self.assertEqual(240, store.usage_total(run_id))
            saved = json.loads(store.model_call(call['call_id'])['result_json'])
            self.assertIn('text', saved)
            expire_job(store, run_id)
            runner = FakeRoleRunner(review_responses=[[]], development_outputs=['fixed'])
            store, worker = reopen_worker(root, runner)
            worker._recover_stale()
            resume(store, worker, run_id)
            worker.run_once()
            self.assertEqual(RunPhase.COMPLETED, store.load_run(run_id).phase)
            self.assertEqual(480, store.usage_total(run_id))
            self.assertFalse(store.has_budget_anomaly(run_id))
            self.assertEqual(saved, json.loads(store.model_call(call['call_id'])['result_json']))

    def _error(self, role, *, kind='timeout', usage=None, startup=False):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            error_type = HermesCancelled if kind in {'pause', 'cancel'} else HermesExecutionError
            class Runner(FakeRoleRunner):
                control = lambda self: None
                def run(self, *args, **kwargs):
                    if args[2] == role:
                        self.calls.append((role, kwargs['allow_writes']))
                        if role != RoleId.REVIEW and not startup:
                            (Path(args[3]) / 'feature.txt').write_text('partial\n', encoding='utf-8')
                        self.control()
                        if kind == 'unexpected':
                            raise RuntimeError('unexpected invocation failure')
                        raise error_type('fixture interruption', usage=usage,
                                         category='startup' if startup else 'timeout')
                    return super().run(*args, **kwargs)
            runner = Runner()
            store, worker, run_id = build_pipeline(root, source, runner)
            if kind == 'pause':
                runner.control = lambda: worker.pause(run_id)
            elif kind == 'cancel':
                runner.control = lambda: worker.cancel(run_id)
            worker.run_once()
            cp = store.pipeline_workspace(run_id)
            prior_cost = 0 if role == RoleId.DEVELOPMENT else 120 if role == RoleId.REVIEW else 240
            unknown = not startup and (usage is None or usage.estimated)
            self.assertEqual(unknown, store.has_budget_anomaly(run_id))
            self.assertEqual(prior_cost + (usage.total_tokens if usage and not unknown else 0), store.usage_total(run_id))
            self.assertEqual(unknown, store.reserved_token_total(run_id) > 0)
            if startup:
                self.assertIsNone(cp['invocation'])
            else:
                self.assertEqual('UNKNOWN' if unknown else 'CANCELLED' if kind in {'pause', 'cancel'} else 'FAILED', cp['invocation']['status'])
            if role != RoleId.REVIEW and not startup:
                self.assertEqual('partial\n', (Path(cp['worktree']['worktree_path']) / 'feature.txt').read_text(encoding='utf-8'))
                self.assertTrue(cp['dirty_fingerprint'])
            if kind == 'cancel':
                self.assertEqual(RunPhase.CANCELLED, store.load_run(run_id).phase)
                return
            if kind == 'unexpected':
                self.assertEqual(RunPhase.FAILED, store.load_run(run_id).phase)
                return
            new_runner = FakeRoleRunner(review_responses=[[]], development_outputs=['fixed'])
            store, worker = reopen_worker(root, new_runner)
            cost = store.usage_total(run_id)
            reserved = store.reserved_token_total(run_id)
            resume(store, worker, run_id)
            worker.run_once()
            if unknown:
                self.assertEqual([], new_runner.calls)
                self.assertEqual(cost, store.usage_total(run_id))
                self.assertEqual(reserved, store.reserved_token_total(run_id))
            else:
                self.assertEqual(RunPhase.COMPLETED, store.load_run(run_id).phase)
                self.assertEqual(cost + 240, store.usage_total(run_id))

    def test_unknown_development_timeout_preserves_cost_and_files(self):
        self._error(RoleId.DEVELOPMENT)

    def test_unknown_review_timeout_does_not_retry(self):
        self._error(RoleId.REVIEW)

    def test_unknown_improvement_timeout_preserves_cost_and_files(self):
        self._error(RoleId.IMPROVEMENT)

    def test_pause_without_usage_preserves_unknown_call(self):
        self._error(RoleId.DEVELOPMENT, kind='pause')

    def test_cancel_without_usage_preserves_unknown_call(self):
        self._error(RoleId.REVIEW, kind='cancel')

    def test_timeout_with_reported_usage_settles_once(self):
        self._error(RoleId.DEVELOPMENT, usage=TokenUsage(12, 5))

    def test_reported_zero_usage_is_distinct_from_missing_report(self):
        self._error(RoleId.DEVELOPMENT, usage=TokenUsage())

    def test_pause_with_reported_usage_preserves_cost_on_resume(self):
        self._error(RoleId.DEVELOPMENT, kind='pause', usage=TokenUsage(12, 5))

    def test_cancel_with_reported_usage_settles_once(self):
        self._error(RoleId.REVIEW, kind='cancel', usage=TokenUsage(12, 5))

    def test_estimated_error_usage_is_conservatively_unknown(self):
        self._error(RoleId.DEVELOPMENT, usage=TokenUsage(12, 5, estimated=True))

    def test_proven_startup_failure_releases_its_reservation(self):
        self._error(RoleId.DEVELOPMENT, startup=True)

    def test_unexpected_exception_preserves_unknown_cost(self):
        self._error(RoleId.DEVELOPMENT, kind='unexpected')

    def test_reported_read_timeout_retries_with_separately_accounted_usage(self):
        class Runner(FakeRoleRunner):
            interrupted = False
            def run(self, *args, **kwargs):
                if args[2] == RoleId.REVIEW and not self.interrupted:
                    self.interrupted = True
                    raise HermesExecutionError('known timeout', category='timeout', usage=TokenUsage(12, 5))
                return super().run(*args, **kwargs)
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            store, worker, run_id = build_pipeline(root, source, Runner(review_responses=[[]], development_outputs=['fixed']))
            worker.run_once()
            self.assertEqual(RunPhase.COMPLETED, store.load_run(run_id).phase)
            self.assertEqual(257, store.usage_total(run_id))
            self.assertEqual(0, store.reserved_token_total(run_id))
            self.assertFalse(store.has_budget_anomaly(run_id))

    def _net_zero(self, restore=False, interrupt=False):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            if restore:
                (source / 'feature.txt').write_text('original\n', encoding='utf-8')
            (source / 'verify.py').write_text(
                "from pathlib import Path\np=Path('feature.txt')\n"
                "assert not p.exists() or p.read_text().strip() in {'fixed', 'original'}\n", encoding='utf-8')
            git(source, 'add', '-A')
            git(source, 'commit', '-m', 'net zero verification fixture')
            initial = git(source, 'rev-parse', 'HEAD')
            class Runner(FakeRoleRunner):
                def run(self, *args, **kwargs):
                    if args[1] == 'stage-002' and args[2] == RoleId.DEVELOPMENT:
                        self.calls.append((args[2], kwargs['allow_writes']))
                        path = Path(args[3]) / 'feature.txt'
                        if restore:
                            path.write_text('original\n', encoding='utf-8')
                        else:
                            path.unlink()
                        return HermesResult('{"summary":"초기 내용 복원","needs_user_input":[]}', TokenUsage(100, 20), 0.01)
                    return super().run(*args, **kwargs)
            runner = Runner(review_responses=[[]], development_outputs=['fixed'])
            store, worker, run_id = build_pipeline(root, source, runner, stage_count=2)
            if interrupt:
                class Interrupted(BaseException):
                    pass
                apply = worker.coordinator._apply_final_candidate
                def interrupted(*args, **kwargs):
                    apply(*args, **kwargs)
                    raise Interrupted()
                worker.coordinator._apply_final_candidate = interrupted
                with self.assertRaises(Interrupted):
                    worker.run_once()
                expire_job(store, run_id)
                resumed = FakeRoleRunner()
                store, worker = reopen_worker(root, resumed)
                worker._recover_stale()
                resume(store, worker, run_id)
                worker.run_once()
                self.assertEqual([], resumed.calls)
            else:
                worker.run_once()
            cp = store.pipeline_workspace(run_id)
            sha = git(source, 'rev-parse', 'HEAD')
            self.assertNotEqual(initial, sha)
            self.assertEqual('', git(source, 'diff', '--stat', initial, sha))
            self.assertEqual(cp['candidate_sha'], sha)
            self.assertEqual(cp['validated_sha'], sha)
            self.assertEqual('COMPLETED', store.pipeline_job(run_id)['status'])
            for name in ('source-application.json', 'stages/stage-002/verification.json'):
                data = json.loads((root / 'artifacts' / run_id / name).read_text(encoding='utf-8'))
                self.assertEqual(sha, data['applied_sha' if name.startswith('source') else 'source_applied_sha'])
            result = [m for m in store.list_messages(run_id) if m['kind'] == 'pipeline_result'][-1]
            self.assertEqual(sha, result['data']['candidate_sha'])

    def test_add_then_remove_applies_exact_net_zero_candidate(self):
        self._net_zero()

    def test_modify_then_restore_applies_exact_net_zero_candidate(self):
        self._net_zero(restore=True)

    def test_net_zero_apply_interruption_recovers_without_another_role(self):
        self._net_zero(interrupt=True)

    def test_same_sha_candidate_is_a_real_noop(self):
        with temporary_directory() as directory:
            source = create_repository(Path(directory))
            repository = GitRepository(source, sandbox=HostGitSandbox())
            snapshot = repository.snapshot()
            self.assertEqual(snapshot.head, repository.apply_candidate(snapshot, snapshot.head, ('feature.txt',)))

    def test_empty_commit_candidate_fast_forwards_even_with_empty_diff(self):
        with temporary_directory() as directory:
            source = create_repository(Path(directory))
            repository = GitRepository(source, sandbox=HostGitSandbox())
            base = repository.snapshot()
            git(source, 'checkout', '--detach')
            git(source, 'commit', '--allow-empty', '-m', 'empty candidate')
            sha = git(source, 'rev-parse', 'HEAD')
            git(source, 'checkout', base.branch)
            self.assertEqual(sha, repository.apply_candidate(base, sha, ('feature.txt',)))
            self.assertEqual(sha, git(source, 'rev-parse', 'HEAD'))

    def test_adapter_returning_wrong_applied_sha_cannot_complete(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            initial = git(source, 'rev-parse', 'HEAD')
            store, worker, run_id = build_pipeline(root, source, FakeRoleRunner())
            with patch.object(GitRepository, 'apply_candidate', return_value=initial):
                worker.run_once()
            self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
            application = json.loads((root / 'artifacts' / run_id / 'source-application.json').read_text(encoding='utf-8'))
            self.assertFalse(application['applied'])

    def _completed_recovery(self, change, boundary='after_completed'):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            store, _, run_id = build_pipeline(root, source, FakeRoleRunner())
            if boundary == 'before_notice':
                child(root, RoleId.REVIEW, boundary)
            else:
                command = 'from tests.pipeline.test_s08_resume import crash_child; import sys; crash_child(*sys.argv[1:])'
                result = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-c', command,
                    str(root), 'after_completed', tempfile.gettempdir()], cwd=AI_ROOT,
                    capture_output=True, text=True, encoding='utf-8', timeout=60,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                self.assertEqual(73, result.returncode, result.stdout + result.stderr)
            cp = store.pipeline_workspace(run_id)
            with closing(sqlite3.connect(store.path)) as connection, connection:
                if change == 'revoked':
                    connection.execute("UPDATE scoped_repository_approvals SET revoked_at = '2026-10-02T00:00:00+00:00'")
                elif change == 'expired':
                    connection.execute("UPDATE scoped_repository_approvals SET expires_at = '2000-01-01T00:00:00+00:00'")
            if change == 'user_change':
                (source / 'user.txt').write_text('user\n', encoding='utf-8')
                git(source, 'add', 'user.txt')
                git(source, 'commit', '-m', 'user changed source after completion')
                (source / 'dirty.txt').write_text('preserve\n', encoding='utf-8')
            head = git(source, 'rev-parse', 'HEAD')
            status = git(source, 'status', '--porcelain')
            cost = store.usage_total(run_id)
            expire_job(store, run_id)
            runner = FakeRoleRunner()
            store, worker = reopen_worker(root, runner)
            worker._recover_stale()
            with (patch.object(worker.coordinator, '_validate_approval', side_effect=AssertionError('no revalidation')),
                 patch.object(worker.coordinator, '_open_workspace', side_effect=AssertionError('no workspace')),
                 patch.object(worker.coordinator.verifier.policy, 'prepare', side_effect=AssertionError('no environment'))):
                worker.run_once()
                self.assertFalse(worker.run_once())
            self.assertEqual('COMPLETED', store.pipeline_job(run_id)['status'])
            self.assertEqual(RunPhase.COMPLETED, store.load_run(run_id).phase)
            self.assertEqual([], runner.calls)
            self.assertEqual(cost, store.usage_total(run_id))
            self.assertEqual(head, git(source, 'rev-parse', 'HEAD'))
            self.assertEqual(status, git(source, 'status', '--porcelain'))
            results = [m for m in store.list_messages(run_id) if m['kind'] == 'pipeline_result']
            self.assertEqual(1, len(results))
            self.assertEqual(cp['validated_sha'], results[0]['data']['validated_sha'])
            with closing(sqlite3.connect(store.path)) as connection, connection:
                count = connection.execute("SELECT COUNT(*) FROM outbound_messages WHERE text = '모든 단계를 완료했습니다.'").fetchone()[0]
            self.assertEqual(1, count)

    def test_completed_queue_recovers_after_approval_revocation(self):
        self._completed_recovery('revoked')

    def test_completed_queue_recovers_after_approval_expiry(self):
        self._completed_recovery('expired')

    def test_completed_queue_preserves_later_user_commit_and_dirty_files(self):
        self._completed_recovery('user_change')

    def test_completed_queue_needs_no_git_or_verification_environment(self):
        self._completed_recovery('environment')

    def test_completed_before_notice_recovers_one_completion_notice(self):
        self._completed_recovery('revoked', 'before_notice')


class HermesUsageReportTests(unittest.TestCase):
    def test_runner_distinguishes_missing_zero_and_nonzero_reports_on_timeout(self):
        from tests.pipeline.test_hermes_contract import HermesContractTests, FakeProcess
        # Reuse only the established fixture, without inheriting its test cases.
        HermesContractTests.setUp(self)
        for usage in (None, {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0},
                      {'input_tokens': 12, 'output_tokens': 5, 'total_tokens': 17}):
            with self.subTest(usage=usage):
                class Stalled(FakeProcess):
                    returncode = None
                    def communicate(self, timeout=None):
                        raise subprocess.TimeoutExpired('fixture', timeout)
                    def poll(self):
                        return None
                def start(command, **kwargs):
                    process = Stalled(command, None, usage=usage)
                    process.returncode = None
                    return process
                with (patch('app.services.hermes.runner.subprocess.Popen', side_effect=start),
                      patch('app.services.hermes.runner.ProcessTree'),
                      patch('app.services.hermes.runner.time.monotonic', side_effect=[0, 2])):
                    with self.assertRaises(HermesExecutionError) as raised:
                        self.runner.run('RUN-CONTRACT', 'chat-001', RoleId.REVIEW,
                                        self.root, '질문', allow_writes=False)
                # Hermes' all-zero aggregate does not establish provider usage;
                # only a validated nonzero report is evidence at this boundary.
                self.assertEqual(bool(usage and usage['total_tokens']), raised.exception.usage_reported)
                self.assertEqual(usage['total_tokens'] if usage else 0, raised.exception.usage.total_tokens)


if __name__ == '__main__':
    unittest.main()
