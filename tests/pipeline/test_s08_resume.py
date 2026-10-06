from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from app.config import FoundationConfig
from app.contracts import RoleId, RunPhase, TokenUsage
from app.orchestrator import RunStateMachine
from app.pipeline import PipelineCoordinator, PipelineWorker
from app.services.budget import BudgetManager, BudgetPolicy
from app.services.git.repository import GitRepository
from app.services.git.worktree import IsolatedGitWorktree, GitWorktreeError
from app.services.hermes import HermesResult
from app.services.logging.audit import AuditLogger
from app.services.verification import VerificationRunner
from app.storage import ArtifactStore, StateStore
from app.storage.sqlite_store import StoreError
from tests.pipeline.support import AI_ROOT, FakeRoleRunner, HostGitSandbox, build_pipeline, create_repository, git, temporary_directory


def reopen_worker(root: Path, runner: FakeRoleRunner):
    store = StateStore(root / 'state.db')
    logger = AuditLogger(root / 'artifacts', store)
    machine = RunStateMachine(store, event_sink=logger, approval_phrase='개발 시작해')
    sandbox = HostGitSandbox()
    coordinator = PipelineCoordinator(
        store, machine, FoundationConfig.load(AI_ROOT), ArtifactStore(root / 'artifacts'), logger,
        BudgetManager(BudgetPolicy.load(AI_ROOT / 'config/limits.json'), store), runner,
        VerificationRunner(timeout_seconds=30, poll_seconds=0.05, sandbox=sandbox), sandbox=sandbox,
    )
    return store, PipelineWorker(store, machine, coordinator, logger, poll_seconds=0.01)


def expire_job(store, run_id):
    with closing(sqlite3.connect(store.path)) as connection, connection:
        connection.execute("UPDATE pipeline_jobs SET lease_until = '2000-01-01T00:00:00+00:00' WHERE run_id = ?", (run_id,))
        connection.execute("UPDATE repository_execution_locks SET lease_until = '2000-01-01T00:00:00+00:00' WHERE run_id = ?", (run_id,))


def resume(store, worker, run_id):
    paused = store.load_run(run_id)
    worker.machine.transition(paused, RunPhase.DEVELOPING, message='보존된 checkpoint 재개 검증')
    return worker.resume(run_id)


class AskingRunner(FakeRoleRunner):
    def __init__(self, *, dirty=False):
        super().__init__(review_responses=[[]], development_outputs=['fixed'])
        self.asked = False
        self.dirty = dirty
        self.observed = []

    def run(self, *args, **kwargs):
        repository = Path(args[3])
        if args[1] == 'stage-002' and args[2] == RoleId.DEVELOPMENT:
            self.observed.append((str(repository), git(repository, 'rev-parse', 'HEAD'), (repository / 'feature.txt').read_text(encoding='utf-8')))
            if not self.asked:
                self.asked = True
                if self.dirty:
                    (repository / 'feature.txt').write_text('partial\n', encoding='utf-8')
                return HermesResult(json.dumps({'summary': '확인 필요', 'needs_user_input': ['계속할까요?']}), TokenUsage(100, 20), 0.01)
        return super().run(*args, **kwargs)


def crash_child(root: str, boundary: str, temp_root: str):
    """Actual abrupt process exit: no coordinator/worker finally blocks run."""
    tempfile.tempdir = temp_root
    def mkdir_temp(suffix=None, prefix=None, dir=None):
        import uuid
        path = Path(dir or tempfile.gettempdir()) / ((prefix or 'tmp-') + uuid.uuid4().hex + (suffix or ''))
        path.mkdir(parents=True)
        return str(path)
    tempfile.mkdtemp = mkdir_temp
    store, worker = reopen_worker(Path(root), FakeRoleRunner(review_responses=[[]], development_outputs=['fixed']))
    commit = GitRepository.commit_scope
    if boundary in {'before_commit', 'after_commit'}:
        def interrupted_commit(repository, message, scope):
            if boundary == 'before_commit':
                os._exit(73)
            commit(repository, message, scope)
            os._exit(73)
        GitRepository.commit_scope = interrupted_commit
    elif boundary == 'after_apply':
        apply = worker.coordinator._apply_final_candidate
        def interrupted_apply(*args, **kwargs):
            apply(*args, **kwargs)
            os._exit(73)
        worker.coordinator._apply_final_candidate = interrupted_apply
    elif boundary == 'after_verified':
        worker.coordinator._finish_verified_stage = lambda *args, **kwargs: os._exit(73)
    elif boundary == 'stage_advance':
        save = store.save_pipeline_workspace
        def interrupted_advance(run_id, owner, checkpoint):
            if checkpoint['stage_index'] == 1:
                os._exit(73)
            save(run_id, owner, checkpoint)
        store.save_pipeline_workspace = interrupted_advance
    elif boundary == 'during_write':
        def interrupted_write(*args, **kwargs):
            (Path(args[3]) / 'feature.txt').write_text('fixed\n', encoding='utf-8')
            os._exit(73)
        worker.coordinator.runner.run = interrupted_write
    elif boundary == 'after_completed':
        execute = worker.coordinator.execute
        def interrupted_completion(*args, **kwargs):
            execute(*args, **kwargs)
            os._exit(73)
        worker.coordinator.execute = interrupted_completion
    else:
        raise ValueError(boundary)
    worker.run_once()
    raise AssertionError('crash boundary was not reached')


class S08ResumeTests(unittest.TestCase):
    def test_two_stages_question_restart_preserves_first_file_candidate_and_workspace(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            runner = AskingRunner()
            store, worker, run_id = build_pipeline(root, source, runner, stage_count=2)
            worker.run_once()
            checkpoint = store.pipeline_workspace(run_id)
            self.assertEqual(1, checkpoint['stage_index'])
            self.assertEqual(checkpoint['candidate_sha'], checkpoint['validated_sha'])
            self.assertFalse((source / 'feature.txt').exists())
            store.answer_execution_question(run_id, '진행해')
            store, worker = reopen_worker(root, runner)
            self.assertEqual('QUEUED', resume(store, worker, run_id))
            self.assertEqual('QUEUED', worker.resume(run_id))
            worker.run_once()
            self.assertEqual(RunPhase.COMPLETED, store.load_run(run_id).phase)
            self.assertEqual(runner.observed[0], runner.observed[1])
            self.assertEqual('fixed\n', (source / 'feature.txt').read_text(encoding='utf-8'))
            results = [message for message in store.list_messages(run_id) if message['kind'] == 'pipeline_result']
            self.assertEqual(git(source, 'rev-parse', 'HEAD'), results[-1]['data']['candidate_sha'])
            self.assertFalse(worker.run_once())
            self.assertEqual(2, len(runner.observed))

    def test_question_with_uncommitted_edit_resumes_the_exact_content(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            runner = AskingRunner(dirty=True)
            store, worker, run_id = build_pipeline(root, source, runner, stage_count=2)
            worker.run_once()
            self.assertTrue(store.pipeline_workspace(run_id)['dirty_fingerprint'])
            store.answer_execution_question(run_id, '진행해')
            resume(store, worker, run_id)
            worker.run_once()
            self.assertEqual('partial\n', runner.observed[-1][2])
            self.assertEqual(RunPhase.COMPLETED, store.load_run(run_id).phase)

    def test_clean_missing_workspace_rebuilds_from_preserved_candidate(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            runner = AskingRunner()
            store, worker, run_id = build_pipeline(root, source, runner, stage_count=2)
            worker.run_once()
            checkpoint = store.pipeline_workspace(run_id)
            owned_path = Path(checkpoint['worktree']['worktree_path'])
            owned_path.resolve().relative_to(Path(tempfile.gettempdir()).resolve())
            git(source, 'worktree', 'remove', '--force', str(owned_path))
            store.answer_execution_question(run_id, '진행해')
            resume(store, worker, run_id)
            worker.run_once()
            self.assertEqual(RunPhase.COMPLETED, store.load_run(run_id).phase)
            self.assertNotEqual(runner.observed[0][0], runner.observed[1][0])
            self.assertEqual(runner.observed[0][1:], runner.observed[1][1:])

    def test_dirty_missing_workspace_blocks_without_discarding_the_checkpoint(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            runner = AskingRunner(dirty=True)
            store, worker, run_id = build_pipeline(root, source, runner, stage_count=2)
            worker.run_once()
            checkpoint = store.pipeline_workspace(run_id)
            owned_path = Path(checkpoint['worktree']['worktree_path'])
            owned_path.resolve().relative_to(Path(tempfile.gettempdir()).resolve())
            git(source, 'worktree', 'remove', '--force', str(owned_path))
            store.answer_execution_question(run_id, '진행해')
            resume(store, worker, run_id)
            worker.run_once()
            self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
            self.assertEqual(checkpoint, store.pipeline_workspace(run_id))
            self.assertEqual(1, len(runner.observed))
            self.assertIn('미커밋', store.pipeline_job(run_id)['last_error'])

    def test_changed_dirty_content_and_changed_head_are_not_silently_reused(self):
        for change in ('dirty', 'head'):
            with self.subTest(change=change), temporary_directory() as directory:
                root = Path(directory)
                source = create_repository(root)
                runner = AskingRunner(dirty=change == 'dirty')
                store, worker, run_id = build_pipeline(root, source, runner, stage_count=2)
                worker.run_once()
                checkpoint = store.pipeline_workspace(run_id)
                workspace = Path(checkpoint['worktree']['worktree_path'])
                (workspace / 'feature.txt').write_text('external\n', encoding='utf-8')
                if change == 'head':
                    git(workspace, 'add', 'feature.txt')
                    git(workspace, 'commit', '-m', 'external edit')
                store.answer_execution_question(run_id, '진행해')
                resume(store, worker, run_id)
                worker.run_once()
                self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
                self.assertEqual(1, len(runner.observed))
                self.assertEqual('external\n', (workspace / 'feature.txt').read_text(encoding='utf-8'))
                self.assertFalse((source / 'feature.txt').exists())

    def test_source_head_change_on_resume_reports_candidate_location(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            runner = AskingRunner()
            store, worker, run_id = build_pipeline(root, source, runner, stage_count=2)
            worker.run_once()
            checkpoint = store.pipeline_workspace(run_id)
            (source / 'user.txt').write_text('user\n', encoding='utf-8')
            git(source, 'add', 'user.txt')
            git(source, 'commit', '-m', 'user change')
            user_head = git(source, 'rev-parse', 'HEAD')
            store.answer_execution_question(run_id, '진행해')
            resume(store, worker, run_id)
            worker.run_once()
            error = store.pipeline_job(run_id)['last_error']
            self.assertIn(checkpoint['candidate_sha'], error)
            self.assertIn(checkpoint['worktree']['worktree_path'], error)
            self.assertEqual(user_head, git(source, 'rev-parse', 'HEAD'))
            self.assertEqual(1, len(runner.observed))

    def test_checkpoint_stage_mismatch_stops_before_another_role(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            runner = AskingRunner()
            store, worker, run_id = build_pipeline(root, source, runner, stage_count=2)
            worker.run_once()
            checkpoint = store.pipeline_workspace(run_id)
            checkpoint['stage_index'] = 0
            with closing(sqlite3.connect(store.path)) as connection, connection:
                connection.execute('UPDATE pipeline_workspaces SET checkpoint_json = ? WHERE run_id = ?', (json.dumps(checkpoint), run_id))
            store.answer_execution_question(run_id, '진행해')
            resume(store, worker, run_id)
            worker.run_once()
            self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
            self.assertEqual(1, len(runner.observed))

    def _abrupt_exit(self, boundary):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            store, _worker, run_id = build_pipeline(root, source, FakeRoleRunner(), stage_count=2 if boundary == 'stage_advance' else 1)
            before = git(source, 'rev-parse', 'HEAD')
            code = 'from tests.pipeline.test_s08_resume import crash_child; import sys; crash_child(*sys.argv[1:])'
            result = subprocess.run(
                [sys.executable, '-B', '-X', 'utf8', '-c', code, str(root), boundary, tempfile.gettempdir()],
                cwd=AI_ROOT, capture_output=True, text=True, encoding='utf-8', timeout=60,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            )
            self.assertEqual(73, result.returncode, result.stdout + result.stderr)
            checkpoint = store.pipeline_workspace(run_id)
            workspace = Path(checkpoint['worktree']['worktree_path'])
            child_head = git(source if boundary == 'after_completed' else workspace, 'rev-parse', 'HEAD')
            self.assertEqual('fixed\n', ((source if boundary == 'after_completed' else workspace) / 'feature.txt').read_text(encoding='utf-8'))
            expire_job(store, run_id)
            class ResumedRunner(FakeRoleRunner):
                def __init__(self):
                    super().__init__(review_responses=[[]], development_outputs=['fixed'])
                    self.stages = []
                def run(self, *args, **kwargs):
                    self.stages.append(args[1])
                    return super().run(*args, **kwargs)
            runner = ResumedRunner()
            store, worker = reopen_worker(root, runner)
            worker._recover_stale()
            if boundary == 'after_completed':
                self.assertEqual(RunPhase.COMPLETED, store.load_run(run_id).phase)
                self.assertEqual('QUEUED', store.pipeline_job(run_id)['status'])
            else:
                self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
                resume(store, worker, run_id)
            reserved = store.reserved_token_total(run_id)
            worker.run_once()
            if boundary == 'during_write':
                self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
                self.assertEqual([], runner.calls)
                self.assertGreater(reserved, 0)
                self.assertEqual(reserved, store.reserved_token_total(run_id))
                self.assertTrue(store.has_budget_anomaly(run_id))
                self.assertEqual('fixed\n', (workspace / 'feature.txt').read_text(encoding='utf-8'))
                self.assertEqual(before, git(source, 'rev-parse', 'HEAD'))
                resume(store, worker, run_id)
                worker.run_once()
                self.assertEqual([], runner.calls)
                return
            self.assertEqual(RunPhase.COMPLETED, store.load_run(run_id).phase,
                             store.pipeline_job(run_id)['last_error'])
            self.assertEqual('fixed\n', (source / 'feature.txt').read_text(encoding='utf-8'))
            self.assertFalse(worker.run_once())
            self.assertEqual(2, store.pipeline_job(run_id)['attempts'])
            if boundary in {'after_verified', 'after_apply', 'after_completed'}:
                self.assertEqual([], runner.calls)
                self.assertEqual(child_head, git(source, 'rev-parse', 'HEAD'))
            if boundary == 'after_apply':
                application = json.loads((root / 'artifacts' / run_id / 'source-application.json').read_text(encoding='utf-8'))
                self.assertTrue(application['recovered'])
            if boundary == 'stage_advance':
                self.assertEqual(0, checkpoint['stage_index'])
                self.assertEqual('stage_verified', checkpoint['status'])
                self.assertTrue(runner.stages)
                self.assertEqual({'stage-002'}, set(runner.stages))
            self.assertNotEqual(before, git(source, 'rev-parse', 'HEAD'))

    def test_real_process_exit_before_stage_commit_and_restart(self):
        self._abrupt_exit('before_commit')

    def test_real_process_exit_after_stage_commit_and_restart(self):
        self._abrupt_exit('after_commit')

    def test_real_process_exit_after_verification_and_restart_skips_roles(self):
        self._abrupt_exit('after_verified')

    def test_real_process_exit_after_source_apply_recovers_without_duplicate_apply(self):
        self._abrupt_exit('after_apply')

    def test_stage_advance_and_checkpoint_are_atomic_during_real_process_exit(self):
        self._abrupt_exit('stage_advance')

    def test_real_process_exit_during_write_preserves_unknown_usage_and_blocks_reexecution(self):
        self._abrupt_exit('during_write')

    def test_real_process_exit_after_completed_state_finishes_the_queue_without_rerunning_roles(self):
        self._abrupt_exit('after_completed')

    def test_process_held_guard_blocks_a_live_stale_worker_until_it_exits(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            first = IsolatedGitWorktree(source, HostGitSandbox(), run_id='GUARD-TEST')
            first.acquire_execution_guard()
            _source, baseline = first.begin()
            first.create(baseline)
            second = IsolatedGitWorktree(source, HostGitSandbox(), run_id='GUARD-TEST', record=first.record.to_dict())
            try:
                with self.assertRaises(GitWorktreeError):
                    second.acquire_execution_guard()
                first.release_execution_guard()
                second.acquire_execution_guard()
                second.attach()
            finally:
                first.release_execution_guard()
                second.release_execution_guard()
                second.cleanup()

    def test_pause_after_a_write_then_resume_keeps_the_same_worktree(self):
        class PausingRunner(FakeRoleRunner):
            def __init__(self):
                super().__init__(review_responses=[[]], development_outputs=['fixed'])
                self.pause = lambda: None
                self.seen = []
            def run(self, *args, **kwargs):
                if args[2] == RoleId.DEVELOPMENT:
                    path = Path(args[3])
                    self.seen.append((str(path), (path / 'feature.txt').read_text(encoding='utf-8') if (path / 'feature.txt').exists() else None))
                result = super().run(*args, **kwargs)
                if len(self.seen) == 1 and args[2] == RoleId.DEVELOPMENT:
                    self.pause()
                return result
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            runner = PausingRunner()
            store, worker, run_id = build_pipeline(root, source, runner)
            runner.pause = lambda: worker.pause(run_id)
            worker.run_once()
            self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
            self.assertFalse((source / 'feature.txt').exists())
            resume(store, worker, run_id)
            worker.run_once()
            self.assertEqual(RunPhase.COMPLETED, store.load_run(run_id).phase)
            self.assertEqual(runner.seen[0][0], runner.seen[1][0])
            self.assertEqual('fixed\n', runner.seen[1][1])

    def test_final_application_rechecks_repository_approval(self):
        class RevokingRunner(FakeRoleRunner):
            def __init__(self):
                super().__init__(review_responses=[[]], development_outputs=['fixed'])
                self.revoke = lambda: None
            def run(self, *args, **kwargs):
                result = super().run(*args, **kwargs)
                if args[2] == RoleId.REVIEW:
                    self.revoke()
                return result
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            runner = RevokingRunner()
            store, worker, run_id = build_pipeline(root, source, runner)
            def revoke():
                with closing(sqlite3.connect(store.path)) as connection, connection:
                    connection.execute("UPDATE scoped_repository_approvals SET revoked_at = '2026-10-02T00:00:00+00:00'")
            runner.revoke = revoke
            worker.run_once()
            self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
            self.assertFalse((source / 'feature.txt').exists())
            self.assertEqual('stage_verified', store.pipeline_workspace(run_id)['status'])

    def test_workspace_migration_is_idempotent_on_an_existing_database(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            runner = AskingRunner()
            store, worker, run_id = build_pipeline(root, source, runner, stage_count=2)
            worker.run_once()
            before = store.pipeline_workspace(run_id)
            store = StateStore(store.path)
            self.assertEqual(before, store.pipeline_workspace(run_id))
            with closing(sqlite3.connect(store.path)) as connection, connection:
                self.assertEqual(1, connection.execute('SELECT COUNT(*) FROM schema_migrations WHERE version = 10').fetchone()[0])

    def test_status_displays_the_same_stage_and_candidate_as_the_checkpoint(self):
        from app.gateway.core.models import IncomingMessage
        from tests.gateway.support import build_application
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            store, worker, run_id = build_pipeline(root, source, AskingRunner(), stage_count=2)
            worker.run_once()
            checkpoint = store.pipeline_workspace(run_id)
            _store, application = build_application(root, object())
            binding = store.load_conversation('telegram', 'chat-1')
            status = application.router._status(IncomingMessage('telegram', 'chat-1', 'test-user', 'status-1', '상태'), binding)
            self.assertIn('작업 단계: 2/2', status.text)
            self.assertIn(checkpoint['candidate_sha'][:12], status.text)
            self.assertIn(checkpoint['validated_sha'][:12], status.text)
            self.assertIn(checkpoint['worktree']['worktree_path'], status.text)

    def test_expired_lease_cannot_renew_or_save_but_new_owner_can_take_over(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            store, worker, run_id = build_pipeline(root, source, FakeRoleRunner())
            job = store.claim_next_pipeline_job('old', lease_seconds=1)
            self.assertTrue(store.acquire_repository_lock(str(source), run_id, 'old', lease_seconds=1))
            expire_job(store, run_id)
            with self.assertRaises(StoreError):
                store.heartbeat_pipeline_job(run_id, 'old')
            with self.assertRaises(StoreError):
                store.renew_repository_lock(str(source), run_id, 'old')
            worker._recover_stale()
            store.requeue_pipeline_job(run_id)
            store.claim_next_pipeline_job('new')
            self.assertTrue(store.acquire_repository_lock(str(source), run_id, 'new'))
            with self.assertRaises(StoreError):
                store.save_pipeline_workspace(run_id, 'old', {})
            store.release_repository_lock(str(source), run_id, 'old')
            self.assertTrue(store.repository_lock_owned(str(source), run_id, 'new'))

    def test_live_pipeline_prevents_takeover_of_an_expired_repository_lock(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            store, worker, run_id = build_pipeline(root, source, FakeRoleRunner())
            store.claim_next_pipeline_job('owner')
            store.acquire_repository_lock(str(source), run_id, 'owner')
            with closing(sqlite3.connect(store.path)) as connection, connection:
                connection.execute("UPDATE repository_execution_locks SET lease_until = '2000-01-01T00:00:00+00:00'")
            self.assertFalse(store.acquire_repository_lock(str(source), run_id, 'other'))

    def test_worker_renews_lease_during_a_runner_that_does_not_heartbeat(self):
        class SlowRunner(FakeRoleRunner):
            def run(self, *args, **kwargs):
                if args[2] == RoleId.DEVELOPMENT:
                    time.sleep(3.5)
                return super().run(*args, **kwargs)
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            store, worker, run_id = build_pipeline(root, source, SlowRunner(review_responses=[[]], development_outputs=['fixed']))
            worker.lease_seconds = 3
            worker.run_once()
            self.assertEqual(RunPhase.COMPLETED, store.load_run(run_id).phase,
                             json.dumps(store.pipeline_job(run_id)) + '\n' + str(store.pipeline_workspace(run_id)))
            self.assertEqual('COMPLETED', store.pipeline_job(run_id)['status'])


if __name__ == '__main__':
    unittest.main()
