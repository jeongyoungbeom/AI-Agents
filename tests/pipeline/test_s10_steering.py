from __future__ import annotations

import json
import unittest
import threading
from dataclasses import replace
from pathlib import Path

from app.contracts import RoleId, RunPhase
from app.gateway.core import AccessPolicy, AgentReply, DialogueRouter, GatewayApplication, IncomingMessage
from app.gateway.conversation_worker import ConversationQueue, ConversationWorker
from app.services.context import ContextService
from app.storage import ArtifactStore
from tests.gateway.support import FakeRepositoryValidator
from tests.pipeline.support import FakeRoleRunner, build_pipeline, create_repository, git, temporary_directory
from tests.pipeline.test_s08_resume import reopen_worker


def incoming(identifier, text, *, user='test-user', metadata=None):
    return IncomingMessage('telegram', 'chat-1', user, str(identifier), text, metadata=metadata or {})


class SteeringBackend:
    supports_cancellation = True
    def __init__(self, action='supplement', paths=()):
        self.action, self.paths = action, paths
        self.calls = []
        self.before_reply = None

    def respond_as(self, state, context, message, role_id, **kwargs):
        self.calls.append((state.run_id, role_id, message.text, kwargs['call_purpose']))
        if self.before_reply:
            self.before_reply()
        return AgentReply('현재 작업을 이어서 확인합니다.', execution_intent={
            'action': self.action, 'requested_scope': list(self.paths),
        })


def gateway(root, store, worker, backend, *, queued=False):
    logger = worker.logger
    queue = ConversationQueue(store) if queued else None
    router = DialogueRouter(store, worker.machine, ContextService(store), ArtifactStore(root / 'artifacts'),
        logger, object(), FakeRepositoryValidator(root / 'repository'), team_backend=backend,
        pipeline_scheduler=worker, conversation_scheduler=queue)
    application = GatewayApplication(store, router, AccessPolicy(allowed_users=frozenset({'test-user'})),
                                     conversation_scheduler=queue)
    conversation = ConversationWorker(store, router, poll_seconds=0.01) if queued else None
    return application, conversation


class InterleavingRunner(FakeRoleRunner):
    def __init__(self, callback=None):
        super().__init__(review_responses=[[]], development_outputs=['fixed'])
        self.callback = callback
        self.prompts = []
        self.paths = []
        self.contents_before = []

    def run(self, *args, **kwargs):
        self.prompts.append(args[4])
        self.paths.append(str(args[3]))
        path = Path(args[3]) / 'feature.txt'
        self.contents_before.append(path.read_text(encoding='utf-8') if path.exists() else None)
        result = super().run(*args, **kwargs)
        if self.callback:
            callback, self.callback = self.callback, None
            callback()
        return result


class S10SteeringTests(unittest.TestCase):
    def test_direction_arriving_with_old_result_runs_tests_without_followup_writes(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            runner = InterleavingRunner()
            store, worker, run_id = build_pipeline(root, source, runner)
            backend = SteeringBackend('test_first')
            application, conversation = gateway(root, store, worker, backend, queued=True)
            runner.callback = lambda: application.handle(incoming(1, '그 기능은 빼고 테스트부터 해'))
            worker.run_once()
            first = store.pipeline_workspace(run_id)
            self.assertEqual('NEEDS_ATTENTION', worker.status(run_id))
            self.assertEqual('RECEIVED', store.execution_inputs(run_id)[0]['status'])
            self.assertEqual(120, store.load_run(run_id).total_tokens)
            conversation.run_once()
            self.assertEqual('QUEUED', worker.status(run_id))
            worker.run_once()
            current = store.pipeline_workspace(run_id)
            self.assertEqual(first['worktree']['worktree_path'], current['worktree']['worktree_path'])
            self.assertEqual([(RoleId.DEVELOPMENT, True)], runner.calls)
            self.assertEqual('fixed\n', (Path(current['worktree']['worktree_path']) / 'feature.txt').read_text(encoding='utf-8'))
            self.assertFalse((source / 'feature.txt').exists())
            self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
            self.assertEqual(0, store.load_run(run_id).stage_index)
            self.assertEqual('APPLIED', store.execution_inputs(run_id)[0]['status'])
            artifact = json.loads((root / 'artifacts' / run_id / 'stages/stage-001/steering-tests.json').read_text(encoding='utf-8'))
            self.assertTrue(artifact['commands'][0]['passed'])
            self.assertFalse(artifact['code_repair_attempted'])
            self.assertEqual(['feature.txt'], current['execution_evidence']['stage-001']['changed_files'])
            self.assertTrue(current['execution_evidence']['stage-001']['uncommitted_changes'])
            self.assertIn('충돌', store.open_execution_question_for_run(run_id)['questions'][0])

    def test_supplement_is_applied_to_next_call_and_keeps_same_workspace_and_approval(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            runner = InterleavingRunner()
            store, worker, run_id = build_pipeline(root, source, runner)
            before = store.load_run(run_id)
            application, _ = gateway(root, store, worker, SteeringBackend('supplement', ['feature.txt']))
            runner.callback = lambda: application.handle(incoming(2, '빈 입력도 처리해 줘'))
            worker.run_once()
            self.assertEqual(RunPhase.COMPLETED, store.load_run(run_id).phase)
            self.assertEqual(3, len(runner.calls))
            self.assertIn('빈 입력도 처리해 줘', runner.prompts[1])
            self.assertEqual('fixed\n', runner.contents_before[1])
            self.assertEqual(1, len(set(runner.paths)))
            self.assertEqual(before.approved_plan_hash, store.load_run(run_id).approved_plan_hash)
            self.assertEqual(360, store.load_run(run_id).total_tokens)
            self.assertEqual('APPLIED', store.execution_inputs(run_id)[0]['status'])

    def test_status_during_development_preserves_goal_owner_and_does_not_call_chat_model(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            runner = InterleavingRunner()
            store, worker, run_id = build_pipeline(root, source, runner)
            backend = SteeringBackend('question')
            application, _ = gateway(root, store, worker, backend, queued=True)
            responses = []
            runner.callback = lambda: responses.extend(application.handle(incoming(3, '어디까지 했어?')))
            objective = store.load_run(run_id).objective
            worker.run_once()
            self.assertIn('개발 작업', responses[0].text)
            self.assertEqual(objective, store.load_run(run_id).objective)
            self.assertEqual(RunPhase.COMPLETED, store.load_run(run_id).phase)
            self.assertFalse(backend.calls)
            self.assertFalse(store.execution_inputs(run_id))
            self.assertEqual(2, len(runner.calls))

    def test_cancel_overlapping_ignored_runner_callback_blocks_commit_review_and_source_apply(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            original = git(source, 'rev-parse', 'HEAD')
            runner = InterleavingRunner()
            store, worker, run_id = build_pipeline(root, source, runner)
            application, _ = gateway(root, store, worker, SteeringBackend())
            runner.callback = lambda: application.handle(incoming(4, '취소'))
            worker.run_once()
            self.assertEqual(RunPhase.CANCELLED, store.load_run(run_id).phase)
            self.assertEqual('CANCELLED', worker.status(run_id))
            self.assertEqual([(RoleId.DEVELOPMENT, True)], runner.calls)
            checkpoint = store.pipeline_workspace(run_id)
            self.assertEqual(original, checkpoint['candidate_sha'])
            self.assertEqual(original, git(source, 'rev-parse', 'HEAD'))
            self.assertEqual('fixed\n', (Path(checkpoint['worktree']['worktree_path']) / 'feature.txt').read_text(encoding='utf-8'))
            self.assertEqual(120, store.load_run(run_id).total_tokens)

    def test_scope_expansion_waits_without_new_call_or_widening_approval(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            runner = InterleavingRunner()
            store, worker, run_id = build_pipeline(root, source, runner)
            approved = store.load_run(run_id).approved_plan_hash
            application, _ = gateway(root, store, worker, SteeringBackend('supplement', ['outside.txt']))
            output = []
            runner.callback = lambda: output.extend(application.handle(incoming(5, 'outside.txt도 수정해 줘')))
            worker.run_once()
            self.assertEqual('WAITING_APPROVAL', store.execution_inputs(run_id)[0]['status'])
            self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
            self.assertEqual(1, len(runner.calls))
            self.assertEqual(approved, store.load_run(run_id).approved_plan_hash)
            self.assertIn('outside.txt', output[0].text)
            self.assertFalse((source / 'feature.txt').exists())

    def test_question_resolves_after_restart_and_resumes_the_same_candidate(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            runner = InterleavingRunner()
            store, worker, run_id = build_pipeline(root, source, runner)
            application, _ = gateway(root, store, worker, SteeringBackend('question'), queued=True)
            runner.callback = lambda: application.handle(incoming(6, '이 방식으로 진행하는 이유가 뭐야?'))
            worker.run_once()
            checkpoint = store.pipeline_workspace(run_id)
            store, worker = reopen_worker(root, runner)
            backend = SteeringBackend('question')
            application, conversation = gateway(root, store, worker, backend, queued=True)
            conversation.run_once()
            self.assertEqual('ANSWERED', store.execution_inputs(run_id)[0]['status'])
            self.assertEqual('QUEUED', worker.status(run_id))
            self.assertEqual(checkpoint, store.pipeline_workspace(run_id))
            worker.run_once()
            self.assertEqual(RunPhase.COMPLETED, store.load_run(run_id).phase)
            self.assertEqual(1, len(set(runner.paths)))
            self.assertIn('fixed\n', runner.contents_before[1:])

    def test_duplicate_and_foreign_sender_do_not_create_directives_or_model_calls(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            store, worker, run_id = build_pipeline(root, source, InterleavingRunner())
            backend = SteeringBackend()
            application, _ = gateway(root, store, worker, backend)
            application.handle(incoming(7, '예외 처리도 해 줘', metadata={'gateway_run_id': 'other', 'gateway_execution_input': 999}))
            application.handle(incoming(7, '예외 처리도 해 줘'))
            application.handle(incoming(8, '범위를 늘려', user='foreign'))
            self.assertEqual(1, len(store.execution_inputs(run_id)))
            self.assertEqual(1, len(backend.calls))
            self.assertEqual(run_id, backend.calls[0][0])

    def test_response_for_replaced_task_is_suppressed_and_cannot_change_new_goal(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            store, worker, run_id = build_pipeline(root, source, InterleavingRunner())
            backend = SteeringBackend('redirect')
            application, _ = gateway(root, store, worker, backend)
            def replace_task():
                application.handle(incoming(9, '취소'))
                application.handle(incoming(10, '새 작업'))
            backend.before_reply = replace_task
            outgoing = application.handle(incoming(11, '방향을 바꿔 줘'))
            self.assertFalse(outgoing)
            self.assertEqual('SUPERSEDED', store.execution_inputs(run_id)[0]['status'])
            binding = store.load_conversation('telegram', 'chat-1')
            self.assertNotEqual(run_id, binding['active_task_id'])
            self.assertEqual('', store.load_run(binding['active_task_id']).objective)

    def test_separate_input_thread_cancels_inflight_execution_without_followup_calls(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            entered, release = threading.Event(), threading.Event()
            class BlockingRunner(InterleavingRunner):
                def run(self, *args, **kwargs):
                    result = super().run(*args, **kwargs)
                    entered.set()
                    if not release.wait(15):
                        raise AssertionError('input thread did not release runner')
                    return result
            runner = BlockingRunner()
            store, worker, run_id = build_pipeline(root, source, runner)
            application, _ = gateway(root, store, worker, SteeringBackend(), queued=True)
            thread = threading.Thread(target=worker.run_once)
            thread.start()
            try:
                self.assertTrue(entered.wait(15))
                output = application.handle(incoming(1, '취소'))
                self.assertTrue(output)
            finally:
                release.set()
                thread.join(15)
            self.assertFalse(thread.is_alive())
            self.assertEqual('CANCELLED', worker.status(run_id))
            self.assertEqual([(RoleId.DEVELOPMENT, True)], runner.calls)
            self.assertFalse((source / 'feature.txt').exists())

    def test_test_first_question_answer_reuses_workspace_without_reverting_completed_content(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            runner = InterleavingRunner()
            store, worker, run_id = build_pipeline(root, source, runner)
            backend = SteeringBackend('test_first')
            application, _ = gateway(root, store, worker, backend)
            runner.callback = lambda: application.handle(incoming(1, '구현은 여기까지 하고 테스트부터 해'))
            worker.run_once()
            checkpoint = store.pipeline_workspace(run_id)
            self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
            application.handle(incoming(2, '현재 변경을 유지하고 승인된 단계의 검증을 마무리해'))
            self.assertEqual('QUEUED', worker.status(run_id))
            worker.run_once()
            self.assertEqual(RunPhase.COMPLETED, store.load_run(run_id).phase)
            self.assertEqual(1, len(set(runner.paths)))
            self.assertIn('fixed\n', runner.contents_before[1:])
            self.assertEqual(1, len(backend.calls))
            self.assertEqual('fixed\n', (source / 'feature.txt').read_text(encoding='utf-8'))

    def test_test_first_failure_does_not_start_finisher_or_consume_code_retry(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root, verifier_passes=False)
            runner = InterleavingRunner()
            store, worker, run_id = build_pipeline(root, source, runner)
            application, _ = gateway(root, store, worker, SteeringBackend('test_first'))
            runner.callback = lambda: application.handle(incoming(1, '개발 말고 테스트부터'))
            worker.run_once()
            self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
            self.assertEqual([(RoleId.DEVELOPMENT, True)], runner.calls)
            evidence = store.pipeline_workspace(run_id)['execution_evidence']['stage-001']
            self.assertEqual('test_failure', evidence['commands'][0]['failure_kind'])
            self.assertEqual(1, evidence['commands'][0]['return_code'])
            self.assertEqual(0, store.retry_count(run_id, 'stage-001', 'verification_failure'))
            self.assertFalse((source / 'feature.txt').exists())
