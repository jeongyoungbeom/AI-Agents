from __future__ import annotations

import json
import sqlite3
import unittest
from contextlib import closing
from pathlib import Path

from app.agents.parsing import parse_team_conversation_reply, InvalidAgentResponse
from app.contracts import RoleId, RunPhase
from app.gateway.conversation_worker import ConversationQueue
from app.gateway.repository_analysis_worker import RepositoryAnalysisQueue
from app.orchestrator import RunStateMachine
from app.services.repository import RepositoryAnalysisRequest
from tests.gateway.support import build_application, temporary_directory
from tests.gateway.test_conversation_foundation import incoming
from tests.pipeline.support import build_pipeline, create_repository
from tests.pipeline.test_s10_steering import InterleavingRunner, SteeringBackend, gateway, incoming as execution_message
from tests.pipeline.test_s08_resume import reopen_worker


class S10MessageTests(unittest.TestCase):
    def test_natural_status_during_analysis_keeps_analysis_and_owner(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store, application = build_application(root, object())
            application.router.repository_analysis_scheduler = RepositoryAnalysisQueue(store)
            application.handle(incoming(1, '안녕'))
            request = RepositoryAnalysisRequest.create(
                channel='telegram', conversation_id='200', user_id='100', source_message_id='2',
                role_id='development', request_text='전체 코드 분석해줘', repository_path=str(root),
                repository_identity='a' * 64, commit_sha='b' * 40, branch='main')
            RunStateMachine(store).create_run(request.analysis_id)
            store.create_repository_analysis(request)
            store.claim_next_repository_analysis('analysis-worker')
            binding = store.load_conversation('telegram', '200')
            before = store.repository_analysis(request.analysis_id)
            response = application.handle(incoming(3, '어디까지 했어?'))
            self.assertIn(request.analysis_id, response[0].text)
            self.assertEqual(before, store.repository_analysis(request.analysis_id))
            self.assertEqual(binding, store.load_conversation('telegram', '200'))

    def test_execution_intent_parsing_and_cache_roundtrip_are_validated(self):
        payload = {'message': '테스트부터 확인', 'execution_intent': {'action': 'test_first', 'requested_scope': ['feature.txt']}}
        reply = parse_team_conversation_reply(json.dumps(payload), RoleId.DEVELOPMENT)
        self.assertEqual('test_first', reply.execution_intent['action'])
        from app.gateway.core import AgentReply
        self.assertEqual(reply, AgentReply.from_dict(reply.to_dict()))
        for value in ({'action': 'approve'}, {'action': 'redirect', 'requested_scope': 'all'},
                      {'action': 'redirect', 'requested_scope': [123]}, 'redirect'):
            with self.subTest(value=value), self.assertRaises(InvalidAgentResponse):
                parse_team_conversation_reply(json.dumps({'message': 'x', 'execution_intent': value}), RoleId.REVIEW)

    def test_queue_failure_rolls_back_input_and_receipt_outbound_together(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            store, worker, run_id = build_pipeline(root, source, InterleavingRunner())
            application, _ = gateway(root, store, worker, SteeringBackend(), queued=True)
            class BrokenQueue(ConversationQueue):
                def enqueue(self, message):
                    super().enqueue(message)
                    raise RuntimeError('queue fault')
            application.conversation_scheduler = BrokenQueue(store)
            with self.assertRaisesRegex(RuntimeError, 'queue fault'):
                application.handle(execution_message(1, '보충 설명'))
            self.assertFalse(store.execution_inputs(run_id))
            self.assertIsNone(store.conversation_job_summary('telegram', 'chat-1'))
            self.assertEqual('FAILED', store.inbound_receipt('telegram', 'chat-1:1')['status'])

    def test_old_queued_message_after_new_task_does_not_call_model(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            store, worker, run_id = build_pipeline(root, source, InterleavingRunner())
            backend = SteeringBackend()
            application, conversation = gateway(root, store, worker, backend, queued=True)
            application.handle(execution_message(1, '이전 기능 보충'))
            # 재시작 복구에서도 외부 message_id와 원래 run의 연결은 변하지 않는다.
            worker.cancel(run_id)
            worker.machine.transition(store.load_run(run_id), RunPhase.CANCELLED)
            new = worker.machine.create_run('RUN-REPLACEMENT', objective='새 목표')
            store.bind_conversation('telegram', 'chat-1', 'test-user', new.run_id, 'review')
            conversation.run_once()
            self.assertFalse(backend.calls)
            self.assertEqual('새 목표', store.load_run(new.run_id).objective)
            self.assertEqual('review', store.load_conversation('telegram', 'chat-1')['active_role'])
            result = store.conversation_result('telegram', 'chat-1', '1')
            self.assertEqual('SUPERSEDED_TASK', result.reason)

    def test_invalid_interpretation_is_preserved_without_treating_it_as_approval(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            runner = InterleavingRunner()
            store, worker, run_id = build_pipeline(root, source, runner)
            class AmbiguousBackend(SteeringBackend):
                def respond_as(self, *args, **kwargs):
                    from app.gateway.core import AgentReply
                    return AgentReply('뜻을 확인해 주세요')
            application, _ = gateway(root, store, worker, AmbiguousBackend())
            runner.callback = lambda: application.handle(execution_message(1, '그거는 조금 다르게'))
            worker.run_once()
            self.assertEqual('NEEDS_ATTENTION', worker.status(run_id))
            self.assertEqual('RECEIVED', store.execution_inputs(run_id)[0]['status'])
            self.assertEqual(1, len(runner.calls))
            self.assertFalse((source / 'feature.txt').exists())

    def test_migration_11_and_input_order_survive_reopen(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            store, worker, run_id = build_pipeline(root, source, InterleavingRunner())
            application, _ = gateway(root, store, worker, SteeringBackend())
            application.handle(execution_message(1, '첫 번째 보충'))
            application.handle(execution_message(2, '두 번째 보충'))
            from app.storage import StateStore
            reopened = StateStore(store.path)
            self.assertEqual(['첫 번째 보충', '두 번째 보충'], [item['text'] for item in reopened.execution_inputs(run_id)])
            with closing(sqlite3.connect(store.path)) as connection:
                self.assertEqual(1, connection.execute('SELECT count(*) FROM schema_migrations WHERE version=11').fetchone()[0])

    def test_resolved_input_replay_does_not_call_model_again(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            store, worker, run_id = build_pipeline(root, source, InterleavingRunner())
            backend = SteeringBackend('question')
            application, _ = gateway(root, store, worker, backend, queued=True)
            application.handle(execution_message(1, '그 선택의 이유는?'))
            job = store.claim_next_conversation_job('test-owner')
            from app.gateway.core import IncomingMessage
            message = IncomingMessage.from_dict(job['message'])
            application.router.route_result(message)
            # 분류 확정 뒤 응답 확정 전 재시작/재전송을 재현한다.
            store, restarted = reopen_worker(root, InterleavingRunner())
            application, _ = gateway(root, store, restarted, backend, queued=True)
            application.router.route_result(message)
            self.assertEqual(1, len(backend.calls))
            self.assertEqual('ANSWERED', store.execution_inputs(run_id)[0]['status'])

    def test_analysis_direction_discards_old_result_and_preserves_scanned_byte_count(self):
        from app.gateway.repository_analysis_worker import RepositoryAnalysisWorker
        from app.gateway.core import AgentReply
        from app.services.context import ContextService
        from app.services.repository import RepositorySnapshotManifest, RepositorySnapshotEntry
        from tests.gateway.test_repository_analysis import batch_reply
        with temporary_directory() as directory:
            root = Path(directory)
            backend = SteeringBackend('supplement')
            store, application = build_application(root, object())
            application.handle(incoming(1, '안녕'))
            application.router.team_backend = backend
            request = RepositoryAnalysisRequest.create(channel='telegram', conversation_id='200', user_id='100',
                source_message_id='2', role_id='development', request_text='전체 코드 분석해줘', repository_path=str(root),
                repository_identity='a' * 64, commit_sha='b' * 40, branch='main')
            RunStateMachine(store).create_run(request.analysis_id, repository=request.repository_path,
                repository_identity=request.repository_identity, repository_head_sha=request.commit_sha, repository_approved=True)
            store.create_repository_analysis(request)
            class Reader:
                def pinned_manifest(self, *args, **kwargs):
                    return RepositorySnapshotManifest('a' * 64, 'b' * 40, 'main', (RepositorySnapshotEntry('app/main.py', 20),))
                def read_pinned_files(self, _path, _manifest, paths, **kwargs):
                    return tuple((path, 'code\nsource') for path in paths)
            class Team:
                supports_cancellation = False
                def __init__(self):
                    self.messages = []
                def respond_as(self, state, context, message, role_id, **kwargs):
                    self.messages.append(message.text)
                    if len(self.messages) == 1:
                        self.before = store.repository_analysis(request.analysis_id)
                        application.handle(incoming(3, '테스트 경계도 함께 조사해 줘'))
                        return batch_reply(context, '이전 목표의 결과')
                    return batch_reply(context, '새 조사 방향 결과')
            team = Team()
            worker = RepositoryAnalysisWorker(store, Reader(), team, ContextService(store),
                max_file_bytes=1024, max_batch_bytes=2048, max_query_rounds=8, max_read_bytes=4096)
            for _ in range(6):
                self.assertTrue(worker.run_once())
                if team.messages:
                    break
            after_old = store.repository_analysis(request.analysis_id)
            self.assertEqual('QUEUED', after_old['status'])
            self.assertEqual(team.before['read_bytes'] + 20, after_old['read_bytes'])
            self.assertEqual(team.before['query_rounds'] + 1, after_old['query_rounds'])
            self.assertFalse(store.repository_analysis_evidence(request.analysis_id))
            for _ in range(6):
                if not worker.run_once():
                    break
            final = store.repository_analysis(request.analysis_id)
            self.assertEqual('COMPLETED', final['status'])
            self.assertEqual('b' * 40, final['commit_sha'])
            self.assertEqual(40, final['read_bytes'])
            self.assertIn('테스트 경계도 함께 조사해 줘', team.messages[1])
            self.assertNotIn('이전 목표의 결과', final['final_response'])
            self.assertEqual('APPLIED', store.execution_inputs(request.analysis_id)[0]['status'])

    def test_late_analysis_input_before_checkpoint_blocks_old_evidence_commit(self):
        from app.gateway.repository_analysis_worker import RepositoryAnalysisWorker
        from app.services.context import ContextService
        from app.services.repository import RepositorySnapshotManifest, RepositorySnapshotEntry
        from tests.gateway.test_repository_analysis import batch_reply
        with temporary_directory() as directory:
            root = Path(directory)
            store, application = build_application(root, object())
            application.handle(incoming(1, '안녕'))
            application.router.team_backend = SteeringBackend('redirect')
            queue = ConversationQueue(store)
            application.conversation_scheduler = queue
            application.router.conversation_scheduler = queue
            request = RepositoryAnalysisRequest.create(channel='telegram', conversation_id='200', user_id='100',
                source_message_id='2', role_id='development', request_text='전체 코드 분석해줘', repository_path=str(root),
                repository_identity='a' * 64, commit_sha='b' * 40, branch='main')
            RunStateMachine(store).create_run(request.analysis_id, repository=request.repository_path,
                repository_identity=request.repository_identity, repository_head_sha=request.commit_sha, repository_approved=True)
            store.create_repository_analysis(request)
            class Reader:
                def pinned_manifest(self, *args, **kwargs):
                    return RepositorySnapshotManifest('a' * 64, 'b' * 40, 'main', (RepositorySnapshotEntry('app/main.py', 20),))
                def read_pinned_files(self, _path, _manifest, paths, **kwargs):
                    return tuple((path, 'code\nsource') for path in paths)
            class Team:
                supports_cancellation = False
                calls = 0
                def respond_as(self, state, context, message, role_id, **kwargs):
                    self.calls += 1
                    return batch_reply(context, '확인 결과')
            team = Team()
            worker = RepositoryAnalysisWorker(store, Reader(), team, ContextService(store),
                max_file_bytes=1024, max_batch_bytes=2048, max_query_rounds=8, max_read_bytes=4096)
            complete = store.complete_repository_analysis_batch
            injected = []
            def delayed_complete(*args, **kwargs):
                if team.calls and not injected:
                    injected.append(True)
                    application.handle(incoming(3, '이제 테스트 관점으로 조사해 줘'))
                return complete(*args, **kwargs)
            store.complete_repository_analysis_batch = delayed_complete
            for _ in range(6):
                worker.run_once()
                if injected:
                    break
            job = store.repository_analysis(request.analysis_id)
            self.assertEqual('PAUSED', job['status'])
            self.assertEqual('IDLE', job['model_call_state'])
            self.assertEqual('STEERING_PENDING', job['stop_reason'])
            self.assertFalse(store.repository_analysis_evidence(request.analysis_id))
            from app.storage import StateStore
            from app.gateway.conversation_worker import ConversationWorker
            reopened = StateStore(store.path)
            _, application = build_application(root, object(), team_backend=SteeringBackend('redirect'))
            conversation = ConversationWorker(reopened, application.router, poll_seconds=0.01)
            self.assertTrue(conversation.run_once())
            self.assertEqual('QUEUED', reopened.repository_analysis(request.analysis_id)['status'])
            self.assertEqual('READY', reopened.execution_inputs(request.analysis_id)[0]['status'])

    def test_saved_model_interpretation_replays_original_input_without_cost_or_provider_repeat(self):
        from app.gateway.core.governed_backend import GovernedTeamConversationBackend
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            store, worker, run_id = build_pipeline(root, source, InterleavingRunner())
            delegate = SteeringBackend('question')
            backend = GovernedTeamConversationBackend(delegate, worker.coordinator.budget)
            application, conversation = gateway(root, store, worker, backend, queued=True)
            application.handle(execution_message(1, '지금 그 방식으로 하는 이유는?'))
            resolve = store.resolve_execution_input
            def resolution_fault(*args, **kwargs):
                raise RuntimeError('classification persistence fault')
            store.resolve_execution_input = resolution_fault
            conversation.run_once()
            cost = store.load_run(run_id).total_tokens
            self.assertGreater(cost, 0)
            self.assertEqual('NEEDS_ATTENTION', store.conversation_job_summary('telegram', 'chat-1')['status'])
            self.assertEqual('RECEIVED', store.execution_inputs(run_id)[0]['status'])
            store, worker = reopen_worker(root, InterleavingRunner())
            backend = GovernedTeamConversationBackend(delegate, worker.coordinator.budget)
            application, conversation = gateway(root, store, worker, backend, queued=True)
            self.assertEqual('QUEUED', store.requeue_conversation_job('telegram', 'chat-1')['status'])
            conversation.run_once()
            self.assertEqual('COMPLETED', store.conversation_job_summary('telegram', 'chat-1')['status'])
            self.assertEqual('ANSWERED', store.execution_inputs(run_id)[0]['status'])
            self.assertEqual(cost, store.load_run(run_id).total_tokens)
            self.assertEqual(1, len(delegate.calls))
            result = store.conversation_result('telegram', 'chat-1', '1')
            self.assertEqual((1, 1), (result.attempts, result.successes))

    def test_interpretation_failure_records_attempt_without_success_or_directive(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            store, worker, run_id = build_pipeline(root, source, InterleavingRunner())
            class FailingBackend(SteeringBackend):
                def respond_as(self, *args, **kwargs):
                    raise RuntimeError('interpretation failed')
            application, _ = gateway(root, store, worker, FailingBackend())
            output = application.handle(execution_message(1, '이것도 함께 반영해 줘'))
            result = store.conversation_result('telegram', 'chat-1', '1')
            self.assertEqual('failed', result.outcome.value)
            self.assertEqual((1, 0), (result.attempts, result.successes))
            self.assertEqual('RECEIVED', store.execution_inputs(run_id)[0]['status'])
            self.assertIn('현재 변경을 보존', output[0].text)
