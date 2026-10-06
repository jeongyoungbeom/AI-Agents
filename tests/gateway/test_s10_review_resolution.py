from __future__ import annotations

import json
import sqlite3
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path

from app.agents.parsing import InvalidAgentResponse, parse_team_conversation_reply
from app.contracts import RoleId, RunPhase, TokenUsage
from app.gateway.core import AgentReply
from app.gateway.core.governed_backend import GovernedTeamConversationBackend
from app.gateway.conversation_worker import ConversationQueue, ConversationWorker
from app.gateway.repository_analysis_worker import RepositoryAnalysisWorker, RepositoryContextLimit
from app.orchestrator import RunStateMachine
from app.services.budget import BudgetExceeded, BudgetManager, BudgetPolicy
from app.services.context import ContextService
from app.services.repository import RepositoryAnalysisRequest, RepositoryAnalysisStatus, RepositorySnapshotEntry, RepositorySnapshotManifest
from app.storage import StateStore, StoreError
from app.storage.sqlite_store import ExecutionInputPending
from tests.gateway.support import build_application
from tests.gateway.test_conversation_foundation import incoming as analysis_message
from tests.gateway.test_repository_analysis import batch_reply
from tests.pipeline.support import build_pipeline, create_repository, git, temporary_directory
from tests.pipeline.test_s08_resume import reopen_worker
from tests.pipeline.test_s10_steering import InterleavingRunner, SteeringBackend, gateway, incoming


class S10ClarificationResolutionTests(unittest.TestCase):
    def _recover(self, first_kind, *, governed=False):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            original = git(source, 'rev-parse', 'HEAD')
            runner = InterleavingRunner()
            store, worker, run_id = build_pipeline(root, source, runner)
            approved = store.load_run(run_id).approved_plan_hash

            class Backend(SteeringBackend):
                def respond_as(self, state, context, message, role_id, **kwargs):
                    self.calls.append(message.text)
                    if len(self.calls) == 1:
                        if first_kind == 'failed':
                            raise RuntimeError('confirmed interpretation failure')
                        if first_kind == 'invalid':
                            return AgentReply('뜻을 설명해 주세요.', execution_intent={'action': 'approve'})
                        return AgentReply('뜻을 설명해 주세요.')
                    pending = [item for item in store.execution_inputs(run_id)
                               if item['intent'].get('clarification_required')]
                    return AgentReply('승인된 feature.txt의 빈 입력 처리만 보충합니다.', execution_intent={
                        'action': 'supplement', 'requested_scope': ['feature.txt'],
                        'clarifies_input_ids': [item['input_id'] for item in pending]})

            delegate = Backend()
            backend = GovernedTeamConversationBackend(delegate, worker.coordinator.budget) if governed else delegate
            application, conversation = gateway(root, store, worker, backend, queued=True)
            runner.callback = lambda: application.handle(incoming(1, '그 부분도 처리해 줘'))
            worker.run_once()
            conversation.run_once()
            first = store.execution_inputs(run_id)[0]
            self.assertEqual('RECEIVED', first['status'])
            self.assertTrue(first['intent']['clarification_required'])
            self.assertEqual('NEEDS_ATTENTION', worker.status(run_id))
            checkpoint = store.pipeline_workspace(run_id)
            workspace = checkpoint['worktree']['worktree_path']
            cost = store.load_run(run_id).total_tokens
            original_message = incoming(1, '그 부분도 처리해 줘', metadata={'gateway_execution_input': first['input_id']})
            application.router.route_result(original_message)
            self.assertEqual(1, len(delegate.calls))
            self.assertEqual(cost, store.load_run(run_id).total_tokens)
            self.assertEqual(original, git(source, 'rev-parse', 'HEAD'))
            self.assertEqual('fixed\n', (Path(workspace) / 'feature.txt').read_text(encoding='utf-8'))

            store, worker = reopen_worker(root, runner)
            backend = GovernedTeamConversationBackend(delegate, worker.coordinator.budget) if governed else delegate
            application, conversation = gateway(root, store, worker, backend, queued=True)
            application.handle(incoming(2, '승인된 feature.txt의 빈 입력만 처리하라는 뜻이야'))
            conversation.run_once()
            entries = store.execution_inputs(run_id)
            self.assertEqual(['SUPERSEDED', 'READY'], [item['status'] for item in entries])
            self.assertEqual(entries[1]['input_id'], entries[0]['intent']['clarified_by_input_id'])
            self.assertEqual('QUEUED', worker.status(run_id))
            clarified_cost = store.load_run(run_id).total_tokens
            store, worker = reopen_worker(root, runner)
            self.assertEqual(clarified_cost, store.load_run(run_id).total_tokens)
            self.assertTrue(worker.run_once())
            self.assertEqual(RunPhase.COMPLETED, store.load_run(run_id).phase)
            self.assertEqual(workspace, store.pipeline_workspace(run_id)['worktree']['worktree_path'])
            self.assertEqual(approved, store.load_run(run_id).approved_plan_hash)
            self.assertEqual(2, len(delegate.calls))
            self.assertEqual(3, len(runner.calls))
            self.assertIn('빈 입력만 처리', runner.prompts[1])
            self.assertEqual('fixed\n', runner.contents_before[1])
            self.assertEqual('fixed\n', (source / 'feature.txt').read_text(encoding='utf-8'))
            self.assertEqual('APPLIED', store.execution_inputs(run_id)[1]['status'])

    def test_confirmed_missing_intent_clarification_restarts_same_workspace_without_rebilling(self):
        self._recover('missing', governed=True)

    def test_invalid_confirmed_intent_can_be_clarified_and_resumed(self):
        self._recover('invalid', governed=True)

    def test_interpretation_failure_can_be_clarified_and_resumed(self):
        self._recover('failed')

    def test_new_directive_does_not_discard_an_unanswered_clarification(self):
        self._guard_resolution('unrelated')

    def test_raw_unprocessed_input_cannot_be_resolved_as_a_clarification(self):
        self._guard_resolution('unprocessed')

    def test_outside_scope_clarification_does_not_release_old_input(self):
        self._guard_resolution('blocked')

    def test_question_does_not_release_old_input(self):
        self._guard_resolution('question')

    def test_cross_run_clarification_is_rejected_atomically(self):
        self._guard_resolution('cross_run')

    def test_multiple_clarifications_roll_back_when_one_target_is_invalid(self):
        self._guard_resolution('invalid_second')

    def _guard_resolution(self, kind):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            store, worker, run_id = build_pipeline(root, source, InterleavingRunner())
            gateway(root, store, worker, SteeringBackend())
            first = store.capture_execution_input('telegram', 'chat-1', 'test-user', '1', '미확정 요청')
            if kind != 'unprocessed':
                store.request_execution_input_clarification(first['input_id'], {'response_text': '뜻 확인'})
            if kind == 'cross_run':
                worker.machine.create_run('RUN-OTHER')
                with closing(sqlite3.connect(store.path)) as connection:
                    connection.execute('UPDATE execution_inputs SET run_id=? WHERE input_id=?', ('RUN-OTHER', first['input_id']))
                    connection.commit()
            second = store.capture_execution_input('telegram', 'chat-1', 'test-user', '2', '보충 답변')
            intent = {'action': 'question' if kind == 'question' else 'supplement',
                      'clarifies_input_ids': [] if kind == 'unrelated' else [first['input_id']]}
            if kind == 'invalid_second':
                intent['clarifies_input_ids'].append(second['input_id'])
            if kind in {'cross_run', 'unprocessed', 'invalid_second'}:
                with self.assertRaises(StoreError):
                    store.resolve_execution_input(second['input_id'], intent)
                self.assertEqual('RECEIVED', store.execution_input(second['input_id'])['status'])
            else:
                store.resolve_execution_input(second['input_id'], intent, blocked=kind == 'blocked')
            self.assertEqual('RECEIVED', store.execution_input(first['input_id'])['status'])

    def test_clarification_ids_are_validated_and_preserved_in_model_contract(self):
        payload = {'message': '답변', 'execution_intent': {'action': 'supplement', 'clarifies_input_ids': [7, 9]}}
        reply = parse_team_conversation_reply(json.dumps(payload), RoleId.DEVELOPMENT)
        self.assertEqual([7, 9], AgentReply.from_dict(reply.to_dict()).execution_intent['clarifies_input_ids'])
        for value in ('7', [True], [0], [-1], [1.0], list(range(1, 34))):
            with self.subTest(value=value), self.assertRaises(InvalidAgentResponse):
                payload['execution_intent']['clarifies_input_ids'] = value
                parse_team_conversation_reply(json.dumps(payload), RoleId.DEVELOPMENT)


class S10PartialSynthesisResolutionTests(unittest.TestCase):
    def _steered_partial(self, *, queued=False, late=False, limit='query', governed=False):
        with temporary_directory() as directory:
            root = Path(directory)
            store, application = build_application(root, object())
            application.handle(analysis_message(1, '안녕'))
            application.router.team_backend = SteeringBackend('redirect')
            if queued:
                queue = ConversationQueue(store)
                application.conversation_scheduler = application.router.conversation_scheduler = queue
            request = RepositoryAnalysisRequest.create(channel='telegram', conversation_id='200', user_id='100',
                source_message_id='2', role_id='development', request_text='전체 코드 분석해줘', repository_path=str(root),
                repository_identity='a' * 64, commit_sha='b' * 40, branch='main')
            RunStateMachine(store).create_run(request.analysis_id, repository=request.repository_path,
                repository_identity=request.repository_identity, repository_head_sha=request.commit_sha, repository_approved=True)
            store.create_repository_analysis(request)

            class Reader:
                def pinned_manifest(self, *args, **kwargs):
                    return RepositorySnapshotManifest('a' * 64, 'b' * 40, 'main', (
                        RepositorySnapshotEntry('app/main.py', 20), RepositorySnapshotEntry('app/second.py', 20)))
                def read_pinned_files(self, _path, _manifest, paths, **kwargs):
                    return tuple((path, 'code\nsource') for path in paths)

            class Team:
                supports_cancellation = False
                def __init__(self):
                    self.calls = []
                    self.injected = False
                def respond_as(self, state, context, message, role_id, **kwargs):
                    self.calls.append(message.text)
                    if context.repository_context['phase'] == 'SYNTHESIS':
                        if not self.injected and not late:
                            self.injected = True
                            application.handle(analysis_message(3, '지금부터 테스트 누락 관점으로 조사 방향을 바꿔 줘'))
                            return AgentReply('이전 목표의 부분 종합')
                        return AgentReply('새 지시의 테스트 누락 관점으로 확인된 파일만 종합했습니다.')
                    return batch_reply(context, '고정 snapshot에서 확인한 파일')

            team = Team()
            errors = []
            backend = team
            if governed:
                respond = team.respond_as
                def reported_reply(*args, **kwargs):
                    return replace(respond(*args, **kwargs), usage=TokenUsage(input_tokens=10, output_tokens=5))
                team.respond_as = reported_reply
                backend = GovernedTeamConversationBackend(team, BudgetManager(BudgetPolicy(), store))
            worker = RepositoryAnalysisWorker(store, Reader(), backend, ContextService(store),
                max_files_per_batch=1, max_file_bytes=1024, max_batch_bytes=2048,
                max_query_rounds=1, max_read_bytes=4096, error_sink=errors.append)
            if limit in {'budget', 'context'}:
                process = worker._process
                def limited(job, lease_lost):
                    if job['completed']:
                        raise BudgetExceeded('test budget') if limit == 'budget' else RepositoryContextLimit('test context')
                    return process(job, lease_lost)
                worker._process = limited
            elif limit == 'time':
                worker._elapsed_limit_reached = lambda job: bool(job['completed'])
            if late:
                finish = store.finish_repository_analysis_with_response
                def delayed_finish(*args, **kwargs):
                    if not team.injected:
                        team.injected = True
                        application.handle(analysis_message(3, '지금부터 테스트 누락 관점으로 조사 방향을 바꿔 줘'))
                    return finish(*args, **kwargs)
                store.finish_repository_analysis_with_response = delayed_finish
            for _ in range(8):
                worker.run_once()
                if team.injected:
                    break
            self.assertTrue(team.injected, (store.repository_analysis(request.analysis_id), errors, team.calls))
            job = store.repository_analysis(request.analysis_id)
            self.assertEqual('PAUSED' if queued else 'QUEUED', job['status'])
            self.assertEqual('', job['final_response'])
            self.assertEqual((2, 2, 2, 2), (job['query_rounds'], job['model_calls'], job['model_attempts'], job['model_successes']))
            if governed:
                self.assertEqual(30, store.load_run(request.analysis_id).total_tokens)
            evidence = store.repository_analysis_evidence(request.analysis_id)
            self.assertTrue(evidence)
            self.assertEqual('RECEIVED' if queued else 'APPLIED' if not late else 'READY', store.execution_inputs(request.analysis_id)[0]['status'])
            if queued:
                reopened = StateStore(store.path)
                conversation = ConversationWorker(reopened, application.router, poll_seconds=0.01)
                self.assertTrue(conversation.run_once())
                self.assertEqual('STEERING_RECEIVED', reopened.conversation_result('telegram', '200', 'foundation-3').reason)
                self.assertEqual('QUEUED', reopened.repository_analysis(request.analysis_id)['status'])
            for _ in range(4):
                if not worker.run_once():
                    break
            final = store.repository_analysis(request.analysis_id)
            self.assertEqual('PARTIAL_COMPLETED', final['status'])
            self.assertEqual({'query': 'QUERY_LIMIT', 'budget': 'BUDGET_LIMIT', 'context': 'CONTEXT_LIMIT', 'time': 'TIME_LIMIT'}[limit], final['stop_reason'])
            self.assertEqual('b' * 40, final['commit_sha'])
            self.assertEqual(evidence, store.repository_analysis_evidence(request.analysis_id))
            self.assertEqual((3, 3, 3, 3), (final['query_rounds'], final['model_calls'], final['model_attempts'], final['model_successes']))
            self.assertIn('테스트 누락 관점으로 조사 방향', team.calls[-1])
            self.assertNotIn('이전 목표의 부분 종합', final['final_response'])
            self.assertIn('새 지시의 테스트 누락 관점', final['final_response'])
            self.assertEqual('APPLIED', store.execution_inputs(request.analysis_id)[0]['status'])
            self.assertFalse(errors)
            if governed:
                self.assertEqual(45, store.load_run(request.analysis_id).total_tokens)

    def test_discarded_partial_synthesis_keeps_confirmed_usage_without_rebilling(self):
        self._steered_partial(governed=True)

    def test_queued_partial_synthesis_keeps_confirmed_usage_after_restart(self):
        self._steered_partial(queued=True, governed=True)

    def test_inline_direction_during_partial_synthesis_requeues_and_keeps_accounting(self):
        self._steered_partial()

    def test_queued_direction_during_partial_synthesis_pauses_then_resumes(self):
        self._steered_partial(queued=True)

    def test_direction_just_before_final_transaction_blocks_old_completion(self):
        self._steered_partial(late=True)

    def test_queued_direction_just_before_final_transaction_survives_restart(self):
        self._steered_partial(queued=True, late=True)

    def test_budget_fallback_uses_the_same_steering_boundary(self):
        self._steered_partial(limit='budget')

    def test_context_fallback_uses_the_same_steering_boundary(self):
        self._steered_partial(queued=True, limit='context')

    def test_time_limit_partial_synthesis_keeps_new_direction(self):
        self._steered_partial(limit='time')

    def test_pending_input_prevents_partial_result_parent_and_outbound_commit(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store, application = build_application(root, object())
            application.handle(analysis_message(1, '안녕'))
            request = RepositoryAnalysisRequest.create(channel='telegram', conversation_id='200', user_id='100',
                source_message_id='2', role_id='development', request_text='분석', repository_path=str(root),
                repository_identity='a' * 64, commit_sha='b' * 40, branch='main')
            RunStateMachine(store).create_run(request.analysis_id)
            store.create_repository_analysis(request)
            store.claim_next_repository_analysis('owner')
            store.capture_execution_input('telegram', '200', '100', '3', '보충')
            for status in (RepositoryAnalysisStatus.COMPLETED, RepositoryAnalysisStatus.PARTIAL_COMPLETED):
                with self.subTest(status=status), self.assertRaises(ExecutionInputPending):
                    store.finish_repository_analysis_with_response(request.analysis_id, 'owner', status, '오래된 결과', reason='QUERY_LIMIT')
            job = store.repository_analysis(request.analysis_id)
            self.assertEqual('PROCESSING', job['status'])
            self.assertEqual('', job['final_response'])
            with closing(sqlite3.connect(store.path)) as connection:
                self.assertEqual(0, connection.execute("SELECT count(*) FROM messages WHERE content LIKE '%오래된 결과%'").fetchone()[0])


if __name__ == '__main__':
    unittest.main()
