import json
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.agents.conversation_backend import HermesConversationBackend
from app.contracts import RoleId, RunState, TokenUsage
from app.gateway.conversation_worker import ConversationQueue, ConversationWorker
from app.gateway.core import AgentReply, IncomingMessage, TeamConversationRequest
from app.gateway.core.governed_backend import GovernedAgentBackend, GovernedTeamConversationBackend
from app.gateway.repository_analysis_worker import RepositoryAnalysisWorker
from app.orchestrator import RunStateMachine
from app.services.budget import BudgetExceeded, BudgetManager, BudgetPolicy
from app.services.context import ContextService
from app.services.git import GitRepositoryCancelled, GitRepositoryError
from app.services.hermes import HermesCancelled, HermesExecutionError, HermesResult
from app.services.repository import RepositoryAnalysisRequest
from app.storage import StateStore
from tests.gateway.support import build_application, temporary_directory
from tests.gateway.test_s03_context import incoming
from tests.gateway.test_s05_agent_loop import CONTEXT, Delegate
from tests.pipeline import test_hermes_contract as contract


class S05ReviewResolutionTests(unittest.TestCase):
    def manager(self, root, **policy):
        store = StateStore(root / 'state.db')
        state = store.create_run(RunState('RUN-S05-REVIEW'))
        return store, state, BudgetManager(BudgetPolicy(**policy), store)

    def runner_fixture(self):
        fixture = contract.HermesContractTests('test_missing_result_file_is_invalid_result')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def test_reportless_and_corrupt_runner_failures_block_followup_and_retry(self):
        fixture = self.runner_fixture()
        for report, usage in [
            ({'status': 'failed', 'text': '', 'error': 'provider unavailable'}, None),
            (None, None),
            ({'status': 'failed', 'text': '', 'error': 'provider unavailable'}, {'total_tokens': -1}),
        ]:
            with self.subTest(report=report, usage=usage), temporary_directory() as directory:
                with self.assertRaises(HermesExecutionError) as raised:
                    fixture.invoke(report, returncode=1, usage=usage)
                self.assertTrue(raised.exception.usage.estimated)
                self.assertGreater(raised.exception.usage.total_tokens, 0)
                store, state, budget = self.manager(Path(directory), conversation_tokens=1000,
                                                   retries={'technical_error': 1})
                delegate = Delegate(raised.exception)
                backend = GovernedTeamConversationBackend(delegate, budget, response_reserve_tokens=100)
                with self.assertRaises(HermesExecutionError):
                    backend.respond_as(state, CONTEXT, incoming(1, '조사'), RoleId.REVIEW)
                self.assertEqual(110, store.usage_total(state.run_id))
                self.assertTrue(store.has_budget_anomaly(state.run_id))
                self.assertEqual(0, store.reserved_token_total(state.run_id))
                delegate.response = AgentReply('후속', usage=TokenUsage(2, 1))
                with self.assertRaises(BudgetExceeded):
                    backend.respond_as(state, CONTEXT, incoming(2, '계속'), RoleId.REVIEW)
                self.assertEqual(1, delegate.calls)

    def test_multi_turn_estimated_failure_uses_entire_invocation_reservation(self):
        class Planning(Delegate):
            def invocation_token_estimate(self, input_tokens, output_tokens):
                return 8 * (input_tokens + output_tokens)

            def respond(self, *args, **kwargs):
                return self.respond_as(*args, **kwargs)

        with temporary_directory() as directory:
            store, state, budget = self.manager(Path(directory), conversation_tokens=1000)
            delegate = Planning(HermesExecutionError('provider', usage=TokenUsage(2, 1, estimated=True)))
            backend = GovernedAgentBackend(delegate, budget, response_reserve_tokens=5)
            with self.assertRaises(HermesExecutionError):
                backend.respond(state, CONTEXT, incoming(1, '계획'))
            self.assertEqual(120, store.usage_total(state.run_id))
            self.assertTrue(store.has_budget_anomaly(state.run_id))
            self.assertEqual(0, store.reserved_token_total(state.run_id))

    def test_runner_stop_before_start_and_after_report_keep_execution_boundary(self):
        for before_start in (True, False):
            with self.subTest(before_start=before_start):
                fixture = self.runner_fixture()
                if before_start:
                    fixture.runner.request_stop()

                def start(command, **kwargs):
                    fixture.runner._stop_requested.set()
                    return contract.FakeProcess(command, {'status': 'succeeded', 'text': '결과'},
                        usage={'input_tokens': 12, 'output_tokens': 5, 'total_tokens': 17})

                with patch('app.services.hermes.runner.subprocess.Popen', side_effect=start) as launch:
                    with self.assertRaises(HermesCancelled) as raised:
                        fixture.runner.run('RUN-CONTRACT', 'chat-001', RoleId.REVIEW,
                                           fixture.root, '질문', allow_writes=False)
                self.assertEqual(int(not before_start), launch.call_count)
                self.assertEqual(0 if before_start else 17, raised.exception.usage.total_tokens)
                self.assertEqual('startup' if before_start else 'execution', raised.exception.category)

    def check_planning(self, *, outcome=None, post_error=None, snapshot_error=None,
                       expected_tokens=80, unknown=False):
        with temporary_directory() as directory:
            root = Path(directory)
            store, state, budget = self.manager(root)
            state = store.save_run(replace(state, repository=str(root)))
            runner = SimpleNamespace(settings=SimpleNamespace(provider='openai-codex',
                planning=SimpleNamespace(model='fixture', reasoning='low'),
                roles={RoleId.DEVELOPMENT: SimpleNamespace(max_turns=8)}))
            calls = []

            def run(*args, **kwargs):
                calls.append(1)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome or HermesResult('{"message":"적용 금지","stages":[]}', TokenUsage(70, 10), 0.1)

            runner.run = run
            foundation = SimpleNamespace(root=root, roles={RoleId.DEVELOPMENT: SimpleNamespace(display_name='빌더')})
            delegate = HermesConversationBackend(foundation, runner, sandbox=object())
            backend = GovernedAgentBackend(delegate, budget, response_reserve_tokens=5)

            def snapshot():
                if snapshot_error:
                    raise snapshot_error
                return object()

            def postcheck(*args, **kwargs):
                raise post_error

            repository = SimpleNamespace(path=root, snapshot=snapshot, assert_position=postcheck)
            source_error = snapshot_error or post_error
            expected_error = HermesCancelled if isinstance(source_error, GitRepositoryCancelled) else type(source_error)
            with patch.object(delegate, '_prompt', return_value='계획 prompt'), \
                 patch('app.agents.conversation_backend.GitRepository', return_value=repository):
                with self.assertRaises(expected_error):
                    backend.respond(state, CONTEXT, incoming(1, '계획'))
            self.assertEqual(int(snapshot_error is None), len(calls))
            self.assertEqual(expected_tokens, store.usage_total(state.run_id))
            self.assertEqual(unknown, store.has_budget_anomaly(state.run_id))
            self.assertEqual(0, store.reserved_token_total(state.run_id))
            with store._connection() as connection:
                call = connection.execute('SELECT * FROM model_calls').fetchone()
            self.assertEqual('FAILED', call['status'])
            self.assertEqual(expected_tokens, json.loads(call['result_json'])['usage']['total_tokens'])

    def test_planning_postcheck_failure_and_cancellation_keep_reported_success_cost(self):
        for error in (GitRepositoryError('HEAD 변경'), GitRepositoryCancelled('Git 취소')):
            with self.subTest(error=type(error).__name__):
                self.check_planning(post_error=error)

    def test_planning_runner_failure_cost_survives_postcheck_error_or_cancel(self):
        for error in (GitRepositoryError('HEAD 변경'), GitRepositoryCancelled('Git 취소')):
            with self.subTest(error=type(error).__name__):
                self.check_planning(outcome=HermesExecutionError('provider', usage=TokenUsage(60, 10)),
                                    post_error=error, expected_tokens=70)

    def test_planning_precheck_and_runner_startup_failures_release_without_cost(self):
        for error in (GitRepositoryError('snapshot 실패'), GitRepositoryCancelled('snapshot 취소')):
            with self.subTest(error=type(error).__name__):
                self.check_planning(snapshot_error=error, expected_tokens=0)
        self.check_planning(outcome=HermesExecutionError('startup', category='startup'),
                            post_error=GitRepositoryError('postcheck 실패'), expected_tokens=0)

    def test_planning_unknown_cost_survives_postcheck(self):
        self.check_planning(outcome=HermesCancelled('report 없는 취소'),
                            post_error=GitRepositoryCancelled('Git 취소'), expected_tokens=144, unknown=True)

    def analysis(self, store, root, budget, *, conversation='200', completed=False):
        request = RepositoryAnalysisRequest.create(channel='telegram', conversation_id=conversation,
            user_id='100', source_message_id='1', role_id=RoleId.REVIEW.value,
            request_text='프로젝트 분석', repository_path=str(root), repository_identity='a' * 64,
            commit_sha='b' * 40, branch='main')
        state = RunStateMachine(store).create_run(request.analysis_id)
        store.create_repository_analysis(request)
        job = store.claim_next_repository_analysis('old-worker', lease_seconds=1)
        store.set_repository_analysis_model_call_state(request.analysis_id, 'old-worker', 'STARTED')
        backend = GovernedTeamConversationBackend(Delegate(), budget, response_reserve_tokens=5)
        message = IncomingMessage('telegram', conversation, '100',
                                 f'repository-analysis:{request.analysis_id}:{job["checkpoint"] + 1}', '묶음')
        invocation = TeamConversationRequest(RoleId.REVIEW, CONTEXT, call_purpose='repository_analysis_batch')
        logical_id = backend._logical_id(state, message, invocation)
        prepared = backend._prepare(state, message, invocation, logical_id, 1)
        if completed:
            backend._finish(state, message, invocation, prepared, AgentReply('보존 결과', usage=TokenUsage(2, 1)))
        return state, backend, prepared, logical_id

    def expire(self, store, state):
        with store._connection() as connection:
            connection.execute('UPDATE repository_analysis_jobs SET lease_until=? WHERE analysis_id=?',
                               ('2000-01-01T00:00:00+00:00', state.run_id))

    def test_stale_analysis_recovery_is_scoped_and_idempotent(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store = StateStore(root / 'state.db')
            budget = BudgetManager(BudgetPolicy(), store)
            state, backend, prepared, _ = self.analysis(store, root, budget)
            live, _, live_prepared, _ = self.analysis(store, root, budget, conversation='201')
            self.expire(store, state)
            worker = RepositoryAnalysisWorker(store, object(), backend, ContextService(store))
            worker._recover_stale()
            worker._recover_stale()
            self.assertEqual('NEEDS_ATTENTION', store.repository_analysis(state.run_id)['status'])
            self.assertEqual('INTERRUPTED', store.model_call(prepared[0])['status'])
            self.assertEqual(15, store.usage_total(state.run_id))
            self.assertEqual(0, store.reserved_token_total(state.run_id))
            self.assertTrue(store.has_budget_anomaly(state.run_id))
            self.assertEqual('RUNNING', store.model_call(live_prepared[0])['status'])
            self.assertEqual(15, store.reserved_token_total(live.run_id))
            self.assertEqual(0, store.usage_total(live.run_id))
            self.assertFalse(store.has_budget_anomaly(live.run_id))
            self.assertEqual(1, len(store.deliverable_outbound('telegram')))

    def test_stale_analysis_recovery_rolls_back_if_cost_or_notification_fails(self):
        for failing_method in ('recover_interrupted_calls', 'queue_outbound'):
            with self.subTest(method=failing_method), temporary_directory() as directory:
                root = Path(directory)
                store = StateStore(root / 'state.db')
                budget = BudgetManager(BudgetPolicy(), store)
                state, backend, prepared, _ = self.analysis(store, root, budget)
                self.expire(store, state)
                worker = RepositoryAnalysisWorker(store, object(), backend, ContextService(store))
                target = budget if failing_method == 'recover_interrupted_calls' else store
                with patch.object(target, failing_method, side_effect=RuntimeError('복구 중 장애')):
                    with self.assertRaises(RuntimeError):
                        worker._recover_stale()
                self.assertEqual('PROCESSING', store.repository_analysis(state.run_id)['status'])
                self.assertEqual('RUNNING', store.model_call(prepared[0])['status'])
                self.assertEqual(15, store.reserved_token_total(state.run_id))
                self.assertEqual(0, store.usage_total(state.run_id))
                self.assertFalse(store.has_budget_anomaly(state.run_id))
                worker._recover_stale()
                self.assertEqual(15, store.usage_total(state.run_id))
                self.assertEqual(0, store.reserved_token_total(state.run_id))

    def test_stale_analysis_keeps_completed_cache_and_actual_usage(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store = StateStore(root / 'state.db')
            budget = BudgetManager(BudgetPolicy(), store)
            state, backend, prepared, logical_id = self.analysis(store, root, budget, completed=True)
            saved = store.completed_model_call(logical_id)
            self.expire(store, state)
            worker = RepositoryAnalysisWorker(store, object(), backend, ContextService(store))
            worker._recover_stale()
            worker._recover_stale()
            self.assertEqual(saved, store.completed_model_call(logical_id))
            self.assertEqual('COMPLETED', store.model_call(prepared[0])['status'])
            self.assertEqual(3, store.usage_total(state.run_id))
            self.assertEqual(0, store.reserved_token_total(state.run_id))
            self.assertFalse(store.has_budget_anomaly(state.run_id))
            self.assertEqual(0, backend.delegate.calls)

    def test_gateway_replays_result_after_save_failure_at_limit_or_overage(self):
        for actual in (15, 30):
            with self.subTest(actual=actual), temporary_directory() as directory:
                root = Path(directory)
                delegate = Delegate(AgentReply('보존된 정상 답변', usage=TokenUsage(actual - 3, 3)))
                store, app = build_application(root, object(), team_backend=delegate)
                budget = BudgetManager(BudgetPolicy(conversation_tokens=15), store)
                app.router.budget = budget
                backend = GovernedTeamConversationBackend(delegate, budget, response_reserve_tokens=5)
                app.router.team_backend = backend
                message = incoming(1, '안녕')
                with patch.object(app.router, '_out', side_effect=RuntimeError('응답 저장 전 장애')):
                    with self.assertRaises(RuntimeError):
                        app.handle(message)
                outputs = app.handle(message)
                self.assertIn('보존된 정상 답변', outputs[0].text)
                self.assertEqual(1, delegate.calls)
                run_id = store.load_conversation('telegram', '200')['run_id']
                self.assertEqual(actual, store.usage_total(run_id))
                self.assertEqual(0, store.reserved_token_total(run_id))
                result = store.conversation_result('telegram', '200', message.external_message_id)
                self.assertEqual(0, result.attempts)
                self.assertEqual(0, result.successes)
                saved_outbound = len(store.deliverable_outbound('telegram'))
                self.assertEqual((), app.handle(message))
                self.assertEqual(saved_outbound, len(store.deliverable_outbound('telegram')))
                self.assertEqual((), app.handle(replace(message, user_id='999')))
                blocked = app.handle(incoming(2, '새 답변'))
                self.assertEqual('BUDGET_LIMIT', blocked[0].metadata['request_result']['reason'])
                self.assertEqual(1, delegate.calls)
                self.assertEqual(actual, store.usage_total(run_id))

    def test_queue_resume_after_response_save_failure_reuses_completed_invocation(self):
        with temporary_directory() as directory:
            root = Path(directory)
            delegate = Delegate(AgentReply('큐 재개 결과', usage=TokenUsage(12, 3)))
            store, app = build_application(root, object(), team_backend=delegate)
            budget = BudgetManager(BudgetPolicy(conversation_tokens=15), store)
            app.router.budget = budget
            app.router.team_backend = GovernedTeamConversationBackend(delegate, budget, response_reserve_tokens=5)
            queue = ConversationQueue(store)
            queue.enqueue(incoming(1, '안녕'))
            worker = ConversationWorker(store, app.router)
            with patch.object(app.router, '_out', side_effect=RuntimeError('응답 저장 전 장애')):
                self.assertTrue(worker.run_once())
            self.assertEqual('NEEDS_ATTENTION', queue.status('telegram', '200'))
            self.assertIsNotNone(queue.resume('telegram', '200'))
            self.assertTrue(worker.run_once())
            self.assertEqual('COMPLETED', queue.status('telegram', '200'))
            self.assertTrue(any('큐 재개 결과' in row['text'] for row in store.deliverable_outbound('telegram')))
            self.assertEqual(1, delegate.calls)
            run_id = store.load_conversation('telegram', '200')['run_id']
            self.assertEqual(15, store.usage_total(run_id))
            self.assertFalse(worker.run_once())

    def test_cache_only_replay_checks_exact_input_and_never_starts_new_calls(self):
        with temporary_directory() as directory:
            store, state, budget = self.manager(Path(directory))
            delegate = Delegate()
            backend = GovernedTeamConversationBackend(delegate, budget, response_reserve_tokens=5)
            message = incoming(1, '원래 입력')
            backend.respond_as(state, CONTEXT, message, RoleId.REVIEW)
            replay = replace(message, metadata={'model_cache_replay': True})
            self.assertTrue(backend.respond_as(state, CONTEXT, replay, RoleId.REVIEW).metadata['cached_call'])
            for next_state, context, next_message, role in [
                (state, CONTEXT, replace(replay, text='변경 입력'), RoleId.REVIEW),
                (state, replace(CONTEXT, characters=10), replay, RoleId.REVIEW),
                (state, CONTEXT, replay, RoleId.DEVELOPMENT),
                (replace(state, run_id='RUN-OTHER'), CONTEXT, replay, RoleId.REVIEW),
                (state, CONTEXT, replace(replay, external_message_id='새 요청'), RoleId.REVIEW),
            ]:
                with self.subTest(message=next_message, role=role):
                    with self.assertRaises(HermesExecutionError):
                        backend.respond_as(next_state, context, next_message, role)
            self.assertEqual(1, delegate.calls)
            self.assertEqual(3, store.usage_total(state.run_id))
            self.assertEqual(0, store.reserved_token_total(state.run_id))

    def test_model_cache_resume_preserves_owner_run_and_incomplete_call_checks(self):
        for changed in ('owner', 'run', 'RUNNING', 'INTERRUPTED', 'FAILED'):
            with self.subTest(changed=changed), temporary_directory() as directory:
                root = Path(directory)
                delegate = Delegate()
                store, app = build_application(root, object(), team_backend=delegate)
                budget = BudgetManager(BudgetPolicy(), store)
                app.router.budget = budget
                app.router.team_backend = GovernedTeamConversationBackend(delegate, budget, response_reserve_tokens=5)
                queue = ConversationQueue(store)
                queue.enqueue(incoming(1, '안녕'))
                worker = ConversationWorker(store, app.router)
                with patch.object(app.router, '_out', side_effect=RuntimeError('저장 전 장애')):
                    self.assertTrue(worker.run_once())
                binding = store.load_conversation('telegram', '200')
                if changed in ('owner', 'run'):
                    store.bind_conversation('telegram', '200',
                        '999' if changed == 'owner' else '100',
                        store.create_run(RunState('RUN-OTHER')).run_id if changed == 'run' else binding['run_id'],
                        binding['active_role'])
                else:
                    with store._connection() as connection:
                        connection.execute('UPDATE model_calls SET status = ?', (changed,))
                self.assertEqual('NEEDS_ATTENTION', queue.resume('telegram', '200')['status'])
                self.assertFalse(worker.run_once())
                self.assertEqual(1, delegate.calls)
