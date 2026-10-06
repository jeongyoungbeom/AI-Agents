import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from app.contracts import RoleId, RunState, TokenUsage
from app.contracts.outcomes import RequestOutcome
from app.gateway.core import AgentReply
from app.gateway.core.governed_backend import GovernedTeamConversationBackend
from app.gateway.core.role_routing import RoleResolver
from app.gateway.conversation_worker import ConversationQueue, ConversationWorker
from app.services.budget import BudgetManager, BudgetPolicy
from app.services.hermes import HermesExecutionError
from app.services.repository import RepositoryToolRequest
from tests.gateway.support import build_application, temporary_directory
from tests.gateway.test_conversation import incoming
from tests.gateway.test_s07_collaboration import ScriptedBackend, consult
from tests.gateway.test_s03_context import Reader, Tools


class S07ReviewResolutionTests(unittest.TestCase):
    def governed(self, root, *steps):
        delegate = ScriptedBackend(*steps)
        store, app = build_application(root, object(), team_backend=delegate)
        budget = BudgetManager(BudgetPolicy(conversation_tokens=1000), store)
        app.router.budget = budget
        app.router.team_backend = GovernedTeamConversationBackend(delegate, budget, response_reserve_tokens=5)
        return store, app, delegate

    def finished(self, store):
        return [item['data'] for item in store.list_events(store.load_conversation('telegram', '200')['run_id'])
                if item['event_type'] == 'AGENT_CONSULTATION_FINISHED']

    def test_queue_recovers_saved_results_at_each_persistence_boundary_without_new_calls(self):
        for boundary in ('notice', 'result', 'final', 'outcome'):
            with self.subTest(boundary=boundary), temporary_directory() as directory:
                store, app, delegate = self.governed(Path(directory), consult(RoleId.DEVELOPMENT),
                    AgentReply('BUILDER_OBTAINED_RESULT'), AgentReply('OWNER_FINAL_SYNTHESIS'))
                queue = ConversationQueue(store)
                message = incoming(1, '센티널아, 빌더한테 물어봐')
                queue.enqueue(message)
                worker = ConversationWorker(store, app.router)
                original_out, original_emit = app.router._out, app.router.logger.emit
                original_result = store.save_conversation_result

                def out(incoming, text, *args, **kwargs):
                    if boundary == 'final' and text == 'OWNER_FINAL_SYNTHESIS':
                        raise RuntimeError('final persistence interrupted')
                    result = original_out(incoming, text, *args, **kwargs)
                    if boundary == 'notice' and '[센티널 → 빌더]' in text:
                        raise RuntimeError('notice persistence interrupted')
                    return result

                def emit(run_id, event_type, *args, **kwargs):
                    result = original_emit(run_id, event_type, *args, **kwargs)
                    if boundary == 'result' and event_type == 'AGENT_CONSULTATION_FINISHED':
                        raise RuntimeError('consultation persistence interrupted')
                    return result

                def save(*args):
                    if boundary == 'outcome' and args[-1].outcome == RequestOutcome.SUCCESS:
                        raise RuntimeError('outcome persistence interrupted')
                    return original_result(*args)

                with patch.object(app.router, '_out', side_effect=out), \
                     patch.object(app.router.logger, 'emit', side_effect=emit), \
                     patch.object(store, 'save_conversation_result', side_effect=save):
                    self.assertTrue(worker.run_once())
                self.assertEqual('NEEDS_ATTENTION', queue.status('telegram', '200'))
                self.assertEqual([], store.conversation_responses('telegram', '200', 'message-1'))
                run_id = store.load_conversation('telegram', '200')['run_id']
                calls_before, usage_before = len(delegate.calls), store.usage_total(run_id)
                self.assertEqual('QUEUED', queue.resume('telegram', '200')['status'])
                self.assertTrue(worker.run_once())
                self.assertEqual('COMPLETED', queue.status('telegram', '200'))
                output = '\n'.join(row['text'] for row in store.deliverable_outbound('telegram'))
                result = store.conversation_result('telegram', '200', 'message-1')
                self.assertEqual(calls_before, len(delegate.calls))
                self.assertEqual(usage_before, store.usage_total(run_id))
                self.assertEqual(0, store.reserved_token_total(run_id))
                self.assertEqual(1, len(self.finished(store)))
                self.assertIn('[센티널]', output)
                if boundary in ('final', 'outcome'):
                    self.assertIn('OWNER_FINAL_SYNTHESIS', output)
                    self.assertEqual(RequestOutcome.SUCCESS, result.outcome)
                    self.assertEqual(['[센티널]\nOWNER_FINAL_SYNTHESIS'],
                        store.conversation_responses('telegram', '200', 'message-1'))
                else:
                    self.assertEqual(RequestOutcome.PARTIAL, result.outcome)
                    self.assertIn('INTERRUPTED', output)
                    self.assertIn('미확인' if boundary == 'notice' else 'BUILDER_OBTAINED_RESULT', output)
                count = len(store.deliverable_outbound('telegram'))
                self.assertFalse(worker.run_once())
                self.assertEqual(count, len(store.deliverable_outbound('telegram')))

    def test_direct_retry_uses_original_final_input_and_preserves_usage(self):
        with temporary_directory() as directory:
            store, app, delegate = self.governed(Path(directory), consult(RoleId.DEVELOPMENT),
                AgentReply('BUILDER_RESULT'), AgentReply('OWNER_RESULT'))
            original = app.router._out
            def out(message, text, *args, **kwargs):
                if text == 'OWNER_RESULT':
                    raise RuntimeError('interrupted')
                return original(message, text, *args, **kwargs)
            message = incoming(1, '센티널아 빌더한테 물어봐')
            with patch.object(app.router, '_out', side_effect=out):
                with self.assertRaises(RuntimeError):
                    app.handle(message)
            output = app.handle(message)
            self.assertIn('OWNER_RESULT', output[-1].text)
            self.assertEqual(3, len(delegate.calls))
            self.assertEqual(9, store.usage_total(store.load_conversation('telegram', '200')['run_id']))
            self.assertEqual(1, len(self.finished(store)))

    def test_recovery_is_atomic_and_another_resume_does_not_duplicate_results(self):
        with temporary_directory() as directory:
            store, app, delegate = self.governed(Path(directory), consult(RoleId.DEVELOPMENT),
                AgentReply('BUILDER_RESULT'), AgentReply('OWNER_RESULT'))
            queue = ConversationQueue(store)
            queue.enqueue(incoming(1, '센티널아 빌더한테 물어봐'))
            worker = ConversationWorker(store, app.router)
            original = app.router._out
            def out(message, text, *args, **kwargs):
                if text == 'OWNER_RESULT':
                    raise RuntimeError('interrupted')
                return original(message, text, *args, **kwargs)
            with patch.object(app.router, '_out', side_effect=out):
                worker.run_once()
            queue.resume('telegram', '200')
            with patch.object(app.router, '_out', side_effect=RuntimeError('recovery interrupted')):
                worker.run_once()
            self.assertEqual([], store.conversation_responses('telegram', '200', 'message-1'))
            queue.resume('telegram', '200')
            worker.run_once()
            self.assertEqual(3, len(delegate.calls))
            self.assertEqual(1, len(self.finished(store)))
            self.assertEqual(['[센티널]\nOWNER_RESULT'], store.conversation_responses('telegram', '200', 'message-1'))

    def test_recovery_rejects_changed_input_owner_or_run_even_with_saved_notice(self):
        for change in ('input', 'owner', 'run'):
            with self.subTest(change=change), temporary_directory() as directory:
                store, app, delegate = self.governed(Path(directory), consult(RoleId.DEVELOPMENT),
                    AgentReply('BUILDER_RESULT'), AgentReply('OWNER_RESULT'))
                message = incoming(1, '센티널아 빌더한테 물어봐')
                queue = ConversationQueue(store)
                queue.enqueue(message)
                worker = ConversationWorker(store, app.router)
                original = app.router._out
                def out(message, text, *args, **kwargs):
                    if text == 'OWNER_RESULT':
                        raise RuntimeError('interrupted')
                    return original(message, text, *args, **kwargs)
                with patch.object(app.router, '_out', side_effect=out):
                    worker.run_once()
                binding = store.load_conversation('telegram', '200')
                if change == 'input':
                    with self.assertRaises(RuntimeError):
                        app.router.route_result(replace(message, text=message.text + ' 원문 변경',
                            metadata={'model_cache_replay': True}))
                else:
                    store.bind_conversation('telegram', '200', '999' if change == 'owner' else '100',
                        store.create_run(RunState('RUN-OTHER')).run_id if change == 'run' else binding['run_id'],
                        binding['active_role'])
                    self.assertEqual('NEEDS_ATTENTION', queue.resume('telegram', '200')['status'])
                    self.assertFalse(worker.run_once())
                self.assertEqual(3, len(delegate.calls))

    def test_recovery_preserves_new_child_evidence_and_untrusted_provenance(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store, app, delegate = self.governed(root, consult(RoleId.DEVELOPMENT),
                AgentReply('', repository_tools=(RepositoryToolRequest('read_file', path='src/b.py'),)),
                AgentReply('src/b.py:1 확보 결과'), AgentReply('근거 종합'))
            app.router.repository_reader, app.router.repository_tools = Reader(), Tools()
            app.handle(incoming(1, str(root / 'selected-repository')))
            app.handle(incoming(2, '이 프로젝트 사용 승인해'))
            message = incoming(3, '센티널아 프로젝트 설명해. 빌더한테 물어봐')
            queue = ConversationQueue(store)
            queue.enqueue(message)
            worker = ConversationWorker(store, app.router)
            emit = app.router.logger.emit
            def fail(run_id, event_type, *args, **kwargs):
                if event_type == 'AGENT_CONSULTATION_FINISHED':
                    raise RuntimeError('interrupted')
                return emit(run_id, event_type, *args, **kwargs)
            with patch.object(app.router.logger, 'emit', side_effect=fail):
                worker.run_once()
            queue.resume('telegram', '200')
            worker.run_once()
            result = self.finished(store)[0]
            self.assertTrue(result['untrusted_repository_data'])
            self.assertEqual('b' * 40, result['head_sha'])
            self.assertTrue(any('src/b.py' in ref for ref in result['evidence_refs']))
            binding = store.load_conversation('telegram', '200')
            final = store.list_messages(binding['run_id'])[-1]
            self.assertTrue(final['data']['untrusted_repository_data'])
            self.assertEqual('b' * 40, final['data']['head_sha'])
            self.assertEqual(3, len(delegate.calls))

    def test_saved_final_time_boundary_keeps_partial_outcome_when_replayed(self):
        with temporary_directory() as directory:
            clock = [0.0]
            def obtained(*args):
                clock[0] = 301.0
                return AgentReply('확보 결과')
            store, app, delegate = self.governed(Path(directory), consult(RoleId.DEVELOPMENT),
                obtained, AgentReply('시간 경계의 최종 종합'))
            queue = ConversationQueue(store)
            queue.enqueue(incoming(1, '센티널아 빌더한테 물어봐'))
            worker = ConversationWorker(store, app.router)
            out = app.router._out
            def fail(message, text, *args, **kwargs):
                if text == '시간 경계의 최종 종합':
                    raise RuntimeError('interrupted')
                return out(message, text, *args, **kwargs)
            with patch('app.gateway.core.conversation.time.monotonic', side_effect=lambda: clock[0]), \
                 patch.object(app.router, '_out', side_effect=fail):
                worker.run_once()
            queue.resume('telegram', '200')
            worker.run_once()
            result = store.conversation_result('telegram', '200', 'message-1')
            self.assertEqual((RequestOutcome.PARTIAL, 'TIME_LIMIT'), (result.outcome, result.reason))
            self.assertEqual(3, len(delegate.calls))

    def test_final_response_cache_also_checks_owner_and_run(self):
        for changed in ('owner', 'run'):
            with self.subTest(changed=changed), temporary_directory() as directory:
                store, app, delegate = self.governed(Path(directory), consult(RoleId.DEVELOPMENT),
                    AgentReply('빌더 결과'), AgentReply('최종 종합'))
                queue = ConversationQueue(store)
                queue.enqueue(incoming(1, '센티널아 빌더한테 물어봐'))
                worker = ConversationWorker(store, app.router)
                finish = store.finish_conversation_job
                def fail(*args, **kwargs):
                    if args[2] == 'COMPLETED':
                        raise RuntimeError('outbox commit interrupted')
                    return finish(*args, **kwargs)
                with patch.object(store, 'finish_conversation_job', side_effect=fail):
                    worker.run_once()
                # 확정 응답 캐시에서도 현재 소유자/run 경계를 우회하지 않는다.
                store.save_conversation_result('telegram', '200', 'message-1',
                    replace(store.conversation_result('telegram', '200', 'message-1'), reason=''))
                binding = store.load_conversation('telegram', '200')
                store.bind_conversation('telegram', '200', '999' if changed == 'owner' else '100',
                    store.create_run(RunState('RUN-OTHER')).run_id if changed == 'run' else binding['run_id'],
                    binding['active_role'])
                self.assertEqual('NEEDS_ATTENTION', queue.resume('telegram', '200')['status'])
                self.assertFalse(worker.run_once())
                self.assertEqual(3, len(delegate.calls))

    def test_recovery_after_a_retry_keeps_the_original_logical_input(self):
        with temporary_directory() as directory:
            store, app, delegate = self.governed(Path(directory),
                HermesExecutionError('temporary startup failure', category='startup', retryable=True),
                consult(RoleId.DEVELOPMENT), AgentReply('빌더 결과'), AgentReply('재시도 뒤 최종 종합'))
            app.router.budget.policy = replace(app.router.budget.policy, retries={'technical_error': 1})
            queue = ConversationQueue(store)
            queue.enqueue(incoming(1, '센티널아 빌더한테 물어봐'))
            worker = ConversationWorker(store, app.router)
            out = app.router._out
            def fail(message, text, *args, **kwargs):
                if text == '재시도 뒤 최종 종합':
                    raise RuntimeError('interrupted')
                return out(message, text, *args, **kwargs)
            with patch.object(app.router, '_out', side_effect=fail):
                worker.run_once()
            queue.resume('telegram', '200')
            worker.run_once()
            self.assertEqual('COMPLETED', queue.status('telegram', '200'))
            self.assertEqual(4, len(delegate.calls))
            self.assertEqual(9, store.usage_total(store.load_conversation('telegram', '200')['run_id']))
            self.assertIn('재시도 뒤 최종 종합', store.conversation_responses('telegram', '200', 'message-1')[0])

    def test_recovered_overage_result_keeps_partial_budget_status(self):
        with temporary_directory() as directory:
            store, app, delegate = self.governed(Path(directory), consult(RoleId.DEVELOPMENT),
                AgentReply('초과 전에 확보한 의견'))
            queue = ConversationQueue(store)
            queue.enqueue(incoming(1, '센티널아 빌더한테 물어봐'))
            worker = ConversationWorker(store, app.router)
            respond, emit = delegate.respond_as, app.router.logger.emit
            def costly(*args, **kwargs):
                reply = respond(*args, **kwargs)
                return replace(reply, usage=TokenUsage(15, 5)) if args[3] == RoleId.DEVELOPMENT else reply
            def fail(run_id, event_type, *args, **kwargs):
                if event_type == 'AGENT_CONSULTATION_FINISHED':
                    raise RuntimeError('interrupted')
                return emit(run_id, event_type, *args, **kwargs)
            with patch.object(delegate, 'respond_as', side_effect=costly), \
                 patch.object(app.router.logger, 'emit', side_effect=fail):
                worker.run_once()
            queue.resume('telegram', '200')
            worker.run_once()
            self.assertEqual(('PARTIAL', 'BUDGET_LIMIT'),
                (self.finished(store)[0]['status'], self.finished(store)[0]['reason']))
            self.assertEqual(2, len(delegate.calls))
            self.assertEqual(23, store.usage_total(store.load_conversation('telegram', '200')['run_id']))
            self.assertIn('초과 전에 확보한 의견', store.conversation_responses('telegram', '200', 'message-1')[0])

    def test_time_boundary_returns_saved_results_once_for_success_budget_denial_or_failure(self):
        for nested in (False, True):
            for ending in ('success', 'budget', 'failure'):
                with self.subTest(nested=nested, ending=ending), temporary_directory() as directory:
                    clock = [0.0]
                    def obtained(*args):
                        clock[0] = 301.0
                        return AgentReply('TIME_BOUNDARY_OBTAINED_RESULT')
                    steps = [consult(RoleId.DEVELOPMENT)]
                    if nested:
                        steps.append(consult(RoleId.IMPROVEMENT, speaker=RoleId.DEVELOPMENT))
                    steps += [obtained, RuntimeError('model failed') if ending == 'failure' else AgentReply('정상 담당 종합')]
                    if nested:
                        steps.append(RuntimeError('owner model failed') if ending == 'failure' else AgentReply('정상 담당 종합'))
                    class TimedBackend(ScriptedBackend):
                        def prompt_token_upper_bound(self, *args, **kwargs):
                            return 40 if ending == 'budget' and kwargs.get('call_purpose') == 'agent_loop_final' else 10
                    delegate = TimedBackend(*steps)
                    store, app = build_application(Path(directory), object(), team_backend=delegate)
                    if ending != 'failure':
                        budget = BudgetManager(BudgetPolicy(conversation_tokens=40), store)
                        app.router.budget = budget
                        app.router.team_backend = GovernedTeamConversationBackend(delegate, budget, response_reserve_tokens=5)
                    with patch('app.gateway.core.conversation.time.monotonic', side_effect=lambda: clock[0]):
                        output = app.handle(incoming(1, '센티널아 다른 에이전트에게 물어봐'))
                    text = '\n'.join(item.text for item in output)
                    self.assertIn('[센티널]', text)
                    self.assertEqual(1, text.count('[센티널]\n'))
                    self.assertEqual(RequestOutcome.PARTIAL, store.conversation_result('telegram', '200', 'message-1').outcome)
                    if ending != 'success':
                        self.assertIn('TIME_BOUNDARY_OBTAINED_RESULT', text)
                        self.assertIn('BUDGET_LIMIT' if ending == 'budget' else 'MODEL_FAILED', text)
                    self.assertEqual(2 + int(nested) + int(ending != 'budget') * (1 + int(nested)), len(delegate.calls))

    def test_target_prohibitions_override_general_permission_without_blocking_other_roles(self):
        resolver = RoleResolver()
        for verb in ('이야기', '논의', '상의', '물어보', '호출', '검토', '말', '부탁', '요청',
                     '이야기 ', '논의 ', '상의 ', '물어 보'):
            negative = verb + ('지' if verb in ('물어보', '물어 보') else '하지')
            text = f'센티널아 다른 에이전트에게 물어봐. 빌더한테는 {negative} 마.'
            with self.subTest(verb=verb), temporary_directory() as directory:
                intent = resolver.interpret(text, RoleId.DEVELOPMENT)
                self.assertNotIn(RoleId.DEVELOPMENT, intent.allowed_delegate_roles)
                self.assertIn(RoleId.IMPROVEMENT, intent.allowed_delegate_roles)
                delegate = ScriptedBackend(consult(RoleId.DEVELOPMENT), AgentReply('금지된 호출'))
                store, app = build_application(Path(directory), object(), team_backend=delegate)
                app.handle(incoming(1, text))
                self.assertEqual([RoleId.REVIEW], [item[0] for item in delegate.calls])
                self.assertEqual([], self.finished(store))

    def test_positive_quoted_and_question_consultation_clauses_keep_permission_contract(self):
        resolver = RoleResolver()
        for verb in ('이야기해', '이야기해서', '논의해', '상의해'):
            with self.subTest(verb=verb):
                self.assertEqual((RoleId.DEVELOPMENT,), resolver.interpret(
                    f'센티널아 빌더한테 {verb}.', RoleId.REVIEW).allowed_delegate_roles)
                for quote in (f'"빌더한테 {verb}"라는 문장을 설명해', f'빌더한테 {verb}라는 게 무슨 뜻이야?',
                              f'빌더한테 {verb}는지 알려줘'):
                    self.assertEqual((), resolver.interpret(quote, RoleId.REVIEW).allowed_delegate_roles)
        text = '센티널아 다른 에이전트에게 물어봐. "빌더한테는 이야기하지 마"라는 문장을 설명해.'
        self.assertIn(RoleId.DEVELOPMENT, resolver.interpret(text, RoleId.REVIEW).allowed_delegate_roles)


if __name__ == '__main__':
    unittest.main()
