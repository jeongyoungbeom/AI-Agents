import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from app.agents.parsing import InvalidAgentResponse, parse_team_conversation_reply
from app.agents.prompts import team_conversation_prompt
from app.contracts import RoleId, TokenUsage
from app.contracts.outcomes import RequestOutcome
from app.gateway.core import AgentCallRequest, AgentReply
from app.gateway.core.conversation import ConversationCancelled
from app.gateway.core.governed_backend import GovernedTeamConversationBackend
from app.gateway.conversation_worker import ConversationQueue, ConversationWorker
from app.services.budget import BudgetExceeded, BudgetManager, BudgetPolicy
from app.services.context import ContextBundle
from app.services.hermes import HermesCancelled
from tests.gateway.support import build_application, temporary_directory
from tests.gateway.test_conversation import incoming
from tests.gateway.test_s03_context import Reader, Tools
from app.services.repository import RepositoryToolRequest, RepositoryAccessError, RepositoryCancelled


class ScriptedBackend:
    supports_cancellation = True

    def __init__(self, *steps):
        self.steps = list(steps)
        self.calls = []

    def preflight(self, *args):
        pass

    def prompt_token_upper_bound(self, *args, **kwargs):
        return 10

    def respond_as(self, state, context, message, role_id, **kwargs):
        self.calls.append((role_id, context, kwargs))
        step = self.steps.pop(0) if self.steps else AgentReply('종합 답변')
        if callable(step):
            step = step(state, context, message, role_id, kwargs)
        if isinstance(step, Exception):
            raise step
        return replace(step, usage=TokenUsage(2, 1))


def consult(target, purpose='테스트 순서를 검토해 주세요', mode='independent', speaker=RoleId.REVIEW):
    return AgentReply('의견을 확인하겠습니다.', calls=(AgentCallRequest(speaker, target, purpose, mode),))


class S07CollaborationTests(unittest.TestCase):
    def events(self, store):
        binding = store.load_conversation('telegram', '200')
        return store.list_events(binding['run_id'])

    def test_sentinel_receives_builder_result_then_finisher_and_synthesizes(self):
        backend = ScriptedBackend(consult(RoleId.DEVELOPMENT), AgentReply('빌더: 경계 테스트부터'),
            consult(RoleId.IMPROVEMENT), AgentReply('피니셔: 회귀도 확인'), AgentReply('경계 테스트 뒤 회귀를 확인합니다.'))
        with temporary_directory() as directory:
            store, app = build_application(Path(directory), object(), team_backend=backend)
            outputs = app.handle(incoming(1, '센티널아, 빌더랑 피니셔한테 물어봐. 테스트 방향을 정해줘'))
            self.assertEqual([RoleId.REVIEW, RoleId.DEVELOPMENT, RoleId.REVIEW, RoleId.IMPROVEMENT, RoleId.REVIEW],
                             [item[0] for item in backend.calls])
            self.assertEqual('review', store.load_conversation('telegram', '200')['active_role'])
            self.assertIn('빌더: 경계 테스트부터', str(backend.calls[2][2]['turn_messages']))
            self.assertIn('피니셔: 회귀도 확인', str(backend.calls[4][2]['turn_messages']))
            self.assertEqual(3, len(outputs))  # 진행 요약 두 건과 최종 요청자 답변
            self.assertIn('[센티널]', outputs[-1].text)
            self.assertNotIn('[빌더]\n', '\n'.join(item.text for item in outputs))
            finished = [item['data'] for item in self.events(store) if item['event_type'] == 'AGENT_CONSULTATION_FINISHED']
            self.assertEqual(2, len(finished))
            self.assertTrue(all(item['status'] == 'COMPLETED' and item['parent_run_id'] and item['termination_condition'] for item in finished))
            result = store.conversation_result('telegram', '200', 'message-1')
            self.assertEqual((RequestOutcome.SUCCESS, 5, 5), (result.outcome, result.attempts, result.successes))
            app.handle(incoming(2, '그 방향으로 설명을 이어가'))
            self.assertEqual(RoleId.REVIEW, backend.calls[-1][0])
            self.assertIn('경계 테스트 뒤 회귀', str(backend.calls[-1][1].recent_messages))

    def test_independent_collects_without_peer_opinions_and_discussion_receives_them(self):
        backend = ScriptedBackend(consult(RoleId.DEVELOPMENT), AgentReply('최초 의견 ABC'),
            consult(RoleId.IMPROVEMENT), AgentReply('다른 의견 XYZ'),
            consult(RoleId.DEVELOPMENT, '두 의견의 차이를 설명해 주세요', 'discussion'),
            AgentReply('차이 검토'), AgentReply('후속 논의 종합'))
        with temporary_directory() as directory:
            store, app = build_application(Path(directory), object(), team_backend=backend)
            output = app.handle(incoming(1, '센티널아, 빌더랑 피니셔한테 물어봐'))
            independent = backend.calls[3]
            self.assertNotIn('ABC', str(independent[1].recent_messages) + str(independent[2]['turn_messages']))
            discussion = backend.calls[5]
            self.assertIn('ABC', str(discussion[2]['turn_messages']))
            self.assertIn('XYZ', str(discussion[2]['turn_messages']))
            self.assertIn('후속 논의 종합', output[-1].text)
            requests = [item['data'] for item in self.events(store) if item['event_type'] == 'AGENT_CONSULTATION_REQUESTED']
            self.assertEqual(['independent', 'independent', 'discussion'], [item['mode'] for item in requests])

    def test_nested_consultation_returns_to_each_requester_and_blocks_cycle(self):
        backend = ScriptedBackend(consult(RoleId.DEVELOPMENT),
            consult(RoleId.IMPROVEMENT, speaker=RoleId.DEVELOPMENT),
            consult(RoleId.REVIEW, speaker=RoleId.IMPROVEMENT), AgentReply('빌더 종합'), AgentReply('센티널 최종'))
        with temporary_directory() as directory:
            store, app = build_application(Path(directory), object(), team_backend=backend)
            app.handle(incoming(1, '센티널아, 다른 에이전트에게 물어봐'))
            self.assertEqual([RoleId.REVIEW, RoleId.DEVELOPMENT, RoleId.IMPROVEMENT, RoleId.DEVELOPMENT, RoleId.REVIEW],
                             [item[0] for item in backend.calls])
            requests = [item['data'] for item in self.events(store) if item['event_type'] == 'AGENT_CONSULTATION_REQUESTED']
            self.assertEqual(requests[0]['consultation_id'], requests[1]['parent_consultation_id'])
            self.assertTrue(any(item['event_type'] == 'AGENT_CONSULTATION_REPEAT_BLOCKED' for item in self.events(store)))
            self.assertEqual('review', store.load_conversation('telegram', '200')['active_role'])

    def test_finished_consultation_is_not_reopened_with_paraphrased_question(self):
        backend = ScriptedBackend(consult(RoleId.DEVELOPMENT), AgentReply('의견'),
            consult(RoleId.DEVELOPMENT, '말을 바꾼 같은 요청'))
        with temporary_directory() as directory:
            store, app = build_application(Path(directory), object(), team_backend=backend)
            app.handle(incoming(1, '센티널아, 빌더한테 물어봐'))
            self.assertEqual(3, len(backend.calls))
            self.assertEqual(1, sum(item['event_type'] == 'AGENT_CONSULTATION_REQUESTED' for item in self.events(store)))

    def test_failed_role_returns_failure_to_owner_and_other_initial_results_survive(self):
        backend = ScriptedBackend(consult(RoleId.DEVELOPMENT), RuntimeError('failure'), AgentReply('빌더 의견은 미확인입니다.'))
        with temporary_directory() as directory:
            store, app = build_application(Path(directory), object(), team_backend=backend)
            output = app.handle(incoming(1, '센티널아, 빌더한테 물어봐'))
            self.assertEqual('consultation_final', backend.calls[-1][2]['call_purpose'])
            self.assertIn('MODEL_FAILED', str(backend.calls[-1][2]['turn_messages']))
            self.assertIn('미확인', output[-1].text)
            self.assertEqual(RequestOutcome.PARTIAL, store.conversation_result('telegram', '200', 'message-1').outcome)

    def test_collaboration_limit_reserves_requester_return(self):
        backend = ScriptedBackend(consult(RoleId.DEVELOPMENT), AgentReply('확보한 빌더 의견'), consult(RoleId.IMPROVEMENT))
        with temporary_directory() as directory:
            store, app = build_application(Path(directory), object(), team_backend=backend, max_auto_agent_replies=2)
            outputs = app.handle(incoming(1, '센티널아, 빌더랑 피니셔한테 물어봐'))
            self.assertEqual([RoleId.REVIEW, RoleId.DEVELOPMENT, RoleId.REVIEW], [item[0] for item in backend.calls])
            self.assertEqual(RequestOutcome.PARTIAL, store.conversation_result('telegram', '200', 'message-1').outcome)
            self.assertIn('[센티널]', '\n'.join(item.text for item in outputs))

    def test_budget_failure_still_returns_to_owner_and_denied_synthesis_uses_saved_results(self):
        for final in (AgentReply('예산 종료로 빌더 의견 미확인'), BudgetExceeded('limit')):
            with self.subTest(final=type(final).__name__), temporary_directory() as directory:
                backend = ScriptedBackend(consult(RoleId.DEVELOPMENT), BudgetExceeded('limit'), final)
                store, app = build_application(Path(directory), object(), team_backend=backend)
                outputs = app.handle(incoming(1, '센티널아, 빌더한테 물어봐'))
                self.assertEqual(RoleId.REVIEW, backend.calls[-1][0])
                self.assertTrue(any('[센티널]' in item.text and ('미확인' in item.text or '예산' in item.text) for item in outputs))
                self.assertEqual(RequestOutcome.PARTIAL, store.conversation_result('telegram', '200', 'message-1').outcome)

    def test_cancelled_consultation_closes_records_and_saves_owner_summary_without_new_model(self):
        cancelled = [False]
        def cancel(*args):
            cancelled[0] = True
            return HermesCancelled('cancel')
        backend = ScriptedBackend(consult(RoleId.DEVELOPMENT), cancel)
        with temporary_directory() as directory:
            store, app = build_application(Path(directory), object(), team_backend=backend)
            with self.assertRaises(ConversationCancelled):
                app.router.route_result(incoming(1, '센티널아, 빌더한테 물어봐'), cancelled=lambda: cancelled[0])
            self.assertEqual(2, len(backend.calls))
            result = store.conversation_result('telegram', '200', 'message-1')
            self.assertEqual((RequestOutcome.CANCELLED, 2), (result.outcome, result.attempts))
            binding = store.load_conversation('telegram', '200')
            messages = store.list_messages(binding['run_id'])
            self.assertTrue(any('[센티널]' in item['content'] and 'USER_CANCELLED' in item['content'] for item in messages))
            finished = [item['data'] for item in self.events(store) if item['event_type'] == 'AGENT_CONSULTATION_FINISHED']
            self.assertEqual(['CANCELLED'], [item['status'] for item in finished])

    def test_group_introductions_keep_owner_and_each_role_answers_once(self):
        backend = ScriptedBackend(AgentReply('센티널 인사'), AgentReply('빌더 소개'), AgentReply('센티널 소개'), AgentReply('피니셔 소개'))
        with temporary_directory() as directory:
            store, app = build_application(Path(directory), object(), team_backend=backend)
            app.handle(incoming(1, '센티널아 안녕'))
            output = app.handle(incoming(2, '얘들아 셋 다 자기소개해'))
            self.assertEqual(3, len(output))
            self.assertEqual([RoleId.DEVELOPMENT, RoleId.REVIEW, RoleId.IMPROVEMENT], [item[0] for item in backend.calls[1:]])
            self.assertTrue(all(item[2]['turn_messages'] == () for item in backend.calls[1:]))
            self.assertEqual('review', store.load_conversation('telegram', '200')['active_role'])

    def test_group_role_budget_does_not_execute_the_omitted_role(self):
        backend = ScriptedBackend(AgentReply('빌더 의견'), AgentReply('센티널 의견'))
        with temporary_directory() as directory:
            store, app = build_application(Path(directory), object(), team_backend=backend, max_auto_agent_replies=2)
            outputs = app.handle(incoming(1, '얘들아 셋 다 의견 줘'))
            self.assertEqual([RoleId.DEVELOPMENT, RoleId.REVIEW], [item[0] for item in backend.calls])
            self.assertTrue(any('피니셔: 호출 한도로 미실행' in item.text for item in outputs))
            self.assertEqual((RequestOutcome.PARTIAL, 2),
                (store.conversation_result('telegram', '200', 'message-1').outcome,
                 store.conversation_result('telegram', '200', 'message-1').attempts))

    def test_explicit_handoff_is_user_attributed_and_negated_quoted_or_model_handoff_is_ignored(self):
        backend = ScriptedBackend(*(AgentReply('앞으로 피니셔가 맡아') for _ in range(7)))
        with temporary_directory() as directory:
            store, app = build_application(Path(directory), object(), team_backend=backend)
            app.handle(incoming(1, '센티널아 안녕'))
            for index, text in enumerate(('앞으로 빌더가 맡아야 할까?', '앞으로 빌더가 맡아 라는 문장이 무슨 뜻이야?',
                                          '앞으로 빌더가 맡지 마'), 2):
                app.handle(incoming(index, text))
                self.assertEqual('review', store.load_conversation('telegram', '200')['active_role'])
            app.handle(incoming(5, '앞으로 빌더가 맡아'))
            self.assertEqual('development', store.load_conversation('telegram', '200')['active_role'])
            app.handle(incoming(6, '계속 설명해'))
            self.assertEqual(RoleId.DEVELOPMENT, backend.calls[-1][0])
            handoffs = [item['data'] for item in self.events(store) if item['event_type'] == 'CONVERSATION_OWNER_HANDED_OFF']
            self.assertEqual(1, len(handoffs))
            self.assertEqual(('review', 'development', 'user', 'message-5'),
                             (handoffs[0]['from_role'], handoffs[0]['to_role'], handoffs[0]['source'], handoffs[0]['source_message_id']))

    def test_unrequested_target_is_blocked_even_without_repository(self):
        backend = ScriptedBackend(consult(RoleId.IMPROVEMENT))
        with temporary_directory() as directory:
            store, app = build_application(Path(directory), object(), team_backend=backend)
            app.handle(incoming(1, '센티널아, 빌더랑 이야기해서 방향 정해봐'))
            self.assertEqual(1, len(backend.calls))
            self.assertTrue(any(item['event_type'] == 'AGENT_CONSULTATION_BLOCKED' for item in self.events(store)))

    def test_consultation_word_without_a_target_does_not_hide_direct_recipient(self):
        from app.gateway.core.role_routing import RoleResolver
        resolver = RoleResolver()
        for text in ('센티널아 이야기해', '센티널아 논의해', '센티널아 상의해'):
            with self.subTest(text=text):
                intent = resolver.interpret(text, RoleId.DEVELOPMENT)
                self.assertEqual((RoleId.REVIEW,), intent.roles)
                self.assertEqual((), intent.allowed_delegate_roles)

    def test_parser_preserves_mode_in_cache_and_rejects_unknown_mode(self):
        reply = parse_team_conversation_reply(json.dumps({'message': '요청', 'calls': [
            {'to_role': 'development', 'purpose': '질문', 'mode': 'discussion', 'from_role': 'improvement'}]}), RoleId.REVIEW)
        cached = AgentReply.from_dict(json.loads(json.dumps(reply.to_dict())))
        self.assertEqual(('discussion', RoleId.REVIEW), (cached.calls[0].mode, cached.calls[0].from_role))
        with self.assertRaises(InvalidAgentResponse):
            parse_team_conversation_reply('{"message":"요청","calls":[{"to_role":"development","purpose":"질문","mode":"handoff"}]}', RoleId.REVIEW)

    def test_role_prompt_distinguishes_independent_discussion_and_requester_return(self):
        prompt = team_conversation_prompt('역할 지침', '센티널', RoleId.REVIEW, ContextBundle((), (), (), False, 0),
            '방향을 정해줘', call_purpose='consultation_return')
        for token in ('independent', 'discussion', 'consultation_return', 'consultation_final', '요청자', '담당권 인계가 아니다'):
            self.assertIn(token, prompt)

    def test_governed_budget_keeps_room_for_owner_return_and_records_real_attempts(self):
        backend = ScriptedBackend(consult(RoleId.DEVELOPMENT), AgentReply('상담 예산 종료를 종합합니다.'))
        with temporary_directory() as directory:
            store, app = build_application(Path(directory), object(), team_backend=backend)
            budget = BudgetManager(BudgetPolicy(conversation_tokens=32, provider_input_overhead_tokens=0), store)
            app.router.budget = budget
            app.router.team_backend = GovernedTeamConversationBackend(backend, budget, response_reserve_tokens=5)
            output = app.handle(incoming(1, '센티널아, 빌더한테 물어봐'))
            self.assertEqual([RoleId.REVIEW, RoleId.REVIEW], [item[0] for item in backend.calls])
            self.assertEqual('consultation_final', backend.calls[-1][2]['call_purpose'])
            run_id = store.load_conversation('telegram', '200')['run_id']
            result = store.conversation_result('telegram', '200', 'message-1')
            self.assertEqual((RequestOutcome.PARTIAL, 2, 2), (result.outcome, result.attempts, result.successes))
            self.assertEqual(6, store.usage_total(run_id))
            self.assertEqual(0, store.reserved_token_total(run_id))
            self.assertTrue(any('예산 종료' in item.text for item in output))

    def test_some_success_then_failure_is_visible_in_requester_synthesis(self):
        backend = ScriptedBackend(consult(RoleId.DEVELOPMENT), AgentReply('확보한 ABC 의견'),
            consult(RoleId.IMPROVEMENT), RuntimeError('failure'), AgentReply('ABC는 확보했고 피니셔는 미확인입니다.'))
        with temporary_directory() as directory:
            store, app = build_application(Path(directory), object(), team_backend=backend)
            output = app.handle(incoming(1, '센티널아, 빌더랑 피니셔한테 물어봐'))
            context = str(backend.calls[-1][2]['turn_messages'])
            self.assertIn('확보한 ABC 의견', context)
            self.assertIn('MODEL_FAILED', context)
            self.assertIn('미확인', output[-1].text)
            self.assertEqual(RequestOutcome.PARTIAL, store.conversation_result('telegram', '200', 'message-1').outcome)

    def test_child_tools_keep_consultation_identity_and_new_evidence_reaches_owner_and_followup(self):
        backend = ScriptedBackend(consult(RoleId.DEVELOPMENT),
            AgentReply('', repository_tools=(RepositoryToolRequest('read_file', path='src/b.py'),)),
            AgentReply('src/b.py:1의 구현 확인'), AgentReply('src/a.py와 src/b.py 관계를 종합합니다.'), AgentReply('이전 종합을 잇습니다.'))
        with temporary_directory() as directory:
            root = Path(directory)
            store, app = build_application(root, object(), team_backend=backend, repository_reader=Reader(), repository_tools=Tools())
            app.handle(incoming(1, str(root / 'selected-repository')))
            app.handle(incoming(2, '이 프로젝트 사용 승인해'))
            app.handle(incoming(3, '센티널아 프로젝트 설명해. 필요하면 빌더한테 물어봐'))
            finished = [item['data'] for item in self.events(store) if item['event_type'] == 'AGENT_CONSULTATION_FINISHED']
            self.assertEqual(1, len(finished))
            self.assertTrue(any('src/b.py' in ref for ref in finished[0]['evidence_refs']))
            self.assertTrue(finished[0]['untrusted_repository_data'])
            self.assertEqual('b' * 40, finished[0]['head_sha'])
            self.assertIn('src/b.py', str(backend.calls[3][1].evidence))
            app.handle(incoming(4, '그중 두 번째 파일부터 설명해'))
            self.assertIn('src/b.py', str(backend.calls[-1][1].evidence) + str(backend.calls[-1][1].recent_messages))

    def test_terminal_result_message_and_event_roll_back_together(self):
        backend = ScriptedBackend(consult(RoleId.DEVELOPMENT), AgentReply('결과'))
        with temporary_directory() as directory:
            store, app = build_application(Path(directory), object(), team_backend=backend)
            emit = app.router.logger.emit
            def fail_on_finish(run_id, event_type, *args, **kwargs):
                if event_type == 'AGENT_CONSULTATION_FINISHED':
                    raise RuntimeError('terminal persistence interrupted')
                return emit(run_id, event_type, *args, **kwargs)
            with patch.object(app.router.logger, 'emit', side_effect=fail_on_finish):
                with self.assertRaises(RuntimeError):
                    app.router.route_result(incoming(1, '센티널아, 빌더한테 물어봐'))
            run_id = store.load_conversation('telegram', '200')['run_id']
            self.assertFalse(any(item['kind'] == 'consultation_result' for item in store.list_messages(run_id)))
            self.assertFalse(any(item['event_type'] == 'AGENT_CONSULTATION_FINISHED' for item in self.events(store)))

    def test_queue_cancellation_keeps_obtained_result_and_owner_summary_in_terminal_card(self):
        with temporary_directory() as directory:
            root = Path(directory)
            backend = ScriptedBackend(consult(RoleId.DEVELOPMENT), AgentReply('이미 확보한 ABC 의견'),
                consult(RoleId.IMPROVEMENT))
            store, app = build_application(root, object(), team_backend=backend)
            queue = ConversationQueue(store)
            def cancel(*args):
                queue.cancel('telegram', '200')
                return HermesCancelled('cancel')
            backend.steps.append(cancel)
            app.router.set_progress_notifier(lambda: None)
            queue.enqueue(incoming(1, '센티널아, 빌더랑 피니셔한테 물어봐'))
            self.assertTrue(ConversationWorker(store, app.router).run_once())
            self.assertEqual('CANCELLED', queue.status('telegram', '200'))
            self.assertEqual(4, len(backend.calls))
            cards = [item for item in store.deliverable_outbound('telegram') if item['delivery_mode'] == 'edit']
            self.assertIn('이미 확보한 ABC 의견', cards[-1]['text'])
            self.assertIn('[센티널]', cards[-1]['text'])
            statuses = [item['data']['status'] for item in self.events(store) if item['event_type'] == 'AGENT_CONSULTATION_FINISHED']
            self.assertEqual(['COMPLETED', 'CANCELLED'], statuses)

    def test_duplicate_completed_input_does_not_repeat_consultation_or_usage(self):
        backend = ScriptedBackend(consult(RoleId.DEVELOPMENT), AgentReply('의견'), AgentReply('종합'))
        with temporary_directory() as directory:
            store, app = build_application(Path(directory), object(), team_backend=backend)
            budget = BudgetManager(BudgetPolicy(conversation_tokens=1000), store)
            app.router.budget = budget
            app.router.team_backend = GovernedTeamConversationBackend(backend, budget, response_reserve_tokens=5)
            message = incoming(1, '센티널아, 빌더한테 물어봐')
            app.handle(message)
            self.assertEqual((), app.handle(message))
            self.assertEqual(3, len(backend.calls))
            run_id = store.load_conversation('telegram', '200')['run_id']
            self.assertEqual(9, store.usage_total(run_id))
            self.assertEqual(1, sum(item['event_type'] == 'AGENT_CONSULTATION_FINISHED' for item in self.events(store)))

    def test_child_tool_failure_and_cancellation_close_consultation(self):
        for failure in (RepositoryAccessError('tool failed'), RepositoryCancelled('tool cancelled')):
            with self.subTest(failure=type(failure).__name__), temporary_directory() as directory:
                class FailingTools:
                    def execute(self, *args, **kwargs):
                        raise failure
                backend = ScriptedBackend(consult(RoleId.DEVELOPMENT),
                    AgentReply('', repository_tools=(RepositoryToolRequest('read_file', path='src/b.py'),)),
                    AgentReply('조회 실패는 미확인으로 종합합니다.'))
                root = Path(directory)
                store, app = build_application(root, object(), team_backend=backend, repository_reader=Reader(), repository_tools=FailingTools())
                app.handle(incoming(1, str(root / 'selected-repository')))
                app.handle(incoming(2, '이 프로젝트 사용 승인해'))
                message = incoming(3, '센티널아 프로젝트 설명해. 빌더한테 물어봐')
                if isinstance(failure, RepositoryCancelled):
                    with self.assertRaises(ConversationCancelled):
                        app.router.route_result(message)
                    self.assertEqual(2, len(backend.calls))
                    expected = 'CANCELLED'
                else:
                    outputs = app.handle(message)
                    self.assertIn('[센티널]', outputs[-1].text)
                    self.assertEqual('consultation_final', backend.calls[-1][2]['call_purpose'])
                    expected = 'FAILED'
                statuses = [item['data']['status'] for item in self.events(store) if item['event_type'] == 'AGENT_CONSULTATION_FINISHED']
                self.assertEqual([expected], statuses)

    def test_child_stubborn_tool_request_at_final_boundary_returns_partial_to_requester(self):
        tool_reply = AgentReply('', repository_tools=(RepositoryToolRequest('read_file', path='src/b.py'),))
        backend = ScriptedBackend(consult(RoleId.DEVELOPMENT), tool_reply, tool_reply, tool_reply,
            AgentReply('근거가 늘지 않아 현재 파일까지만 종합했습니다.'))
        with temporary_directory() as directory:
            root = Path(directory)
            store, app = build_application(root, object(), team_backend=backend, repository_reader=Reader(), repository_tools=Tools())
            app.handle(incoming(1, str(root / 'selected-repository')))
            app.handle(incoming(2, '이 프로젝트 사용 승인해'))
            outputs = app.handle(incoming(3, '센티널아 프로젝트 설명해. 빌더한테 물어봐'))
            self.assertEqual([RoleId.REVIEW, RoleId.DEVELOPMENT, RoleId.DEVELOPMENT, RoleId.DEVELOPMENT, RoleId.REVIEW],
                             [item[0] for item in backend.calls])
            self.assertEqual('consultation_final', backend.calls[-1][2]['call_purpose'])
            self.assertIn('NO_PROGRESS', outputs[-1].text)
            self.assertNotIn('[빌더]\n', '\n'.join(item.text for item in outputs))
            self.assertEqual(RequestOutcome.PARTIAL, store.conversation_result('telegram', '200', 'message-3').outcome)


if __name__ == '__main__':
    unittest.main()
