import unittest
from pathlib import Path

from app.contracts import TokenUsage
from app.gateway.core import GovernedTeamConversationBackend
from app.services.budget import BudgetManager, BudgetPolicy
from tests.gateway.support import build_application, temporary_directory
from tests.gateway.test_conversation_foundation import incoming


class NoModelExpected:
    def respond_as(self, *_args, **_kwargs):
        raise AssertionError('budget-blocked request reached model')


class S11BudgetDiagnosticsTests(unittest.TestCase):
    def setup_budget(self, root):
        store, application = build_application(root, None)
        binding = application.router._create_binding(incoming(0, 'fixture'))
        manager = BudgetManager(BudgetPolicy(conversation_tokens=400000, per_stage_tokens=300000,
            whole_task_tokens=1200000, completion_reserve_tokens=80000), store)
        application.router.budget = manager
        return store, application, manager, binding['run_id']

    def test_known_overrun_reports_actual_and_reservation_without_reopening_budget(self):
        with temporary_directory() as directory:
            store, _, manager, run_id = self.setup_budget(Path(directory))
            reservation = manager.reserve(run_id, 'stage-001', 'development', 'agent', 48488)
            manager.record_usage(run_id, 'stage-001', 'development', 'agent',
                                 TokenUsage(total_tokens=57680), reservation=reservation)
            before = store.list_events(run_id)
            decision = manager.can_spend(run_id, 'stage-001', 1000)
            self.assertFalse(decision.allowed)
            self.assertIn('57,680', decision.reason)
            self.assertIn('48,488', decision.reason)
            self.assertIn('예약 초과', decision.reason)
            self.assertNotIn('사용량이 미확인', decision.reason)
            self.assertEqual(before, store.list_events(run_id))
            self.assertEqual(57680, store.usage_total(run_id))
            self.assertTrue(store.has_budget_anomaly(run_id))

    def test_unknown_usage_reports_estimate_and_stays_blocked(self):
        with temporary_directory() as directory:
            store, _, manager, run_id = self.setup_budget(Path(directory))
            manager.record_usage(run_id, 'stage-001', 'development', 'agent',
                                 TokenUsage(total_tokens=13987, estimated=True))
            store.append_event(run_id, '2026-10-03T00:00:00+00:00', 'MODEL_USAGE_UNKNOWN', '미확인')
            decision = manager.can_spend(run_id, 'stage-001', 1000)
            self.assertFalse(decision.allowed)
            self.assertIn('사용량이 미확인', decision.reason)
            self.assertIn('실제 0 / 추정 13,987', decision.reason)

    def test_conversation_limit_preserves_results_reports_usage_and_resume_condition(self):
        with temporary_directory() as directory:
            store, app, manager, run_id = self.setup_budget(Path(directory))
            store.append_message(run_id, 'development', 'agent_reply', '확보된 부분 결과')
            manager.record_usage(run_id, 'chat-previous', 'development', 'conversation', TokenUsage(total_tokens=6654))
            manager.policy = BudgetPolicy(conversation_tokens=6655, whole_task_tokens=1000000)
            app.router.team_backend = GovernedTeamConversationBackend(NoModelExpected(), manager)
            messages_before = store.list_messages(run_id)
            outputs = app.handle(incoming(1, '센티널아 확보한 결과를 요약해줘'))
            text = '\n'.join(o.text for o in outputs)
            self.assertIn('6,654', text)
            self.assertIn('확보한 답변', text)
            self.assertIn('예상 비용', text)
            self.assertIn('다시 요청', text)
            self.assertEqual(6654, store.usage_total(run_id))
            self.assertFalse(store.model_calls_for_request(store.model_request_key('telegram', '200', '1')))
            self.assertTrue(all(m in store.list_messages(run_id) for m in messages_before))


if __name__ == '__main__':
    unittest.main()
