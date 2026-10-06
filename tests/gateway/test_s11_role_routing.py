import unittest
from pathlib import Path

from app.contracts import RoleId
from app.gateway.core.role_routing import ROLE_ORDER, RoleResolver
from tests.gateway.support import build_application, temporary_directory
from tests.gateway.test_conversation_foundation import incoming
from tests.gateway.test_team_conversation import StaticTeamBackend


class S11RoleRoutingTests(unittest.TestCase):
    def test_unnamed_each_role_response_request_addresses_the_whole_team(self):
        for text in ('각자 맡은 일만 한 번 알려줘', '각각 맡은 역할만 설명해줘'):
            with self.subTest(text=text):
                intent = RoleResolver().interpret(text, RoleId.REVIEW)
                self.assertEqual(ROLE_ORDER, intent.roles)
                self.assertTrue(intent.group_call)
                self.assertEqual((), intent.allowed_delegate_roles)

    def test_each_role_followup_calls_each_role_once_and_keeps_the_owner(self):
        with temporary_directory() as directory:
            team = StaticTeamBackend()
            store, app = build_application(Path(directory), object(), team_backend=team)
            app.handle(incoming(1, '센티널아 안녕'))
            app.handle(incoming(2, '셋 다 자기 역할만 짧게 소개해줘'))
            output = app.handle(incoming(3, '각자 맡은 일만 한 번 알려줘'))
            self.assertEqual(list(ROLE_ORDER), [call[0] for call in team.calls[-3:]])
            self.assertEqual(7, len(team.calls))
            self.assertEqual(3, len(output))
            self.assertEqual('review', store.load_conversation('telegram', '200')['active_role'])
            self.assertEqual(3, output[0].metadata['request_result']['attempts'])
            self.assertEqual(3, output[0].metadata['request_result']['successes'])

    def test_plural_response_does_not_expand_named_or_single_recipients(self):
        resolver = RoleResolver()
        intent = resolver.interpret('빌더랑 센티널 각자 맡은 일을 알려줘', RoleId.IMPROVEMENT)
        self.assertEqual((RoleId.DEVELOPMENT, RoleId.REVIEW), intent.roles)
        self.assertFalse(intent.group_call)
        intent = resolver.interpret('각자 맡은 일 대신 센티널만 답해', RoleId.DEVELOPMENT)
        self.assertEqual((RoleId.REVIEW,), intent.roles)
        self.assertFalse(intent.group_call)

    def test_negation_condition_and_word_explanation_do_not_call_the_team(self):
        for text in ('각자 설명하지 마', '각각 답하지 마',
                     '필요하면 각자 소개해줘', '각자라는 말이 무슨 뜻인지 설명해줘',
                     '각자 라는 단어의 뜻을 알려줘', '"각자 소개해줘"라는 문장을 설명해줘'):
            with self.subTest(text=text):
                intent = RoleResolver().interpret(text, RoleId.REVIEW)
                self.assertEqual((RoleId.REVIEW,), intent.roles)
                self.assertFalse(intent.group_call)

    def test_plural_response_uses_configured_role_names(self):
        resolver = RoleResolver({'development': '개발자', 'review': '검토자', 'improvement': '보완자'})
        self.assertEqual(ROLE_ORDER, resolver.interpret('각자 역할을 알려줘', RoleId.REVIEW).roles)
        self.assertEqual((RoleId.REVIEW,), resolver.interpret('검토자만 답해', RoleId.DEVELOPMENT).roles)


if __name__ == '__main__':
    unittest.main()
