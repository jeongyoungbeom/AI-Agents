import unittest
from pathlib import Path

from app.contracts import RoleId
from app.gateway.core.role_routing import ROLE_ORDER, RoleResolver
from tests.gateway.support import build_application, temporary_directory
from tests.gateway.test_conversation_foundation import incoming
from tests.gateway.test_team_conversation import StaticTeamBackend


class S11ReviewResolutionTests(unittest.TestCase):
    def test_negated_plural_responses_preserve_each_active_role(self):
        texts = (
            '각자 맡은 일은 알려주지 마',
            '각자 역할 설명은 금지야',
            '각각 맡은 역할을 알려주지 말아줘',
            '각자 역할을 알려 주지 마세요',
            '각자 역할을 알려주지 않아도 돼',
            '각자 역할을 알려주면 안 돼',
            '셋 다 역할 설명은 금지야',
            '모두 역할을 알려주지 마',
        )
        for role in ROLE_ORDER:
            for text in texts:
                with self.subTest(role=role, text=text):
                    intent = RoleResolver().interpret(text, role)
                    self.assertEqual((role,), intent.roles)
                    self.assertFalse(intent.group_call)
                    self.assertFalse(intent.explicit)
                    self.assertEqual((), intent.allowed_delegate_roles)

    def test_negative_plural_gateway_requests_call_and_reserve_only_the_owner(self):
        for text in ('각자 맡은 일은 알려주지 마', '각자 역할 설명은 금지야'):
            with self.subTest(text=text), temporary_directory() as directory:
                team = StaticTeamBackend()
                store, app = build_application(Path(directory), object(), team_backend=team)
                app.handle(incoming(1, '센티널아 안녕'))
                offset = len(team.calls)
                outputs = app.handle(incoming(2, text))
                self.assertEqual([RoleId.REVIEW], [call[0] for call in team.calls[offset:]])
                self.assertEqual(1, team.preflights[-1])
                self.assertEqual(1, len(outputs))
                self.assertEqual('review', store.load_conversation('telegram', '200')['active_role'])
                result = outputs[0].metadata['request_result']
                self.assertEqual('success', result['outcome'])
                self.assertEqual(1, result['attempts'])
                self.assertEqual(1, result['successes'])

    def test_negative_plural_request_keeps_explicit_conditional_consultation_permission(self):
        for text in ('각자 맡은 일은 알려주지 마. 필요하면 빌더한테 물어봐',
                     '각자 역할 설명은 금지야. 필요하면 빌더한테 물어봐'):
            with self.subTest(text=text):
                intent = RoleResolver().interpret(text, RoleId.REVIEW)
                self.assertEqual((RoleId.REVIEW,), intent.roles)
                self.assertFalse(intent.group_call)
                self.assertEqual((RoleId.DEVELOPMENT,), intent.allowed_delegate_roles)


if __name__ == '__main__':
    unittest.main()
