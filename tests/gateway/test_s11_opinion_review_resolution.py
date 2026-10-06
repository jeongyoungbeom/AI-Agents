import unittest
from pathlib import Path

from app.contracts import RoleId
from app.gateway.core import AgentReply
from app.gateway.core.role_routing import ROLE_ORDER, RoleResolver
from tests.gateway.support import build_application, temporary_directory
from tests.gateway.test_conversation import incoming
from tests.gateway.test_s07_collaboration import ScriptedBackend, consult


class S11OpinionReviewResolutionTests(unittest.TestCase):
    def test_description_suffixes_ignore_spacing_without_granting_authority(self):
        resolver = RoleResolver()
        for whitespace in ('', ' ', '  ', '\t', '\n'):
            for ending in ('라는 문장을 설명해줘', '라고 말하면 무슨 뜻이야',
                           '란 말이 무슨 뜻이야', '인지 알려줘', '는지 설명해줘'):
                text = f'센티널아 빌더 의견을 받아줘{whitespace}{ending}'
                with self.subTest(text=text):
                    intent = resolver.interpret(text, RoleId.DEVELOPMENT)
                    self.assertEqual((RoleId.REVIEW,), intent.roles)
                    self.assertEqual((), intent.allowed_delegate_roles)
                    self.assertEqual(text, intent.question_purpose)
        # 설명하는 구절과 별도의 실제 상담 요청은 구분한다.
        text = '센티널아 빌더 의견을 받아줘 라는 문장을 설명해줘. 피니셔 의견을 들어줘'
        self.assertEqual((RoleId.IMPROVEMENT,), resolver.interpret(text, RoleId.REVIEW).allowed_delegate_roles)

    def test_opinion_denials_override_generic_permission_only_for_the_named_role(self):
        names = {'development': '개발자', 'review': '검토자', 'improvement': '보완자'}
        for resolver in (RoleResolver(), RoleResolver(names)):
            owner = resolver.names[RoleId.REVIEW]
            for target in (RoleId.DEVELOPMENT, RoleId.IMPROVEMENT):
                name = resolver.names[target]
                for denial in ('의견은 받지 마', '의견을 듣지 말아줘',
                               '의견도 받아주지 마', '의견은 들어 주지 마',
                               '한테는 의견을 받지 마'):
                    with self.subTest(names=resolver.names, target=target, denial=denial):
                        text = f'{owner}아 다른 에이전트 의견을 받아줘. {name}{"" if denial.startswith("한테") else " "}{denial}'
                        intent = resolver.interpret(text, RoleId.REVIEW)
                        self.assertEqual(tuple(role for role in ROLE_ORDER if role != target), intent.allowed_delegate_roles)
                        self.assertEqual((RoleId.REVIEW,), intent.roles)

    def _assert_consultation(self, text, target, allowed, repository):
        with temporary_directory() as directory:
            backend = ScriptedBackend(consult(target), AgentReply('동료 의견'), AgentReply('담당 종합'))
            store, app = build_application(Path(directory), object(), team_backend=backend)
            if repository:
                context = {
                    'source': 'approved_committed_git_snapshot',
                    'untrusted_repository_data': True,
                    'identity_hash': 'a' * 64, 'head_sha': 'b' * 40,
                    'tree': ['feature.py'], 'documents': [],
                }
                app.router._repository_context_for_chat = lambda *_args, **_kwargs: (context, '')
            output = app.handle(incoming(1, text))
            binding = store.load_conversation('telegram', '200')
            events = store.list_events(binding['run_id'])
            expected = [RoleId.REVIEW, target, RoleId.REVIEW] if allowed else [RoleId.REVIEW]
            self.assertEqual(expected, [call[0] for call in backend.calls])
            requested = [event for event in events if event['event_type'] == 'AGENT_CONSULTATION_REQUESTED']
            self.assertEqual(int(allowed), len(requested))
            blocked_type = 'REPOSITORY_CONTEXT_CALL_BLOCKED' if repository else 'AGENT_CONSULTATION_BLOCKED'
            self.assertEqual(int(not allowed), sum(event['event_type'] == blocked_type for event in events))
            self.assertEqual('review', binding['active_role'])
            self.assertIn('[센티널]', output[-1].text)
            result = store.conversation_result('telegram', '200', 'message-1')
            self.assertEqual((len(expected), len(expected)), (result.attempts, result.successes))
            self.assertEqual('success', result.outcome.value)
            if allowed and repository:
                self.assertTrue(backend.calls[1][1].repository_context['untrusted_repository_data'])
                self.assertEqual(1, sum(event['event_type'] == 'REPOSITORY_CONTEXT_CALL_GUARDED' for event in events))

    def test_spaced_descriptions_block_model_suggested_calls_in_both_contexts(self):
        for repository in (False, True):
            for text in ('센티널아 빌더 의견을 받아줘 라는 문장을 설명해줘',
                         '센티널아 다른 에이전트 의견을 받아주세요  라고 말하면 무슨 뜻이야'):
                with self.subTest(repository=repository, text=text):
                    self._assert_consultation(text, RoleId.DEVELOPMENT, False, repository)

    def test_named_denial_blocks_builder_and_keeps_finisher_and_owner_in_both_contexts(self):
        text = '센티널아 다른 에이전트 의견을 받아줘. 빌더 의견은 받지 마'
        for repository in (False, True):
            for target, allowed in ((RoleId.DEVELOPMENT, False), (RoleId.IMPROVEMENT, True)):
                with self.subTest(repository=repository, target=target):
                    self._assert_consultation(text, target, allowed, repository)

    def test_positive_and_quoted_denial_requests_keep_consultation_and_return(self):
        for repository in (False, True):
            for text in ('센티널아 빌더 의견을 받아주세요',
                         '센티널아 빌더 의견도 들어줘',
                         '센티널아 다른 에이전트 의견을 받아줘. "빌더 의견은 받지 마"라는 문장을 설명해줘'):
                with self.subTest(repository=repository, text=text):
                    self._assert_consultation(text, RoleId.DEVELOPMENT, True, repository)


if __name__ == '__main__':
    unittest.main()
