import unittest
from dataclasses import replace
from pathlib import Path

from app.contracts import RoleId
from app.gateway.core import AgentReply
from app.gateway.core.role_routing import RoleResolver
from app.gateway.repository_analysis_worker import RepositoryAnalysisQueue
from app.services.logging.redaction import SecretRedactor
from app.services.message_intent import TaskIntent, is_full_repository_audit_request
from app.services.repository import build_repository_analysis_plan, SafeRepositoryReader
from tests.gateway.support import build_application, temporary_directory
from tests.gateway.test_conversation_foundation import incoming
from tests.gateway.test_s11_a09_full_audit import ManifestReader, REQUESTS
from tests.gateway.test_team_conversation import StaticTeamBackend


NON_READING_REQUESTS = (
    '전체 감사는 금지야',
    'full 감사는 금지야',
    '전체 감사 계획을 설명해줘',
    'full 감사 전에 필요한 준비를 설명해줘',
    '전체 감사를 제안하면 어떤 일이 생겨?',
)


class ObservedReader(ManifestReader):
    def __init__(self):
        self.redactor = SecretRedactor()
        self.manifest_calls = 0
        self.inspect_calls = 0

    def should_inspect(self, text):
        return SafeRepositoryReader.should_inspect(text)

    def pinned_manifest(self, *args, **kwargs):
        self.manifest_calls += 1
        return super().pinned_manifest(*args, **kwargs)

    def inspect(self, *args, **kwargs):
        self.inspect_calls += 1
        raise AssertionError('금지·설명·가정 질문은 저장소를 조회하지 않는다')


class S11A09ReviewResolutionTests(unittest.TestCase):
    def test_review_expressions_keep_answer_and_read_denial_in_the_shared_contract(self):
        variants = ('전체 감사를 안 해도 돼', '전체 감사는 하지 말아 줘',
                    'full 감사 방법을 알려줘', '전체 감사 전략을 설명해줘')
        for text in NON_READING_REQUESTS + variants:
            with self.subTest(text=text):
                intent = RoleResolver().interpret(text, RoleId.REVIEW)
                self.assertEqual(TaskIntent.ANSWER, intent.task_intent)
                self.assertTrue(intent.read_forbidden)
                self.assertFalse(is_full_repository_audit_request(text))
                self.assertEqual((RoleId.REVIEW,), intent.roles)
                self.assertEqual((), intent.allowed_delegate_roles)

    def assert_chat_without_reading(self, text, *, wrong_model=False):
        with temporary_directory() as directory:
            root = Path(directory)
            team = StaticTeamBackend([AgentReply(
                '요청한 설명입니다',
                task_intent=TaskIntent.ANALYZE_REPOSITORY if wrong_model else TaskIntent.ANSWER,
            )])
            reader = ObservedReader()
            store, app = build_application(root, object(), team_backend=team, repository_reader=reader)
            app.router.repository_analysis_scheduler = RepositoryAnalysisQueue(store)
            app.handle(incoming(1, str(root / 'selected-repository')))
            app.handle(incoming(2, '이 프로젝트 사용 승인해'))
            message = incoming(3, text)
            if wrong_model:
                message = replace(message, metadata={'user_intent': {'read_forbidden': False}})
            output = app.handle(message)
            self.assertEqual((0, 0), (reader.manifest_calls, reader.inspect_calls))
            self.assertIsNone(store.load_pending_repository_audit('telegram', '200', '100'))
            self.assertIsNone(store.repository_analysis_summary('telegram', '200'))
            self.assertEqual(1, len(team.calls))
            self.assertIn('요청한 설명입니다', output[0].text)
            result = output[0].metadata['request_result']
            self.assertEqual(('success', 1, 1), (result['outcome'], result['attempts'], result['successes']))
            self.assertNotEqual('AUDIT_APPROVAL', result['reason'])
            self.assertEqual('free_chat', store.load_conversation('telegram', '200')['mode'])

    def test_review_expressions_answer_without_manifest_pending_or_analysis(self):
        for text in NON_READING_REQUESTS + ('"전체 감사를 제안해"라고 말하면 어떤 일이 생겨?',):
            with self.subTest(text=text):
                self.assert_chat_without_reading(text)

    def test_wrong_model_analysis_and_external_metadata_cannot_override_read_denial(self):
        for text in NON_READING_REQUESTS:
            with self.subTest(text=text):
                self.assert_chat_without_reading(text, wrong_model=True)

    def test_review_expressions_block_the_detailed_repository_tool_boundary(self):
        class Tools:
            def execute(self, *args, **kwargs):
                raise AssertionError('읽기 금지 요청의 상세 도구를 실행하면 안 된다')

        for text in NON_READING_REQUESTS:
            with self.subTest(text=text), temporary_directory() as directory:
                store, app = build_application(Path(directory), object(), repository_tools=Tools())
                message = incoming(1, text)
                app.handle(message)
                state = store.load_run(store.load_conversation('telegram', '200')['run_id'])
                context, notice = app.router._repository_tool_context_for_chat(
                    message, state, RoleId.REVIEW, {'head_sha': 'b' * 40}, (), tool_round=1,
                )
                self.assertEqual({}, context)
                self.assertIn('읽기 금지', notice)

    def test_positive_proposals_and_read_before_explanation_preserve_full_plan(self):
        manifest = ManifestReader().pinned_manifest()
        for text in REQUESTS + (
            '전체 코드를 읽고 테스트 계획을 설명해줘',
            '수정하지 말고 전체 코드를 읽고 어떻게 테스트하면 좋을지 알려줘.',
        ):
            with self.subTest(text=text):
                intent = RoleResolver().interpret(text, RoleId.REVIEW)
                self.assertEqual(TaskIntent.ANALYZE_REPOSITORY, intent.task_intent)
                self.assertFalse(intent.read_forbidden)
                self.assertTrue(is_full_repository_audit_request(text))
                plan = build_repository_analysis_plan(manifest, text, max_files_per_batch=3, max_file_bytes=49152)
                self.assertEqual('full', plan['mode'])
                self.assertEqual(131, sum(item['selected'] for item in plan['files']))
                self.assertGreater(len(plan['batches']), 40)
