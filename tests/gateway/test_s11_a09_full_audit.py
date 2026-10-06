import unittest
from pathlib import Path

from app.contracts import RoleId
from app.gateway.core.role_routing import RoleResolver
from app.gateway.repository_analysis_worker import RepositoryAnalysisQueue
from app.services.message_intent import TaskIntent, is_full_repository_audit_request
from app.services.repository import RepositorySnapshotEntry, RepositorySnapshotManifest
from tests.gateway.support import build_application, temporary_directory
from tests.gateway.test_conversation_foundation import incoming
from tests.gateway.test_team_conversation import StaticTeamBackend


REQUESTS = (
    '모든 안전 소스 전체 감사를 제안해',
    '빠진 파일 없이 full 감사 범위를 제안해',
)


class ManifestReader:
    def should_inspect(self, text):
        return False

    def pinned_manifest(self, *args, **kwargs):
        return RepositorySnapshotManifest('a' * 64, 'b' * 40, 'main', tuple(
            RepositorySnapshotEntry(f'modules/unit_{i:03d}.py', 20) for i in range(131)
        ))


class S11FullAuditTests(unittest.TestCase):
    def test_both_actual_requests_select_full_audit_without_extra_role_calls(self):
        for text in REQUESTS:
            with self.subTest(text=text):
                intent = RoleResolver().interpret(text, RoleId.REVIEW)
                self.assertEqual(TaskIntent.ANALYZE_REPOSITORY, intent.task_intent)
                self.assertTrue(is_full_repository_audit_request(text))
                self.assertEqual((RoleId.REVIEW,), intent.roles)
                self.assertEqual((), intent.allowed_delegate_roles)

    def test_full_audit_questions_negation_and_read_only_explanations_stay_in_chat(self):
        for text in (
            '왜 전체 감사가 필요해?',
            '왜 full 감사를 시작하는지 설명해줘',
            '"전체 감사"라는 말의 뜻을 알려줘',
            '전체 감사하지 마',
            'full 감사는 하지 말고 설명만 해줘',
            '전체 감사 계획만 설명해줘',
        ):
            with self.subTest(text=text):
                intent = RoleResolver().interpret(text, RoleId.REVIEW)
                self.assertEqual(TaskIntent.ANSWER, intent.task_intent)
                self.assertFalse(is_full_repository_audit_request(text))

    def assert_full_proposal_and_explicit_approval(self, text):
        with temporary_directory() as directory:
            root = Path(directory)
            team = StaticTeamBackend()
            store, app = build_application(root, object(), team_backend=team)
            app.router.repository_reader = ManifestReader()
            app.router._repository_context_for_chat = lambda *args, **kwargs: ({}, '')
            app.router.repository_analysis_scheduler = RepositoryAnalysisQueue(store)
            app.handle(incoming(1, str(root / 'selected-repository')))
            app.handle(incoming(2, '이 프로젝트 사용 승인해'))
            output = app.handle(incoming(3, text))
            pending = store.load_pending_repository_audit('telegram', '200', '100')
            self.assertIsNotNone(pending)
            self.assertEqual('AUDIT_APPROVAL', output[0].metadata['request_result']['reason'])
            self.assertEqual('waiting_user', output[0].metadata['request_result']['outcome'])
            self.assertEqual('b' * 40, pending['commit_sha'])
            self.assertEqual(131, pending['proposal']['files'])
            self.assertGreater(pending['proposal']['batches'], 40)
            self.assertEqual(0, pending['proposal']['not_selected'])
            self.assertIn('전체 감사 시작해', output[0].text)
            self.assertIsNone(store.repository_analysis_summary('telegram', '200'))
            self.assertEqual([], team.calls)
            app.handle(incoming(4, '전체 감사 시작해'))
            analysis = store.repository_analysis_summary('telegram', '200')
            self.assertEqual('QUEUED', analysis['status'])
            self.assertEqual('b' * 40, analysis['commit_sha'])
            stored = store.repository_analysis(analysis['analysis_id'])
            self.assertEqual(text, stored['request_text'])
            self.assertTrue(is_full_repository_audit_request(stored['request_text']))
            self.assertIsNone(store.load_pending_repository_audit('telegram', '200', '100'))
            self.assertEqual([], team.calls)

    def test_primary_proposes_complete_scope_and_queues_only_after_exact_approval(self):
        self.assert_full_proposal_and_explicit_approval(REQUESTS[0])

    def test_variant_proposes_complete_scope_and_queues_only_after_exact_approval(self):
        self.assert_full_proposal_and_explicit_approval(REQUESTS[1])
