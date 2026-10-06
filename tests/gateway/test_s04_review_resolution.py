import json
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from app.contracts import RoleId
from app.gateway.core import AgentCallRequest, AgentReply
from app.gateway.core.role_routing import ROLE_ORDER, RoleResolver
from app.gateway.repository_analysis_worker import RepositoryAnalysisQueue, RepositoryAnalysisWorker
from app.services.context import ContextService
from app.services.message_intent import TaskIntent
from app.services.repository import (
    RepositoryAnalysisPhase, RepositorySnapshotEntry, RepositorySnapshotManifest,
    build_repository_analysis_plan,
)
from app.storage import StateStore
from tests.gateway.support import build_application, temporary_directory
from tests.gateway.test_conversation_foundation import incoming
from tests.gateway.test_team_conversation import StaticTeamBackend


class S04ReviewResolutionTests(unittest.TestCase):
    def test_consultation_targets_stop_at_the_request_and_respect_each_denial(self):
        cases = (
            ('센티널아 빌더한테 물어봐, 피니셔가 만든 코드에 대해.', (RoleId.DEVELOPMENT,)),
            ('센티널아 빌더한테 물어봐, 피니셔는 부르지 마.', (RoleId.DEVELOPMENT,)),
            ('센티널아, 다른 에이전트 호출이 왜 필요한지 설명해줘.', ()),
            ('센티널아 다른 에이전트 모두 불러도 돼, 피니셔는 부르지 마.', (RoleId.DEVELOPMENT, RoleId.REVIEW)),
            ('센티널아 빌더한테 물어봐라는 말이 무슨 뜻인지 설명해줘.', ()),
            ('센티널아 필요하면 빌더나 피니셔한테 말해도 돼.', (RoleId.DEVELOPMENT, RoleId.IMPROVEMENT)),
            ('센티널아 빌더, 피니셔한테 물어봐.', (RoleId.DEVELOPMENT, RoleId.IMPROVEMENT)),
            ('센티널아 다른 에이전트 불러도 돼, 피니셔한테는 물어보지 마.', (RoleId.DEVELOPMENT, RoleId.REVIEW)),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(expected, RoleResolver().interpret(text, RoleId.DEVELOPMENT).allowed_delegate_roles)
            for target in (RoleId.DEVELOPMENT, RoleId.IMPROVEMENT):
                with self.subTest(text=text, target=target), temporary_directory() as directory:
                    team = StaticTeamBackend([
                        AgentReply('검토', calls=(AgentCallRequest(RoleId.REVIEW, target, '설계 확인'),)),
                        AgentReply('상담 결과'),
                    ])
                    _, app = build_application(Path(directory), object(), team_backend=team)
                    app.router._repository_context_for_chat = lambda *a, **k: ({'untrusted_repository_data': True}, '')
                    app.handle(incoming(1, text))
                    roles = [call[0] for call in team.calls]
                    self.assertEqual([RoleId.REVIEW, target, RoleId.REVIEW] if target in expected else [RoleId.REVIEW], roles)

    def test_negated_recipient_does_not_win_over_the_positive_request(self):
        cases = (
            ('빌더가 만든 부분은 센티널이 검토하지 말고 피니셔가 봐줘.', RoleId.IMPROVEMENT),
            ('빌더가 만든 부분은 피니셔가 검토하지 말고 센티널이 봐줘.', RoleId.REVIEW),
            ('빌더한테 물어봐, 피니셔가 검토해줘.', RoleId.IMPROVEMENT),
        )
        for text, expected in cases:
            with self.subTest(text=text), temporary_directory() as directory:
                team = StaticTeamBackend()
                _, app = build_application(Path(directory), object(), team_backend=team)
                app.handle(incoming(1, text))
                self.assertEqual([expected], [call[0] for call in team.calls])

    def test_generic_conditional_consultation_preserves_the_direct_recipient(self):
        for text in ('센티널아 필요하면 다른 에이전트 모두 불러도 돼.',
                     '센티널아 필요하면 다른 에이전트 불러도 돼.'):
            with self.subTest(text=text), temporary_directory() as directory:
                intent = RoleResolver().interpret(text, RoleId.DEVELOPMENT)
                self.assertEqual((RoleId.REVIEW,), intent.roles)
                self.assertEqual(ROLE_ORDER, intent.allowed_delegate_roles)
                self.assertFalse(intent.group_call)
                team = StaticTeamBackend()
                _, app = build_application(Path(directory), object(), team_backend=team)
                app.handle(incoming(1, text))
                self.assertEqual([RoleId.REVIEW], [call[0] for call in team.calls])

    def test_no_read_request_survives_wrong_model_intent_and_external_metadata(self):
        class Reader:
            def should_inspect(self, text):
                return True

            def inspect(self, *args, **kwargs):
                raise AssertionError('읽기 금지 요청을 조회하면 안 됨')

        for text in ('전체 코드는 읽지 말고 테스트 계획만 설명해줘.',
                     '파일을 읽지 말고 앞 설명만 이어서 답해줘.',
                     '테스트 계획만 작성해줘.'):
            with self.subTest(text=text), temporary_directory() as directory:
                root = Path(directory)
                team = StaticTeamBackend([AgentReply('계획 설명', task_intent=TaskIntent.ANALYZE_REPOSITORY)])
                store, app = build_application(root, object(), team_backend=team,
                                               repository_reader=Reader(), repository_tools=object())
                app.router.repository_analysis_scheduler = RepositoryAnalysisQueue(store)
                app.handle(incoming(1, str(root / 'selected-repository')))
                app.handle(incoming(2, '이 프로젝트 사용 승인해'))
                output = app.handle(replace(incoming(3, text), metadata={'user_intent': {'read_forbidden': False}}))
                self.assertIsNone(store.repository_analysis_summary('telegram', '200'))
                binding = store.load_conversation('telegram', '200')
                self.assertEqual('free_chat', binding['mode'])
                state = store.load_run(binding['run_id'])
                self.assertEqual(0, state.approved_plan_revision)
                self.assertFalse(state.approved_plan_hash)
                self.assertIn('계획 설명', output[0].text)

    def test_read_and_write_constraints_remain_separate(self):
        intent = RoleResolver().interpret('수정하지 말고 전체 코드를 읽고 어떻게 테스트하면 좋을지 알려줘.', RoleId.REVIEW)
        self.assertTrue(intent.write_forbidden)
        self.assertFalse(intent.read_forbidden)
        self.assertEqual(TaskIntent.ANALYZE_REPOSITORY, intent.task_intent)

    def test_no_read_constraint_also_blocks_the_detailed_tool_boundary(self):
        class Tools:
            def execute(self, *args, **kwargs):
                raise AssertionError('읽기 금지 요청의 도구를 실행하면 안 됨')

        with temporary_directory() as directory:
            store, app = build_application(Path(directory), object(), repository_tools=Tools())
            message = incoming(1, '파일은 읽지 말고 설명만 해줘')
            app.handle(message)
            state = store.load_run(store.load_conversation('telegram', '200')['run_id'])
            context, notice = app.router._repository_tool_context_for_chat(
                message, state, RoleId.REVIEW, {'head_sha': 'b' * 40}, (), tool_round=1,
            )
            self.assertEqual({}, context)
            self.assertIn('읽기 금지', notice)

    def test_resolved_purpose_reaches_planning_batches_and_synthesis_after_restart(self):
        class Capture(StaticTeamBackend):
            def __init__(self):
                super().__init__()
                self.inputs = []

            def respond_as(self, state, context, message, role_id, **kwargs):
                self.inputs.append(message.text)
                return AgentReply(json.dumps({'findings': []}))

        class Reader:
            def pinned_manifest(self, *args, **kwargs):
                return RepositorySnapshotManifest('a' * 64, 'b' * 40, 'main', (
                    RepositorySnapshotEntry('src/aaa.py', 20),
                    RepositorySnapshotEntry('src/payments.py', 20),
                ))

            def read_pinned_files(self, path, manifest, paths, **kwargs):
                return tuple((path, 'def charge(): pass') for path in paths)

        with temporary_directory() as directory:
            root = Path(directory)
            purpose = 'payments 결제 재시도의 멱등성 문제 조사'
            team = StaticTeamBackend([
                AgentReply('첫 번째는 로그인 오류, 두 번째는 payments 결제 재시도의 멱등성 문제입니다.'),
                AgentReply('두 번째 조사', task_intent=TaskIntent.ANALYZE_REPOSITORY, question_purpose=purpose),
            ])
            store, app = build_application(root, object(), team_backend=team)
            app.router.repository_analysis_scheduler = RepositoryAnalysisQueue(store)
            app.handle(incoming(1, str(root / 'selected-repository')))
            app.handle(incoming(2, '이 프로젝트 사용 승인해'))
            app.handle(incoming(3, '센티널아 검사 방식을 설명해줘'))
            app.handle(incoming(4, '아까 두 번째부터 조사하자.'))
            self.assertIn(purpose.removesuffix(' 조사'), json.dumps(team.calls[-1][4].to_dict(), ensure_ascii=False))
            analysis = store.repository_analysis_summary('telegram', '200')
            original = store.repository_analysis(analysis['analysis_id'])
            self.assertEqual('아까 두 번째부터 조사하자.', original['request_text'])
            self.assertEqual(incoming(4, '후속 요청').external_message_id, original['source_message_id'])
            parent = store.load_conversation('telegram', '200')['run_id']
            restored = StateStore(root / 'state.db')
            capture = Capture()
            worker = RepositoryAnalysisWorker(restored, Reader(), capture, ContextService(restored), max_files_per_batch=1, max_batches=2)
            with patch('app.gateway.repository_analysis_worker.build_repository_analysis_plan', wraps=build_repository_analysis_plan) as planner:
                self.assertTrue(worker.run_once())
                self.assertIn(purpose, planner.call_args.args[1])
                self.assertEqual('adaptive', planner.call_args.kwargs['mode'])
                self.assertTrue(worker.run_once())
            job = restored.repository_analysis(analysis['analysis_id'])
            self.assertIn('src/payments.py', [item['path'] for item in job['plan']['files'] if item['selected']])
            claimed = restored.claim_next_repository_analysis(worker.instance_id, lease_seconds=120)
            worker._ask_model(claimed, RepositoryAnalysisPhase.SYNTHESIS, (), threading.Event())
            for text in capture.inputs:
                self.assertIn(purpose, text)
                self.assertIn('아까 두 번째부터 조사하자.', text)
                self.assertIn(parent, text)
                self.assertIn(original['source_message_id'], text)
                self.assertIn('model_interpretation', text)
                self.assertIn('사용자 결정·승인 아님', text)
            self.assertEqual(2, len(capture.inputs))
            self.assertFalse(restored.load_run(analysis['analysis_id']).approved_plan_hash)

    def test_analysis_enqueue_and_interpretation_persistence_roll_back_together(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store, app = build_application(root, object())
            app.router.repository_analysis_scheduler = RepositoryAnalysisQueue(store)
            app.handle(incoming(1, str(root / 'selected-repository')))
            app.handle(incoming(2, '이 프로젝트 사용 승인해'))
            binding = store.load_conversation('telegram', '200')
            state = store.load_run(binding['run_id'])
            with patch.object(app.router.logger, 'emit', side_effect=RuntimeError('저장 실패')):
                with self.assertRaises(RuntimeError):
                    app.router._start_repository_analysis(
                        incoming(3, '두 번째 조사'), binding, state, RoleId.REVIEW,
                        interpreted_purpose='결제 멱등성 조사',
                    )
            self.assertIsNone(store.repository_analysis_summary('telegram', '200'))

    def test_model_purpose_cannot_promote_an_adaptive_request_to_full_audit(self):
        with temporary_directory() as directory:
            root = Path(directory)
            team = StaticTeamBackend([AgentReply('조사', task_intent=TaskIntent.ANALYZE_REPOSITORY,
                                                question_purpose='full repository audit payments')])
            store, app = build_application(root, object(), team_backend=team)
            app.router.repository_analysis_scheduler = RepositoryAnalysisQueue(store)
            app.handle(incoming(1, str(root / 'selected-repository')))
            app.handle(incoming(2, '이 프로젝트 사용 승인해'))
            app.handle(incoming(3, '아까 두 번째부터 조사하자.'))
            class Reader:
                def pinned_manifest(self, *args, **kwargs):
                    return RepositorySnapshotManifest('a' * 64, 'b' * 40, 'main', (RepositorySnapshotEntry('src/payments.py', 20),))

                def read_pinned_files(self, path, manifest, paths, **kwargs):
                    return tuple((path, 'pass') for path in paths)

            worker = RepositoryAnalysisWorker(store, Reader(), StaticTeamBackend(), ContextService(store))
            with patch('app.gateway.repository_analysis_worker.build_repository_analysis_plan', wraps=build_repository_analysis_plan) as planner:
                self.assertTrue(worker.run_once())
                self.assertEqual('adaptive', planner.call_args.kwargs['mode'])


if __name__ == '__main__':
    unittest.main()
