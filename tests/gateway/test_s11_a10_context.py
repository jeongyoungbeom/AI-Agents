import unittest
from pathlib import Path

from app.contracts import RoleId, RunState
from app.orchestrator import RunStateMachine
from app.services.repository import RepositoryAnalysisRequest, RepositoryAnalysisStatus
from tests.gateway.support import (
    TEST_REPOSITORY_HEAD, TEST_REPOSITORY_IDENTITY, build_application,
    temporary_directory,
)
from tests.gateway.test_s03_context import Team, incoming


class S11A10ContextTests(unittest.TestCase):
    def setup_analysis(self, root, *, active_task=True):
        team = Team()
        store, app = build_application(root, object(), team_backend=team)
        app.handle(incoming(1, 'D:\\projects\\sample'))
        app.handle(incoming(2, '이 프로젝트 사용 승인해'))
        session = store.load_conversation_session('telegram', '200')
        if active_task:
            store.create_run(RunState('RUN-A10-ACTIVE', repository=str(root),
                repository_identity=TEST_REPOSITORY_IDENTITY,
                repository_head_sha=TEST_REPOSITORY_HEAD))
            store.set_conversation_task('telegram', '200', 'RUN-A10-ACTIVE')
        request = RepositoryAnalysisRequest.create(
            channel='telegram', conversation_id='200', user_id='100',
            source_message_id='analysis-input', role_id=RoleId.REVIEW.value,
            request_text='전체 분석', repository_path=str(root),
            repository_identity=TEST_REPOSITORY_IDENTITY,
            commit_sha=TEST_REPOSITORY_HEAD, branch='main')
        RunStateMachine(store).create_run(request.analysis_id)
        store.create_repository_analysis(request)
        store.claim_next_repository_analysis('worker')
        store.add_repository_analysis_evidence(request.analysis_id,
            commit_sha=TEST_REPOSITORY_HEAD, path='feature.py', start_line=1,
            end_line=2, phase='CORE', kind='source_read', summary='첫 번째 산술 문제')
        store.finish_repository_analysis_with_response(request.analysis_id, 'worker',
            RepositoryAnalysisStatus.PARTIAL_COMPLETED,
            '첫 번째 문제: feature.py:1-2의 add(2, 3)은 -1을 반환하며 기대값은 5다. 미처리 127개.',
            reason='USER_STOPPED')
        return store, app, session, request, team

    def test_active_task_followup_reads_partial_analysis_after_restart(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store, app, session, request, team = self.setup_analysis(root)
            self.assertNotEqual(session.session_run_id, store.load_conversation('telegram', '200')['run_id'])
            reopened, restarted = build_application(root, object(), team_backend=team)
            for n, text in [(3, '그중 첫 번째부터 설명해줘'),
                            (4, '긴 분석에서 처음 찾은 문제를 이어서 설명해')]:
                restarted.handle(incoming(n, text))
                results = [m for m in team.contexts[-1].recent_messages if m['kind'] == 'analysis_result']
                self.assertEqual(1, len(results))
                data = results[0]['data']
                self.assertEqual(request.analysis_id, data['analysis_id'])
                self.assertEqual(TEST_REPOSITORY_HEAD, data['head_sha'])
                self.assertTrue(data['untrusted_repository_data'])
                self.assertEqual('PARTIAL_COMPLETED', data['status'])
                self.assertEqual('USER_STOPPED', data['reason'])
                self.assertEqual('feature.py', data['evidence_refs'][0]['path'])
                self.assertIn('기대값은 5', results[0]['content'])
                self.assertLessEqual(team.contexts[-1].characters, restarted.router.context.policy.max_characters)
            self.assertEqual(1, sum(m['kind']=='analysis_result' for m in reopened.list_messages(session.session_run_id)))
            self.assertFalse(any(m['kind']=='analysis_result' for m in reopened.list_messages('RUN-A10-ACTIVE')))

    def test_parent_result_keeps_snapshot_and_project_boundaries(self):
        for identity in (TEST_REPOSITORY_IDENTITY, 'c'*64):
            with self.subTest(identity=identity), temporary_directory() as directory:
                store, app, session, request, team = self.setup_analysis(Path(directory))
                if identity == TEST_REPOSITORY_IDENTITY:
                    store.refresh_project_head('telegram', '200', '100', identity, 'd'*40)
                else:
                    store.set_current_project('telegram', '200', '100', 'D:\\projects\\other',
                        repository_identity=identity, head_sha='d'*40)
                app.handle(incoming(3, '긴 분석에서 처음 찾은 문제를 이어서 설명해'))
                results=[m for m in team.contexts[-1].recent_messages if m['kind']=='analysis_result']
                if identity == TEST_REPOSITORY_IDENTITY:
                    self.assertEqual(1,len(results))
                    self.assertEqual(TEST_REPOSITORY_HEAD,results[0]['data']['head_sha'])
                    self.assertTrue(results[0]['data']['stale_repository_snapshot'])
                    self.assertTrue(results[0]['data']['untrusted_repository_data'])
                else:
                    self.assertEqual([],results)

    def test_same_session_does_not_duplicate_result(self):
        with temporary_directory() as directory:
            store, app, session, request, team = self.setup_analysis(Path(directory), active_task=False)
            app.handle(incoming(3,'그중 첫 번째부터 설명해줘'))
            self.assertEqual(1,sum(m['kind']=='analysis_result' for m in team.contexts[-1].recent_messages))

    def test_active_task_reads_parent_pipeline_result_without_unrelated_chat(self):
        with temporary_directory() as directory:
            store, app, session, request, team = self.setup_analysis(Path(directory))
            app.router.context.add_message(session.session_run_id,'development',
                '완료된 개발 결과: feature.py 산술 검증 통과',kind='pipeline_result',
                data={'run_id':'RUN-A10-PREVIOUS','repository_identity':TEST_REPOSITORY_IDENTITY,
                    'head_sha':TEST_REPOSITORY_HEAD,'untrusted_repository_data':True})
            app.router.context.add_message(session.session_run_id,'user',
                '별도 세션의 일반 메시지',data={'repository_identity':TEST_REPOSITORY_IDENTITY})
            app.handle(incoming(3,'앞 결과를 설명해'))
            context=team.contexts[-1]
            self.assertEqual(1,sum(m['kind']=='pipeline_result' for m in context.recent_messages))
            self.assertFalse(any(m['content']=='별도 세션의 일반 메시지' for m in context.recent_messages))
