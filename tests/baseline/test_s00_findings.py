"""S00 baseline: passing reproduces a defect; it does not mean the defect is fixed."""
from pathlib import Path
import json
import unittest

from app.contracts import RoleId, TokenUsage, RunPhase
from app.gateway.core import AgentReply
from app.gateway.core.conversation import DialogueRouter
from app.services.context import ContextService
from app.services.repository import SafeRepositoryReader, RepositoryAnalysisRequest, RepositoryAnalysisStatus, RepositorySnapshotManifest, RepositorySnapshotEntry, build_repository_analysis_plan, is_long_repository_analysis_request
from app.services.hermes import HermesResult
from app.orchestrator import RunStateMachine
from app.storage import StateStore
from tests.gateway.support import temporary_directory, build_application, TEST_REPOSITORY_IDENTITY, future_expiry
from tests.gateway.test_conversation_foundation import incoming
from tests.pipeline.support import build_pipeline, create_repository, FakeRoleRunner


class AuditReproductions(unittest.TestCase):
    def test_followup_excludes_previous_repository_answer(self):
        class Team:
            def preflight(self,*args): pass
            def respond_as(self,*args,**kwargs): return AgentReply('ok')
        with temporary_directory() as d:
            store, app = build_application(Path(d), None, team_backend=Team())
            app.handle(incoming(1,'안녕'))
            binding=store.load_conversation('telegram','200')
            store.set_current_project('telegram','200','100',str(Path(d)/'repo'), repository_identity=TEST_REPOSITORY_IDENTITY, head_sha='b'*40,approved=True,approval_expires_at=future_expiry())
            state=store.load_run(binding['run_id'])
            app.router.context.add_message(state.run_id,'review','ARL은 신뢰성 실험 워크벤치입니다.',data={'repository_identity':TEST_REPOSITORY_IDENTITY,'untrusted_repository_data':True})
            question='그럼 지금은 테스트를 해볼때라는 것인가?'
            self.assertFalse(SafeRepositoryReader.should_inspect(question))
            context=app.router._conversation_context(incoming(2,question),state,RoleId.REVIEW)
            self.assertFalse(any('신뢰성 실험' in x['content'] for x in context.recent_messages))
            print('CONFIRMED: followup loses the preceding project answer')

    def test_named_delegation_is_authorized_after_s04(self):
        text='센티널아 테스트를 정리해줘. 니가 못할 것 같으면 빌더나 피니셔한테 말해도돼'
        self.assertTrue(DialogueRouter._user_authorized_repository_consultation(incoming(1,text)))
        self.assertTrue(DialogueRouter._user_authorized_repository_consultation(incoming(2,'다른 에이전트에게 의견을 물어봐')))
        print('S04 REGRESSION: named conditional consultation is authorized')

    def test_natural_requests_route_correctly_after_s04(self):
        self.assertFalse(DialogueRouter._is_work_intent('코드는 수정하지 말고 테스트 계획만 작성해줘'))
        self.assertTrue(is_long_repository_analysis_request('전체 코드를 읽고 어떻게 테스트하면 좋을지 알려줘'))
        print('S04 REGRESSION: read-only planning stays in chat; broad reading retains its purpose question')

    def test_error_notice_is_marked_failed(self):
        class Team:
            def preflight(self,*args): pass
            def respond_as(self,*args,**kwargs): raise RuntimeError('fixture model failure')
        with temporary_directory() as d:
            store, app=build_application(Path(d),None,team_backend=Team())
            app.router.set_progress_notifier(lambda:None)
            app.handle(incoming(1,'안녕'))
            records=store.deliverable_outbound('telegram')
            self.assertTrue(any('· 실패]' in x['text'] for x in records))
            self.assertFalse(any('4/4' in x['text'] for x in records))
            self.assertTrue(any('문제가 생겼습니다' in x['text'] for x in records))
            print('REGRESSION: failed response produces a failed progress card')

    def test_full_audit_keeps_larger_files_after_s06(self):
        entries=(RepositorySnapshotEntry('app/core.py',12000),RepositorySnapshotEntry('app/small.py',100))
        cap=(24000-1024)//2
        plan=build_repository_analysis_plan(RepositorySnapshotManifest('a'*64,'b'*40,'main',entries),'전체 코드 분석해줘',max_files_per_batch=3,max_file_bytes=49152,max_batch_bytes=cap,max_context_bytes=cap)
        core=next(x for x in plan['files'] if x['path']=='app/core.py')
        self.assertTrue(core['selected'])
        self.assertEqual(core['exclude_reason'],'')
        print(f'S06 REGRESSION: full audit retains a 12000-byte core file for ranged reads (chunk cap {cap})')

    def test_long_result_is_not_saved_to_conversation(self):
        with temporary_directory() as d:
            store=StateStore(Path(d)/'state.db')
            machine=RunStateMachine(store)
            machine.create_run('session')
            store.create_conversation_session('telegram','200','100','session','review')
            req=RepositoryAnalysisRequest.create(channel='telegram',conversation_id='200',user_id='100',source_message_id='1',role_id='review',request_text='프로젝트 전반 분석해줘',repository_path=d,repository_identity='a'*64,commit_sha='b'*40,branch='main')
            machine.create_run(req.analysis_id)
            store.create_repository_analysis(req)
            store.claim_next_repository_analysis('worker')
            store.queue_repository_analysis_progress(req.analysis_id,'worker','진행 중')
            store.finish_repository_analysis_with_response(req.analysis_id,'worker',RepositoryAnalysisStatus.NEEDS_ATTENTION,'모델 실행 실패',reason='MODEL_OUTCOME_UNKNOWN')
            self.assertEqual([],store.list_messages('session'))
            self.assertEqual([],store.list_messages(req.analysis_id))
            self.assertTrue(any('장기 저장소 분석 · 실패' in x['text'] for x in store.deliverable_outbound('telegram')))
            print('S02 REGRESSION: error finalizer shows failure; S03 chat history defect remains')

    def test_resume_preserves_completed_stage_workspace(self):
        class Runner(FakeRoleRunner):
            def __init__(self):
                super().__init__(review_responses=[[]],development_outputs=['fixed'])
                self.stage2_seen=[]
            def run(self,run_id,stage_id,role_id,repository,prompt,**kwargs):
                if stage_id=='stage-002' and role_id==RoleId.DEVELOPMENT:
                    self.stage2_seen.append((Path(repository)/'feature.txt').exists())
                    return HermesResult(json.dumps({'summary':'확인 필요','needs_user_input':['이어서 진행할까요?']}),TokenUsage(100,20),0.01)
                return super().run(run_id,stage_id,role_id,repository,prompt,**kwargs)
        with temporary_directory() as d:
            root=Path(d)
            repo=create_repository(root)
            runner=Runner()
            store,worker,run_id=build_pipeline(root,repo,runner,stage_count=2)
            worker.run_once()
            paused=store.load_run(run_id)
            self.assertEqual(RunPhase.PAUSED,paused.phase)
            self.assertEqual(1,paused.stage_index)
            worker.machine.transition(paused,RunPhase.DEVELOPING,message='audit resume')
            worker.resume(run_id)
            worker.run_once()
            self.assertEqual([True,True],runner.stage2_seen)
            print('S08 REGRESSION: stage 2 resume preserves stage 1 output and candidate')

if __name__=='__main__': unittest.main(verbosity=2)
