"""S00 baseline: passing reproduces a defect; it does not mean the defect is fixed."""
from pathlib import Path
import unittest
from app.agents.parsing import parse_team_conversation_reply, InvalidAgentResponse
from app.agents.team_conversation_backend import NO_TOOLS_TOOLSET
from app.contracts import RoleId
from app.services.repository import RepositoryToolBatch, RepositoryToolRequest
from tests.gateway.support import temporary_directory, build_application, TEST_REPOSITORY_IDENTITY, future_expiry
from tests.gateway.test_conversation_foundation import incoming

class BoundaryReproductions(unittest.TestCase):
    def test_warning_before_valid_json_breaks_response_contract(self):
        payload='{"message":"정상 응답","calls":[],"memory_updates":[],"repository_tools":[]}'
        self.assertEqual('정상 응답', parse_team_conversation_reply(payload, RoleId.REVIEW).text)
        actual_prefix=f'Warning: Unknown toolsets: {NO_TOOLS_TOOLSET}\n'
        with self.assertRaises(InvalidAgentResponse):
            parse_team_conversation_reply(actual_prefix+payload,RoleId.REVIEW)
        print('CONFIRMED: observed Hermes warning corrupts an otherwise valid response')

    def test_tool_round_replaces_earlier_file_evidence(self):
        class Tools:
            def execute(self,*args,**kwargs):
                return RepositoryToolBatch(TEST_REPOSITORY_IDENTITY,'b'*40,())
        with temporary_directory() as d:
            store,app=build_application(Path(d),None)
            app.handle(incoming(1,'안녕'))
            binding=store.load_conversation('telegram','200')
            state=store.load_run(binding['run_id'])
            store.set_current_project('telegram','200','100',d,repository_identity=TEST_REPOSITORY_IDENTITY,head_sha='b'*40,approved=True,approval_expires_at=future_expiry())
            app.router.repository_tools=Tools()
            old={'head_sha':'b'*40,'documents':[{'path':'README.md','content':'initial evidence'}],'tool_results':[{'content':'earlier evidence'}]}
            new,error=app.router._repository_tool_context_for_chat(incoming(2,'다음 파일도 봐'),state,RoleId.REVIEW,old,(RepositoryToolRequest('read_file',path='next.py'),),tool_round=2)
            self.assertFalse(error)
            self.assertEqual([],new['documents'])
            self.assertNotIn('earlier evidence',str(new))
            print('CONFIRMED: next tool round removes original documents and previous tool evidence')

if __name__=='__main__': unittest.main(verbosity=2)
