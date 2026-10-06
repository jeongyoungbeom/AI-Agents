import unittest
from pathlib import Path

from app.contracts import RoleId
from app.gateway.core.role_routing import RoleResolver
from app.services.toolchains import ToolchainService
from tests.gateway.test_toolchain_approval import ToolchainApprovalTests, incoming
from tests.gateway.support import temporary_directory
from tests.pipeline.test_toolchains import ProbeSandbox, write_catalog


class S11A11A15Tests(unittest.TestCase):
    def test_opinion_receipt_request_allows_only_named_peer_and_keeps_owner(self):
        for text in ('센티널이 정리하되 빌더 의견을 받아줘',
                     '센티널아 빌더 의견도 들어줘'):
            with self.subTest(text=text):
                intent=RoleResolver().interpret(text,RoleId.DEVELOPMENT)
                self.assertEqual((RoleId.REVIEW,),intent.roles)
                self.assertEqual((RoleId.DEVELOPMENT,),intent.allowed_delegate_roles)
                self.assertFalse(intent.group_call)

    def test_negated_quoted_or_description_opinions_do_not_grant_consultation(self):
        for text in ('센티널아 빌더 의견은 받지 마',
                     '센티널아 빌더 의견을 받아주지 마',
                     '센티널아 "빌더 의견을 받아줘"라는 문장을 설명해줘',
                     '센티널아 빌더 의견을 받아줘라는 말이 무슨 뜻이야',
                     '센티널아 빌더 의견을 설명해줘'):
            with self.subTest(text=text):
                self.assertEqual((),RoleResolver().interpret(text,RoleId.REVIEW).allowed_delegate_roles)

    def test_approval_uses_real_toolchain_service_repository_contract(self):
        with temporary_directory() as directory:
            root=Path(directory)
            (root/'selected-repository').mkdir()
            sandbox=ProbeSandbox()
            preflight=ToolchainService.load(write_catalog(root),sandbox=sandbox)
            helper=ToolchainApprovalTests()
            store,application,scheduler,run_id=helper._planned_application(root,preflight)
            reply=application.handle(incoming('real-toolchain-approval','개발 시작해'))
            self.assertIn('실행 큐',reply[0].text)
            self.assertTrue(store.load_run(run_id).approval_granted)
            self.assertEqual([run_id],scheduler.enqueued)
            self.assertTrue(any(call[0][0]=='git' for call in sandbox.calls))
            self.assertTrue(any(call[0][0]=='python' for call in sandbox.calls))


if __name__=='__main__':
    unittest.main()
