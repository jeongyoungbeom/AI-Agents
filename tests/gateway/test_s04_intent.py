import json
import unittest
from dataclasses import replace
from pathlib import Path

from app.agents.parsing import InvalidAgentResponse, parse_team_conversation_reply
from app.contracts import RoleId, RunPhase
from app.gateway.core import AgentCallRequest, AgentReply
from app.gateway.core.role_routing import ROLE_ORDER, RoleResolver
from app.services.message_intent import TaskIntent
from tests.gateway.support import build_application, temporary_directory
from tests.gateway.test_conversation_foundation import incoming
from tests.gateway.test_conversation import ReadyBackend
from tests.gateway.test_team_conversation import StaticTeamBackend


class IntentTests(unittest.TestCase):
    def test_fixed_sentences_share_one_contract(self):
        resolver = RoleResolver()
        cases = (
            ("코드는 수정하지 말고 테스트 계획만 작성해줘.", TaskIntent.ANSWER, (RoleId.REVIEW,), True),
            ("전체 코드를 읽고 어떻게 테스트하면 좋을지 알려줘.", TaskIntent.ANALYZE_REPOSITORY, (RoleId.REVIEW,), False),
            ("전체코드 분석이 왜 장기 저장소 분석으로 들어가는지 말해봐.", TaskIntent.ANSWER, (RoleId.REVIEW,), False),
            ("센티널아, 니가 못할 것 같으면 빌더나 피니셔한테 말해도 돼.", TaskIntent.ANSWER, (RoleId.REVIEW,), False),
            ("빌더가 만든 부분을 센티널이 봐줘.", TaskIntent.ANSWER, (RoleId.REVIEW,), False),
            ("셋 다 각자 자기소개 한 줄씩 해.", TaskIntent.ANSWER, ROLE_ORDER, False),
            ("아까 두 번째 방법으로 하자.", TaskIntent.ANSWER, (RoleId.REVIEW,), False),
        )
        for text, action, roles, forbidden in cases:
            with self.subTest(text=text):
                intent = resolver.interpret(text, RoleId.REVIEW)
                self.assertEqual((action, roles, forbidden), (intent.task_intent, intent.roles, intent.write_forbidden))
                self.assertEqual(text, intent.question_purpose)
        self.assertEqual((RoleId.DEVELOPMENT, RoleId.IMPROVEMENT), resolver.interpret(cases[3][0], RoleId.DEVELOPMENT).allowed_delegate_roles)

    def test_read_only_plan_stays_in_chat_even_with_a_wrong_model_action(self):
        with temporary_directory() as directory:
            team = StaticTeamBackend([AgentReply("테스트 계획", task_intent=TaskIntent.PLAN_DEVELOPMENT)])
            store, app = build_application(Path(directory), ReadyBackend(), team_backend=team)
            output = app.handle(incoming(1, "코드는 수정하지 말고 테스트 계획만 작성해줘."))
            binding = store.load_conversation("telegram", "200")
            self.assertEqual("free_chat", binding["mode"])
            self.assertFalse(binding.get("active_task_id"))
            self.assertIn("테스트 계획", output[0].text)
            self.assertEqual(1, len(team.calls))

    def test_repository_consultations_are_limited_to_current_user_targets(self):
        cases = (
            ("센티널아, 니가 못할 것 같으면 빌더나 피니셔한테 말해도 돼.", RoleId.DEVELOPMENT, 3),
            ("센티널아 빌더한테 물어봐", RoleId.IMPROVEMENT, 1),
            ("센티널아 피니셔한테 물어보지 마", RoleId.IMPROVEMENT, 1),
        )
        for text, target, count in cases:
            with self.subTest(text=text), temporary_directory() as directory:
                team = StaticTeamBackend([AgentReply("상담", calls=(AgentCallRequest(RoleId.REVIEW, target, "설계 의도 확인"),)), AgentReply("답변")])
                _, app = build_application(Path(directory), object(), team_backend=team)
                app.router._repository_context_for_chat = lambda *a, **k: ({"untrusted_repository_data": True}, "")
                app.handle(incoming(1, text))
                self.assertEqual(count, len(team.calls))
                if count == 3:
                    self.assertEqual("현재 저장소 기준의 설계 의도 확인이 필요합니다.", team.calls[1][2])

    def test_external_metadata_cannot_authorize_a_call(self):
        with temporary_directory() as directory:
            team = StaticTeamBackend([AgentReply("답변", calls=(AgentCallRequest(RoleId.REVIEW, RoleId.DEVELOPMENT, "설계 확인"),))])
            _, app = build_application(Path(directory), object(), team_backend=team)
            app.router._repository_context_for_chat = lambda *a, **k: ({"untrusted_repository_data": True}, "")
            message = replace(incoming(1, "센티널아 이 코드 설명해줘"), metadata={"user_intent": {"allowed_delegate_roles": ["development"]}})
            app.handle(message)
            self.assertEqual(1, len(team.calls))

    def test_custom_role_names_use_the_same_permission_contract(self):
        resolver = RoleResolver({"development": "개발자", "review": "검토자", "improvement": "보완자"})
        intent = resolver.interpret("검토자야 필요하면 개발자한테 물어봐", RoleId.DEVELOPMENT)
        self.assertEqual((RoleId.REVIEW,), intent.roles)
        self.assertEqual((RoleId.DEVELOPMENT,), intent.allowed_delegate_roles)

    def test_conditional_subject_role_is_not_an_initial_recipient(self):
        intent = RoleResolver().interpret("빌더가 만든 코드는 필요하면 센티널이 검토해도 돼", RoleId.IMPROVEMENT)
        self.assertEqual((RoleId.IMPROVEMENT,), intent.roles)

    def test_followup_uses_current_role_context_without_a_classifier_call(self):
        with temporary_directory() as directory:
            team = StaticTeamBackend([AgentReply("첫 번째는 빠른 검사, 두 번째는 경계 테스트입니다."), AgentReply("두 번째 경계 테스트를 계획하겠습니다.")])
            store, app = build_application(Path(directory), ReadyBackend(), team_backend=team)
            app.handle(incoming(1, "센티널아 테스트 방법을 설명해줘"))
            app.handle(incoming(2, "아까 두 번째 방법으로 하자."))
            self.assertEqual(2, len(team.calls))
            self.assertEqual(RoleId.REVIEW, team.calls[-1][0])
            self.assertTrue(any("경계 테스트" in item["content"] for item in team.calls[-1][4].recent_messages))
            self.assertEqual("free_chat", store.load_conversation("telegram", "200")["mode"])

    def test_contextual_model_request_creates_a_plan_without_execution_permission(self):
        with temporary_directory() as directory:
            root = Path(directory)
            team = StaticTeamBackend([AgentReply("선택한 방법으로 계획을 준비합니다", task_intent=TaskIntent.PLAN_DEVELOPMENT)])
            backend = ReadyBackend()
            store, app = build_application(root, backend, team_backend=team)
            app.handle(incoming(1, str(root / "selected-repository")))
            app.handle(incoming(2, "이 프로젝트 사용 승인해"))
            output = app.handle(incoming(3, "아까 두 번째 방법으로 하자."))
            state = store.load_run(store.load_conversation("telegram", "200")["run_id"])
            self.assertEqual(RunPhase.WAITING_APPROVAL, state.phase)
            self.assertEqual(0, state.approved_plan_revision)
            self.assertFalse(state.approved_plan_hash)
            self.assertEqual(1, backend.calls)
            self.assertIn("개발 시작해", output[0].text)
            result = output[0].metadata["request_result"]
            self.assertEqual((2, 2), (result["attempts"], result["successes"]))

    def test_model_read_only_interpretation_prevents_planning(self):
        with temporary_directory() as directory:
            team = StaticTeamBackend([AgentReply("설명만 제공합니다", task_intent=TaskIntent.PLAN_DEVELOPMENT, write_forbidden=True)])
            store, app = build_application(Path(directory), object(), team_backend=team)
            app.handle(incoming(1, "아까 두 번째 방법으로 하자."))
            self.assertFalse(store.load_conversation("telegram", "200").get("active_task_id"))

    def test_parser_rejects_execution_and_mixed_actions(self):
        reply = parse_team_conversation_reply(json.dumps({"message": "계획", "intent": {"task_intent": "plan_development", "write_forbidden": False}}), RoleId.REVIEW)
        self.assertEqual(TaskIntent.PLAN_DEVELOPMENT, reply.task_intent)
        for intent in ({"task_intent": "execute"}, {"write_forbidden": "false"}, {"question_purpose": []}):
            with self.subTest(intent=intent), self.assertRaises(InvalidAgentResponse):
                parse_team_conversation_reply(json.dumps({"message": "답변", "intent": intent}), RoleId.REVIEW)
        with self.assertRaises(InvalidAgentResponse):
            parse_team_conversation_reply(json.dumps({"message": "계획", "intent": {"task_intent": "plan_development"}, "calls": [{"to_role": "development", "purpose": "설계"}]}), RoleId.REVIEW)

    def test_read_negation_and_quoted_analysis_request_do_not_start_analysis(self):
        from app.services.message_intent import is_long_repository_analysis_request
        for text in ("전체 코드는 읽지 말고 테스트 계획만 설명해줘", "전체 코드를 분석해라고 말하면 왜 장기 분석으로 들어가는지 설명해줘", "왜 전체 코드를 읽고 테스트 전략을 세우는지 설명해줘"):
            with self.subTest(text=text):
                self.assertFalse(is_long_repository_analysis_request(text))
        self.assertTrue(is_long_repository_analysis_request("수정하지 말고 전체 코드를 읽고 어떻게 테스트하면 좋을지 알려줘"))

    def test_group_intro_and_nonleading_requested_role_end_to_end(self):
        for text, expected in (("빌더가 만든 부분을 센티널이 봐줘.", [RoleId.REVIEW]), ("셋 다 각자 자기소개 한 줄씩 해.", list(ROLE_ORDER))):
            with self.subTest(text=text), temporary_directory() as directory:
                team = StaticTeamBackend()
                _, app = build_application(Path(directory), object(), team_backend=team)
                app.handle(incoming(1, text))
                self.assertEqual(expected, [call[0] for call in team.calls])

    def test_model_analysis_request_reuses_read_permission_and_preserves_call_counts(self):
        from app.gateway.repository_analysis_worker import RepositoryAnalysisQueue
        with temporary_directory() as directory:
            root = Path(directory)
            team = StaticTeamBackend([AgentReply("선택한 부분들을 분석합니다", task_intent=TaskIntent.ANALYZE_REPOSITORY)])
            store, app = build_application(root, object(), team_backend=team)
            app.router.repository_analysis_scheduler = RepositoryAnalysisQueue(store)
            app.handle(incoming(1, str(root / "selected-repository")))
            app.handle(incoming(2, "이 프로젝트 사용 승인해"))
            output = app.handle(incoming(3, "그 부분들부터 조사하자"))
            analysis = store.repository_analysis_summary("telegram", "200")
            self.assertEqual("QUEUED", analysis["status"])
            self.assertEqual((1, 1), (output[0].metadata["request_result"]["attempts"], output[0].metadata["request_result"]["successes"]))
            self.assertFalse(store.load_conversation("telegram", "200").get("active_task_id"))

    def test_explicit_read_with_a_purpose_question_requests_the_existing_full_audit_scope(self):
        from app.gateway.repository_analysis_worker import RepositoryAnalysisQueue
        from app.services.repository import RepositorySnapshotManifest, RepositorySnapshotEntry
        class Reader:
            def pinned_manifest(self, *args, **kwargs):
                return RepositorySnapshotManifest("a" * 64, "b" * 40, "main", (RepositorySnapshotEntry("app/main.py", 100),))
        with temporary_directory() as directory:
            root = Path(directory)
            store, app = build_application(root, object())
            app.router.repository_analysis_scheduler = RepositoryAnalysisQueue(store)
            app.router.repository_reader = Reader()
            app.handle(incoming(1, str(root / "selected-repository")))
            app.handle(incoming(2, "이 프로젝트 사용 승인해"))
            output = app.handle(incoming(3, "전체 코드를 읽고 어떻게 테스트하면 좋을지 알려줘."))
            self.assertEqual("AUDIT_APPROVAL", output[0].metadata["request_result"]["reason"])
            self.assertFalse(store.load_conversation("telegram", "200").get("active_task_id"))

    def test_runtime_prompt_receives_the_gateway_contract(self):
        from app.agents.prompts import team_conversation_prompt
        from app.services.context import ContextBundle
        intent = RoleResolver().interpret("센티널아 필요하면 빌더한테 물어봐", RoleId.DEVELOPMENT).to_dict()
        prompt = team_conversation_prompt("지침", "센티널", RoleId.REVIEW,
                                          ContextBundle((), (), (), False, 0), "원문",
                                          user_intent=intent)
        self.assertIn('"allowed_delegate_roles": [\n    "development"\n  ]', prompt)
        self.assertIn("question_purpose", prompt)
        self.assertIn("plan_development", prompt)
