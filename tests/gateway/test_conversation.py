import unittest
from collections.abc import Callable
from pathlib import Path

from app.contracts import RunPhase, TokenUsage
from app.gateway.core import AgentReply, IncomingMessage, ProposedStage, RepositoryInfo
from app.services.budget import BudgetManager, BudgetPolicy
from tests.gateway.support import build_application, temporary_directory


def incoming(identifier: int, text: str, *, channel: str = "telegram"):
    return IncomingMessage(
        channel=channel,
        conversation_id="200",
        user_id="100",
        external_message_id=f"message-{identifier}",
        text=text,
    )


class ReadyBackend:
    def __init__(self):
        self.calls = 0

    def respond(self, state, context, message):
        self.calls += 1
        return AgentReply(
            "한 단계 계획을 만들었습니다.",
            stages=(
                ProposedStage(
                    objective="기능 구현",
                    scope=("app/",),
                    acceptance_criteria=("테스트 통과",),
                    verification_commands=("python -m unittest",),
                ),
            ),
        )


class CapturingReadyBackend(ReadyBackend):
    def __init__(self):
        super().__init__()
        self.contexts = []

    def respond(self, state, context, message):
        self.contexts.append(context)
        return super().respond(state, context, message)


class CapturingTeamBackend:
    def __init__(self):
        self.contexts = []

    def preflight(self, state, context, message, reply_count):
        return None

    def respond_as(
        self,
        state,
        context,
        message,
        role_id,
        *,
        caller_role=None,
        call_purpose="",
        turn_messages=(),
        call_index=1,
    ):
        self.contexts.append(context)
        return AgentReply(f"{role_id.value} 답변")


class SwitchingRepositoryValidator:
    """Returns distinct stable identities without needing real test repositories."""

    def validate(
        self,
        raw_path: str,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> RepositoryInfo:
        normalized = raw_path.replace("/", "\\").lower()
        if normalized.endswith("\\alpha"):
            return RepositoryInfo(
                Path("D:/projects/alpha"), "main", "a" * 64, "1" * 40
            )
        if normalized.endswith("\\beta"):
            return RepositoryInfo(
                Path("D:/projects/beta"), "main", "b" * 64, "2" * 40
            )
        raise ValueError("테스트용 저장소를 찾을 수 없습니다")


class ConversationTests(unittest.TestCase):
    def test_free_chat_context_is_handed_off_to_a_new_development_task(self):
        with temporary_directory() as directory:
            root = Path(directory)
            backend = CapturingReadyBackend()
            store, application = build_application(root, backend)

            application.handle(
                incoming(1, "센티널아 로그인은 이메일과 비밀번호만 쓰기로 했어")
            )
            application.handle(incoming(2, "그 기준으로 로그인 기능을 개발해"))
            application.handle(incoming(3, "D:\\projects\\sample"))
            application.handle(incoming(4, "이 프로젝트 사용 승인해"))

            self.assertEqual(1, backend.calls)
            decisions = [
                item["content"] for item in backend.contexts[-1].decisions
            ]
            handoff = next(
                item for item in decisions if item.startswith("자유 대화에서 이어진 작업 맥락:")
            )
            self.assertIn("이메일과 비밀번호만 쓰기로 했어", handoff)
            binding = store.load_conversation("telegram", "200")
            self.assertIsNotNone(binding)
            assert binding is not None
            event_types = {
                event["event_type"]
                for event in store.list_events(binding["active_task_id"])
            }
            self.assertIn("SESSION_CONTEXT_HANDED_OFF", event_types)

    def test_new_work_keeps_a_new_task_separate_from_free_chat_context(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store, application = build_application(root, ReadyBackend())

            application.handle(incoming(1, "센티널아 이전 작업의 조건이야"))
            application.handle(incoming(2, "새 작업"))

            binding = store.load_conversation("telegram", "200")
            self.assertIsNotNone(binding)
            assert binding is not None
            contents = [
                item["content"]
                for item in store.list_messages(binding["active_task_id"])
            ]
            self.assertFalse(
                any(item.startswith("자유 대화에서 이어진 작업 맥락:") for item in contents)
            )

    def test_free_chat_handoff_is_bounded_before_it_reaches_a_task(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store, application = build_application(root, ReadyBackend())
            application.handle(incoming(1, "대화 시작"))
            session = store.load_conversation_session("telegram", "200")
            self.assertIsNotNone(session)
            assert session is not None
            for _index in range(10):
                application.router.context.add_message(
                    session.session_run_id, "100", "x" * 1400
                )

            handoff = application.router._session_task_handoff(session.session_run_id)

            self.assertLessEqual(len(handoff), 6000)
            self.assertIn("길이 제한으로 생략", handoff)

    def test_handoff_excludes_context_from_a_previous_repository(self):
        with temporary_directory() as directory:
            backend = CapturingReadyBackend()
            _store, application = build_application(
                Path(directory),
                backend,
                repository_validator=SwitchingRepositoryValidator(),
            )

            application.handle(incoming(1, "D:\\projects\\alpha"))
            application.handle(incoming(2, "이 프로젝트 사용 승인해"))
            application.handle(incoming(3, "알파 저장소에서만 쓰는 인증 규칙이야"))
            application.handle(incoming(4, "D:\\projects\\beta"))
            application.handle(incoming(5, "이 프로젝트 사용 승인해"))
            application.handle(incoming(6, "베타 저장소의 로그인 기능을 개발해"))

            self.assertEqual(1, backend.calls)
            decisions = [
                item["content"] for item in backend.contexts[-1].decisions
            ]
            handoff = next(
                item
                for item in decisions
                if item.startswith("자유 대화에서 이어진 작업 맥락:")
            )
            self.assertNotIn("알파 저장소에서만 쓰는 인증 규칙", handoff)
            self.assertIn("D:\\projects\\beta", handoff)

    def test_repository_switch_also_cuts_legacy_unscoped_context(self):
        with temporary_directory() as directory:
            backend = CapturingReadyBackend()
            store, application = build_application(
                Path(directory),
                backend,
                repository_validator=SwitchingRepositoryValidator(),
            )
            application.handle(incoming(1, "일반 대화로 시작"))
            session = store.load_conversation_session("telegram", "200")
            self.assertIsNotNone(session)
            assert session is not None
            application.router.context.add_message(
                session.session_run_id, "100", "이전 저장소의 비밀 규칙"
            )
            store.set_current_project(
                "telegram",
                "200",
                "100",
                "D:\\projects\\alpha",
                repository_identity="a" * 64,
                head_sha="1" * 40,
            )

            application.handle(incoming(2, "D:\\projects\\beta"))
            application.handle(incoming(3, "이 프로젝트 사용 승인해"))
            application.handle(incoming(4, "베타 저장소 기능을 개발해"))

            decisions = [
                item["content"] for item in backend.contexts[-1].decisions
            ]
            handoff = next(
                item
                for item in decisions
                if item.startswith("자유 대화에서 이어진 작업 맥락:")
            )
            self.assertNotIn("이전 저장소의 비밀 규칙", handoff)

    def test_repository_switch_excludes_prior_free_chat_context_and_memory(self):
        with temporary_directory() as directory:
            team = CapturingTeamBackend()
            store, application = build_application(
                Path(directory),
                object(),
                team_backend=team,
                repository_validator=SwitchingRepositoryValidator(),
            )

            application.handle(incoming(1, "D:\\projects\\alpha"))
            application.handle(incoming(2, "이 프로젝트 사용 승인해"))
            application.handle(incoming(3, "ALPHA_ONLY"))
            session = store.load_conversation_session("telegram", "200")
            self.assertIsNotNone(session)
            assert session is not None
            application.router.context.save_memory(
                "conversation",
                f"telegram:200:repository:{'a' * 64}",
                "ALPHA_CONVERSATION_MEMORY",
            )

            application.handle(incoming(4, "D:\\projects\\beta"))
            application.handle(incoming(5, "이 프로젝트 사용 승인해"))
            application.handle(incoming(6, "BETA_ONLY"))

            context = team.contexts[-1]
            contents = [item["content"] for item in context.recent_messages]
            memories = [item["content"] for item in context.memories]
            self.assertIn("BETA_ONLY", contents)
            self.assertNotIn("ALPHA_ONLY", contents)
            self.assertNotIn("ALPHA_CONVERSATION_MEMORY", memories)

    def test_natural_conversation_collects_request_and_dynamic_project(self):
        with temporary_directory() as directory:
            root = Path(directory)
            backend = ReadyBackend()
            store, application = build_application(root, backend)

            first = application.handle(incoming(1, "로그인 기능을 만들어줘"))
            self.assertIn("절대경로", first[0].text)
            second = application.handle(incoming(2, "D:\\projects\\sample"))
            self.assertIn("프로젝트 사용", second[0].text)
            third = application.handle(incoming(3, "이 프로젝트 사용 승인해"))
            self.assertIn("계획이 준비", third[0].text)
            self.assertIn("수정 범위: app/", third[0].text)
            self.assertIn("완료 조건: 테스트 통과", third[0].text)
            self.assertIn("검증: python -m unittest", third[0].text)

            binding = store.load_conversation("telegram", "200")
            state = store.load_run(binding["run_id"] if binding else "")
            self.assertEqual(str((root / "selected-repository").resolve()), state.repository)
            self.assertEqual("로그인 기능을 만들어줘", state.objective)
            self.assertEqual(RunPhase.WAITING_APPROVAL, state.phase)
            self.assertTrue(
                (root / "artifacts" / state.run_id / "plans" / "revision-0001" / "plan.json").is_file()
            )

    def test_exact_phrase_is_required_for_approval(self):
        with temporary_directory() as directory:
            store, application = build_application(Path(directory), ReadyBackend())
            application.handle(incoming(1, "기능 개발"))
            application.handle(incoming(2, "D:\\projects\\sample"))
            application.handle(incoming(3, "이 프로젝트 사용 승인해"))

            application.handle(incoming(4, "승인합니다"))
            binding = store.load_conversation("telegram", "200")
            not_approved = store.load_run(binding["run_id"] if binding else "")
            self.assertFalse(not_approved.approval_granted)

            application.handle(incoming(5, "개발 시작해"))
            binding = store.load_conversation("telegram", "200")
            state = store.load_run(binding["run_id"] if binding else "")
            self.assertTrue(state.approval_granted)
            self.assertEqual("100", state.approved_by)

    def test_waiting_approval_keeps_plan_for_general_questions_and_revises_explicitly(self):
        with temporary_directory() as directory:
            backend = ReadyBackend()
            store, application = build_application(Path(directory), backend)
            application.handle(incoming(1, "기능 개발"))
            application.handle(incoming(2, "D:\\projects\\sample"))
            application.handle(incoming(3, "이 프로젝트 사용 승인해"))
            binding = store.load_conversation("telegram", "200")
            initial = store.load_run(binding["run_id"])

            question = application.handle(incoming(4, "이 검증 명령은 왜 필요한 거야?"))
            unchanged = store.load_run(initial.run_id)

            self.assertEqual(1, backend.calls)
            self.assertEqual(1, unchanged.plan_revision)
            self.assertEqual(RunPhase.WAITING_APPROVAL, unchanged.phase)
            self.assertIn("범위 선택 이유", question[0].text)
            self.assertIn("계획 버전 1", question[0].text)
            self.assertIn("고정 commit", question[0].text)
            risk = application.handle(incoming(6, "이 계획에서 어떤 위험이 있어?"))
            self.assertIn("주요 위험", risk[0].text)
            self.assertEqual(1, store.load_run(initial.run_id).plan_revision)
            self.assertEqual(1, backend.calls)

            application.handle(incoming(5, "계획 수정: 검증 명령을 pytest로 바꿔줘"))
            revised = store.load_run(initial.run_id)
            self.assertEqual(2, backend.calls)
            self.assertEqual(2, revised.plan_revision)
            self.assertEqual(RunPhase.WAITING_APPROVAL, revised.phase)
            self.assertTrue(
                application.router._is_plan_revision_request(
                    "로그인 예외 처리도 추가해줘"
                )
            )
            self.assertFalse(
                application.router._is_plan_revision_request(
                    "이 검증은 왜 필요한 거야?"
                )
            )

    def test_duplicate_message_does_not_call_backend_twice(self):
        with temporary_directory() as directory:
            backend = ReadyBackend()
            _, application = build_application(Path(directory), backend)
            application.handle(incoming(1, "기능 개발"))
            application.handle(incoming(2, "D:\\projects\\sample"))
            approval_message = incoming(3, "이 프로젝트 사용 승인해")
            self.assertTrue(application.handle(approval_message))
            self.assertEqual((), application.handle(approval_message))
            self.assertEqual(1, backend.calls)

    def test_natural_status_and_stop_phrases_are_immediate_controls(self):
        with temporary_directory() as directory:
            _store, application = build_application(Path(directory), ReadyBackend())

            status = application.handle(incoming(1, "진행 상황 알려줘"))
            stopped = application.handle(incoming(2, "멈춰줘"))

            self.assertIn("아직 시작한 작업", status[0].text)
            self.assertIn("진행 중인 작업이 없습니다", stopped[0].text)
            self.assertTrue(application.router.is_immediate_control(incoming(3, "뭐 하고 있어")))
            self.assertTrue(application.router.is_stop_control(incoming(4, "중단해줘")))

    def test_usage_phrase_is_an_immediate_control_and_reports_actual_totals(self):
        with temporary_directory() as directory:
            store, application = build_application(Path(directory), ReadyBackend())
            application.handle(incoming(1, "안녕"))
            binding = store.load_conversation("telegram", "200")
            assert binding is not None
            budget = BudgetManager(BudgetPolicy(), store)
            budget.record_usage(
                binding["run_id"],
                "chat-test",
                "development",
                "conversation",
                TokenUsage(input_tokens=12, output_tokens=3),
            )
            application.router.budget = budget

            usage = application.handle(incoming(2, "사용량"))

            self.assertTrue(application.router.is_immediate_control(incoming(3, "토큰")))
            self.assertIn("토큰 사용: 15", usage[0].text)
            self.assertIn("대화 한도: 측정 중", usage[0].text)

    def test_router_is_not_tied_to_telegram(self):
        with temporary_directory() as directory:
            root = Path(directory)
            backend = ReadyBackend()
            store, original = build_application(root, backend)
            original.access_policy = original.access_policy.__class__(
                allowed_users=frozenset({"100"})
            )
            first = incoming(1, "기능 개발", channel="discord")
            result = original.handle(first)
            self.assertEqual("discord", result[0].channel)
            self.assertIsNotNone(store.load_conversation("discord", "200"))

    def test_unauthorized_message_creates_no_run(self):
        with temporary_directory() as directory:
            root = Path(directory)
            _, application = build_application(root, ReadyBackend())
            blocked = IncomingMessage(
                channel="telegram",
                conversation_id="200",
                user_id="999",
                external_message_id="blocked-1",
                text="비인가 요청",
            )
            self.assertEqual((), application.handle(blocked))
            self.assertIsNone(
                application.store.load_conversation("telegram", "200")
            )


if __name__ == "__main__":
    unittest.main()
