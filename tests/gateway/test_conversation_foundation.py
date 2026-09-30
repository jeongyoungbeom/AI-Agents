import sqlite3
import threading
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from app.contracts import RoleId, RunState
from app.gateway.conversation_worker import ConversationQueue, ConversationWorker
from app.gateway.core import (
    AgentCallRequest,
    AgentReply,
    ConversationCancelled,
    ConversationMode,
    IncomingMessage,
    LocalGitRepositoryValidator,
    ProposedStage,
    RepositoryInfo,
    RoleResolver,
)
from app.services.context import ContextService
from app.services.repository import RepositoryCancelled
from app.storage import StateStore
from tests.gateway.support import build_application, future_expiry, temporary_directory


def incoming(identifier: int, text: str, *, conversation: str = "200"):
    return IncomingMessage(
        channel="telegram",
        conversation_id=conversation,
        user_id="100",
        external_message_id=f"foundation-{identifier}",
        text=text,
    )


class CountingBackend:
    def __init__(self):
        self.calls = 0

    def respond(self, state, context, message):
        self.calls += 1
        raise AssertionError("free-chat foundation must not call the planning backend")


class RoleFoundationTests(unittest.TestCase):
    def test_role_resolver_separates_conditional_delegation_from_initial_requests(self):
        resolver = RoleResolver(
            {
                "development": "빌더",
                "review": "센티널",
                "improvement": "피니셔",
            }
        )

        conditional = resolver.resolve(
            "테스트 어떻게 하면 좋을지 내가 이해하기 쉽게 정리해서 말해줄 수 있을까?\n"
            "니가 못할 것 같으면 빌더나 피니셔한테 말해도돼",
            RoleId.REVIEW,
        )
        self.assertEqual((RoleId.REVIEW,), conditional.roles)
        self.assertFalse(conditional.explicit)

        direct = resolver.resolve(
            "센티널아, 모르면 빌더한테 물어봐", RoleId.DEVELOPMENT
        )
        self.assertEqual((RoleId.REVIEW,), direct.roles)
        self.assertTrue(direct.explicit)

        direct_before_plural = resolver.resolve(
            "센티널아, 빌더랑 피니셔 둘 다 답해줘", RoleId.DEVELOPMENT
        )
        self.assertEqual((RoleId.REVIEW,), direct_before_plural.roles)
        self.assertTrue(direct_before_plural.explicit)

        direct_without_delimiter = resolver.resolve(
            "센티널아 빌더한테 물어봐", RoleId.DEVELOPMENT
        )
        self.assertEqual((RoleId.REVIEW,), direct_without_delimiter.roles)
        self.assertTrue(direct_without_delimiter.explicit)

        second_direct_without_delimiter = resolver.resolve(
            "빌더야 센티널한테 검토 부탁해", RoleId.IMPROVEMENT
        )
        self.assertEqual(
            (RoleId.DEVELOPMENT,), second_direct_without_delimiter.roles
        )
        self.assertTrue(second_direct_without_delimiter.explicit)

        plural = resolver.resolve("빌더,\n피니셔 둘 다 답해줘", RoleId.REVIEW)
        self.assertEqual((RoleId.DEVELOPMENT, RoleId.IMPROVEMENT), plural.roles)
        self.assertTrue(plural.explicit)
        self.assertFalse(plural.group_call)

        group = resolver.resolve("셋 다 의견 줘", RoleId.REVIEW)
        self.assertEqual(
            (RoleId.DEVELOPMENT, RoleId.REVIEW, RoleId.IMPROVEMENT), group.roles
        )
        self.assertTrue(group.group_call)

        denied = resolver.resolve("모두를 부르지 마, 센티널만 답해", RoleId.DEVELOPMENT)
        self.assertEqual((RoleId.REVIEW,), denied.roles)
        self.assertFalse(denied.group_call)
        for phrase in ("셋 다 호출하지 말아줘", "모두라는 단어가 왜 단체 호출이지?"):
            self.assertFalse(resolver.resolve(phrase, RoleId.DEVELOPMENT).group_call)

        fallback = resolver.resolve("이어서 설명해줘", RoleId.IMPROVEMENT)
        self.assertEqual((RoleId.IMPROVEMENT,), fallback.roles)
        self.assertFalse(fallback.explicit)

        for conditional_text in (
            "모르면 빌더한테 물어봐",
            "필요하면 피니셔 불러도 돼",
            "빌더나 피니셔한테 말해도 돼",
            "센티널 불러도 돼",
        ):
            selection = resolver.resolve(conditional_text, RoleId.REVIEW)
            self.assertEqual((RoleId.REVIEW,), selection.roles)
            self.assertFalse(selection.explicit)

        renamed = RoleResolver(
            {
                "development": "개발자",
                "review": "검토자",
                "improvement": "보완자",
            }
        )
        renamed_direct = renamed.resolve(
            "  검토자야,\n개발자한테 물어봐", RoleId.DEVELOPMENT
        )
        self.assertEqual((RoleId.REVIEW,), renamed_direct.roles)
        self.assertTrue(renamed_direct.explicit)

    def test_user_can_select_one_last_or_all_roles(self):
        resolver = RoleResolver(
            {
                "development": "빌더",
                "review": "센티널",
                "improvement": "피니셔",
            }
        )

        explicit = resolver.resolve("센티널 의견은 어때?", RoleId.DEVELOPMENT)
        self.assertEqual((RoleId.REVIEW,), explicit.roles)
        self.assertTrue(explicit.explicit)

        fallback = resolver.resolve("계속 이야기해줘", RoleId.IMPROVEMENT)
        self.assertEqual((RoleId.IMPROVEMENT,), fallback.roles)
        self.assertFalse(fallback.explicit)

        group = resolver.resolve("얘들아 다들 봐줘", RoleId.REVIEW)
        self.assertEqual(
            (RoleId.DEVELOPMENT, RoleId.REVIEW, RoleId.IMPROVEMENT),
            group.roles,
        )
        self.assertTrue(group.group_call)

    def test_agents_can_call_any_other_agent_without_changing_handoff_rules(self):
        calls = (
            AgentCallRequest(RoleId.DEVELOPMENT, RoleId.IMPROVEMENT, "구현 의견"),
            AgentCallRequest(RoleId.REVIEW, RoleId.DEVELOPMENT, "설계 의도 확인"),
            AgentCallRequest(RoleId.IMPROVEMENT, RoleId.REVIEW, "리뷰 근거 확인"),
        )
        self.assertEqual(3, len(calls))
        with self.assertRaises(ValueError):
            AgentCallRequest(RoleId.REVIEW, RoleId.REVIEW, "자기 호출")


class ModeAndMemoryTests(unittest.TestCase):
    def test_approved_project_owner_can_view_and_clear_project_facts(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store, application = build_application(root, CountingBackend())
            identity = "a" * 64
            application.handle(incoming(0, "기억 조회"))
            store.set_current_project(
                "telegram", "200", "100", str(root / "project"),
                repository_identity=identity, head_sha="b" * 40,
                approved=True, approval_expires_at=future_expiry(),
            )
            store.save_memory("project", identity, "shared", "프로젝트 규칙")
            store.save_memory("project", str(root / "project"), "shared", "이전 경로 기억")
            viewed = application.handle(incoming(1, "프로젝트 기억 조회"))
            self.assertIn("프로젝트 규칙", viewed[0].text)
            self.assertIn("이전 경로 기억", viewed[0].text)
            removed = application.handle(incoming(2, "프로젝트 기억 삭제"))
            self.assertIn("2건", removed[0].text)
            self.assertEqual([], store.list_memories((("project", identity),)))
            self.assertEqual([], store.list_memories((("project", str(root / "project")),)))

    def test_user_can_replace_and_delete_one_fact_without_losing_another(self):
        with temporary_directory() as directory:
            store, application = build_application(Path(directory), CountingBackend())
            application.handle(incoming(1, "기억 수정: 한국어 선호"))
            application.handle(incoming(2, "기억 수정: 짧은 답변 선호"))
            facts = store.list_memories((("user", "100"),))
            self.assertEqual(2, len(facts))
            first_id = facts[0]["fact_id"]
            reply = application.handle(incoming(3, f"기억 교체: {first_id}: 영어 선호"))
            self.assertIn("교체했습니다", reply[0].text)
            facts = store.list_memories((("user", "100"),))
            self.assertEqual({"영어 선호", "짧은 답변 선호"},
                             {item["content"] for item in facts})
            new_id = next(item["fact_id"] for item in facts if item["content"] == "영어 선호")
            application.handle(incoming(4, f"기억 삭제: {new_id}"))
            self.assertEqual(["짧은 답변 선호"],
                             [item["content"] for item in store.list_memories((("user", "100"),))])

    def test_user_can_view_update_and_delete_own_long_term_memory(self):
        with temporary_directory() as directory:
            _store, application = build_application(Path(directory), CountingBackend())

            updated = application.handle(incoming(1, "기억 수정: 답변은 한국어로 짧게"))
            viewed = application.handle(incoming(2, "기억 조회"))
            deleted = application.handle(incoming(3, "기억 삭제"))
            empty = application.handle(incoming(4, "기억 조회"))

            self.assertIn("저장했습니다", updated[0].text)
            self.assertIn("답변은 한국어로 짧게", viewed[0].text)
            self.assertIn("삭제했습니다", deleted[0].text)
            self.assertIn("없습니다", empty[0].text)

    def test_casual_chat_does_not_require_a_repository(self):
        with temporary_directory() as directory:
            backend = CountingBackend()
            store, application = build_application(Path(directory), backend)

            greeting = application.handle(incoming(1, "안녕 친구들"))
            self.assertNotIn("절대경로", greeting[0].text)
            self.assertIn("[빌더]", greeting[0].text)
            binding = store.load_conversation("telegram", "200")
            self.assertEqual(ConversationMode.FREE_CHAT.value, binding["mode"])
            self.assertEqual(0, backend.calls)

            sentinel = application.handle(incoming(2, "센티널은 어떻게 생각해?"))
            self.assertIn("[센티널]", sentinel[0].text)
            self.assertEqual(
                RoleId.REVIEW.value,
                store.load_conversation("telegram", "200")["active_role"],
            )

            everyone = application.handle(incoming(3, "얘들아 모두 모여봐"))
            self.assertEqual(3, len(everyone))
            self.assertIn("[피니셔]", everyone[-1].text)

    def test_development_intent_switches_to_planning_and_requests_project(self):
        with temporary_directory() as directory:
            store, application = build_application(Path(directory), CountingBackend())

            result = application.handle(incoming(1, "로그인 기능을 만들어줘"))
            binding = store.load_conversation("telegram", "200")

            self.assertIn("절대경로", result[0].text)
            self.assertEqual(ConversationMode.PLANNING.value, binding["mode"])
            self.assertEqual(RoleId.DEVELOPMENT.value, binding["active_role"])

    def test_work_intent_distinguishes_requests_from_questions(self):
        from app.gateway.core.conversation import DialogueRouter

        for phrase in ("전체 코드를 수정해", "로그인 기능을 추가하고 싶어", "버그를 잡아줘", "테스트를 작성해줘"):
            self.assertTrue(DialogueRouter._is_work_intent(phrase), phrase)
        for phrase in ("수정해도 되는지 판단해줘", "왜 수정해라고 말하면 실행돼?"):
            self.assertFalse(DialogueRouter._is_work_intent(phrase), phrase)

    def test_concept_question_does_not_inspect_repository(self):
        from app.services.repository import SafeRepositoryReader

        self.assertFalse(SafeRepositoryReader.should_inspect("코드 리뷰란 뭐야?"))
        self.assertFalse(SafeRepositoryReader.should_inspect("코드가 없을 때 어떻게 설명해?"))
        self.assertFalse(SafeRepositoryReader.should_inspect("전체코드 분석이 왜 장기 저장소 분석으로 들어가는지 말해봐"))
        self.assertTrue(SafeRepositoryReader.should_inspect("이 프로젝트의 코드를 확인해줘"))

    def test_scoped_memories_survive_restart_and_filter_by_role(self):
        with temporary_directory() as directory:
            database = Path(directory) / "state.db"
            store = StateStore(database)
            store.create_run(RunState("RUN-MEMORY"))
            context = ContextService(store)
            context.save_memory("conversation", "telegram:200", "공통 대화 요약")
            context.save_memory(
                "conversation", "telegram:200", "센티널 메모", role_id="review"
            )
            context.save_memory(
                "conversation", "telegram:200", "빌더 메모", role_id="development"
            )
            context.save_memory("user", "100", "사용자는 한국어를 선호함")
            context.save_memory("project", "D:\\repo", "프로젝트 규칙")

            restarted = ContextService(StateStore(database))
            bundle = restarted.build(
                "RUN-MEMORY",
                conversation_key="telegram:200",
                user_id="100",
                role_id="review",
                repository="D:\\repo",
            )
            contents = {item["content"] for item in bundle.memories}

            self.assertIn("공통 대화 요약", contents)
            self.assertIn("센티널 메모", contents)
            self.assertIn("사용자는 한국어를 선호함", contents)
            self.assertIn("프로젝트 규칙", contents)
            self.assertNotIn("빌더 메모", contents)
            self.assertGreaterEqual(bundle.memory_revision, 1)


class ConversationQueueTests(unittest.TestCase):
    def test_repository_validator_forwards_and_preserves_cancellation(self):
        with temporary_directory() as directory:
            repository = Path(directory) / "repository"
            (repository / ".git").mkdir(parents=True)
            cancelled = lambda: False
            validator = LocalGitRepositoryValidator(sandbox=object())

            with patch(
                "app.gateway.core.repository.inspect_repository_identity",
                side_effect=RepositoryCancelled("test cancellation"),
            ) as inspect:
                with self.assertRaises(RepositoryCancelled):
                    validator.validate(str(repository), cancelled=cancelled)

            self.assertIs(inspect.call_args.kwargs["cancelled"], cancelled)

    def test_worker_cancellation_during_repository_validation_is_terminal(self):
        class PlanningBackend:
            def __init__(self):
                self.calls = 0

            def respond(self, _state, _context, _message):
                self.calls += 1
                return AgentReply(
                    "한 단계 계획을 만들었습니다.",
                    stages=(
                        ProposedStage(
                            objective="기능 구현",
                            scope=("app/",),
                            acceptance_criteria=("테스트 통과",),
                            verification_commands=("py -3 -m unittest",),
                        ),
                    ),
                )

        class BlockingRepositoryValidator:
            def __init__(self, path: Path):
                self.path = path
                self.block = False
                self.entered = threading.Event()
                self.received_cancelled = None

            def validate(self, _raw_path, *, cancelled=None):
                if not self.block:
                    return RepositoryInfo(
                        self.path,
                        "main",
                        "a" * 64,
                        "b" * 40,
                    )
                self.received_cancelled = cancelled
                self.entered.set()
                while cancelled is not None and not cancelled():
                    threading.Event().wait(0.01)
                if cancelled is None:
                    raise AssertionError("repository validation did not receive cancellation")
                raise RepositoryCancelled("test cancellation")

        with temporary_directory() as directory:
            root = Path(directory)
            backend = PlanningBackend()
            validator = BlockingRepositoryValidator(root / "selected-repository")
            store, application = build_application(
                root, backend, repository_validator=validator
            )

            application.handle(incoming(1, "로그인 기능을 만들어줘"))
            application.handle(incoming(2, "D:\\projects\\sample"))
            application.handle(incoming(3, "이 프로젝트 사용 승인해"))
            self.assertEqual(1, backend.calls)
            pending_before = store.deliverable_outbound("telegram")

            queue = ConversationQueue(store)
            application.conversation_scheduler = queue
            application.router.conversation_scheduler = queue
            worker = ConversationWorker(store, application.router, poll_seconds=0.01)
            validator.block = True

            self.assertEqual(
                (), application.handle(incoming(4, "계획 수정: 근거를 추가해줘"))
            )
            thread = threading.Thread(target=worker.run_once)
            thread.start()
            self.assertTrue(validator.entered.wait(timeout=1))

            self.assertIn(queue.cancel("telegram", "200"), {"CANCEL_REQUESTED", "CANCELLED"})
            thread.join(timeout=2)

            self.assertFalse(thread.is_alive())
            self.assertIsNotNone(validator.received_cancelled)
            self.assertEqual("CANCELLED", queue.status("telegram", "200"))
            self.assertEqual(1, backend.calls)
            self.assertEqual(pending_before, store.deliverable_outbound("telegram"))

    def test_status_and_stop_remain_immediate_while_first_message_is_queued(self):
        with temporary_directory() as directory:
            store, application = build_application(Path(directory), CountingBackend())
            queue = ConversationQueue(store)
            application.conversation_scheduler = queue
            application.router.conversation_scheduler = queue

            self.assertEqual((), application.handle(incoming(1, "안녕")))
            status = application.handle(incoming(2, "상태"))
            stopped = application.handle(incoming(3, "중지"))

            self.assertIn("QUEUED", status[0].text)
            self.assertIn("중지", stopped[0].text)
            self.assertEqual("CANCELLED", queue.status("telegram", "200"))

    def test_new_work_cancels_queued_messages_before_they_reach_the_new_task(self):
        with temporary_directory() as directory:
            store, application = build_application(Path(directory), CountingBackend())
            queue = ConversationQueue(store)
            application.conversation_scheduler = queue
            application.router.conversation_scheduler = queue
            worker = ConversationWorker(store, application.router)

            self.assertEqual(
                (), application.handle(incoming(1, "이전 대화의 요구사항"))
            )
            new_work = application.handle(incoming(2, "새 작업"))

            self.assertIn("새 작업 대화", new_work[0].text)
            self.assertEqual("CANCELLED", queue.status("telegram", "200"))
            self.assertFalse(worker.run_once())
            binding = store.load_conversation("telegram", "200")
            self.assertIsNotNone(binding)
            assert binding is not None
            contents = [
                item["content"]
                for item in store.list_messages(binding["active_task_id"])
            ]
            self.assertNotIn("이전 대화의 요구사항", contents)

    def test_application_persists_normal_messages_for_background_processing(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store, application = build_application(root, CountingBackend())
            queue = ConversationQueue(store)
            application.conversation_scheduler = queue
            application.router.conversation_scheduler = queue
            activity = []
            worker = ConversationWorker(
                store,
                application.router,
                activity_notifier=lambda channel, conversation: activity.append(
                    (channel, conversation)
                ),
            )

            self.assertEqual((), application.handle(incoming(1, "안녕")))
            self.assertEqual(
                "COMPLETED",
                store.inbound_receipt("telegram", "200:foundation-1")["status"],
            )
            self.assertEqual("QUEUED", queue.status("telegram", "200"))

            self.assertTrue(worker.run_once())
            self.assertEqual([("telegram", "200")], activity)
            self.assertEqual("COMPLETED", queue.status("telegram", "200"))
            pending = store.deliverable_outbound("telegram")
            self.assertEqual(1, len(pending))
            self.assertIn("[빌더]", pending[0]["text"])

    def test_one_conversation_is_serialized_while_another_can_run(self):
        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            queue = ConversationQueue(store)
            first = queue.enqueue(incoming(1, "첫 번째", conversation="200"))
            second = queue.enqueue(incoming(2, "두 번째", conversation="200"))
            other = queue.enqueue(incoming(3, "다른 대화", conversation="201"))

            claimed_first = store.claim_next_conversation_job("worker-a")
            claimed_other = store.claim_next_conversation_job("worker-b")

            self.assertEqual(first["job_id"], claimed_first["job_id"])
            self.assertEqual(other["job_id"], claimed_other["job_id"])
            self.assertEqual("QUEUED", store.conversation_job(second["job_id"])["status"])

            store.finish_conversation_job(
                claimed_first["job_id"], "worker-a", "COMPLETED"
            )
            claimed_second = store.claim_next_conversation_job("worker-a")
            self.assertEqual(second["job_id"], claimed_second["job_id"])

    def test_worker_pool_processes_different_conversations_in_parallel(self):
        class BlockingRouter:
            def __init__(self):
                self.started = threading.Event()
                self.release = threading.Event()
                self._lock = threading.Lock()
                self.calls = 0

            def route(self, _message, *, cancelled=None):
                with self._lock:
                    self.calls += 1
                    if self.calls == 2:
                        self.started.set()
                self.release.wait(timeout=2)
                return ()

            def refresh_conversation_progress(self, *_args):
                return None

            def cancel_conversation_progress(self, _message):
                return None

            def complete_conversation_progress(self, _message):
                return None

            def fail_conversation_progress(self, *_args):
                return None

        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            queue = ConversationQueue(store)
            queue.enqueue(incoming(1, "첫 대화", conversation="200"))
            queue.enqueue(incoming(2, "둘째 대화", conversation="201"))
            router = BlockingRouter()
            worker = ConversationWorker(
                store, router, worker_count=2, poll_seconds=0.01
            )
            worker.start()
            try:
                self.assertTrue(router.started.wait(timeout=1))
                self.assertEqual(2, router.calls)
            finally:
                router.release.set()
                worker.stop()
                worker.join(timeout=2)

            self.assertFalse(worker.is_alive())
            self.assertEqual("COMPLETED", queue.status("telegram", "200"))
            self.assertEqual("COMPLETED", queue.status("telegram", "201"))

    def test_processing_conversation_can_receive_a_cancel_request(self):
        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            queue = ConversationQueue(store)
            queued = queue.enqueue(incoming(1, "긴 대화"))
            store.claim_next_conversation_job("worker-a")

            self.assertEqual("CANCEL_REQUESTED", queue.cancel("telegram", "200"))
            self.assertTrue(
                store.conversation_job_cancel_requested(queued["job_id"])
            )

    def test_running_conversation_cancellation_is_terminal_without_duplicate_reply(self):
        class BlockingRouter:
            def __init__(self):
                self.entered = threading.Event()
                self.cancelled_progress = 0
                self.failed_progress = 0

            def route(self, _message, *, cancelled=None):
                self.entered.set()
                while cancelled is not None and not cancelled():
                    threading.Event().wait(0.01)
                raise ConversationCancelled("test cancellation")

            def refresh_conversation_progress(self, *_args):
                return None

            def cancel_conversation_progress(self, _message):
                self.cancelled_progress += 1

            def complete_conversation_progress(self, _message):
                return None

            def fail_conversation_progress(self, *_args):
                self.failed_progress += 1

        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            queue = ConversationQueue(store)
            queue.enqueue(incoming(1, "긴 대화"))
            router = BlockingRouter()
            worker = ConversationWorker(store, router, poll_seconds=0.01)
            thread = threading.Thread(target=worker.run_once)
            thread.start()
            self.assertTrue(router.entered.wait(timeout=1))

            self.assertIn(queue.cancel("telegram", "200"), {"CANCEL_REQUESTED", "CANCELLED"})
            thread.join(timeout=2)

            self.assertFalse(thread.is_alive())
            self.assertEqual("CANCELLED", queue.status("telegram", "200"))
            self.assertEqual(1, router.cancelled_progress)
            self.assertEqual(0, router.failed_progress)
            self.assertEqual([], store.deliverable_outbound("telegram"))

    def test_stale_worker_replays_a_durable_response_only_after_same_job_resume(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store, application = build_application(root, CountingBackend())
            queue = ConversationQueue(store)
            application.conversation_scheduler = queue
            application.router.conversation_scheduler = queue
            message = incoming(9, "안녕")
            queued = queue.enqueue(message)
            claimed = store.claim_next_conversation_job(
                "crashed-worker", lease_seconds=1
            )
            self.assertIsNotNone(claimed)
            application.router.route(message)
            with closing(sqlite3.connect(root / "state.db")) as connection:
                connection.execute(
                    "UPDATE conversation_jobs SET lease_until = ? WHERE job_id = ?",
                    ("2000-01-01T00:00:00+00:00", claimed["job_id"]),
                )
                connection.commit()

            events = []
            worker = ConversationWorker(
                store, application.router, error_sink=events.append
            )
            worker._recover_stale()

            self.assertEqual("NEEDS_ATTENTION", queue.status("telegram", "200"))
            self.assertEqual([], store.deliverable_outbound("telegram"))
            status = application.handle(incoming(10, "상태"))
            self.assertIn(f"job={queued['job_id']}", status[0].text)
            self.assertIn("시도=1", status[0].text)
            self.assertIn("worker stopped", status[0].text)

            resumed = application.handle(incoming(11, "재개"))
            self.assertIn(f"대화 작업 {queued['job_id']}", resumed[0].text)
            self.assertEqual("QUEUED", queue.status("telegram", "200"))
            self.assertTrue(worker.run_once())
            self.assertEqual("COMPLETED", queue.status("telegram", "200"))
            pending = store.deliverable_outbound("telegram")
            self.assertEqual(1, sum("[빌더]" in item["text"] for item in pending))
            self.assertTrue(any("job=" in event and "attempt=2" in event for event in events))

    def test_stale_worker_without_a_durable_response_stays_in_attention(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store, application = build_application(root, CountingBackend())
            queue = ConversationQueue(store)
            application.conversation_scheduler = queue
            application.router.conversation_scheduler = queue
            message = incoming(12, "중단된 요청")
            queued = queue.enqueue(message)
            claimed = store.claim_next_conversation_job(
                "crashed-worker", lease_seconds=1
            )
            self.assertIsNotNone(claimed)
            with closing(sqlite3.connect(root / "state.db")) as connection:
                connection.execute(
                    "UPDATE conversation_jobs SET lease_until = ? WHERE job_id = ?",
                    ("2000-01-01T00:00:00+00:00", queued["job_id"]),
                )
                connection.commit()

            ConversationWorker(store, application.router)._recover_stale()
            resumed = application.handle(incoming(13, "재개"))

            self.assertIn(f"대화 작업 {queued['job_id']}", resumed[0].text)
            self.assertIn("안전하게 재개할 수 없습니다", resumed[0].text)
            self.assertEqual("NEEDS_ATTENTION", queue.status("telegram", "200"))


if __name__ == "__main__":
    unittest.main()
