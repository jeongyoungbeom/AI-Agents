import json
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.agents.conversation_backend import HermesConversationBackend
from app.agents.parsing import InvalidAgentResponse, parse_team_conversation_reply
from app.contracts import RoleId, RunState, TokenUsage
from app.gateway.conversation_worker import ConversationQueue, ConversationWorker
from app.gateway.core import AgentReply, MemoryScope, MemoryUpdate, ProposedStage, TeamConversationRequest, TeamConversationBatchResult
from app.gateway.core.governed_backend import GovernedAgentBackend, GovernedTeamConversationBackend
from app.services.budget import BudgetExceeded, BudgetManager, BudgetPolicy
from app.services.context import ContextBundle
from app.services.hermes import HermesCancelled, HermesExecutionError
from app.services.repository import RepositoryToolBatch, RepositoryToolRequest, RepositoryToolResult
from app.storage import StateStore
from tests.gateway.support import build_application, temporary_directory, TEST_REPOSITORY_HEAD, TEST_REPOSITORY_IDENTITY
from tests.gateway.test_s03_context import Reader, Tools, incoming


CONTEXT = ContextBundle((), (), (), False, 0)


class Delegate:
    supports_output_token_limit = False

    def preflight(self, *_args):
        pass

    def __init__(self, response=None):
        self.calls = 0
        self.response = response or AgentReply("결과", usage=TokenUsage(2, 1))

    def prompt_token_upper_bound(self, *_args, **_kwargs):
        return 10

    def respond_as(self, *_args, **_kwargs):
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class S05AgentLoopTests(unittest.TestCase):
    def manager(self, root, **policy):
        store = StateStore(root / "state.db")
        state = store.create_run(RunState("RUN-S05"))
        return store, state, BudgetManager(BudgetPolicy(**policy), store)

    def test_completed_result_and_usage_are_reused_after_restart(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store, state, budget = self.manager(root, conversation_tokens=1000)
            reply = AgentReply("확인 질문", usage=TokenUsage(4, 2),
                needs_user_input=("어떤 범위인가요?",),
                stages=(ProposedStage("목표", ("src/",), ("조건",), ("python check.py",)),),
                memory_updates=(MemoryUpdate(MemoryScope.RUN, "기억"),))
            delegate = Delegate(reply)
            governed = GovernedTeamConversationBackend(delegate, budget, response_reserve_tokens=5)
            first = governed.respond_as(state, CONTEXT, incoming(1, "안녕"), RoleId.REVIEW)
            reopened = StateStore(root / "state.db")
            next_backend = GovernedTeamConversationBackend(delegate,
                BudgetManager(BudgetPolicy(conversation_tokens=1), reopened), response_reserve_tokens=5)
            cached = next_backend.respond_as(state, CONTEXT, incoming(1, "안녕"), RoleId.REVIEW)
            self.assertEqual(first.text, cached.text)
            self.assertEqual(first.stages, cached.stages)
            self.assertEqual(first.memory_updates, cached.memory_updates)
            self.assertEqual(first.needs_user_input, cached.needs_user_input)
            self.assertTrue(cached.metadata["cached_call"])
            self.assertEqual(1, delegate.calls)
            self.assertEqual(6, reopened.usage_total(state.run_id))
            self.assertEqual(0, reopened.reserved_token_total(state.run_id))

    def test_parallel_duplicate_call_cannot_execute_or_settle_twice(self):
        with temporary_directory() as directory:
            store, state, budget = self.manager(Path(directory), conversation_tokens=1000)
            entered, proceed = threading.Event(), threading.Event()

            class Blocking(Delegate):
                def respond_as(self, *args, **kwargs):
                    entered.set()
                    if not proceed.wait(3):
                        raise AssertionError("duplicate test did not release")
                    return super().respond_as(*args, **kwargs)

            delegate = Blocking()
            backend = GovernedTeamConversationBackend(delegate, budget, response_reserve_tokens=5)
            results = []
            thread = threading.Thread(target=lambda: results.append(
                backend.respond_as(state, CONTEXT, incoming(1, "안녕"), RoleId.REVIEW)))
            thread.start()
            self.assertTrue(entered.wait(3))
            try:
                with self.assertRaises(HermesExecutionError):
                    backend.respond_as(state, CONTEXT, incoming(1, "안녕"), RoleId.REVIEW)
            finally:
                proceed.set()
                thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(1, len(results))
            self.assertEqual(1, delegate.calls)
            self.assertEqual(3, store.usage_total(state.run_id))
            self.assertEqual(0, store.reserved_token_total(state.run_id))

    def test_overage_preserves_success_and_blocks_next_invocation(self):
        with temporary_directory() as directory:
            store, state, budget = self.manager(Path(directory), conversation_tokens=1000)
            delegate = Delegate(AgentReply("여러 내부 호출의 결과", usage=TokenUsage(80, 20)))
            backend = GovernedTeamConversationBackend(delegate, budget, response_reserve_tokens=5)
            reply = backend.respond_as(state, CONTEXT, incoming(1, "조사"), RoleId.REVIEW)
            self.assertEqual("여러 내부 호출의 결과", reply.text)
            self.assertTrue(reply.metadata["budget_stop"])
            self.assertEqual(100, store.usage_total(state.run_id))
            with self.assertRaises(BudgetExceeded):
                backend.respond_as(state, CONTEXT, incoming(2, "더 조사"), RoleId.REVIEW)
            self.assertEqual(1, delegate.calls)

    def test_multi_turn_invocation_estimate_and_actual_usage_have_same_unit(self):
        with temporary_directory() as directory:
            store, state, budget = self.manager(Path(directory), conversation_tokens=1000)

            class Multi(Delegate):
                supports_output_token_limit = True

                def invocation_token_estimate(self, input_tokens, output_tokens):
                    return 4 * (input_tokens + output_tokens)

                def respond(self, *_args, **kwargs):
                    self.calls += 1
                    self.cap = kwargs["max_output_tokens"]
                    self.reserved = store.reserved_token_total(state.run_id)
                    return AgentReply("4회 내부 실행 결과", usage=TokenUsage(40, 12))

            delegate = Multi()
            reply = GovernedAgentBackend(delegate, budget, response_reserve_tokens=5).respond(
                state, CONTEXT, incoming(1, "계획"))
            self.assertEqual(60, delegate.reserved)
            self.assertEqual(5, delegate.cap)
            self.assertEqual(52, store.usage_total(state.run_id))
            self.assertFalse(reply.metadata["budget_stop"])
            runner = SimpleNamespace(settings=SimpleNamespace(provider="openai-codex",
                roles={RoleId.DEVELOPMENT: SimpleNamespace(max_turns=160)}))
            real = HermesConversationBackend(SimpleNamespace(root=Path(directory)), runner, sandbox=object())
            self.assertEqual(120, real.invocation_token_estimate(10, 5))
            self.assertFalse(real.supports_output_token_limit)

    def test_failed_and_cancelled_calls_keep_reported_or_unknown_usage(self):
        for error, expected, anomaly in [
            (HermesExecutionError("provider", usage=TokenUsage(7, 2)), 9, False),
            (HermesCancelled("취소", usage=TokenUsage(8, 3)), 11, False),
            (HermesCancelled("취소"), 15, True),
            (HermesExecutionError("시작 실패", category="startup"), 0, False),
        ]:
            with self.subTest(error=error), temporary_directory() as directory:
                store, state, budget = self.manager(Path(directory), conversation_tokens=1000)
                backend = GovernedTeamConversationBackend(Delegate(error), budget, response_reserve_tokens=5)
                with self.assertRaises(type(error)):
                    backend.respond_as(state, CONTEXT, incoming(1, "조사"), RoleId.REVIEW)
                self.assertEqual(expected, store.usage_total(state.run_id))
                self.assertEqual(0, store.reserved_token_total(state.run_id))
                self.assertEqual(anomaly, store.has_budget_anomaly(state.run_id))

    def test_cancel_before_start_releases_reservation_without_call(self):
        with temporary_directory() as directory:
            store, state, budget = self.manager(Path(directory))
            delegate = Delegate()
            backend = GovernedTeamConversationBackend(delegate, budget, response_reserve_tokens=5)
            with self.assertRaises(HermesCancelled):
                backend.respond_as(state, CONTEXT, incoming(1, "조사"), RoleId.REVIEW, cancelled=lambda: True)
            self.assertEqual(0, delegate.calls)
            self.assertEqual(0, store.reserved_token_total(state.run_id))
            self.assertEqual(0, store.usage_total(state.run_id))

    def test_restart_recovery_settles_only_the_stale_message_once(self):
        with temporary_directory() as directory:
            root = Path(directory)
            delegate = Delegate()
            store, application = build_application(root, object(), team_backend=delegate)
            application.handle(incoming(1, "인사"))
            binding = store.load_conversation("telegram", "200")
            state = store.load_run(binding["run_id"])
            budget = BudgetManager(BudgetPolicy(conversation_tokens=1000), store)
            application.router.budget = budget
            governed = GovernedAgentBackend(delegate, budget, response_reserve_tokens=5)
            stale_message, live_message = incoming(2, "조사"), incoming(3, "다른 조사")
            request = TeamConversationRequest(RoleId.REVIEW, CONTEXT)
            prepared = governed._prepare(state, stale_message, request, governed._logical_id(state, stale_message, request), 1)
            other = governed._prepare(state, live_message, request, governed._logical_id(state, live_message, request), 1)
            ConversationQueue(store).enqueue(stale_message)
            store.claim_next_conversation_job("old-worker", lease_seconds=1)
            with store._connection() as connection:
                connection.execute("UPDATE conversation_jobs SET lease_until='2000-01-01T00:00:00+00:00'")
            worker = ConversationWorker(store, application.router)
            worker._recover_stale()
            worker._recover_stale()
            self.assertEqual(15, store.usage_total(state.run_id))
            self.assertEqual(15, store.reserved_token_total(state.run_id))
            self.assertEqual("INTERRUPTED", store.model_call(prepared[0])["status"])
            self.assertEqual("RUNNING", store.model_call(other[0])["status"])
            self.assertTrue(store.has_budget_anomaly(state.run_id))
            budget.release_reservation(other[1])

    def test_duplicate_settlement_and_release_do_not_duplicate_usage(self):
        with temporary_directory() as directory:
            store, state, budget = self.manager(Path(directory), conversation_tokens=100)
            reservation = budget.reserve(state.run_id, "stage-001", "review", "conversation", 20)
            for _ in range(2):
                budget.record_usage(state.run_id, "stage-001", "review", "conversation",
                                    TokenUsage(5, 2), reservation=reservation)
            budget.release_reservation(reservation)
            self.assertEqual(7, store.usage_total(state.run_id))
            self.assertEqual(0, store.reserved_token_total(state.run_id))

    def test_call_schema_upgrade_keeps_prior_usage_and_is_idempotent(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store, state, budget = self.manager(root)
            budget.record_usage(state.run_id, "stage-001", "review", "conversation", TokenUsage(5, 2))
            with store._connection() as connection:
                connection.execute("DROP TABLE model_calls")  # 격리 fixture의 S04 schema 상태
            upgraded = StateStore(root / "state.db")
            reopened = StateStore(root / "state.db")
            self.assertEqual(7, reopened.usage_total(state.run_id))
            self.assertEqual(7, reopened.load_run(state.run_id).total_tokens)
            self.assertIsNone(upgraded.completed_model_call("new-call"))

    def loop(self, root, team, tools=None):
        store, application = build_application(root, object(), team_backend=team,
                                               repository_reader=Reader(), repository_tools=tools or Tools())
        application.handle(incoming(1, "D:\\projects\\sample"))
        application.handle(incoming(2, "이 프로젝트 사용 승인해"))
        return store, application

    def test_one_role_connects_five_reads_without_consuming_consultation_limit(self):
        class Reads:
            def __init__(self):
                self.contexts = []

            def preflight(self, *_args):
                pass

            def respond_as(self, _state, context, _message, role, **_kwargs):
                self.contexts.append(context)
                index = len(self.contexts)
                if index <= 5:
                    return AgentReply("", repository_tools=(RepositoryToolRequest("read_file", path=f"src/{index}.py"),))
                return AgentReply("5개 파일의 관계를 연결했습니다.")

        with temporary_directory() as directory:
            team = Reads()
            store, application = self.loop(Path(directory), team)
            output = application.handle(incoming(3, "센티널아 프로젝트 설명해"))
            self.assertEqual(6, len(team.contexts))
            self.assertIn("5개 파일", output[0].text)
            self.assertEqual(5, len(team.contexts[-1].repository_context["tool_results"]))
            paths = {item["data"]["path"] for item in team.contexts[-1].evidence}
            self.assertTrue({f"src/{i}.py" for i in range(1, 6)} <= paths)
            self.assertEqual("success", output[0].metadata["request_result"]["outcome"])

    def test_repeat_and_empty_results_stop_with_partial_synthesis(self):
        class Repeats:
            def __init__(self, vary=False):
                self.purposes = []
                self.vary = vary
            def preflight(self, *_args):
                pass
            def respond_as(self, _state, context, _message, _role, **kwargs):
                self.purposes.append(kwargs["call_purpose"])
                if kwargs["call_purpose"] == "agent_loop_final":
                    self.reason = context.repository_context["agent_loop_stop_reason"]
                    return AgentReply("src/a.py:1 확인. 나머지는 미확인입니다.")
                path = f"src/{len(self.purposes)}.py" if self.vary else "src/a.py"
                return AgentReply("", repository_tools=(RepositoryToolRequest("read_file", path=path),))

        class EmptyTools(Tools):
            def execute(self, _path, requests, **_kwargs):
                return RepositoryToolBatch(TEST_REPOSITORY_IDENTITY, TEST_REPOSITORY_HEAD,
                    (RepositoryToolResult(requests[0], "denied", {}),))

        for tools, calls, vary in [(Tools(), 3, False), (EmptyTools(), 3, False), (EmptyTools(), 3, True)]:
            with self.subTest(tools=tools), temporary_directory() as directory:
                team = Repeats(vary)
                _, application = self.loop(Path(directory), team, tools)
                output = application.handle(incoming(3, "센티널아 프로젝트 설명해"))
                self.assertEqual(calls, len(team.purposes))
                self.assertEqual("agent_loop_final", team.purposes[-1])
                self.assertEqual("NO_PROGRESS", team.reason)
                self.assertEqual("partial", output[0].metadata["request_result"]["outcome"])
                self.assertIn("미확인", output[0].text)

    def test_round_and_elapsed_limits_reserve_a_final_answer(self):
        class Reads:
            calls = 0
            def preflight(self, *_args):
                pass
            def respond_as(self, _state, context, _message, _role, **kwargs):
                self.calls += 1
                if kwargs["call_purpose"] == "agent_loop_final":
                    self.reason = context.repository_context["agent_loop_stop_reason"]
                    return AgentReply("확인한 범위의 부분 답변")
                return AgentReply("", repository_tools=(RepositoryToolRequest("read_file", path=f"src/{self.calls}.py"),))

        with temporary_directory() as directory:
            team = Reads()
            _, application = self.loop(Path(directory), team)
            application.router.max_repository_tool_rounds = 1
            output = application.handle(incoming(3, "센티널아 프로젝트 설명해"))
            self.assertEqual("TOOL_ROUND_LIMIT", team.reason)
            self.assertEqual(3, team.calls)
            self.assertEqual("partial", output[0].metadata["request_result"]["outcome"])
        with temporary_directory() as directory:
            team = Reads()
            _, application = self.loop(Path(directory), team)
            application.router.max_agent_loop_seconds = 1
            ticks = iter(range(0, 1000, 2))
            with patch("app.gateway.core.conversation.time.monotonic", side_effect=lambda: next(ticks)):
                output = application.handle(incoming(3, "센티널아 프로젝트 설명해"))
            self.assertEqual("TIME_LIMIT", team.reason)
            self.assertEqual(1, team.calls)
            self.assertEqual("partial", output[0].metadata["request_result"]["outcome"])

    def test_user_question_stops_actions_and_records_waiting(self):
        class Question(Delegate):
            def preflight(self, *_args):
                pass
        with temporary_directory() as directory:
            delegate = Question(AgentReply("결정이 필요합니다.", needs_user_input=("대상은 어느 기능인가요?",)))
            _, application = build_application(Path(directory), object(), team_backend=delegate)
            output = application.handle(incoming(1, "안녕"))
            self.assertEqual("waiting_user", output[0].metadata["request_result"]["outcome"])
            self.assertIn("어느 기능", output[0].text)
            self.assertEqual(1, delegate.calls)
        with self.assertRaises(InvalidAgentResponse):
            parse_team_conversation_reply(json.dumps({"message":"질문", "needs_user_input":["범위?"],
                "repository_tools":[{"tool":"read_file","path":"src/a.py"}]}), RoleId.REVIEW)

    def test_budget_allowance_stops_exploration_and_keeps_final_result(self):
        class Reads(Delegate):
            def respond_as(self, _state, context, _message, _role, **kwargs):
                self.calls += 1
                if kwargs["call_purpose"] == "agent_loop_final":
                    return AgentReply("src/1.py:1까지만 확인한 부분 결과", usage=TokenUsage(10, 5))
                return AgentReply("", usage=TokenUsage(10, 5), repository_tools=(
                    RepositoryToolRequest("read_file", path=f"src/{self.calls}.py"),))
        with temporary_directory() as directory:
            delegate = Reads()
            store, application = self.loop(Path(directory), delegate)
            budget = BudgetManager(BudgetPolicy(conversation_tokens=40), store)
            application.router.budget = budget
            application.router.team_backend = GovernedTeamConversationBackend(delegate, budget, response_reserve_tokens=5)
            output = application.handle(incoming(3, "센티널아 프로젝트 설명해"))
            self.assertEqual(2, delegate.calls)
            self.assertTrue(any("부분 결과" in item.text for item in output))
            self.assertEqual("partial", output[0].metadata["request_result"]["outcome"])
            self.assertEqual(30, store.usage_total(store.load_conversation("telegram", "200")["run_id"]))

    def test_startup_retry_has_separate_attempt_and_usage_records(self):
        class StartsOnce(Delegate):
            def respond_as(self, *_args, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise HermesExecutionError("시작 실패", category="startup", retryable=True)
                return AgentReply("정상 결과", usage=TokenUsage(2, 1))
        with temporary_directory() as directory:
            store, state, budget = self.manager(Path(directory), conversation_tokens=1000,
                                               retries={"technical_error":1})
            delegate = StartsOnce()
            backend = GovernedTeamConversationBackend(delegate, budget, response_reserve_tokens=5)
            reply = backend.respond_as(state, CONTEXT, incoming(1, "조사"), RoleId.REVIEW)
            self.assertEqual("정상 결과", reply.text)
            self.assertEqual(2, delegate.calls)
            self.assertEqual(3, store.usage_total(state.run_id))
            self.assertEqual(1, store.retry_count(state.run_id, backend._stage_id(incoming(1,"조사")), "technical_error"))
            self.assertEqual(0, store.reserved_token_total(state.run_id))

    def test_planning_overage_keeps_answer_without_registering_a_new_plan(self):
        with temporary_directory() as directory:
            store, application = build_application(Path(directory), object())
            application.handle(incoming(1, "D:\\projects\\sample"))
            binding = store.load_conversation("telegram", "200")
            state = store.load_run(binding["run_id"])
            reply = AgentReply("보존할 계획 근거", metadata={"budget_stop":True},
                stages=(ProposedStage("목표", ("src/",), ("조건",), ("python check.py",)),))
            output = application.router._apply_backend_reply(incoming(2,"계획"), binding, state, reply)
            self.assertIn("보존할 계획 근거", output.text)
            self.assertEqual("partial", output.metadata["request_result"]["outcome"])
            self.assertEqual(0, store.load_run(state.run_id).plan_revision)

    def test_router_cached_result_is_not_counted_as_a_new_model_call(self):
        class Cached(Delegate):
            def respond_batch(self, *_args, **_kwargs):
                return TeamConversationBatchResult((AgentReply("저장된 결과", metadata={"cached_call":True}),), 0)
        with temporary_directory() as directory:
            _, application = build_application(Path(directory), object(), team_backend=Cached())
            output = application.handle(incoming(1,"안녕"))
            result = output[0].metadata["request_result"]
            self.assertEqual("success", result["outcome"])
            self.assertEqual(0, result["attempts"])
            self.assertEqual(0, result["successes"])


if __name__ == "__main__":
    unittest.main()
