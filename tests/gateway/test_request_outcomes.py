from __future__ import annotations

import sqlite3
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from app.contracts import RoleId
from app.contracts.outcomes import RequestOutcome, RequestResult
from app.gateway.conversation_worker import ConversationQueue, ConversationWorker
from app.gateway.core import AgentCallRequest, AgentReply, RepositoryInfo
from app.gateway.repository_analysis_worker import RepositoryAnalysisWorker
from app.services.budget import BudgetExceeded
from app.services.context import ContextService
from app.services.repository import RepositoryAccessError, RepositoryAnalysisRequest, RepositoryAnalysisStatus
from app.services.toolchains import ToolchainPreflightError
from app.storage import StateStore, StoreError
from tests.gateway.support import build_application, future_expiry, temporary_directory
from tests.gateway.test_conversation_foundation import incoming
from tests.gateway.test_conversation import ReadyBackend


class Team:
    def __init__(self, replies=(), preflight_error=None):
        self.replies = list(replies)
        self.preflight_error = preflight_error
        self.calls = 0

    def preflight(self, *_args):
        if self.preflight_error:
            raise self.preflight_error

    def respond_as(self, *_args, **_kwargs):
        self.calls += 1
        reply = self.replies.pop(0) if self.replies else AgentReply("정상 답변")
        if isinstance(reply, Exception):
            raise reply
        return reply


class RequestOutcomeTests(unittest.TestCase):

    def test_denied_owner_request_is_persisted_as_failed_by_worker(self):
        with temporary_directory() as directory:
            store, app = build_application(Path(directory), None, team_backend=Team())
            binding = app.router._create_binding(incoming(0, "fixture"))
            denied = replace(incoming(1, "안녕"), user_id="101")
            queue = ConversationQueue(store)
            queue.enqueue(denied)
            self.assertTrue(ConversationWorker(store, app.router).run_once())
            expected = RequestResult(RequestOutcome.FAILED, "ACCESS_DENIED")
            self.assertEqual("COMPLETED", queue.summary("telegram", "200")["status"])
            self.assertEqual(expected.to_dict(), queue.summary("telegram", "200")["result"])
            self.assertEqual(expected, store.conversation_result("telegram", "200", denied.external_message_id))
            self.assertEqual(binding, store.load_conversation("telegram", "200"))
            self.assertEqual(0, app.router.team_backend.calls)
            self.assertIn("작업 소유자", store.deliverable_outbound("telegram")[-1]["text"])

    def test_planning_repository_boundaries_have_explicit_outcomes(self):
        cases = (
            ("missing_approval", RequestOutcome.WAITING_USER, "PROJECT_APPROVAL"),
            ("expired_approval", RequestOutcome.WAITING_USER, "PROJECT_APPROVAL"),
            ("validation_failed", RequestOutcome.FAILED, "REPOSITORY_UNAVAILABLE"),
            ("identity_changed", RequestOutcome.WAITING_USER, "PROJECT_APPROVAL"),
        )
        for boundary, outcome, reason in cases:
            with self.subTest(boundary=boundary), temporary_directory() as directory:
                backend = ReadyBackend()
                store, app = build_application(Path(directory), backend)
                app.router.route_result(incoming(1, "기능 개발"))
                app.router.route_result(incoming(2, r"D:\projects\sample"))
                binding = store.load_conversation("telegram", "200")
                state = store.load_run(binding["run_id"])
                store.save_run(replace(state, repository_approved=True))
                if boundary != "missing_approval":
                    store.approve_repository("telegram", "200", "100", state.repository, state.repository_identity, future_expiry())
                    if boundary == "expired_approval":
                        with store._connection() as connection:
                            connection.execute("UPDATE scoped_repository_approvals SET expires_at = ?", ("2000-01-01T00:00:00+00:00",))
                repository = RepositoryInfo(Path(state.repository), "main", "c" * 64, state.repository_head_sha)
                with patch.object(
                    app.router, "_validate_repository",
                    side_effect=ValueError("fixture validation failed") if boundary == "validation_failed" else None,
                    return_value=repository,
                ):
                    message = incoming(3, "계획을 설명해줘")
                    routed = app.router.route_result(message)
                expected = RequestResult(outcome, reason)
                self.assertEqual(expected, routed.result)
                self.assertEqual(expected.to_dict(), routed.messages[0].metadata["request_result"])
                self.assertEqual(expected, store.conversation_result("telegram", "200", message.external_message_id))
                self.assertEqual(0, backend.calls)

    def test_planning_approval_errors_and_waits_are_not_success(self):
        cases = (
            ("read_validation", "이 프로젝트 사용 승인해", RequestOutcome.FAILED, "REPOSITORY_UNAVAILABLE"),
            ("execution_validation", "개발 시작해", RequestOutcome.FAILED, "REPOSITORY_UNAVAILABLE"),
            ("read_identity", "이 프로젝트 사용 승인해", RequestOutcome.WAITING_USER, "PROJECT_APPROVAL"),
            ("execution_identity", "개발 시작해", RequestOutcome.WAITING_USER, "PLAN_REVISION_REQUIRED"),
            ("expired", "개발 시작해", RequestOutcome.WAITING_USER, "PROJECT_APPROVAL"),
            ("preflight", "개발 시작해", RequestOutcome.FAILED, "EXECUTION_PREFLIGHT_FAILED"),
            ("pipeline_missing", "개발 시작해", RequestOutcome.FAILED, "PIPELINE_UNAVAILABLE"),
            ("read_renewed", "이 프로젝트 사용 승인해", RequestOutcome.WAITING_USER, "EXECUTION_APPROVAL"),
            ("reminder", "승인합니다", RequestOutcome.WAITING_USER, "EXECUTION_APPROVAL"),
        )
        for boundary, text, outcome, reason in cases:
            with self.subTest(boundary=boundary), temporary_directory() as directory:
                backend = ReadyBackend()
                store, app = build_application(Path(directory), backend)
                for identifier, setup in enumerate(("기능 개발", r"D:\projects\sample", "이 프로젝트 사용 승인해"), 1):
                    app.router.route_result(incoming(identifier, setup))
                state = store.load_run(store.load_conversation("telegram", "200")["run_id"])
                if boundary == "expired":
                    with store._connection() as connection:
                        connection.execute("UPDATE scoped_repository_approvals SET expires_at = ?", ("2000-01-01T00:00:00+00:00",))
                repository = RepositoryInfo(Path(state.repository), "main", "c" * 64 if "identity" in boundary else state.repository_identity, state.repository_head_sha)
                with patch.object(
                    app.router, "_validate_repository", return_value=repository,
                    side_effect=ValueError("fixture validation failed") if "validation" in boundary else None,
                ), patch.object(
                    app.router, "_preflight_execution_plan",
                    side_effect=ToolchainPreflightError("tool", "fixture preflight failed") if boundary == "preflight" else None,
                ):
                    message = incoming(4, text)
                    routed = app.router.route_result(message)
                expected = RequestResult(outcome, reason)
                self.assertEqual(expected, routed.result)
                self.assertEqual(expected, store.conversation_result("telegram", "200", message.external_message_id))
                self.assertEqual(1, backend.calls)
                self.assertEqual(boundary == "pipeline_missing", store.load_run(state.run_id).approval_granted)

    def test_control_response_success_does_not_mean_execution_success(self):
        with temporary_directory() as directory:
            backend = ReadyBackend()
            store, app = build_application(Path(directory), backend)
            for identifier, setup in enumerate(("기능 개발", r"D:\projects\sample", "이 프로젝트 사용 승인해"), 1):
                app.router.route_result(incoming(identifier, setup))
            for identifier, text in enumerate(("도움말", "상태", "이 검증 명령은 왜 필요한 거야?"), 4):
                with self.subTest(text=text):
                    message = incoming(identifier, text)
                    result = app.router.route_result(message).result
                    self.assertEqual(RequestResult(RequestOutcome.SUCCESS, "CONTROL_RESPONSE"), result)
                    self.assertEqual(result, store.conversation_result("telegram", "200", message.external_message_id))
            self.assertEqual(1, backend.calls)
            self.assertFalse(store.load_run(store.load_conversation("telegram", "200")["run_id"]).approval_granted)

    def test_queued_stop_and_replacement_persist_cancelled_results(self):
        for control in ("중지", "새 작업"):
            with self.subTest(control=control), temporary_directory() as directory:
                root = Path(directory)
                store, app = build_application(root, None, team_backend=Team())
                queue = ConversationQueue(store)
                app.conversation_scheduler = app.router.conversation_scheduler = queue
                first, second = incoming(1, "첫 요청"), incoming(2, "두 번째 요청")
                app.handle(first)
                app.handle(second)
                app.handle(incoming(3, control))
                self.assertEqual("CANCELLED", queue.status("telegram", "200"))
                expected = RequestResult(RequestOutcome.CANCELLED, "USER_CANCELLED")
                for message in (first, second):
                    self.assertEqual(expected, store.conversation_result("telegram", "200", message.external_message_id))
                queue.cancel("telegram", "200")
                reopened = StateStore(root / "state.db")
                self.assertEqual(expected.to_dict(), ConversationQueue(reopened).summary("telegram", "200")["result"])
                self.assertIn("요청 결과=취소됨", app.handle(incoming(4, "상태"))[0].text)
                self.assertFalse(ConversationWorker(store, app.router).run_once())
                self.assertEqual(0, app.router.team_backend.calls)

    def test_queued_cancellation_preserves_cached_counts_and_other_conversations(self):
        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            queue = ConversationQueue(store)
            first, other = incoming(1, "첫 요청"), incoming(2, "다른 요청", conversation="201")
            queue.enqueue(first)
            queue.enqueue(other)
            cached = RequestResult(RequestOutcome.PARTIAL, "INTERRUPTED", 3, 2, 5)
            store.save_conversation_result("telegram", "200", first.external_message_id, cached)
            queue.cancel("telegram", "200")
            expected = replace(cached, outcome=RequestOutcome.CANCELLED, reason="USER_CANCELLED")
            self.assertEqual(expected, store.conversation_result("telegram", "200", first.external_message_id))
            self.assertEqual("QUEUED", queue.status("telegram", "201"))
            self.assertIsNone(queue.summary("telegram", "201")["result"])
            self.assertEqual("201", store.claim_next_conversation_job("worker")["conversation_id"])

    def test_cancel_result_failure_rolls_back_all_job_transitions(self):
        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            queue = ConversationQueue(store)
            running = incoming(1, "진행 요청")
            queue.enqueue(running)
            claimed = store.claim_next_conversation_job("worker")
            queued = (incoming(2, "대기 요청"), incoming(3, "추가 대기 요청"))
            for message in queued:
                queue.enqueue(message)
            original = store.save_conversation_result
            saves = 0

            def fail_second_save(*args):
                nonlocal saves
                original(*args)
                saves += 1
                if saves == 2:
                    raise sqlite3.OperationalError("fixture result write failed")

            with patch.object(store, "save_conversation_result", side_effect=fail_second_save):
                with self.assertRaises(sqlite3.OperationalError):
                    queue.cancel("telegram", "200")
            self.assertEqual(2, saves)
            self.assertFalse(store.conversation_job_cancel_requested(claimed["job_id"]))
            self.assertEqual("QUEUED", queue.status("telegram", "200"))
            for message in (running, *queued):
                self.assertIsNone(store.conversation_result("telegram", "200", message.external_message_id))
            queue.cancel("telegram", "200")
            self.assertTrue(store.conversation_job_cancel_requested(claimed["job_id"]))
            self.assertIsNone(store.conversation_result("telegram", "200", running.external_message_id))
            for message in queued:
                self.assertEqual(RequestOutcome.CANCELLED, store.conversation_result("telegram", "200", message.external_message_id).outcome)

    def test_queue_result_and_card_follow_model_results_not_notice_count(self):
        cases = (
            ("안녕", [AgentReply("정상 답변")], "success", 1, 1, "완료"),
            ("안녕", [RuntimeError("model failed")], "failed", 1, 0, "실패"),
            ("얘들아 의견 줘", [AgentReply("첫 답변"), RuntimeError("failed"), AgentReply("세 번째 답변")],
             "partial", 3, 2, "부분 완료"),
            ("얘들아 의견 줘", [AgentReply("첫 답변"), BudgetExceeded("limit"), AgentReply("세 번째 답변")],
             "partial", 3, 2, "부분 완료"),
        )
        for text, replies, outcome, attempts, successes, label in cases:
            with self.subTest(outcome=outcome, replies=replies), temporary_directory() as directory:
                store, app = build_application(Path(directory), None, team_backend=Team(replies))
                queue = ConversationQueue(store)
                app.router.set_progress_notifier(lambda: None)
                queue.enqueue(incoming(1, text))
                self.assertTrue(ConversationWorker(store, app.router).run_once())
                summary = queue.summary("telegram", "200")
                self.assertEqual("COMPLETED", summary["status"])
                self.assertEqual(outcome, summary["result"]["outcome"])
                self.assertEqual((attempts, successes), (summary["result"]["attempts"], summary["result"]["successes"]))
                self.assertIsNone(summary["result"]["provider_requests"])
                cards = [item for item in store.deliverable_outbound("telegram") if item["delivery_mode"] == "edit"]
                self.assertIn(f"· {label}]", cards[-1]["text"])
                self.assertNotIn("4/4", cards[-1]["text"])
                self.assertIn("provider 요청 미확인", cards[-1]["text"])

    def test_budget_preflight_is_failed_with_no_model_calls(self):
        with temporary_directory() as directory:
            team = Team(preflight_error=BudgetExceeded("limit"))
            store, app = build_application(Path(directory), None, team_backend=team)
            app.router.set_progress_notifier(lambda: None)
            routed = app.router.route_result(incoming(1, "안녕"))
            self.assertEqual(RequestResult(RequestOutcome.FAILED, "BUDGET_LIMIT"), routed.result)
            self.assertEqual(0, team.calls)
            self.assertIn("· 실패]", store.deliverable_outbound("telegram")[-1]["text"])

    def test_call_limit_reports_partial_even_with_a_valid_answer(self):
        with temporary_directory() as directory:
            team = Team([AgentReply("확인한 내용", calls=(
                AgentCallRequest(RoleId.DEVELOPMENT, RoleId.REVIEW, "테스트 관점 확인"),
            ))])
            _, app = build_application(Path(directory), None, team_backend=team, max_auto_agent_replies=1)
            routed = app.router.route_result(incoming(1, "필요하면 센티널에게 물어봐"))
            self.assertEqual(RequestResult(RequestOutcome.PARTIAL, "CALL_LIMIT", 1, 1), routed.result)

    def test_missing_project_is_waiting_user(self):
        with temporary_directory() as directory:
            store, app = build_application(Path(directory), None)
            message = incoming(1, "로그인 기능을 만들어줘")
            routed = app.router.route_result(message)
            self.assertEqual(RequestOutcome.WAITING_USER, routed.result.outcome)
            self.assertEqual(routed.result, store.conversation_result("telegram", "200", message.external_message_id))

    def test_repository_read_and_project_selection_failures_are_not_success(self):
        class Reader:
            @staticmethod
            def should_inspect(_text):
                return True

            def inspect(self, *_args, **_kwargs):
                raise RepositoryAccessError("fixture read failed")

        with temporary_directory() as directory:
            root = Path(directory)
            store, app = build_application(root, None, team_backend=Team())
            app.router._create_binding(incoming(0, "fixture"))
            app.router.repository_reader = Reader()
            app.router.set_progress_notifier(lambda: None)
            store.set_current_project(
                "telegram", "200", "100", str(root / "selected-repository"),
                repository_identity="a" * 64, head_sha="b" * 40,
                approved=True, approval_expires_at=future_expiry(),
            )
            result = app.router.route_result(incoming(1, "프로젝트 설명해줘"))
            self.assertEqual(RequestOutcome.FAILED, result.result.outcome)
            self.assertEqual(0, app.router.team_backend.calls)
            self.assertIn("· 실패]", store.deliverable_outbound("telegram")[-1]["text"])
            result = app.router.route_result(incoming(2, "D:\\invalid\\repository"))
            self.assertEqual(RequestOutcome.FAILED, result.result.outcome)

    def test_cancellation_between_result_and_job_commit_removes_answer(self):
        with temporary_directory() as directory:
            store, app = build_application(Path(directory), None, team_backend=Team())
            app.router.set_progress_notifier(lambda: None)
            queue = ConversationQueue(store)
            queue.enqueue(incoming(1, "안녕"))

            def prepare(message):
                queue.cancel("telegram", "200")
                return (message,)

            original_cancel = app.router.cancel_conversation_progress

            def cancel_progress(message):
                self.assertIsNone(getattr(store._local, "connection", None))
                original_cancel(message)

            with patch.object(app.router, "cancel_conversation_progress", side_effect=cancel_progress):
                ConversationWorker(store, app.router, message_preparer=prepare).run_once()
            self.assertEqual("CANCELLED", queue.summary("telegram", "200")["status"])
            self.assertEqual("cancelled", queue.summary("telegram", "200")["result"]["outcome"])
            pending = store.deliverable_outbound("telegram")
            self.assertFalse(any("정상 답변" in item["text"] for item in pending))
            self.assertIn("· 취소됨]", pending[-1]["text"])

    def test_late_progress_cannot_replace_terminal_or_a_new_request(self):
        with temporary_directory() as directory:
            store, app = build_application(Path(directory), None, team_backend=Team())
            router = app.router
            router.set_progress_notifier(lambda: None)
            first = incoming(1, "안녕")
            router.route_result(first)
            old = router._active_progress[("telegram", "200")].progress
            before = store.deliverable_outbound("telegram")
            router._update_conversation_progress(first, old, completed_steps=1, active_step=2, detail="오래된 진행")
            router._update_conversation_progress(first, old, completed_steps=1, active_step=2, detail="오래된 실패", outcome="failed")
            self.assertEqual(before, store.deliverable_outbound("telegram"))
            router.complete_conversation_progress(first)
            router._update_conversation_progress(first, old, completed_steps=4, active_step=4, detail="오래된 완료", outcome="success")
            self.assertEqual(before, store.deliverable_outbound("telegram"))
            second = incoming(2, "다음 질문")
            router._start_conversation_progress(second, (RoleId.DEVELOPMENT,), repository_required=False)
            router.cancel_conversation_progress(first)
            router.complete_conversation_progress(first)
            self.assertEqual(second.external_message_id, router._active_progress[("telegram", "200")].progress.reply_to)

    def test_outbound_retry_reuses_answer_and_discards_older_edits(self):
        with temporary_directory() as directory:
            store, app = build_application(Path(directory), None, team_backend=Team())
            queue = ConversationQueue(store)
            queue.enqueue(incoming(1, "안녕"))
            worker = ConversationWorker(store, app.router)
            worker.run_once()
            first = store.claim_next_outbound("telegram", "sender")
            store.fail_outbound(first["outbound_id"], "temporary", lease_owner="sender")
            retried = store.claim_next_outbound("telegram", "sender")
            self.assertEqual(first["outbound_id"], retried["outbound_id"])
            store.complete_outbound(retried["outbound_id"], ("10",), lease_owner="sender")
            self.assertFalse(worker.run_once())
            self.assertEqual(1, app.router.team_backend.calls)
            target = store.queue_outbound("telegram", "200", "진행 카드")
            older = store.queue_outbound("telegram", "200", "과거 갱신", delivery_mode="edit", target_outbound_id=target)
            store.complete_outbound(target, ("11",))
            store.fail_outbound(older, "temporary")
            newer = store.queue_outbound("telegram", "200", "최신 실패", delivery_mode="edit", target_outbound_id=target)
            claimed = store.claim_next_outbound("telegram", "sender")
            self.assertEqual(newer, claimed["outbound_id"])
            self.assertEqual("DEAD", store.outbound_record(older)["status"])

    def analysis(self, store, root):
        request = RepositoryAnalysisRequest.create(
            channel="telegram", conversation_id="200", user_id="100", source_message_id="1",
            role_id="development", request_text="저장소 전반 분석", repository_path=str(root),
            repository_identity="a" * 64, commit_sha="b" * 40, branch="main",
        )
        store.create_repository_analysis(request)
        store.claim_next_repository_analysis("worker")
        store.queue_repository_analysis_progress(request.analysis_id, "worker", "[장기 저장소 분석 · 진행 중]")
        return request.analysis_id

    def test_analysis_final_card_uses_outcome_and_does_not_duplicate_final(self):
        cases = (
            (RepositoryAnalysisStatus.COMPLETED, "COMPLETED", "success", "완료"),
            (RepositoryAnalysisStatus.PARTIAL_COMPLETED, "BUDGET_LIMIT", "partial", "부분 완료"),
            (RepositoryAnalysisStatus.NEEDS_ATTENTION, "MODEL_OUTCOME_UNKNOWN", "failed", "실패"),
            (RepositoryAnalysisStatus.NEEDS_ATTENTION, "SNAPSHOT_UNAVAILABLE", "waiting_user", "사용자 대기"),
            (RepositoryAnalysisStatus.CANCELLED, "USER_STOPPED", "cancelled", "취소됨"),
        )
        for status, reason, outcome, label in cases:
            with self.subTest(status=status), temporary_directory() as directory:
                root = Path(directory)
                store = StateStore(root / "state.db")
                analysis_id = self.analysis(store, root)
                store.record_repository_analysis_call(analysis_id, "worker")
                job = store.finish_repository_analysis_with_response(analysis_id, "worker", status, "최종 결과", reason=reason)
                self.assertEqual(outcome, job["result"]["outcome"])
                self.assertEqual((1, 0), (job["model_attempts"], job["model_successes"]))
                self.assertEqual("PENDING", job["delivery_status"])
                with self.assertRaises(StoreError):
                    store.finish_repository_analysis_with_response(analysis_id, "worker", status, "중복 결과", reason=reason)
                pending = store.deliverable_outbound("telegram")
                self.assertEqual(1, sum(item["text"] == "최종 결과" for item in pending))
                self.assertIn(f"· {label}]", next(item["text"] for item in pending if item["delivery_mode"] == "edit"))
                store.complete_outbound(job["final_outbound_id"], ("20",))
                self.assertEqual("DELIVERED", store.repository_analysis(analysis_id)["delivery_status"])
                self.assertEqual(outcome, store.repository_analysis(analysis_id)["result"]["outcome"])

    def test_analysis_failed_model_attempt_is_persisted_before_checkpoint(self):
        from app.contracts import RunState
        from app.services.repository import RepositoryAnalysisPhase
        import threading
        with temporary_directory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.db")
            analysis_id = self.analysis(store, root)
            store.create_run(RunState(analysis_id))
            worker = RepositoryAnalysisWorker(store, object(), Team([RuntimeError("model failed")]), ContextService(store))
            worker.instance_id = "worker"
            job = store.repository_analysis(analysis_id)
            with self.assertRaises(RuntimeError):
                worker._ask_model(job, RepositoryAnalysisPhase.STRUCTURE, (("app/a.py", "code"),), threading.Event())
            saved = store.repository_analysis(analysis_id)
            self.assertEqual((1, 0, 0), (saved["model_attempts"], saved["model_successes"], saved["checkpoint"]))

    def test_v8_database_copy_migrates_without_losing_messages(self):
        from app.contracts import RunState
        with temporary_directory() as directory:
            root = Path(directory)
            original = root / "v8.db"
            store = StateStore(original)
            store.create_run(RunState("existing"))
            ContextService(store).add_message("existing", "user", "기존 기록")
            legacy_analysis = self.analysis(store, root)
            with closing(sqlite3.connect(original)) as connection:
                connection.execute("DROP TABLE conversation_request_results")
                connection.execute("ALTER TABLE repository_analysis_jobs DROP COLUMN model_attempts")
                connection.execute("ALTER TABLE repository_analysis_jobs DROP COLUMN model_successes")
                connection.execute("DELETE FROM schema_migrations WHERE version = 9")
                connection.commit()
                with closing(sqlite3.connect(root / "copy.db")) as backup:
                    connection.backup(backup)
            class FailingConnection(sqlite3.Connection):
                def execute(self, sql, *args):
                    if "ADD COLUMN model_successes" in sql:
                        raise sqlite3.OperationalError("fixture migration failure")
                    return super().execute(sql, *args)

            original_connect = sqlite3.connect
            with patch("app.storage.sqlite_store.sqlite3.connect", side_effect=lambda *args, **kwargs: original_connect(*args, **kwargs, factory=FailingConnection)):
                with self.assertRaises(sqlite3.OperationalError):
                    StateStore(root / "copy.db")
            with closing(sqlite3.connect(root / "copy.db")) as connection:
                self.assertIsNone(connection.execute("SELECT name FROM sqlite_master WHERE name = 'conversation_request_results'").fetchone())
                self.assertNotIn("model_attempts", [row[1] for row in connection.execute("PRAGMA table_info(repository_analysis_jobs)")])
            migrated = StateStore(root / "copy.db")
            self.assertEqual("기존 기록", migrated.list_messages("existing")[0]["content"])
            self.assertIsNone(migrated.repository_analysis(legacy_analysis)["model_attempts"])
            legacy = migrated.finish_repository_analysis_with_response(
                legacy_analysis, "worker", RepositoryAnalysisStatus.NEEDS_ATTENTION,
                "이전 실행 결과 미확인", reason="MODEL_OUTCOME_UNKNOWN",
            )
            self.assertIsNone(legacy["result"]["attempts"])
            migrated.save_conversation_result("telegram", "200", "1", RequestResult(RequestOutcome.FAILED))
            reopened = StateStore(root / "copy.db")
            self.assertEqual(RequestOutcome.FAILED, reopened.conversation_result("telegram", "200", "1").outcome)
            with closing(sqlite3.connect(root / "copy.db")) as connection:
                self.assertEqual("ok", connection.execute("PRAGMA integrity_check").fetchone()[0])


if __name__ == "__main__":
    unittest.main()
