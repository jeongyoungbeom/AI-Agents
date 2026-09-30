import sqlite3
import unittest
from pathlib import Path
from unittest.mock import patch

from app.gateway.adapters.telegram import TelegramAdapter, TelegramClient
from app.gateway.core import AgentReply, IncomingMessage, PollBatch
from app.gateway.core.runner import GatewayRunner
from app.contracts import RoleId
from app.orchestrator import RunStateMachine
from app.services.repository import RepositoryAnalysisRequest, RepositoryAnalysisStatus
from tests.gateway.support import build_application, temporary_directory


class PendingBackend:
    def respond(self, state, context, message):
        return AgentReply("계속 대화할 수 있습니다.")


class FakeTelegramClient:
    def __init__(self):
        self.sent = []
        self.edited = []
        self.offsets = []

    def get_me(self):
        return {"username": "test_bot"}

    def get_updates(self, offset, *, timeout_seconds):
        self.offsets.append(offset)
        if offset is not None:
            return []
        return [
            {
                "update_id": 4,
                "message": {
                    "message_id": 9,
                    "text": "새 기능을 만들어줘",
                    "from": {"id": 100},
                    "chat": {"id": 200, "type": "private"},
                },
            }
        ]

    def send_message(self, chat_id, text):
        self.sent.append((chat_id, text))
        return str(len(self.sent))

    def edit_message_text(self, chat_id, message_id, text):
        self.edited.append((chat_id, message_id, text))
        return str(message_id)


class FailFirstTelegramClient(FakeTelegramClient):
    def __init__(self):
        super().__init__()
        self.failures = 1

    def send_message(self, chat_id, text):
        if self.failures:
            self.failures -= 1
            raise RuntimeError("temporary network error")
        return super().send_message(chat_id, text)


class AdapterAndRunnerTests(unittest.TestCase):
    def test_telegram_adapter_and_runner_persist_poll_cursor(self):
        with temporary_directory() as directory:
            store, application = build_application(Path(directory), PendingBackend())
            client = FakeTelegramClient()
            adapter = TelegramAdapter(client, poll_timeout_seconds=1)
            runner = GatewayRunner(adapter, application)

            self.assertEqual(1, runner.run_once())
            self.assertEqual("5", store.load_gateway_cursor("telegram"))
            self.assertEqual(0, runner.run_once())
            self.assertEqual([None, 5], client.offsets)
            self.assertEqual("200", client.sent[0][0])

    def test_online_check_uses_adapter_client(self):
        adapter = TelegramAdapter(FakeTelegramClient(), poll_timeout_seconds=1)
        self.assertIn("@test_bot", adapter.check(online=True))

    def test_telegram_typing_action_uses_chat_action_api(self):
        client = TelegramClient("test-token")
        with patch.object(client, "_call", return_value=True) as call:
            client.send_chat_action("200")
        call.assert_called_once_with(
            "sendChatAction", {"chat_id": "200", "action": "typing"}
        )

    def test_failed_send_is_retried_from_durable_outbox(self):
        with temporary_directory() as directory:
            store, application = build_application(Path(directory), PendingBackend())
            client = FailFirstTelegramClient()
            errors = []
            runner = GatewayRunner(
                TelegramAdapter(client, poll_timeout_seconds=1),
                application,
                error_sink=errors.append,
            )

            self.assertEqual(0, runner.run_once())
            self.assertEqual("5", store.load_gateway_cursor("telegram"))
            pending = store.deliverable_outbound("telegram")
            self.assertEqual(1, len(pending))
            self.assertEqual("FAILED", pending[0]["status"])

            self.assertEqual(1, runner.run_once())
            self.assertEqual([], store.deliverable_outbound("telegram"))
            self.assertTrue(errors)

    def test_restart_does_not_retry_a_delivery_with_unknown_result(self):
        with temporary_directory() as directory:
            store, application = build_application(Path(directory), PendingBackend())
            client = FakeTelegramClient()
            runner = GatewayRunner(
                TelegramAdapter(client, poll_timeout_seconds=1), application
            )
            outbound_id = store.queue_outbound("telegram", "200", "최종 응답")

            with patch.object(
                store, "complete_outbound", side_effect=SystemExit("gateway stopped")
            ):
                with self.assertRaises(SystemExit):
                    runner.flush_outbound()

            self.assertEqual([("200", "최종 응답")], client.sent)
            connection = sqlite3.connect(store.path)
            try:
                connection.execute(
                    "UPDATE outbound_messages SET lease_until = ? WHERE outbound_id = ?",
                    ("2000-01-01T00:00:00+00:00", outbound_id),
                )
                connection.commit()
            finally:
                connection.close()

            restarted = GatewayRunner(
                TelegramAdapter(client, poll_timeout_seconds=1), application
            )
            self.assertEqual(0, restarted.flush_outbound())
            record = store.outbound_record(outbound_id)
            self.assertEqual("NEEDS_ATTENTION", record["status"] if record else None)
            self.assertEqual([("200", "최종 응답")], client.sent)

    def test_progress_outbound_edits_the_original_telegram_message(self):
        with temporary_directory() as directory:
            store, application = build_application(Path(directory), PendingBackend())
            client = FakeTelegramClient()
            runner = GatewayRunner(
                TelegramAdapter(client, poll_timeout_seconds=1), application
            )
            original_id = store.queue_outbound(
                "telegram", "200", "[센티널 · 진행 중]\n진행: 1/4"
            )
            store.queue_outbound(
                "telegram",
                "200",
                "[센티널 · 진행 중]\n진행: 3/4",
                delivery_mode="edit",
                target_outbound_id=original_id,
            )

            self.assertEqual(2, runner.flush_outbound())
            self.assertEqual([("200", "[센티널 · 진행 중]\n진행: 1/4")], client.sent)
            self.assertEqual(
                [("200", "1", "[센티널 · 진행 중]\n진행: 3/4")],
                client.edited,
            )

    def test_long_analysis_result_delivers_after_progress_id_becomes_unknown(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store, application = build_application(root, PendingBackend())
            request = RepositoryAnalysisRequest.create(
                channel="telegram", conversation_id="200", user_id="100",
                source_message_id="1", role_id=RoleId.DEVELOPMENT.value,
                request_text="전체 코드 분석해줘", repository_path=str(root),
                repository_identity="a" * 64, commit_sha="b" * 40, branch="main",
            )
            RunStateMachine(store).create_run(request.analysis_id)
            store.create_repository_analysis(request)
            store.claim_next_repository_analysis("analysis-worker")
            progress_id = store.queue_repository_analysis_progress(
                request.analysis_id, "analysis-worker", "분석 중"
            )
            progress_outbound = store.claim_next_outbound("telegram", "crashed-gateway")
            self.assertEqual(progress_id, progress_outbound["outbound_id"])
            store.mark_outbound_uncertain(
                progress_id, "gateway stopped after send", lease_owner="crashed-gateway"
            )
            result = "분석 결과\n" + "가" * 9000
            store.finish_repository_analysis_with_response(
                request.analysis_id, "analysis-worker",
                RepositoryAnalysisStatus.COMPLETED, result, reason="COMPLETED",
            )
            client = FakeTelegramClient()
            runner = GatewayRunner(TelegramAdapter(client, poll_timeout_seconds=1), application)
            self.assertEqual(1, runner.flush_outbound())
            self.assertEqual(result, "".join(text for _chat, text in client.sent))
            self.assertTrue(all(len(text) <= 3900 for _chat, text in client.sent))
            self.assertEqual([], client.edited)
            analysis = store.repository_analysis(request.analysis_id)
            self.assertEqual("DELIVERED", analysis["delivery_status"])
            self.assertEqual(result, analysis["final_response"])

    def test_partial_multimessage_delivery_is_unknown_and_result_is_retrievable(self):
        class FailSecondClient(FakeTelegramClient):
            def send_message(self, chat_id, text):
                if len(self.sent) == 1:
                    raise RuntimeError("second part failed")
                return super().send_message(chat_id, text)

        with temporary_directory() as directory:
            root = Path(directory)
            store, application = build_application(root, PendingBackend())
            request = RepositoryAnalysisRequest.create(
                channel="telegram", conversation_id="200", user_id="100",
                source_message_id="1", role_id=RoleId.DEVELOPMENT.value,
                request_text="전체 코드 분석해줘", repository_path=str(root),
                repository_identity="a" * 64, commit_sha="b" * 40, branch="main",
            )
            RunStateMachine(store).create_run(request.analysis_id)
            store.create_repository_analysis(request)
            store.claim_next_repository_analysis("analysis-worker")
            result = "가" * 8000
            store.finish_repository_analysis_with_response(
                request.analysis_id, "analysis-worker",
                RepositoryAnalysisStatus.COMPLETED, result, reason="COMPLETED",
            )
            client = FailSecondClient()
            runner = GatewayRunner(
                TelegramAdapter(client, poll_timeout_seconds=1), application,
                error_sink=lambda _message: None,
            )
            self.assertEqual(0, runner.flush_outbound())
            self.assertEqual(1, len(client.sent))
            self.assertEqual("UNKNOWN", store.repository_analysis(request.analysis_id)["delivery_status"])
            retrieved = application.handle(IncomingMessage(
                channel="telegram", conversation_id="200", user_id="100",
                external_message_id="result-retrieval", text="분석 결과",
            ))
            self.assertIn(result, retrieved[0].text)


if __name__ == "__main__":
    unittest.main()
