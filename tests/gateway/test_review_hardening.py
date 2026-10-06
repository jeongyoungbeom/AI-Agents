import io
import subprocess
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from app.gateway.adapters.telegram import TelegramAdapter, TelegramAPIError, TelegramClient
from app.gateway.core import (
    AccessPolicy,
    AgentReply,
    GatewayApplication,
    GovernedAgentBackend,
    IncomingMessage,
    LocalGitRepositoryValidator,
    OutgoingMessage,
    ProposedStage,
)
from app.gateway.core.runner import GatewayRunner
from app.services.budget import BudgetManager, BudgetPolicy
from app.services.verification import SafeVerificationPolicy, UnsafeVerificationCommand
from app.storage import StateStore, StoreError
from tests.gateway.support import build_application, temporary_directory


def incoming(identifier, text, *, conversation="200", user="100"):
    return IncomingMessage(
        channel="telegram",
        conversation_id=conversation,
        user_id=user,
        external_message_id=str(identifier),
        text=text,
    )


class ReadyBackend:
    def __init__(self, *, fail_once=False):
        self.fail_once = fail_once
        self.calls = 0

    def respond(self, state, context, message):
        self.calls += 1
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("temporary token=secret")
        return AgentReply(
            "계획을 준비했습니다.",
            stages=(
                ProposedStage(
                    objective="구현",
                    scope=("app/",),
                    acceptance_criteria=("테스트 통과",),
                    verification_commands=("py -3 -m unittest",),
                ),
            ),
        )


class ReviewHardeningTests(unittest.TestCase):
    def test_backend_crash_keeps_explicit_project_approval_and_can_retry(self):
        with temporary_directory() as directory:
            root = Path(directory)
            backend = ReadyBackend(fail_once=True)
            store, application = build_application(root, backend)
            application.handle(incoming("m1", "기능 개발"))
            application.handle(incoming("m2", "D:\\projects\\sample"))
            approval = incoming("m3", "이 프로젝트 사용 승인해")

            with self.assertRaises(RuntimeError):
                application.handle(approval)
            binding = store.load_conversation("telegram", "200")
            state = store.load_run(binding["run_id"])
            self.assertTrue(state.repository_approved)
            self.assertTrue(
                store.repository_is_approved(
                    "telegram", "200", "100", state.repository,
                    state.repository_identity,
                )
            )
            self.assertEqual("FAILED", store.inbound_receipt("telegram", "200:m3")["status"])

            retried = application.handle(approval)
            self.assertIn("계획이 준비", retried[0].text)

    def test_same_external_id_in_two_conversations_is_not_a_collision(self):
        class EchoRouter:
            def route(self, message):
                return (OutgoingMessage(message.channel, message.conversation_id, "ok"),)

        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            application = GatewayApplication(
                store,
                EchoRouter(),
                AccessPolicy(allowed_users=frozenset({"100"})),
            )
            self.assertTrue(application.handle(incoming("same", "a", conversation="200")))
            self.assertTrue(application.handle(incoming("same", "b", conversation="201")))

    def test_plan_revision_must_be_reapproved(self):
        with temporary_directory() as directory:
            store, application = build_application(Path(directory), ReadyBackend())
            application.handle(incoming("p1", "기능 개발"))
            application.handle(incoming("p2", "D:\\projects\\sample"))
            application.handle(incoming("p3", "이 프로젝트 사용 승인해"))
            binding = store.load_conversation("telegram", "200")
            first = store.load_run(binding["run_id"])

            application.handle(incoming("p4", "범위를 바꿔줘"))
            second = store.load_run(binding["run_id"])
            self.assertEqual(2, second.plan_revision)
            self.assertFalse(second.approval_granted)
            self.assertEqual("SUPERSEDED", store.load_plan_revision(first.run_id, 1)["status"])
            with self.assertRaises(StoreError):
                store.approve_plan_revision(first.run_id, 1, first.plan_hash)

            application.handle(incoming("p5", "개발 시작해"))
            approved = store.load_run(binding["run_id"])
            self.assertEqual(approved.plan_revision, approved.approved_plan_revision)
            self.assertEqual(approved.plan_hash, approved.approved_plan_hash)

    def test_conversation_owner_is_enforced(self):
        with temporary_directory() as directory:
            store, application = build_application(Path(directory), ReadyBackend())
            application.handle(incoming("o1", "기능 개발"))
            binding = store.load_conversation("telegram", "200")
            before = len(store.list_messages(binding["run_id"]))
            application.access_policy = AccessPolicy(
                allowed_users=frozenset({"100", "101"})
            )

            denied = application.handle(incoming("o2", "상태", user="101"))
            self.assertIn("소유자", denied[0].text)
            self.assertEqual(before, len(store.list_messages(binding["run_id"])))

    def test_unknown_failure_cost_is_recorded_and_blocks_retry(self):
        with temporary_directory() as directory:
            delegate = ReadyBackend(fail_once=True)
            store, application = build_application(Path(directory), delegate)
            application.router.backend = GovernedAgentBackend(
                delegate,
                BudgetManager(
                    BudgetPolicy(
                        conversation_tokens=5000,
                        retries={"technical_error": 1},
                    ),
                    store,
                ),
                response_reserve_tokens=10,
            )
            application.handle(incoming("b1", "기능 개발"))
            application.handle(incoming("b2", "D:\\projects\\sample"))
            application.handle(incoming("b3", "이 프로젝트 사용 승인해"))
            binding = store.load_conversation("telegram", "200")
            self.assertGreater(store.usage_total(binding["run_id"]), 0)
            self.assertEqual(
                0,
                store.retry_count(binding["run_id"], "stage-001", "technical_error"),
            )

    def test_outbox_lease_allows_only_one_process_to_claim(self):
        with temporary_directory() as directory:
            database = Path(directory) / "state.db"
            first = StateStore(database)
            outbound_id = first.queue_outbound("telegram", "200", "hello")
            second = StateStore(database)
            claimed = first.claim_next_outbound("telegram", "runner-one")
            self.assertEqual(outbound_id, claimed["outbound_id"])
            self.assertIsNone(second.claim_next_outbound("telegram", "runner-two"))
            first.complete_outbound(outbound_id, ("1",), lease_owner="runner-one")
            self.assertEqual("SENT", second.outbound_record(outbound_id)["status"])

    def test_verification_policy_rejects_shell_chaining(self):
        policy = SafeVerificationPolicy()
        self.assertEqual(("py", "-3", "-m", "unittest"), policy.prepare("py -3 -m unittest").argv)
        self.assertEqual(("node", "verify.js"), policy.prepare("node verify.js").argv)
        with self.assertRaises(UnsafeVerificationCommand):
            policy.prepare("py -m unittest; Remove-Item -Recurse D:\\")
        with self.assertRaises(UnsafeVerificationCommand):
            policy.prepare("python -c \"import shutil\"")
        with self.assertRaises(UnsafeVerificationCommand):
            policy.prepare("python -m pip install example-package")
        with self.assertRaises(UnsafeVerificationCommand):
            policy.prepare("npm install")
        with self.assertRaises(UnsafeVerificationCommand):
            policy.prepare("npx --package example-package test")

    def test_telegram_rate_limit_exposes_retry_after(self):
        body = io.BytesIO(
            b'{"ok":false,"description":"Too Many Requests","parameters":{"retry_after":7}}'
        )
        error = urllib.error.HTTPError("https://example.invalid", 429, "rate", {}, body)
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(TelegramAPIError) as raised:
                TelegramClient("dummy-token").get_me()
        self.assertEqual(7, raised.exception.retry_after)

    def test_real_git_repository_is_resolved_and_unc_is_rejected(self):
        with temporary_directory() as directory:
            repository = Path(directory) / "repository"
            repository.mkdir()
            initialized = subprocess.run(
                ["git", "init", "--quiet", str(repository)],
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, initialized.returncode)
            (repository / "README.md").write_text("test\n", encoding="utf-8")
            subprocess.run(
                ["git", "-C", str(repository), "config", "user.email", "test@example.com"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repository), "config", "user.name", "Test"],
                check=True,
            )
            subprocess.run(["git", "-C", str(repository), "add", "README.md"], check=True)
            subprocess.run(
                ["git", "-C", str(repository), "commit", "--quiet", "-m", "initial"],
                check=True,
            )
            nested = repository / "src"
            nested.mkdir()
            selected = LocalGitRepositoryValidator().validate(str(nested))
            self.assertEqual(repository.resolve(), selected.path)
            with self.assertRaises(ValueError):
                LocalGitRepositoryValidator().validate("\\\\server\\share\\repo")

    def test_runner_honors_telegram_retry_after(self):
        class RateLimitedAdapter:
            name = "telegram"

            def __init__(self):
                self.calls = 0

            def poll(self, cursor):
                self.calls += 1
                if self.calls == 1:
                    raise TelegramAPIError("rate limited", retry_after=7)
                raise KeyboardInterrupt

            def prepare(self, message):
                return (message,)

            def send(self, message):
                return ("1",)

            def check(self, *, online=True):
                return "ok"

        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            adapter = RateLimitedAdapter()
            application = GatewayApplication(
                store,
                object(),
                AccessPolicy(allowed_users=frozenset({"100"})),
            )
            runner = GatewayRunner(adapter, application, error_sink=lambda _: None)
            with patch("app.gateway.core.runner.time.sleep") as sleep:
                with self.assertRaises(KeyboardInterrupt):
                    runner.run_forever()
            self.assertAlmostEqual(7.0, sum(call.args[0] for call in sleep.call_args_list))
            self.assertTrue(all(call.args[0] <= 0.5 for call in sleep.call_args_list))


class ChunkClient:
    def __init__(self):
        self.sent = []
        self.send_attempts = 0
        self.polled = False

    def get_me(self):
        return {"username": "chunk_bot"}

    def get_updates(self, offset, *, timeout_seconds):
        if self.polled:
            return []
        self.polled = True
        return [{
            "update_id": 1,
            "message": {
                "message_id": 1,
                "text": "go",
                "from": {"id": 100},
                "chat": {"id": 200, "type": "private"},
            },
        }]

    def send_message(self, chat_id, text):
        self.send_attempts += 1
        if self.send_attempts == 2:
            raise RuntimeError("temporary")
        self.sent.append(text)
        return str(self.send_attempts)


class ChunkRetryTests(unittest.TestCase):
    def test_failed_second_chunk_does_not_resend_first_chunk(self):
        class LongRouter:
            def route(self, message):
                return (
                    OutgoingMessage(
                        message.channel,
                        message.conversation_id,
                        "A" * 100 + "B" * 100 + "C" * 50,
                    ),
                )

        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            client = ChunkClient()
            adapter = TelegramAdapter(
                client, poll_timeout_seconds=1, max_message_characters=100
            )
            application = GatewayApplication(
                store,
                LongRouter(),
                AccessPolicy(allowed_users=frozenset({"100"})),
                message_preparer=adapter.prepare,
            )
            runner = GatewayRunner(adapter, application, error_sink=lambda _: None)

            self.assertEqual(1, runner.run_once())
            self.assertEqual(2, runner.run_once())
            self.assertEqual(["A" * 100, "B" * 100, "C" * 50], client.sent)


if __name__ == "__main__":
    unittest.main()
