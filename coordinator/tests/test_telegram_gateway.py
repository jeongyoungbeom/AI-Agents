import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


COORDINATOR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COORDINATOR))

import agent_pipeline as pipeline
import telegram_gateway as gateway


class FakeAPI:
    def __init__(self):
        self.messages = []

    def send(self, chat_id, text):
        self.messages.append((chat_id, text))


def make_repository(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Gateway Test"],
        cwd=path,
        check=True,
    )
    (path / "README.md").write_text("test\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True)


class TelegramGatewayTests(unittest.TestCase):
    def settings(self, root: Path) -> pipeline.Settings:
        return pipeline.Settings(
            hermes_executable=root / "hermes.exe",
            hermes_home=root / "hermes-home",
            provider="openai-codex",
            model="test-model",
            timeout_seconds=30,
            reasoning={},
            toolsets={},
        )

    def test_bot_config_requires_token_and_user_allowlist(self):
        with self.assertRaises(gateway.GatewayError):
            gateway.BotConfig("", frozenset({1}), frozenset()).validate()
        with self.assertRaises(gateway.GatewayError):
            gateway.BotConfig("token", frozenset(), frozenset()).validate()

    def test_only_allowed_users_and_explicit_group_chats_are_authorized(self):
        config = gateway.BotConfig("token", frozenset({11}), frozenset({-22}))
        instance = gateway.TelegramGateway.__new__(gateway.TelegramGateway)
        instance.config = config
        self.assertTrue(instance.authorized(11, 11, "private"))
        self.assertTrue(instance.authorized(11, -22, "group"))
        self.assertFalse(instance.authorized(12, 12, "private"))
        self.assertFalse(instance.authorized(11, -33, "group"))

    def test_project_is_selected_per_chat_from_any_absolute_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repo with spaces"
            make_repository(repository)
            api = FakeAPI()
            store = gateway.GatewayStore(root / "state.json")
            instance = gateway.TelegramGateway(
                gateway.BotConfig("token", frozenset({1}), frozenset()),
                self.settings(root),
                api=api,
                store=store,
            )
            instance.command_project(1, f'"{repository}"')
            self.assertEqual(str(repository.resolve()), store.chat(1)["repository"])
            self.assertIn("프로젝트 선택됨", api.messages[-1][1])

    def test_planning_json_and_plan_normalization(self):
        planning = gateway.extract_planning_json(
            "prefix\n"
            + json.dumps(
                {
                    "status": "ready",
                    "summary": "test",
                    "stages": [
                        {
                            "objective": "Implement",
                            "acceptance_criteria": ["Works"],
                            "verification_commands": ["git diff --check"],
                        }
                    ],
                }
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            plan = gateway.normalize_plan(planning, repository, "TG-TEST-001")
        self.assertEqual("stage-001", plan["stages"][0]["id"])
        self.assertEqual("ai/tg-test-001", plan["branch"])

    def test_transient_state_is_marked_interrupted_after_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            store = gateway.GatewayStore(Path(directory) / "state.json")
            store.update_chat(10, phase="RUNNING")
            store.recover_interrupted()
            self.assertEqual("INTERRUPTED", store.chat(10)["phase"])

    def test_long_messages_are_split_within_telegram_limit(self):
        chunks = gateway.split_message("x" * 9000)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(len(chunk) <= 3900 for chunk in chunks))

    def test_unsafe_verification_command_is_rejected(self):
        with self.assertRaises(gateway.GatewayError):
            gateway.validate_verification_command("git clean -fd")
        with self.assertRaises(gateway.GatewayError):
            gateway.validate_verification_command("npm test && git push")
        self.assertEqual(
            "npm run test",
            gateway.validate_verification_command("npm run test"),
        )

    def test_coordinator_cancel_hook(self):
        instance = pipeline.Coordinator(
            self.settings(Path(".")),
            notifier=FakeAPI(),
            cancel_check=lambda: True,
        )
        with self.assertRaises(pipeline.PipelineCancelled):
            instance._raise_if_cancelled()


if __name__ == "__main__":
    unittest.main()
