import os
import unittest
from pathlib import Path
from unittest.mock import patch

from app.gateway.config import ConversationSettings, GatewaySettings


AI_ROOT = Path(__file__).resolve().parents[2]


class GatewayConfigTests(unittest.TestCase):
    def test_secret_example_starts_blank_and_real_file_is_gitignored(self):
        example = AI_ROOT / "config" / "secrets.env.example"
        values = []
        for line in example.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                values.append(line.split("=", 1)[1])
        self.assertTrue(values)
        self.assertTrue(all(value == "" for value in values))
        gitignore = (AI_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("config/secrets.env", gitignore)

    def test_environment_values_override_blank_file(self):
        values = {
            "TELEGRAM_BOT_TOKEN": "test-token",
            "TELEGRAM_ALLOWED_USERS": "100, 101",
            "TELEGRAM_ALLOWED_CHATS": "-200",
        }
        with patch.dict(os.environ, values, clear=True):
            settings = GatewaySettings.load(AI_ROOT)
        settings.validate()
        self.assertEqual(frozenset({"100", "101"}), settings.telegram.allowed_users)
        self.assertEqual(frozenset({"-200"}), settings.telegram.allowed_chats)
        self.assertEqual(3, settings.conversation.group_parallel_workers)
        self.assertEqual(2, settings.conversation.worker_count)
        self.assertEqual(8 * 1024 * 1024, settings.attachments.max_bytes)
        self.assertEqual(24_000, settings.attachments.max_text_characters)
        self.assertEqual(5 * 1024 * 1024, settings.logging.gateway_max_bytes)
        self.assertEqual(14, settings.logging.gateway_backup_count)

    def test_group_parallel_workers_is_limited_to_the_three_roles(self):
        with self.assertRaises(ValueError):
            ConversationSettings(group_parallel_workers=4).validate()

    def test_model_calls_per_message_cannot_be_configured_above_four(self):
        with self.assertRaises(ValueError):
            ConversationSettings(max_auto_agent_replies=5).validate()

    def test_conversation_worker_pool_is_bounded(self):
        with self.assertRaises(ValueError):
            ConversationSettings(worker_count=9).validate()


if __name__ == "__main__":
    unittest.main()
