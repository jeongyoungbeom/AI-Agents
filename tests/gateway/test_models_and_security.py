import unittest

from app.gateway.adapters.telegram import TelegramAdapter
from app.gateway.core import AccessPolicy, IncomingMessage


def message(*, user: str = "100", private: bool = True, conversation: str = "200"):
    return IncomingMessage(
        channel="telegram",
        conversation_id=conversation,
        user_id=user,
        external_message_id="update-1",
        text="hello",
        is_private=private,
    )


class ModelsAndSecurityTests(unittest.TestCase):
    def test_access_policy_denies_by_default_and_allows_listed_private_user(self):
        self.assertFalse(AccessPolicy().allows(message()))
        policy = AccessPolicy(allowed_users=frozenset({"100"}))
        self.assertTrue(policy.allows(message()))
        self.assertFalse(policy.allows(message(user="101")))

    def test_group_requires_explicit_enable_and_conversation_allowlist(self):
        private_only = AccessPolicy(
            allowed_users=frozenset({"100"}),
            allowed_group_conversations=frozenset({"200"}),
        )
        self.assertFalse(private_only.allows(message(private=False)))
        enabled = AccessPolicy(
            allowed_users=frozenset({"100"}),
            allowed_group_conversations=frozenset({"200"}),
            private_only=False,
        )
        self.assertTrue(enabled.allows(message(private=False)))
        self.assertFalse(enabled.allows(message(private=False, conversation="201")))

    def test_telegram_update_is_normalized_to_common_message(self):
        incoming = TelegramAdapter.normalize_update(
            {
                "update_id": 91,
                "message": {
                    "message_id": 7,
                    "text": "기능을 만들어줘",
                    "from": {"id": 100, "first_name": "사용자"},
                    "chat": {"id": 200, "type": "private"},
                },
            }
        )
        self.assertIsNotNone(incoming)
        assert incoming is not None
        self.assertEqual("telegram", incoming.channel)
        self.assertEqual("update-91", incoming.external_message_id)
        self.assertEqual("100", incoming.user_id)

    def test_long_telegram_messages_are_split_within_limit(self):
        parts = TelegramAdapter.split_text(("문장 " * 80).strip(), limit=100)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(0 < len(part) <= 100 for part in parts))
        self.assertEqual(("문장 " * 80).strip(), " ".join(parts))


if __name__ == "__main__":
    unittest.main()
