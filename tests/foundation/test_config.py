import unittest

from app.config import FoundationConfig
from app.contracts import RoleId
from app.services.budget import BudgetPolicy
from app.services.context import ContextPolicy
from app.services.retention import RetentionPolicy
from tests.foundation.support import AI_ROOT


class ConfigTests(unittest.TestCase):
    def test_foundation_configuration_has_selected_role_names(self):
        config = FoundationConfig.load(AI_ROOT)
        self.assertEqual(set(RoleId), set(config.roles))
        self.assertEqual("빌더", config.roles[RoleId.DEVELOPMENT].display_name)
        self.assertEqual("센티널", config.roles[RoleId.REVIEW].display_name)
        self.assertEqual("피니셔", config.roles[RoleId.IMPROVEMENT].display_name)
        self.assertTrue(all(not role.model for role in config.roles.values()))
        self.assertTrue(all(role.instructions.is_file() for role in config.roles.values()))
        self.assertTrue(
            all(role.conversation_instructions.is_file() for role in config.roles.values())
        )
        self.assertEqual("개발 시작해", config.approval_phrase)
        self.assertEqual("이 프로젝트 사용 승인해", config.repository_approval_phrase)
        self.assertEqual(720, config.repository_approval_ttl_hours)
        self.assertEqual(24, config.pending_project_request_ttl_hours)

    def test_calibrated_limits_enable_hard_budgets_and_retention(self):
        policy = BudgetPolicy.load(AI_ROOT / "config" / "limits.json")
        self.assertFalse(policy.calibration_mode)
        self.assertEqual(400_000, policy.conversation_tokens)
        self.assertEqual(300_000, policy.per_stage_tokens)
        self.assertEqual(1_200_000, policy.whole_task_tokens)
        self.assertEqual(80_000, policy.completion_reserve_tokens)
        self.assertEqual(2_048, policy.provider_input_overhead_tokens)
        self.assertEqual(1, policy.retries["technical_error"])
        context = ContextPolicy.load(AI_ROOT / "config" / "limits.json")
        self.assertEqual(24_000, context.max_characters)
        self.assertEqual(6_000, context.decision_summary_characters)
        retention = RetentionPolicy.load(AI_ROOT / "config" / "limits.json")
        self.assertEqual(14, retention.attachment_ttl_days)
        self.assertEqual(30, retention.backup_ttl_days)
        self.assertEqual(7, retention.backup_count)

    def test_secret_template_contains_no_values(self):
        path = AI_ROOT / "config" / "secrets.env.example"
        values = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                values.append(line.split("=", 1)[1])
        self.assertTrue(values)
        self.assertTrue(all(value == "" for value in values))


if __name__ == "__main__":
    unittest.main()
