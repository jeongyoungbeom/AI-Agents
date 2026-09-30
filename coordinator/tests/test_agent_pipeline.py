import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


COORDINATOR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COORDINATOR))

import agent_pipeline as pipeline


class PipelineTests(unittest.TestCase):
    def test_extract_review_json(self):
        parsed = pipeline.extract_review_json(
            '{"verdict":"approved","summary":"ok","findings":[]}'
        )
        self.assertEqual("approved", parsed["verdict"])

    def test_rejects_invalid_review_verdict(self):
        with self.assertRaises(pipeline.PipelineError):
            pipeline.extract_review_json(
                '{"verdict":"maybe","summary":"","findings":[]}'
            )

    def test_plan_validation_and_git_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.invalid"],
                cwd=repository,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Pipeline Test"],
                cwd=repository,
                check=True,
            )
            (repository / "README.md").write_text("test\n", encoding="utf-8")
            subprocess.run(["git", "add", "README.md"], cwd=repository, check=True)
            subprocess.run(["git", "commit", "-m", "initial"], cwd=repository, check=True)
            plan = {
                "run_id": "TEST-001",
                "repository": {"path": str(repository), "mode": "in_place"},
                "stages": [
                    {
                        "id": "STAGE-001",
                        "objective": "Test validation",
                        "acceptance_criteria": ["Validation succeeds"],
                        "verification_commands": ["git diff --check"],
                    }
                ],
            }
            validated = pipeline.validate_plan(plan)
            git_repository = pipeline.GitRepository(Path(validated["repository"]["path"]))
            git_repository.validate()
            git_repository.require_clean()

    def test_blank_telegram_configuration_is_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "TELEGRAM_BOT_TOKEN=\nTELEGRAM_CHAT_ID=\n", encoding="utf-8"
            )
            self.assertFalse(pipeline.TelegramNotifier(env_path).enabled)

    def test_full_stage_order_with_fake_hermes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            repository.mkdir()
            subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.invalid"],
                cwd=repository,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Pipeline Test"],
                cwd=repository,
                check=True,
            )
            (repository / "README.md").write_text("initial\n", encoding="utf-8")
            subprocess.run(["git", "add", "README.md"], cwd=repository, check=True)
            subprocess.run(["git", "commit", "-m", "initial"], cwd=repository, check=True)

            plan_path = root / "plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "run_id": "FLOW-TEST-001",
                        "repository": {
                            "path": str(repository),
                            "mode": "in_place",
                            "require_clean": True,
                        },
                        "branch": "ai/FLOW-TEST-001",
                        "stages": [
                            {
                                "id": "STAGE-001",
                                "objective": "Exercise the pipeline",
                                "acceptance_criteria": ["Flow completes"],
                                "verification_commands": ["git diff --check"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            old_artifacts = pipeline.ARTIFACTS_ROOT
            old_locks = pipeline.LOCKS_ROOT
            pipeline.ARTIFACTS_ROOT = root / "artifacts"
            pipeline.LOCKS_ROOT = pipeline.ARTIFACTS_ROOT / "locks"
            calls = []

            class FakeHermes:
                def preflight(self, require_auth=True):
                    return []

                def run(self, profile, prompt, repository_path, output_dir, label):
                    calls.append(profile)
                    if profile == "developer":
                        (repository_path / "feature.txt").write_text(
                            "implemented\n", encoding="utf-8"
                        )
                        return "implemented"
                    if profile == "reviewer":
                        return json.dumps(
                            {
                                "verdict": "changes_requested",
                                "summary": "Add final polish",
                                "findings": [
                                    {
                                        "id": "R-001",
                                        "severity": "low",
                                        "file": "feature.txt",
                                        "line": 1,
                                        "evidence": "test",
                                        "required_change": "append improved",
                                    }
                                ],
                            }
                        )
                    with (repository_path / "feature.txt").open(
                        "a", encoding="utf-8"
                    ) as handle:
                        handle.write("improved\n")
                    return "R-001 fixed"

            try:
                settings = pipeline.Settings(
                    hermes_executable=Path("unused"),
                    hermes_home=root / "hermes-home",
                    provider="openai-codex",
                    model="test-model",
                    timeout_seconds=30,
                    reasoning={},
                    toolsets={},
                )
                coordinator = pipeline.Coordinator(settings)
                coordinator.hermes = FakeHermes()
                run_dir = coordinator.run(plan_path)
                state = pipeline.read_json(run_dir / "state.json")
                self.assertEqual(["developer", "reviewer", "improver"], calls)
                self.assertEqual("COMPLETED", state["status"])
                self.assertEqual("COMPLETED", state["stages"][0]["status"])
                self.assertEqual(
                    "implemented\nimproved\n",
                    (repository / "feature.txt").read_text(encoding="utf-8"),
                )
            finally:
                pipeline.ARTIFACTS_ROOT = old_artifacts
                pipeline.LOCKS_ROOT = old_locks


if __name__ == "__main__":
    unittest.main()
