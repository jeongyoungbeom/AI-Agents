from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.contracts import RoleId, RunState
from app.services.hermes import (
    HermesModelSettings,
    HermesRoleSettings,
    HermesRunner,
    HermesSettings,
)
from app.storage import ArtifactStore, StateStore


class QueueTests(unittest.TestCase):
    def test_pipeline_job_can_only_be_claimed_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.db")
            store.create_run(RunState("RUN-QUEUE-TEST"))
            store.enqueue_pipeline_job("RUN-QUEUE-TEST", "telegram", "chat")
            first = store.claim_next_pipeline_job("worker-a")
            second = store.claim_next_pipeline_job("worker-b")
            self.assertIsNotNone(first)
            self.assertIsNone(second)

    def test_same_repository_lock_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.db")
            store.create_run(RunState("RUN-LOCK-A"))
            store.create_run(RunState("RUN-LOCK-B"))
            repository = str(root / "repository")
            self.assertTrue(store.acquire_repository_lock(repository, "RUN-LOCK-A", "a"))
            self.assertFalse(store.acquire_repository_lock(repository, "RUN-LOCK-B", "b"))


class FakeProcess:
    returncode = 0

    def __init__(self, command):
        path = Path(command[command.index("--result-file") + 1])
        path.write_text(json.dumps({"status": "succeeded", "text": '{"summary":"ok","needs_user_input":[]}',
                                    "error": "", "failure_reason": "", "session_id": "fixture"}), encoding="utf-8")

    def communicate(self, timeout=None):
        return '{"summary":"ok","needs_user_input":[]}', ""

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


class HermesPermissionTests(unittest.TestCase):
    def test_project_model_routing_matches_the_selected_models(self):
        project_root = Path(__file__).resolve().parents[2]
        settings = HermesSettings.load(project_root)

        self.assertEqual(settings.conversation, HermesModelSettings("gpt-6-sol", "xhigh"))
        self.assertEqual(settings.planning, HermesModelSettings("gpt-6-sol", "xhigh"))
        self.assertTrue(all(role.model == "gpt-6-sol" for role in settings.roles.values()))
        self.assertTrue(all(role.reasoning == "xhigh" for role in settings.roles.values()))
        self.assertTrue(settings.docker_required)

    def test_yolo_is_only_added_for_approved_write_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "hermes.exe"
            executable.write_text("fake", encoding="utf-8")
            home = root / "home"
            (home / "profiles" / "developer").mkdir(parents=True)
            (home / "profiles" / "reviewer").mkdir(parents=True)
            (home / "profiles" / "improver").mkdir(parents=True)
            (home / "auth.json").write_text("{}", encoding="utf-8")
            roles = {
                RoleId.DEVELOPMENT: HermesRoleSettings(
                    "developer", "gpt-5.6-terra", "xhigh", "coding", 10
                ),
                RoleId.REVIEW: HermesRoleSettings(
                    "reviewer", "gpt-5.6-sol", "xhigh", "debugging", 10
                ),
                RoleId.IMPROVEMENT: HermesRoleSettings(
                    "improver", "gpt-5.6-terra", "xhigh", "coding", 10
                ),
            }
            settings = HermesSettings(
                root,
                executable,
                home,
                "openai-codex",
                10,
                0.01,
                HermesModelSettings("gpt-5.6-terra", "xhigh"),
                HermesModelSettings("gpt-5.6-sol", "xhigh"),
                roles,
            )
            repository = root / "repository"
            repository.mkdir()
            image = repository / "reference.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\nreference")
            runner = HermesRunner(settings, ArtifactStore(root / "artifacts"))
            commands = []

            def fake_popen(command, **kwargs):
                commands.append(command)
                return FakeProcess(command)

            with patch("app.services.hermes.runner.subprocess.Popen", side_effect=fake_popen):
                runner.run(
                    "RUN-HERMES-TEST",
                    "stage-001",
                    RoleId.REVIEW,
                    repository,
                    "review",
                    allow_writes=False,
                    image_path=image,
                )
                runner.run("RUN-HERMES-TEST", "stage-001", RoleId.DEVELOPMENT, repository, "develop", allow_writes=True)

            self.assertNotIn("--yolo", commands[0])
            self.assertIn("--yolo", commands[1])
            self.assertIn("--checkpoints", commands[1])
            self.assertIn("--no-restore-cwd", commands[0])
            self.assertIn("--query-file", commands[0])
            self.assertEqual(commands[0][commands[0].index("--model") + 1], "gpt-5.6-sol")
            self.assertEqual(commands[1][commands[1].index("--model") + 1], "gpt-5.6-terra")
            self.assertEqual(commands[0][commands[0].index("--reasoning") + 1], "xhigh")
            self.assertIn("--usage-file", commands[0])
            self.assertIn("--oneshot", commands[0])
            self.assertEqual(commands[0][commands[0].index("--image") + 1], str(image.resolve()))

    def test_actual_hermes_usage_report_is_preferred_to_character_estimate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "hermes.exe"
            executable.write_text("fake", encoding="utf-8")
            home = root / "home"
            (home / "profiles" / "developer").mkdir(parents=True)
            (home / "auth.json").write_text("{}", encoding="utf-8")
            settings = HermesSettings(
                root,
                executable,
                home,
                "openai-codex",
                10,
                0.01,
                HermesModelSettings("gpt-5.6-terra", "xhigh"),
                HermesModelSettings("gpt-5.6-sol", "xhigh"),
                {
                    RoleId.DEVELOPMENT: HermesRoleSettings(
                        "developer", "gpt-5.6-terra", "xhigh", "coding", 10
                    )
                },
            )
            repository = root / "repository"
            repository.mkdir()
            runner = HermesRunner(settings, ArtifactStore(root / "artifacts"))

            def fake_popen(command, **_kwargs):
                usage_path = Path(command[command.index("--usage-file") + 1])
                usage_path.write_text(
                    json.dumps(
                        {
                            "input_tokens": 120,
                            "output_tokens": 30,
                            "total_tokens": 170,
                        }
                    ),
                    encoding="utf-8",
                )
                return FakeProcess(command)

            with patch("app.services.hermes.runner.subprocess.Popen", side_effect=fake_popen):
                result = runner.run(
                    "RUN-USAGE-TEST",
                    "stage-001",
                    RoleId.DEVELOPMENT,
                    repository,
                    "actual usage",
                    allow_writes=False,
                )

            self.assertEqual(120, result.usage.input_tokens)
            self.assertEqual(30, result.usage.output_tokens)
            self.assertEqual(170, result.usage.total_tokens)
            self.assertFalse(result.usage.estimated)

    def test_planning_can_override_the_builder_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "hermes.exe"
            executable.write_text("fake", encoding="utf-8")
            home = root / "home"
            (home / "profiles" / "developer").mkdir(parents=True)
            (home / "auth.json").write_text("{}", encoding="utf-8")
            roles = {
                RoleId.DEVELOPMENT: HermesRoleSettings(
                    "developer", "gpt-5.6-terra", "xhigh", "coding", 10
                )
            }
            settings = HermesSettings(
                root,
                executable,
                home,
                "openai-codex",
                10,
                0.01,
                HermesModelSettings("gpt-5.6-terra", "xhigh"),
                HermesModelSettings("gpt-5.6-sol", "xhigh"),
                roles,
            )
            repository = root / "repository"
            repository.mkdir()
            runner = HermesRunner(settings, ArtifactStore(root / "artifacts"))
            commands = []

            def fake_popen(command, **kwargs):
                commands.append(command)
                return FakeProcess(command)

            with patch("app.services.hermes.runner.subprocess.Popen", side_effect=fake_popen):
                runner.run(
                    "RUN-PLANNING-TEST",
                    "planning-r001",
                    RoleId.DEVELOPMENT,
                    repository,
                    "plan",
                    allow_writes=False,
                    model="gpt-5.6-sol",
                    reasoning="xhigh",
                )

            self.assertEqual(commands[0][commands[0].index("--model") + 1], "gpt-5.6-sol")


if __name__ == "__main__":
    unittest.main()
