from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.agents.parsing import InvalidAgentResponse, parse_team_conversation_reply
from app.contracts import RoleId
from app.services.hermes import (
    HermesCancelled, HermesExecutionError, HermesModelSettings, HermesRoleSettings,
    HermesRunner, HermesSettings,
)
from app.storage import ArtifactStore


class FakeProcess:
    def __init__(self, command, report, *, stdout="", stderr="", returncode=0, usage=None):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        if report is not None:
            Path(command[command.index("--result-file") + 1]).write_text(json.dumps(report), encoding="utf-8")
        if usage is not None:
            Path(command[command.index("--usage-file") + 1]).write_text(json.dumps(usage), encoding="utf-8")

    def communicate(self, timeout=None):
        return self.stdout, self.stderr

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


class HermesContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        executable = self.root / "hermes.exe"
        executable.write_text("fixture", encoding="utf-8")
        home = self.root / "home"
        (home / "profiles" / "reviewer").mkdir(parents=True)
        (home / "auth.json").write_text("{}", encoding="utf-8")
        settings = HermesSettings(
            self.root, executable, home, "openai-codex", 1, .01,
            HermesModelSettings("fixture", "low"), HermesModelSettings("fixture", "low"),
            {RoleId.REVIEW: HermesRoleSettings("reviewer", "fixture", "low", "coding", 1)},
        )
        self.runner = HermesRunner(settings, ArtifactStore(self.root / "artifacts"))

    def invoke(self, report, **options):
        process_options = {key: options.pop(key) for key in tuple(options)
                           if key in {"stdout", "stderr", "returncode", "usage"}}
        commands = []

        def start(command, **_kwargs):
            commands.append(command)
            return FakeProcess(command, report, **process_options)

        with patch("app.services.hermes.runner.subprocess.Popen", side_effect=start):
            result = self.runner.run("RUN-CONTRACT", "chat-001", RoleId.REVIEW,
                                     self.root, "한국어 질문", allow_writes=False, **options)
        return result, commands[0]

    def test_structured_payload_survives_stdout_warning_and_tracks_actual_usage(self):
        payload = '{"message":"정상 답변","calls":[],"memory_updates":[],"repository_tools":[]}'
        result, command = self.invoke(
            {"status": "succeeded", "text": payload, "internal_stream_retries": 2}, stdout="Warning: harmless notice\n",
            stderr="session_id: fixture", no_tools=True, max_output_tokens=100,
            usage={"input_tokens": 12, "output_tokens": 5, "total_tokens": 17},
        )
        self.assertEqual("정상 답변", parse_team_conversation_reply(result.text, RoleId.REVIEW).text)
        self.assertEqual(17, result.usage.total_tokens)
        self.assertFalse(result.usage.estimated)
        self.assertIn("--no-tools", command)
        self.assertNotIn("--toolsets", command)
        self.assertNotIn("--max-output-tokens", command)
        role_log = (self.root / "artifacts" / "RUN-CONTRACT" / "stages" / "chat-001" / "review.log")
        self.assertIn("internal_stream_retries=2", role_log.read_text(encoding="utf-8"))

    def test_invalid_json_and_authentication_failure_are_not_success(self):
        result, _ = self.invoke({"status": "succeeded", "text": "{broken"})
        with self.assertRaises(InvalidAgentResponse):
            parse_team_conversation_reply(result.text, RoleId.REVIEW)
        with self.assertRaises(HermesExecutionError) as raised:
            self.invoke({"status": "failed", "text": "", "error": "401 unauthorized"},
                        returncode=1, stderr="session_id: fixture")
        self.assertEqual("authentication", raised.exception.category)
        self.assertFalse(raised.exception.retryable)
        self.assertIn("401 unauthorized", str(raised.exception))

    def test_missing_result_file_is_invalid_result(self):
        with self.assertRaises(HermesExecutionError) as raised:
            self.invoke(None, stdout="Warning only")
        self.assertEqual("invalid_result", raised.exception.category)

    def test_timeout_and_cancellation_are_distinct(self):
        class StalledProcess:
            returncode = None

            def communicate(self, timeout=None):
                raise subprocess.TimeoutExpired("hermes", timeout)

            def poll(self):
                return None

        with patch("app.services.hermes.runner.subprocess.Popen", return_value=StalledProcess()), \
             patch("app.services.hermes.runner.ProcessTree"), \
             patch("app.services.hermes.runner.time.monotonic", side_effect=[0, 2]):
            with self.assertRaises(HermesExecutionError) as raised:
                self.runner.run("RUN-CONTRACT", "chat-001", RoleId.REVIEW,
                                self.root, "질문", allow_writes=False)
        self.assertEqual("timeout", raised.exception.category)
        with patch("app.services.hermes.runner.subprocess.Popen", return_value=StalledProcess()), \
             patch("app.services.hermes.runner.ProcessTree"):
            with self.assertRaises(HermesCancelled):
                self.runner.run("RUN-CONTRACT", "chat-001", RoleId.REVIEW,
                                self.root, "질문", allow_writes=False, cancelled=lambda: True)

    def test_cancellation_keeps_a_report_written_before_termination(self):
        class ReportedProcess(FakeProcess):
            def communicate(self, timeout=None):
                raise subprocess.TimeoutExpired("hermes", timeout)

        def start(command, **_kwargs):
            return ReportedProcess(command, None,
                usage={"input_tokens": 12, "output_tokens": 5, "total_tokens": 17})

        with patch("app.services.hermes.runner.subprocess.Popen", side_effect=start), \
             patch("app.services.hermes.runner.ProcessTree"):
            with self.assertRaises(HermesCancelled) as raised:
                self.runner.run("RUN-CONTRACT", "chat-001", RoleId.REVIEW, self.root,
                                "질문", allow_writes=False, cancelled=lambda: True)
        self.assertEqual(17, raised.exception.usage.total_tokens)
        self.assertFalse(raised.exception.usage.estimated)


if __name__ == "__main__":
    unittest.main()
