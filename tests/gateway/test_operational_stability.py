from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from app.contracts import RoleId
from app.gateway.core import GatewayRunner, GatewayWorkerStopped, PollBatch
from app.gateway.outbound_worker import ActivityWorker, OutboundWorker
from app.gateway.runtime_guard import GatewayInstanceLock
from app.services.hermes import (
    HermesCancelled,
    HermesModelSettings,
    HermesRoleSettings,
    HermesRunner,
    HermesSettings,
)
from app.services.process_tree import ProcessTree, isolated_process_options
from app.storage import ArtifactStore
from tests.gateway.support import build_application, temporary_directory


class IdleAdapter:
    name = "idle"

    def __init__(self):
        self.polls = 0

    def check(self, *, online=True):
        return "ready"

    def poll(self, cursor):
        self.polls += 1
        return PollBatch(cursor=cursor)

    def prepare(self, message):
        return (message,)

    def send(self, message):
        return ("sent",)


class OperationalStabilityTests(unittest.TestCase):
    def test_gateway_instance_lock_rejects_a_second_process_and_releases(self):
        with temporary_directory() as directory:
            path = Path(directory) / "gateway.lock"
            first = GatewayInstanceLock(path)
            second = GatewayInstanceLock(path)

            first.acquire()
            try:
                with self.assertRaisesRegex(RuntimeError, "이미 실행 중"):
                    second.acquire()
            finally:
                first.release()

            second.acquire()
            second.release()

    def test_gateway_lock_is_released_after_a_separate_process_crashes(self):
        with temporary_directory() as directory:
            path = Path(directory) / "gateway.lock"
            code = (
                "import time\n"
                "from pathlib import Path\n"
                "from app.gateway.runtime_guard import GatewayInstanceLock\n"
                f"lock = GatewayInstanceLock(Path({str(path)!r}))\n"
                "lock.acquire()\n"
                "print('LOCKED', flush=True)\n"
                "time.sleep(60)\n"
            )
            process = subprocess.Popen(
                [sys._base_executable, "-u", "-c", code],
                cwd=Path(__file__).resolve().parents[2],
                text=True,
                encoding="utf-8",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                self.assertEqual("LOCKED", process.stdout.readline().strip())
                with self.assertRaisesRegex(RuntimeError, str(process.pid)):
                    GatewayInstanceLock(path).acquire()
            finally:
                process.kill()
                process.wait(timeout=5)
                if process.stdout is not None:
                    process.stdout.close()
                if process.stderr is not None:
                    process.stderr.close()

            recovered = GatewayInstanceLock(path)
            recovered.acquire()
            recovered.release()

    def test_invalid_lock_owner_is_not_rendered(self):
        with temporary_directory() as directory:
            path = Path(directory) / "gateway.lock"
            path.write_bytes(b"0\x1b[31mnot-a-pid")

            self.assertEqual("", GatewayInstanceLock(path)._owner_hint())

    def test_dead_required_worker_stops_before_the_next_poll(self):
        with temporary_directory() as directory:
            _store, application = build_application(Path(directory), object())
            adapter = IdleAdapter()
            errors = []

            def failed_health_check():
                raise GatewayWorkerStopped("pipeline worker stopped")

            runner = GatewayRunner(
                adapter,
                application,
                error_sink=errors.append,
                health_check=failed_health_check,
            )

            with self.assertRaises(GatewayWorkerStopped):
                runner.run_forever(retry_delay_seconds=0.001)

            self.assertEqual(0, adapter.polls)
            self.assertEqual(1, len(errors))
            self.assertIn("워커 오류", errors[0])

    def test_outbound_notification_never_sends_on_the_calling_thread(self):
        caller_thread = threading.get_ident()
        sent = threading.Event()
        sender_threads = []

        def sender():
            sender_threads.append(threading.get_ident())
            sent.set()
            return 0

        worker = OutboundWorker(sender, poll_seconds=0.05)
        worker.start()
        try:
            worker.notify()
            self.assertTrue(sent.wait(timeout=1))
        finally:
            worker.stop()
            worker.join(timeout=1)

        self.assertFalse(worker.is_alive())
        self.assertTrue(sender_threads)
        self.assertTrue(all(item != caller_thread for item in sender_threads))

    def test_activity_notification_never_uses_the_work_calling_thread(self):
        caller_thread = threading.get_ident()
        sent = threading.Event()
        sender_threads = []

        def sender(_channel, _conversation_id):
            sender_threads.append(threading.get_ident())
            sent.set()

        worker = ActivityWorker(sender)
        worker.start()
        try:
            worker.notify("telegram", "chat-1")
            self.assertTrue(sent.wait(timeout=1))
        finally:
            worker.stop()
            worker.join(timeout=1)

        self.assertFalse(worker.is_alive())
        self.assertTrue(sender_threads)
        self.assertTrue(all(item != caller_thread for item in sender_threads))

    def test_gateway_stop_terminates_active_hermes_process(self):
        with temporary_directory() as directory:
            root = Path(directory)
            executable = root / "hermes.exe"
            executable.write_bytes(b"test")
            home = root / "hermes-home"
            (home / "profiles" / "test").mkdir(parents=True)
            (home / "auth.json").write_text("{}", encoding="utf-8")
            role = HermesRoleSettings(
                profile="test",
                model="gpt-5.6-terra",
                reasoning="xhigh",
                toolsets="coding",
                max_turns=1,
            )
            settings = HermesSettings(
                root=root,
                executable=executable,
                home=home,
                provider="openai-codex",
                timeout_seconds=10,
                poll_seconds=0.05,
                conversation=HermesModelSettings("gpt-5.6-terra", "xhigh"),
                planning=HermesModelSettings("gpt-5.6-sol", "xhigh"),
                roles={role_id: role for role_id in RoleId},
            )
            runner = HermesRunner(settings, ArtifactStore(root / "artifacts"))

            class FakeProcess:
                def __init__(self):
                    self.returncode = None
                    self.terminated = threading.Event()

                def communicate(self, timeout=None):
                    if self.terminated.wait(timeout):
                        return "", ""
                    raise subprocess.TimeoutExpired("hermes", timeout)

                def poll(self):
                    return self.returncode

                def terminate(self):
                    self.returncode = -15
                    self.terminated.set()

                def wait(self, timeout=None):
                    if not self.terminated.wait(timeout):
                        raise subprocess.TimeoutExpired("hermes", timeout)
                    return self.returncode

                def kill(self):
                    self.returncode = -9
                    self.terminated.set()

            fake_process = FakeProcess()
            errors = []
            popen_called = threading.Event()

            def create_process(*_args, **_kwargs):
                popen_called.set()
                return fake_process

            def invoke():
                try:
                    runner.run(
                        "RUN-STOP",
                        "stage-001",
                        RoleId.DEVELOPMENT,
                        root,
                        "stop test",
                        allow_writes=False,
                    )
                except Exception as exc:
                    errors.append(exc)

            with patch(
                "app.services.hermes.runner.subprocess.Popen",
                side_effect=create_process,
            ):
                thread = threading.Thread(target=invoke)
                thread.start()
                self.assertTrue(popen_called.wait(timeout=1))
                runner.request_stop()
                thread.join(timeout=2)

            self.assertFalse(thread.is_alive())
            self.assertTrue(fake_process.terminated.is_set())
            self.assertEqual(0, runner.active_process_count())
            self.assertEqual(1, len(errors))
            self.assertIsInstance(errors[0], HermesCancelled)

    def test_process_tree_termination_stops_a_real_child_process(self):
        with temporary_directory() as directory:
            root = Path(directory)
            heartbeat = root / "heartbeat"
            child_code = (
                "import sys, time\n"
                "from pathlib import Path\n"
                "path = Path(sys.argv[1])\n"
                "while True:\n"
                "    path.write_text(str(time.monotonic()), encoding='utf-8')\n"
                "    time.sleep(0.05)\n"
            )
            parent_code = (
                "import subprocess, sys, time\n"
                f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}, {str(heartbeat)!r}])\n"
                "print(child.pid, flush=True)\n"
                "while True:\n"
                "    time.sleep(1)\n"
            )
            process = subprocess.Popen(
                [sys._base_executable, "-u", "-c", parent_code],
                text=True,
                encoding="utf-8",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **isolated_process_options(),
            )
            process_tree = ProcessTree(process)
            try:
                child_pid = process.stdout.readline().strip()
                self.assertTrue(child_pid.isdigit())
                deadline = time.monotonic() + 5
                while not heartbeat.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(heartbeat.exists())

                process_tree.terminate(grace_seconds=1)
                time.sleep(0.2)
                stopped_value = heartbeat.read_text(encoding="utf-8")
                time.sleep(0.2)

                self.assertEqual(stopped_value, heartbeat.read_text(encoding="utf-8"))
                self.assertIsNotNone(process.poll())
            finally:
                process_tree.close()
                if process.stdout is not None:
                    process.stdout.close()
                if process.stderr is not None:
                    process.stderr.close()

    @unittest.skipUnless(os.name == "nt", "Windows Job Object 전용 테스트")
    def test_windows_job_assignment_failure_is_fail_closed(self):
        class SuspendedProcess:
            pid = 12345
            _handle = 67890

            def __init__(self):
                self.returncode = None
                self.killed = False

            def poll(self):
                return self.returncode

            def kill(self):
                self.killed = True
                self.returncode = -9

            def wait(self, timeout=None):
                return self.returncode

        process = SuspendedProcess()
        with patch(
            "app.services.process_tree._assign_windows_kill_job", return_value=None
        ), patch("app.services.process_tree._resume_windows_process") as resume:
            with self.assertRaisesRegex(RuntimeError, "Job Object 격리"):
                ProcessTree(process)

        self.assertTrue(process.killed)
        resume.assert_not_called()

    def test_root_identity_rejects_a_reused_pid(self):
        class ReusedProcess:
            @staticmethod
            def create_time():
                return 200.0

        tree = object.__new__(ProcessTree)
        tree._root_identity = (4242, 100.0)
        with patch("psutil.Process", return_value=ReusedProcess()):
            self.assertIsNone(tree._root_process())


if __name__ == "__main__":
    unittest.main()
