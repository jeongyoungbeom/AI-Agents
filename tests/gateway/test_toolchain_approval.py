from __future__ import annotations

import unittest
from pathlib import Path

from app.contracts import RunPhase, StageContract
from app.gateway.core import AgentReply, IncomingMessage
from app.services.toolchains import ToolchainPreflightError
from app.services.git import GitRepository
from tests.gateway.support import (
    TEST_REPOSITORY_HEAD,
    TEST_REPOSITORY_IDENTITY,
    build_application,
    future_expiry,
    temporary_directory,
)


class _UnusedBackend:
    def respond(self, *_args, **_kwargs) -> AgentReply:
        return AgentReply("not used")


class _PipelineScheduler:
    def __init__(self) -> None:
        self.enqueued: list[str] = []

    def enqueue(self, run_id, _channel, _conversation_id):
        self.enqueued.append(run_id)
        return "QUEUED"

    def cancel(self, _run_id):
        return None

    def pause(self, _run_id):
        return None

    def resume(self, _run_id):
        return None

    def status(self, _run_id):
        return None


class _ToolchainPreflight:
    def __init__(self, error: ToolchainPreflightError | None = None) -> None:
        self.error = error
        self.calls: list[tuple[Path, tuple[str, ...], str]] = []

    def preflight(self, repository, commands, *, operation_id=None, **_kwargs) -> None:
        if not isinstance(repository, GitRepository):
            raise TypeError('실제 ToolchainService는 GitRepository를 받습니다.')
        self.calls.append((repository.path, tuple(commands), str(operation_id)))
        if self.error is not None:
            raise self.error


def incoming(identifier: str, text: str) -> IncomingMessage:
    return IncomingMessage("telegram", "200", "100", identifier, text)


class ToolchainApprovalTests(unittest.TestCase):
    def _planned_application(self, root: Path, preflight: _ToolchainPreflight):
        (root / 'selected-repository').mkdir(exist_ok=True)
        scheduler = _PipelineScheduler()
        store, application = build_application(
            root,
            _UnusedBackend(),
            pipeline_scheduler=scheduler,
            execution_preflight=preflight,
        )
        machine = application.router.state_machine
        state = machine.create_run(
            "RUN-TOOLCHAIN-APPROVAL",
            repository=str(root / "selected-repository"),
            repository_identity=TEST_REPOSITORY_IDENTITY,
            repository_head_sha=TEST_REPOSITORY_HEAD,
            repository_approved=True,
            objective="fixture",
        )
        contract = StageContract(
            run_id=state.run_id,
            stage_id="stage-001",
            objective="fixture",
            scope=("app/",),
            acceptance_criteria=("passes",),
            verification_commands=("python -m unittest",),
        )
        state = machine.register_plan(state, {"stages": [contract.to_dict()]})
        state = machine.request_approval(state)
        self.assertEqual(RunPhase.WAITING_APPROVAL, state.phase)
        store.bind_conversation("telegram", "200", "100", state.run_id, "development")
        store.approve_repository(
            "telegram", "200", "100", state.repository,
            TEST_REPOSITORY_IDENTITY, future_expiry(),
        )
        return store, application, scheduler, state.run_id

    def test_preflight_failure_keeps_approval_and_pipeline_queue_unmodified(self):
        with temporary_directory() as directory:
            preflight = _ToolchainPreflight(
                ToolchainPreflightError("tool", "python 3.11 is missing")
            )
            store, application, scheduler, run_id = self._planned_application(Path(directory), preflight)

            reply = application.handle(incoming("approval-fails", "개발 시작해"))
            self.assertIn("실행 환경", reply[0].text)
            self.assertIn("python", reply[0].text)
            self.assertFalse(store.load_run(run_id).approval_granted)
            self.assertEqual([], scheduler.enqueued)
            self.assertEqual([("python -m unittest",)], [call[1] for call in preflight.calls])

    def test_preflight_success_happens_before_approval_and_queue_registration(self):
        with temporary_directory() as directory:
            preflight = _ToolchainPreflight()
            store, application, scheduler, run_id = self._planned_application(Path(directory), preflight)

            reply = application.handle(incoming("approval-passes", "개발 시작해"))
            self.assertIn("실행 큐", reply[0].text)
            self.assertTrue(store.load_run(run_id).approval_granted)
            self.assertEqual([run_id], scheduler.enqueued)
            self.assertEqual(run_id, preflight.calls[0][2])


if __name__ == "__main__":
    unittest.main()
