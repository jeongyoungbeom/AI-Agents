import unittest
from dataclasses import replace
from pathlib import Path

from app.contracts import RunPhase
from app.orchestrator import ApprovalRequired, InvalidTransition, RunStateMachine
from app.services.logging.audit import AuditLogger
from app.storage import ConcurrentUpdateError, StateStore
from tests.foundation.support import temporary_directory


class StateMachineTests(unittest.TestCase):
    def make_system(self, root: Path):
        store = StateStore(root / "state.db")
        logger = AuditLogger(root / "artifacts", store)
        return store, logger, RunStateMachine(store, logger)

    @staticmethod
    def prepared_run(machine, run_id: str, *, stage_count: int = 1):
        state = machine.create_run(
            run_id,
            repository="D:\\test-repository",
            repository_identity="a" * 64,
            repository_head_sha="b" * 40,
            repository_approved=True,
            stage_count=stage_count,
        )
        return machine.register_plan(
            state,
            {"stages": [{"objective": f"stage-{index + 1}"} for index in range(stage_count)]},
        )

    def test_development_is_blocked_until_exact_approval(self):
        with temporary_directory() as directory:
            store, _, machine = self.make_system(Path(directory))
            state = self.prepared_run(machine, "RUN-APPROVAL")
            state = machine.request_approval(state)
            with self.assertRaises(ApprovalRequired):
                machine.transition(state, RunPhase.DEVELOPING)
            with self.assertRaises(ApprovalRequired):
                machine.approve(state, "응 진행해", "user-1")
            state = machine.approve(state, "개발 시작해", "user-1")
            state = machine.transition(state, RunPhase.DEVELOPING)
            self.assertEqual(RunPhase.DEVELOPING, state.phase)
            self.assertTrue(store.latest_checkpoint(state.run_id).approval_granted)

    def test_invalid_phase_order_is_rejected(self):
        with temporary_directory() as directory:
            _, _, machine = self.make_system(Path(directory))
            state = machine.create_run("RUN-ORDER")
            with self.assertRaises(InvalidTransition):
                machine.transition(state, RunPhase.REVIEWING)

    def test_approval_without_plan_metadata_cannot_be_stored(self):
        with temporary_directory() as directory:
            store, _, machine = self.make_system(Path(directory))
            state = machine.create_run(
                "RUN-LEGACY-APPROVAL",
                repository="D:\\test-repository",
                repository_identity="a" * 64,
                repository_head_sha="b" * 40,
                repository_approved=True,
            )
            with self.assertRaises(ValueError):
                store.save_run(
                    replace(
                        state,
                        phase=RunPhase.WAITING_APPROVAL,
                        approval_granted=True,
                    )
                )

    def test_two_stage_dry_flow_completes_in_order(self):
        with temporary_directory() as directory:
            store, _, machine = self.make_system(Path(directory))
            state = self.prepared_run(machine, "RUN-FLOW", stage_count=2)
            state = machine.request_approval(state)
            state = machine.approve(state, "개발 시작해", "user-1")
            state = machine.transition(state, RunPhase.DEVELOPING)
            for expected_stage in range(2):
                state = machine.transition(state, RunPhase.REVIEWING)
                state = machine.transition(state, RunPhase.IMPROVING)
                state = machine.transition(state, RunPhase.VERIFYING)
                state = machine.complete_stage(state)
                if expected_stage == 0:
                    self.assertEqual(RunPhase.DEVELOPING, state.phase)
                    self.assertEqual(1, state.stage_index)
            self.assertEqual(RunPhase.COMPLETED, state.phase)
            self.assertGreater(len(store.list_events(state.run_id)), 8)

    def test_pause_and_resume_return_to_checkpointed_phase(self):
        with temporary_directory() as directory:
            store, _, machine = self.make_system(Path(directory))
            state = self.prepared_run(machine, "RUN-PAUSE")
            state = machine.request_approval(state)
            state = machine.approve(state, "개발 시작해", "user-1")
            state = machine.transition(state, RunPhase.DEVELOPING)
            state = machine.pause(state, "사용자 중단")
            self.assertEqual(RunPhase.PAUSED, state.phase)
            state = machine.resume(store.load_run(state.run_id))
            self.assertEqual(RunPhase.DEVELOPING, state.phase)

    def test_review_can_branch_to_builder_finsher_or_verification(self):
        with temporary_directory() as directory:
            _, _, machine = self.make_system(Path(directory))
            for run_id, target in (
                ("RUN-DESIGN-BRANCH", RunPhase.DEVELOPING),
                ("RUN-IMPLEMENTATION-BRANCH", RunPhase.IMPROVING),
                ("RUN-CLEAN-BRANCH", RunPhase.VERIFYING),
            ):
                state = self.prepared_run(machine, run_id)
                state = machine.request_approval(state)
                state = machine.approve(state, "개발 시작해", "user-1")
                state = machine.transition(state, RunPhase.DEVELOPING)
                state = machine.transition(state, RunPhase.REVIEWING)
                state = machine.transition(state, target)
                self.assertEqual(target, state.phase)

    def test_finisher_can_return_important_fix_to_reviewer(self):
        with temporary_directory() as directory:
            _, _, machine = self.make_system(Path(directory))
            state = self.prepared_run(machine, "RUN-CLOSURE-REVIEW")
            state = machine.request_approval(state)
            state = machine.approve(state, "개발 시작해", "user-1")
            state = machine.transition(state, RunPhase.DEVELOPING)
            state = machine.transition(state, RunPhase.REVIEWING)
            state = machine.transition(state, RunPhase.IMPROVING)
            state = machine.transition(state, RunPhase.REVIEWING)
            self.assertEqual(RunPhase.REVIEWING, state.phase)

    def test_optimistic_version_prevents_lost_updates(self):
        with temporary_directory() as directory:
            store, _, machine = self.make_system(Path(directory))
            original = self.prepared_run(machine, "RUN-VERSION")
            machine.request_approval(original)
            with self.assertRaises(ConcurrentUpdateError):
                store.save_run(original)


if __name__ == "__main__":
    unittest.main()
