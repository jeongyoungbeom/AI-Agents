import unittest
from pathlib import Path

from app.contracts import AgentHandoff, RoleId, RunPhase, StageContract, TokenUsage
from app.orchestrator import RunStateMachine
from app.services.budget import BudgetManager, BudgetPolicy
from app.services.context import ContextService
from app.services.logging.audit import AuditLogger
from app.storage import ArtifactStore, StateStore
from tests.foundation.support import temporary_directory


class FoundationDryRunTests(unittest.TestCase):
    def test_fake_job_produces_checkpoint_handoff_usage_and_summary(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.db")
            logger = AuditLogger(root / "artifacts", store)
            artifacts = ArtifactStore(root / "artifacts")
            machine = RunStateMachine(store, logger)
            context = ContextService(store)
            budget = BudgetManager(
                BudgetPolicy(whole_task_tokens=1000, completion_reserve_tokens=100),
                store,
            )

            state = machine.create_run(
                "RUN-DRY",
                repository="D:\\example-repository",
                repository_identity="a" * 64,
                repository_head_sha="b" * 40,
                repository_approved=True,
                objective="가짜 기반 작업",
            )
            context.add_message("RUN-DRY", "user", "설계부터 이야기하자")
            context.add_decision("RUN-DRY", "단계별 파이프라인을 사용한다")
            contract = StageContract(
                run_id="RUN-DRY",
                stage_id="stage-001",
                objective="가짜 구현",
                scope=("app/",),
                acceptance_criteria=("가짜 검증 통과",),
                verification_commands=("fake-test",),
            )
            state = machine.register_plan(
                state, {"stages": [contract.to_dict()]}
            )
            state = machine.request_approval(state)
            state = machine.approve(state, "개발 시작해", "test-user")
            state = machine.transition(state, RunPhase.DEVELOPING)
            artifacts.save_contract(contract)
            artifacts.save_handoff(
                AgentHandoff(
                    contract=contract,
                    from_role=RoleId.DEVELOPMENT,
                    to_role=RoleId.REVIEW,
                    summary="가짜 개발 완료",
                    changed_files=("app/fake.py",),
                )
            )
            budget.record_usage(
                "RUN-DRY",
                "stage-001",
                "development",
                "agent",
                TokenUsage(input_tokens=100, output_tokens=25),
            )

            state = store.load_run("RUN-DRY")
            state = machine.transition(state, RunPhase.REVIEWING)
            state = machine.transition(state, RunPhase.IMPROVING)
            state = machine.transition(state, RunPhase.VERIFYING)
            state = machine.complete_stage(state)
            summary = logger.write_summary(state)

            self.assertEqual(RunPhase.COMPLETED, state.phase)
            self.assertEqual(125, state.total_tokens)
            self.assertTrue(summary.is_file())
            self.assertTrue(
                (root / "artifacts" / "RUN-DRY" / "events.jsonl").is_file()
            )
            self.assertTrue(
                root.joinpath(
                    "artifacts",
                    "RUN-DRY",
                    "stages",
                    "stage-001",
                    "handoff-development-to-review.json",
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
