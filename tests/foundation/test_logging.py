import json
import unittest
from pathlib import Path

from app.contracts import AgentHandoff, RoleId, RunState, StageContract
from app.services.logging.audit import AuditLogger
from app.storage import ArtifactStore, StateStore
from tests.foundation.support import temporary_directory


class LoggingTests(unittest.TestCase):
    def test_event_is_written_to_database_jsonl_and_timeline(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.db")
            store.create_run(RunState(run_id="RUN-LOG"))
            logger = AuditLogger(root / "artifacts", store)
            logger.emit(
                "RUN-LOG",
                "AGENT_FAILED",
                "Authorization: Bearer abcdefghijklmnop",
                stage_id="stage-001",
                role_id="development",
                status="FAILED",
                data={"telegram_bot_token": "123456:abcdefghijklmnopqrstuv"},
            )
            run_dir = root / "artifacts" / "RUN-LOG"
            timeline = (run_dir / "timeline.log").read_text(encoding="utf-8")
            event = json.loads(
                (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertIn("[REDACTED]", timeline)
            self.assertNotIn("abcdefghijklmnop", timeline)
            self.assertEqual("[REDACTED]", event["data"]["telegram_bot_token"])
            self.assertEqual("[REDACTED]", store.list_events("RUN-LOG")[0]["data"]["telegram_bot_token"])

    def test_artifact_store_writes_readable_contract_and_handoff(self):
        with temporary_directory() as directory:
            root = Path(directory)
            artifacts = ArtifactStore(root / "artifacts")
            contract = StageContract(
                run_id="RUN-ARTIFACT",
                stage_id="stage-001",
                objective="Create logs",
                scope=("app/",),
                acceptance_criteria=("Files exist",),
                verification_commands=("python -m unittest",),
            )
            handoff = AgentHandoff(
                contract=contract,
                from_role=RoleId.DEVELOPMENT,
                to_role=RoleId.REVIEW,
                summary="Development completed",
                changed_files=("app/example.py",),
            )
            contract_path = artifacts.save_contract(contract)
            handoff_path = artifacts.save_handoff(handoff)
            artifacts.append_role_log(
                contract.run_id,
                contract.stage_id,
                "development",
                "token=secret-value",
            )
            self.assertTrue(contract_path.is_file())
            self.assertTrue(handoff_path.is_file())
            role_log = contract_path.parent / "development.log"
            self.assertIn("[REDACTED]", role_log.read_text(encoding="utf-8"))

    def test_artifact_paths_cannot_escape_run_directory(self):
        with temporary_directory() as directory:
            artifacts = ArtifactStore(Path(directory) / "artifacts")
            with self.assertRaises(ValueError):
                artifacts.write_text("RUN-SAFE", Path("..") / "escape.txt", "no")

    def test_summary_contains_last_state_and_events(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.db")
            state = RunState(
                run_id="RUN-SUMMARY", objective="Foundation", repository="D:\\repo"
            )
            store.create_run(state)
            logger = AuditLogger(root / "artifacts", store)
            logger.emit("RUN-SUMMARY", "RUN_CREATED", "시작")
            path = logger.write_summary(store.load_run("RUN-SUMMARY"))
            summary = path.read_text(encoding="utf-8")
            self.assertIn("RUN-SUMMARY", summary)
            self.assertIn("Foundation", summary)
            self.assertIn("RUN_CREATED", summary)


if __name__ == "__main__":
    unittest.main()
