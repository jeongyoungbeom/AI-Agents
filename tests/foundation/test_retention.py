from __future__ import annotations

import os
import sqlite3
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.contracts import RunPhase, RunState
from app.contracts import TokenUsage
from app.services.retention import RetentionManager, RetentionPolicy
from app.storage import StateStore, StoreError
from tests.foundation.support import temporary_directory


class RetentionTests(unittest.TestCase):
    def test_purge_clears_inactive_old_session_and_project_selection(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store = StateStore(root / "data" / "state.db")
            store.create_run(RunState("RUN-OLD-SESSION"))
            store.bind_conversation("telegram", "chat", "person", "RUN-OLD-SESSION", "development")
            store.clear_conversation_task("telegram", "chat")
            store.set_current_project("telegram", "chat", "person", str(root / "project"))
            old = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
            connection = sqlite3.connect(store.path)
            try:
                connection.execute("UPDATE conversation_sessions SET updated_at = ?", (old,))
                connection.execute("UPDATE conversation_projects SET selected_at = ?", (old,))
                connection.commit()
            finally:
                connection.close()
            result = RetentionManager(root, store, RetentionPolicy()).purge()
            self.assertEqual(1, result.database_rows["conversation_sessions"])
            self.assertEqual(1, result.database_rows["conversation_projects"])
            self.assertIsNone(store.load_conversation_session("telegram", "chat"))
            self.assertIsNone(store.load_project_selection("telegram", "chat"))

    def test_purge_removes_expired_scoped_memories_and_scope_delete(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store = StateStore(root / "data" / "state.db")
            for scope in ("user", "project", "conversation"):
                store.save_memory(scope, "key", "shared", f"{scope} fact")
            self.assertEqual(1, store.delete_memory("project", "key"))
            self.assertEqual([], store.list_memories((("project", "key"),)))
            now = datetime.now(timezone.utc) + timedelta(days=31)
            result = RetentionManager(root, store, RetentionPolicy()).purge(now=now)
            self.assertEqual(1, result.database_rows["user_memories"])
            self.assertEqual(1, result.database_rows["conversation_memories"])
            self.assertEqual(1, result.database_rows["project_memories"])

    def test_backup_can_restore_a_consistent_database_snapshot(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store = StateStore(root / "data" / "state.db")
            store.create_run(RunState("RUN-BACKUP"))
            store.save_memory("user", "person", "shared", "복구할 사실")
            manager = RetentionManager(root, store, RetentionPolicy())
            backup = manager.backup_database(
                now=datetime(2026, 1, 2, tzinfo=timezone.utc)
            )
            store.create_run(RunState("RUN-AFTER-BACKUP"))

            manager.restore_database(backup)

            restored = StateStore(root / "data" / "state.db")
            self.assertEqual("RUN-BACKUP", restored.load_run("RUN-BACKUP").run_id)
            self.assertEqual("복구할 사실", restored.list_memories(
                (("user", "person"),)
            )[0]["content"])
            with self.assertRaises(StoreError):
                restored.load_run("RUN-AFTER-BACKUP")

    def test_purge_removes_expired_terminal_data_but_keeps_active_artifacts(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store = StateStore(root / "data" / "state.db")
            store.create_run(RunState("RUN-ACTIVE"))
            store.create_run(RunState("RUN-DONE", phase=RunPhase.COMPLETED))
            store.add_usage(
                "RUN-DONE",
                "stage-001",
                "development",
                "agent",
                TokenUsage(total_tokens=10),
            )
            outbound = store.queue_outbound("telegram", "chat", "완료")
            store.complete_outbound(outbound, ())
            active_artifact = root / "artifacts" / "RUN-ACTIVE" / "recovery.json"
            done_artifact = root / "artifacts" / "RUN-DONE" / "recovery.json"
            attachment = root / "data" / "attachments" / "old.txt"
            log = root / "logs" / "gateway.log"
            for path in (active_artifact, done_artifact, attachment, log):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("old", encoding="utf-8")
            now = datetime.now(timezone.utc) + timedelta(days=2)
            old_timestamp = (now - timedelta(days=2)).timestamp()
            for path in (active_artifact, done_artifact, attachment, log):
                os.utime(path, (old_timestamp, old_timestamp))
            os.utime(done_artifact.parent, (old_timestamp, old_timestamp))
            policy = RetentionPolicy(
                database_ttl_days=1,
                artifact_ttl_days=1,
                attachment_ttl_days=1,
                log_ttl_days=1,
                backup_ttl_days=1,
            )

            result = RetentionManager(root, store, policy).purge(now=now)

            self.assertTrue(active_artifact.is_file())
            self.assertFalse(done_artifact.parent.exists())
            self.assertFalse(attachment.exists())
            self.assertFalse(log.exists())
            self.assertEqual(1, result.database_rows["outbound"])
            self.assertEqual(1, result.database_rows["runs"])
            self.assertEqual("RUN-ACTIVE", store.load_run("RUN-ACTIVE").run_id)
            with self.assertRaises(StoreError):
                store.load_run("RUN-DONE")
            self.assertEqual(0, store.usage_total("RUN-DONE"))
