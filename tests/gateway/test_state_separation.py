from __future__ import annotations

import json
import sqlite3
import threading
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from app.contracts import RoleId, RunState
from app.gateway.core import AccessPolicy, GatewayApplication, OutgoingMessage
from app.orchestrator import RunStateMachine
from app.contracts.models import utc_now
from app.storage import StateStore
from tests.gateway.support import (
    TEST_REPOSITORY_HEAD,
    TEST_REPOSITORY_IDENTITY,
    build_application,
    future_expiry,
    temporary_directory,
)
from tests.gateway.test_conversation import ReadyBackend, incoming


class StateSeparationTests(unittest.TestCase):
    def test_ungranted_execution_approval_rejects_partial_metadata(self):
        with self.assertRaises(ValueError):
            RunState(
                "RUN-PARTIAL-APPROVAL",
                approved_plan_revision=1,
            )

    def test_free_chat_session_has_no_active_task(self):
        with temporary_directory() as directory:
            store, application = build_application(Path(directory), ReadyBackend())

            response = application.handle(incoming(1, "안녕"))

            self.assertTrue(response)
            session = store.load_conversation_session("telegram", "200")
            binding = store.load_conversation("telegram", "200")
            self.assertIsNotNone(session)
            self.assertIsNotNone(binding)
            assert session is not None and binding is not None
            self.assertEqual("", session.active_task_id)
            self.assertEqual(session.session_run_id, binding["run_id"])

    def test_run_json_contains_only_execution_state(self):
        with temporary_directory() as directory:
            database = Path(directory) / "state.db"
            store = StateStore(database)
            plan_hash = "a" * 64
            state = RunState(
                run_id="RUN-SPLIT",
                repository="D:\\projects\\sample",
                repository_identity=TEST_REPOSITORY_IDENTITY,
                repository_head_sha=TEST_REPOSITORY_HEAD,
                repository_approved=True,
                objective="기능 구현",
                plan_revision=1,
                plan_hash=plan_hash,
                approval_granted=True,
                approved_plan_revision=1,
                approved_plan_hash=plan_hash,
                approved_at=utc_now(),
                approved_by="100",
            )
            store.create_run(state)

            with closing(sqlite3.connect(database)) as connection:
                raw = json.loads(
                    connection.execute(
                        "SELECT state_json FROM runs WHERE run_id = ?", (state.run_id,)
                    ).fetchone()[0]
                )

            self.assertNotIn("repository", raw)
            self.assertNotIn("objective", raw)
            self.assertNotIn("plan_revision", raw)
            self.assertNotIn("approval_granted", raw)
            self.assertEqual("기능 구현", store.load_task_definition(state.run_id).objective)
            self.assertEqual(1, store.load_plan_pointer(state.run_id).revision)
            self.assertTrue(store.load_execution_approval(state.run_id).granted)
            self.assertEqual(state, store.load_run(state.run_id))

    def test_project_role_and_active_task_change_independently(self):
        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            store.create_run(RunState("RUN-FIRST"))
            store.create_run(RunState("RUN-SECOND", objective="두 번째 작업"))
            store.bind_conversation(
                "telegram", "200", "100", "RUN-FIRST", RoleId.DEVELOPMENT.value
            )
            project = store.set_current_project(
                "telegram", "200", "100", "D:\\projects\\sample",
                repository_identity=TEST_REPOSITORY_IDENTITY,
                head_sha=TEST_REPOSITORY_HEAD,
                approved=True,
                approval_expires_at=future_expiry(),
            )

            store.set_conversation_role("telegram", "200", RoleId.REVIEW.value)
            store.set_conversation_task("telegram", "200", "RUN-SECOND")

            session = store.load_conversation_session("telegram", "200")
            selected = store.load_project_selection("telegram", "200")
            self.assertIsNotNone(session)
            self.assertIsNotNone(selected)
            assert session is not None and selected is not None
            self.assertEqual(RoleId.REVIEW, session.active_role)
            self.assertEqual("RUN-SECOND", session.active_task_id)
            self.assertEqual(project.repository_path, selected.repository_path)
            self.assertTrue(selected.approved)
            self.assertEqual("", store.load_task_definition("RUN-FIRST").objective)
            self.assertEqual(
                "두 번째 작업", store.load_task_definition("RUN-SECOND").objective
            )

    def test_new_task_keeps_the_current_project_but_not_the_old_objective(self):
        with temporary_directory() as directory:
            store, application = build_application(Path(directory), ReadyBackend())
            application.handle(incoming(1, "로그인 기능을 만들어줘"))
            application.handle(incoming(2, "D:\\projects\\sample"))
            application.handle(incoming(3, "이 프로젝트 사용 승인해"))
            old_binding = store.load_conversation("telegram", "200")
            self.assertIsNotNone(old_binding)
            store.set_conversation_role("telegram", "200", RoleId.REVIEW.value)

            application.handle(incoming(4, "새 작업"))

            new_binding = store.load_conversation("telegram", "200")
            self.assertIsNotNone(new_binding)
            assert old_binding is not None and new_binding is not None
            self.assertNotEqual(old_binding["run_id"], new_binding["run_id"])
            new_state = store.load_run(new_binding["run_id"])
            selected = store.load_project_selection("telegram", "200")
            self.assertIsNotNone(selected)
            assert selected is not None
            self.assertEqual(selected.repository_path, new_state.repository)
            self.assertTrue(new_state.repository_approved)
            self.assertEqual("", new_state.objective)
            self.assertEqual(RoleId.REVIEW.value, new_binding["active_role"])

    def test_split_state_survives_a_store_restart(self):
        with temporary_directory() as directory:
            database = Path(directory) / "state.db"
            first = StateStore(database)
            first.create_run(RunState("RUN-RESTART", objective="재시작 작업"))
            first.bind_conversation(
                "telegram", "200", "100", "RUN-RESTART", RoleId.IMPROVEMENT.value
            )
            first.set_current_project(
                "telegram", "200", "100", "D:\\projects\\restart",
                repository_identity=TEST_REPOSITORY_IDENTITY,
                head_sha=TEST_REPOSITORY_HEAD,
                approved=True,
                approval_expires_at=future_expiry(),
            )

            restarted = StateStore(database)
            session = restarted.load_conversation_session("telegram", "200")
            project = restarted.load_project_selection("telegram", "200")
            self.assertEqual("RUN-RESTART", session.active_task_id if session else None)
            self.assertEqual(RoleId.IMPROVEMENT, session.active_role if session else None)
            self.assertTrue(project.approved if project else False)
            self.assertEqual(
                "재시작 작업",
                restarted.load_task_definition("RUN-RESTART").objective,
            )

    def test_legacy_combined_state_is_migrated_without_losing_data(self):
        with temporary_directory() as directory:
            database = Path(directory) / "legacy.db"
            state = RunState(
                "RUN-LEGACY-SPLIT",
                repository="D:\\projects\\legacy",
                repository_identity=TEST_REPOSITORY_IDENTITY,
                repository_head_sha=TEST_REPOSITORY_HEAD,
                repository_approved=True,
                objective="기존 작업",
            )
            legacy_payload = state.to_dict()
            legacy_payload.pop("repository_identity")
            legacy_payload.pop("repository_head_sha")
            payload = json.dumps(legacy_payload, ensure_ascii=False, sort_keys=True)
            with closing(sqlite3.connect(database)) as connection:
                connection.executescript(
                    """
                    CREATE TABLE runs (
                        run_id TEXT PRIMARY KEY, state_json TEXT NOT NULL,
                        version INTEGER NOT NULL, created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE checkpoints (
                        checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                        phase TEXT NOT NULL, state_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    CREATE TABLE conversation_bindings (
                        channel TEXT NOT NULL, conversation_id TEXT NOT NULL,
                        user_id TEXT NOT NULL, run_id TEXT NOT NULL REFERENCES runs(run_id),
                        active_role TEXT NOT NULL, mode TEXT NOT NULL DEFAULT 'free_chat',
                        created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                        PRIMARY KEY(channel, conversation_id)
                    );
                    """
                )
                connection.execute(
                    "INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
                    (state.run_id, payload, 0, state.created_at, state.updated_at),
                )
                connection.execute(
                    "INSERT INTO checkpoints(run_id, phase, state_json, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (state.run_id, state.phase.value, payload, state.created_at),
                )
                connection.execute(
                    "INSERT INTO conversation_bindings VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "telegram",
                        "200",
                        "100",
                        state.run_id,
                        RoleId.REVIEW.value,
                        "free_chat",
                        state.created_at,
                        state.updated_at,
                    ),
                )
                connection.commit()

            migrated = StateStore(database)

            migrated_state = migrated.load_run(state.run_id)
            self.assertFalse(migrated_state.repository_approved)
            self.assertEqual("", migrated_state.repository_identity)
            self.assertEqual(state.objective, migrated_state.objective)
            session = migrated.load_conversation_session("telegram", "200")
            project = migrated.load_project_selection("telegram", "200")
            self.assertEqual(RoleId.REVIEW, session.active_role if session else None)
            self.assertEqual(state.repository, project.repository_path if project else None)
            self.assertFalse(project.approved if project else True)
            self.assertFalse(
                migrated.repository_is_approved(
                    "telegram", "200", "100", state.repository, ""
                )
            )
            with closing(sqlite3.connect(database)) as connection:
                raw = json.loads(
                    connection.execute(
                        "SELECT state_json FROM runs WHERE run_id = ?", (state.run_id,)
                    ).fetchone()[0]
                )
            self.assertNotIn("repository", raw)

    def test_execution_approval_is_invalidated_by_a_new_plan(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store, application = build_application(root, ReadyBackend())
            application.handle(incoming(1, "기능 개발"))
            application.handle(incoming(2, "D:\\projects\\sample"))
            application.handle(incoming(3, "이 프로젝트 사용 승인해"))
            binding = store.load_conversation("telegram", "200")
            state = store.load_run(binding["run_id"] if binding else "")
            application.router.state_machine.approve(state, "개발 시작해", "100")
            approved = store.load_execution_approval(state.run_id)
            self.assertTrue(approved.granted)

            current = store.load_run(state.run_id)
            revised = application.router.state_machine.register_plan(
                current,
                {
                    "stages": [
                        {
                            "objective": "다른 구현",
                            "scope": ["app/"],
                            "acceptance_criteria": ["테스트 통과"],
                            "verification_commands": ["python -m unittest"],
                        }
                    ]
                },
            )

            self.assertFalse(revised.approval_granted)
            self.assertFalse(store.load_execution_approval(state.run_id).granted)

    def test_repository_approval_is_scoped_to_channel_and_conversation(self):
        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            repository = "D:\\projects\\sample"
            store.approve_repository(
                "telegram", "200", "100", repository,
                TEST_REPOSITORY_IDENTITY, future_expiry(),
            )

            self.assertTrue(
                store.repository_is_approved(
                    "telegram", "200", "100", repository,
                    TEST_REPOSITORY_IDENTITY,
                )
            )
            self.assertFalse(
                store.repository_is_approved(
                    "discord", "200", "100", repository,
                    TEST_REPOSITORY_IDENTITY,
                )
            )
            self.assertFalse(
                store.repository_is_approved(
                    "telegram", "201", "100", repository,
                    TEST_REPOSITORY_IDENTITY,
                )
            )

    def test_legacy_migration_runs_once_and_does_not_resurrect_binding(self):
        with temporary_directory() as directory:
            database = Path(directory) / "state.db"
            store = StateStore(database)
            store.create_run(RunState("RUN-MIGRATION-ONCE"))
            now = utc_now()
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    "INSERT INTO conversation_bindings(channel, conversation_id, user_id, "
                    "run_id, active_role, mode, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "telegram", "legacy", "100", "RUN-MIGRATION-ONCE",
                        RoleId.DEVELOPMENT.value, "free_chat", now, now,
                    ),
                )
                connection.commit()

            restarted = StateStore(database)
            self.assertIsNone(restarted.load_conversation("telegram", "legacy"))

    def test_plan_and_approval_updates_roll_back_as_units(self):
        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            machine = RunStateMachine(store)
            state = machine.create_run(
                "RUN-ATOMIC-PLAN",
                repository="D:\\projects\\sample",
                repository_identity=TEST_REPOSITORY_IDENTITY,
                repository_head_sha=TEST_REPOSITORY_HEAD,
                repository_approved=True,
            )
            with patch.object(store, "save_run", side_effect=RuntimeError("fail")):
                with self.assertRaises(RuntimeError):
                    machine.register_plan(
                        state, {"stages": [{"objective": "one"}]}
                    )
            self.assertIsNone(store.load_plan_revision(state.run_id, 1))

            planned = machine.register_plan(
                store.load_run(state.run_id),
                {"stages": [{"objective": "one"}]},
            )
            waiting = machine.request_approval(planned)
            with patch.object(store, "save_run", side_effect=RuntimeError("fail")):
                with self.assertRaises(RuntimeError):
                    machine.approve(waiting, "개발 시작해", "100")
            self.assertEqual(
                "CURRENT",
                store.load_plan_revision(state.run_id, 1)["status"],
            )
            self.assertFalse(store.load_execution_approval(state.run_id).granted)

    def test_slow_router_does_not_hold_sqlite_write_transaction(self):
        class BlockingRouter:
            entered = threading.Event()
            release = threading.Event()

            @staticmethod
            def is_immediate_control(_message):
                return True

            @staticmethod
            def is_stop_control(_message):
                return False

            def route(self, message):
                self.entered.set()
                self.release.wait(timeout=2)
                return (
                    OutgoingMessage(
                        message.channel, message.conversation_id, "ok"
                    ),
                )

        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            router = BlockingRouter()
            application = GatewayApplication(
                store,
                router,
                AccessPolicy(allowed_users=frozenset({"100"})),
            )
            worker = threading.Thread(
                target=application.handle,
                args=(incoming(99, "hello"),),
            )
            worker.start()
            self.assertTrue(router.entered.wait(timeout=1))

            store.save_gateway_cursor("telegram", "during-agent-call")

            router.release.set()
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(
                "during-agent-call", store.load_gateway_cursor("telegram")
            )

    def test_project_selection_and_approval_roll_back_on_mid_write_failure(self):
        with temporary_directory() as directory:
            store, application = build_application(Path(directory), ReadyBackend())
            application.handle(incoming(1, "기능 개발"))
            with patch.object(
                store, "set_current_project", side_effect=RuntimeError("fail")
            ):
                with self.assertRaises(RuntimeError):
                    application.handle(incoming(2, "D:\\projects\\sample"))
            binding = store.load_conversation("telegram", "200")
            state = store.load_run(binding["run_id"] if binding else "")
            self.assertEqual("", state.repository)
            self.assertIsNone(store.load_project_selection("telegram", "200"))

            application.handle(incoming(3, "D:\\projects\\sample"))
            with patch.object(
                store,
                "set_current_project",
                side_effect=RuntimeError("fail"),
            ):
                with self.assertRaises(RuntimeError):
                    application.handle(incoming(4, "이 프로젝트 사용 승인해"))
            state = store.load_run(binding["run_id"] if binding else "")
            self.assertFalse(state.repository_approved)
            self.assertFalse(
                store.repository_is_approved(
                    "telegram", "200", "100", state.repository,
                    state.repository_identity,
                )
            )

    def test_plan_approval_and_pipeline_enqueue_roll_back_together(self):
        class FailingPipeline:
            @staticmethod
            def enqueue(_run_id, _channel, _conversation_id):
                raise RuntimeError("queue failed")

            @staticmethod
            def status(_run_id):
                return None

            @staticmethod
            def cancel(_run_id):
                return None

        with temporary_directory() as directory:
            store, application = build_application(
                Path(directory), ReadyBackend(), pipeline_scheduler=FailingPipeline()
            )
            application.handle(incoming(1, "기능 개발"))
            application.handle(incoming(2, "D:\\projects\\sample"))
            application.handle(incoming(3, "이 프로젝트 사용 승인해"))
            binding = store.load_conversation("telegram", "200")
            with self.assertRaises(RuntimeError):
                application.handle(incoming(4, "개발 시작해"))

            state = store.load_run(binding["run_id"] if binding else "")
            self.assertFalse(state.approval_granted)
            self.assertEqual(
                "CURRENT",
                store.load_plan_revision(state.run_id, state.plan_revision)["status"],
            )

    def test_upgraded_conversation_session_has_session_run_foreign_key(self):
        with temporary_directory() as directory:
            database = Path(directory) / "old-5a.db"
            state = RunState("RUN-OLD-SESSION")
            with closing(sqlite3.connect(database)) as connection:
                connection.executescript(
                    """
                    PRAGMA foreign_keys = ON;
                    CREATE TABLE runs (
                        run_id TEXT PRIMARY KEY, state_json TEXT NOT NULL,
                        version INTEGER NOT NULL, created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE conversation_sessions (
                        channel TEXT NOT NULL, conversation_id TEXT NOT NULL,
                        user_id TEXT NOT NULL, active_role TEXT NOT NULL,
                        mode TEXT NOT NULL, active_task_id TEXT REFERENCES runs(run_id),
                        created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                        PRIMARY KEY(channel, conversation_id)
                    );
                    """
                )
                connection.execute(
                    "INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
                    (
                        state.run_id,
                        json.dumps(state.to_dict(), ensure_ascii=False),
                        state.version,
                        state.created_at,
                        state.updated_at,
                    ),
                )
                connection.execute(
                    "INSERT INTO conversation_sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "telegram", "200", "100", RoleId.DEVELOPMENT.value,
                        "free_chat", state.run_id, state.created_at, state.updated_at,
                    ),
                )
                connection.commit()

            StateStore(database)

            with closing(sqlite3.connect(database)) as connection:
                connection.execute("PRAGMA foreign_keys = ON")
                keys = {
                    row[3]
                    for row in connection.execute(
                        "PRAGMA foreign_key_list(conversation_sessions)"
                    )
                }
                self.assertIn("session_run_id", keys)
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                        "UPDATE conversation_sessions SET session_run_id = 'MISSING'"
                    )


if __name__ == "__main__":
    unittest.main()
