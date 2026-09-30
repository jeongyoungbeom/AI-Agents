from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import warnings
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from app.contracts import (
    ConversationSessionState,
    ExecutionApprovalState,
    PlanPointerState,
    ProjectSelectionState,
    RoleId,
    RunState,
    TaskDefinitionState,
    TokenUsage,
)
from app.contracts.models import utc_now
from app.services.repository.analysis import (
    RepositoryAnalysisPhase,
    RepositoryAnalysisRequest,
    RepositoryAnalysisStatus,
)


class StoreError(RuntimeError):
    pass


class ConcurrentUpdateError(StoreError):
    pass


_RUN_EXTERNAL_FIELDS = {
    "repository",
    "repository_identity",
    "repository_head_sha",
    "objective",
    "repository_approved",
    "plan_revision",
    "plan_hash",
    "approval_granted",
    "approved_plan_revision",
    "approved_plan_hash",
    "approved_at",
    "approved_by",
}


class StateStore:
    """SQLite-backed source of truth for runs, checkpoints, usage, and messages."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._local = threading.local()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Commit or roll back, then always release the Windows file handle."""
        active = getattr(self._local, "connection", None)
        if active is not None:
            yield active
            return
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """여러 저장소 호출을 하나의 SQLite 트랜잭션으로 묶는다."""
        if getattr(self._local, "connection", None) is not None:
            yield
            return
        callbacks: list = []
        with self._lock:
            connection = self._connect()
            self._local.connection = connection
            self._local.after_commit = callbacks
            committed = False
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield
                connection.commit()
                committed = True
            except Exception:
                connection.rollback()
                raise
            finally:
                del self._local.connection
                del self._local.after_commit
                connection.close()
            if committed:
                for callback in callbacks:
                    try:
                        callback()
                    except Exception as exc:
                        warnings.warn(
                            f"post-commit callback failed: {type(exc).__name__}",
                            RuntimeWarning,
                            stacklevel=2,
                        )

    def after_commit(self, callback) -> None:
        callbacks = getattr(self._local, "after_commit", None)
        if callbacks is None:
            callback()
            return
        callbacks.append(callback)

    def _initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            state_json TEXT NOT NULL,
            version INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS checkpoints (
            checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            phase TEXT NOT NULL,
            state_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            timestamp TEXT NOT NULL,
            stage_id TEXT NOT NULL,
            role_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            status TEXT NOT NULL,
            message TEXT NOT NULL,
            data_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            timestamp TEXT NOT NULL,
            sender TEXT NOT NULL,
            kind TEXT NOT NULL,
            content TEXT NOT NULL,
            data_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS usage_ledger (
            usage_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            stage_id TEXT NOT NULL,
            role_id TEXT NOT NULL,
            category TEXT NOT NULL,
            input_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            total_tokens INTEGER NOT NULL,
            estimated INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS retry_ledger (
            retry_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            stage_id TEXT NOT NULL,
            category TEXT NOT NULL,
            attempt_number INTEGER NOT NULL,
            reason TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS budget_warning_ledger (
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            scope TEXT NOT NULL,
            stage_id TEXT NOT NULL DEFAULT '',
            threshold_percent INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING',
            created_at TEXT NOT NULL,
            PRIMARY KEY(run_id, scope, stage_id, threshold_percent)
        );
        CREATE TABLE IF NOT EXISTS token_reservations (
            reservation_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            stage_id TEXT NOT NULL,
            role_id TEXT NOT NULL,
            category TEXT NOT NULL,
            reserved_tokens INTEGER NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            settled_at TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS conversation_bindings (
            channel TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            run_id TEXT NOT NULL REFERENCES runs(run_id),
            active_role TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'free_chat',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(channel, conversation_id)
        );
        CREATE TABLE IF NOT EXISTS conversation_sessions (
            channel TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            active_role TEXT NOT NULL,
            mode TEXT NOT NULL,
            session_run_id TEXT NOT NULL REFERENCES runs(run_id),
            active_task_id TEXT REFERENCES runs(run_id),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(channel, conversation_id)
        );
        CREATE TABLE IF NOT EXISTS conversation_projects (
            channel TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            repository_path TEXT NOT NULL,
            repository_identity TEXT NOT NULL DEFAULT '',
            head_sha TEXT NOT NULL DEFAULT '',
            approved INTEGER NOT NULL,
            selected_at TEXT NOT NULL,
            approved_at TEXT NOT NULL,
            approval_expires_at TEXT NOT NULL DEFAULT '',
            PRIMARY KEY(channel, conversation_id)
        );
        CREATE TABLE IF NOT EXISTS pending_project_requests (
            channel TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            request_text TEXT NOT NULL,
            source_message_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(channel, conversation_id)
        );
        CREATE TABLE IF NOT EXISTS task_definitions (
            run_id TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
            objective TEXT NOT NULL,
            repository_path TEXT NOT NULL,
            repository_identity TEXT NOT NULL DEFAULT '',
            repository_head_sha TEXT NOT NULL DEFAULT '',
            repository_approved INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS task_plan_state (
            run_id TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
            revision INTEGER NOT NULL,
            plan_hash TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS task_execution_approvals (
            run_id TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
            granted INTEGER NOT NULL,
            plan_revision INTEGER NOT NULL,
            plan_hash TEXT NOT NULL,
            approved_at TEXT NOT NULL,
            approved_by TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS conversation_memories (
            scope TEXT NOT NULL,
            scope_key TEXT NOT NULL,
            role_id TEXT NOT NULL,
            content TEXT NOT NULL,
            revision INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(scope, scope_key, role_id)
        );
        CREATE TABLE IF NOT EXISTS conversation_jobs (
            job_id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            external_message_id TEXT NOT NULL,
            message_json TEXT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL,
            lease_owner TEXT NOT NULL,
            lease_until TEXT NOT NULL,
            last_error TEXT NOT NULL,
            replay_cached INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(channel, conversation_id, external_message_id)
        );
        CREATE TABLE IF NOT EXISTS inbound_receipts (
            channel TEXT NOT NULL,
            external_message_id TEXT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL,
            last_error TEXT NOT NULL,
            first_seen_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(channel, external_message_id)
        );
        CREATE TABLE IF NOT EXISTS gateway_cursors (
            channel TEXT PRIMARY KEY,
            cursor TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS outbound_messages (
            outbound_id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            text TEXT NOT NULL,
            reply_to TEXT NOT NULL,
            delivery_mode TEXT NOT NULL DEFAULT 'send',
            target_outbound_id INTEGER NOT NULL DEFAULT 0,
            coalesce_key TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL,
            external_ids_json TEXT NOT NULL,
            last_error TEXT NOT NULL,
            lease_owner TEXT NOT NULL DEFAULT '',
            lease_until TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS plan_revisions (
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            revision INTEGER NOT NULL,
            plan_hash TEXT NOT NULL,
            plan_json TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            approved_at TEXT NOT NULL,
            PRIMARY KEY(run_id, revision)
        );
        CREATE TABLE IF NOT EXISTS repository_approvals (
            user_id TEXT NOT NULL,
            repository_path TEXT NOT NULL,
            path_hash TEXT NOT NULL,
            approved_at TEXT NOT NULL,
            revoked_at TEXT NOT NULL,
            PRIMARY KEY(user_id, path_hash)
        );
        CREATE TABLE IF NOT EXISTS scoped_repository_approvals (
            channel TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            repository_path TEXT NOT NULL,
            path_hash TEXT NOT NULL,
            repository_identity TEXT NOT NULL DEFAULT '',
            approved_at TEXT NOT NULL,
            expires_at TEXT NOT NULL DEFAULT '',
            revoked_at TEXT NOT NULL,
            PRIMARY KEY(channel, conversation_id, user_id, path_hash)
        );
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS pipeline_jobs (
            run_id TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
            channel TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL,
            lease_owner TEXT NOT NULL,
            lease_until TEXT NOT NULL,
            last_error TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS execution_questions (
            question_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            stage_id TEXT NOT NULL,
            role_id TEXT NOT NULL,
            questions_json TEXT NOT NULL,
            status TEXT NOT NULL,
            answer TEXT NOT NULL,
            asked_at TEXT NOT NULL,
            answered_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS repository_execution_locks (
            path_hash TEXT PRIMARY KEY,
            repository_path TEXT NOT NULL,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            lease_owner TEXT NOT NULL,
            lease_until TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, event_id);
        CREATE INDEX IF NOT EXISTS idx_messages_run ON messages(run_id, message_id);
        CREATE INDEX IF NOT EXISTS idx_usage_run ON usage_ledger(run_id, usage_id);
        CREATE INDEX IF NOT EXISTS idx_retry_run ON retry_ledger(run_id, retry_id);
        CREATE INDEX IF NOT EXISTS idx_budget_warning_run ON budget_warning_ledger(run_id);
        CREATE INDEX IF NOT EXISTS idx_token_reservation_active
            ON token_reservations(run_id, status, stage_id, category);
        CREATE INDEX IF NOT EXISTS idx_conversation_run
            ON conversation_bindings(run_id);
        CREATE INDEX IF NOT EXISTS idx_conversation_session_task
            ON conversation_sessions(active_task_id);
        CREATE INDEX IF NOT EXISTS idx_conversation_jobs_delivery
            ON conversation_jobs(status, job_id);
        CREATE INDEX IF NOT EXISTS idx_conversation_jobs_thread
            ON conversation_jobs(channel, conversation_id, job_id);
        CREATE INDEX IF NOT EXISTS idx_inbound_status
            ON inbound_receipts(channel, status);
        CREATE INDEX IF NOT EXISTS idx_outbound_delivery
            ON outbound_messages(channel, status, outbound_id);
        CREATE INDEX IF NOT EXISTS idx_pipeline_delivery
            ON pipeline_jobs(status, created_at);
        CREATE INDEX IF NOT EXISTS idx_execution_questions_open
            ON execution_questions(run_id, status, question_id);
        """
        with self._lock, self._connection() as connection:
            connection.executescript(schema)
            existing = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(outbound_messages)")
            }
            for column, declaration in (
                ("lease_owner", "TEXT NOT NULL DEFAULT ''"),
                ("lease_until", "TEXT NOT NULL DEFAULT ''"),
                ("delivery_mode", "TEXT NOT NULL DEFAULT 'send'"),
                ("target_outbound_id", "INTEGER NOT NULL DEFAULT 0"),
                ("coalesce_key", "TEXT NOT NULL DEFAULT ''"),
            ):
                if column not in existing:
                    connection.execute(
                        f"ALTER TABLE outbound_messages ADD COLUMN {column} {declaration}"
                    )
            conversation_job_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(conversation_jobs)")
            }
            if "replay_cached" not in conversation_job_columns:
                connection.execute(
                    "ALTER TABLE conversation_jobs "
                    "ADD COLUMN replay_cached INTEGER NOT NULL DEFAULT 0"
                )
            binding_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(conversation_bindings)")
            }
            for column, declaration in (
                ("mode", "TEXT NOT NULL DEFAULT 'free_chat'"),
            ):
                if column not in binding_columns:
                    connection.execute(
                        f"ALTER TABLE conversation_bindings ADD COLUMN {column} {declaration}"
                    )
            for table_name, additions in (
                (
                    "budget_warning_ledger",
                    (("status", "TEXT NOT NULL DEFAULT 'PENDING'"),),
                ),
                (
                    "conversation_projects",
                    (
                        ("repository_identity", "TEXT NOT NULL DEFAULT ''"),
                        ("head_sha", "TEXT NOT NULL DEFAULT ''"),
                        ("approval_expires_at", "TEXT NOT NULL DEFAULT ''"),
                    ),
                ),
                (
                    "task_definitions",
                    (
                        ("repository_identity", "TEXT NOT NULL DEFAULT ''"),
                        ("repository_head_sha", "TEXT NOT NULL DEFAULT ''"),
                    ),
                ),
                (
                    "scoped_repository_approvals",
                    (
                        ("repository_identity", "TEXT NOT NULL DEFAULT ''"),
                        ("expires_at", "TEXT NOT NULL DEFAULT ''"),
                    ),
                ),
            ):
                columns = {
                    str(row["name"])
                    for row in connection.execute(f"PRAGMA table_info({table_name})")
                }
                for column, declaration in additions:
                    if column not in columns:
                        connection.execute(
                            f"ALTER TABLE {table_name} ADD COLUMN {column} {declaration}"
                        )
            self._apply_schema_migrations(connection)

    @staticmethod
    def _execution_payload(state: RunState) -> str:
        value = state.to_dict()
        for field_name in _RUN_EXTERNAL_FIELDS:
            value.pop(field_name, None)
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    def _apply_schema_migrations(self, connection: sqlite3.Connection) -> None:
        """레거시 투영은 버전별로 정확히 한 번만 실행한다."""
        session_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(conversation_sessions)")
        }
        if "session_run_id" not in session_columns:
            connection.execute(
                "ALTER TABLE conversation_sessions "
                "ADD COLUMN session_run_id TEXT NOT NULL DEFAULT ''"
            )

        migrated = connection.execute(
            "SELECT 1 FROM schema_migrations WHERE version = 1"
        ).fetchone()
        if migrated is None:
            self._migrate_split_state(connection)
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (1, ?)",
                (utc_now(),),
            )

        session_migration = connection.execute(
            "SELECT 1 FROM schema_migrations WHERE version = 2"
        ).fetchone()
        if session_migration is None:
            connection.execute(
                "UPDATE conversation_sessions SET session_run_id = active_task_id "
                "WHERE session_run_id = '' AND active_task_id IS NOT NULL"
            )
            missing = connection.execute(
                "SELECT channel, conversation_id FROM conversation_sessions "
                "WHERE session_run_id = '' LIMIT 1"
            ).fetchone()
            if missing is not None:
                raise StoreError(
                    "conversation session is missing its log run: "
                    f"{missing['channel']}/{missing['conversation_id']}"
                )
            foreign_keys = {
                str(row["from"])
                for row in connection.execute(
                    "PRAGMA foreign_key_list(conversation_sessions)"
                )
            }
            if "session_run_id" not in foreign_keys:
                connection.executescript(
                    """
                    CREATE TABLE conversation_sessions_v2 (
                        channel TEXT NOT NULL,
                        conversation_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        active_role TEXT NOT NULL,
                        mode TEXT NOT NULL,
                        session_run_id TEXT NOT NULL REFERENCES runs(run_id),
                        active_task_id TEXT REFERENCES runs(run_id),
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY(channel, conversation_id)
                    );
                    INSERT INTO conversation_sessions_v2(
                        channel, conversation_id, user_id, active_role, mode,
                        session_run_id, active_task_id, created_at, updated_at
                    )
                    SELECT channel, conversation_id, user_id, active_role, mode,
                        session_run_id, active_task_id, created_at, updated_at
                    FROM conversation_sessions;
                    DROP TABLE conversation_sessions;
                    ALTER TABLE conversation_sessions_v2 RENAME TO conversation_sessions;
                    CREATE INDEX idx_conversation_session_task
                        ON conversation_sessions(active_task_id);
                    """
                )
            violations = connection.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                raise StoreError("foreign key validation failed after session migration")
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (2, ?)",
                (utc_now(),),
            )
        identity_migration = connection.execute(
            "SELECT 1 FROM schema_migrations WHERE version = 3"
        ).fetchone()
        if identity_migration is None:
            now = utc_now()
            connection.execute(
                "UPDATE conversation_projects SET approved = 0, approved_at = '', "
                "approval_expires_at = '' WHERE repository_identity = ''"
            )
            connection.execute(
                "UPDATE task_definitions SET repository_approved = 0 "
                "WHERE repository_identity = ''"
            )
            connection.execute(
                "UPDATE scoped_repository_approvals SET revoked_at = ? "
                "WHERE repository_identity = '' AND revoked_at = ''",
                (now,),
            )
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (3, ?)",
                (now,),
            )
        analysis_migration = connection.execute(
            "SELECT 1 FROM schema_migrations WHERE version = 4"
        ).fetchone()
        if analysis_migration is None:
            connection.executescript(
                """
                CREATE TABLE repository_analysis_jobs (
                    analysis_id TEXT PRIMARY KEY,
                    channel TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    source_message_id TEXT NOT NULL,
                    role_id TEXT NOT NULL,
                    request_text TEXT NOT NULL,
                    repository_path TEXT NOT NULL,
                    repository_identity TEXT NOT NULL,
                    commit_sha TEXT NOT NULL,
                    branch TEXT NOT NULL,
                    status TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    stop_reason TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    completed_json TEXT NOT NULL,
                    remaining_json TEXT NOT NULL,
                    checkpoint INTEGER NOT NULL,
                    model_call_state TEXT NOT NULL,
                    query_rounds INTEGER NOT NULL,
                    model_calls INTEGER NOT NULL,
                    read_bytes INTEGER NOT NULL,
                    no_progress_count INTEGER NOT NULL,
                    progress_outbound_id INTEGER NOT NULL,
                    last_progress_at TEXT NOT NULL,
                    lease_owner TEXT NOT NULL,
                    lease_until TEXT NOT NULL,
                    attempts INTEGER NOT NULL,
                    last_error TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE repository_analysis_evidence (
                    evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id TEXT NOT NULL REFERENCES repository_analysis_jobs(analysis_id)
                        ON DELETE CASCADE,
                    commit_sha TEXT NOT NULL,
                    path TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    phase TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    redaction_status TEXT NOT NULL,
                    untrusted_repository_data INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(analysis_id, commit_sha, path, start_line, end_line, kind)
                );
                CREATE TABLE repository_analysis_batches (
                    analysis_id TEXT NOT NULL REFERENCES repository_analysis_jobs(analysis_id)
                        ON DELETE CASCADE,
                    batch_index INTEGER NOT NULL,
                    phase TEXT NOT NULL,
                    target_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    PRIMARY KEY(analysis_id, batch_index)
                );
                CREATE UNIQUE INDEX idx_repository_analysis_one_active_per_conversation
                    ON repository_analysis_jobs(channel, conversation_id)
                    WHERE status IN ('QUEUED', 'PROCESSING', 'STOP_REQUESTED', 'PAUSED');
                CREATE INDEX idx_repository_analysis_delivery
                    ON repository_analysis_jobs(status, created_at);
                CREATE INDEX idx_repository_analysis_evidence
                    ON repository_analysis_evidence(analysis_id, evidence_id);
                """
            )
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (4, ?)",
                (utc_now(),),
            )
        audit_migration = connection.execute(
            "SELECT 1 FROM schema_migrations WHERE version = 5"
        ).fetchone()
        if audit_migration is None:
            connection.executescript(
                """
                CREATE TABLE pending_repository_audits (
                    channel TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    request_text TEXT NOT NULL,
                    commit_sha TEXT NOT NULL,
                    repository_identity TEXT NOT NULL,
                    proposal_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(channel, conversation_id, user_id)
                );
                """
            )
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (5, ?)",
                (utc_now(),),
            )
        active_time_migration = connection.execute(
            "SELECT 1 FROM schema_migrations WHERE version = 6"
        ).fetchone()
        if active_time_migration is None:
            connection.execute(
                "ALTER TABLE repository_analysis_jobs ADD COLUMN active_seconds REAL NOT NULL DEFAULT 0"
            )
            connection.execute(
                "ALTER TABLE repository_analysis_jobs ADD COLUMN active_since TEXT NOT NULL DEFAULT ''"
            )
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (6, ?)",
                (utc_now(),),
            )
        delivery_migration = connection.execute(
            "SELECT 1 FROM schema_migrations WHERE version = 7"
        ).fetchone()
        if delivery_migration is None:
            connection.execute(
                "ALTER TABLE repository_analysis_jobs ADD COLUMN final_response TEXT NOT NULL DEFAULT ''"
            )
            connection.execute(
                "ALTER TABLE repository_analysis_jobs ADD COLUMN final_outbound_id INTEGER NOT NULL DEFAULT 0"
            )
            connection.execute(
                "ALTER TABLE repository_analysis_jobs ADD COLUMN delivery_status TEXT NOT NULL DEFAULT ''"
            )
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (7, ?)",
                (utc_now(),),
            )
        memory_migration = connection.execute(
            "SELECT 1 FROM schema_migrations WHERE version = 8"
        ).fetchone()
        if memory_migration is None:
            connection.execute(
                "CREATE TABLE memory_facts ("
                "fact_id INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT NOT NULL, "
                "scope_key TEXT NOT NULL, role_id TEXT NOT NULL, content TEXT NOT NULL, "
                "revision INTEGER NOT NULL, source_kind TEXT NOT NULL, "
                "source_ref TEXT NOT NULL, confirmed_at TEXT NOT NULL, "
                "expires_at TEXT NOT NULL, supersedes_id INTEGER, "
                "superseded_at TEXT NOT NULL DEFAULT '', deleted_at TEXT NOT NULL DEFAULT '')"
            )
            connection.execute(
                "CREATE INDEX idx_memory_facts_scope ON memory_facts"
                "(scope, scope_key, role_id, deleted_at, expires_at)"
            )
            for old in connection.execute(
                "SELECT scope, scope_key, role_id, content, revision, created_at, updated_at "
                "FROM conversation_memories WHERE role_id != 'compaction'"
            ).fetchall():
                confirmed = str(old["updated_at"])
                expiry = (datetime.fromisoformat(confirmed) + timedelta(days=30)).isoformat()
                connection.execute(
                    "INSERT INTO memory_facts(scope, scope_key, role_id, content, revision, "
                    "source_kind, source_ref, confirmed_at, expires_at) "
                    "VALUES (?, ?, ?, ?, ?, 'legacy', '', ?, ?)",
                    (old["scope"], old["scope_key"], old["role_id"], old["content"],
                     old["revision"], confirmed, expiry),
                )
            connection.execute("DELETE FROM conversation_memories WHERE role_id != 'compaction'")
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (8, ?)",
                (utc_now(),),
            )

    @staticmethod
    def _queue_repository_analysis_final(
        connection: sqlite3.Connection, job: sqlite3.Row, response: str, now: str,
    ) -> int:
        progress_id = int(job["progress_outbound_id"])
        if progress_id:
            connection.execute(
                "INSERT INTO outbound_messages(channel, conversation_id, text, reply_to, "
                "delivery_mode, target_outbound_id, coalesce_key, status, attempts, "
                "external_ids_json, last_error, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 'edit', ?, ?, 'PENDING', 0, '[]', '', ?, ?)",
                (
                    str(job["channel"]), str(job["conversation_id"]),
                    "[장기 저장소 분석 · 완료]\n최종 답변을 별도 메시지로 전송합니다. "
                    "전달에 문제가 있으면 '분석 결과'로 조회해 주세요.",
                    str(job["source_message_id"]), progress_id,
                    f"repository-analysis:{job['analysis_id']}", now, now,
                ),
            )
        cursor = connection.execute(
            "INSERT INTO outbound_messages(channel, conversation_id, text, reply_to, "
            "delivery_mode, target_outbound_id, coalesce_key, status, attempts, "
            "external_ids_json, last_error, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 'send', 0, '', 'PENDING', 0, '[]', '', ?, ?)",
            (
                str(job["channel"]), str(job["conversation_id"]),
                response.strip(), str(job["source_message_id"]), now, now,
            ),
        )
        return int(cursor.lastrowid)

    def save_pending_repository_audit(
        self, channel: str, conversation_id: str, user_id: str,
        request_text: str, commit_sha: str, repository_identity: str,
        proposal: dict[str, Any],
    ) -> None:
        with self._lock, self._connection() as connection:
            connection.execute(
                "INSERT INTO pending_repository_audits(channel, conversation_id, user_id, "
                "request_text, commit_sha, repository_identity, proposal_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(channel, conversation_id, user_id) DO UPDATE SET "
                "request_text = excluded.request_text, commit_sha = excluded.commit_sha, "
                "repository_identity = excluded.repository_identity, proposal_json = excluded.proposal_json, "
                "created_at = excluded.created_at",
                (channel, conversation_id, user_id, request_text, commit_sha,
                 repository_identity, json.dumps(proposal, ensure_ascii=False), utc_now()),
            )

    def load_pending_repository_audit(
        self, channel: str, conversation_id: str, user_id: str,
    ) -> dict[str, Any] | None:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM pending_repository_audits WHERE channel = ? "
                "AND conversation_id = ? AND user_id = ?",
                (channel, conversation_id, user_id),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["proposal"] = json.loads(result.pop("proposal_json"))
        return result

    def clear_pending_repository_audit(
        self, channel: str, conversation_id: str, user_id: str,
    ) -> None:
        with self._lock, self._connection() as connection:
            connection.execute(
                "DELETE FROM pending_repository_audits WHERE channel = ? "
                "AND conversation_id = ? AND user_id = ?",
                (channel, conversation_id, user_id),
            )

    def _migrate_split_state(self, connection: sqlite3.Connection) -> None:
        """기존 단일 JSON/바인딩을 분리 테이블로 안전하게 투영한다."""
        for row in connection.execute(
            "SELECT run_id, state_json, updated_at FROM runs"
        ).fetchall():
            raw = json.loads(str(row["state_json"]))
            if raw.get("repository_approved") and not (
                raw.get("repository_identity") and raw.get("repository_head_sha")
            ):
                # 5-B 이전의 경로 전용 승인은 실제 저장소 식별값이 없어
                # 안전하게 재사용할 수 없으므로 이관 시 fail-closed 한다.
                raw["repository_approved"] = False
            state = RunState.from_dict(raw)
            now = str(row["updated_at"])
            connection.execute(
                "INSERT OR IGNORE INTO task_definitions(run_id, objective, repository_path, "
                "repository_identity, repository_head_sha, repository_approved, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    state.run_id,
                    state.objective,
                    state.repository,
                    state.repository_identity,
                    state.repository_head_sha,
                    int(state.repository_approved),
                    now,
                ),
            )
            connection.execute(
                "INSERT OR IGNORE INTO task_plan_state(run_id, revision, plan_hash, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (state.run_id, state.plan_revision, state.plan_hash, now),
            )
            connection.execute(
                "INSERT OR IGNORE INTO task_execution_approvals(run_id, granted, plan_revision, "
                "plan_hash, approved_at, approved_by, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    state.run_id,
                    int(state.approval_granted),
                    state.approved_plan_revision,
                    state.approved_plan_hash,
                    state.approved_at,
                    state.approved_by,
                    now,
                ),
            )
            if any(field_name in raw for field_name in _RUN_EXTERNAL_FIELDS):
                connection.execute(
                    "UPDATE runs SET state_json = ? WHERE run_id = ?",
                    (self._execution_payload(state), state.run_id),
                )

        legacy_bindings = connection.execute(
            "SELECT channel, conversation_id, user_id, run_id, active_role, mode, "
            "created_at, updated_at FROM conversation_bindings"
        ).fetchall()
        for binding in legacy_bindings:
            connection.execute(
                "INSERT OR IGNORE INTO conversation_sessions(channel, conversation_id, user_id, "
                "active_role, mode, session_run_id, active_task_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(binding["channel"]),
                    str(binding["conversation_id"]),
                    str(binding["user_id"]),
                    str(binding["active_role"]),
                    str(binding["mode"]),
                    str(binding["run_id"]),
                    str(binding["run_id"]),
                    str(binding["created_at"]),
                    str(binding["updated_at"]),
                ),
            )
            task = connection.execute(
                "SELECT repository_path, repository_identity, repository_head_sha "
                "FROM task_definitions "
                "WHERE run_id = ?",
                (str(binding["run_id"]),),
            ).fetchone()
            if task is not None and str(task["repository_path"]):
                repository_path = str(task["repository_path"])
                connection.execute(
                    "INSERT OR IGNORE INTO conversation_projects(channel, conversation_id, "
                    "user_id, repository_path, repository_identity, head_sha, approved, "
                    "selected_at, approved_at, approval_expires_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, 0, ?, '', '')",
                    (
                        str(binding["channel"]),
                        str(binding["conversation_id"]),
                        str(binding["user_id"]),
                        repository_path,
                        str(task["repository_identity"]),
                        str(task["repository_head_sha"]),
                        str(binding["updated_at"]),
                    ),
                )

    @staticmethod
    def _write_state_parts(
        connection: sqlite3.Connection, state: RunState, now: str
    ) -> None:
        task = TaskDefinitionState(
            run_id=state.run_id,
            objective=state.objective,
            repository_path=state.repository,
            repository_identity=state.repository_identity,
            repository_head_sha=state.repository_head_sha,
            repository_approved=state.repository_approved,
        )
        plan = PlanPointerState(
            run_id=state.run_id,
            revision=state.plan_revision,
            plan_hash=state.plan_hash,
        )
        approval = ExecutionApprovalState(
            run_id=state.run_id,
            granted=state.approval_granted,
            plan_revision=state.approved_plan_revision,
            plan_hash=state.approved_plan_hash,
            approved_at=state.approved_at,
            approved_by=state.approved_by,
        )
        connection.execute(
            "INSERT INTO task_definitions(run_id, objective, repository_path, "
            "repository_identity, repository_head_sha, repository_approved, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id) DO UPDATE SET objective = excluded.objective, "
            "repository_path = excluded.repository_path, "
            "repository_identity = excluded.repository_identity, "
            "repository_head_sha = excluded.repository_head_sha, "
            "repository_approved = excluded.repository_approved, updated_at = excluded.updated_at",
            (
                task.run_id,
                task.objective,
                task.repository_path,
                task.repository_identity,
                task.repository_head_sha,
                int(task.repository_approved),
                now,
            ),
        )
        connection.execute(
            "INSERT INTO task_plan_state(run_id, revision, plan_hash, updated_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(run_id) DO UPDATE SET "
            "revision = excluded.revision, plan_hash = excluded.plan_hash, "
            "updated_at = excluded.updated_at",
            (plan.run_id, plan.revision, plan.plan_hash, now),
        )
        connection.execute(
            "INSERT INTO task_execution_approvals(run_id, granted, plan_revision, plan_hash, "
            "approved_at, approved_by, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id) DO UPDATE SET granted = excluded.granted, "
            "plan_revision = excluded.plan_revision, plan_hash = excluded.plan_hash, "
            "approved_at = excluded.approved_at, approved_by = excluded.approved_by, "
            "updated_at = excluded.updated_at",
            (
                approval.run_id,
                int(approval.granted),
                approval.plan_revision,
                approval.plan_hash,
                approval.approved_at,
                approval.approved_by,
                now,
            ),
        )

    @staticmethod
    def _hydrate_run_parts(
        connection: sqlite3.Connection, value: dict[str, Any], run_id: str
    ) -> dict[str, Any]:
        task = connection.execute(
            "SELECT objective, repository_path, repository_identity, "
            "repository_head_sha, repository_approved FROM task_definitions "
            "WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        plan = connection.execute(
            "SELECT revision, plan_hash FROM task_plan_state WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        approval = connection.execute(
            "SELECT granted, plan_revision, plan_hash, approved_at, approved_by "
            "FROM task_execution_approvals WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if task is not None:
            value.update(
                repository=str(task["repository_path"]),
                repository_identity=str(task["repository_identity"]),
                repository_head_sha=str(task["repository_head_sha"]),
                objective=str(task["objective"]),
                repository_approved=bool(task["repository_approved"]),
            )
        if plan is not None:
            value.update(
                plan_revision=int(plan["revision"]),
                plan_hash=str(plan["plan_hash"]),
            )
        if approval is not None:
            value.update(
                approval_granted=bool(approval["granted"]),
                approved_plan_revision=int(approval["plan_revision"]),
                approved_plan_hash=str(approval["plan_hash"]),
                approved_at=str(approval["approved_at"]),
                approved_by=str(approval["approved_by"]),
            )
        return value

    def create_run(self, state: RunState) -> RunState:
        if state.version != 0:
            raise StoreError("new runs must start at version zero")
        payload = self._execution_payload(state)
        checkpoint = json.dumps(state.to_dict(), ensure_ascii=False, sort_keys=True)
        with self._lock, self._connection() as connection:
            try:
                connection.execute(
                    "INSERT INTO runs(run_id, state_json, version, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (state.run_id, payload, state.version, state.created_at, state.updated_at),
                )
                connection.execute(
                    "INSERT INTO checkpoints(run_id, phase, state_json, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (state.run_id, state.phase.value, checkpoint, state.updated_at),
                )
                self._write_state_parts(connection, state, state.updated_at)
            except sqlite3.IntegrityError as exc:
                raise StoreError(f"run already exists: {state.run_id}") from exc
        return state

    def load_run(self, run_id: str) -> RunState:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT state_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise StoreError(f"run not found: {run_id}")
            value = self._hydrate_run_parts(
                connection, json.loads(str(row["state_json"])), run_id
            )
        return RunState.from_dict(value)

    def save_run(self, state: RunState) -> RunState:
        now = utc_now()
        next_state = replace(state, version=state.version + 1, updated_at=now)
        payload = self._execution_payload(next_state)
        checkpoint = json.dumps(next_state.to_dict(), ensure_ascii=False, sort_keys=True)
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "UPDATE runs SET state_json = ?, version = ?, updated_at = ? "
                "WHERE run_id = ? AND version = ?",
                (payload, next_state.version, now, state.run_id, state.version),
            )
            if cursor.rowcount != 1:
                raise ConcurrentUpdateError(
                    f"run changed concurrently or does not exist: {state.run_id}"
                )
            self._write_state_parts(connection, next_state, now)
            connection.execute(
                "INSERT INTO checkpoints(run_id, phase, state_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (state.run_id, next_state.phase.value, checkpoint, now),
            )
        return next_state

    def latest_checkpoint(self, run_id: str) -> RunState:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT state_json FROM checkpoints WHERE run_id = ? "
                "ORDER BY checkpoint_id DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        if row is None:
            raise StoreError(f"checkpoint not found: {run_id}")
        return RunState.from_dict(json.loads(str(row["state_json"])))

    def append_event(
        self,
        run_id: str,
        timestamp: str,
        event_type: str,
        message: str,
        *,
        stage_id: str = "",
        role_id: str = "",
        status: str = "",
        data: dict[str, Any] | None = None,
    ) -> None:
        with self._lock, self._connection() as connection:
            connection.execute(
                "INSERT INTO events(run_id, timestamp, stage_id, role_id, event_type, "
                "status, message, data_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    timestamp,
                    stage_id,
                    role_id,
                    event_type,
                    status,
                    message,
                    json.dumps(data or {}, ensure_ascii=False, sort_keys=True),
                ),
            )

    def list_events(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT timestamp, stage_id, role_id, event_type, status, message, data_json "
                "FROM events WHERE run_id = ? ORDER BY event_id",
                (run_id,),
            ).fetchall()
        return [
            {
                "timestamp": str(row["timestamp"]),
                "stage_id": str(row["stage_id"]),
                "role_id": str(row["role_id"]),
                "event_type": str(row["event_type"]),
                "status": str(row["status"]),
                "message": str(row["message"]),
                "data": json.loads(str(row["data_json"])),
            }
            for row in rows
        ]

    def append_message(
        self,
        run_id: str,
        sender: str,
        kind: str,
        content: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        with self._lock, self._connection() as connection:
            connection.execute(
                "INSERT INTO messages(run_id, timestamp, sender, kind, content, data_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    utc_now(),
                    sender,
                    kind,
                    content,
                    json.dumps(data or {}, ensure_ascii=False, sort_keys=True),
                ),
            )

    def list_messages(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT timestamp, sender, kind, content, data_json FROM messages "
                "WHERE run_id = ? ORDER BY message_id",
                (run_id,),
            ).fetchall()
        return [
            {
                "timestamp": str(row["timestamp"]),
                "sender": str(row["sender"]),
                "kind": str(row["kind"]),
                "content": str(row["content"]),
                "data": json.loads(str(row["data_json"])),
            }
            for row in rows
        ]

    def conversation_responses(
        self, channel: str, conversation_id: str, external_message_id: str
    ) -> list[str]:
        """중단된 워커가 이미 확정한 응답을 발신함으로 복구한다."""
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT content, data_json FROM messages "
                "WHERE kind = 'assistant_message' ORDER BY message_id"
            ).fetchall()
        responses: list[str] = []
        for row in rows:
            data = json.loads(str(row["data_json"]))
            if (
                str(data.get("channel", "")) == channel
                and str(data.get("conversation_id", "")) == conversation_id
                and str(data.get("external_message_id", ""))
                == external_message_id
            ):
                responses.append(str(row["content"]))
        return responses

    def add_usage(
        self,
        run_id: str,
        stage_id: str,
        role_id: str,
        category: str,
        usage: TokenUsage,
    ) -> None:
        with self._lock, self._connection() as connection:
            connection.execute(
                "INSERT INTO usage_ledger(run_id, stage_id, role_id, category, "
                "input_tokens, output_tokens, total_tokens, estimated, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    stage_id,
                    role_id,
                    category,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.total_tokens,
                    int(usage.estimated),
                    utc_now(),
                ),
            )

    def usage_total(self, run_id: str, stage_id: str | None = None) -> int:
        query = "SELECT COALESCE(SUM(total_tokens), 0) AS total FROM usage_ledger WHERE run_id = ?"
        parameters: list[Any] = [run_id]
        if stage_id is not None:
            query += " AND stage_id = ?"
            parameters.append(stage_id)
        with self._lock, self._connection() as connection:
            row = connection.execute(query, parameters).fetchone()
        return int(row["total"] if row else 0)

    def usage_total_by_category(self, run_id: str, category: str) -> int:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(SUM(total_tokens), 0) AS total FROM usage_ledger "
                "WHERE run_id = ? AND category = ?",
                (run_id, category),
            ).fetchone()
        return int(row["total"] if row else 0)

    def reserved_token_total(
        self,
        run_id: str,
        *,
        stage_id: str | None = None,
        category: str | None = None,
    ) -> int:
        query = (
            "SELECT COALESCE(SUM(reserved_tokens), 0) AS total FROM token_reservations "
            "WHERE run_id = ? AND status = 'RESERVED'"
        )
        parameters: list[Any] = [run_id]
        if stage_id is not None:
            query += " AND stage_id = ?"
            parameters.append(stage_id)
        if category is not None:
            query += " AND category = ?"
            parameters.append(category)
        with self._lock, self._connection() as connection:
            row = connection.execute(query, parameters).fetchone()
        return int(row["total"] if row else 0)

    def create_token_reservation(
        self,
        reservation_id: str,
        run_id: str,
        stage_id: str,
        role_id: str,
        category: str,
        reserved_tokens: int,
    ) -> None:
        if not reservation_id or reserved_tokens < 1:
            raise ValueError("token reservation requires an id and positive tokens")
        with self._lock, self._connection() as connection:
            connection.execute(
                "INSERT INTO token_reservations("
                "reservation_id, run_id, stage_id, role_id, category, reserved_tokens, "
                "status, created_at, settled_at"
                ") VALUES (?, ?, ?, ?, ?, ?, 'RESERVED', ?, '')",
                (
                    reservation_id,
                    run_id,
                    stage_id,
                    role_id,
                    category,
                    reserved_tokens,
                    utc_now(),
                ),
            )

    def settle_token_reservation(self, reservation_id: str) -> bool:
        if not reservation_id:
            return False
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "UPDATE token_reservations SET status = 'SETTLED', settled_at = ? "
                "WHERE reservation_id = ? AND status = 'RESERVED'",
                (utc_now(), reservation_id),
            )
        return cursor.rowcount == 1

    def release_token_reservation(self, reservation_id: str) -> bool:
        if not reservation_id:
            return False
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "UPDATE token_reservations SET status = 'RELEASED', settled_at = ? "
                "WHERE reservation_id = ? AND status = 'RESERVED'",
                (utc_now(), reservation_id),
            )
        return cursor.rowcount == 1

    def usage_breakdown(self, run_id: str) -> dict[str, int]:
        """Return durable actual/estimated totals without exposing raw prompts."""
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(SUM(total_tokens), 0) AS total_tokens, "
                "COALESCE(SUM(CASE WHEN estimated = 0 THEN total_tokens ELSE 0 END), 0) AS actual_tokens, "
                "COALESCE(SUM(CASE WHEN estimated = 1 THEN total_tokens ELSE 0 END), 0) AS estimated_tokens "
                "FROM usage_ledger WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return {
            "total_tokens": int(row["total_tokens"] if row else 0),
            "actual_tokens": int(row["actual_tokens"] if row else 0),
            "estimated_tokens": int(row["estimated_tokens"] if row else 0),
        }

    def conversation_targets_for_run(self, run_id: str) -> tuple[dict[str, str], ...]:
        """Find active owner conversations without making the store Telegram-specific."""
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT channel, conversation_id FROM conversation_sessions "
                "WHERE session_run_id = ? OR active_task_id = ? "
                "UNION "
                "SELECT channel, conversation_id FROM conversation_bindings WHERE run_id = ?",
                (run_id, run_id, run_id),
            ).fetchall()
        return tuple(
            {
                "channel": str(row["channel"]),
                "conversation_id": str(row["conversation_id"]),
            }
            for row in rows
        )

    def claim_budget_warning(
        self, run_id: str, scope: str, stage_id: str, threshold_percent: int
    ) -> bool:
        if scope not in {"conversation", "stage", "task"}:
            raise ValueError("unsupported budget warning scope")
        if not 1 <= threshold_percent <= 99:
            raise ValueError("budget warning threshold must be between 1 and 99")
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO budget_warning_ledger("
                "run_id, scope, stage_id, threshold_percent, status, created_at"
                ") VALUES (?, ?, ?, ?, 'PENDING', ?)",
                (run_id, scope, stage_id, threshold_percent, utc_now()),
            )
        return cursor.rowcount == 1

    def pending_budget_warning_claims(self, run_id: str) -> tuple[dict[str, Any], ...]:
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT scope, stage_id, threshold_percent FROM budget_warning_ledger "
                "WHERE run_id = ? AND status = 'PENDING' "
                "ORDER BY created_at, scope, stage_id",
                (run_id,),
            ).fetchall()
        return tuple(
            {
                "scope": str(row["scope"]),
                "stage_id": str(row["stage_id"]),
                "threshold_percent": int(row["threshold_percent"]),
            }
            for row in rows
        )

    def pending_budget_warning_run_ids(self) -> tuple[str, ...]:
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT DISTINCT run_id FROM budget_warning_ledger "
                "WHERE status = 'PENDING' ORDER BY run_id"
            ).fetchall()
        return tuple(str(row["run_id"]) for row in rows)

    def queue_budget_warning_delivery(
        self,
        run_id: str,
        warnings: tuple[dict[str, Any], ...],
        targets: tuple[dict[str, str], ...],
    ) -> int:
        """Atomically persist warning outbox rows before marking them queued."""
        if not warnings or not targets:
            return 0
        now = utc_now()
        queued = 0
        with self.transaction():
            with self._connection() as connection:
                for warning in warnings:
                    cursor = connection.execute(
                        "UPDATE budget_warning_ledger SET status = 'QUEUED' "
                        "WHERE run_id = ? AND scope = ? AND stage_id = ? "
                        "AND threshold_percent = ? AND status = 'PENDING'",
                        (
                            run_id,
                            str(warning["scope"]),
                            str(warning.get("stage_id", "")),
                            int(warning["threshold_percent"]),
                        ),
                    )
                    if cursor.rowcount != 1:
                        continue
                    for target in targets:
                        connection.execute(
                            "INSERT INTO outbound_messages(channel, conversation_id, text, reply_to, "
                            "status, attempts, external_ids_json, last_error, created_at, updated_at) "
                            "VALUES (?, ?, ?, '', 'PENDING', 0, '[]', '', ?, ?)",
                            (
                                str(target["channel"]),
                                str(target["conversation_id"]),
                                str(warning["text"]),
                                now,
                                now,
                            ),
                        )
                        queued += 1
        return queued

    def retry_count(self, run_id: str, stage_id: str, category: str) -> int:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM retry_ledger "
                "WHERE run_id = ? AND stage_id = ? AND category = ?",
                (run_id, stage_id, category),
            ).fetchone()
        return int(row["total"] if row else 0)

    def add_retry(
        self, run_id: str, stage_id: str, category: str, reason: str
    ) -> int:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM retry_ledger "
                "WHERE run_id = ? AND stage_id = ? AND category = ?",
                (run_id, stage_id, category),
            ).fetchone()
            attempt = int(row["total"] if row else 0) + 1
            connection.execute(
                "INSERT INTO retry_ledger(run_id, stage_id, category, attempt_number, "
                "reason, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, stage_id, category, attempt, reason, utc_now()),
            )
        return attempt

    def save_plan_revision(
        self,
        run_id: str,
        revision: int,
        plan_hash: str,
        plan: dict[str, Any],
    ) -> None:
        if revision < 1:
            raise ValueError("plan revision must be positive")
        now = utc_now()
        payload = json.dumps(plan, ensure_ascii=False, sort_keys=True)
        with self._lock, self._connection() as connection:
            connection.execute(
                "UPDATE plan_revisions SET status = 'SUPERSEDED' "
                "WHERE run_id = ? AND status IN ('CURRENT', 'APPROVED')",
                (run_id,),
            )
            try:
                connection.execute(
                    "INSERT INTO plan_revisions(run_id, revision, plan_hash, plan_json, "
                    "status, created_at, approved_at) VALUES (?, ?, ?, ?, 'CURRENT', ?, '')",
                    (run_id, revision, plan_hash, payload, now),
                )
            except sqlite3.IntegrityError as exc:
                raise StoreError(
                    f"plan revision already exists: {run_id}/{revision}"
                ) from exc

    def approve_plan_revision(
        self, run_id: str, revision: int, plan_hash: str
    ) -> None:
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "UPDATE plan_revisions SET status = 'APPROVED', approved_at = ? "
                "WHERE run_id = ? AND revision = ? AND plan_hash = ? "
                "AND status = 'CURRENT'",
                (utc_now(), run_id, revision, plan_hash),
            )
            if cursor.rowcount != 1:
                raise StoreError("the current plan revision does not match approval")

    def load_plan_revision(
        self, run_id: str, revision: int
    ) -> dict[str, Any] | None:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT plan_hash, plan_json, status, created_at, approved_at "
                "FROM plan_revisions WHERE run_id = ? AND revision = ?",
                (run_id, revision),
            ).fetchone()
        if row is None:
            return None
        return {
            "run_id": run_id,
            "revision": revision,
            "plan_hash": str(row["plan_hash"]),
            "plan": json.loads(str(row["plan_json"])),
            "status": str(row["status"]),
            "created_at": str(row["created_at"]),
            "approved_at": str(row["approved_at"]),
        }

    @staticmethod
    def _repository_hash(repository_path: str) -> str:
        normalized = str(Path(repository_path).resolve()).casefold()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def approve_repository(
        self,
        channel: str,
        conversation_id: str,
        user_id: str,
        repository_path: str,
        repository_identity: str,
        expires_at: str,
    ) -> None:
        if not all(
            value.strip()
            for value in (
                channel,
                conversation_id,
                user_id,
                repository_path,
                repository_identity,
                expires_at,
            )
        ):
            raise ValueError("approval scope, repository identity, and expiry are required")
        if not re.fullmatch(r"[0-9a-f]{64}", repository_identity):
            raise ValueError("repository identity must be a SHA-256 hex digest")
        if not self._timestamp_is_future(expires_at):
            raise ValueError("repository approval expiry must be in the future")
        now = utc_now()
        path_hash = self._repository_hash(repository_path)
        with self._lock, self._connection() as connection:
            connection.execute(
                "INSERT INTO scoped_repository_approvals(channel, conversation_id, user_id, "
                "repository_path, path_hash, repository_identity, approved_at, expires_at, revoked_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, '') "
                "ON CONFLICT(channel, conversation_id, user_id, path_hash) DO UPDATE SET "
                "repository_path = excluded.repository_path, "
                "repository_identity = excluded.repository_identity, "
                "approved_at = excluded.approved_at, expires_at = excluded.expires_at, revoked_at = ''",
                (
                    channel.strip(),
                    conversation_id.strip(),
                    user_id.strip(),
                    str(Path(repository_path).resolve()),
                    path_hash,
                    repository_identity,
                    now,
                    expires_at,
                ),
            )

    def repository_is_approved(
        self,
        channel: str,
        conversation_id: str,
        user_id: str,
        repository_path: str,
        repository_identity: str,
    ) -> bool:
        if not repository_identity.strip():
            return False
        path_hash = self._repository_hash(repository_path)
        now = utc_now()
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM scoped_repository_approvals "
                "WHERE channel = ? AND conversation_id = ? AND user_id = ? "
                "AND path_hash = ? AND repository_identity = ? AND revoked_at = '' "
                "AND expires_at > ?",
                (
                    channel.strip(),
                    conversation_id.strip(),
                    user_id.strip(),
                    path_hash,
                    repository_identity.strip(),
                    now,
                ),
            ).fetchone()
        return row is not None

    def repository_approval_expiry(
        self,
        channel: str,
        conversation_id: str,
        user_id: str,
        repository_path: str,
        repository_identity: str,
    ) -> str:
        if not repository_identity.strip():
            return ""
        path_hash = self._repository_hash(repository_path)
        now = utc_now()
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT expires_at FROM scoped_repository_approvals "
                "WHERE channel = ? AND conversation_id = ? AND user_id = ? "
                "AND path_hash = ? AND repository_identity = ? AND revoked_at = '' "
                "AND expires_at > ?",
                (
                    channel.strip(),
                    conversation_id.strip(),
                    user_id.strip(),
                    path_hash,
                    repository_identity.strip(),
                    now,
                ),
            ).fetchone()
        return str(row["expires_at"]) if row is not None else ""

    @staticmethod
    def _timestamp_is_future(value: str) -> bool:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return False
        if parsed.tzinfo is None:
            return False
        return parsed > datetime.now(timezone.utc)

    def load_task_definition(self, run_id: str) -> TaskDefinitionState:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT objective, repository_path, repository_identity, "
                "repository_head_sha, repository_approved "
                "FROM task_definitions WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise StoreError(f"task definition not found: {run_id}")
        return TaskDefinitionState(
            run_id=run_id,
            objective=str(row["objective"]),
            repository_path=str(row["repository_path"]),
            repository_identity=str(row["repository_identity"]),
            repository_head_sha=str(row["repository_head_sha"]),
            repository_approved=bool(row["repository_approved"]),
        )

    def load_plan_pointer(self, run_id: str) -> PlanPointerState:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT revision, plan_hash FROM task_plan_state WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise StoreError(f"plan state not found: {run_id}")
        return PlanPointerState(
            run_id=run_id,
            revision=int(row["revision"]),
            plan_hash=str(row["plan_hash"]),
        )

    def load_execution_approval(self, run_id: str) -> ExecutionApprovalState:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT granted, plan_revision, plan_hash, approved_at, approved_by "
                "FROM task_execution_approvals WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise StoreError(f"execution approval not found: {run_id}")
        return ExecutionApprovalState(
            run_id=run_id,
            granted=bool(row["granted"]),
            plan_revision=int(row["plan_revision"]),
            plan_hash=str(row["plan_hash"]),
            approved_at=str(row["approved_at"]),
            approved_by=str(row["approved_by"]),
        )

    def load_conversation_session(
        self, channel: str, conversation_id: str
    ) -> ConversationSessionState | None:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT channel, conversation_id, user_id, active_role, mode, "
                "session_run_id, active_task_id, created_at, updated_at "
                "FROM conversation_sessions "
                "WHERE channel = ? AND conversation_id = ?",
                (channel, conversation_id),
            ).fetchone()
        if row is None:
            return None
        return ConversationSessionState(
            channel=str(row["channel"]),
            conversation_id=str(row["conversation_id"]),
            user_id=str(row["user_id"]),
            active_role=RoleId(str(row["active_role"])),
            mode=str(row["mode"]),
            session_run_id=str(row["session_run_id"]),
            active_task_id=str(row["active_task_id"] or ""),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def load_project_selection(
        self, channel: str, conversation_id: str
    ) -> ProjectSelectionState | None:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT channel, conversation_id, user_id, repository_path, approved, "
                "repository_identity, head_sha, selected_at, approved_at, "
                "approval_expires_at FROM conversation_projects "
                "WHERE channel = ? AND conversation_id = ?",
                (channel, conversation_id),
            ).fetchone()
        if row is None:
            return None
        approved = bool(row["approved"]) and self._timestamp_is_future(
            str(row["approval_expires_at"])
        )
        return ProjectSelectionState(
            channel=str(row["channel"]),
            conversation_id=str(row["conversation_id"]),
            user_id=str(row["user_id"]),
            repository_path=str(row["repository_path"]),
            repository_identity=str(row["repository_identity"]),
            head_sha=str(row["head_sha"]),
            approved=approved,
            selected_at=str(row["selected_at"]),
            approved_at=str(row["approved_at"]),
            approval_expires_at=str(row["approval_expires_at"]),
        )

    def set_current_project(
        self,
        channel: str,
        conversation_id: str,
        user_id: str,
        repository_path: str,
        *,
        repository_identity: str = "",
        head_sha: str = "",
        approved: bool = False,
        approval_expires_at: str = "",
    ) -> ProjectSelectionState:
        if not repository_path.strip():
            raise ValueError("repository path is required")
        if repository_identity and not re.fullmatch(
            r"[0-9a-f]{64}", repository_identity
        ):
            raise ValueError("repository identity must be a SHA-256 hex digest")
        if head_sha and not re.fullmatch(r"[0-9a-f]{40,64}", head_sha):
            raise ValueError("repository head must be a Git object id")
        if bool(repository_identity) != bool(head_sha):
            raise ValueError("repository identity and HEAD must be set together")
        now = utc_now()
        resolved_path = str(Path(repository_path).resolve())
        if approved:
            if not re.fullmatch(r"[0-9a-f]{64}", repository_identity):
                raise ValueError("approved project requires a repository identity")
            if not re.fullmatch(r"[0-9a-f]{40,64}", head_sha):
                raise ValueError("approved project requires a Git HEAD")
            if not self._timestamp_is_future(approval_expires_at):
                raise ValueError("approved project requires a future expiry")
        with self._lock, self._connection() as connection:
            session = connection.execute(
                "SELECT user_id FROM conversation_sessions "
                "WHERE channel = ? AND conversation_id = ?",
                (channel, conversation_id),
            ).fetchone()
            if session is None:
                raise StoreError(f"conversation is not bound: {channel}/{conversation_id}")
            if str(session["user_id"]) != user_id.strip():
                raise StoreError("project selection owner does not match the conversation")
            connection.execute(
                "INSERT INTO conversation_projects(channel, conversation_id, user_id, "
                "repository_path, repository_identity, head_sha, approved, selected_at, "
                "approved_at, approval_expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(channel, conversation_id) DO UPDATE SET "
                "user_id = excluded.user_id, repository_path = excluded.repository_path, "
                "repository_identity = excluded.repository_identity, head_sha = excluded.head_sha, "
                "approved = excluded.approved, selected_at = excluded.selected_at, "
                "approved_at = excluded.approved_at, "
                "approval_expires_at = excluded.approval_expires_at",
                (
                    channel,
                    conversation_id,
                    user_id.strip(),
                    resolved_path,
                    repository_identity,
                    head_sha,
                    int(approved),
                    now,
                    now if approved else "",
                    approval_expires_at if approved else "",
                ),
            )
            if approved:
                path_hash = self._repository_hash(resolved_path)
                connection.execute(
                    "INSERT INTO scoped_repository_approvals(channel, conversation_id, "
                    "user_id, repository_path, path_hash, repository_identity, approved_at, "
                    "expires_at, revoked_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, '') "
                    "ON CONFLICT(channel, conversation_id, user_id, path_hash) DO UPDATE SET "
                    "repository_path = excluded.repository_path, "
                    "repository_identity = excluded.repository_identity, "
                    "approved_at = excluded.approved_at, expires_at = excluded.expires_at, "
                    "revoked_at = ''",
                    (
                        channel,
                        conversation_id,
                        user_id.strip(),
                        resolved_path,
                        path_hash,
                        repository_identity,
                        now,
                        approval_expires_at,
                    ),
                )
        selected = self.load_project_selection(channel, conversation_id)
        if selected is None:
            raise StoreError("project selection was not saved")
        return selected

    def set_current_project_approval(
        self,
        channel: str,
        conversation_id: str,
        user_id: str,
        approved: bool,
        *,
        approval_expires_at: str = "",
    ) -> ProjectSelectionState:
        now = utc_now()
        with self._lock, self._connection() as connection:
            project = connection.execute(
                "SELECT repository_path, repository_identity, head_sha "
                "FROM conversation_projects "
                "WHERE channel = ? AND conversation_id = ? AND user_id = ?",
                (channel, conversation_id, user_id.strip()),
            ).fetchone()
            if project is None:
                raise StoreError("current project is missing or owned by another user")
            repository_identity = str(project["repository_identity"])
            if approved and (
                not repository_identity
                or not str(project["head_sha"])
                or not self._timestamp_is_future(approval_expires_at)
            ):
                raise ValueError("project approval requires identity, HEAD, and future expiry")
            cursor = connection.execute(
                "UPDATE conversation_projects SET approved = ?, approved_at = ?, "
                "approval_expires_at = ? "
                "WHERE channel = ? AND conversation_id = ? AND user_id = ?",
                (
                    int(approved),
                    now if approved else "",
                    approval_expires_at if approved else "",
                    channel,
                    conversation_id,
                    user_id.strip(),
                ),
            )
            if cursor.rowcount != 1:
                raise StoreError("current project is missing or owned by another user")
            repository_path = str(project["repository_path"])
            path_hash = self._repository_hash(repository_path)
            if approved:
                connection.execute(
                    "INSERT INTO scoped_repository_approvals(channel, conversation_id, "
                    "user_id, repository_path, path_hash, repository_identity, approved_at, "
                    "expires_at, revoked_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, '') "
                    "ON CONFLICT(channel, conversation_id, user_id, path_hash) DO UPDATE SET "
                    "repository_path = excluded.repository_path, "
                    "repository_identity = excluded.repository_identity, "
                    "approved_at = excluded.approved_at, expires_at = excluded.expires_at, "
                    "revoked_at = ''",
                    (
                        channel,
                        conversation_id,
                        user_id.strip(),
                        repository_path,
                        path_hash,
                        repository_identity,
                        now,
                        approval_expires_at,
                    ),
                )
            else:
                connection.execute(
                    "UPDATE scoped_repository_approvals SET revoked_at = ? "
                    "WHERE channel = ? AND conversation_id = ? AND user_id = ? "
                    "AND path_hash = ?",
                    (
                        now,
                        channel,
                        conversation_id,
                        user_id.strip(),
                        path_hash,
                    ),
                )
        selected = self.load_project_selection(channel, conversation_id)
        if selected is None:
            raise StoreError("project selection was not saved")
        return selected

    def clear_current_project(
        self, channel: str, conversation_id: str, user_id: str
    ) -> None:
        now = utc_now()
        with self._lock, self._connection() as connection:
            project = connection.execute(
                "SELECT repository_path FROM conversation_projects WHERE channel = ? "
                "AND conversation_id = ? AND user_id = ?",
                (channel, conversation_id, user_id.strip()),
            ).fetchone()
            if project is not None:
                connection.execute(
                    "UPDATE scoped_repository_approvals SET revoked_at = ? "
                    "WHERE channel = ? AND conversation_id = ? AND user_id = ? "
                    "AND path_hash = ? AND revoked_at = ''",
                    (
                        now,
                        channel,
                        conversation_id,
                        user_id.strip(),
                        self._repository_hash(str(project["repository_path"])),
                    ),
                )
            connection.execute(
                "DELETE FROM conversation_projects WHERE channel = ? "
                "AND conversation_id = ? AND user_id = ?",
                (channel, conversation_id, user_id.strip()),
            )

    def bind_conversation(
        self,
        channel: str,
        conversation_id: str,
        user_id: str,
        run_id: str,
        active_role: str,
        mode: str = "free_chat",
    ) -> None:
        now = utc_now()
        with self._lock, self._connection() as connection:
            connection.execute(
                "INSERT INTO conversation_sessions(channel, conversation_id, user_id, "
                "active_role, mode, session_run_id, active_task_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(channel, conversation_id) DO UPDATE SET "
                "user_id = excluded.user_id, session_run_id = excluded.session_run_id, "
                "active_task_id = excluded.active_task_id, "
                "active_role = excluded.active_role, mode = excluded.mode, "
                "updated_at = excluded.updated_at",
                (
                    channel,
                    conversation_id,
                    user_id,
                    active_role,
                    mode,
                    run_id,
                    run_id,
                    now,
                    now,
                ),
            )
            task = connection.execute(
                "SELECT repository_path, repository_identity, repository_head_sha "
                "FROM task_definitions "
                "WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if task is not None and str(task["repository_path"]):
                repository_path = str(task["repository_path"])
                connection.execute(
                    "INSERT OR IGNORE INTO conversation_projects(channel, conversation_id, "
                    "user_id, repository_path, repository_identity, head_sha, approved, "
                    "selected_at, approved_at, approval_expires_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, 0, ?, '', '')",
                    (
                        channel,
                        conversation_id,
                        user_id,
                        repository_path,
                        str(task["repository_identity"]),
                        str(task["repository_head_sha"]),
                        now,
                    ),
                )

    def create_conversation_session(
        self,
        channel: str,
        conversation_id: str,
        user_id: str,
        session_run_id: str,
        active_role: str,
        mode: str = "free_chat",
    ) -> None:
        """작업을 만들지 않고 대화 로그용 실행 ID로 세션을 연다."""
        now = utc_now()
        session = ConversationSessionState(
            channel=channel,
            conversation_id=conversation_id,
            user_id=user_id,
            active_role=RoleId(active_role),
            mode=mode,
            session_run_id=session_run_id,
            active_task_id="",
            created_at=now,
            updated_at=now,
        )
        with self._lock, self._connection() as connection:
            connection.execute(
                "INSERT INTO conversation_sessions(channel, conversation_id, user_id, "
                "active_role, mode, session_run_id, active_task_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?) "
                "ON CONFLICT(channel, conversation_id) DO UPDATE SET "
                "user_id = excluded.user_id, active_role = excluded.active_role, "
                "mode = excluded.mode, session_run_id = excluded.session_run_id, "
                "active_task_id = NULL, updated_at = excluded.updated_at",
                (
                    session.channel,
                    session.conversation_id,
                    session.user_id,
                    session.active_role.value,
                    session.mode,
                    session.session_run_id,
                    session.created_at,
                    session.updated_at,
                ),
            )

    def load_conversation(
        self, channel: str, conversation_id: str
    ) -> dict[str, str] | None:
        session = self.load_conversation_session(channel, conversation_id)
        if session is None:
            return None
        return {
            "channel": session.channel,
            "conversation_id": session.conversation_id,
            "user_id": session.user_id,
            # 호환 필드: 작업이 없으면 대화 로그 실행 ID를 사용한다.
            "run_id": session.active_task_id or session.session_run_id,
            "session_run_id": session.session_run_id,
            "active_task_id": session.active_task_id,
            "active_role": session.active_role.value,
            "mode": session.mode,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
        }

    def set_conversation_task(
        self, channel: str, conversation_id: str, run_id: str | None
    ) -> None:
        task_id = run_id.strip() if run_id else None
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "UPDATE conversation_sessions SET active_task_id = ?, updated_at = ? "
                "WHERE channel = ? AND conversation_id = ?",
                (task_id, utc_now(), channel, conversation_id),
            )
            if cursor.rowcount != 1:
                raise StoreError(f"conversation is not bound: {channel}/{conversation_id}")

    def clear_conversation_task(self, channel: str, conversation_id: str) -> None:
        self.set_conversation_task(channel, conversation_id, None)

    def save_pending_project_request(
        self,
        channel: str,
        conversation_id: str,
        user_id: str,
        request_text: str,
        source_message_id: str,
    ) -> None:
        """Store the non-path portion of a message until project read approval."""
        text = request_text.strip()
        message_id = source_message_id.strip()
        if not text or not message_id:
            raise ValueError("pending project request text and source message are required")
        with self._lock, self._connection() as connection:
            connection.execute(
                "INSERT INTO pending_project_requests(channel, conversation_id, user_id, "
                "request_text, source_message_id, created_at) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(channel, conversation_id) DO UPDATE SET "
                "user_id = excluded.user_id, request_text = excluded.request_text, "
                "source_message_id = excluded.source_message_id, created_at = excluded.created_at",
                (
                    channel,
                    conversation_id,
                    user_id,
                    text,
                    message_id,
                    utc_now(),
                ),
            )

    def take_pending_project_request(
        self, channel: str, conversation_id: str, user_id: str
    ) -> dict[str, str] | None:
        """Return and clear the request that was waiting for project approval."""
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT request_text, source_message_id, created_at "
                "FROM pending_project_requests "
                "WHERE channel = ? AND conversation_id = ? AND user_id = ?",
                (channel, conversation_id, user_id),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "DELETE FROM pending_project_requests "
                "WHERE channel = ? AND conversation_id = ? AND user_id = ?",
                (channel, conversation_id, user_id),
            )
        return {
            "request_text": str(row["request_text"]),
            "source_message_id": str(row["source_message_id"]),
            "created_at": str(row["created_at"]),
        }

    def load_pending_project_request(
        self, channel: str, conversation_id: str, user_id: str
    ) -> dict[str, str] | None:
        """Read a request waiting for project approval without consuming it."""
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT request_text, source_message_id, created_at FROM pending_project_requests "
                "WHERE channel = ? AND conversation_id = ? AND user_id = ?",
                (channel, conversation_id, user_id),
            ).fetchone()
            if row is None:
                return None
        return {
            "request_text": str(row["request_text"]),
            "source_message_id": str(row["source_message_id"]),
            "created_at": str(row["created_at"]),
        }

    def clear_pending_project_request(
        self, channel: str, conversation_id: str, user_id: str
    ) -> bool:
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM pending_project_requests "
                "WHERE channel = ? AND conversation_id = ? AND user_id = ?",
                (channel, conversation_id, user_id),
            )
        return cursor.rowcount == 1

    def set_conversation_role(
        self, channel: str, conversation_id: str, active_role: str
    ) -> None:
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "UPDATE conversation_sessions SET active_role = ?, updated_at = ? "
                "WHERE channel = ? AND conversation_id = ?",
                (active_role, utc_now(), channel, conversation_id),
            )
            if cursor.rowcount != 1:
                raise StoreError(
                    f"conversation is not bound: {channel}/{conversation_id}"
                )

    def set_conversation_mode(
        self, channel: str, conversation_id: str, mode: str
    ) -> None:
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "UPDATE conversation_sessions SET mode = ?, updated_at = ? "
                "WHERE channel = ? AND conversation_id = ?",
                (mode, utc_now(), channel, conversation_id),
            )
            if cursor.rowcount != 1:
                raise StoreError(
                    f"conversation is not bound: {channel}/{conversation_id}"
                )

    def save_memory(
        self,
        scope: str,
        scope_key: str,
        role_id: str,
        content: str,
        *,
        source_kind: str = "agent",
        source_ref: str = "",
    ) -> int:
        now = utc_now()
        expires = (datetime.fromisoformat(now) + timedelta(days=30)).isoformat()
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT revision FROM memory_facts WHERE scope = ? AND scope_key = ? "
                "AND role_id = ? AND content = ? AND deleted_at = '' "
                "AND superseded_at = '' AND expires_at > ?",
                (scope, scope_key, role_id, content, now),
            ).fetchone()
            if row is not None:
                return int(row["revision"])
            next_row = connection.execute(
                "SELECT COALESCE(MAX(revision), 0) + 1 AS next_revision FROM memory_facts "
                "WHERE scope = ? AND scope_key = ? AND role_id = ?",
                (scope, scope_key, role_id),
            ).fetchone()
            revision = int(next_row["next_revision"])
            connection.execute(
                "INSERT INTO memory_facts(scope, scope_key, role_id, content, revision, "
                "source_kind, source_ref, confirmed_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (scope, scope_key, role_id, content, revision, source_kind,
                 source_ref, now, expires),
            )
            return revision

    def replace_memory_fact(
        self, fact_id: int, scope: str, scope_key: str, content: str,
        *, source_kind: str = "user", source_ref: str = "",
    ) -> int:
        now = utc_now()
        expires = (datetime.fromisoformat(now) + timedelta(days=30)).isoformat()
        with self._lock, self._connection() as connection:
            old = connection.execute(
                "SELECT role_id FROM memory_facts WHERE fact_id = ? AND scope = ? "
                "AND scope_key = ? AND deleted_at = '' AND superseded_at = '' "
                "AND expires_at > ?",
                (fact_id, scope, scope_key, now),
            ).fetchone()
            if old is None:
                raise StoreError("active memory fact does not exist")
            role_id = str(old["role_id"])
            next_row = connection.execute(
                "SELECT COALESCE(MAX(revision), 0) + 1 AS next_revision FROM memory_facts "
                "WHERE scope = ? AND scope_key = ? AND role_id = ?",
                (scope, scope_key, role_id),
            ).fetchone()
            revision = int(next_row["next_revision"])
            connection.execute(
                "UPDATE memory_facts SET superseded_at = ? WHERE fact_id = ?",
                (now, fact_id),
            )
            cursor = connection.execute(
                "INSERT INTO memory_facts(scope, scope_key, role_id, content, revision, "
                "source_kind, source_ref, confirmed_at, expires_at, supersedes_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (scope, scope_key, role_id, content, revision, source_kind,
                 source_ref, now, expires, fact_id),
            )
            return int(cursor.lastrowid)

    def append_compacted_memory(
        self,
        scope: str,
        scope_key: str,
        role_id: str,
        content: str,
        *,
        max_characters: int,
    ) -> int:
        if max_characters < 100:
            raise ValueError("compacted memory limit is too small")
        line = "- " + content.strip().lstrip("- ")
        now = utc_now()
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT content, revision FROM conversation_memories "
                "WHERE scope = ? AND scope_key = ? AND role_id = ?",
                (scope, scope_key, role_id),
            ).fetchone()
            lines = str(row["content"]).splitlines() if row else []
            lines.append(line)
            while lines and len("\n".join(lines)) > max_characters:
                lines.pop(0)
            revision = (int(row["revision"]) if row else 0) + 1
            connection.execute(
                "INSERT INTO conversation_memories(scope, scope_key, role_id, content, "
                "revision, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(scope, scope_key, role_id) DO UPDATE SET "
                "content = excluded.content, revision = excluded.revision, "
                "updated_at = excluded.updated_at",
                (scope, scope_key, role_id, "\n".join(lines), revision, now, now),
            )
        return revision

    def list_memories(
        self, keys: tuple[tuple[str, str], ...], *, role_id: str = ""
    ) -> list[dict[str, Any]]:
        if not keys:
            return []
        clauses = " OR ".join("(scope = ? AND scope_key = ?)" for _ in keys)
        parameters: list[Any] = [item for key in keys for item in key]
        role_clause = ""
        if role_id:
            role_clause = " AND role_id IN ('shared', 'compaction', ?)"
            parameters.append(role_id)
        with self._lock, self._connection() as connection:
            compacted = connection.execute(
                "SELECT scope, scope_key, role_id, content, revision, updated_at "
                f"FROM conversation_memories WHERE ({clauses}){role_clause} "
                "ORDER BY scope, role_id",
                parameters,
            ).fetchall()
            facts = connection.execute(
                "SELECT fact_id, scope, scope_key, role_id, content, revision, "
                "source_kind, source_ref, confirmed_at, expires_at "
                f"FROM memory_facts WHERE ({clauses}){role_clause} "
                "AND deleted_at = '' AND superseded_at = '' AND expires_at > ? "
                "ORDER BY scope, role_id, fact_id",
                [*parameters, utc_now()],
            ).fetchall()
        return [
            {
                "fact_id": 0,
                "scope": str(row["scope"]),
                "scope_key": str(row["scope_key"]),
                "role_id": str(row["role_id"]),
                "content": str(row["content"]),
                "revision": int(row["revision"]),
                "updated_at": str(row["updated_at"]),
            }
            for row in compacted
        ] + [
            {
                "fact_id": int(row["fact_id"]),
                "scope": str(row["scope"]),
                "scope_key": str(row["scope_key"]),
                "role_id": str(row["role_id"]),
                "content": str(row["content"]),
                "revision": int(row["revision"]),
                "updated_at": str(row["confirmed_at"]),
                "source_kind": str(row["source_kind"]),
                "source_ref": str(row["source_ref"]),
                "confirmed_at": str(row["confirmed_at"]),
                "expires_at": str(row["expires_at"]),
            }
            for row in facts
        ]

    def delete_memory(self, scope: str, scope_key: str, *, role_id: str = "") -> int:
        if not scope.strip() or not scope_key.strip():
            raise ValueError("memory scope and key are required")
        query = "UPDATE memory_facts SET deleted_at = ? WHERE scope = ? AND scope_key = ? "
        query += "AND deleted_at = '' AND superseded_at = ''"
        parameters: list[Any] = [scope.strip(), scope_key.strip()]
        if role_id:
            query += " AND role_id = ?"
            parameters.append(role_id)
        with self._lock, self._connection() as connection:
            cursor = connection.execute(query, [utc_now(), *parameters])
            legacy_query = "DELETE FROM conversation_memories WHERE scope = ? AND scope_key = ?"
            if role_id:
                legacy_query += " AND role_id = ?"
            legacy = connection.execute(legacy_query, parameters)
        return int(cursor.rowcount) + int(legacy.rowcount)

    def delete_memory_fact(self, fact_id: int, scope: str, scope_key: str) -> bool:
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "UPDATE memory_facts SET deleted_at = ? WHERE fact_id = ? AND scope = ? "
                "AND scope_key = ? AND deleted_at = '' AND superseded_at = ''",
                (utc_now(), fact_id, scope, scope_key),
            )
        return cursor.rowcount == 1

    def purge_expired_memory_facts(self, now: str) -> dict[str, int]:
        deleted: dict[str, int] = {}
        with self._lock, self._connection() as connection:
            for scope in ("user", "project", "conversation", "run"):
                deleted[scope] = int(connection.execute(
                    "DELETE FROM memory_facts WHERE scope = ? AND "
                    "(expires_at <= ? OR (deleted_at != '' AND deleted_at <= ?))",
                    (scope, now, now),
                ).rowcount)
        return deleted

    def backup_to(self, destination: Path) -> Path:
        destination = destination.resolve()
        if destination == self.path:
            raise ValueError("backup destination must differ from the state database")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            source = self._connect()
            target = sqlite3.connect(destination)
            try:
                source.backup(target)
            finally:
                target.close()
                source.close()
        return destination

    def purge_expired_operational_rows(self, cutoff: str) -> dict[str, int]:
        """Remove expired delivery history and terminal task records atomically."""
        deleted: dict[str, int] = {}
        statements = {
            "outbound": (
                "DELETE FROM outbound_messages WHERE status IN ('SENT', 'DEAD') "
                "AND updated_at < ?",
                (cutoff,),
            ),
            "inbound": (
                "DELETE FROM inbound_receipts WHERE status IN ('COMPLETED', 'FAILED') "
                "AND updated_at < ?",
                (cutoff,),
            ),
            "conversation_jobs": (
                "DELETE FROM conversation_jobs WHERE status IN ('COMPLETED', 'CANCELLED', 'FAILED') "
                "AND updated_at < ?",
                (cutoff,),
            ),
            "events": ("DELETE FROM events WHERE timestamp < ?", (cutoff,)),
            "messages": ("DELETE FROM messages WHERE timestamp < ?", (cutoff,)),
        }
        with self._lock, self._connection() as connection:
            for name, (statement, parameters) in statements.items():
                deleted[name] = int(connection.execute(statement, parameters).rowcount)
            deleted["conversation_sessions"] = int(connection.execute(
                "DELETE FROM conversation_sessions WHERE updated_at < ? "
                "AND active_task_id IS NULL",
                (cutoff,),
            ).rowcount)
            deleted["conversation_projects"] = int(connection.execute(
                "DELETE FROM conversation_projects WHERE selected_at < ? "
                "AND NOT EXISTS (SELECT 1 FROM conversation_sessions AS session "
                "WHERE session.channel = conversation_projects.channel "
                "AND session.conversation_id = conversation_projects.conversation_id)",
                (cutoff,),
            ).rowcount)
            terminal_run_ids = [
                str(row["run_id"])
                for row in connection.execute(
                    "SELECT run_id, state_json FROM runs WHERE updated_at < ?", (cutoff,)
                ).fetchall()
                if str(json.loads(str(row["state_json"])).get("phase", ""))
                in {"COMPLETED", "FAILED", "CANCELLED"}
                and connection.execute(
                    "SELECT 1 FROM conversation_sessions WHERE session_run_id = ?",
                    (str(row["run_id"]),),
                ).fetchone()
                is None
            ]
            deleted["runs"] = 0
            for run_id in terminal_run_ids:
                # A completed task can remain selected until the user sends another
                # message. Detach it before deleting its cascaded operational rows.
                connection.execute(
                    "UPDATE conversation_sessions SET active_task_id = NULL "
                    "WHERE active_task_id = ?",
                    (run_id,),
                )
                connection.execute(
                    "DELETE FROM conversation_bindings WHERE run_id = ?", (run_id,)
                )
                connection.execute(
                    "DELETE FROM conversation_memories WHERE scope = 'run' "
                    "AND scope_key = ?",
                    (run_id,),
                )
                connection.execute(
                    "DELETE FROM memory_facts WHERE scope = 'run' AND scope_key = ?",
                    (run_id,),
                )
                deleted["runs"] += int(
                    connection.execute("DELETE FROM runs WHERE run_id = ?", (run_id,)).rowcount
                )
        return deleted

    def active_run_ids(self) -> set[str]:
        terminal = {"COMPLETED", "FAILED", "CANCELLED"}
        with self._lock, self._connection() as connection:
            rows = connection.execute("SELECT run_id, state_json FROM runs").fetchall()
        return {
            str(row["run_id"])
            for row in rows
            if str(json.loads(str(row["state_json"])).get("phase", "")) not in terminal
        }

    def create_repository_analysis(
        self, request: RepositoryAnalysisRequest
    ) -> tuple[dict[str, Any], bool]:
        """Create one durable analysis per conversation, preserving an active request."""
        now = utc_now()
        active_statuses = tuple(
            status.value
            for status in RepositoryAnalysisStatus
            if status.active
        )
        placeholders = ", ".join("?" for _ in active_statuses)
        with self.transaction():
            with self._connection() as connection:
                existing = connection.execute(
                    "SELECT * FROM repository_analysis_jobs WHERE channel = ? "
                    "AND conversation_id = ? AND status IN (" + placeholders + ") "
                    "ORDER BY created_at DESC LIMIT 1",
                    (request.channel, request.conversation_id, *active_statuses),
                ).fetchone()
                if existing is not None:
                    return self._repository_analysis_row(existing), False
                try:
                    connection.execute(
                        "INSERT INTO repository_analysis_jobs(analysis_id, channel, conversation_id, "
                        "user_id, source_message_id, role_id, request_text, repository_path, "
                        "repository_identity, commit_sha, branch, status, phase, stop_reason, "
                        "plan_json, completed_json, remaining_json, checkpoint, model_call_state, "
                        "query_rounds, model_calls, read_bytes, no_progress_count, progress_outbound_id, "
                        "last_progress_at, lease_owner, lease_until, attempts, last_error, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', '{}', '[]', '[]', 0, "
                        "'IDLE', 0, 0, 0, 0, 0, '', '', '', 0, '', ?, ?)",
                        (
                            request.analysis_id,
                            request.channel,
                            request.conversation_id,
                            request.user_id,
                            request.source_message_id,
                            request.role_id,
                            request.request_text,
                            request.repository_path,
                            request.repository_identity,
                            request.commit_sha,
                            request.branch,
                            RepositoryAnalysisStatus.QUEUED.value,
                            RepositoryAnalysisPhase.STRUCTURE.value,
                            now,
                            now,
                        ),
                    )
                except sqlite3.IntegrityError:
                    existing = connection.execute(
                        "SELECT * FROM repository_analysis_jobs WHERE channel = ? "
                        "AND conversation_id = ? AND status IN (" + placeholders + ") "
                        "ORDER BY created_at DESC LIMIT 1",
                        (request.channel, request.conversation_id, *active_statuses),
                    ).fetchone()
                    if existing is None:
                        raise
                    return self._repository_analysis_row(existing), False
                saved = connection.execute(
                    "SELECT * FROM repository_analysis_jobs WHERE analysis_id = ?",
                    (request.analysis_id,),
                ).fetchone()
        if saved is None:
            raise StoreError("repository analysis was not saved")
        return self._repository_analysis_row(saved), True

    def repository_analysis(
        self, analysis_id: str
    ) -> dict[str, Any] | None:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM repository_analysis_jobs WHERE analysis_id = ?",
                (analysis_id,),
            ).fetchone()
        return self._repository_analysis_row(row) if row else None

    def active_repository_analysis(
        self, channel: str, conversation_id: str
    ) -> dict[str, Any] | None:
        active_statuses = tuple(
            status.value
            for status in RepositoryAnalysisStatus
            if status.active
        )
        placeholders = ", ".join("?" for _ in active_statuses)
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM repository_analysis_jobs WHERE channel = ? "
                "AND conversation_id = ? AND status IN (" + placeholders + ") "
                "ORDER BY created_at DESC LIMIT 1",
                (channel, conversation_id, *active_statuses),
            ).fetchone()
        return self._repository_analysis_row(row) if row else None

    def repository_analysis_summary(
        self, channel: str, conversation_id: str
    ) -> dict[str, Any] | None:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM repository_analysis_jobs WHERE channel = ? "
                "AND conversation_id = ? ORDER BY created_at DESC LIMIT 1",
                (channel, conversation_id),
            ).fetchone()
        return self._repository_analysis_row(row) if row else None

    def claim_next_repository_analysis(
        self, lease_owner: str, *, lease_seconds: int = 120
    ) -> dict[str, Any] | None:
        if not lease_owner.strip() or lease_seconds < 1:
            raise ValueError("repository analysis claim settings are invalid")
        now = datetime.now(timezone.utc)
        now_text = now.isoformat()
        lease_until = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self.transaction():
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT * FROM repository_analysis_jobs WHERE status = 'QUEUED' "
                    "ORDER BY created_at LIMIT 1"
                ).fetchone()
                if row is None:
                    return None
                analysis_id = str(row["analysis_id"])
                cursor = connection.execute(
                    "UPDATE repository_analysis_jobs SET status = 'PROCESSING', attempts = attempts + 1, "
                    "lease_owner = ?, lease_until = ?, active_since = ?, updated_at = ? "
                    "WHERE analysis_id = ? AND status = 'QUEUED'",
                    (lease_owner, lease_until, now_text, now_text, analysis_id),
                )
                if cursor.rowcount != 1:
                    return None
                claimed = connection.execute(
                    "SELECT * FROM repository_analysis_jobs WHERE analysis_id = ?",
                    (analysis_id,),
                ).fetchone()
        return self._repository_analysis_row(claimed) if claimed else None

    def heartbeat_repository_analysis(
        self, analysis_id: str, lease_owner: str, *, lease_seconds: int = 120
    ) -> None:
        now = datetime.now(timezone.utc)
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "UPDATE repository_analysis_jobs SET lease_until = ?, updated_at = ? "
                "WHERE analysis_id = ? AND lease_owner = ? "
                "AND status IN ('PROCESSING', 'STOP_REQUESTED')",
                (
                    (now + timedelta(seconds=lease_seconds)).isoformat(),
                    now.isoformat(),
                    analysis_id,
                    lease_owner,
                ),
            )
            if cursor.rowcount != 1:
                raise StoreError(f"repository analysis lease was lost: {analysis_id}")

    def set_repository_analysis_model_call_state(
        self, analysis_id: str, lease_owner: str, state: str
    ) -> None:
        if state not in {"IDLE", "STARTED"}:
            raise ValueError("repository analysis model call state is invalid")
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "UPDATE repository_analysis_jobs SET model_call_state = ?, updated_at = ? "
                "WHERE analysis_id = ? AND lease_owner = ? "
                "AND status IN ('PROCESSING', 'STOP_REQUESTED')",
                (state, utc_now(), analysis_id, lease_owner),
            )
            if cursor.rowcount != 1:
                raise StoreError(f"repository analysis is not owned: {analysis_id}")

    def save_repository_analysis_plan(
        self, analysis_id: str, lease_owner: str, plan: dict[str, Any]
    ) -> dict[str, Any]:
        batches = plan.get("batches")
        if not isinstance(batches, list) or not batches:
            raise ValueError("repository analysis plan requires batches")
        now = utc_now()
        with self.transaction():
            with self._connection() as connection:
                current = connection.execute(
                    "SELECT commit_sha, repository_identity FROM repository_analysis_jobs "
                    "WHERE analysis_id = ? AND lease_owner = ? AND status = 'PROCESSING'",
                    (analysis_id, lease_owner),
                ).fetchone()
                if current is None:
                    raise StoreError(f"repository analysis plan is not owned: {analysis_id}")
                if (
                    str(plan.get("commit_sha", "")) != str(current["commit_sha"])
                    or str(plan.get("identity_hash", ""))
                    != str(current["repository_identity"])
                ):
                    raise StoreError("repository analysis plan snapshot does not match the job")
                connection.execute(
                    "DELETE FROM repository_analysis_batches WHERE analysis_id = ?",
                    (analysis_id,),
                )
                for index, batch in enumerate(batches):
                    if (
                        not isinstance(batch, dict)
                        or int(batch.get("batch_index", -1)) != index
                        or not isinstance(batch.get("phase"), str)
                        or not isinstance(batch.get("paths"), list)
                    ):
                        raise ValueError("repository analysis batch is invalid")
                    connection.execute(
                        "INSERT INTO repository_analysis_batches(analysis_id, batch_index, phase, target_json, "
                        "status, created_at, completed_at) VALUES (?, ?, ?, ?, 'PENDING', ?, '')",
                        (
                            analysis_id,
                            index,
                            str(batch["phase"]),
                            json.dumps(batch, ensure_ascii=False, sort_keys=True),
                            now,
                        ),
                    )
                cursor = connection.execute(
                    "UPDATE repository_analysis_jobs SET plan_json = ?, remaining_json = ?, "
                    "updated_at = ? WHERE analysis_id = ? AND lease_owner = ? AND status = 'PROCESSING'",
                    (
                        json.dumps(plan, ensure_ascii=False, sort_keys=True),
                        json.dumps(batches, ensure_ascii=False, sort_keys=True),
                        now,
                        analysis_id,
                        lease_owner,
                    ),
                )
                if cursor.rowcount != 1:
                    raise StoreError(f"repository analysis plan is not owned: {analysis_id}")
                row = connection.execute(
                    "SELECT * FROM repository_analysis_jobs WHERE analysis_id = ?",
                    (analysis_id,),
                ).fetchone()
        if row is None:
            raise StoreError(f"repository analysis disappeared: {analysis_id}")
        return self._repository_analysis_row(row)

    def next_repository_analysis_batch(
        self, analysis_id: str
    ) -> dict[str, Any] | None:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM repository_analysis_batches WHERE analysis_id = ? "
                "AND status = 'PENDING' ORDER BY batch_index LIMIT 1",
                (analysis_id,),
            ).fetchone()
        return self._repository_analysis_batch_row(row) if row else None

    def complete_repository_analysis_batch(
        self,
        analysis_id: str,
        lease_owner: str,
        batch_index: int,
        *,
        completed: list[dict[str, Any]],
        remaining: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        query_rounds: int,
        model_calls: int,
        read_bytes: int,
        no_progress_count: int,
        file_exclusions: list[dict[str, str]] | None = None,
        analysis_partial_reasons: list[str] | None = None,
        final_response: str = "",
        final_status: RepositoryAnalysisStatus = RepositoryAnalysisStatus.COMPLETED,
        final_reason: str = "COMPLETED",
    ) -> dict[str, Any]:
        if (
            min(query_rounds, model_calls, read_bytes, no_progress_count) < 0
            or final_status not in {
                RepositoryAnalysisStatus.COMPLETED,
                RepositoryAnalysisStatus.PARTIAL_COMPLETED,
            }
            or not final_reason.strip()
        ):
            raise ValueError("repository analysis completion is invalid")
        now = utc_now()
        with self.transaction():
            with self._connection() as connection:
                job = connection.execute(
                    "SELECT * FROM repository_analysis_jobs WHERE analysis_id = ? AND lease_owner = ? "
                    "AND status = 'PROCESSING'",
                    (analysis_id, lease_owner),
                ).fetchone()
                if job is None:
                    raise StoreError(f"repository analysis batch is not owned: {analysis_id}")
                batch = connection.execute(
                    "SELECT * FROM repository_analysis_batches WHERE analysis_id = ? AND batch_index = ? "
                    "AND status = 'PENDING'",
                    (analysis_id, batch_index),
                ).fetchone()
                if batch is None:
                    raise StoreError(f"repository analysis batch is not pending: {batch_index}")
                phase = str(batch["phase"])
                plan = json.loads(str(job["plan_json"]))
                if file_exclusions:
                    planned_paths = set(json.loads(str(batch["target_json"])).get("paths", []))
                    for exclusion in file_exclusions:
                        path = str(exclusion["path"])
                        reason = str(exclusion["reason"])
                        if path not in planned_paths or reason not in {"binary_content", "non_utf8"}:
                            raise ValueError("repository analysis file exclusion is invalid")
                        file_entry = next(
                            (item for item in plan["files"] if item["path"] == path), None
                        )
                        if file_entry is None:
                            raise ValueError("repository analysis excluded path is not planned")
                        file_entry["exclude_reason"] = reason
                        file_entry["eligible"] = False
                        plan["excluded"][reason] = plan["excluded"].get(reason, 0) + 1
                    if "UNREADABLE_FILE" not in plan["partial_reasons"]:
                        plan["partial_reasons"].append("UNREADABLE_FILE")
                for reason in analysis_partial_reasons or []:
                    if reason not in {"UNATTRIBUTED_ANALYSIS"}:
                        raise ValueError("repository analysis partial reason is invalid")
                    if reason not in plan["partial_reasons"]:
                        plan["partial_reasons"].append(reason)
                for item in evidence:
                    path = str(item.get("path", "")).strip()
                    start_line = int(item.get("start_line", 0))
                    end_line = int(item.get("end_line", 0))
                    kind = str(item.get("kind", "")).strip()
                    if not path or start_line < 1 or end_line < start_line or not kind:
                        raise ValueError("repository analysis evidence location is invalid")
                    connection.execute(
                        "INSERT OR IGNORE INTO repository_analysis_evidence(analysis_id, commit_sha, path, "
                        "start_line, end_line, phase, kind, summary, redaction_status, untrusted_repository_data, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)",
                        (
                            analysis_id,
                            str(job["commit_sha"]),
                            path,
                            start_line,
                            end_line,
                            phase,
                            kind,
                            str(item.get("summary", "")),
                            str(item.get("redaction_status", "redacted")),
                            now,
                        ),
                    )
                connection.execute(
                    "UPDATE repository_analysis_batches SET status = 'COMPLETED', completed_at = ? "
                    "WHERE analysis_id = ? AND batch_index = ? AND status = 'PENDING'",
                    (now, analysis_id, batch_index),
                )
                cursor = connection.execute(
                    "UPDATE repository_analysis_jobs SET phase = ?, completed_json = ?, remaining_json = ?, "
                    "checkpoint = checkpoint + 1, model_call_state = 'IDLE', query_rounds = ?, "
                    "model_calls = ?, read_bytes = ?, no_progress_count = ?, plan_json = ?, updated_at = ? "
                    "WHERE analysis_id = ? AND lease_owner = ? AND status = 'PROCESSING'",
                    (
                        phase,
                        json.dumps(completed, ensure_ascii=False, sort_keys=True),
                        json.dumps(remaining, ensure_ascii=False, sort_keys=True),
                        query_rounds,
                        model_calls,
                        read_bytes,
                        no_progress_count,
                        json.dumps(plan, ensure_ascii=False, sort_keys=True),
                        now,
                        analysis_id,
                        lease_owner,
                    ),
                )
                if cursor.rowcount != 1:
                    raise StoreError(f"repository analysis checkpoint is not owned: {analysis_id}")
                if final_response.strip():
                    final_outbound_id = self._queue_repository_analysis_final(
                        connection, job, final_response, now
                    )
                    connection.execute(
                        "UPDATE repository_analysis_jobs SET status = ?, stop_reason = ?, "
                        "final_response = ?, final_outbound_id = ?, delivery_status = 'PENDING', "
                        "lease_owner = '', lease_until = '', "
                        "active_seconds = active_seconds + CASE WHEN active_since = '' THEN 0 "
                        "ELSE MAX(0, (julianday(?) - julianday(active_since)) * 86400) END, "
                        "active_since = '', updated_at = ? WHERE analysis_id = ?",
                        (final_status.value, final_reason, final_response.strip(),
                         final_outbound_id, now, now, analysis_id),
                    )
                row = connection.execute(
                    "SELECT * FROM repository_analysis_jobs WHERE analysis_id = ?",
                    (analysis_id,),
                ).fetchone()
        if row is None:
            raise StoreError(f"repository analysis disappeared: {analysis_id}")
        return self._repository_analysis_row(row)

    def checkpoint_repository_analysis(
        self,
        analysis_id: str,
        lease_owner: str,
        *,
        phase: str,
        completed: list[dict[str, Any]],
        remaining: list[dict[str, Any]],
        plan: dict[str, Any] | None = None,
        query_rounds: int | None = None,
        model_calls: int | None = None,
        read_bytes: int | None = None,
        no_progress_count: int | None = None,
    ) -> dict[str, Any]:
        if phase not in {item.value for item in RepositoryAnalysisPhase}:
            raise ValueError("repository analysis phase is invalid")
        values = {
            "phase": phase,
            "completed_json": json.dumps(completed, ensure_ascii=False, sort_keys=True),
            "remaining_json": json.dumps(remaining, ensure_ascii=False, sort_keys=True),
            "plan_json": json.dumps(plan or {}, ensure_ascii=False, sort_keys=True),
            "updated_at": utc_now(),
        }
        for key, value in {
            "query_rounds": query_rounds,
            "model_calls": model_calls,
            "read_bytes": read_bytes,
            "no_progress_count": no_progress_count,
        }.items():
            if value is not None:
                if value < 0:
                    raise ValueError(f"repository analysis {key} cannot be negative")
                values[key] = value
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "UPDATE repository_analysis_jobs SET checkpoint = checkpoint + 1, model_call_state = 'IDLE', "
                + assignments
                + " WHERE analysis_id = ? AND lease_owner = ? AND status = 'PROCESSING'",
                (*values.values(), analysis_id, lease_owner),
            )
            if cursor.rowcount != 1:
                raise StoreError(f"repository analysis checkpoint is not owned: {analysis_id}")
            row = connection.execute(
                "SELECT * FROM repository_analysis_jobs WHERE analysis_id = ?",
                (analysis_id,),
            ).fetchone()
        if row is None:
            raise StoreError(f"repository analysis disappeared: {analysis_id}")
        return self._repository_analysis_row(row)

    def add_repository_analysis_evidence(
        self,
        analysis_id: str,
        *,
        commit_sha: str,
        path: str,
        start_line: int,
        end_line: int,
        phase: str,
        kind: str,
        summary: str,
        redaction_status: str = "redacted",
    ) -> None:
        if start_line < 1 or end_line < start_line or not path.strip() or not kind.strip():
            raise ValueError("repository analysis evidence location is invalid")
        if phase not in {item.value for item in RepositoryAnalysisPhase}:
            raise ValueError("repository analysis evidence phase is invalid")
        with self._lock, self._connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO repository_analysis_evidence(analysis_id, commit_sha, path, "
                "start_line, end_line, phase, kind, summary, redaction_status, untrusted_repository_data, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)",
                (
                    analysis_id,
                    commit_sha,
                    path,
                    start_line,
                    end_line,
                    phase,
                    kind,
                    summary,
                    redaction_status,
                    utc_now(),
                ),
            )

    def repository_analysis_evidence(self, analysis_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM repository_analysis_evidence WHERE analysis_id = ? "
                "ORDER BY evidence_id",
                (analysis_id,),
            ).fetchall()
        return [
            {
                "evidence_id": int(row["evidence_id"]),
                "commit_sha": str(row["commit_sha"]),
                "path": str(row["path"]),
                "start_line": int(row["start_line"]),
                "end_line": int(row["end_line"]),
                "phase": str(row["phase"]),
                "kind": str(row["kind"]),
                "summary": str(row["summary"]),
                "redaction_status": str(row["redaction_status"]),
                "untrusted_repository_data": bool(row["untrusted_repository_data"]),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

    def requeue_repository_analysis(self, analysis_id: str, lease_owner: str) -> dict[str, Any]:
        now = utc_now()
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "UPDATE repository_analysis_jobs SET status = 'QUEUED', lease_owner = '', lease_until = '', "
                "active_seconds = active_seconds + CASE WHEN active_since = '' THEN 0 "
                "ELSE MAX(0, (julianday(?) - julianday(active_since)) * 86400) END, "
                "active_since = '', updated_at = ? WHERE analysis_id = ? AND lease_owner = ? AND status = 'PROCESSING' "
                "AND model_call_state = 'IDLE'",
                (now, now, analysis_id, lease_owner),
            )
            if cursor.rowcount != 1:
                raise StoreError(f"repository analysis cannot be requeued: {analysis_id}")
            row = connection.execute(
                "SELECT * FROM repository_analysis_jobs WHERE analysis_id = ?",
                (analysis_id,),
            ).fetchone()
        if row is None:
            raise StoreError(f"repository analysis disappeared: {analysis_id}")
        return self._repository_analysis_row(row)

    def queue_repository_analysis_progress(
        self, analysis_id: str, lease_owner: str, text: str
    ) -> int:
        if not text.strip():
            raise ValueError("repository analysis progress text is required")
        now = utc_now()
        with self.transaction():
            with self._connection() as connection:
                job = connection.execute(
                    "SELECT * FROM repository_analysis_jobs WHERE analysis_id = ? AND lease_owner = ? "
                    "AND status IN ('PROCESSING', 'STOP_REQUESTED')",
                    (analysis_id, lease_owner),
                ).fetchone()
                if job is None:
                    raise StoreError(f"repository analysis progress is not owned: {analysis_id}")
                progress_id = int(job["progress_outbound_id"])
                if progress_id:
                    cursor = connection.execute(
                        "INSERT INTO outbound_messages(channel, conversation_id, text, reply_to, "
                        "delivery_mode, target_outbound_id, coalesce_key, status, attempts, "
                        "external_ids_json, last_error, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, 'edit', ?, ?, 'PENDING', 0, '[]', '', ?, ?)",
                        (
                            str(job["channel"]),
                            str(job["conversation_id"]),
                            text.strip(),
                            str(job["source_message_id"]),
                            progress_id,
                            f"repository-analysis:{analysis_id}",
                            now,
                            now,
                        ),
                    )
                    return int(cursor.lastrowid)
                cursor = connection.execute(
                    "INSERT INTO outbound_messages(channel, conversation_id, text, reply_to, "
                    "delivery_mode, target_outbound_id, coalesce_key, status, attempts, "
                    "external_ids_json, last_error, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, 'send', 0, '', 'PENDING', 0, '[]', '', ?, ?)",
                    (
                        str(job["channel"]),
                        str(job["conversation_id"]),
                        text.strip(),
                        str(job["source_message_id"]),
                        now,
                        now,
                    ),
                )
                progress_id = int(cursor.lastrowid)
                connection.execute(
                    "UPDATE repository_analysis_jobs SET progress_outbound_id = ?, last_progress_at = ?, "
                    "updated_at = ? WHERE analysis_id = ? AND lease_owner = ?",
                    (progress_id, now, now, analysis_id, lease_owner),
                )
                return progress_id

    def finish_repository_analysis(
        self,
        analysis_id: str,
        lease_owner: str,
        status: RepositoryAnalysisStatus,
        *,
        reason: str = "",
        error: str = "",
    ) -> dict[str, Any]:
        if status.active or status == RepositoryAnalysisStatus.QUEUED:
            raise ValueError("repository analysis finish status must be terminal")
        now = utc_now()
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "UPDATE repository_analysis_jobs SET status = ?, stop_reason = ?, last_error = ?, "
                "lease_owner = '', lease_until = '', "
                "active_seconds = active_seconds + CASE WHEN active_since = '' THEN 0 "
                "ELSE MAX(0, (julianday(?) - julianday(active_since)) * 86400) END, "
                "active_since = '', updated_at = ? "
                "WHERE analysis_id = ? AND lease_owner = ? "
                "AND status IN ('PROCESSING', 'STOP_REQUESTED')",
                (status.value, reason, error, now, now, analysis_id, lease_owner),
            )
            if cursor.rowcount != 1:
                raise StoreError(f"repository analysis cannot finish: {analysis_id}")
            row = connection.execute(
                "SELECT * FROM repository_analysis_jobs WHERE analysis_id = ?",
                (analysis_id,),
            ).fetchone()
        if row is None:
            raise StoreError(f"repository analysis disappeared: {analysis_id}")
        return self._repository_analysis_row(row)

    def pause_repository_analysis(self, analysis_id: str, lease_owner: str) -> dict[str, Any]:
        now = utc_now()
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "UPDATE repository_analysis_jobs SET status = 'PAUSED', model_call_state = 'IDLE', "
                "lease_owner = '', lease_until = '', "
                "active_seconds = active_seconds + CASE WHEN active_since = '' THEN 0 "
                "ELSE MAX(0, (julianday(?) - julianday(active_since)) * 86400) END, "
                "active_since = '', updated_at = ? WHERE analysis_id = ? "
                "AND lease_owner = ? AND status IN ('PROCESSING', 'STOP_REQUESTED')",
                (now, now, analysis_id, lease_owner),
            )
            if cursor.rowcount != 1:
                raise StoreError(f"repository analysis cannot pause: {analysis_id}")
            row = connection.execute(
                "SELECT * FROM repository_analysis_jobs WHERE analysis_id = ?",
                (analysis_id,),
            ).fetchone()
        if row is None:
            raise StoreError(f"repository analysis disappeared: {analysis_id}")
        return self._repository_analysis_row(row)

    def requeue_cancelled_repository_analysis(
        self, analysis_id: str, lease_owner: str
    ) -> dict[str, Any]:
        """A locally confirmed cancellation is safe to resume from its prior checkpoint."""
        now = utc_now()
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "UPDATE repository_analysis_jobs SET status = 'QUEUED', model_call_state = 'IDLE', "
                "lease_owner = '', lease_until = '', "
                "active_seconds = active_seconds + CASE WHEN active_since = '' THEN 0 "
                "ELSE MAX(0, (julianday(?) - julianday(active_since)) * 86400) END, "
                "active_since = '', updated_at = ? WHERE analysis_id = ? "
                "AND lease_owner = ? AND status = 'PROCESSING'",
                (now, now, analysis_id, lease_owner),
            )
            if cursor.rowcount != 1:
                raise StoreError(f"repository analysis cannot be safely requeued: {analysis_id}")
            row = connection.execute(
                "SELECT * FROM repository_analysis_jobs WHERE analysis_id = ?",
                (analysis_id,),
            ).fetchone()
        if row is None:
            raise StoreError(f"repository analysis disappeared: {analysis_id}")
        return self._repository_analysis_row(row)

    def finish_repository_analysis_with_response(
        self,
        analysis_id: str,
        lease_owner: str,
        status: RepositoryAnalysisStatus,
        response: str,
        *,
        reason: str,
        error: str = "",
        model_calls_increment: int = 0,
        query_rounds_increment: int = 0,
    ) -> dict[str, Any]:
        if (
            status.active
            or not response.strip()
            or min(model_calls_increment, query_rounds_increment) < 0
        ):
            raise ValueError("repository analysis response finish is invalid")
        now = utc_now()
        with self.transaction():
            with self._connection() as connection:
                job = connection.execute(
                    "SELECT * FROM repository_analysis_jobs WHERE analysis_id = ? AND lease_owner = ? "
                    "AND status IN ('PROCESSING', 'STOP_REQUESTED')",
                    (analysis_id, lease_owner),
                ).fetchone()
                if job is None:
                    raise StoreError(f"repository analysis cannot finish: {analysis_id}")
                final_outbound_id = self._queue_repository_analysis_final(
                    connection, job, response, now
                )
                cursor = connection.execute(
                    "UPDATE repository_analysis_jobs SET status = ?, stop_reason = ?, last_error = ?, "
                    "final_response = ?, final_outbound_id = ?, delivery_status = 'PENDING', "
                    "model_call_state = 'IDLE', model_calls = model_calls + ?, "
                    "query_rounds = query_rounds + ?, lease_owner = '', lease_until = '', "
                    "active_seconds = active_seconds + CASE WHEN active_since = '' THEN 0 "
                    "ELSE MAX(0, (julianday(?) - julianday(active_since)) * 86400) END, "
                    "active_since = '', updated_at = ? "
                    "WHERE analysis_id = ? AND lease_owner = ? "
                    "AND status IN ('PROCESSING', 'STOP_REQUESTED')",
                    (
                        status.value, reason, error, response.strip(), final_outbound_id,
                        model_calls_increment, query_rounds_increment,
                        now, now, analysis_id, lease_owner,
                    ),
                )
                if cursor.rowcount != 1:
                    raise StoreError(f"repository analysis cannot finish: {analysis_id}")
                row = connection.execute(
                    "SELECT * FROM repository_analysis_jobs WHERE analysis_id = ?",
                    (analysis_id,),
                ).fetchone()
        if row is None:
            raise StoreError(f"repository analysis disappeared: {analysis_id}")
        return self._repository_analysis_row(row)

    def request_repository_analysis_stop(
        self, channel: str, conversation_id: str
    ) -> dict[str, Any] | None:
        now = utc_now()
        with self._lock, self._connection() as connection:
            connection.execute(
                "UPDATE repository_analysis_jobs SET status = 'PAUSED', stop_reason = 'USER_STOPPED', "
                "updated_at = ? WHERE channel = ? AND conversation_id = ? AND status = 'QUEUED'",
                (now, channel, conversation_id),
            )
            connection.execute(
                "UPDATE repository_analysis_jobs SET status = 'STOP_REQUESTED', stop_reason = 'USER_STOPPED', "
                "updated_at = ? WHERE channel = ? AND conversation_id = ? AND status = 'PROCESSING'",
                (now, channel, conversation_id),
            )
        return self.active_repository_analysis(channel, conversation_id) or self.repository_analysis_summary(
            channel, conversation_id
        )

    def supersede_repository_analysis(
        self, channel: str, conversation_id: str
    ) -> dict[str, Any] | None:
        """Replace a queued/paused analysis or request safe replacement of an active call."""
        now = utc_now()
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM repository_analysis_jobs WHERE channel = ? AND conversation_id = ? "
                "AND status IN ('QUEUED', 'PROCESSING', 'STOP_REQUESTED', 'PAUSED') "
                "ORDER BY created_at DESC LIMIT 1",
                (channel, conversation_id),
            ).fetchone()
            if row is None:
                return None
            analysis_id = str(row["analysis_id"])
            status = str(row["status"])
            if status in {"QUEUED", "PAUSED"}:
                progress_id = int(row["progress_outbound_id"])
                if progress_id:
                    connection.execute(
                        "INSERT INTO outbound_messages(channel, conversation_id, text, reply_to, "
                        "delivery_mode, target_outbound_id, coalesce_key, status, attempts, "
                        "external_ids_json, last_error, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, 'edit', ?, ?, 'PENDING', 0, '[]', '', ?, ?)",
                        (
                            str(row["channel"]),
                            str(row["conversation_id"]),
                            "[장기 저장소 분석 · 교체됨]\n\n"
                            "사용자가 새 작업을 시작해 이 분석을 종료했습니다.",
                            str(row["source_message_id"]),
                            progress_id,
                            f"repository-analysis:{analysis_id}",
                            now,
                            now,
                        ),
                    )
                connection.execute(
                    "UPDATE repository_analysis_jobs SET status = 'SUPERSEDED', "
                    "stop_reason = 'SUPERSEDED', model_call_state = 'IDLE', "
                    "lease_owner = '', lease_until = '', updated_at = ? WHERE analysis_id = ?",
                    (now, analysis_id),
                )
            elif status in {"PROCESSING", "STOP_REQUESTED"}:
                connection.execute(
                    "UPDATE repository_analysis_jobs SET status = 'STOP_REQUESTED', "
                    "stop_reason = 'SUPERSEDED', updated_at = ? WHERE analysis_id = ?",
                    (now, analysis_id),
                )
            updated = connection.execute(
                "SELECT * FROM repository_analysis_jobs WHERE analysis_id = ?",
                (analysis_id,),
            ).fetchone()
        return self._repository_analysis_row(updated) if updated else None

    def resume_repository_analysis(
        self, channel: str, conversation_id: str
    ) -> dict[str, Any] | None:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM repository_analysis_jobs WHERE channel = ? AND conversation_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (channel, conversation_id),
            ).fetchone()
            if row is None:
                return None
            job = self._repository_analysis_row(row)
            if job["status"] != RepositoryAnalysisStatus.PAUSED.value:
                return job
            if job["model_call_state"] != "IDLE":
                cursor = connection.execute(
                    "UPDATE repository_analysis_jobs SET status = 'NEEDS_ATTENTION', "
                    "stop_reason = 'MODEL_OUTCOME_UNKNOWN', updated_at = ? WHERE analysis_id = ?",
                    (utc_now(), job["analysis_id"]),
                )
                if cursor.rowcount != 1:
                    raise StoreError(f"repository analysis cannot be recovered: {job['analysis_id']}")
            else:
                connection.execute(
                    "UPDATE repository_analysis_jobs SET status = 'QUEUED', stop_reason = '', "
                    "updated_at = ? WHERE analysis_id = ?",
                    (utc_now(), job["analysis_id"]),
                )
            updated = connection.execute(
                "SELECT * FROM repository_analysis_jobs WHERE analysis_id = ?",
                (job["analysis_id"],),
            ).fetchone()
        return self._repository_analysis_row(updated) if updated else None

    def recover_stale_repository_analyses(self) -> list[dict[str, Any]]:
        now = utc_now()
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM repository_analysis_jobs WHERE status IN ('PROCESSING', 'STOP_REQUESTED') "
                "AND lease_until <> '' AND lease_until <= ?",
                (now,),
            ).fetchall()
            for row in rows:
                call_state = str(row["model_call_state"])
                stop_reason = str(row["stop_reason"])
                if call_state == "STARTED":
                    status = RepositoryAnalysisStatus.NEEDS_ATTENTION.value
                    reason = "MODEL_OUTCOME_UNKNOWN"
                elif str(row["status"]) == "STOP_REQUESTED" and stop_reason == "USER_STOPPED":
                    status = RepositoryAnalysisStatus.PAUSED.value
                    reason = "USER_STOPPED"
                elif str(row["status"]) == "STOP_REQUESTED" and stop_reason == "SUPERSEDED":
                    status = RepositoryAnalysisStatus.SUPERSEDED.value
                    reason = "SUPERSEDED"
                else:
                    status = RepositoryAnalysisStatus.QUEUED.value
                    reason = ""
                connection.execute(
                    "UPDATE repository_analysis_jobs SET status = ?, stop_reason = ?, lease_owner = '', "
                    "lease_until = '', active_seconds = active_seconds + CASE "
                    "WHEN active_since = '' THEN 0 ELSE MAX(0, (julianday(lease_until) - "
                    "julianday(active_since)) * 86400) END, active_since = '', updated_at = ? "
                    "WHERE analysis_id = ?",
                    (status, reason, now, str(row["analysis_id"])),
                )
        recovered: list[dict[str, Any]] = []
        for row in rows:
            current = self.repository_analysis(str(row["analysis_id"]))
            if current is not None:
                recovered.append(current)
        return recovered

    def enqueue_conversation_job(
        self,
        channel: str,
        conversation_id: str,
        user_id: str,
        external_message_id: str,
        message: dict[str, Any],
    ) -> dict[str, Any]:
        now = utc_now()
        payload = json.dumps(message, ensure_ascii=False, sort_keys=True)
        with self._lock, self._connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO conversation_jobs(channel, conversation_id, "
                "user_id, external_message_id, message_json, status, attempts, "
                "lease_owner, lease_until, last_error, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 'QUEUED', 0, '', '', '', ?, ?)",
                (
                    channel,
                    conversation_id,
                    user_id,
                    external_message_id,
                    payload,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM conversation_jobs WHERE channel = ? "
                "AND conversation_id = ? AND external_message_id = ?",
                (channel, conversation_id, external_message_id),
            ).fetchone()
        if row is None:
            raise StoreError("conversation job was not saved")
        return self._conversation_job_row(row)

    def claim_next_conversation_job(
        self,
        lease_owner: str,
        *,
        lease_seconds: int = 120,
        max_attempts: int = 2,
    ) -> dict[str, Any] | None:
        if not lease_owner.strip() or lease_seconds < 1 or max_attempts < 1:
            raise ValueError("conversation claim settings are invalid")
        now = datetime.now(timezone.utc)
        now_text = now.isoformat()
        lease_until = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self.transaction():
            with self._connection() as connection:
                connection.execute(
                    "UPDATE conversation_jobs SET status = 'NEEDS_ATTENTION', "
                    "last_error = 'conversation retry limit reached', "
                    "lease_owner = '', lease_until = '', updated_at = ? "
                    "WHERE status = 'QUEUED' AND attempts >= ? AND replay_cached = 0",
                    (now_text, max_attempts),
                )
                row = connection.execute(
                    "SELECT queued.* FROM conversation_jobs AS queued "
                    "WHERE queued.status = 'QUEUED' AND "
                    "(queued.attempts < ? OR queued.replay_cached = 1) "
                    "AND NOT EXISTS ("
                    "SELECT 1 FROM conversation_jobs AS active "
                    "WHERE active.channel = queued.channel "
                    "AND active.conversation_id = queued.conversation_id "
                    "AND active.status IN ('PROCESSING', 'CANCEL_REQUESTED')) "
                    "ORDER BY queued.job_id LIMIT 1",
                    (max_attempts,),
                ).fetchone()
                if row is None:
                    return None
                job_id = int(row["job_id"])
                cursor = connection.execute(
                    "UPDATE conversation_jobs SET status = 'PROCESSING', "
                    "attempts = attempts + 1, lease_owner = ?, lease_until = ?, "
                    "updated_at = ? WHERE job_id = ? AND status = 'QUEUED' "
                    "AND (attempts < ? OR replay_cached = 1)",
                    (lease_owner, lease_until, now_text, job_id, max_attempts),
                )
                if cursor.rowcount != 1:
                    return None
                claimed = connection.execute(
                    "SELECT * FROM conversation_jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
        return self._conversation_job_row(claimed) if claimed else None

    def finish_conversation_job(
        self,
        job_id: int,
        lease_owner: str,
        status: str,
        messages: list[dict[str, str]] | None = None,
        error: str = "",
    ) -> list[int]:
        if status not in {"COMPLETED", "FAILED", "CANCELLED", "NEEDS_ATTENTION"}:
            raise ValueError(f"unsupported conversation job status: {status}")
        now = utc_now()
        identifiers: list[int] = []
        with self.transaction():
            with self._connection() as connection:
                cursor = connection.execute(
                    "UPDATE conversation_jobs SET status = ?, last_error = ?, replay_cached = 0, "
                    "lease_owner = '', lease_until = '', updated_at = ? "
                    "WHERE job_id = ? AND lease_owner = ? "
                    "AND status IN ('PROCESSING', 'CANCEL_REQUESTED')",
                    (status, error, now, job_id, lease_owner),
                )
                if cursor.rowcount != 1:
                    raise StoreError(f"conversation job is not owned: {job_id}")
                for message in messages or []:
                    queued = connection.execute(
                        "INSERT INTO outbound_messages(channel, conversation_id, text, "
                        "reply_to, status, attempts, external_ids_json, last_error, "
                        "created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, 'PENDING', 0, '[]', '', ?, ?)",
                        (
                            message["channel"],
                            message["conversation_id"],
                            message["text"],
                            message.get("reply_to", ""),
                            now,
                            now,
                        ),
                    )
                    identifiers.append(int(queued.lastrowid))
        return identifiers

    def heartbeat_conversation_job(
        self,
        job_id: int,
        lease_owner: str,
        *,
        lease_seconds: int = 120,
    ) -> None:
        now = datetime.now(timezone.utc)
        lease_until = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "UPDATE conversation_jobs SET lease_until = ?, updated_at = ? "
                "WHERE job_id = ? AND lease_owner = ? "
                "AND status IN ('PROCESSING', 'CANCEL_REQUESTED')",
                (lease_until, now.isoformat(), job_id, lease_owner),
            )
            if cursor.rowcount != 1:
                raise StoreError(f"conversation job lease was lost: {job_id}")

    def request_conversation_cancel(
        self, channel: str, conversation_id: str
    ) -> str | None:
        now = utc_now()
        with self._lock, self._connection() as connection:
            connection.execute(
                "UPDATE conversation_jobs SET status = 'CANCELLED', updated_at = ? "
                "WHERE channel = ? AND conversation_id = ? AND status = 'QUEUED'",
                (now, channel, conversation_id),
            )
            connection.execute(
                "UPDATE conversation_jobs SET status = 'CANCEL_REQUESTED', updated_at = ? "
                "WHERE channel = ? AND conversation_id = ? AND status = 'PROCESSING'",
                (now, channel, conversation_id),
            )
        return self.conversation_job_status(channel, conversation_id)

    def conversation_job_cancel_requested(self, job_id: int) -> bool:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT status FROM conversation_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return bool(row and str(row["status"]) in {"CANCEL_REQUESTED", "CANCELLED"})

    def conversation_job_status(
        self, channel: str, conversation_id: str
    ) -> str | None:
        job = self.conversation_job_summary(channel, conversation_id)
        return str(job["status"]) if job else None

    def conversation_job_summary(
        self, channel: str, conversation_id: str
    ) -> dict[str, Any] | None:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM conversation_jobs WHERE channel = ? "
                "AND conversation_id = ? ORDER BY job_id DESC LIMIT 1",
                (channel, conversation_id),
            ).fetchone()
        return self._conversation_job_row(row) if row else None

    def requeue_conversation_job(
        self, channel: str, conversation_id: str
    ) -> dict[str, Any] | None:
        """Queue one interrupted job only when its completed response is durable."""
        with self.transaction():
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT * FROM conversation_jobs WHERE channel = ? "
                    "AND conversation_id = ? ORDER BY job_id DESC LIMIT 1",
                    (channel, conversation_id),
                ).fetchone()
                if row is None:
                    return None
                job = self._conversation_job_row(row)
                if job["status"] != "NEEDS_ATTENTION":
                    return job
                cached = self.conversation_responses(
                    channel, conversation_id, job["external_message_id"]
                )
                if not cached:
                    return job
                now = utc_now()
                cursor = connection.execute(
                    "UPDATE conversation_jobs SET status = 'QUEUED', replay_cached = 1, "
                    "lease_owner = '', lease_until = '', updated_at = ? "
                    "WHERE job_id = ? AND status = 'NEEDS_ATTENTION'",
                    (now, job["job_id"]),
                )
                if cursor.rowcount != 1:
                    raise StoreError(f"conversation job is not resumable: {job['job_id']}")
                return {
                    **job,
                    "status": "QUEUED",
                    "replay_cached": True,
                    "lease_owner": "",
                    "lease_until": "",
                    "updated_at": now,
                }

    def conversation_job(self, job_id: int) -> dict[str, Any] | None:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM conversation_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return self._conversation_job_row(row) if row else None

    def recover_stale_conversation_jobs(self) -> list[dict[str, Any]]:
        now = utc_now()
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM conversation_jobs WHERE status IN "
                "('PROCESSING', 'CANCEL_REQUESTED') AND lease_until <> '' "
                "AND lease_until <= ?",
                (now,),
            ).fetchall()
            if rows:
                connection.execute(
                    "UPDATE conversation_jobs SET status = 'NEEDS_ATTENTION', "
                    "last_error = 'worker stopped while processing conversation', "
                    "lease_owner = '', lease_until = '', updated_at = ? "
                    "WHERE status IN ('PROCESSING', 'CANCEL_REQUESTED') "
                    "AND lease_until <> '' AND lease_until <= ?",
                    (now, now),
                )
        return [self._conversation_job_row(row) for row in rows]

    def claim_inbound(
        self,
        channel: str,
        external_message_id: str,
        *,
        max_attempts: int = 2,
    ) -> bool:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        now = utc_now()
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT status, attempts FROM inbound_receipts "
                "WHERE channel = ? AND external_message_id = ?",
                (channel, external_message_id),
            ).fetchone()
            if row is None:
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO inbound_receipts(channel, external_message_id, status, "
                    "attempts, last_error, first_seen_at, updated_at) "
                    "VALUES (?, ?, 'PROCESSING', 1, '', ?, ?)",
                    (channel, external_message_id, now, now),
                )
                return cursor.rowcount == 1
            attempts = int(row["attempts"])
            if str(row["status"]) != "FAILED" or attempts >= max_attempts:
                return False
            cursor = connection.execute(
                "UPDATE inbound_receipts SET status = 'PROCESSING', attempts = ?, "
                "last_error = '', updated_at = ? "
                "WHERE channel = ? AND external_message_id = ? "
                "AND status = 'FAILED' AND attempts = ?",
                (attempts + 1, now, channel, external_message_id, attempts),
            )
            return cursor.rowcount == 1

    def complete_inbound(self, channel: str, external_message_id: str) -> None:
        self._finish_inbound(channel, external_message_id, "COMPLETED", "")

    def fail_inbound(
        self, channel: str, external_message_id: str, error: str
    ) -> None:
        self._finish_inbound(channel, external_message_id, "FAILED", error)

    def _finish_inbound(
        self, channel: str, external_message_id: str, status: str, error: str
    ) -> None:
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "UPDATE inbound_receipts SET status = ?, last_error = ?, updated_at = ? "
                "WHERE channel = ? AND external_message_id = ? AND status = 'PROCESSING'",
                (status, error, utc_now(), channel, external_message_id),
            )
            if cursor.rowcount != 1:
                raise StoreError(
                    f"inbound message is not processing: {channel}/{external_message_id}"
                )

    def recover_processing_inbound(self, channel: str) -> int:
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "UPDATE inbound_receipts SET status = 'FAILED', "
                "last_error = 'gateway restarted while processing', updated_at = ? "
                "WHERE channel = ? AND status = 'PROCESSING'",
                (utc_now(), channel),
            )
            return int(cursor.rowcount)

    def inbound_receipt(
        self, channel: str, external_message_id: str
    ) -> dict[str, Any] | None:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT status, attempts, last_error, first_seen_at, updated_at "
                "FROM inbound_receipts WHERE channel = ? AND external_message_id = ?",
                (channel, external_message_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "status": str(row["status"]),
            "attempts": int(row["attempts"]),
            "last_error": str(row["last_error"]),
            "first_seen_at": str(row["first_seen_at"]),
            "updated_at": str(row["updated_at"]),
        }

    def load_gateway_cursor(self, channel: str) -> str | None:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT cursor FROM gateway_cursors WHERE channel = ?", (channel,)
            ).fetchone()
        return str(row["cursor"]) if row else None

    def save_gateway_cursor(self, channel: str, cursor: str) -> None:
        with self._lock, self._connection() as connection:
            connection.execute(
                "INSERT INTO gateway_cursors(channel, cursor, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(channel) DO UPDATE SET cursor = excluded.cursor, "
                "updated_at = excluded.updated_at",
                (channel, cursor, utc_now()),
            )

    def queue_outbound(
        self,
        channel: str,
        conversation_id: str,
        text: str,
        reply_to: str = "",
        *,
        delivery_mode: str = "send",
        target_outbound_id: int = 0,
        coalesce_key: str = "",
    ) -> int:
        if delivery_mode not in {"send", "edit"}:
            raise ValueError("unsupported outbound delivery mode")
        if delivery_mode == "edit" and target_outbound_id < 1:
            raise ValueError("edit outbound requires a target outbound id")
        if delivery_mode == "send" and target_outbound_id != 0:
            raise ValueError("send outbound cannot have a target outbound id")
        coalesce_key = coalesce_key.strip()
        if coalesce_key and delivery_mode != "edit":
            raise ValueError("only edit outbound can be coalesced")
        if len(coalesce_key) > 300:
            raise ValueError("outbound coalesce key is too long")
        now = utc_now()
        with self._lock, self._connection() as connection:
            if coalesce_key:
                current = connection.execute(
                    "SELECT outbound_id, status FROM outbound_messages "
                    "WHERE coalesce_key = ? ORDER BY outbound_id DESC LIMIT 1",
                    (coalesce_key,),
                ).fetchone()
                if current is not None and str(current["status"]) != "PROCESSING":
                    outbound_id = int(current["outbound_id"])
                    connection.execute(
                        "UPDATE outbound_messages SET text = ?, reply_to = ?, "
                        "status = 'PENDING', attempts = 0, last_error = '', "
                        "lease_owner = '', lease_until = '', updated_at = ? "
                        "WHERE outbound_id = ?",
                        (text, reply_to, now, outbound_id),
                    )
                    return outbound_id
            cursor = connection.execute(
                "INSERT INTO outbound_messages(channel, conversation_id, text, reply_to, "
                "delivery_mode, target_outbound_id, coalesce_key, status, attempts, external_ids_json, "
                "last_error, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', 0, '[]', '', ?, ?)",
                (
                    channel,
                    conversation_id,
                    text,
                    reply_to,
                    delivery_mode,
                    target_outbound_id,
                    coalesce_key,
                    now,
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def progress_outbound_for_reply(
        self,
        channel: str,
        conversation_id: str,
        reply_to: str,
    ) -> int | None:
        """Find the latest editable progress card for one inbound message."""
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT outbound_id FROM outbound_messages "
                "WHERE channel = ? AND conversation_id = ? AND reply_to = ? "
                "AND delivery_mode = 'send' AND text LIKE ? "
                "ORDER BY outbound_id DESC LIMIT 1",
                (channel, conversation_id, reply_to, "% · 진행 중]%"),
            ).fetchone()
        return int(row["outbound_id"]) if row is not None else None

    def complete_inbound_with_outbound(
        self,
        channel: str,
        external_message_id: str,
        messages: list[dict[str, str]],
    ) -> list[int]:
        """한 트랜잭션에서 수신 완료와 발신 보관을 함께 확정한다."""
        now = utc_now()
        identifiers: list[int] = []
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "UPDATE inbound_receipts SET status = 'COMPLETED', last_error = '', "
                "updated_at = ? WHERE channel = ? AND external_message_id = ? "
                "AND status = 'PROCESSING'",
                (now, channel, external_message_id),
            )
            if cursor.rowcount != 1:
                raise StoreError(
                    f"inbound message is not processing: {channel}/{external_message_id}"
                )
            for message in messages:
                queued = connection.execute(
                    "INSERT INTO outbound_messages(channel, conversation_id, text, "
                    "reply_to, status, attempts, external_ids_json, last_error, "
                    "created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, 'PENDING', 0, '[]', '', ?, ?)",
                    (
                        message["channel"],
                        message["conversation_id"],
                        message["text"],
                        message.get("reply_to", ""),
                        now,
                        now,
                    ),
                )
                identifiers.append(int(queued.lastrowid))
        return identifiers

    def deliverable_outbound(
        self, channel: str, *, max_attempts: int = 2, limit: int = 100
    ) -> list[dict[str, Any]]:
        if max_attempts < 1 or limit < 1:
            raise ValueError("outbound limits must be positive")
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT outbound_id, channel, conversation_id, text, reply_to, "
                "delivery_mode, target_outbound_id, status, attempts "
                "FROM outbound_messages WHERE channel = ? "
                "AND status IN ('PENDING', 'FAILED') AND attempts < ? "
                "ORDER BY outbound_id LIMIT ?",
                (channel, max_attempts, limit),
            ).fetchall()
        return [
            {
                "outbound_id": int(row["outbound_id"]),
                "channel": str(row["channel"]),
                "conversation_id": str(row["conversation_id"]),
                "text": str(row["text"]),
                "reply_to": str(row["reply_to"]),
                "delivery_mode": str(row["delivery_mode"]),
                "target_outbound_id": int(row["target_outbound_id"]),
                "status": str(row["status"]),
                "attempts": int(row["attempts"]),
            }
            for row in rows
        ]

    def complete_outbound(
        self,
        outbound_id: int,
        external_ids: tuple[str, ...],
        *,
        lease_owner: str = "",
    ) -> None:
        with self._lock, self._connection() as connection:
            owner_clause = " AND lease_owner = ?" if lease_owner else ""
            parameters: list[Any] = [
                json.dumps(list(external_ids), ensure_ascii=False),
                utc_now(),
                outbound_id,
            ]
            if lease_owner:
                parameters.append(lease_owner)
            cursor = connection.execute(
                "UPDATE outbound_messages SET status = 'SENT', attempts = attempts + 1, "
                "external_ids_json = ?, last_error = '', lease_owner = '', lease_until = '', "
                "updated_at = ? WHERE outbound_id = ? "
                + ("AND status = 'PROCESSING'" if lease_owner else "AND status IN ('PENDING', 'FAILED', 'PROCESSING')")
                + owner_clause,
                parameters,
            )
            if cursor.rowcount != 1:
                raise StoreError(f"outbound message is not deliverable: {outbound_id}")
            connection.execute(
                "UPDATE repository_analysis_jobs SET delivery_status = 'DELIVERED', updated_at = ? "
                "WHERE final_outbound_id = ?",
                (utc_now(), outbound_id),
            )

    def fail_outbound(
        self,
        outbound_id: int,
        error: str,
        *,
        lease_owner: str = "",
        max_attempts: int = 2,
    ) -> None:
        with self._lock, self._connection() as connection:
            owner_clause = " AND lease_owner = ?" if lease_owner else ""
            parameters: list[Any] = [max_attempts, error, utc_now(), outbound_id]
            if lease_owner:
                parameters.append(lease_owner)
            cursor = connection.execute(
                "UPDATE outbound_messages SET status = CASE WHEN attempts + 1 >= ? "
                "THEN 'DEAD' ELSE 'FAILED' END, attempts = attempts + 1, "
                "last_error = ?, lease_owner = '', lease_until = '', updated_at = ? "
                "WHERE outbound_id = ? "
                + ("AND status = 'PROCESSING'" if lease_owner else "AND status IN ('PENDING', 'FAILED', 'PROCESSING')")
                + owner_clause,
                parameters,
            )
            if cursor.rowcount != 1:
                raise StoreError(f"outbound message is not deliverable: {outbound_id}")
            connection.execute(
                "UPDATE repository_analysis_jobs SET delivery_status = CASE "
                "WHEN (SELECT status FROM outbound_messages WHERE outbound_id = ?) = 'DEAD' "
                "THEN 'FAILED' ELSE 'RETRYING' END, updated_at = ? "
                "WHERE final_outbound_id = ?",
                (outbound_id, utc_now(), outbound_id),
            )

    def mark_outbound_uncertain(
        self, outbound_id: int, error: str, external_ids: tuple[str, ...] = (),
        *, lease_owner: str,
    ) -> None:
        now = utc_now()
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "UPDATE outbound_messages SET status = 'NEEDS_ATTENTION', attempts = attempts + 1, "
                "external_ids_json = ?, last_error = ?, lease_owner = '', lease_until = '', "
                "updated_at = ? WHERE outbound_id = ? AND status = 'PROCESSING' AND lease_owner = ?",
                (json.dumps(list(external_ids)), error, now, outbound_id, lease_owner),
            )
            if cursor.rowcount != 1:
                raise StoreError(f"outbound message is not owned: {outbound_id}")
            connection.execute(
                "UPDATE repository_analysis_jobs SET delivery_status = 'UNKNOWN', updated_at = ? "
                "WHERE final_outbound_id = ?",
                (now, outbound_id),
            )

    def claim_next_outbound(
        self,
        channel: str,
        lease_owner: str,
        *,
        max_attempts: int = 2,
        lease_seconds: int = 60,
    ) -> dict[str, Any] | None:
        """채널별 맨 앞 발신 한 건에 임대를 잡아 다중 프로세스 중복 전송을 막는다."""
        if not lease_owner.strip():
            raise ValueError("lease_owner is required")
        if max_attempts < 1 or lease_seconds < 1:
            raise ValueError("outbound claim limits must be positive")
        now = datetime.now(timezone.utc)
        now_text = now.isoformat()
        lease_until = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self.transaction():
            with self._connection() as connection:
                connection.execute(
                    "UPDATE outbound_messages SET status = 'NEEDS_ATTENTION', lease_owner = '', "
                    "lease_until = '', last_error = 'delivery outcome unknown after gateway stopped', updated_at = ? "
                    "WHERE channel = ? AND status = 'PROCESSING' "
                    "AND lease_until <> '' AND lease_until <= ?",
                    (now_text, channel, now_text),
                )
                connection.execute(
                    "UPDATE repository_analysis_jobs SET delivery_status = 'UNKNOWN', updated_at = ? "
                    "WHERE final_outbound_id IN (SELECT outbound_id FROM outbound_messages "
                    "WHERE channel = ? AND status = 'NEEDS_ATTENTION') "
                    "AND delivery_status NOT IN ('DELIVERED', 'UNKNOWN')",
                    (now_text, channel),
                )
                connection.execute(
                    "UPDATE outbound_messages SET status = 'DEAD', updated_at = ? "
                    "WHERE channel = ? AND status = 'FAILED' AND attempts >= ?",
                    (now_text, channel, max_attempts),
                )
                row = connection.execute(
                    "SELECT outbound_id, channel, conversation_id, text, reply_to, "
                    "delivery_mode, target_outbound_id, status, attempts "
                    "FROM outbound_messages WHERE channel = ? "
                    "AND status NOT IN ('SENT', 'DEAD', 'NEEDS_ATTENTION') "
                    "ORDER BY outbound_id LIMIT 1",
                    (channel,),
                ).fetchone()
                if row is None or str(row["status"]) == "PROCESSING":
                    return None
                cursor = connection.execute(
                    "UPDATE outbound_messages SET status = 'PROCESSING', lease_owner = ?, "
                    "lease_until = ?, updated_at = ? WHERE outbound_id = ? "
                    "AND status IN ('PENDING', 'FAILED') AND attempts < ?",
                    (
                        lease_owner,
                        lease_until,
                        now_text,
                        int(row["outbound_id"]),
                        max_attempts,
                    ),
                )
                if cursor.rowcount != 1:
                    return None
                return {
                    "outbound_id": int(row["outbound_id"]),
                    "channel": str(row["channel"]),
                    "conversation_id": str(row["conversation_id"]),
                    "text": str(row["text"]),
                    "reply_to": str(row["reply_to"]),
                    "delivery_mode": str(row["delivery_mode"]),
                    "target_outbound_id": int(row["target_outbound_id"]),
                    "status": "PROCESSING",
                    "attempts": int(row["attempts"]),
                }

    def outbound_record(self, outbound_id: int) -> dict[str, Any] | None:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT status, attempts, external_ids_json, last_error, "
                "delivery_mode, target_outbound_id, "
                "lease_owner, lease_until "
                "FROM outbound_messages WHERE outbound_id = ?",
                (outbound_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "status": str(row["status"]),
            "attempts": int(row["attempts"]),
            "external_ids": json.loads(str(row["external_ids_json"])),
            "delivery_mode": str(row["delivery_mode"]),
            "target_outbound_id": int(row["target_outbound_id"]),
            "last_error": str(row["last_error"]),
            "lease_owner": str(row["lease_owner"]),
            "lease_until": str(row["lease_until"]),
        }

    def enqueue_pipeline_job(
        self, run_id: str, channel: str, conversation_id: str
    ) -> dict[str, Any]:
        now = utc_now()
        with self._lock, self._connection() as connection:
            try:
                connection.execute(
                    "INSERT INTO pipeline_jobs(run_id, channel, conversation_id, status, "
                    "attempts, lease_owner, lease_until, last_error, created_at, updated_at) "
                    "VALUES (?, ?, ?, 'QUEUED', 0, '', '', '', ?, ?)",
                    (run_id, channel, conversation_id, now, now),
                )
            except sqlite3.IntegrityError:
                existing = self.pipeline_job(run_id)
                if existing is None:
                    raise
                return existing
        queued = self.pipeline_job(run_id)
        if queued is None:
            raise StoreError(f"pipeline job was not saved: {run_id}")
        return queued

    def requeue_pipeline_job(self, run_id: str) -> str | None:
        """Return an attention-paused pipeline job to the durable queue."""
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT status FROM pipeline_jobs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                return None
            status = str(row["status"])
            if status == "NEEDS_ATTENTION":
                connection.execute(
                    "UPDATE pipeline_jobs SET status = 'QUEUED', last_error = '', "
                    "lease_owner = '', lease_until = '', updated_at = ? "
                    "WHERE run_id = ? AND status = 'NEEDS_ATTENTION'",
                    (utc_now(), run_id),
                )
                return "QUEUED"
            return status

    def pipeline_job(self, run_id: str) -> dict[str, Any] | None:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT run_id, channel, conversation_id, status, attempts, lease_owner, "
                "lease_until, last_error, created_at, updated_at "
                "FROM pipeline_jobs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return self._pipeline_row(row) if row else None

    def claim_next_pipeline_job(
        self, lease_owner: str, *, lease_seconds: int = 120
    ) -> dict[str, Any] | None:
        if not lease_owner.strip() or lease_seconds < 1:
            raise ValueError("pipeline lease owner and positive lease are required")
        now = datetime.now(timezone.utc)
        now_text = now.isoformat()
        lease_until = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self.transaction():
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT run_id FROM pipeline_jobs WHERE status = 'QUEUED' "
                    "ORDER BY created_at LIMIT 1"
                ).fetchone()
                if row is None:
                    return None
                run_id = str(row["run_id"])
                cursor = connection.execute(
                    "UPDATE pipeline_jobs SET status = 'RUNNING', attempts = attempts + 1, "
                    "lease_owner = ?, lease_until = ?, updated_at = ? "
                    "WHERE run_id = ? AND status = 'QUEUED'",
                    (lease_owner, lease_until, now_text, run_id),
                )
                if cursor.rowcount != 1:
                    return None
        return self.pipeline_job(run_id)

    def heartbeat_pipeline_job(
        self, run_id: str, lease_owner: str, *, lease_seconds: int = 120
    ) -> None:
        now = datetime.now(timezone.utc)
        lease_until = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "UPDATE pipeline_jobs SET lease_until = ?, updated_at = ? "
                "WHERE run_id = ? AND lease_owner = ? "
                "AND status IN ('RUNNING', 'CANCEL_REQUESTED', 'PAUSE_REQUESTED')",
                (lease_until, now.isoformat(), run_id, lease_owner),
            )
            if cursor.rowcount != 1:
                raise StoreError(f"pipeline job lease was lost: {run_id}")

    def finish_pipeline_job(
        self,
        run_id: str,
        lease_owner: str,
        status: str,
        error: str = "",
    ) -> None:
        if status not in {"COMPLETED", "FAILED", "CANCELLED", "NEEDS_ATTENTION"}:
            raise ValueError(f"unsupported pipeline terminal status: {status}")
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "UPDATE pipeline_jobs SET status = ?, last_error = ?, lease_owner = '', "
                "lease_until = '', updated_at = ? WHERE run_id = ? AND lease_owner = ? "
                "AND status IN ('RUNNING', 'CANCEL_REQUESTED', 'PAUSE_REQUESTED')",
                (status, error, utc_now(), run_id, lease_owner),
            )
            if cursor.rowcount != 1:
                raise StoreError(f"pipeline job is not owned by this worker: {run_id}")

    def request_pipeline_cancel(self, run_id: str) -> str | None:
        now = utc_now()
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT status FROM pipeline_jobs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                return None
            status = str(row["status"])
            if status == "QUEUED":
                connection.execute(
                    "UPDATE pipeline_jobs SET status = 'CANCELLED', updated_at = ? "
                    "WHERE run_id = ? AND status = 'QUEUED'",
                    (now, run_id),
                )
                return "CANCELLED"
            if status == "RUNNING":
                connection.execute(
                    "UPDATE pipeline_jobs SET status = 'CANCEL_REQUESTED', updated_at = ? "
                    "WHERE run_id = ? AND status = 'RUNNING'",
                    (now, run_id),
                )
                return "CANCEL_REQUESTED"
            if status == "NEEDS_ATTENTION":
                connection.execute(
                    "UPDATE pipeline_jobs SET status = 'CANCELLED', updated_at = ? "
                    "WHERE run_id = ? AND status = 'NEEDS_ATTENTION'",
                    (now, run_id),
                )
                return "CANCELLED"
            return status

    def request_pipeline_pause(self, run_id: str) -> str | None:
        """Ask the active worker to stop at its next cooperative checkpoint."""
        now = utc_now()
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT status FROM pipeline_jobs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                return None
            status = str(row["status"])
            if status == "RUNNING":
                connection.execute(
                    "UPDATE pipeline_jobs SET status = 'PAUSE_REQUESTED', updated_at = ? "
                    "WHERE run_id = ? AND status = 'RUNNING'",
                    (now, run_id),
                )
                return "PAUSE_REQUESTED"
            if status == "QUEUED":
                connection.execute(
                    "UPDATE pipeline_jobs SET status = 'NEEDS_ATTENTION', updated_at = ? "
                    "WHERE run_id = ? AND status = 'QUEUED'",
                    (now, run_id),
                )
                return "NEEDS_ATTENTION"
            return status

    def pipeline_cancel_requested(self, run_id: str) -> bool:
        job = self.pipeline_job(run_id)
        return bool(job and job["status"] in {"CANCEL_REQUESTED", "CANCELLED"})

    def pipeline_pause_requested(self, run_id: str) -> bool:
        job = self.pipeline_job(run_id)
        return bool(job and job["status"] == "PAUSE_REQUESTED")

    def recover_stale_pipeline_jobs(self) -> list[dict[str, Any]]:
        now = utc_now()
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT run_id, channel, conversation_id, status, attempts, lease_owner, "
                "lease_until, last_error, created_at, updated_at FROM pipeline_jobs "
                "WHERE status IN ('RUNNING', 'CANCEL_REQUESTED', 'PAUSE_REQUESTED') "
                "AND lease_until <> '' AND lease_until <= ?",
                (now,),
            ).fetchall()
            if rows:
                connection.execute(
                    "UPDATE pipeline_jobs SET status = 'NEEDS_ATTENTION', "
                    "last_error = 'worker stopped during repository mutation', "
                    "lease_owner = '', lease_until = '', updated_at = ? "
                    "WHERE status IN ('RUNNING', 'CANCEL_REQUESTED', 'PAUSE_REQUESTED') "
                    "AND lease_until <> '' AND lease_until <= ?",
                    (now, now),
                )
        return [self._pipeline_row(row) for row in rows]

    def open_execution_question(
        self,
        run_id: str,
        stage_id: str,
        role_id: str,
        questions: tuple[str, ...],
    ) -> dict[str, Any]:
        cleaned = tuple(item.strip() for item in questions if item.strip())
        if not run_id.strip() or not stage_id.strip() or not role_id.strip() or not cleaned:
            raise ValueError("execution question requires run, stage, role, and question text")
        if any(len(item) > 2000 for item in cleaned):
            raise ValueError("execution question is too long")
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT question_id, run_id, stage_id, role_id, questions_json, status, "
                "answer, asked_at, answered_at FROM execution_questions "
                "WHERE run_id = ? AND status = 'PENDING' ORDER BY question_id DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            if row is None:
                now = utc_now()
                cursor = connection.execute(
                    "INSERT INTO execution_questions(run_id, stage_id, role_id, questions_json, "
                    "status, answer, asked_at, answered_at) VALUES (?, ?, ?, ?, 'PENDING', '', ?, '')",
                    (run_id, stage_id, role_id, json.dumps(cleaned, ensure_ascii=False), now),
                )
                row = connection.execute(
                    "SELECT question_id, run_id, stage_id, role_id, questions_json, status, "
                    "answer, asked_at, answered_at FROM execution_questions WHERE question_id = ?",
                    (int(cursor.lastrowid),),
                ).fetchone()
        if row is None:
            raise StoreError("execution question was not saved")
        return self._execution_question_row(row)

    def open_execution_question_for_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT question_id, run_id, stage_id, role_id, questions_json, status, "
                "answer, asked_at, answered_at FROM execution_questions "
                "WHERE run_id = ? AND status = 'PENDING' ORDER BY question_id DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        return self._execution_question_row(row) if row else None

    def answer_execution_question(self, run_id: str, answer: str) -> dict[str, Any] | None:
        cleaned = answer.strip()
        if not cleaned:
            raise ValueError("execution question answer cannot be blank")
        if len(cleaned) > 12000:
            raise ValueError("execution question answer is too long")
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT question_id FROM execution_questions WHERE run_id = ? "
                "AND status = 'PENDING' ORDER BY question_id DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            question_id = int(row["question_id"])
            connection.execute(
                "UPDATE execution_questions SET status = 'ANSWERED', answer = ?, answered_at = ? "
                "WHERE question_id = ? AND status = 'PENDING'",
                (cleaned, utc_now(), question_id),
            )
            answered = connection.execute(
                "SELECT question_id, run_id, stage_id, role_id, questions_json, status, "
                "answer, asked_at, answered_at FROM execution_questions WHERE question_id = ?",
                (question_id,),
            ).fetchone()
        return self._execution_question_row(answered) if answered else None

    def answered_execution_questions(self, run_id: str, stage_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT question_id, run_id, stage_id, role_id, questions_json, status, "
                "answer, asked_at, answered_at FROM execution_questions "
                "WHERE run_id = ? AND stage_id = ? AND status = 'ANSWERED' "
                "ORDER BY question_id",
                (run_id, stage_id),
            ).fetchall()
        return [self._execution_question_row(row) for row in rows]

    @staticmethod
    def _execution_question_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "question_id": int(row["question_id"]),
            "run_id": str(row["run_id"]),
            "stage_id": str(row["stage_id"]),
            "role_id": str(row["role_id"]),
            "questions": tuple(str(item) for item in json.loads(str(row["questions_json"]))),
            "status": str(row["status"]),
            "answer": str(row["answer"]),
            "asked_at": str(row["asked_at"]),
            "answered_at": str(row["answered_at"]),
        }

    def acquire_repository_lock(
        self,
        repository_path: str,
        run_id: str,
        lease_owner: str,
        *,
        lease_seconds: int = 120,
    ) -> bool:
        now = datetime.now(timezone.utc)
        now_text = now.isoformat()
        lease_until = (now + timedelta(seconds=lease_seconds)).isoformat()
        path_hash = self._repository_hash(repository_path)
        normalized = str(Path(repository_path).resolve())
        with self.transaction():
            with self._connection() as connection:
                connection.execute(
                    "DELETE FROM repository_execution_locks WHERE lease_until <= ?",
                    (now_text,),
                )
                try:
                    connection.execute(
                        "INSERT INTO repository_execution_locks(path_hash, repository_path, "
                        "run_id, lease_owner, lease_until, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            path_hash,
                            normalized,
                            run_id,
                            lease_owner,
                            lease_until,
                            now_text,
                            now_text,
                        ),
                    )
                    return True
                except sqlite3.IntegrityError:
                    return False

    def renew_repository_lock(
        self,
        repository_path: str,
        run_id: str,
        lease_owner: str,
        *,
        lease_seconds: int = 120,
    ) -> None:
        now = datetime.now(timezone.utc)
        lease_until = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "UPDATE repository_execution_locks SET lease_until = ?, updated_at = ? "
                "WHERE path_hash = ? AND run_id = ? AND lease_owner = ?",
                (
                    lease_until,
                    now.isoformat(),
                    self._repository_hash(repository_path),
                    run_id,
                    lease_owner,
                ),
            )
            if cursor.rowcount != 1:
                raise StoreError(f"repository execution lock was lost: {repository_path}")

    def release_repository_lock(
        self, repository_path: str, run_id: str, lease_owner: str
    ) -> None:
        with self._lock, self._connection() as connection:
            connection.execute(
                "DELETE FROM repository_execution_locks WHERE path_hash = ? "
                "AND run_id = ? AND lease_owner = ?",
                (self._repository_hash(repository_path), run_id, lease_owner),
            )

    @staticmethod
    def _repository_analysis_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "analysis_id": str(row["analysis_id"]),
            "channel": str(row["channel"]),
            "conversation_id": str(row["conversation_id"]),
            "user_id": str(row["user_id"]),
            "source_message_id": str(row["source_message_id"]),
            "role_id": str(row["role_id"]),
            "request_text": str(row["request_text"]),
            "repository_path": str(row["repository_path"]),
            "repository_identity": str(row["repository_identity"]),
            "commit_sha": str(row["commit_sha"]),
            "branch": str(row["branch"]),
            "status": str(row["status"]),
            "phase": str(row["phase"]),
            "stop_reason": str(row["stop_reason"]),
            "plan": json.loads(str(row["plan_json"])),
            "completed": json.loads(str(row["completed_json"])),
            "remaining": json.loads(str(row["remaining_json"])),
            "checkpoint": int(row["checkpoint"]),
            "model_call_state": str(row["model_call_state"]),
            "query_rounds": int(row["query_rounds"]),
            "model_calls": int(row["model_calls"]),
            "read_bytes": int(row["read_bytes"]),
            "no_progress_count": int(row["no_progress_count"]),
            "progress_outbound_id": int(row["progress_outbound_id"]),
            "last_progress_at": str(row["last_progress_at"]),
            "final_response": str(row["final_response"]),
            "final_outbound_id": int(row["final_outbound_id"]),
            "delivery_status": str(row["delivery_status"]),
            "active_seconds": float(row["active_seconds"]),
            "active_since": str(row["active_since"]),
            "lease_owner": str(row["lease_owner"]),
            "lease_until": str(row["lease_until"]),
            "attempts": int(row["attempts"]),
            "last_error": str(row["last_error"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    @staticmethod
    def _repository_analysis_batch_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "analysis_id": str(row["analysis_id"]),
            "batch_index": int(row["batch_index"]),
            "phase": str(row["phase"]),
            "target": json.loads(str(row["target_json"])),
            "status": str(row["status"]),
            "created_at": str(row["created_at"]),
            "completed_at": str(row["completed_at"]),
        }

    @staticmethod
    def _conversation_job_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "job_id": int(row["job_id"]),
            "channel": str(row["channel"]),
            "conversation_id": str(row["conversation_id"]),
            "user_id": str(row["user_id"]),
            "external_message_id": str(row["external_message_id"]),
            "message": json.loads(str(row["message_json"])),
            "status": str(row["status"]),
            "attempts": int(row["attempts"]),
            "lease_owner": str(row["lease_owner"]),
            "lease_until": str(row["lease_until"]),
            "last_error": str(row["last_error"]),
            "replay_cached": bool(row["replay_cached"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    @staticmethod
    def _pipeline_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "run_id": str(row["run_id"]),
            "channel": str(row["channel"]),
            "conversation_id": str(row["conversation_id"]),
            "status": str(row["status"]),
            "attempts": int(row["attempts"]),
            "lease_owner": str(row["lease_owner"]),
            "lease_until": str(row["lease_until"]),
            "last_error": str(row["last_error"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }
