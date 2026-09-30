import unittest
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.contracts import RunState
from app.services.context import ContextPolicy, ContextService
from app.storage import StateStore
from tests.foundation.support import temporary_directory


class ContextTests(unittest.TestCase):
    def test_distinct_facts_survive_and_replacement_keeps_history(self):
        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            context = ContextService(store)
            context.save_memory("user", "person", "한국어 선호", source_ref="m1")
            context.save_memory("user", "person", "짧은 답변 선호", source_ref="m1")
            facts = context.user_memories("person")
            self.assertEqual({"한국어 선호", "짧은 답변 선호"},
                             {item["content"] for item in facts})
            old_id = next(item["fact_id"] for item in facts
                          if item["content"] == "한국어 선호")
            new_id = context.replace_memory(old_id, "user", "person", "영어 선호")
            self.assertEqual({"영어 선호", "짧은 답변 선호"},
                             {item["content"] for item in context.user_memories("person")})
            connection = sqlite3.connect(store.path)
            try:
                history = connection.execute(
                    "SELECT supersedes_id, source_kind FROM memory_facts WHERE fact_id = ?",
                    (new_id,),
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual((old_id, "user"), history)

    def test_legacy_memory_migrates_without_loss(self):
        with temporary_directory() as directory:
            path = Path(directory) / "state.db"
            StateStore(path)
            timestamp = datetime.now(timezone.utc).isoformat()
            connection = sqlite3.connect(path)
            try:
                connection.execute("DELETE FROM schema_migrations WHERE version = 8")
                connection.execute("DROP TABLE memory_facts")
                connection.execute(
                    "INSERT INTO conversation_memories VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ("user", "legacy-user", "shared", "기존 사실", 4, timestamp, timestamp),
                )
                connection.commit()
            finally:
                connection.close()
            migrated = StateStore(path)
            facts = migrated.list_memories((("user", "legacy-user"),))
            self.assertEqual(["기존 사실"], [item["content"] for item in facts])
            self.assertEqual(4, facts[0]["revision"])
            self.assertEqual("legacy", facts[0]["source_kind"])
            self.assertEqual(["기존 사실"], [item["content"] for item in
                StateStore(path).list_memories((("user", "legacy-user"),))])

    def test_expired_fact_is_excluded_from_context_before_purge(self):
        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            store.create_run(RunState(run_id="RUN-EXPIRED-MEMORY"))
            context = ContextService(store)
            context.save_memory("user", "person", "만료된 사실")
            connection = sqlite3.connect(store.path)
            try:
                connection.execute(
                    "UPDATE memory_facts SET expires_at = ? WHERE scope = 'user'",
                    ((datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),),
                )
                connection.commit()
            finally:
                connection.close()
            bundle = context.build("RUN-EXPIRED-MEMORY", user_id="person")
            self.assertEqual((), bundle.memories)

    def test_context_keeps_decisions_and_only_recent_messages(self):
        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            store.create_run(RunState(run_id="RUN-CONTEXT"))
            context = ContextService(store)
            context.add_decision("RUN-CONTEXT", "세션 인증을 사용한다")
            for index in range(6):
                context.add_message("RUN-CONTEXT", "user", f"message-{index}")
            bundle = context.build("RUN-CONTEXT", recent_limit=2, max_characters=1000)
            self.assertEqual(1, len(bundle.decisions))
            self.assertEqual(2, len(bundle.recent_messages))
            self.assertEqual("message-4", bundle.recent_messages[0]["content"])
            self.assertTrue(bundle.truncated)

    def test_conversation_secrets_are_redacted_before_storage(self):
        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            store.create_run(RunState(run_id="RUN-SECRET"))
            context = ContextService(store)
            context.add_message("RUN-SECRET", "user", "password=my-password")
            saved = store.list_messages("RUN-SECRET")[0]["content"]
            self.assertEqual("password=[REDACTED]", saved)

    def test_decisions_also_obey_context_character_limit(self):
        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            store.create_run(RunState(run_id="RUN-CONTEXT-LIMIT"))
            context = ContextService(store)
            context.add_decision("RUN-CONTEXT-LIMIT", "a" * 70)
            context.add_decision("RUN-CONTEXT-LIMIT", "b" * 70)

            bundle = context.build(
                "RUN-CONTEXT-LIMIT", recent_limit=1, max_characters=100
            )

            self.assertLessEqual(bundle.characters, 100)
            self.assertEqual(1, len(bundle.decisions))
            self.assertEqual("b" * 70, bundle.decisions[0]["content"])
            self.assertTrue(bundle.truncated)

    def test_compacted_decision_memory_retains_older_confirmed_decisions(self):
        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            store.create_run(RunState(run_id="RUN-COMPACTION"))
            context = ContextService(
                store,
                policy=ContextPolicy(
                    recent_messages=1,
                    recent_decisions=1,
                    max_characters=300,
                    decision_summary_characters=200,
                ),
            )
            context.add_decision("RUN-COMPACTION", "첫 번째 확정 결정")
            context.add_decision("RUN-COMPACTION", "두 번째 확정 결정")

            bundle = context.build("RUN-COMPACTION")

            self.assertEqual(("두 번째 확정 결정",), tuple(
                item["content"] for item in bundle.decisions
            ))
            summary = next(item for item in bundle.memories if item["role_id"] == "compaction")
            self.assertIn("첫 번째 확정 결정", summary["content"])
            self.assertLessEqual(bundle.characters, 300)

    def test_project_memory_is_not_included_for_another_project(self):
        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            store.create_run(RunState(run_id="RUN-PROJECT-ISOLATION"))
            context = ContextService(store)
            context.save_memory("project", "D:\\project-a", "A의 규칙")
            context.save_memory("project", "D:\\project-b", "B의 규칙")

            bundle = context.build("RUN-PROJECT-ISOLATION", repository="D:\\project-b")

            contents = {item["content"] for item in bundle.memories}
            self.assertIn("B의 규칙", contents)
            self.assertNotIn("A의 규칙", contents)

    def test_project_context_does_not_import_legacy_path_after_identity_selection(self):
        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            store.create_run(RunState(run_id="RUN-PROJECT-ALIASES"))
            context = ContextService(store)
            context.save_memory("project", "identity", "새 사실")
            context.save_memory("project", "D:\\old-project", "이관된 사실")
            bundle = context.build(
                "RUN-PROJECT-ALIASES", repository="identity",
            )
            self.assertEqual({"새 사실"},
                             {item["content"] for item in bundle.memories})
            self.assertEqual({"새 사실", "이관된 사실"}, {
                item["content"] for item in context.project_memories(
                    "identity", "D:\\old-project"
                )
            })

    def test_repository_identity_excludes_prior_messages_and_run_memory(self):
        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            store.create_run(RunState(run_id="RUN-REPOSITORY-CONTEXT"))
            context = ContextService(store)
            alpha = "a" * 64
            beta = "b" * 64
            context.add_message(
                "RUN-REPOSITORY-CONTEXT",
                "user",
                "ALPHA_ONLY",
                data={"repository_identity": alpha},
            )
            context.add_message(
                "RUN-REPOSITORY-CONTEXT",
                "user",
                "BETA_ONLY",
                data={"repository_identity": beta},
            )
            context.save_memory("run", "RUN-REPOSITORY-CONTEXT", "ALPHA_MEMORY")
            context.save_memory("conversation", f"telegram:200:repository:{alpha}", "ALPHA_CONVERSATION_MEMORY")
            context.save_memory("conversation", f"telegram:200:repository:{beta}", "BETA_CONVERSATION_MEMORY")

            bundle = context.build(
                "RUN-REPOSITORY-CONTEXT",
                conversation_key=f"telegram:200:repository:{beta}",
                repository=beta,
                repository_identity=beta,
            )

            self.assertEqual(
                ("BETA_ONLY",),
                tuple(item["content"] for item in bundle.recent_messages),
            )
            contents = {item["content"] for item in bundle.memories}
            self.assertIn("BETA_CONVERSATION_MEMORY", contents)
            self.assertNotIn("ALPHA_CONVERSATION_MEMORY", contents)
            self.assertNotIn("ALPHA_MEMORY", contents)


if __name__ == "__main__":
    unittest.main()
