import unittest
from pathlib import Path

from app.contracts import RunState
from app.storage import StateStore
from tests.gateway.support import temporary_directory


class GatewayStorageTests(unittest.TestCase):
    def test_conversation_binding_and_cursor_survive_new_store_instance(self):
        with temporary_directory() as directory:
            database = Path(directory) / "state.db"
            first = StateStore(database)
            first.create_run(RunState(run_id="RUN-GATEWAY"))
            first.bind_conversation(
                "telegram", "200", "100", "RUN-GATEWAY", "development"
            )
            first.save_gateway_cursor("telegram", "44")

            restarted = StateStore(database)
            binding = restarted.load_conversation("telegram", "200")
            self.assertIsNotNone(binding)
            assert binding is not None
            self.assertEqual("RUN-GATEWAY", binding["run_id"])
            self.assertEqual("44", restarted.load_gateway_cursor("telegram"))

    def test_inbound_message_is_deduplicated_and_failed_message_retries_once(self):
        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            self.assertTrue(store.claim_inbound("telegram", "update-1"))
            store.complete_inbound("telegram", "update-1")
            self.assertFalse(store.claim_inbound("telegram", "update-1"))

            self.assertTrue(store.claim_inbound("telegram", "update-2"))
            store.fail_inbound("telegram", "update-2", "temporary token=secret")
            self.assertTrue(store.claim_inbound("telegram", "update-2"))
            store.fail_inbound("telegram", "update-2", "failed again")
            self.assertFalse(store.claim_inbound("telegram", "update-2"))
            receipt = store.inbound_receipt("telegram", "update-2")
            self.assertEqual(2, receipt["attempts"] if receipt else None)

    def test_restart_recovers_messages_left_processing(self):
        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            store.claim_inbound("telegram", "update-3")
            self.assertEqual(1, store.recover_processing_inbound("telegram"))
            receipt = store.inbound_receipt("telegram", "update-3")
            self.assertEqual("FAILED", receipt["status"] if receipt else None)
            self.assertTrue(store.claim_inbound("telegram", "update-3"))

    def test_inbound_completion_and_outbound_queue_are_persisted_together(self):
        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            store.claim_inbound("telegram", "update-4")
            identifiers = store.complete_inbound_with_outbound(
                "telegram",
                "update-4",
                [
                    {
                        "channel": "telegram",
                        "conversation_id": "200",
                        "text": "응답",
                        "reply_to": "update-4",
                    }
                ],
            )
            self.assertEqual("COMPLETED", store.inbound_receipt("telegram", "update-4")["status"])
            pending = store.deliverable_outbound("telegram")
            self.assertEqual(identifiers[0], pending[0]["outbound_id"])

            store.fail_outbound(identifiers[0], "network")
            self.assertEqual("FAILED", store.outbound_record(identifiers[0])["status"])
            self.assertEqual(1, store.outbound_record(identifiers[0])["attempts"])
            store.complete_outbound(identifiers[0], ("telegram-9",))
            self.assertEqual("SENT", store.outbound_record(identifiers[0])["status"])
            self.assertEqual(2, store.outbound_record(identifiers[0])["attempts"])

    def test_pending_progress_edits_are_coalesced_into_one_outbox_row(self):
        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            progress_id = store.queue_outbound(
                "telegram", "200", "[빌더 · 진행 중]\n진행: 1/4"
            )
            first_edit = store.queue_outbound(
                "telegram",
                "200",
                "[빌더 · 진행 중]\n진행: 2/4",
                delivery_mode="edit",
                target_outbound_id=progress_id,
                coalesce_key=f"progress:{progress_id}",
            )
            latest_edit = store.queue_outbound(
                "telegram",
                "200",
                "[빌더 · 진행 중]\n진행: 3/4",
                delivery_mode="edit",
                target_outbound_id=progress_id,
                coalesce_key=f"progress:{progress_id}",
            )

            self.assertEqual(first_edit, latest_edit)
            pending = store.deliverable_outbound("telegram")
            self.assertEqual(2, len(pending))
            self.assertEqual(
                "[빌더 · 진행 중]\n진행: 3/4",
                pending[-1]["text"],
            )


if __name__ == "__main__":
    unittest.main()
