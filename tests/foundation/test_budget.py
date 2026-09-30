import unittest
from pathlib import Path

from app.contracts import RunState, TokenUsage
from app.services.budget import BudgetExceeded, BudgetManager, BudgetPolicy
from app.storage import StateStore
from tests.foundation.support import temporary_directory


class BudgetTests(unittest.TestCase):
    def make_manager(self, root: Path) -> tuple[StateStore, BudgetManager]:
        store = StateStore(root / "state.db")
        store.create_run(RunState(run_id="RUN-BUDGET"))
        policy = BudgetPolicy(
            calibration_mode=False,
            conversation_tokens=100,
            per_stage_tokens=160,
            whole_task_tokens=220,
            completion_reserve_tokens=20,
            retries={"technical_error": 1, "verification_failure": 1},
        )
        return store, BudgetManager(policy, store)

    def test_usage_is_persisted_and_limits_are_checked_before_call(self):
        with temporary_directory() as directory:
            store, manager = self.make_manager(Path(directory))
            manager.record_usage(
                "RUN-BUDGET",
                "stage-001",
                "development",
                "agent",
                TokenUsage(input_tokens=80, output_tokens=20),
            )
            allowed = manager.can_spend("RUN-BUDGET", "stage-001", 50)
            denied = manager.can_spend("RUN-BUDGET", "stage-001", 70)
            self.assertTrue(allowed.allowed)
            self.assertFalse(denied.allowed)
            self.assertEqual(100, store.load_run("RUN-BUDGET").total_tokens)

    def test_final_synthesis_can_use_only_the_reserved_completion_budget(self):
        with temporary_directory() as directory:
            _, manager = self.make_manager(Path(directory))
            manager.record_usage(
                "RUN-BUDGET",
                "stage-001",
                "development",
                "agent",
                TokenUsage(total_tokens=100),
            )

            ordinary = manager.can_spend("RUN-BUDGET", "stage-002", 120)
            synthesis = manager.can_spend(
                "RUN-BUDGET",
                "stage-002",
                120,
                use_completion_reserve=True,
            )

            self.assertFalse(ordinary.allowed)
            self.assertTrue(synthesis.allowed)

    def test_conversation_has_separate_limit(self):
        with temporary_directory() as directory:
            _, manager = self.make_manager(Path(directory))
            manager.record_usage(
                "RUN-BUDGET",
                "stage-001",
                "development",
                "conversation",
                TokenUsage(total_tokens=90),
            )
            decision = manager.can_spend(
                "RUN-BUDGET", "stage-001", 11, category="conversation"
            )
            self.assertFalse(decision.allowed)
            self.assertIn("conversation", decision.reason)

    def test_repository_analysis_has_its_own_limit_and_keeps_completion_reserve(self):
        with temporary_directory() as directory:
            store, manager = self.make_manager(Path(directory))
            manager.record_usage(
                "RUN-BUDGET",
                "stage-001",
                "development",
                "conversation",
                TokenUsage(total_tokens=90),
            )

            batch = manager.can_spend(
                "RUN-BUDGET", "stage-002", 11, category="repository_analysis"
            )
            synthesis = manager.can_spend(
                "RUN-BUDGET",
                "stage-003",
                120,
                category="repository_analysis",
                use_completion_reserve=True,
            )
            batch_without_reserve = manager.can_spend(
                "RUN-BUDGET", "stage-003", 120, category="repository_analysis"
            )

            self.assertTrue(batch.allowed)
            self.assertTrue(synthesis.allowed)
            self.assertFalse(batch_without_reserve.allowed)

    def test_retry_is_counted_and_never_resets_budget(self):
        with temporary_directory() as directory:
            _, manager = self.make_manager(Path(directory))
            self.assertTrue(
                manager.can_retry("RUN-BUDGET", "stage-001", "technical_error")
            )
            attempt = manager.record_retry(
                "RUN-BUDGET", "stage-001", "technical_error", "timeout"
            )
            self.assertEqual(1, attempt)
            self.assertFalse(
                manager.can_retry("RUN-BUDGET", "stage-001", "technical_error")
            )
            with self.assertRaises(RuntimeError):
                manager.record_retry(
                    "RUN-BUDGET", "stage-001", "technical_error", "timeout again"
                )

    def test_threshold_warning_is_once_per_scope_and_status_separates_actual_usage(self):
        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            store.create_run(RunState(run_id="RUN-WARNING"))
            manager = BudgetManager(
                BudgetPolicy(
                    conversation_tokens=100,
                    per_stage_tokens=100,
                    whole_task_tokens=200,
                    warning_threshold_percent=80,
                ),
                store,
            )
            warnings = manager.record_usage(
                "RUN-WARNING",
                "stage-001",
                "development",
                "conversation",
                TokenUsage(input_tokens=60, output_tokens=20),
            )
            self.assertEqual({"conversation", "stage"}, {item.scope for item in warnings})
            self.assertEqual(
                (),
                manager.record_usage(
                    "RUN-WARNING",
                    "stage-001",
                    "development",
                    "conversation",
                    TokenUsage(total_tokens=1),
                ),
            )
            manager.record_usage(
                "RUN-WARNING",
                "stage-001",
                "review",
                "agent",
                TokenUsage(total_tokens=10, estimated=True),
            )
            snapshot = manager.snapshot("RUN-WARNING", stage_id="stage-001")
            self.assertEqual(91, snapshot.total_tokens)
            self.assertEqual(81, snapshot.actual_tokens)
            self.assertEqual(10, snapshot.estimated_tokens)
            self.assertIn("실제 81 / 추정 10", manager.render_status("RUN-WARNING", stage_id="stage-001"))

    def test_warning_sink_can_find_the_active_conversation_for_delivery(self):
        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            store.create_run(RunState(run_id="RUN-DELIVERY"))
            store.create_conversation_session(
                "telegram", "chat-1", "owner-1", "RUN-DELIVERY", "development"
            )
            delivered = []
            manager = BudgetManager(
                BudgetPolicy(per_stage_tokens=10, warning_threshold_percent=80),
                store,
                warning_sink=lambda warnings, run_id: delivered.append((warnings, run_id)),
            )

            manager.record_usage(
                "RUN-DELIVERY",
                "stage-001",
                "development",
                "agent",
                TokenUsage(total_tokens=8),
            )

            self.assertEqual("RUN-DELIVERY", delivered[0][1])
            self.assertEqual("stage", delivered[0][0][0].scope)
            self.assertEqual(
                ({"channel": "telegram", "conversation_id": "chat-1"},),
                store.conversation_targets_for_run("RUN-DELIVERY"),
            )

    def test_parallel_reservations_cannot_overbook_a_conversation_limit(self):
        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            store.create_run(RunState(run_id="RUN-RESERVATION"))
            manager = BudgetManager(BudgetPolicy(conversation_tokens=100), store)

            first = manager.reserve(
                "RUN-RESERVATION", "chat-001", "development", "conversation", 60
            )
            with self.assertRaises(BudgetExceeded):
                manager.reserve(
                    "RUN-RESERVATION", "chat-001", "review", "conversation", 60
                )

            manager.record_usage(
                "RUN-RESERVATION",
                "chat-001",
                "development",
                "conversation",
                TokenUsage(total_tokens=45),
                reservation=first,
            )
            second = manager.reserve(
                "RUN-RESERVATION", "chat-001", "review", "conversation", 55
            )
            self.assertTrue(second.active)

    def test_usage_over_a_reservation_is_recorded_then_stops_the_call_path(self):
        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            store.create_run(RunState(run_id="RUN-RESERVATION-OVERAGE"))
            manager = BudgetManager(BudgetPolicy(conversation_tokens=100), store)
            reservation = manager.reserve(
                "RUN-RESERVATION-OVERAGE",
                "chat-001",
                "development",
                "conversation",
                10,
            )

            with self.assertRaises(BudgetExceeded):
                manager.record_usage(
                    "RUN-RESERVATION-OVERAGE",
                    "chat-001",
                    "development",
                    "conversation",
                    TokenUsage(total_tokens=11),
                    reservation=reservation,
                )

            self.assertEqual(11, store.usage_total("RUN-RESERVATION-OVERAGE"))
            self.assertEqual(
                0, store.reserved_token_total("RUN-RESERVATION-OVERAGE")
            )

    def test_failed_warning_queue_is_retried_without_reclaiming_the_warning(self):
        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            store.create_run(RunState(run_id="RUN-WARNING-RETRY"))
            store.create_conversation_session(
                "telegram", "chat-1", "owner-1", "RUN-WARNING-RETRY", "development"
            )
            attempts = []

            def failing_sink(warnings, run_id):
                attempts.append((warnings, run_id))
                raise OSError("temporary outbox failure")

            manager = BudgetManager(
                BudgetPolicy(per_stage_tokens=10, warning_threshold_percent=80),
                store,
                warning_sink=failing_sink,
            )
            manager.record_usage(
                "RUN-WARNING-RETRY",
                "stage-001",
                "development",
                "agent",
                TokenUsage(total_tokens=8),
            )
            self.assertEqual(1, len(attempts))
            self.assertEqual(1, len(store.pending_budget_warning_claims("RUN-WARNING-RETRY")))

            def durable_sink(warnings, run_id):
                store.queue_budget_warning_delivery(
                    run_id,
                    tuple(
                        {
                            "scope": warning.scope,
                            "stage_id": warning.stage_id,
                            "threshold_percent": warning.threshold_percent,
                            "text": BudgetManager.render_warning(warning),
                        }
                        for warning in warnings
                    ),
                    store.conversation_targets_for_run(run_id),
                )

            manager.set_warning_sink(durable_sink)
            manager.retry_pending_warnings()
            self.assertEqual((), store.pending_budget_warning_claims("RUN-WARNING-RETRY"))
            self.assertEqual(
                1,
                len(store.deliverable_outbound("telegram")),
            )


if __name__ == "__main__":
    unittest.main()
