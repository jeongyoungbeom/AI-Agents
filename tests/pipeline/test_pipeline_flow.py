from __future__ import annotations

import json
import subprocess
import threading
import unittest
from pathlib import Path

from app.agents.parsing import InvalidAgentResponse, parse_review_summary
from app.contracts import RoleId, RunPhase, TokenUsage
from app.services.hermes import HermesCancelled, HermesResult

from tests.pipeline.support import (
    FakeRoleRunner,
    build_pipeline,
    create_repository,
    git,
    temporary_directory,
)


class PipelineFlowTests(unittest.TestCase):
    def test_user_question_pauses_persists_answer_and_restarts_with_answer_context(self):
        class AskingRunner(FakeRoleRunner):
            def __init__(self):
                super().__init__(review_responses=[[]], development_outputs=["fixed"])
                self.prompts = []
                self.development_calls = 0

            def run(self, *args, **kwargs):
                role_id = args[2]
                prompt = args[4]
                if role_id != RoleId.DEVELOPMENT:
                    return super().run(*args, **kwargs)
                self.calls.append((role_id, kwargs["allow_writes"]))
                self.prompts.append(prompt)
                if self.development_calls == 0:
                    self.development_calls += 1
                    return HermesResult(
                        '{"summary":"선택 확인 필요","needs_user_input":["PostgreSQL을 사용할까요?"]}',
                        TokenUsage(100, 20),
                        0.01,
                    )
                self.development_calls += 1
                repository = Path(args[3])
                (repository / "feature.txt").write_text("fixed\n", encoding="utf-8")
                return HermesResult(
                    '{"summary":"구현 완료","needs_user_input":[]}',
                    TokenUsage(100, 20),
                    0.01,
                )

        with temporary_directory() as directory:
            root = Path(directory)
            repository = create_repository(root)
            runner = AskingRunner()
            store, worker, run_id = build_pipeline(root, repository, runner)

            self.assertTrue(worker.run_once())

            paused = store.load_run(run_id)
            question = store.open_execution_question_for_run(run_id)
            self.assertEqual(RunPhase.PAUSED, paused.phase)
            self.assertEqual("NEEDS_ATTENTION", store.pipeline_job(run_id)["status"])
            self.assertEqual(("PostgreSQL을 사용할까요?",), question["questions"])
            self.assertIn(
                "PostgreSQL을 사용할까요?",
                "\n".join(item["text"] for item in store.deliverable_outbound("telegram")),
            )

            store.answer_execution_question(run_id, "PostgreSQL로 진행해줘")
            application_state = worker.machine.transition(
                paused,
                RunPhase.DEVELOPING,
                message="사용자 답변 뒤 안전한 단계 시작 지점에서 재개",
            )
            self.assertEqual(RunPhase.DEVELOPING, application_state.phase)
            self.assertEqual("QUEUED", worker.resume(run_id))
            self.assertTrue(worker.run_once())

            self.assertEqual(RunPhase.COMPLETED, store.load_run(run_id).phase)
            self.assertIn("PostgreSQL로 진행해줘", runner.prompts[-1])

    def test_question_after_a_write_is_stopped_without_creating_a_resumable_question(self):
        class MutatingQuestionRunner(FakeRoleRunner):
            def run(self, *args, **kwargs):
                role_id = args[2]
                if role_id != RoleId.DEVELOPMENT:
                    return super().run(*args, **kwargs)
                self.calls.append((role_id, kwargs["allow_writes"]))
                repository = Path(args[3])
                (repository / "feature.txt").write_text("partial\n", encoding="utf-8")
                return HermesResult(
                    '{"summary":"확인 필요","needs_user_input":["진행할까요?"]}',
                    TokenUsage(100, 20),
                    0.01,
                )

        with temporary_directory() as directory:
            root = Path(directory)
            repository = create_repository(root)
            store, worker, run_id = build_pipeline(
                root, repository, MutatingQuestionRunner()
            )

            self.assertTrue(worker.run_once())

            self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
            self.assertEqual("NEEDS_ATTENTION", store.pipeline_job(run_id)["status"])
            self.assertIsNone(store.open_execution_question_for_run(run_id))
            messages = "\n".join(
                item["text"] for item in store.deliverable_outbound("telegram")
            )
            self.assertIn("미커밋 파일 변경", messages)
    def test_progress_is_flushed_immediately_and_activity_is_reported(self):
        with temporary_directory() as directory:
            root = Path(directory)
            repository = create_repository(root)
            runner = FakeRoleRunner(review_responses=[[]])
            store, worker, _run_id = build_pipeline(root, repository, runner)
            outbound_notifications = []
            activity = []
            worker.coordinator.set_outbound_notifier(
                lambda: outbound_notifications.append(
                    len(store.deliverable_outbound("telegram"))
                )
            )
            worker.activity_notifier = (
                lambda channel, conversation: activity.append((channel, conversation))
            )
            worker.activity_interval_seconds = 0.0001

            self.assertTrue(worker.run_once())

            self.assertGreaterEqual(len(outbound_notifications), 4)
            self.assertTrue(all(count >= 1 for count in outbound_notifications))
            self.assertGreaterEqual(len(activity), 2)
            self.assertTrue(
                all(item == ("telegram", "chat-1") for item in activity)
            )

    @staticmethod
    def finding(category: str, *, severity: str = "medium", finding_id: str = "F-001"):
        return {
            "finding_id": finding_id,
            "severity": severity,
            "category": category,
            "evidence": "검토에서 확인된 문제",
            "required_change": "단계 범위 안에서 수정",
            "file": "feature.txt",
            "line": 1,
            "status": "open",
        }

    def test_full_stage_runs_development_review_improvement_and_verification(self):
        with temporary_directory() as directory:
            root = Path(directory)
            repository = create_repository(root)
            runner = FakeRoleRunner()
            store, worker, run_id = build_pipeline(root, repository, runner)

            self.assertTrue(worker.run_once())

            state = store.load_run(run_id)
            self.assertEqual(RunPhase.COMPLETED, state.phase)
            self.assertEqual("COMPLETED", store.pipeline_job(run_id)["status"])
            self.assertEqual("fixed", (repository / "feature.txt").read_text().strip())
            self.assertEqual(
                [
                    (RoleId.DEVELOPMENT, True),
                    (RoleId.REVIEW, False),
                    (RoleId.IMPROVEMENT, True),
                ],
                runner.calls,
            )
            self.assertEqual("3", git(repository, "rev-list", "--count", "HEAD"))
            stage = root / "artifacts" / run_id / "stages" / "stage-001"
            self.assertTrue((stage / "contract.json").is_file())
            self.assertTrue((stage / "handoff-development-to-review.json").is_file())
            self.assertTrue((stage / "handoff-review-to-improvement.json").is_file())
            self.assertTrue((stage / "verification.json").is_file())
            messages = store.deliverable_outbound("telegram", limit=100)
            text = "\n".join(item["text"] for item in messages)
            self.assertIn("빌더 개발 완료 → 센티널이 리뷰", text)
            self.assertIn("센티널 구현 문제 1건 → 피니셔가 수정을 시작", text)
            self.assertIn("모든 단계를 완료", text)

    def test_every_stage_repeats_the_three_role_flow(self):
        with temporary_directory() as directory:
            root = Path(directory)
            repository = create_repository(root)
            runner = FakeRoleRunner()
            store, worker, run_id = build_pipeline(
                root, repository, runner, stage_count=2
            )

            worker.run_once()

            self.assertEqual(RunPhase.COMPLETED, store.load_run(run_id).phase)
            self.assertEqual(
                [
                    RoleId.DEVELOPMENT,
                    RoleId.REVIEW,
                    RoleId.IMPROVEMENT,
                    RoleId.DEVELOPMENT,
                    RoleId.REVIEW,
                    RoleId.IMPROVEMENT,
                ],
                [role for role, _allow in runner.calls],
            )
            self.assertEqual("5", git(repository, "rev-list", "--count", "HEAD"))

    def test_clean_review_skips_finisher_and_verifies_directly(self):
        with temporary_directory() as directory:
            root = Path(directory)
            repository = create_repository(root)
            runner = FakeRoleRunner(
                review_responses=[[]], development_outputs=["fixed"]
            )
            store, worker, run_id = build_pipeline(root, repository, runner)

            worker.run_once()

            self.assertEqual(RunPhase.COMPLETED, store.load_run(run_id).phase)
            self.assertEqual(
                [(RoleId.DEVELOPMENT, True), (RoleId.REVIEW, False)],
                runner.calls,
            )
            stage = root / "artifacts" / run_id / "stages" / "stage-001"
            self.assertFalse((stage / "handoff-review-to-improvement.json").exists())
            messages = store.deliverable_outbound("telegram", limit=100)
            text = "\n".join(item["text"] for item in messages)
            self.assertIn("피니셔 보완 없이 먼저 검증", text)

    def test_design_finding_returns_to_builder_then_runs_one_rereview(self):
        with temporary_directory() as directory:
            root = Path(directory)
            repository = create_repository(root)
            runner = FakeRoleRunner(
                review_responses=[[self.finding("design")], []],
                development_outputs=["draft", "fixed"],
            )
            store, worker, run_id = build_pipeline(root, repository, runner)

            worker.run_once()

            self.assertEqual(RunPhase.COMPLETED, store.load_run(run_id).phase)
            self.assertEqual(
                [
                    RoleId.DEVELOPMENT,
                    RoleId.REVIEW,
                    RoleId.DEVELOPMENT,
                    RoleId.REVIEW,
                ],
                [role for role, _allow in runner.calls],
            )
            self.assertEqual(1, store.retry_count(run_id, "stage-001", "design_rework"))
            stage = root / "artifacts" / run_id / "stages" / "stage-001"
            self.assertTrue((stage / "handoff-review-to-development.json").is_file())
            self.assertTrue(
                (stage / "handoff-development-to-review-after-design-rework.json").is_file()
            )
            self.assertTrue((stage / "review-after-design-rework.json").is_file())
            rework = json.loads(
                (stage / "development-rework-result.json").read_text(encoding="utf-8")
            )
            actual = tuple(
                item
                for item in git(
                    repository,
                    "diff",
                    "--name-only",
                    f"{rework['base_sha']}..{rework['candidate_sha']}",
                ).splitlines()
                if item
            )
            self.assertEqual(tuple(rework["changed_files"]), actual)

    def test_unresolved_design_finding_pauses_after_one_rework(self):
        with temporary_directory() as directory:
            root = Path(directory)
            repository = create_repository(root)
            design = self.finding("design")
            runner = FakeRoleRunner(
                review_responses=[[design], [design]],
                development_outputs=["draft", "fixed"],
            )
            store, worker, run_id = build_pipeline(root, repository, runner)

            worker.run_once()

            self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
            self.assertEqual("NEEDS_ATTENTION", store.pipeline_job(run_id)["status"])
            self.assertEqual(1, store.retry_count(run_id, "stage-001", "design_rework"))
            self.assertNotIn(RoleId.IMPROVEMENT, [role for role, _ in runner.calls])

    def test_high_implementation_finding_gets_one_closure_review(self):
        with temporary_directory() as directory:
            root = Path(directory)
            repository = create_repository(root)
            runner = FakeRoleRunner(
                review_responses=[
                    [self.finding("implementation", severity="high")],
                    [],
                ]
            )
            store, worker, run_id = build_pipeline(root, repository, runner)

            worker.run_once()

            self.assertEqual(RunPhase.COMPLETED, store.load_run(run_id).phase)
            self.assertEqual(
                [
                    RoleId.DEVELOPMENT,
                    RoleId.REVIEW,
                    RoleId.IMPROVEMENT,
                    RoleId.REVIEW,
                ],
                [role for role, _allow in runner.calls],
            )
            self.assertEqual(1, store.retry_count(run_id, "stage-001", "closure_review"))
            stage = root / "artifacts" / run_id / "stages" / "stage-001"
            self.assertTrue((stage / "handoff-improvement-to-review.json").is_file())
            self.assertTrue((stage / "review-closure-result.json").is_file())
            closure = json.loads(
                (stage / "handoff-improvement-to-review.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertGreater(closure["usage"]["total_tokens"], 0)

    def test_unresolved_closure_review_pauses_without_second_finisher_pass(self):
        with temporary_directory() as directory:
            root = Path(directory)
            repository = create_repository(root)
            high = self.finding("implementation", severity="high")
            runner = FakeRoleRunner(review_responses=[[high], [high]])
            store, worker, run_id = build_pipeline(root, repository, runner)

            worker.run_once()

            self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
            self.assertEqual(1, [role for role, _ in runner.calls].count(RoleId.IMPROVEMENT))
            self.assertEqual(1, store.retry_count(run_id, "stage-001", "closure_review"))
            verification = json.loads(
                (
                    root
                    / "artifacts"
                    / run_id
                    / "stages"
                    / "stage-001"
                    / "verification.json"
                ).read_text(encoding="utf-8")
            )
            self.assertTrue(verification["passed"])
            self.assertTrue(verification["closure_reviewed"])
            self.assertFalse(verification["closure_review_passed"])

    def test_finisher_without_a_real_change_pauses(self):
        with temporary_directory() as directory:
            root = Path(directory)
            repository = create_repository(root)
            runner = FakeRoleRunner(improvement_fixes=False)
            store, worker, run_id = build_pipeline(root, repository, runner)

            worker.run_once()

            self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
            self.assertEqual("NEEDS_ATTENTION", store.pipeline_job(run_id)["status"])
            stage = root / "artifacts" / run_id / "stages" / "stage-001"
            result = json.loads(
                (stage / "improvement-result.json").read_text(encoding="utf-8")
            )
            self.assertFalse(result["changed"])

    def test_closure_review_targets_sha_after_verification_retry(self):
        with temporary_directory() as directory:
            root = Path(directory)
            repository = create_repository(root)
            runner = FakeRoleRunner(
                review_responses=[
                    [self.finding("implementation", severity="critical")],
                    [],
                ],
                improvement_outputs=["almost", "fixed"],
            )
            store, worker, run_id = build_pipeline(root, repository, runner)

            worker.run_once()

            self.assertEqual(RunPhase.COMPLETED, store.load_run(run_id).phase)
            self.assertEqual(
                [
                    RoleId.DEVELOPMENT,
                    RoleId.REVIEW,
                    RoleId.IMPROVEMENT,
                    RoleId.IMPROVEMENT,
                    RoleId.REVIEW,
                ],
                [role for role, _allow in runner.calls],
            )
            stage = root / "artifacts" / run_id / "stages" / "stage-001"
            verification = json.loads(
                (stage / "verification.json").read_text(encoding="utf-8")
            )
            closure = json.loads(
                (stage / "handoff-improvement-to-review.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(verification["verification_retry_sha"], closure["candidate_sha"])
            self.assertEqual(verification["final_sha"], closure["candidate_sha"])

    def test_live_review_requires_an_explicit_category(self):
        response = json.dumps(
            {
                "summary": "분류 누락",
                "findings": [
                    {
                        "finding_id": "F-001",
                        "severity": "high",
                        "evidence": "근거",
                        "required_change": "수정",
                    }
                ],
                "needs_user_input": [],
            }
        )
        with self.assertRaises(InvalidAgentResponse):
            parse_review_summary(response)

    def test_reviewer_mutation_stops_without_reverting_user_visible_evidence(self):
        with temporary_directory() as directory:
            root = Path(directory)
            repository = create_repository(root)
            store, worker, run_id = build_pipeline(
                root, repository, FakeRoleRunner(reviewer_mutates=True)
            )

            self.assertTrue(worker.run_once())

            self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
            self.assertEqual("NEEDS_ATTENTION", store.pipeline_job(run_id)["status"])
            self.assertFalse((repository / "reviewer-was-here.txt").exists())
            recovery = json.loads(
                (root / "artifacts" / run_id / "worktree-recovery.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn(
                "reviewer-was-here.txt", recovery["recovery"]["changed_files"]
            )

    def test_dirty_repository_is_blocked_before_agent_call(self):
        with temporary_directory() as directory:
            root = Path(directory)
            repository = create_repository(root)
            runner = FakeRoleRunner()
            store, worker, run_id = build_pipeline(root, repository, runner)
            (repository / "local-user-change.txt").write_text("keep me\n", encoding="utf-8")

            worker.run_once()

            self.assertEqual([], runner.calls)
            self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
            self.assertEqual("NEEDS_ATTENTION", store.pipeline_job(run_id)["status"])
            self.assertTrue((repository / "local-user-change.txt").is_file())

    def test_out_of_scope_write_is_not_committed_and_has_recovery_record(self):
        with temporary_directory() as directory:
            root = Path(directory)
            repository = create_repository(root)
            runner = FakeRoleRunner(development_out_of_scope_write=True)
            store, worker, run_id = build_pipeline(root, repository, runner)

            self.assertTrue(worker.run_once())

            self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
            self.assertEqual("NEEDS_ATTENTION", store.pipeline_job(run_id)["status"])
            self.assertEqual("1", git(repository, "rev-list", "--count", "HEAD"))
            self.assertFalse((repository / "feature.txt").exists())
            self.assertFalse((repository / "outside-stage-scope.txt").exists())
            self.assertEqual("", git(repository, "status", "--short"))
            recovery = json.loads(
                (
                    root
                    / "artifacts"
                    / run_id
                    / "stages"
                    / "stage-001"
                    / "recovery.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual("development", recovery["operation"])
            self.assertIn("outside-stage-scope.txt", recovery["recovery"]["changed_files"])
            worktree_recovery = json.loads(
                (root / "artifacts" / run_id / "worktree-recovery.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn(
                "outside-stage-scope.txt",
                worktree_recovery["recovery"]["changed_files"],
            )

    def test_verification_write_stops_and_has_recovery_record(self):
        with temporary_directory() as directory:
            root = Path(directory)
            repository = create_repository(root)
            (repository / "verify.py").write_text(
                "from pathlib import Path\n"
                "Path('README.md').write_text('verification changed this\\n', encoding='utf-8')\n",
                encoding="utf-8",
            )
            git(repository, "add", "verify.py")
            git(repository, "commit", "-m", "verification mutates")
            runner = FakeRoleRunner(review_responses=[[]], development_outputs=["fixed"])
            store, worker, run_id = build_pipeline(root, repository, runner)

            self.assertTrue(worker.run_once())

            self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
            self.assertEqual("", git(repository, "status", "--short"))
            self.assertEqual("test\n", (repository / "README.md").read_text(encoding="utf-8"))
            recovery = json.loads(
                (
                    root
                    / "artifacts"
                    / run_id
                    / "stages"
                    / "stage-001"
                    / "recovery.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual("verification", recovery["operation"])
            self.assertEqual("verification", recovery["role_id"])
            patch = (
                root
                / "artifacts"
                / run_id
                / "stages"
                / "stage-001"
                / "recovery.patch"
            ).read_text(encoding="utf-8")
            self.assertIn("README.md", patch)

    def test_verification_failure_uses_one_bounded_improvement_retry(self):
        with temporary_directory() as directory:
            root = Path(directory)
            repository = create_repository(root, verifier_passes=False)
            runner = FakeRoleRunner(improvement_fixes=True)
            store, worker, run_id = build_pipeline(root, repository, runner)

            worker.run_once()

            roles = [role for role, _allow in runner.calls]
            self.assertEqual(2, roles.count(RoleId.IMPROVEMENT))
            self.assertEqual(1, store.retry_count(run_id, "stage-001", "verification_failure"))
            self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
            self.assertEqual("NEEDS_ATTENTION", store.pipeline_job(run_id)["status"])

    def test_running_job_honors_a_user_cancellation_request(self):
        with temporary_directory() as directory:
            root = Path(directory)
            repository = create_repository(root)
            runner = FakeRoleRunner()
            store, worker, run_id = build_pipeline(root, repository, runner)
            original_run = runner.run

            def cancel_after_development(*args, **kwargs):
                result = original_run(*args, **kwargs)
                if args[2] == RoleId.DEVELOPMENT:
                    store.request_pipeline_cancel(run_id)
                return result

            runner.run = cancel_after_development

            worker.run_once()

            self.assertEqual(RunPhase.CANCELLED, store.load_run(run_id).phase)
            self.assertEqual("CANCELLED", store.pipeline_job(run_id)["status"])
            self.assertEqual([RoleId.DEVELOPMENT], [role for role, _ in runner.calls])

    def test_worker_stop_cancels_a_running_agent_before_join(self):
        class BlockingRunner(FakeRoleRunner):
            def __init__(self):
                super().__init__()
                self.entered = threading.Event()
                self.pause = threading.Event()

            def run(self, *args, cancelled=None, **kwargs):
                self.entered.set()
                while cancelled is not None and not cancelled():
                    self.pause.wait(0.01)
                raise HermesCancelled("gateway stopping")

        with temporary_directory() as directory:
            root = Path(directory)
            repository = create_repository(root)
            runner = BlockingRunner()
            store, worker, run_id = build_pipeline(root, repository, runner)
            thread = threading.Thread(target=worker.run_once)
            thread.start()
            try:
                self.assertTrue(runner.entered.wait(timeout=30))
            finally:
                worker.stop()
                thread.join(timeout=30)

            self.assertFalse(thread.is_alive())
            self.assertEqual(RunPhase.CANCELLED, store.load_run(run_id).phase)
            self.assertEqual("CANCELLED", store.pipeline_job(run_id)["status"])


if __name__ == "__main__":
    unittest.main()
