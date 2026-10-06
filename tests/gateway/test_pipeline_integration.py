import unittest
from pathlib import Path

from app.contracts import RunPhase

from tests.gateway.support import build_application, temporary_directory
from tests.gateway.test_conversation import ReadyBackend, incoming


class FakeScheduler:
    def __init__(self):
        self.jobs = {}

    def enqueue(self, run_id, channel, conversation_id):
        self.jobs[run_id] = "QUEUED"
        return "QUEUED"

    def cancel(self, run_id):
        if run_id not in self.jobs:
            return None
        self.jobs[run_id] = "CANCELLED"
        return "CANCELLED"

    def pause(self, run_id):
        if run_id not in self.jobs:
            return None
        self.jobs[run_id] = "NEEDS_ATTENTION"
        return "NEEDS_ATTENTION"

    def resume(self, run_id):
        if self.jobs.get(run_id) != "NEEDS_ATTENTION":
            return self.jobs.get(run_id)
        self.jobs[run_id] = "QUEUED"
        return "QUEUED"

    def status(self, run_id):
        return self.jobs.get(run_id)


class PipelineGatewayIntegrationTests(unittest.TestCase):
    def test_exact_approval_enqueues_and_status_reports_pipeline_job(self):
        with temporary_directory() as directory:
            scheduler = FakeScheduler()
            store, application = build_application(
                Path(directory), ReadyBackend(), pipeline_scheduler=scheduler
            )
            application.handle(incoming(1, "기능 개발"))
            application.handle(incoming(2, "D:\\projects\\sample"))
            application.handle(incoming(3, "이 프로젝트 사용 승인해"))

            approved = application.handle(incoming(4, "개발 시작해"))
            status = application.handle(incoming(5, "상태"))

            binding = store.load_conversation("telegram", "200")
            run_id = binding["run_id"]
            self.assertIn("실행 큐에 등록", approved[0].text)
            self.assertEqual("QUEUED", scheduler.jobs[run_id])
            self.assertIn("실행 큐: QUEUED", status[0].text)

    def test_stop_cancels_a_queued_pipeline_job(self):
        with temporary_directory() as directory:
            scheduler = FakeScheduler()
            store, application = build_application(
                Path(directory), ReadyBackend(), pipeline_scheduler=scheduler
            )
            for identifier, text in enumerate(
                (
                    "기능 개발",
                    "D:\\projects\\sample",
                    "이 프로젝트 사용 승인해",
                    "개발 시작해",
                ),
                start=1,
            ):
                application.handle(incoming(identifier, text))

            stopped = application.handle(incoming(5, "중지"))

            binding = store.load_conversation("telegram", "200")
            state = store.load_run(binding["run_id"])
            self.assertEqual(RunPhase.CANCELLED, state.phase)
            self.assertEqual("CANCELLED", scheduler.jobs[state.run_id])
            self.assertIn("작업을 중지", stopped[0].text)

    def test_execution_question_answer_is_saved_then_requeues_the_paused_stage(self):
        with temporary_directory() as directory:
            scheduler = FakeScheduler()
            store, application = build_application(
                Path(directory), ReadyBackend(), pipeline_scheduler=scheduler
            )
            for identifier, text in enumerate(
                (
                    "기능 개발",
                    "D:\\projects\\sample",
                    "이 프로젝트 사용 승인해",
                    "개발 시작해",
                ),
                start=1,
            ):
                application.handle(incoming(identifier, text))

            binding = store.load_conversation("telegram", "200")
            state = store.load_run(binding["run_id"])
            developing = application.router.state_machine.transition(
                state, RunPhase.DEVELOPING, message="테스트 실행 시작"
            )
            paused = application.router.state_machine.pause(
                developing, "데이터베이스 선택 확인 필요"
            )
            store.open_execution_question(
                paused.run_id,
                "stage-001",
                "development",
                ("PostgreSQL을 사용할까요?",),
            )
            scheduler.jobs[paused.run_id] = "NEEDS_ATTENTION"

            answered = application.handle(incoming(5, "PostgreSQL로 진행해줘"))

            current = store.load_run(paused.run_id)
            question = store.open_execution_question_for_run(paused.run_id)
            self.assertEqual(RunPhase.DEVELOPING, current.phase)
            self.assertEqual("QUEUED", scheduler.jobs[paused.run_id])
            self.assertIsNone(question)
            self.assertIn("보존된 작업 공간", answered[0].text)
            self.assertIn("QUEUED", answered[0].text)
            decisions = [item["content"] for item in store.list_messages(paused.run_id)]
            self.assertIn("실행 중 사용자 답변: PostgreSQL로 진행해줘", decisions)

    def test_resume_control_explains_that_a_pending_question_needs_an_answer(self):
        with temporary_directory() as directory:
            scheduler = FakeScheduler()
            store, application = build_application(
                Path(directory), ReadyBackend(), pipeline_scheduler=scheduler
            )
            for identifier, text in enumerate(
                (
                    "기능 개발",
                    "D:\\projects\\sample",
                    "이 프로젝트 사용 승인해",
                    "개발 시작해",
                ),
                start=1,
            ):
                application.handle(incoming(identifier, text))
            binding = store.load_conversation("telegram", "200")
            state = store.load_run(binding["run_id"])
            developing = application.router.state_machine.transition(
                state, RunPhase.DEVELOPING, message="테스트 실행 시작"
            )
            paused = application.router.state_machine.pause(developing, "질문")
            store.open_execution_question(
                paused.run_id, "stage-001", "development", ("선택해 주세요",)
            )
            scheduler.jobs[paused.run_id] = "NEEDS_ATTENTION"

            response = application.handle(incoming(5, "재개"))

            self.assertIn("먼저 아래 질문", response[0].text)
            self.assertEqual(RunPhase.PAUSED, store.load_run(paused.run_id).phase)


if __name__ == "__main__":
    unittest.main()
