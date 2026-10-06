import json
import unittest
from pathlib import Path

from app.contracts import RoleId
from app.gateway.core import AgentReply, IncomingMessage
from app.orchestrator import RunStateMachine
from app.services.context import ContextService
from app.services.repository import (
    RepositoryAnalysisRequest, RepositoryAnalysisStatus, RepositoryReadContext,
    RepositoryToolBatch, RepositoryToolRequest, RepositoryToolResult,
)
from app.storage import StateStore, StoreError
from tests.gateway.support import (
    TEST_REPOSITORY_HEAD, TEST_REPOSITORY_IDENTITY, build_application,
    temporary_directory,
)


def incoming(number: int, text: str) -> IncomingMessage:
    return IncomingMessage(
        channel="telegram", conversation_id="200", user_id="100",
        external_message_id=f"message-{number}", text=text,
    )


class Team:
    def __init__(self):
        self.contexts = []

    def preflight(self, *_args):
        return None

    def respond_as(self, _state, context, _message, _role_id, **_kwargs):
        self.contexts.append(context)
        return AgentReply("첫 번째 문제는 테스트 누락입니다. 근거는 src/a.py:1입니다.")


class Reader:
    def __init__(self):
        self.inspections = 0

    def should_inspect(self, text):
        return "프로젝트 설명" in text

    def inspect(self, *_args, **_kwargs):
        self.inspections += 1
        return RepositoryReadContext(
            TEST_REPOSITORY_IDENTITY, TEST_REPOSITORY_HEAD, "main",
            ("src/a.py",), (("src/a.py", "def first():\n    pass\n"),),
        )


class Tools:
    def execute(self, _path, requests, **_kwargs):
        request = requests[0]
        content = f"1: {request.path}의 내용"
        return RepositoryToolBatch(
            TEST_REPOSITORY_IDENTITY, TEST_REPOSITORY_HEAD,
            (RepositoryToolResult(
                request, "ok", {"path": request.path, "start_line": 1,
                                 "end_line": 1, "content": content},
            ),),
        )


class S03ContextTests(unittest.TestCase):
    def test_recent_analysis_result_has_priority_over_old_memory(self):
        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            RunStateMachine(store).create_run("RUN-S03-PRIORITY")
            context = ContextService(store)
            context.save_memory("user", "100", "오래된 기억 " + "M" * 80)
            context.add_message(
                "RUN-S03-PRIORITY", "review", "첫 번째 분석 결과 " + "A" * 65,
                kind="analysis_result",
            )
            bundle = context.build("RUN-S03-PRIORITY", user_id="100",
                                   max_characters=100)
            self.assertEqual(1, len(bundle.recent_messages))
            self.assertEqual((), bundle.memories)

    def test_followup_keeps_project_answer_and_attributed_evidence(self):
        with temporary_directory() as directory:
            team, reader = Team(), Reader()
            store, application = build_application(
                Path(directory), object(), team_backend=team,
                repository_reader=reader, repository_tools=Tools(),
            )
            application.handle(incoming(1, "D:\\projects\\sample"))
            application.handle(incoming(2, "이 프로젝트 사용 승인해"))
            application.handle(incoming(3, "센티널아 프로젝트 설명해"))
            application.handle(incoming(4, "그럼 테스트할 때야?"))
            application.handle(incoming(5, "첫 번째를 설명해"))

            self.assertEqual(1, reader.inspections)
            for context in team.contexts[1:]:
                self.assertTrue(any(
                    "첫 번째 문제는 테스트 누락" in item["content"]
                    and item["data"].get("untrusted_repository_data")
                    for item in context.recent_messages
                ))
                self.assertTrue(any(
                    item["data"].get("path") == "src/a.py"
                    and item["data"].get("head_sha") == TEST_REPOSITORY_HEAD
                    for item in context.evidence
                ))
            session = store.load_conversation_session("telegram", "200")
            reopened = StateStore(Path(directory) / "state.db")
            restored = ContextService(reopened).build(
                session.session_run_id, repository_identity=TEST_REPOSITORY_IDENTITY,
            )
            self.assertEqual(1, len(restored.evidence))

    def test_tool_rounds_keep_prior_documents_and_both_file_reads(self):
        with temporary_directory() as directory:
            store, application = build_application(
                Path(directory), object(), team_backend=Team(),
                repository_tools=Tools(),
            )
            application.handle(incoming(1, "D:\\projects\\sample"))
            application.handle(incoming(2, "이 프로젝트 사용 승인해"))
            binding = store.load_conversation("telegram", "200")
            state = store.load_run(binding["run_id"])
            original = {
                "source": "approved_committed_git_snapshot",
                "untrusted_repository_data": True,
                "identity_hash": TEST_REPOSITORY_IDENTITY,
                "head_sha": TEST_REPOSITORY_HEAD,
                "branch": "main", "tree": ["README.md", "src/a.py", "src/b.py"],
                "documents": [{"path": "README.md", "content": "프로젝트 안내"}],
            }
            first, notice = application.router._repository_tool_context_for_chat(
                incoming(3, "두 파일 비교"), state, RoleId.REVIEW, original,
                (RepositoryToolRequest("read_file", path="src/a.py"),), tool_round=1,
            )
            second, notice2 = application.router._repository_tool_context_for_chat(
                incoming(3, "두 파일 비교"), state, RoleId.REVIEW, first,
                (RepositoryToolRequest("read_file", path="src/b.py"),), tool_round=2,
            )
            self.assertEqual(("", ""), (notice, notice2))
            self.assertEqual("README.md", second["documents"][0]["path"])
            self.assertEqual(
                ["src/a.py", "src/b.py"],
                [batch["results"][0]["data"]["path"] for batch in second["tool_results"]],
            )
            evidence = ContextService(store).build(
                state.run_id, repository_identity=TEST_REPOSITORY_IDENTITY,
            ).evidence
            self.assertEqual({"src/a.py", "src/b.py"},
                             {item["data"]["path"] for item in evidence})

    def test_large_tool_context_retains_locations_inside_context_cap(self):
        with temporary_directory() as directory:
            _, application = build_application(Path(directory), object(), team_backend=Team())
            context = {
                "identity_hash": TEST_REPOSITORY_IDENTITY,
                "head_sha": TEST_REPOSITORY_HEAD,
                "untrusted_repository_data": True,
                "tree": [f"src/{index}.py" for index in range(100)],
                "documents": [{"path": "README.md", "content": "A" * 18000}],
                "tool_results": [
                    {"head_sha": TEST_REPOSITORY_HEAD, "results": [
                        {"request": {"tool": "read_file", "path": path},
                         "status": "ok", "data": {"path": path,
                         "start_line": 1, "end_line": 500, "content": "B" * 12000}}
                    ]}
                    for path in ("src/a.py", "src/b.py")
                ],
            }
            bounded = application.router._bound_repository_context(context)
            self.assertLessEqual(
                len(json.dumps(bounded, ensure_ascii=False, separators=(",", ":"))),
                application.router.context.policy.max_characters - 4000,
            )
            self.assertEqual("README.md", bounded["documents"][0]["path"])
            self.assertEqual(
                ["src/a.py", "src/b.py"],
                [batch["results"][0]["data"]["path"]
                 for batch in bounded["tool_results"]],
            )

    def test_analysis_finalization_links_once_and_stays_project_scoped(self):
        with temporary_directory() as directory:
            root = Path(directory)
            team = Team()
            store, application = build_application(root, object(), team_backend=team)
            application.handle(incoming(1, "D:\\projects\\sample"))
            application.handle(incoming(2, "이 프로젝트 사용 승인해"))
            session = store.load_conversation_session("telegram", "200")
            request = RepositoryAnalysisRequest.create(
                channel="telegram", conversation_id="200", user_id="100",
                source_message_id="message-3", role_id=RoleId.REVIEW.value,
                request_text="전체 분석", repository_path=str(root),
                repository_identity=TEST_REPOSITORY_IDENTITY,
                commit_sha=TEST_REPOSITORY_HEAD, branch="main",
            )
            RunStateMachine(store).create_run(request.analysis_id)
            store.create_repository_analysis(request)
            store.claim_next_repository_analysis("worker")
            store.add_repository_analysis_evidence(
                request.analysis_id, commit_sha=TEST_REPOSITORY_HEAD,
                path="src/a.py", start_line=1, end_line=2, phase="STRUCTURE",
                kind="source_read", summary="첫 번째 문제",
            )
            store.finish_repository_analysis_with_response(
                request.analysis_id, "worker", RepositoryAnalysisStatus.COMPLETED,
                "첫 번째 문제: 테스트 누락", reason="COMPLETED",
            )
            with self.assertRaises(StoreError):
                store.finish_repository_analysis_with_response(
                    request.analysis_id, "worker", RepositoryAnalysisStatus.COMPLETED,
                    "첫 번째 문제: 테스트 누락", reason="COMPLETED",
                )
            reopened = StateStore(root / "state.db")
            messages = reopened.list_messages(session.session_run_id)
            results = [item for item in messages if item["kind"] == "analysis_result"]
            self.assertEqual(1, len(results))
            self.assertEqual(request.analysis_id, results[0]["data"]["analysis_id"])
            self.assertEqual("src/a.py", results[0]["data"]["evidence_refs"][0]["path"])
            alpha = ContextService(reopened).build(
                session.session_run_id, repository_identity=TEST_REPOSITORY_IDENTITY,
            )
            beta = ContextService(reopened).build(
                session.session_run_id, repository_identity="c" * 64,
            )
            self.assertTrue(any("첫 번째 문제" in item["content"]
                                for item in alpha.recent_messages))
            self.assertFalse(any("첫 번째 문제" in item["content"]
                                 for item in beta.recent_messages))
            application.handle(incoming(4, "그중 첫 번째부터 설명해"))
            self.assertTrue(any(
                item["kind"] == "analysis_result" and "첫 번째 문제" in item["content"]
                for item in team.contexts[-1].recent_messages
            ))

    def test_old_snapshot_is_marked_as_history_after_head_changes(self):
        with temporary_directory() as directory:
            team, reader = Team(), Reader()
            store, application = build_application(
                Path(directory), object(), team_backend=team,
                repository_reader=reader, repository_tools=Tools(),
            )
            application.handle(incoming(1, "D:\\projects\\sample"))
            application.handle(incoming(2, "이 프로젝트 사용 승인해"))
            application.handle(incoming(3, "센티널아 프로젝트 설명해"))
            next_head = "c" * 40
            store.refresh_project_head(
                "telegram", "200", "100", TEST_REPOSITORY_IDENTITY, next_head,
            )
            application.handle(incoming(4, "그럼 테스트할 때야?"))
            context = team.contexts[-1]
            self.assertTrue(any(
                item["data"].get("stale_repository_snapshot")
                for item in context.recent_messages
                if "첫 번째 문제" in item["content"]
            ))
            self.assertEqual((), context.evidence)

    def test_pipeline_result_links_once_to_its_parent_session(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store, application = build_application(root, object(), team_backend=Team())
            application.handle(incoming(1, "안녕"))
            session = store.load_conversation_session("telegram", "200")
            run_id = "RUN-S03-PIPELINE"
            RunStateMachine(store).create_run(run_id)
            store.set_conversation_task("telegram", "200", run_id)
            store.enqueue_pipeline_job(run_id, "telegram", "200")
            store.claim_next_pipeline_job("worker")
            store.finish_pipeline_job(run_id, "worker", "COMPLETED")
            with self.assertRaises(StoreError):
                store.finish_pipeline_job(run_id, "worker", "COMPLETED")
            reopened = StateStore(root / "state.db")
            results = [item for item in reopened.list_messages(session.session_run_id)
                       if item["kind"] == "pipeline_result"]
            self.assertEqual(1, len(results))
            self.assertEqual(f"pipeline:{run_id}", results[0]["data"]["source_ref"])


if __name__ == "__main__":
    unittest.main()
