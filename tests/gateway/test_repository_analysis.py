from __future__ import annotations

import unittest
import sqlite3
from pathlib import Path

from app.contracts import RoleId
from app.gateway.core import AgentReply
from app.gateway.repository_analysis_worker import RepositoryAnalysisQueue
from app.gateway.repository_analysis_worker import RepositoryAnalysisWorker
from app.orchestrator import RunStateMachine
from app.services.context import ContextPolicy, ContextService
from app.services.repository import (
    RepositoryAnalysisRequest,
    RepositoryAnalysisStatus,
    RepositorySnapshotEntry,
    RepositorySnapshotManifest,
    build_repository_analysis_plan,
    is_long_repository_analysis_request,
)
from app.storage import StateStore

from tests.gateway.support import build_application, future_expiry, temporary_directory
from tests.gateway.test_conversation_foundation import incoming


class RepositoryAnalysisPlanTests(unittest.TestCase):
    def test_full_audit_plans_every_eligible_file_before_worker_limits(self):
        entries = tuple(
            RepositorySnapshotEntry(f"app/module_{index:03}.py", 100)
            for index in range(140)
        ) + (
            RepositorySnapshotEntry("scripts/chat-gateway.ps1", 100),
            RepositorySnapshotEntry("assets/logo.bin", 100),
        )
        plan = build_repository_analysis_plan(
            RepositorySnapshotManifest("a" * 64, "b" * 40, "main", entries),
            "전체 코드 분석해줘", max_files_per_batch=3,
            max_file_bytes=1000,
        )
        self.assertEqual("full", plan["mode"])
        self.assertEqual(141, sum(item["selected"] for item in plan["files"]))
        self.assertEqual(0, len(plan["not_selected"]))
        self.assertEqual(1, plan["excluded"]["binary_extension"])
        self.assertGreater(len(plan["batches"]), 40)

    def test_unselected_source_files_make_plan_partial_and_include_script_languages(self):
        entries = [RepositorySnapshotEntry(f"app/module_{index:03}.py", 100) for index in range(140)]
        entries += [
            RepositorySnapshotEntry("scripts/chat-gateway.ps1", 100),
            RepositorySnapshotEntry("scripts/build.sh", 100),
            RepositorySnapshotEntry("db/schema.sql", 100),
            RepositorySnapshotEntry("web/index.html", 100),
        ]
        manifest = RepositorySnapshotManifest("a" * 64, "b" * 40, "main", tuple(entries))
        plan = build_repository_analysis_plan(
            manifest, "프로젝트 전반 구조를 분석해줘", max_files_per_batch=3,
            max_file_bytes=1000, adaptive_file_limit=117,
        )
        self.assertEqual(117, sum(item["selected"] for item in plan["files"]))
        self.assertEqual(27, len(plan["not_selected"]))
        self.assertIn("SELECTION_LIMIT", plan["partial_reasons"])
        for suffix in (".ps1", ".sh", ".sql", ".html"):
            self.assertEqual("source", next(item["category"] for item in plan["files"] if item["path"].endswith(suffix)))

    def test_plan_respects_batch_bytes_and_includes_configuration_and_documentation(self):
        manifest = RepositorySnapshotManifest(
            "a" * 64,
            "b" * 40,
            "main",
            (
                RepositorySnapshotEntry("README.md", 1024),
                RepositorySnapshotEntry("config/app.yaml", 4096),
                RepositorySnapshotEntry("docs/guide.md", 4096),
                RepositorySnapshotEntry("app/a.py", 40 * 1024),
                RepositorySnapshotEntry("app/b.py", 40 * 1024),
                RepositorySnapshotEntry("app/c.py", 40 * 1024),
                RepositorySnapshotEntry("assets/logo.bin", 12),
            ),
        )

        plan = build_repository_analysis_plan(
            manifest,
            "전체 테스트 전략",
            max_files_per_batch=3,
            max_file_bytes=48 * 1024,
            max_batch_bytes=96 * 1024,
        )

        analysis_batches = [
            batch for batch in plan["batches"] if batch["phase"] != "SYNTHESIS"
        ]
        self.assertTrue(all(batch["estimated_bytes"] <= 96 * 1024 for batch in analysis_batches))
        planned_paths = {path for batch in analysis_batches for path in batch["paths"]}
        self.assertTrue({"README.md", "config/app.yaml", "docs/guide.md"} <= planned_paths)
        source_batches = [batch for batch in analysis_batches if "source" in batch["categories"]]
        self.assertEqual([2, 1], [len(batch["paths"]) for batch in source_batches])
        self.assertEqual(
            "category_not_selected",
            next(item["reason"] for item in plan["not_selected"] if item["path"] == "assets/logo.bin"),
        )

    def test_plan_marks_files_that_cannot_fit_context_instead_of_scheduling_them(self):
        manifest = RepositorySnapshotManifest(
            "a" * 64,
            "b" * 40,
            "main",
            (RepositorySnapshotEntry("app/large.py", 26_000),),
        )

        plan = build_repository_analysis_plan(
            manifest,
            "전체 코드",
            max_files_per_batch=3,
            max_file_bytes=48 * 1024,
            max_batch_bytes=20_000,
            max_context_bytes=20_000,
        )

        self.assertFalse(plan["files"][0]["eligible"])
        self.assertEqual("context_size_limit", plan["files"][0]["exclude_reason"])
        self.assertIn("CONTEXT_LIMIT", plan["partial_reasons"])
        self.assertNotIn(
            "app/large.py", [path for batch in plan["batches"] for path in batch["paths"]]
        )


class RepositoryAnalysisRequestTests(unittest.TestCase):
    def test_final_evidence_lists_each_analyzed_file_beyond_twenty(self):
        evidence = [
            {"path": f"app/file_{index:02}.py", "start_line": 1,
             "end_line": 2, "kind": "batch_analysis",
             "summary": f"파일 {index} 결론"}
            for index in range(25)
        ]
        locations = RepositoryAnalysisWorker._evidence_locations(evidence)
        self.assertEqual(25, len(locations.splitlines()))
        self.assertIn("app/file_24.py:1-2 — 파일 24 결론", locations)

    def test_unreadable_file_is_excluded_without_stopping_other_batches(self):
        class Reader:
            def pinned_manifest(self, *_args, **_kwargs):
                return RepositorySnapshotManifest(
                    "a" * 64, "b" * 40, "main",
                    (RepositorySnapshotEntry("app/bad.py", 4), RepositorySnapshotEntry("app/good.py", 4)),
                )

            def read_pinned_files_with_exclusions(self, _path, _manifest, paths, **_kwargs):
                documents = tuple((path, "good") for path in paths if path == "app/good.py")
                excluded = tuple((path, "non_utf8") for path in paths if path == "app/bad.py")
                return documents, excluded

        class Team:
            def preflight(self, *_args):
                pass

            def respond_as(self, *_args, **_kwargs):
                return AgentReply("확인한 코드의 역할을 요약했습니다.")

        with temporary_directory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.db")
            request = RepositoryAnalysisRequest.create(
                channel="telegram", conversation_id="200", user_id="100",
                source_message_id="1", role_id=RoleId.DEVELOPMENT.value,
                request_text="전체 코드 분석해줘", repository_path=str(root),
                repository_identity="a" * 64, commit_sha="b" * 40, branch="main",
            )
            RunStateMachine(store).create_run(request.analysis_id)
            store.create_repository_analysis(request)
            worker = RepositoryAnalysisWorker(
                store, Reader(), Team(), ContextService(store),
                max_files_per_batch=2, max_file_bytes=1024,
                max_batch_bytes=2048, max_batches=8,
                max_query_rounds=8, max_read_bytes=4096,
            )
            for _ in range(8):
                if not worker.run_once():
                    break
            analysis = store.repository_analysis(request.analysis_id)
            self.assertIsNotNone(analysis)
            self.assertEqual("PARTIAL_COMPLETED", analysis["status"])
            self.assertEqual("UNREADABLE_FILE", analysis["stop_reason"])
            self.assertEqual("non_utf8", next(item["exclude_reason"] for item in analysis["plan"]["files"] if item["path"] == "app/bad.py"))
            self.assertIn("app/good.py", [item["path"] for item in store.repository_analysis_evidence(request.analysis_id)])

    def test_batch_findings_are_validated_against_each_read_file(self):
        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            worker = RepositoryAnalysisWorker(
                store, object(), object(), ContextService(store)
            )
            findings = worker._validated_batch_findings(
                '{"findings":['
                '{"path":"app/a.py","start_line":2,"end_line":2,"summary":"A 결론"},'
                '{"path":"app/b.py","start_line":1,"end_line":1,"summary":"B 결론"},'
                '{"path":"app/missing.py","start_line":1,"end_line":1,"summary":"추측"},'
                '{"path":"app/b.py","start_line":9,"end_line":9,"summary":"범위 밖"}]}' ,
                (("app/a.py", "one\ntwo\n"), ("app/b.py", "only\n")),
            )
            self.assertEqual(["app/a.py", "app/b.py"], [item["path"] for item in findings])
            self.assertEqual([2, 1], [item["start_line"] for item in findings])

    def test_explanation_of_full_analysis_does_not_start_analysis(self):
        class Team:
            def __init__(self):
                self.calls = 0

            def preflight(self, *_args):
                pass

            def respond_as(self, *_args, **_kwargs):
                self.calls += 1
                return AgentReply("분석 등록 기준을 설명합니다")

        with temporary_directory() as directory:
            root = Path(directory)
            team = Team()
            store, application = build_application(root, object(), team_backend=team)
            application.router.repository_analysis_scheduler = RepositoryAnalysisQueue(store)
            response = application.handle(incoming(1, "전체코드 분석이 왜 장기 저장소 분석으로 들어가는지 말해봐"))
            self.assertIn("분석 등록 기준", response[0].text)
            self.assertEqual(1, team.calls)
            self.assertIsNone(store.repository_analysis_summary("telegram", "200"))

    def test_status_first_line_reports_processing_analysis(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store, application = build_application(root, object())
            application.router.repository_analysis_scheduler = RepositoryAnalysisQueue(store)
            application.handle(incoming(1, "안녕"))
            request = RepositoryAnalysisRequest.create(
                channel="telegram", conversation_id="200", user_id="100",
                source_message_id="2", role_id=RoleId.DEVELOPMENT.value,
                request_text="전체 코드 분석해줘", repository_path=str(root),
                repository_identity="a" * 64, commit_sha="b" * 40, branch="main",
            )
            RunStateMachine(store).create_run(request.analysis_id)
            store.create_repository_analysis(request)
            store.claim_next_repository_analysis("test-worker")
            status = application.handle(incoming(3, "상태"))
            self.assertTrue(status[0].text.startswith("진행 중: 장기 저장소 분석"))
            self.assertIn(request.analysis_id, status[0].text.splitlines()[0])

    def test_detector_separates_broad_requests_from_an_ordinary_file_question(self):
        self.assertTrue(is_long_repository_analysis_request("전체 코드와 테스트 전략을 확인해줘"))
        self.assertTrue(is_long_repository_analysis_request("프로젝트 전반 구조를 분석해줘"))
        self.assertFalse(is_long_repository_analysis_request("AuthService 클래스만 설명해줘"))
        self.assertFalse(is_long_repository_analysis_request("전체코드 분석이 왜 장기 저장소 분석으로 들어가는지 말해봐"))
        self.assertFalse(is_long_repository_analysis_request("전체 코드를 수정해"))

    def test_store_allows_only_one_active_analysis_per_conversation(self):
        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            request = RepositoryAnalysisRequest.create(
                channel="telegram",
                conversation_id="200",
                user_id="100",
                source_message_id="1",
                role_id="development",
                request_text="전체 코드 확인",
                repository_path=str(Path(directory)),
                repository_identity="a" * 64,
                commit_sha="b" * 40,
                branch="main",
            )
            first, created = store.create_repository_analysis(request)
            second, created_again = store.create_repository_analysis(
                RepositoryAnalysisRequest.create(
                    channel="telegram",
                    conversation_id="200",
                    user_id="100",
                    source_message_id="2",
                    role_id="development",
                    request_text="프로젝트 전반 분석",
                    repository_path=str(Path(directory)),
                    repository_identity="a" * 64,
                    commit_sha="b" * 40,
                    branch="main",
                )
            )

            self.assertTrue(created)
            self.assertFalse(created_again)
            self.assertEqual(first["analysis_id"], second["analysis_id"])
            self.assertEqual(RepositoryAnalysisStatus.QUEUED.value, first["status"])

    def test_new_work_can_supersede_a_not_started_analysis(self):
        with temporary_directory() as directory:
            store = StateStore(Path(directory) / "state.db")
            request = RepositoryAnalysisRequest.create(
                channel="telegram",
                conversation_id="200",
                user_id="100",
                source_message_id="1",
                role_id="development",
                request_text="전체 코드 확인",
                repository_path=str(Path(directory)),
                repository_identity="a" * 64,
                commit_sha="b" * 40,
                branch="main",
            )
            store.create_repository_analysis(request)

            replaced = store.supersede_repository_analysis("telegram", "200")

            self.assertIsNotNone(replaced)
            assert replaced is not None
            self.assertEqual("SUPERSEDED", replaced["status"])
            _next, created = store.create_repository_analysis(
                RepositoryAnalysisRequest.create(
                    channel="telegram",
                    conversation_id="200",
                    user_id="100",
                    source_message_id="2",
                    role_id="development",
                    request_text="다른 범위의 전체 코드 확인",
                    repository_path=str(Path(directory)),
                    repository_identity="a" * 64,
                    commit_sha="b" * 40,
                    branch="main",
                )
            )
            self.assertTrue(created)

    def test_broad_approved_request_is_registered_without_calling_chat_backend(self):
        class Reader:
            def pinned_manifest(self, *_args, **_kwargs):
                return RepositorySnapshotManifest(
                    "a" * 64, "b" * 40, "main",
                    (RepositorySnapshotEntry("app/main.py", 100),),
                )

        with temporary_directory() as directory:
            root = Path(directory)
            store, application = build_application(
                root,
                object(),
            )
            application.router.repository_analysis_scheduler = RepositoryAnalysisQueue(store)
            application.router.repository_reader = Reader()
            application.handle(incoming(1, str(root / "selected-repository")))
            store.set_current_project(
                "telegram",
                "200",
                "100",
                str(root / "selected-repository"),
                repository_identity="a" * 64,
                head_sha="b" * 40,
                approved=True,
                approval_expires_at=future_expiry(),
            )

            proposal = application.handle(incoming(2, "전체 코드와 전체 테스트 전략을 분석해줘"))
            self.assertIn("전체 코드 감사 제안", proposal[0].text)
            self.assertIsNone(store.repository_analysis_summary("telegram", "200"))
            reply = application.handle(incoming(3, "전체 감사 시작해"))

            self.assertIn("장기 저장소 분석을 등록", reply[0].text)
            analysis = store.repository_analysis_summary("telegram", "200")
            self.assertIsNotNone(analysis)
            assert analysis is not None
            self.assertEqual("b" * 40, analysis["commit_sha"])
            self.assertEqual(0, analysis["checkpoint"])

            replacement = application.handle(incoming(4, "새 작업"))

            self.assertIn("새 작업 대화를 시작", replacement[0].text)
            replaced = store.repository_analysis(analysis["analysis_id"])
            self.assertIsNotNone(replaced)
            assert replaced is not None
            self.assertEqual("SUPERSEDED", replaced["status"])

    def test_worker_checkpoints_each_batch_then_delivers_one_final_synthesis(self):
        class Reader:
            def __init__(self):
                self.manifest_calls = 0

            def pinned_manifest(self, *_args, **_kwargs):
                self.manifest_calls += 1
                return RepositorySnapshotManifest(
                    "a" * 64,
                    "b" * 40,
                    "main",
                    (
                        RepositorySnapshotEntry("README.md", 20),
                        RepositorySnapshotEntry("app/main.py", 20),
                    ),
                )

            def read_pinned_files(self, _path, _manifest, paths, **_kwargs):
                return tuple((path, f"{path}\nsource") for path in paths)

        class Team:
            supports_cancellation = False

            def __init__(self):
                self.calls = []

            def preflight(self, *_args):
                return None

            def respond_as(self, _state, _context, message, role_id, **_kwargs):
                self.calls.append((message.text, role_id, _context.repository_context))
                return AgentReply("확인한 파일의 역할과 테스트 관점을 요약했습니다.")

        with temporary_directory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.db")
            request = RepositoryAnalysisRequest.create(
                channel="telegram",
                conversation_id="200",
                user_id="100",
                source_message_id="1",
                role_id=RoleId.DEVELOPMENT.value,
                request_text="전체 코드와 테스트 전략을 분석해줘",
                repository_path=str(root),
                repository_identity="a" * 64,
                commit_sha="b" * 40,
                branch="main",
            )
            RunStateMachine(store).create_run(
                request.analysis_id,
                repository=request.repository_path,
                repository_identity=request.repository_identity,
                repository_head_sha=request.commit_sha,
                repository_approved=True,
            )
            store.create_repository_analysis(request)
            team = Team()
            reader = Reader()
            worker = RepositoryAnalysisWorker(
                store,
                reader,
                team,
                ContextService(store),
                max_files_per_batch=1,
                max_file_bytes=1024,
                max_batch_bytes=2048,
                max_batches=8,
                max_query_rounds=8,
                max_read_bytes=4096,
            )

            while worker.run_once():
                if store.repository_analysis(request.analysis_id)["status"] == "COMPLETED":
                    break

            analysis = store.repository_analysis(request.analysis_id)
            self.assertIsNotNone(analysis)
            assert analysis is not None
            self.assertEqual("COMPLETED", analysis["status"])
            self.assertEqual(3, analysis["checkpoint"])
            self.assertEqual(3, analysis["model_calls"])
            self.assertEqual(3, len(team.calls))
            self.assertEqual(1, reader.manifest_calls)
            self.assertEqual("SYNTHESIS", analysis["phase"])
            self.assertGreaterEqual(len(store.repository_analysis_evidence(request.analysis_id)), 3)
            self.assertEqual([], team.calls[-1][2]["documents"])
            self.assertTrue(
                all(
                    len(item["summary"]) <= 800
                    for item in team.calls[-1][2]["evidence"]
                )
            )
            delivered = store.deliverable_outbound("telegram")
            self.assertTrue(any("장기 저장소 분석 · 완료" in item["text"] for item in delivered))

    def test_restart_marks_an_inflight_model_call_for_attention_without_replaying_it(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.db")
            request = RepositoryAnalysisRequest.create(
                channel="telegram",
                conversation_id="200",
                user_id="100",
                source_message_id="1",
                role_id=RoleId.DEVELOPMENT.value,
                request_text="전체 코드 확인",
                repository_path=str(root),
                repository_identity="a" * 64,
                commit_sha="b" * 40,
                branch="main",
            )
            RunStateMachine(store).create_run(request.analysis_id)
            store.create_repository_analysis(request)
            claimed = store.claim_next_repository_analysis("crashed-worker", lease_seconds=1)
            self.assertIsNotNone(claimed)
            store.set_repository_analysis_model_call_state(
                request.analysis_id, "crashed-worker", "STARTED"
            )
            connection = sqlite3.connect(root / "state.db")
            try:
                connection.execute(
                    "UPDATE repository_analysis_jobs SET lease_until = ? WHERE analysis_id = ?",
                    ("2000-01-01T00:00:00+00:00", request.analysis_id),
                )
                connection.commit()
            finally:
                connection.close()

            recovered = store.recover_stale_repository_analyses()

            self.assertEqual(1, len(recovered))
            self.assertEqual("NEEDS_ATTENTION", recovered[0]["status"])
            self.assertEqual("MODEL_OUTCOME_UNKNOWN", recovered[0]["stop_reason"])
            self.assertEqual(
                "NEEDS_ATTENTION",
                store.resume_repository_analysis("telegram", "200")["status"],
            )

    def test_stale_user_stop_and_supersede_requests_are_preserved(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.db")
            request = RepositoryAnalysisRequest.create(
                channel="telegram",
                conversation_id="200",
                user_id="100",
                source_message_id="1",
                role_id=RoleId.DEVELOPMENT.value,
                request_text="전체 코드 확인",
                repository_path=str(root),
                repository_identity="a" * 64,
                commit_sha="b" * 40,
                branch="main",
            )
            store.create_repository_analysis(request)
            store.claim_next_repository_analysis("stale-worker", lease_seconds=1)
            store.request_repository_analysis_stop("telegram", "200")

            connection = sqlite3.connect(root / "state.db")
            try:
                connection.execute(
                    "UPDATE repository_analysis_jobs SET lease_until = ? WHERE analysis_id = ?",
                    ("2000-01-01T00:00:00+00:00", request.analysis_id),
                )
                connection.commit()
            finally:
                connection.close()

            recovered = store.recover_stale_repository_analyses()
            self.assertEqual("PAUSED", recovered[0]["status"])
            self.assertEqual("USER_STOPPED", recovered[0]["stop_reason"])

            store.resume_repository_analysis("telegram", "200")
            store.claim_next_repository_analysis("stale-worker", lease_seconds=1)
            store.supersede_repository_analysis("telegram", "200")
            connection = sqlite3.connect(root / "state.db")
            try:
                connection.execute(
                    "UPDATE repository_analysis_jobs SET lease_until = ? WHERE analysis_id = ?",
                    ("2000-01-01T00:00:00+00:00", request.analysis_id),
                )
                connection.commit()
            finally:
                connection.close()

            recovered = store.recover_stale_repository_analyses()
            self.assertEqual("SUPERSEDED", recovered[0]["status"])
            self.assertEqual("SUPERSEDED", recovered[0]["stop_reason"])

    def test_worker_does_not_record_evidence_when_repository_context_is_omitted(self):
        class Reader:
            def pinned_manifest(self, *_args, **_kwargs):
                return RepositorySnapshotManifest(
                    "a" * 64,
                    "b" * 40,
                    "main",
                    (RepositorySnapshotEntry("README.md", 100),),
                )

            def read_pinned_files(self, _path, _manifest, paths, **_kwargs):
                return tuple((path, "x" * 26_000) for path in paths)

        class Team:
            supports_cancellation = False

            def __init__(self):
                self.calls = 0

            def respond_as(self, *_args, **_kwargs):
                self.calls += 1
                return AgentReply("분석 결과")

        with temporary_directory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.db")
            request = RepositoryAnalysisRequest.create(
                channel="telegram",
                conversation_id="200",
                user_id="100",
                source_message_id="1",
                role_id=RoleId.DEVELOPMENT.value,
                request_text="전체 코드 확인",
                repository_path=str(root),
                repository_identity="a" * 64,
                commit_sha="b" * 40,
                branch="main",
            )
            RunStateMachine(store).create_run(request.analysis_id)
            store.create_repository_analysis(request)
            team = Team()
            worker = RepositoryAnalysisWorker(
                store,
                Reader(),
                team,
                ContextService(store, policy=ContextPolicy(max_characters=24_000)),
            )

            worker.run_once()  # save plan
            worker.run_once()  # context omission must not confirm the file

            analysis = store.repository_analysis(request.analysis_id)
            self.assertEqual("PARTIAL_COMPLETED", analysis["status"])
            self.assertEqual("CONTEXT_LIMIT", analysis["stop_reason"])
            self.assertEqual([], store.repository_analysis_evidence(request.analysis_id))
            self.assertEqual(0, team.calls)
            self.assertIn("부분 종합", store.deliverable_outbound("telegram")[-1]["text"])

    def test_batch_limit_returns_partial_summary_with_unprocessed_scope(self):
        class Reader:
            def pinned_manifest(self, *_args, **_kwargs):
                return RepositorySnapshotManifest(
                    "a" * 64,
                    "b" * 40,
                    "main",
                    (
                        RepositorySnapshotEntry("README.md", 20),
                        RepositorySnapshotEntry("pyproject.toml", 20),
                        RepositorySnapshotEntry("package.json", 20),
                        RepositorySnapshotEntry("settings.gradle", 20),
                        RepositorySnapshotEntry("config/app.yaml", 20),
                        RepositorySnapshotEntry("docs/guide.md", 20),
                        RepositorySnapshotEntry("app/main.py", 20),
                    ),
                )

            def read_pinned_files(self, _path, _manifest, paths, **_kwargs):
                return tuple((path, f"{path}\nsource") for path in paths)

        class Team:
            supports_cancellation = False

            def __init__(self):
                self.calls = []

            def respond_as(self, _state, context, _message, _role_id, **_kwargs):
                self.calls.append(context.repository_context)
                return AgentReply("저장된 근거에서 테스트 설정과 실행 흐름을 확인했습니다.")

        with temporary_directory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.db")
            request = RepositoryAnalysisRequest.create(
                channel="telegram",
                conversation_id="200",
                user_id="100",
                source_message_id="1",
                role_id=RoleId.DEVELOPMENT.value,
                request_text="전체 테스트 전략을 분석해줘",
                repository_path=str(root),
                repository_identity="a" * 64,
                commit_sha="b" * 40,
                branch="main",
            )
            RunStateMachine(store).create_run(request.analysis_id)
            store.create_repository_analysis(request)
            team = Team()
            worker = RepositoryAnalysisWorker(
                store,
                Reader(),
                team,
                ContextService(store),
                max_files_per_batch=1,
                max_file_bytes=1024,
                max_batch_bytes=2048,
                max_batches=2,
                max_query_rounds=8,
                max_read_bytes=4096,
            )

            while worker.run_once():
                analysis = store.repository_analysis(request.analysis_id)
                if analysis["status"] in {"PARTIAL_COMPLETED", "COMPLETED"}:
                    break

            analysis = store.repository_analysis(request.analysis_id)
            response = store.deliverable_outbound("telegram")[-1]["text"]
            self.assertEqual("PARTIAL_COMPLETED", analysis["status"])
            self.assertEqual("BATCH_LIMIT", analysis["stop_reason"])
            self.assertTrue(analysis["plan"]["unprocessed_paths"])
            self.assertGreaterEqual(len(team.calls), 2)
            self.assertIn("저장된 근거에서 테스트 설정", response)
            self.assertIn("계획됐지만 미처리", response)
            self.assertTrue(
                any(
                    item["path"] == "config/app.yaml" and item["selected"]
                    for item in analysis["plan"]["files"]
                )
            )
            self.assertTrue(
                any(
                    item["path"] == "docs/guide.md" and item["selected"]
                    for item in analysis["plan"]["files"]
                )
            )

    def test_elapsed_time_limit_returns_a_partial_result_without_reading_files(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.db")
            request = RepositoryAnalysisRequest.create(
                channel="telegram",
                conversation_id="200",
                user_id="100",
                source_message_id="1",
                role_id=RoleId.DEVELOPMENT.value,
                request_text="전체 코드 확인",
                repository_path=str(root),
                repository_identity="a" * 64,
                commit_sha="b" * 40,
                branch="main",
            )
            RunStateMachine(store).create_run(request.analysis_id)
            store.create_repository_analysis(request)
            connection = sqlite3.connect(root / "state.db")
            try:
                connection.execute(
                    "UPDATE repository_analysis_jobs SET created_at = ?, active_seconds = 61 WHERE analysis_id = ?",
                    ("2000-01-01T00:00:00+00:00", request.analysis_id),
                )
                connection.commit()
            finally:
                connection.close()

            worker = RepositoryAnalysisWorker(
                store,
                object(),
                object(),
                ContextService(store),
                max_elapsed_seconds=60,
            )
            self.assertTrue(worker.run_once())
            analysis = store.repository_analysis(request.analysis_id)

            self.assertIsNotNone(analysis)
            assert analysis is not None
            self.assertEqual("PARTIAL_COMPLETED", analysis["status"])
            self.assertEqual("TIME_LIMIT", analysis["stop_reason"])

    def test_paused_wall_time_does_not_exhaust_active_time(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.db")
            request = RepositoryAnalysisRequest.create(
                channel="telegram", conversation_id="200", user_id="100",
                source_message_id="1", role_id=RoleId.DEVELOPMENT.value,
                request_text="전체 코드 확인", repository_path=str(root),
                repository_identity="a" * 64, commit_sha="b" * 40, branch="main",
            )
            RunStateMachine(store).create_run(request.analysis_id)
            store.create_repository_analysis(request)
            store.claim_next_repository_analysis("first-worker")
            store.pause_repository_analysis(request.analysis_id, "first-worker")
            connection = sqlite3.connect(root / "state.db")
            try:
                connection.execute(
                    "UPDATE repository_analysis_jobs SET created_at = ? WHERE analysis_id = ?",
                    ("2000-01-01T00:00:00+00:00", request.analysis_id),
                )
                connection.commit()
            finally:
                connection.close()
            store.resume_repository_analysis("telegram", "200")
            claimed = store.claim_next_repository_analysis("second-worker")
            self.assertIsNotNone(claimed)
            worker = RepositoryAnalysisWorker(
                store, object(), object(), ContextService(store), max_elapsed_seconds=60,
            )
            self.assertFalse(worker._elapsed_limit_reached(claimed))


if __name__ == "__main__":
    unittest.main()
