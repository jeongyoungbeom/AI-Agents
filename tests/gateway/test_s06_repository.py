from __future__ import annotations

import json
import base64
import sqlite3
import subprocess
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from app.contracts import RoleId, TokenUsage
from app.gateway.core import AgentReply
from app.gateway.core.governed_backend import GovernedTeamConversationBackend
from app.services.budget import BudgetManager, BudgetPolicy
from app.gateway.repository_analysis_worker import RepositoryAnalysisWorker
from app.orchestrator import RunStateMachine
from app.services.context import ContextPolicy, ContextService
from app.services.repository import RepositoryAnalysisRequest, RepositorySnapshotEntry, RepositorySnapshotManifest, SafeRepositoryReader
from app.services.repository import protocol, reader as reader_module
from app.services.repository.analysis_plan import build_repository_analysis_plan
from app.storage import StateStore
from tests.gateway.support import temporary_directory


class FixtureReader:
    def __init__(self, files, *, dirty=False):
        self.files = files
        self.reads = []
        self.manifest = RepositorySnapshotManifest("a" * 64, "b" * 40, "main", tuple(
            RepositorySnapshotEntry(path, len(text.encode())) for path, text in files.items()),
            working_tree_dirty=dirty)

    def pinned_manifest(self, *args, **kwargs):
        return self.manifest

    def read_pinned_ranges(self, _root, _manifest, paths, *, start_lines, max_chunk_bytes, **kwargs):
        rows = []
        for path in paths:
            start = start_lines.get(path, 1)
            self.reads.append((path, start))
            lines = self.files[path].splitlines(keepends=True)
            selected = []
            for line in lines[start - 1:]:
                if len("".join(selected).encode()) + len(line.encode()) > max_chunk_bytes:
                    break
                selected.append(line)
            end = start + len(selected) - 1
            rows.append(dict(path=path, content="".join(selected), start_line=start, end_line=end,
                             total_lines=len(lines), next_line=end + 1 if end < len(lines) else 0))
        return tuple(rows), ()


class EvidenceTeam:
    def __init__(self, *, next_paths=(), malformed=False, missing=False):
        self.contexts = []
        self.next_paths = next_paths
        self.malformed = malformed
        self.missing = missing

    def respond_as(self, _state, context, message, role, **kwargs):
        data = context.repository_context
        self.contexts.append(data)
        if data["phase"] == "SYNTHESIS":
            return AgentReply("고정 근거로 종합했습니다. " + " ".join(item["summary"] for item in data["evidence"]))
        if self.malformed:
            return AgentReply("JSON 형식 없는 응답")
        findings = []
        for item in data["documents"][:1 if self.missing else None]:
            tail = next((index for index, line in enumerate(item["content"].splitlines()) if "TAIL_DEFECT" in line), None)
            start = item["start_line"] + tail if tail is not None else item["start_line"]
            findings.append(dict(path=item["path"], start_line=start, end_line=start,
                                 summary="TAIL_DEFECT: 0으로 나누는 결함" if tail is not None else "이 행 조각의 입력과 호출 흐름을 확인했습니다."))
        return AgentReply(json.dumps(dict(findings=findings, next_paths=list(self.next_paths),
                                         open_questions=["호출부와 테스트의 연결 확인"]), ensure_ascii=False))


class S06RepositoryTests(unittest.TestCase):
    def setup_job(self, root, reader, team, *, full=True, **settings):
        store = StateStore(root / "state.db")
        RunStateMachine(store).create_run("parent")
        store.create_conversation_session("telegram", "200", "100", "parent", "review")
        request = RepositoryAnalysisRequest.create(channel="telegram", conversation_id="200", user_id="100",
            source_message_id="1", role_id=RoleId.REVIEW.value, request_text="전체 코드 감사" if full else "프로젝트 구조 조사",
            repository_path=str(root), repository_identity=reader.manifest.identity_hash,
            commit_sha=reader.manifest.commit_sha, branch="main")
        RunStateMachine(store).create_run(request.analysis_id)
        store.create_repository_analysis(request)
        worker = RepositoryAnalysisWorker(store, reader, team, ContextService(store), **settings)
        return store, request.analysis_id, worker

    def drain(self, store, analysis_id, worker):
        for _ in range(100):
            if not worker.run_once():
                break
        job = store.repository_analysis(analysis_id)
        self.assertIn(job["status"], {"COMPLETED", "PARTIAL_COMPLETED"})
        return job

    def test_real_git_large_file_tail_is_grounded_in_absolute_lines(self):
        for size in (13_000, 60_000):
            with self.subTest(size=size), temporary_directory() as directory:
                root = Path(directory)
                repo = root / "source"
                repo.mkdir()
                text = "value = 1  # " + "x" * 65 + "\n"
                text = text * (size // len(text) + 1) + "return value // 0  # TAIL_DEFECT\n"
                (repo / "main.py").write_text(text, encoding="utf-8")
                for args in (("init",), ("add", "main.py"), ("-c", "user.name=fixture", "-c", "user.email=fixture@local", "commit", "-m", "fixture")):
                    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
                sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"]).decode().strip()
                oid = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD:main.py"]).decode().strip()
                fixture = FixtureReader({"main.py": text})
                fixture.manifest = RepositorySnapshotManifest("a" * 64, sha, "main", (RepositorySnapshotEntry("main.py", len(text.encode()), oid),))
                actual = SafeRepositoryReader()
                fixture.read_pinned_ranges = actual.read_pinned_ranges
                team = EvidenceTeam()
                store, analysis_id, worker = self.setup_job(root, fixture, team, max_files_per_batch=1)

                def host_git(arguments, *, maximum, input_data=None, **kwargs):
                    result = subprocess.run(["git", "--no-replace-objects", *arguments], cwd=repo,
                        input=input_data, capture_output=True, check=True)
                    self.assertLessEqual(len(result.stdout), maximum)
                    return result.stdout

                def host_protocol(_root, config, **kwargs):
                    with patch.object(protocol, "_git", side_effect=host_git):
                        return protocol._pinned_read(config, {"identity_hash": "a" * 64})

                with patch.object(reader_module, "_repository_protocol", side_effect=host_protocol):
                    job = self.drain(store, analysis_id, worker)
                evidence = store.repository_analysis_evidence(analysis_id)
                tail = [item for item in evidence if "TAIL_DEFECT" in item["summary"]]
                self.assertEqual("COMPLETED", job["status"])
                self.assertEqual(1, len(tail))
                self.assertEqual(len(text.splitlines()), tail[0]["start_line"])
                self.assertEqual(sha, tail[0]["commit_sha"])
                reads = [item for item in evidence if item["kind"] == "source_read"]
                self.assertGreater(len(reads), 1)
                self.assertEqual(1, reads[0]["start_line"])
                self.assertEqual(len(text.splitlines()), reads[-1]["end_line"])
                self.assertTrue(all(left["end_line"] + 1 == right["start_line"] for left, right in zip(reads, reads[1:])))
                followup = ContextService(store).build("parent", conversation_key="telegram:200", user_id="100")
                self.assertTrue(any("TAIL_DEFECT" in item["content"] for item in followup.recent_messages))

    def test_full_plan_preserves_more_than_forty_batches_and_continues(self):
        with temporary_directory() as directory:
            reader = FixtureReader({f"app/file_{i:03}.py": "value = 1\n" for i in range(125)})
            team = EvidenceTeam()
            store, aid, worker = self.setup_job(Path(directory), reader, team, max_batches=50)
            job = self.drain(store, aid, worker)
            self.assertEqual("COMPLETED", job["status"])
            self.assertGreater(len(job["completed"]), 40)
            self.assertEqual(125, len(reader.reads))
            self.assertTrue(all(context["evidence"] for context in team.contexts[1:]))
            self.assertTrue(team.contexts[1]["open_questions"])

    def test_default_batch_limit_preserves_precise_unread_scope(self):
        with temporary_directory() as directory:
            reader = FixtureReader({f"app/file_{i:03}.py": "value = 1\n" for i in range(125)})
            store, aid, worker = self.setup_job(Path(directory), reader, EvidenceTeam())
            job = self.drain(store, aid, worker)
            self.assertEqual("BATCH_LIMIT", job["stop_reason"])
            self.assertEqual(117, len(reader.reads))
            unread = {path for batch in job["remaining"] for path in batch["paths"]}
            self.assertEqual(8, len(unread))
            self.assertIn("app/file_124.py", unread)
            self.assertIn("미처리 8개", job["final_response"])
            self.assertIn("종합 요약 선택", job["final_response"])

    def test_adaptive_evidence_selects_previously_unselected_caller(self):
        with temporary_directory() as directory:
            reader = FixtureReader({"main.py": "from zz.dependency import call\n", **{
                f"app/file_{i:03}.py": "value = 1\n" for i in range(130)}, "zz/dependency.py": "def call(): pass\n"})
            team = EvidenceTeam(next_paths=("zz/dependency.py", "../../secret.env"))
            store, aid, worker = self.setup_job(Path(directory), reader, team, full=False, max_files_per_batch=1)
            worker.run_once()
            self.assertFalse(next(item for item in store.repository_analysis(aid)["plan"]["files"] if item["path"] == "zz/dependency.py")["selected"])
            worker.run_once()
            self.assertEqual(["zz/dependency.py"], store.repository_analysis(aid)["remaining"][0]["paths"])
            worker.run_once()
            self.assertEqual(["main.py", "zz/dependency.py"], [item[0] for item in reader.reads])
            self.assertTrue(team.contexts[1]["evidence"])
            self.assertEqual(["호출부와 테스트의 연결 확인"], team.contexts[1]["open_questions"])

    def test_restart_resumes_next_line_without_duplicate_checkpoint(self):
        with temporary_directory() as directory:
            root = Path(directory)
            reader = FixtureReader({"main.py": "value = 1 # comment\n" * 800})
            team = EvidenceTeam()
            store, aid, worker = self.setup_job(root, reader, team, max_files_per_batch=1)
            worker.run_once()
            worker.run_once()
            next_line = store.repository_analysis(aid)["remaining"][0]["start_lines"]["main.py"]
            restarted = RepositoryAnalysisWorker(StateStore(root / "state.db"), reader, team, ContextService(store), max_files_per_batch=1)
            self.drain(store, aid, restarted)
            self.assertEqual(next_line, reader.reads[1][1])
            self.assertEqual(len(reader.reads), len(set(reader.reads)))
            self.assertEqual(len(reader.reads), sum(item["kind"] == "source_read" for item in store.repository_analysis_evidence(aid)))

    def test_checkpoint_failure_rolls_back_evidence_and_continuations(self):
        with temporary_directory() as directory:
            reader = FixtureReader({"main.py": "value = 1 # comment\n" * 800})
            store, aid, worker = self.setup_job(Path(directory), reader, EvidenceTeam(), max_files_per_batch=1)
            worker.run_once()
            job = store.claim_next_repository_analysis(worker.instance_id)
            before = store.repository_analysis(aid)
            with sqlite3.connect(store.path) as connection:
                connection.execute("CREATE TRIGGER fail_evidence BEFORE INSERT ON repository_analysis_evidence BEGIN SELECT RAISE(FAIL, 'injected'); END")
            with self.assertRaisesRegex(sqlite3.IntegrityError, "injected"):
                worker._process(job, threading.Event())
            after = store.repository_analysis(aid)
            self.assertEqual(before["remaining"], after["remaining"])
            self.assertEqual(before["checkpoint"], after["checkpoint"])
            self.assertEqual([], store.repository_analysis_evidence(aid))
            self.assertEqual(before["remaining"][0], store.next_repository_analysis_batch(aid)["target"])

    def test_malformed_reply_is_read_only_and_stops_without_progress(self):
        with temporary_directory() as directory:
            reader = FixtureReader({f"app/file_{i}.py": "value = 1\n" for i in range(4)})
            store, aid, worker = self.setup_job(Path(directory), reader, EvidenceTeam(malformed=True), max_files_per_batch=1)
            job = self.drain(store, aid, worker)
            self.assertEqual("NO_PROGRESS", job["stop_reason"])
            evidence = store.repository_analysis_evidence(aid)
            self.assertEqual(2, len(evidence))
            self.assertTrue(all(item["kind"] == "source_read" for item in evidence))
            self.assertIn("분석 결과 파일 0개", job["final_response"])

    def test_missing_file_result_is_partial_with_separate_read_count(self):
        with temporary_directory() as directory:
            reader = FixtureReader({"a.py": "value = 1\n", "b.py": "value = 2\n"})
            store, aid, worker = self.setup_job(Path(directory), reader, EvidenceTeam(missing=True))
            job = self.drain(store, aid, worker)
            self.assertEqual("UNATTRIBUTED_ANALYSIS", job["stop_reason"])
            self.assertIn("읽은 파일 2개/2개 행 조각 · 분석 결과 파일 1개", job["final_response"])

    def test_range_validation_rejects_unread_lines_and_malformed_fields(self):
        with temporary_directory() as directory:
            store, aid, worker = self.setup_job(Path(directory), FixtureReader({"main.py": "ok\n"}), EvidenceTeam())
            payload = {"findings": [dict(path="main.py", start_line=1, end_line=1, summary="unread"),
                dict(path="main.py", start_line=True, end_line=20, summary="bool"),
                dict(path="main.py", start_line=20, end_line=21, summary={"invalid": "object"}),
                dict(path="main.py", start_line=20, end_line=21, summary="valid") ]}
            found = worker._validated_batch_findings(json.dumps(payload), (("main.py", "a\nb\n"),),
                                                    (dict(path="main.py", start_line=20, end_line=21),))
            self.assertEqual(["valid"], [item["summary"] for item in found])

    def test_synthesis_context_includes_tail_with_bounded_json_and_omission_count(self):
        with temporary_directory() as directory:
            store, aid, worker = self.setup_job(Path(directory), FixtureReader({"main.py": "ok\n"}), EvidenceTeam())
            evidence = [dict(path=f"app/file_{i:03}.py", start_line=1, end_line=10, phase="CORE",
                             kind="batch_analysis", summary="한글 근거" * 100) for i in range(500)]
            selected = worker._synthesis_evidence_context(evidence, max_bytes=4000)
            self.assertLessEqual(len(json.dumps(selected, ensure_ascii=False).encode()), 4000)
            self.assertIn("app/file_499.py", [item["path"] for item in selected])
            self.assertLess(len(selected), len(evidence))

    def test_dirty_snapshot_and_scan_exclusions_are_disclosed(self):
        with temporary_directory() as directory:
            reader = FixtureReader({"main.py": "ok\n"}, dirty=True)
            reader.manifest = RepositorySnapshotManifest("a" * 64, "b" * 40, "main",
                (*reader.manifest.entries, RepositorySnapshotEntry("huge.py", 5 * 1024 * 1024),
                 RepositorySnapshotEntry("logo.bin", 10)), working_tree_dirty=True)
            store, aid, worker = self.setup_job(Path(directory), reader, EvidenceTeam())
            job = self.drain(store, aid, worker)
            self.assertEqual("SCAN_SIZE_LIMIT", job["stop_reason"])
            self.assertIn("huge.py: scan_size_limit", job["final_response"])
            self.assertIn("logo.bin: binary_extension", job["final_response"])
            self.assertIn("미커밋 변경", job["final_response"])
            self.assertIn("b" * 40, job["final_response"])

    def test_budget_limit_uses_reserved_synthesis_and_settles_once(self):
        class UsageTeam(EvidenceTeam):
            def prompt_token_upper_bound(self, *args, **kwargs):
                return 10

            def respond_as(self, *args, **kwargs):
                reply = super().respond_as(*args, **kwargs)
                return AgentReply(reply.text, usage=TokenUsage(10, 5))

        with temporary_directory() as directory:
            reader = FixtureReader({f"app/file_{i}.py": "value = 1\n" for i in range(3)})
            delegate = UsageTeam()
            store, aid, worker = self.setup_job(Path(directory), reader, delegate, max_files_per_batch=1)
            budget = BudgetManager(BudgetPolicy(calibration_mode=False, whole_task_tokens=40, completion_reserve_tokens=15), store)
            worker.team_backend = GovernedTeamConversationBackend(delegate, budget, response_reserve_tokens=5)
            job = self.drain(store, aid, worker)
            self.assertEqual("BUDGET_LIMIT", job["stop_reason"])
            self.assertEqual(2, len(delegate.contexts))
            self.assertEqual("SYNTHESIS", delegate.contexts[-1]["phase"])
            self.assertEqual(30, store.usage_total(aid))
            self.assertEqual(0, store.reserved_token_total(aid))
            self.assertEqual(1, sum(item["kind"] == "source_read" for item in store.repository_analysis_evidence(aid)))
            self.assertFalse(worker.run_once())
            self.assertEqual(30, store.usage_total(aid))

    def test_protocol_unicode_chunks_and_oversize_line_exclusion(self):
        data = "한글\n다음\n끝\n".encode()
        config = dict(expected_identity="a" * 64, commit_sha="b" * 40, chunk_bytes=8,
                      max_git_bytes=1024, policy=dict(sensitive_parts=[], sensitive_suffixes=[]),
                      entries=[dict(path="main.py", size=len(data), object_id="c" * 40, start_line=1)])
        raw = f"{'c' * 40} blob {len(data)}\n".encode() + data + b"\n"
        with patch.object(protocol, "_git", side_effect=lambda args, **kwargs: raw if args[-1] == "--batch" else b""):
            first = protocol._pinned_read(config, {"identity_hash": "a" * 64})["results"][0]
            self.assertEqual("한글\n", base64.b64decode(first["data"]).decode())
            self.assertEqual(2, first["next_line"])
            config["chunk_bytes"] = 4
            excluded = protocol._pinned_read(config, {"identity_hash": "a" * 64})["results"][0]
            self.assertEqual("line_size_limit", excluded["reason"])

    def test_reader_rejects_forged_snapshot_and_range_metadata(self):
        with temporary_directory() as directory:
            root = Path(directory)
            reader = SafeRepositoryReader()
            manifest = RepositorySnapshotManifest("a" * 64, "b" * 40, "main", (RepositorySnapshotEntry("main.py", 3),))
            row = dict(path="main.py", status="ok", data=base64.b64encode(b"ok\n").decode(),
                       start_line=2, end_line=2, total_lines=2, next_line=0)
            payload = dict(identity_hash="a" * 64, commit_sha="b" * 40, results=[row])
            with patch.object(reader_module, "_repository_protocol", return_value=payload):
                with self.assertRaisesRegex(reader_module.RepositoryAccessError, "손상"):
                    reader.read_pinned_ranges(root, manifest, ("main.py",), start_lines={}, max_chunk_bytes=10, max_scan_bytes=10)
            payload["commit_sha"] = "c" * 40
            with patch.object(reader_module, "_repository_protocol", return_value=payload):
                with self.assertRaises(reader_module.RepositoryPinnedSnapshotUnavailable):
                    reader.read_pinned_ranges(root, manifest, ("main.py",), start_lines={}, max_chunk_bytes=10, max_scan_bytes=10)

    def test_partial_evidence_locations_keep_unanalyzed_ranges_of_same_file(self):
        evidence = [dict(path="main.py", start_line=1, end_line=10, kind="source_read", summary="read"),
                    dict(path="main.py", start_line=11, end_line=20, kind="source_read", summary="read"),
                    dict(path="main.py", start_line=2, end_line=2, kind="batch_analysis", summary="finding")]
        locations = RepositoryAnalysisWorker._evidence_locations(evidence)
        self.assertIn("main.py:11-20 — 읽기만 확인됨", locations)
        self.assertIn("main.py:2-2 — finding", locations)

    def test_entrypoint_is_read_before_related_tests_and_remaining_sources(self):
        reader = FixtureReader({"README.md": "docs\n", "app/main.py": "entry\n",
                                "tests/test_main.py": "test\n", "app/other.py": "other\n"})
        plan = build_repository_analysis_plan(reader.manifest, "프로젝트 구조", max_files_per_batch=1, max_file_bytes=1000)
        self.assertEqual(["README.md", "app/main.py", "tests/test_main.py", "app/other.py"],
                         [path for batch in plan["batches"] for path in batch["paths"]])


if __name__ == "__main__":
    unittest.main()
