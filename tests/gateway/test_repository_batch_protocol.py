from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import app.services.repository.protocol as protocol
import app.services.repository.reader as reader_module
import app.services.repository.tools as tools_module
from app.services.repository import (
    RepositoryToolRequest,
    RepositoryCancelled,
    RepositorySnapshotEntry,
    RepositorySnapshotManifest,
    SafeRepositoryReader,
    SafeRepositoryToolLayer,
)


class RepositoryBatchProtocolTests(unittest.TestCase):
    def test_protocol_retries_once_when_container_cannot_decode_stdin(self):
        malformed = b'{"ok":false,"kind":"request_decode","error":"request not received"}'
        valid = b'{"ok":true,"result":{"head_sha":"abc"}}'
        with patch.object(reader_module, "_protocol_bytes", side_effect=(malformed, valid)) as call:
            result = reader_module._repository_protocol(
                Path.cwd(), {"mode": "identity"}, maximum=1024, sandbox=None,
                operation_id=None, component="repository-reader", cancelled=None,
            )
        self.assertEqual({"head_sha": "abc"}, result)
        self.assertEqual(2, call.call_count)

        with patch.object(reader_module, "_protocol_bytes", return_value=malformed) as call:
            with self.assertRaisesRegex(reader_module.RepositoryAccessError, "request not received"):
                reader_module._repository_protocol(
                    Path.cwd(), {"mode": "identity"}, maximum=1024, sandbox=None,
                    operation_id=None, component="repository-reader", cancelled=None,
                )
        self.assertEqual(2, call.call_count)

    def test_protocol_reports_unreadable_request_without_raw_json_error(self):
        with patch.object(protocol.sys, "stdin", io.StringIO("")), patch.object(protocol.sys, "stdout", io.StringIO()) as output:
            protocol.main()
            result = json.loads(output.getvalue())
        self.assertEqual("request_decode", result["kind"])
        self.assertNotIn("JSONDecodeError", result["error"])

    def test_large_protocol_request_is_streamed_over_stdin(self):
        class LocalSandbox:
            def __init__(self):
                self.argv = ()
                self.interactive = False

            def popen(self, _root, argv, **kwargs):
                self.argv = tuple(argv)
                self.interactive = kwargs.pop("interactive")
                for key in ("writable_workspace", "operation_id", "component"):
                    kwargs.pop(key)
                return subprocess.Popen(
                    [sys._base_executable, "-u", "-c", "import sys; sys.stdout.write(sys.stdin.read())"],
                    **kwargs,
                )

            def cleanup_process(self, process, *, reason):
                if process.poll() is None:
                    process.terminate()

        sandbox = LocalSandbox()
        request = "x" * 40_000
        result = reader_module._protocol_bytes(
            Path.cwd(), "unused", request, maximum=50_000, sandbox=sandbox,
            operation_id="large-request", component="repository-reader", cancelled=None,
        )
        self.assertEqual(request.encode(), result)
        self.assertTrue(sandbox.interactive)
        self.assertNotIn(request, sandbox.argv)

    def test_pinned_batch_excludes_binary_and_non_utf8_without_losing_text(self):
        content = (b"print('ok')\n", b"a\0b", b"\xff\xfe")
        object_ids = ("a" * 40, "b" * 40, "c" * 40)
        raw = b"".join(
            f"{object_id} blob {len(data)}\n".encode() + data + b"\n"
            for object_id, data in zip(object_ids, content)
        )
        git_calls = []

        def fake_git(arguments, **_kwargs):
            git_calls.append(arguments)
            return raw if arguments[-1] == "--batch" else b""

        config = {
            "expected_identity": "a" * 64, "commit_sha": "b" * 40,
            "entries": [
                {"path": name, "size": len(data), "object_id": oid}
                for name, data, oid in zip(("one.py", "two.py", "three.py"), content, object_ids)
            ],
            "max_git_bytes": 4096,
            "policy": {"sensitive_parts": [], "sensitive_suffixes": []},
        }
        with patch.object(protocol, "_git", side_effect=fake_git):
            result = protocol._pinned_read(config, {"identity_hash": "a" * 64})
        self.assertEqual(["ok", "excluded", "excluded"], [item["status"] for item in result["results"]])
        self.assertEqual(["binary_content", "non_utf8"], [item["reason"] for item in result["results"][1:]])
        self.assertEqual(2, len(git_calls))

    def test_reader_pinned_batch_uses_one_protocol_call_and_reports_exclusions(self):
        manifest = RepositorySnapshotManifest(
            "a" * 64, "b" * 40, "main",
            (RepositorySnapshotEntry("one.py", 2), RepositorySnapshotEntry("two.py", 2)),
        )
        payload = {
            "identity_hash": "a" * 64, "commit_sha": "b" * 40,
            "results": [
                {"path": "one.py", "status": "ok", "data": base64.b64encode(b"ok").decode()},
                {"path": "two.py", "status": "excluded", "reason": "non_utf8"},
            ],
        }
        with patch.object(reader_module, "_repository_protocol", return_value=payload) as operation:
            documents, excluded = SafeRepositoryReader().read_pinned_files_with_exclusions(
                Path.cwd(), manifest, ("one.py", "two.py"),
                max_total_bytes=10, max_file_bytes=10,
            )
        self.assertEqual(1, operation.call_count)
        self.assertEqual((("one.py", "ok"),), documents)
        self.assertEqual((("two.py", "non_utf8"),), excluded)

    def test_protocol_wait_observes_cancellation_and_cleans_up_the_process(self):
        class SlowSandbox:
            def __init__(self):
                self.cleanup_reasons: list[str] = []

            def popen(self, _root, _argv, **kwargs):
                for key in ("writable_workspace", "operation_id", "component", "interactive"):
                    kwargs.pop(key)
                return subprocess.Popen(
                    [sys._base_executable, "-u", "-c", "import time; time.sleep(30)"],
                    **kwargs,
                )

            def note_process_termination(self, _process, *, reason):
                self.cleanup_reasons.append(f"note:{reason}")

            def cleanup_process(self, process, *, reason):
                self.cleanup_reasons.append(reason)
                if process.poll() is None:
                    process.terminate()

        sandbox = SlowSandbox()

        with self.assertRaises(RepositoryCancelled):
            reader_module._protocol_bytes(
                Path.cwd(),
                "print('unreachable')",
                "{}",
                maximum=1_024,
                sandbox=sandbox,
                operation_id="cancel-test",
                component="repository-tools",
                cancelled=lambda: True,
            )

        self.assertIn("note:cancelled", sandbox.cleanup_reasons)
        self.assertIn("cancelled", sandbox.cleanup_reasons)

    def test_search_reaches_a_match_after_the_first_thousand_tracked_files(self):
        head = "a" * 40
        entries = [
            protocol.TreeEntry(f"src/{index:04d}.py", f"{index:040x}", 32)
            for index in range(1_000)
        ]
        entries.append(protocol.TreeEntry("zz/last.py", "f" * 40, 32))
        calls: list[list[str]] = []

        def git(arguments, **_kwargs):
            calls.append(arguments)
            if arguments[0] != "grep" or ":(literal)zz/last.py" not in arguments:
                return b""
            return f"{head}:zz/last.py\0".encode() + b"7\0needle is here\0"

        with patch.object(protocol, "_git", side_effect=git):
            result = protocol._search(
                entries,
                head,
                {"query": "needle"},
                {
                    "max_blob_bytes": 1_024,
                    "max_search_bytes": 16_384,
                    "max_search_results": 20,
                    "max_result_characters": 12_000,
                },
            )

        grep_calls = [arguments for arguments in calls if arguments[0] == "grep"]
        self.assertEqual(1, len(grep_calls))
        self.assertTrue(all("--fixed-strings" in arguments for arguments in grep_calls))
        self.assertTrue(all("-e" in arguments for arguments in grep_calls))
        self.assertEqual(
            "needle is here",
            base64.b64decode(result["matches"][0]["excerpt_b64"]).decode(),
        )
        self.assertEqual("zz/last.py", result["matches"][0]["path"])
        self.assertEqual(1_001, result["searched_tracked_files"])

    def test_search_passes_option_like_literal_as_an_explicit_pattern(self):
        head = "a" * 40
        entries = [protocol.TreeEntry("src/options.py", "b" * 40, 32)]
        calls: list[list[str]] = []

        def git(arguments, **_kwargs):
            calls.append(arguments)
            return f"{head}:src/options.py\0".encode() + b"1\0--help\0"

        with patch.object(protocol, "_git", side_effect=git):
            result = protocol._search(
                entries,
                head,
                {"query": "--help"},
                {
                    "max_blob_bytes": 1_024,
                    "max_search_bytes": 16_384,
                    "max_search_results": 20,
                    "max_result_characters": 12_000,
                },
            )

        grep = next(arguments for arguments in calls if arguments[0] == "grep")
        self.assertEqual("-e", grep[grep.index("--fixed-strings") + 1])
        self.assertEqual("--help", grep[grep.index("-e") + 1])
        self.assertEqual("src/options.py", result["matches"][0]["path"])

    def test_git_stops_both_output_streams_at_the_configured_limit(self):
        class FakeProcess:
            def __init__(self):
                self.stdout = io.BytesIO(b"x" * 2_048)
                self.stderr = io.BytesIO(b"y" * 2_048)
                self.stdin = io.BytesIO()
                self.terminated = False

            def terminate(self):
                self.terminated = True

            def kill(self):
                self.terminated = True

            def wait(self, timeout=None):
                return 0

        process = FakeProcess()
        with patch.object(protocol.subprocess, "Popen", return_value=process):
            with self.assertRaises(protocol.ProtocolError):
                protocol._git(["status"], maximum=1_024)

        self.assertTrue(process.terminated)

    def test_inspect_marks_ranked_documents_omitted_by_selection_limit(self):
        entries = [
            protocol.TreeEntry(f"docs/guide-{index}.md", f"{index:040x}", 10)
            for index in range(10)
        ]
        config = {
            "query": "guide",
            "query_tokens": ["guide"],
            "limits": {
                "max_tree_bytes": 2_000_000,
                "max_files": 8,
                "max_file_bytes": 100,
            },
            "policy": {"foundation_names": {}},
        }
        identity = {"identity_hash": "a" * 64, "head_sha": "b" * 40, "branch": "main"}

        with (
            patch.object(protocol, "_tree", return_value=(entries, {"unsafe_path": 0})),
            patch.object(
                protocol,
                "_batch_blobs",
                side_effect=lambda selected, **_kwargs: {entry.path: b"contents" for entry in selected},
            ),
        ):
            result = protocol._inspect(config, identity)

        self.assertEqual(8, len(result["document_blobs"]))
        self.assertEqual(2, result["exclusions"]["document_selection_limit"])

    def test_tools_marks_excess_detail_reads_truncated_before_response_overflow(self):
        entries = [
            protocol.TreeEntry(f"src/{name}.txt", object_id, 262_144)
            for name, object_id in zip(("a", "b", "c"), ("a" * 40, "b" * 40, "c" * 40))
        ]
        config = {
            "requests": [
                {"tool": "read_file", "path": entry.path}
                for entry in entries
            ],
            "limits": {
                "max_tree_bytes": 2_000_000,
                "max_blob_bytes": 262_144,
                "max_response_bytes": 1_000_000,
            },
            "policy": {},
        }
        identity = {"identity_hash": "a" * 64, "head_sha": "b" * 40, "branch": "main"}
        blobs = {entry.path: b"x" * entry.size for entry in entries}

        with (
            patch.object(protocol, "_tree", return_value=(entries, {"unsafe_path": 0})),
            patch.object(protocol, "_batch_blobs", return_value=blobs),
        ):
            result = protocol._tools(config, identity)

        rendered = json.dumps(
            {"ok": True, "result": result}, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        self.assertLessEqual(len(rendered), 1_000_000)
        self.assertEqual(["ok", "ok", "truncated"], [item["status"] for item in result["results"]])
        self.assertEqual("batch_response_byte_limit", result["results"][2]["reason"])

    def test_search_reports_oversize_and_binary_exclusions(self):
        entry = protocol.TreeEntry("source.dat", "b" * 40, 262_145)

        with patch.object(protocol, "_git", return_value=b""):
            result = protocol._search(
                [entry],
                "a" * 40,
                {"query": "needle"},
                {
                    "max_blob_bytes": 262_144,
                    "max_search_bytes": 16_384,
                    "max_search_results": 20,
                    "max_result_characters": 12_000,
                },
            )

        self.assertTrue(result["truncated"])
        self.assertIn("oversize_file_content_excluded", result["truncation_reasons"])
        self.assertEqual(1, result["excluded"]["oversize_file_content"])
        self.assertIn("binary", result["excluded"]["binary_content"])

    def test_reader_uses_one_batched_protocol_result_and_keeps_exclusion_reason(self):
        payload = {
            "identity_hash": "a" * 64,
            "head_sha": "b" * 40,
            "branch": "main",
            "paths": ["README.md", "app.py"],
            "document_blobs": [
                {
                    "path": "app.py",
                    "data": base64.b64encode(
                        b"API_TOKEN=must-not-be-exposed\nprint('ready')\n"
                    ).decode(),
                }
            ],
            "exclusions": {"document_file_size_limit": 1},
        }
        with patch.object(reader_module, "_repository_protocol", return_value=payload) as operation:
            context = SafeRepositoryReader().inspect(
                Path.cwd(),
                "프로젝트 코드 구조",
                expected_identity="a" * 64,
            )

        self.assertEqual(1, operation.call_count)
        self.assertEqual("b" * 40, context.head_sha)
        self.assertTrue(context.truncated)
        self.assertEqual(1, context.to_dict()["exclusions"]["document_file_size_limit"])
        self.assertIn("[REDACTED]", dict(context.documents)["app.py"])

    def test_reader_marks_document_selection_and_tree_limits_as_truncated(self):
        payload = {
            "identity_hash": "a" * 64,
            "head_sha": "b" * 40,
            "branch": "main",
            "paths": ["README.md", "app.py", "docs/guide.md"],
            "document_blobs": [],
            "exclusions": {"document_selection_limit": 2},
        }
        with patch.object(reader_module, "_repository_protocol", return_value=payload):
            context = SafeRepositoryReader(max_tree_entries=2).inspect(
                Path.cwd(),
                "프로젝트 코드 구조",
                expected_identity="a" * 64,
            )

        self.assertTrue(context.truncated)
        self.assertEqual(2, context.to_dict()["exclusions"]["document_selection_limit"])
        self.assertEqual(1, context.to_dict()["exclusions"]["tree_entry_limit"])

    def test_tool_batch_uses_one_protocol_result_and_marks_limited_search(self):
        payload = {
            "identity_hash": "a" * 64,
            "head_sha": "b" * 40,
            "branch": "main",
            "results": [
                {
                    "tool": "search_files",
                    "data": {
                        "matches": [
                            {
                                "path": "zz/last.py",
                                "kind": "content",
                                "line": 3,
                                "excerpt_b64": base64.b64encode(
                                    b"API_TOKEN=must-not-be-exposed"
                                ).decode(),
                            }
                        ],
                        "searched_tracked_files": 1_001,
                        "truncated": True,
                        "truncation_reasons": ["search_output_byte_limit"],
                        "excluded": {
                            "oversize_file_content": 2,
                            "binary_content": "git grep -I excludes binary content",
                        },
                    },
                }
            ],
        }
        with patch.object(tools_module, "_repository_protocol", return_value=payload) as operation:
            batch = SafeRepositoryToolLayer().execute(
                Path.cwd(),
                (RepositoryToolRequest("search_files", query="token"),),
                expected_identity="a" * 64,
                expected_head="b" * 40,
            )

        self.assertEqual(1, operation.call_count)
        result = batch.results[0].data
        self.assertTrue(batch.truncated)
        self.assertEqual(1_001, result["searched_tracked_files"])
        self.assertIn("search_output_byte_limit", result["truncation_reasons"])
        self.assertEqual(2, result["excluded"]["oversize_file_content"])
        self.assertNotIn("must-not-be-exposed", result["matches"][0]["excerpt"])

    def test_tool_batch_renders_protocol_response_limited_read_as_truncated(self):
        payload = {
            "identity_hash": "a" * 64,
            "head_sha": "b" * 40,
            "branch": "main",
            "results": [
                {
                    "tool": "read_file",
                    "status": "truncated",
                    "reason": "batch_response_byte_limit",
                }
            ],
        }
        request = RepositoryToolRequest("read_file", path="app.py", start_line=1, end_line=2)
        with patch.object(tools_module, "_repository_protocol", return_value=payload):
            batch = SafeRepositoryToolLayer().execute(
                Path.cwd(),
                (request,),
                expected_identity="a" * 64,
                expected_head="b" * 40,
            )

        self.assertTrue(batch.truncated)
        self.assertEqual("truncated", batch.results[0].status)
        self.assertEqual("batch_response_byte_limit", batch.results[0].data["reason"])


if __name__ == "__main__":
    unittest.main()
