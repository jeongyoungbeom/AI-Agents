from __future__ import annotations

import base64
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.context import ContextService
from app.services.logging.redaction import SecretRedactor
from app.services.repository import RepositorySnapshotEntry, RepositorySnapshotManifest, SafeRepositoryReader
from app.services.repository import protocol, reader as reader_module
from tests.gateway.support import temporary_directory
from tests.gateway import test_s06_repository as support


BODY = "U1lOVEhFVElDX09OTFlfTk9UX0FfUkVBTF9LRVk="  # 사용 불가능한 합성 본문
BEGIN = "-----BEGIN PRIVATE KEY-----"
END = "-----END PRIVATE KEY-----"


class S06ReviewResolutionTests(unittest.TestCase):
    def read_range(self, root, text, *, start=1, cap=4096):
        data = text.encode("utf-8")
        raw = f"{'c' * 40} blob {len(data)}\n".encode() + data + b"\n"
        manifest = RepositorySnapshotManifest("a" * 64, "b" * 40, "main",
            (RepositorySnapshotEntry("main.py", len(data), "c" * 40),))
        responses = []

        def local_protocol(_root, config, **kwargs):
            self.assertEqual(len(data) + 160 + 1024, config["max_git_bytes"])
            with patch.object(protocol, "_git", side_effect=lambda args, **kw: raw if args[-1] == "--batch" else b""):
                result = protocol._pinned_read(config, {"identity_hash": "a" * 64})
            responses.append(result["results"][0])
            return result

        with patch.object(reader_module, "_repository_protocol", side_effect=local_protocol):
            docs, excluded = SafeRepositoryReader().read_pinned_ranges(root, manifest, ("main.py",),
                start_lines={"main.py": start}, max_chunk_bytes=cap, max_scan_bytes=len(data))
        self.assertFalse(excluded)
        return docs[0], responses[0]

    def test_pem_is_masked_before_each_slice_with_original_bytes_and_continuation(self):
        for newline in ("\n", "\r\n"):
            text = newline.join((BEGIN, BODY, "합성 본문", END, "TAIL_DEFECT")) + newline
            original_lines = text.encode().splitlines(keepends=True)
            for start in (1, 2, 3, 4, 5):
                with self.subTest(newline=repr(newline), start=start):
                    doc, row = self.read_range(Path.cwd(), text, start=start, cap=len(original_lines[start - 1]))
                    emitted = base64.b64decode(row["data"])
                    self.assertEqual(len(original_lines[start - 1]), len(emitted))
                    self.assertLessEqual(len(emitted), len(original_lines[start - 1]))
                    self.assertEqual((start, start, 5, start + 1 if start < 5 else 0),
                        (doc["start_line"], doc["end_line"], doc["total_lines"], doc["next_line"]))
                    if start < 5:
                        self.assertEqual(b"*" * (len(emitted) - len(newline)) + newline.encode(), emitted)
                        self.assertNotIn(BODY, doc["content"])
                    else:
                        self.assertEqual("TAIL_DEFECT" + newline, doc["content"])

    def test_reader_preserves_lf_crlf_for_complete_pem_and_general_credentials(self):
        for newline in ("\n", "\r\n"):
            text = newline.join(("sample = '''", BEGIN, BODY, END, "'''",
                "Authorization: Bearer", "synthBearerValue123", "password =", "synthPasswordValue",
                '"api_key": "synthJsonValue"', "TAIL_DEFECT")) + newline
            with self.subTest(newline=repr(newline)):
                doc, _ = self.read_range(Path.cwd(), text)
                self.assertEqual(len(text.splitlines()), len(doc["content"].splitlines()))
                self.assertEqual(text.count(newline), doc["content"].count(newline))
                self.assertEqual("TAIL_DEFECT", doc["content"].splitlines()[10])
                for secret in (BODY, "synthBearerValue123", "synthPasswordValue", "synthJsonValue"):
                    self.assertNotIn(secret, doc["content"])

    def test_opt_in_preserves_lines_and_default_redaction_contract(self):
        redactor = SecretRedactor()
        for newline in ("\n", "\r\n"):
            text = newline.join((BEGIN, BODY, "", END, "TAIL_DEFECT")) + newline
            self.assertEqual("[REDACTED]" + newline + "TAIL_DEFECT" + newline, redactor.text(text))
            safe = redactor.text(text, preserve_lines=True)
            self.assertEqual(len(text.splitlines()), len(safe.splitlines()))
            self.assertEqual(text.count(newline), safe.count(newline))
            self.assertNotIn(BODY, safe)
        values = ('{"api_key": "synthJsonValue"}', 'token=synthTokenValue',
            'https://fixture:synthPasswordValue@example.invalid/path', 'Bearer synthBearerValue123')
        for value in values:
            self.assertEqual(redactor.text(value), redactor.text(value, preserve_lines=True))
            self.assertEqual(redactor.text(value), redactor.value(value))

    def test_isolated_protocol_source_masks_without_application_imports(self):
        data = (BEGIN + "\n" + BODY + "\n" + END + "\nTAIL_DEFECT\n").encode()
        raw = f"{'c' * 40} blob {len(data)}\n".encode() + data + b"\n"
        config = dict(expected_identity="a" * 64, commit_sha="b" * 40, chunk_bytes=128,
            max_git_bytes=4096, policy=dict(sensitive_parts=[], sensitive_suffixes=[]),
            entries=[dict(path="main.py", size=len(data), object_id="c" * 40, start_line=2)])
        source = Path(protocol.__file__).read_text(encoding="utf-8").rsplit('if __name__ == "__main__":', 1)[0]
        source += f"\n_git = lambda arguments, **kwargs: {raw!r} if arguments[-1] == '--batch' else b''\n"
        source += "print(json.dumps(_pinned_read(json.load(sys.stdin), {'identity_hash': 'a' * 64})))\n"
        result = subprocess.run([sys.executable, "-I", "-S", "-c", source],
            input=json.dumps(config).encode(), capture_output=True, check=True)
        row = json.loads(result.stdout)["results"][0]
        self.assertNotIn(BODY.encode(), base64.b64decode(row["data"]))
        self.assertEqual((2, 4, 0), (row["start_line"], row["end_line"], row["next_line"]))

    def test_real_git_worker_model_storage_and_followup_keep_safe_absolute_evidence(self):
        cap = (24_000 - 4096) // 4
        for newline in ("\n", "\r\n"):
            for split in (False, True):
                with self.subTest(newline=repr(newline), split=split), temporary_directory() as directory:
                    root = Path(directory)
                    git_root = root / "source"
                    git_root.mkdir()
                    prefix = "sample = '''" + newline
                    if split:
                        # 첫 조각은 BEGIN까지, 다음 조각은 본문, 마지막 조각은 END·결함을 읽는다.
                        prefix = "# " + "x" * (cap - len((BEGIN + newline).encode()) - 2 - len(newline)) + newline
                    body = BODY + ("x" * (cap - len(BODY) - len(newline)) if split else "") + newline
                    text = prefix + BEGIN + newline + body + END + newline + "'''" + newline + "return value // 0  # TAIL_DEFECT" + newline
                    (git_root / "main.py").write_bytes(text.encode())
                    for args in (("init",), ("-c", "core.autocrlf=false", "add", "main.py"), ("-c", "user.name=fixture",
                            "-c", "user.email=fixture@local", "commit", "-m", "fixture")):
                        subprocess.run(["git", "-C", str(git_root), *args], check=True, capture_output=True)
                    sha = subprocess.check_output(["git", "-C", str(git_root), "rev-parse", "HEAD"]).decode().strip()
                    oid = subprocess.check_output(["git", "-C", str(git_root), "rev-parse", "HEAD:main.py"]).decode().strip()
                    fixture = support.FixtureReader({"main.py": text})
                    fixture.manifest = RepositorySnapshotManifest("a" * 64, sha, "main",
                        (RepositorySnapshotEntry("main.py", len(text.encode()), oid),))
                    fixture.read_pinned_ranges = SafeRepositoryReader().read_pinned_ranges
                    team = support.EvidenceTeam()
                    harness = support.S06RepositoryTests()
                    store, aid, worker = harness.setup_job(root, fixture, team, max_files_per_batch=1)

                    def local_git(arguments, *, maximum, input_data=None, **kwargs):
                        result = subprocess.run(["git", "--no-replace-objects", *arguments], cwd=git_root,
                            input=input_data, capture_output=True, check=True)
                        self.assertLessEqual(len(result.stdout), maximum)
                        return result.stdout

                    def local_protocol(_root, config, **kwargs):
                        with patch.object(protocol, "_git", side_effect=local_git):
                            return protocol._pinned_read(config, {"identity_hash": "a" * 64})

                    with patch.object(reader_module, "_repository_protocol", side_effect=local_protocol):
                        job = harness.drain(store, aid, worker)
                    self.assertEqual("COMPLETED", job["status"])
                    evidence = store.repository_analysis_evidence(aid)
                    tail = [row for row in evidence if "TAIL_DEFECT" in row["summary"]]
                    self.assertEqual(1, len(tail))
                    self.assertEqual(("main.py", 6, 6, sha),
                        (tail[0]["path"], tail[0]["start_line"], tail[0]["end_line"], tail[0]["commit_sha"]))
                    self.assertIn("main.py:6-6", job["final_response"])
                    followup = ContextService(store).build("parent", conversation_key="telegram:200", user_id="100")
                    visible = json.dumps(team.contexts, ensure_ascii=False) + json.dumps(evidence) + job["final_response"] + str(followup)
                    self.assertNotIn(BODY, visible)
                    self.assertNotIn(BODY.encode(), store.path.read_bytes())
                    self.assertTrue(any("TAIL_DEFECT" in message["content"] for message in followup.recent_messages))
                    reads = [row for row in evidence if row["kind"] == "source_read"]
                    self.assertEqual(3 if split else 1, len(reads))
                    self.assertTrue(all(left["end_line"] + 1 == right["start_line"] for left, right in zip(reads, reads[1:])))
                    for context in team.contexts:
                        self.assertTrue(context["untrusted_repository_data"])
                        for doc in context["documents"]:
                            self.assertEqual(doc["end_line"] - doc["start_line"] + 1, len(doc["content"].splitlines()))


if __name__ == "__main__":
    unittest.main()
