"""Fixed, read-only Git snapshot protocol executed inside the repository sandbox.

This module is passed as static source to ``python3 -I -S -c`` in the
container.  It never receives shell text: the host sends one JSON request on stdin and
this program invokes Git with explicit argv lists only.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any


_HEX_SHA = re.compile(r"[0-9a-f]{40,64}")
_PRIVATE_KEY = re.compile(
    rb"-----BEGIN [^-\r\n]*PRIVATE KEY-----.*?-----END [^-\r\n]*PRIVATE KEY-----",
    re.DOTALL,
)
_PRIVATE_KEY_MASK = bytes(byte if byte in (10, 13) else ord("*") for byte in range(256))


class ProtocolError(RuntimeError):
    def __init__(self, message: str, *, kind: str = "access"):
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class TreeEntry:
    path: str
    object_id: str
    size: int


def _git(
    arguments: list[str],
    *,
    maximum: int,
    input_data: bytes | None = None,
    allow_failure: bool = False,
    timeout_seconds: int = 30,
) -> bytes:
    process = subprocess.Popen(
        ["git", "--no-replace-objects", "-c", "core.pager=cat", *arguments],
        cwd="/workspace",
        stdin=subprocess.PIPE if input_data is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout = bytearray()
    stderr = bytearray()
    overflow = threading.Event()

    def drain(stream, target: bytearray, limit: int) -> None:
        try:
            while True:
                chunk = stream.read(65_536)
                if not chunk:
                    return
                remaining = limit - len(target)
                if remaining > 0:
                    target.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    overflow.set()
                    process.terminate()
                    return
        finally:
            stream.close()

    assert process.stdout is not None and process.stderr is not None
    stdout_thread = threading.Thread(
        target=drain, args=(process.stdout, stdout, maximum), daemon=True
    )
    stderr_thread = threading.Thread(
        target=drain, args=(process.stderr, stderr, min(maximum, 65_536)), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()
    if input_data is not None:
        assert process.stdin is not None
        try:
            process.stdin.write(input_data)
        except BrokenPipeError:
            pass
        finally:
            process.stdin.close()
    try:
        return_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        return_code = process.wait()
        raise ProtocolError("Git 읽기 작업 시간이 초과했습니다.")
    finally:
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
    if stdout_thread.is_alive() or stderr_thread.is_alive():
        raise ProtocolError("Git 출력 수집을 안전하게 완료하지 못했습니다.")
    if overflow.is_set():
        raise ProtocolError("Git 조회 결과가 안전 크기 제한을 초과했습니다.")
    if return_code and not (allow_failure and return_code == 1):
        detail = bytes(stderr).decode("utf-8", errors="replace").strip()
        raise ProtocolError(f"Git 읽기 작업이 실패했습니다: {detail[:300]}")
    return bytes(stdout)


def _git_text(
    arguments: list[str], *, allow_empty: bool = False, maximum: int = 2_000_000
) -> str:
    value = _git(arguments, maximum=maximum, allow_failure=allow_empty).decode(
        "utf-8", errors="replace"
    ).strip()
    if not value and not allow_empty:
        raise ProtocolError(f"Git 조회 결과가 비어 있습니다: {' '.join(arguments)}")
    return value


def _remote_identity(remote: str) -> str:
    value = remote.strip()
    value = re.sub(r"(?i)(https?://)[^/@\s]+@", r"\1", value)
    value = re.sub(r"(?i)(ssh://)[^/@\s]+@", r"\1", value)
    return value.casefold()


def _identity(config: dict[str, Any]) -> dict[str, str]:
    top = _git_text(["rev-parse", "--show-toplevel"])
    if top != "/workspace":
        raise ProtocolError("승인 경로가 Git 최상위 폴더가 아닙니다.")
    head = _git_text(["rev-parse", "HEAD"]).lower()
    if not _HEX_SHA.fullmatch(head):
        raise ProtocolError("유효한 Git HEAD를 확인하지 못했습니다.")
    branch = _git_text(["branch", "--show-current"], allow_empty=True)
    roots = sorted(
        line.strip().lower()
        for line in _git_text(
            ["rev-list", "--max-parents=0", "--all"], allow_empty=True
        ).splitlines()
        if line.strip()
    )
    if not roots or any(not _HEX_SHA.fullmatch(item) for item in roots):
        raise ProtocolError("저장소의 최초 커밋 식별값을 확인하지 못했습니다.")
    common = _git_text(["rev-parse", "--git-common-dir"])
    remote = _git_text(["config", "--get", "remote.origin.url"], allow_empty=True)
    payload = json.dumps(
        {
            "root": str(config["identity_root"]).casefold(),
            "git_common_dir": common.casefold(),
            "root_commits": roots,
            "origin": _remote_identity(remote),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "identity_hash": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "head_sha": head,
        "branch": branch,
    }


def _path_is_safe(path: str, policy: dict[str, Any]) -> bool:
    if not path or "\\" in path or any(ord(character) < 32 for character in path):
        return False
    value = PurePosixPath(path)
    if value.is_absolute() or ".." in value.parts:
        return False
    lowered = tuple(part.casefold() for part in value.parts)
    sensitive_parts = set(policy["sensitive_parts"])
    if any(part in sensitive_parts for part in lowered):
        return False
    if any(
        marker in part
        for part in lowered[:-1]
        for marker in ("credential", "secret", "private_key")
    ):
        return False
    name = lowered[-1]
    if (
        name.startswith(".env")
        or name.endswith(".env")
        or ".env." in name
        or any(marker in name for marker in ("credential", "secret", "private_key"))
    ):
        return False
    return PurePosixPath(name).suffix.casefold() not in set(policy["sensitive_suffixes"])


def _tree(
    head: str, *, maximum: int, policy: dict[str, Any]
) -> tuple[list[TreeEntry], dict[str, int]]:
    raw = _git(
        ["ls-tree", "-r", "-z", "-l", "--full-tree", head], maximum=maximum
    )
    entries: list[TreeEntry] = []
    excluded = {"unsafe_path": 0}
    try:
        for item in raw.split(b"\0"):
            if not item:
                continue
            metadata, separator, raw_path = item.partition(b"\t")
            fields = metadata.split()
            if not separator or len(fields) != 4 or fields[1] != b"blob":
                continue
            path = raw_path.decode("utf-8", errors="strict")
            size = int(fields[3])
            object_id = fields[2].decode("ascii", errors="strict")
            if _path_is_safe(path, policy):
                entries.append(TreeEntry(path, object_id, size))
            else:
                excluded["unsafe_path"] += 1
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProtocolError("Git 트리의 파일 이름 또는 크기 정보가 올바르지 않습니다.") from exc
    return entries, excluded


def _batch_blobs(entries: list[TreeEntry], *, maximum: int) -> dict[str, bytes]:
    if not entries:
        return {}
    request = b"".join(f"{entry.object_id}\n".encode("ascii") for entry in entries)
    raw = _git(
        ["cat-file", "--batch"],
        maximum=maximum,
        input_data=request,
    )
    cursor = 0
    result: dict[str, bytes] = {}
    for entry in entries:
        header_end = raw.find(b"\n", cursor)
        if header_end < 0:
            raise ProtocolError("Git 일괄 파일 조회 결과가 잘렸습니다.")
        fields = raw[cursor:header_end].split()
        if len(fields) != 3 or fields[0].decode("ascii", errors="ignore") != entry.object_id or fields[1] != b"blob":
            raise ProtocolError("Git 일괄 파일 조회 결과가 올바르지 않습니다.")
        try:
            size = int(fields[2])
        except ValueError as exc:
            raise ProtocolError("Git 일괄 파일 크기 결과가 올바르지 않습니다.") from exc
        if size != entry.size:
            raise ProtocolError("조회 중 Git 스냅샷의 파일 크기가 달라져 작업을 중단했습니다.", kind="snapshot")
        content_start = header_end + 1
        content_end = content_start + size
        if content_end >= len(raw) or raw[content_end : content_end + 1] != b"\n":
            raise ProtocolError("Git 일괄 파일 내용이 올바르지 않습니다.")
        result[entry.path] = raw[content_start:content_end]
        cursor = content_end + 1
    return result


def _pinned_read(config: dict[str, Any], identity: dict[str, str]) -> dict[str, Any]:
    """Read a fixed commit with one cat-file batch and isolate bad text files."""
    if identity["identity_hash"] != config["expected_identity"]:
        raise ProtocolError("저장소 식별값이 분석 시작 당시와 다릅니다.", kind="identity")
    commit = str(config["commit_sha"])
    if not _HEX_SHA.fullmatch(commit):
        raise ProtocolError("고정 commit 식별값이 올바르지 않습니다.")
    try:
        _git(["cat-file", "-e", f"{commit}^{{commit}}"], maximum=128)
    except ProtocolError as exc:
        raise ProtocolError("고정 commit object를 읽을 수 없습니다.", kind="unavailable") from exc
    entries = config["entries"]
    if not isinstance(entries, list) or not entries:
        raise ProtocolError("고정 파일 묶음이 비었습니다.")
    specs: list[str] = []
    for entry in entries:
        path = str(entry["path"])
        size = int(entry["size"])
        oid = str(entry.get("object_id", ""))
        if not _path_is_safe(path, config["policy"]) or size < 0 or (oid and not _HEX_SHA.fullmatch(oid)):
            raise ProtocolError("고정 파일 묶음의 경로 또는 크기가 올바르지 않습니다.")
        specs.append(f"{commit}:{path}")
    raw = _git(
        ["cat-file", "--batch"], maximum=int(config["max_git_bytes"]),
        input_data=("\n".join(specs) + "\n").encode("utf-8"),
        timeout_seconds=min(40, max(5, len(specs) // 3)),
    )
    cursor = 0
    results: list[dict[str, str]] = []
    for entry in entries:
        header_end = raw.find(b"\n", cursor)
        if header_end < 0:
            raise ProtocolError("고정 파일 묶음 결과가 잘렸습니다.")
        fields = raw[cursor:header_end].split()
        if len(fields) != 3 or fields[1] != b"blob":
            raise ProtocolError("고정 파일 묶음 결과가 올바르지 않습니다.", kind="snapshot")
        object_id = fields[0].decode("ascii", errors="strict")
        expected_oid = str(entry.get("object_id", ""))
        size = int(fields[2])
        if not _HEX_SHA.fullmatch(object_id) or (expected_oid and object_id != expected_oid) or size != int(entry["size"]):
            raise ProtocolError("고정 파일의 object 또는 크기가 달라졌습니다.", kind="snapshot")
        content_start = header_end + 1
        content_end = content_start + size
        if content_end >= len(raw) or raw[content_end:content_end + 1] != b"\n":
            raise ProtocolError("고정 파일 내용이 잘렸습니다.")
        data = raw[content_start:content_end]
        cursor = content_end + 1
        path = str(entry["path"])
        if b"\0" in data:
            results.append({"path": path, "status": "excluded", "reason": "binary_content"})
            continue
        try:
            data.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            results.append({"path": path, "status": "excluded", "reason": "non_utf8"})
            continue
        row = {"path": path, "status": "ok"}
        if config.get("chunk_bytes"):
            # 조각 경계 밖의 PEM도 격리 blob 안에서 가리고 원본 byte·CR/LF를 보존한다.
            data = _PRIVATE_KEY.sub(lambda match: match.group().translate(_PRIVATE_KEY_MASK), data)
            start = int(entry.get("start_line", 1))
            cap = int(config["chunk_bytes"])
            lines = data.splitlines(keepends=True) or [b""]
            if not 1 <= start <= len(lines) or cap < 1:
                raise ProtocolError("고정 파일 행 범위가 올바르지 않습니다.")
            chunk = bytearray()
            end = start - 1
            for line in lines[start - 1:]:
                if len(chunk) + len(line) > cap:
                    break
                chunk.extend(line)
                end += 1
            if end < start:
                results.append({"path": path, "status": "excluded", "reason": "line_size_limit"})
                continue
            row.update(start_line=start, end_line=end, total_lines=len(lines),
                       next_line=end + 1 if end < len(lines) else 0)
            data = bytes(chunk)
        row["data"] = base64.b64encode(data).decode("ascii")
        results.append(row)
    return {"identity_hash": identity["identity_hash"], "commit_sha": commit, "results": results}


def _rank_paths(entries: list[TreeEntry], config: dict[str, Any]) -> list[TreeEntry]:
    tokens = set(config["query_tokens"])
    foundation = dict(config["policy"]["foundation_names"])
    query = str(config["query"]).casefold()
    code_markers = ("코드", "소스", "파일", "구조", "프로젝트")
    code_suffixes = {
        ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".rs", ".go", ".cs"
    }
    scored: list[tuple[int, TreeEntry]] = []
    for entry in entries:
        lowered = entry.path.casefold()
        name = PurePosixPath(lowered).name
        score = int(foundation.get(name, 0))
        score += sum(30 for token in tokens if token in lowered)
        if not score and any(marker in query for marker in code_markers) and PurePosixPath(name).suffix in code_suffixes:
            score = 10
        if score:
            scored.append((score, entry))
    scored.sort(key=lambda item: (-item[0], len(item[1].path), item[1].path.casefold()))
    return [entry for _score, entry in scored]


def _encoded_blobs(blobs: dict[str, bytes]) -> list[dict[str, str]]:
    return [
        {"path": path, "data": base64.b64encode(value).decode("ascii")}
        for path, value in blobs.items()
    ]


def _inspect(config: dict[str, Any], identity: dict[str, str]) -> dict[str, Any]:
    limits = config["limits"]
    entries, exclusions = _tree(
        identity["head_sha"], maximum=int(limits["max_tree_bytes"]), policy=config["policy"]
    )
    ranked = _rank_paths(entries, config)
    selected = ranked[: int(limits["max_files"])]
    if len(ranked) > len(selected):
        exclusions["document_selection_limit"] = len(ranked) - len(selected)
    readable = [entry for entry in selected if entry.size <= int(limits["max_file_bytes"])]
    too_large = len(selected) - len(readable)
    if too_large:
        exclusions["document_file_size_limit"] = too_large
    batch_limit = sum(entry.size for entry in readable) + (len(readable) * 128) + 1024
    return {
        **identity,
        "paths": [entry.path for entry in entries],
        "document_blobs": _encoded_blobs(_batch_blobs(readable, maximum=batch_limit)),
        "exclusions": exclusions,
    }


def _add_match(matches: list[dict[str, Any]], value: dict[str, Any], limits: dict[str, Any]) -> bool:
    if len(matches) >= int(limits["max_search_results"]):
        return False
    rendered = json.dumps(matches + [value], ensure_ascii=False, separators=(",", ":"))
    if len(rendered) > int(limits["max_result_characters"]):
        return False
    matches.append(value)
    return True


def _search(
    entries: list[TreeEntry], head: str, request: dict[str, Any], limits: dict[str, Any]
) -> dict[str, Any]:
    needle = str(request["query"]).casefold()
    matches: list[dict[str, Any]] = []
    reasons: list[str] = []
    for entry in sorted(entries, key=lambda item: item.path.casefold()):
        if needle in entry.path.casefold() and not _add_match(
            matches, {"path": entry.path, "kind": "path", "line": None, "excerpt": ""}, limits
        ):
            reasons.append("result_limit")
            return _search_result(matches, reasons, entries, limits)
    eligible = [entry for entry in entries if entry.size <= int(limits["max_blob_bytes"])]
    if len(eligible) != len(entries):
        reasons.append("oversize_file_content_excluded")
    for offset in range(0, len(eligible), 4096):
        chunk = eligible[offset : offset + 4096]
        pathspecs = [f":(literal){entry.path}" for entry in chunk]
        try:
            raw = _git(
                [
                    "grep", "-i", "-I", "-n", "-z", "-m", "3", "--fixed-strings",
                    "-e", str(request["query"]), head, "--", *pathspecs,
                ],
                maximum=int(limits["max_search_bytes"]),
                allow_failure=True,
            )
        except ProtocolError as exc:
            if "안전 크기 제한" not in str(exc):
                raise
            reasons.append("search_output_byte_limit")
            break
        parts = raw.split(b"\0")
        prefix = f"{head}:".encode("ascii")
        for index in range(0, len(parts) - 2, 3):
            raw_path, raw_line, raw_excerpt = parts[index : index + 3]
            if not raw_path.startswith(prefix):
                continue
            try:
                path = raw_path[len(prefix) :].decode("utf-8", errors="strict")
                line = int(raw_line)
            except (UnicodeDecodeError, ValueError):
                continue
            if not _add_match(
                matches,
                {
                    "path": path,
                    "kind": "content",
                    "line": line,
                    "excerpt_b64": base64.b64encode(raw_excerpt[:16_384]).decode("ascii"),
                },
                limits,
            ):
                reasons.append("result_limit")
                return _search_result(matches, reasons, entries, limits)
    return _search_result(matches, reasons, entries, limits)


def _search_result(
    matches: list[dict[str, Any]], reasons: list[str], entries: list[TreeEntry], limits: dict[str, Any]
) -> dict[str, Any]:
    return {
        "matches": matches,
        "searched_tracked_files": len(entries),
        "truncated": bool(reasons),
        "truncation_reasons": reasons,
        "excluded": {
            "oversize_file_content": sum(
                entry.size > int(limits["max_blob_bytes"]) for entry in entries
            ),
            "binary_content": "git grep -I excludes binary content",
        },
    }


def _tools(config: dict[str, Any], identity: dict[str, str]) -> dict[str, Any]:
    limits = config["limits"]
    entries, exclusions = _tree(
        identity["head_sha"], maximum=int(limits["max_tree_bytes"]), policy=config["policy"]
    )
    by_path = {entry.path: entry for entry in entries}
    requests = list(config["requests"])
    read_entries = [
        by_path[request["path"]]
        for request in requests
        if request["tool"] == "read_file"
        and request["path"] in by_path
        and by_path[request["path"]].size <= int(limits["max_blob_bytes"])
    ]
    unique_reads = list({entry.path: entry for entry in read_entries}.values())
    batch_limit = sum(entry.size for entry in unique_reads) + (len(unique_reads) * 128) + 1024
    blobs = _batch_blobs(unique_reads, maximum=batch_limit)
    results: list[dict[str, Any]] = []
    response_limit = int(limits.get("max_response_bytes", 1_000_000))

    def result_payload(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {**identity, "results": items, "exclusions": exclusions}

    def response_fits(items: list[dict[str, Any]]) -> bool:
        rendered = json.dumps(
            {"ok": True, "result": result_payload(items)},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return len(rendered.encode("utf-8")) <= response_limit

    def append_result(value: dict[str, Any], fallback: dict[str, Any]) -> None:
        if response_fits([*results, value]):
            results.append(value)
            return
        if response_fits([*results, fallback]):
            results.append(fallback)
            return
        raise ProtocolError("저장소 도구 응답 예산이 기본 결과보다 작습니다.")

    for request in requests:
        if request["tool"] == "search_files":
            search_result = {
                "tool": "search_files",
                "data": _search(entries, identity["head_sha"], request, limits),
            }
            append_result(
                search_result,
                {
                    "tool": "search_files",
                    "data": {
                        "matches": [],
                        "searched_tracked_files": len(entries),
                        "truncated": True,
                        "truncation_reasons": ["protocol_response_byte_limit"],
                        "excluded": {},
                    },
                },
            )
            continue
        entry = by_path.get(request["path"])
        if entry is None:
            append_result(
                {"tool": "read_file", "status": "denied", "reason": "safe_path_not_found"},
                {"tool": "read_file", "status": "denied", "reason": "safe_path_not_found"},
            )
        elif entry.size > int(limits["max_blob_bytes"]):
            append_result(
                {"tool": "read_file", "status": "denied", "reason": "file_size_limit"},
                {"tool": "read_file", "status": "denied", "reason": "file_size_limit"},
            )
        else:
            append_result(
                {
                    "tool": "read_file",
                    "status": "ok",
                    "path": entry.path,
                    "data": base64.b64encode(blobs[entry.path]).decode("ascii"),
                },
                {
                    "tool": "read_file",
                    "status": "truncated",
                    "reason": "batch_response_byte_limit",
                },
            )
    return result_payload(results)


def run(config: dict[str, Any]) -> dict[str, Any]:
    identity = _identity(config)
    mode = config["mode"]
    if mode == "identity":
        return identity
    if mode == "inspect":
        return _inspect(config, identity)
    if mode == "tools":
        return _tools(config, identity)
    if mode == "pinned_read":
        return _pinned_read(config, identity)
    raise ProtocolError("지원하지 않는 저장소 조회 작업입니다.")


def main() -> None:
    try:
        config = json.load(sys.stdin)
        result = {"ok": True, "result": run(config)}
    except json.JSONDecodeError:
        result = {"ok": False, "kind": "request_decode", "error": "격리 저장소 조회 요청 JSON을 읽지 못했습니다."}
    except ProtocolError as exc:
        result = {"ok": False, "kind": exc.kind, "error": str(exc)}
    except (KeyError, TypeError, ValueError) as exc:
        result = {"ok": False, "kind": "access", "error": f"저장소 조회 요청 형식 오류: {type(exc).__name__}"}
    sys.stdout.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
