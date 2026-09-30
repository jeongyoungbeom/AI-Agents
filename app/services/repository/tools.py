from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from collections.abc import Callable

from app.services.logging.redaction import SecretRedactor
from app.services.sandbox import DockerSandbox

from .reader import (
    RepositoryAccessError,
    RepositoryIdentity,
    RepositoryIdentityChanged,
    RepositoryProtocolSnapshotChanged,
    SafeRepositoryReader,
    _git_bytes,
    _git_text,
    _protocol_config,
    _protocol_identity,
    _repository_protocol,
)


class RepositoryToolError(RepositoryAccessError):
    pass


class RepositorySnapshotChanged(RepositoryToolError):
    pass


@dataclass(frozen=True)
class _SafeTreeEntry:
    path: str
    size: int


_TOOL_NAMES = frozenset({"search_files", "read_file"})
_SEARCHABLE_SUFFIXES = frozenset(
    {
        "",
        ".c",
        ".cc",
        ".cpp",
        ".cs",
        ".css",
        ".go",
        ".gradle",
        ".h",
        ".hpp",
        ".html",
        ".java",
        ".js",
        ".json",
        ".jsx",
        ".kt",
        ".kts",
        ".md",
        ".php",
        ".properties",
        ".py",
        ".rb",
        ".rs",
        ".scss",
        ".sh",
        ".sql",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
)


@dataclass(frozen=True)
class RepositoryToolRequest:
    """모델이 요청할 수 있는 두 가지 읽기 전용 저장소 작업."""

    tool: str
    query: str = ""
    path: str = ""
    start_line: int = 1
    end_line: int | None = None

    def __post_init__(self) -> None:
        if self.tool not in _TOOL_NAMES:
            raise ValueError(f"지원하지 않는 저장소 도구입니다: {self.tool}")
        if self.tool == "search_files":
            query = self.query.strip()
            if len(query) < 2 or len(query) > 200 or any(
                character in query for character in ("\x00", "\r", "\n")
            ):
                raise ValueError("검색어는 한 줄의 2~200자여야 합니다.")
            if self.path:
                raise ValueError("search_files에는 path를 사용할 수 없습니다.")
        else:
            if self.query:
                raise ValueError("read_file에는 query를 사용할 수 없습니다.")
            if not SafeRepositoryReader._path_is_safe(self.path):
                raise ValueError("안전하지 않은 저장소 상대경로입니다.")
            if len(self.path) > 500:
                raise ValueError("저장소 상대경로가 너무 깁니다.")
            if self.start_line < 1:
                raise ValueError("start_line은 1 이상이어야 합니다.")
            if self.end_line is not None and self.end_line < self.start_line:
                raise ValueError("end_line은 start_line 이상이어야 합니다.")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RepositoryToolRequest":
        tool = str(value.get("tool", "")).strip()
        start_line = _positive_integer(value.get("start_line", 1), "start_line")
        raw_end = value.get("end_line")
        end_line = (
            None
            if raw_end in (None, "")
            else _positive_integer(raw_end, "end_line")
        )
        return cls(
            tool=tool,
            query=str(value.get("query", "")).strip(),
            path=str(value.get("path", "")).strip(),
            start_line=start_line,
            end_line=end_line,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"tool": self.tool}
        if self.tool == "search_files":
            result["query"] = self.query
        else:
            result.update(
                {
                    "path": self.path,
                    "start_line": self.start_line,
                    "end_line": self.end_line,
                }
            )
        return result


@dataclass(frozen=True)
class RepositoryToolResult:
    request: RepositoryToolRequest
    status: str
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "status": self.status,
            "data": dict(self.data),
            "untrusted_repository_data": True,
        }


@dataclass(frozen=True)
class RepositoryToolBatch:
    identity_hash: str
    head_sha: str
    results: tuple[RepositoryToolResult, ...]
    truncated: bool = False

    @property
    def characters(self) -> int:
        return len(
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": "approved_committed_git_snapshot_tools",
            "untrusted_repository_data": True,
            "identity_hash": self.identity_hash,
            "head_sha": self.head_sha,
            "results": [item.to_dict() for item in self.results],
            "truncated": self.truncated,
        }


class SafeRepositoryToolLayer:
    """모델 요청을 검증하고 승인된 하나의 Git HEAD에서만 실행한다."""

    def __init__(
        self,
        *,
        max_requests: int = 3,
        max_tree_bytes: int = 2_000_000,
        max_search_files: int = 400,
        max_search_bytes: int = 4_000_000,
        max_search_results: int = 20,
        max_blob_bytes: int = 262_144,
        max_read_lines: int = 240,
        max_result_characters: int = 12_000,
        redactor: SecretRedactor | None = None,
        sandbox: DockerSandbox | None = None,
    ):
        limits = (
            max_requests,
            max_tree_bytes,
            max_search_files,
            max_search_bytes,
            max_search_results,
            max_blob_bytes,
            max_read_lines,
            max_result_characters,
        )
        if min(limits) < 1:
            raise ValueError("repository tool limits must be positive")
        self.max_requests = max_requests
        self.max_tree_bytes = max_tree_bytes
        # Kept only for constructor compatibility. The F-2 protocol never
        # stops a search after a number of scanned files.
        self.max_search_files = max_search_files
        self.max_search_bytes = max_search_bytes
        self.max_search_results = max_search_results
        self.max_blob_bytes = max_blob_bytes
        self.max_read_lines = max_read_lines
        self.max_result_characters = max_result_characters
        self.redactor = redactor or SecretRedactor()
        self.sandbox = sandbox or DockerSandbox()

    def execute(
        self,
        repository_path: str | Path,
        requests: tuple[RepositoryToolRequest, ...],
        *,
        expected_identity: str,
        expected_head: str,
        operation_id: str | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> RepositoryToolBatch:
        if not requests:
            raise RepositoryToolError("저장소 도구 요청이 비어 있습니다.")
        if len(requests) > self.max_requests:
            raise RepositoryToolError(
                f"한 번에 저장소 도구를 {self.max_requests}개까지만 사용할 수 있습니다."
            )
        try:
            root = Path(repository_path).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise RepositoryToolError("프로젝트 경로를 확인할 수 없습니다.") from exc
        transport_maximum = max(
            1_000_000,
            (self.max_result_characters * self.max_requests * 4) + 65_536,
        )
        config = _protocol_config(
            root,
            mode="tools",
            requests=[request.to_dict() for request in requests],
            max_tree_bytes=self.max_tree_bytes,
            max_blob_bytes=self.max_blob_bytes,
            max_search_bytes=self.max_search_bytes,
            max_search_results=self.max_search_results,
            max_result_characters=self.max_result_characters,
            max_response_bytes=transport_maximum,
        )
        try:
            raw = _repository_protocol(
                root,
                config,
                maximum=transport_maximum,
                sandbox=self.sandbox,
                operation_id=operation_id,
                component="repository-tools",
                cancelled=cancelled,
            )
        except RepositoryProtocolSnapshotChanged as exc:
            raise RepositorySnapshotChanged(str(exc)) from exc
        identity = _protocol_identity(root, raw)
        if identity.identity_hash != expected_identity:
            raise RepositoryIdentityChanged(
                "저장소 식별값이 승인 당시와 달라 다시 승인이 필요합니다."
            )
        if identity.head_sha != expected_head:
            raise RepositorySnapshotChanged(
                "도구 실행 전에 Git HEAD가 바뀌어 일관된 조회를 중단했습니다."
            )
        raw_results = raw.get("results")
        if not isinstance(raw_results, list) or len(raw_results) != len(requests):
            raise RepositoryToolError("격리 저장소 도구 결과가 올바르지 않습니다.")
        results: list[RepositoryToolResult] = []
        used = 0
        truncated = False
        for request, raw_result in zip(requests, raw_results, strict=True):
            remaining = self.max_result_characters - used
            if remaining < 200:
                truncated = True
                break
            if not isinstance(raw_result, dict) or raw_result.get("tool") != request.tool:
                raise RepositoryToolError("격리 저장소 도구 결과가 요청과 일치하지 않습니다.")
            result = (
                self._render_protocol_search(request, raw_result, remaining)
                if request.tool == "search_files"
                else self._render_protocol_read(request, raw_result, remaining)
            )
            results.append(result)
            used += len(
                json.dumps(
                    result.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            truncated = truncated or bool(result.data.get("truncated", False))
        return RepositoryToolBatch(
            identity_hash=identity.identity_hash,
            head_sha=identity.head_sha,
            results=tuple(results),
            truncated=truncated or len(results) < len(requests),
        )

    def _render_protocol_search(
        self,
        request: RepositoryToolRequest,
        raw_result: dict[str, Any],
        character_limit: int,
    ) -> RepositoryToolResult:
        raw_data = raw_result.get("data")
        if not isinstance(raw_data, dict) or not isinstance(raw_data.get("matches"), list):
            raise RepositoryToolError("격리 검색 결과가 올바르지 않습니다.")
        matches: list[dict[str, Any]] = []
        truncated = bool(raw_data.get("truncated", False))
        reasons = raw_data.get("truncation_reasons", [])
        if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
            raise RepositoryToolError("격리 검색 제한 정보가 올바르지 않습니다.")

        def add_match(value: dict[str, Any]) -> bool:
            projected = len(
                json.dumps(matches + [value], ensure_ascii=False, separators=(",", ":"))
            )
            if len(matches) >= self.max_search_results or projected > character_limit:
                return False
            matches.append(value)
            return True

        for raw_match in raw_data["matches"]:
            if not isinstance(raw_match, dict) or not isinstance(raw_match.get("path"), str):
                raise RepositoryToolError("격리 검색 일치 결과가 올바르지 않습니다.")
            path = raw_match["path"]
            kind = raw_match.get("kind")
            if not SafeRepositoryReader._path_is_safe(path) or kind not in {"path", "content"}:
                raise RepositoryToolError("격리 검색 일치 결과가 올바르지 않습니다.")
            if kind == "path":
                value = {"path": path, "kind": "path", "line": None, "excerpt": ""}
            else:
                try:
                    line = int(raw_match["line"])
                    excerpt = base64.b64decode(raw_match["excerpt_b64"], validate=True).decode("utf-8")
                except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
                    raise RepositoryToolError("격리 검색 본문 결과가 올바르지 않습니다.") from exc
                if line < 1:
                    raise RepositoryToolError("격리 검색 본문 줄 번호가 올바르지 않습니다.")
                value = {
                    "path": path,
                    "kind": "content",
                    "line": line,
                    "excerpt": self.redactor.text(excerpt.strip())[:400],
                }
            if not add_match(value):
                truncated = True
                reasons = [*reasons, "result_character_limit"]
                break
        excluded = raw_data.get("excluded", {})
        if not isinstance(excluded, dict) or not all(
            isinstance(key, str) and isinstance(value, (int, str))
            for key, value in excluded.items()
        ):
            raise RepositoryToolError("격리 검색 제외 정보가 올바르지 않습니다.")
        searched_files = raw_data.get("searched_tracked_files", 0)
        if isinstance(searched_files, bool) or not isinstance(searched_files, int) or searched_files < 0:
            raise RepositoryToolError("격리 검색 파일 수가 올바르지 않습니다.")
        return RepositoryToolResult(
            request,
            "ok",
            {
                "matches": matches,
                "searched_tracked_files": searched_files,
                "truncated": truncated,
                "truncation_reasons": list(dict.fromkeys(reasons)),
                "excluded": dict(excluded),
            },
        )

    def _render_protocol_read(
        self,
        request: RepositoryToolRequest,
        raw_result: dict[str, Any],
        character_limit: int,
    ) -> RepositoryToolResult:
        if raw_result.get("status") == "truncated":
            reason = raw_result.get("reason")
            if not isinstance(reason, str) or not reason:
                raise RepositoryToolError("격리 상세 읽기 제한 사유가 올바르지 않습니다.")
            return RepositoryToolResult(
                request,
                "truncated",
                {
                    "error": "일괄 상세 읽기 응답 크기 제한으로 파일을 생략했습니다.",
                    "reason": reason,
                    "truncated": True,
                },
            )
        if raw_result.get("status") == "denied":
            reason = raw_result.get("reason")
            message = (
                "파일 크기가 안전 제한을 초과했습니다."
                if reason == "file_size_limit"
                else "승인된 HEAD에 안전하게 읽을 수 있는 해당 파일이 없습니다."
            )
            return RepositoryToolResult(request, "denied", {"error": message, "reason": reason})
        if raw_result.get("status") != "ok" or raw_result.get("path") != request.path:
            raise RepositoryToolError("격리 상세 읽기 결과가 올바르지 않습니다.")
        try:
            data = base64.b64decode(raw_result["data"], validate=True)
            if b"\0" in data[:4096]:
                raise UnicodeDecodeError("utf-8", data, 0, 1, "binary")
            text = data.decode("utf-8")
        except (KeyError, TypeError, ValueError, UnicodeDecodeError):
            return RepositoryToolResult(
                request,
                "denied",
                {"error": "바이너리이거나 UTF-8 텍스트가 아닙니다.", "reason": "binary_or_non_utf8"},
            )
        lines = text.splitlines()
        start = request.start_line
        requested_end = request.end_line or (start + self.max_read_lines - 1)
        end = min(requested_end, start + self.max_read_lines - 1, len(lines))
        if start > len(lines) and lines:
            return RepositoryToolResult(
                request,
                "error",
                {"error": f"파일은 {len(lines)}줄인데 {start}줄부터 요청했습니다."},
            )
        selected: list[str] = []
        truncated = requested_end > end
        for line_number in range(start, end + 1):
            raw_line = lines[line_number - 1] if lines else ""
            rendered = f"{line_number}: {self.redactor.text(raw_line)}"
            if len("\n".join(selected + [rendered])) > character_limit:
                truncated = True
                break
            selected.append(rendered)
        actual_end = start + len(selected) - 1 if selected else start - 1
        return RepositoryToolResult(
            request,
            "ok",
            {
                "path": request.path,
                "start_line": start,
                "end_line": actual_end,
                "total_lines": len(lines),
                "content": "\n".join(selected),
                "truncated": truncated,
            },
        )

    def _safe_entries(
        self,
        identity: RepositoryIdentity,
        *,
        operation_id: str | None,
        cancelled: Callable[[], bool] | None,
    ) -> tuple[_SafeTreeEntry, ...]:
        raw = _git_bytes(
            identity.root,
            "ls-tree",
            "-r",
            "-z",
            "-l",
            "--full-tree",
            identity.head_sha,
            maximum=self.max_tree_bytes,
            sandbox=self.sandbox,
            operation_id=operation_id,
            component="repository-tools",
            cancelled=cancelled,
        )
        entries: list[_SafeTreeEntry] = []
        try:
            for item in raw.split(b"\0"):
                if not item:
                    continue
                metadata, separator, raw_path = item.partition(b"\t")
                fields = metadata.split()
                if not separator or len(fields) != 4 or fields[1] != b"blob":
                    continue
                path = raw_path.decode("utf-8", errors="strict")
                if "\n" in path or "\r" in path:
                    continue
                size = int(fields[3])
                if SafeRepositoryReader._path_is_safe(path):
                    entries.append(_SafeTreeEntry(path=path, size=size))
        except (UnicodeDecodeError, ValueError) as exc:
            raise RepositoryToolError(
                "Git 트리의 파일 이름 또는 크기 정보가 올바르지 않습니다."
            ) from exc
        return tuple(entries)

    def _search(
        self,
        identity: RepositoryIdentity,
        safe_entries: tuple[_SafeTreeEntry, ...],
        request: RepositoryToolRequest,
        character_limit: int,
        *,
        operation_id: str | None,
        cancelled: Callable[[], bool] | None,
    ) -> RepositoryToolResult:
        needle = request.query.casefold()
        ordered = sorted(
            safe_entries,
            key=lambda entry: (
                needle not in entry.path.casefold(),
                PurePosixPath(entry.path).suffix.casefold()
                not in _SEARCHABLE_SUFFIXES,
                len(entry.path),
                entry.path.casefold(),
            ),
        )
        matches: list[dict[str, Any]] = []
        scanned_files = 0
        scanned_bytes = 0
        truncated = False

        def add_match(value: dict[str, Any]) -> bool:
            projected = len(
                json.dumps(
                    matches + [value], ensure_ascii=False, separators=(",", ":")
                )
            )
            if len(matches) >= self.max_search_results or projected > character_limit:
                return False
            matches.append(value)
            return True

        candidates: list[_SafeTreeEntry] = []
        for entry in ordered:
            if (
                scanned_files >= self.max_search_files
                or scanned_bytes >= self.max_search_bytes
            ):
                truncated = True
                break
            path = entry.path
            if needle in path.casefold() and not add_match(
                {"path": path, "kind": "path", "line": None, "excerpt": ""}
            ):
                truncated = True
                break
            if PurePosixPath(path).suffix.casefold() not in _SEARCHABLE_SUFFIXES:
                continue
            scanned_files += 1
            if entry.size > self.max_blob_bytes:
                continue
            if scanned_bytes + entry.size > self.max_search_bytes:
                truncated = True
                break
            scanned_bytes += entry.size
            candidates.append(entry)

        for entry, text in self._batch_read_text(
            identity,
            candidates,
            operation_id=operation_id,
            cancelled=cancelled,
        ):
            if text is None:
                continue
            hits_in_file = 0
            for line_number, line in enumerate(text.splitlines(), start=1):
                if needle not in line.casefold():
                    continue
                excerpt = self.redactor.text(line.strip())[:400]
                if not add_match(
                    {
                        "path": entry.path,
                        "kind": "content",
                        "line": line_number,
                        "excerpt": excerpt,
                    }
                ):
                    truncated = True
                    break
                hits_in_file += 1
                if hits_in_file >= 3:
                    break
            if truncated or len(matches) >= self.max_search_results:
                truncated = True
                break
        return RepositoryToolResult(
            request,
            "ok",
            {
                "matches": matches,
                "scanned_files": scanned_files,
                "truncated": truncated,
            },
        )

    def _read(
        self,
        identity: RepositoryIdentity,
        safe_entries: tuple[_SafeTreeEntry, ...],
        request: RepositoryToolRequest,
        character_limit: int,
        *,
        operation_id: str | None,
        cancelled: Callable[[], bool] | None,
    ) -> RepositoryToolResult:
        entry_by_path = {entry.path: entry for entry in safe_entries}
        entry = entry_by_path.get(request.path)
        if entry is None:
            return RepositoryToolResult(
                request,
                "denied",
                {"error": "승인된 HEAD에 안전하게 읽을 수 있는 해당 파일이 없습니다."},
            )
        text, _size = self._read_text(
            identity,
            request.path,
            known_size=entry.size,
            operation_id=operation_id,
            cancelled=cancelled,
        )
        if text is None:
            return RepositoryToolResult(
                request,
                "denied",
                {"error": "바이너리이거나 안전한 파일 크기 제한을 초과했습니다."},
            )
        lines = text.splitlines()
        start = request.start_line
        requested_end = request.end_line or (start + self.max_read_lines - 1)
        end = min(requested_end, start + self.max_read_lines - 1, len(lines))
        if start > len(lines) and lines:
            return RepositoryToolResult(
                request,
                "error",
                {"error": f"파일은 {len(lines)}줄인데 {start}줄부터 요청했습니다."},
            )
        selected: list[str] = []
        truncated = requested_end > end
        for line_number in range(start, end + 1):
            raw_line = lines[line_number - 1] if lines else ""
            rendered = f"{line_number}: {self.redactor.text(raw_line)}"
            if len("\n".join(selected + [rendered])) > character_limit:
                truncated = True
                break
            selected.append(rendered)
        actual_end = start + len(selected) - 1 if selected else start - 1
        return RepositoryToolResult(
            request,
            "ok",
            {
                "path": request.path,
                "start_line": start,
                "end_line": actual_end,
                "total_lines": len(lines),
                "content": "\n".join(selected),
                "truncated": truncated,
            },
        )

    def _read_text(
        self,
        identity: RepositoryIdentity,
        relative: str,
        *,
        known_size: int | None = None,
        operation_id: str | None,
        cancelled: Callable[[], bool] | None,
    ) -> tuple[str | None, int]:
        object_name = f"{identity.head_sha}:{relative}"
        if known_size is None:
            raw_size = _git_text(
                identity.root,
                "cat-file",
                "-s",
                object_name,
                sandbox=self.sandbox,
                operation_id=operation_id,
                component="repository-tools",
                cancelled=cancelled,
            )
            try:
                size = int(raw_size)
            except ValueError as exc:
                raise RepositoryToolError("Git 파일 크기 결과가 올바르지 않습니다.") from exc
        else:
            size = known_size
        if size > self.max_blob_bytes:
            return None, size
        data = _git_bytes(
            identity.root,
            "show",
            "--no-ext-diff",
            object_name,
            maximum=self.max_blob_bytes,
            sandbox=self.sandbox,
            operation_id=operation_id,
            component="repository-tools",
            cancelled=cancelled,
        )
        if b"\0" in data[:4096]:
            return None, size
        try:
            return data.decode("utf-8"), size
        except UnicodeDecodeError:
            return None, size

    def _batch_read_text(
        self,
        identity: RepositoryIdentity,
        entries: list[_SafeTreeEntry],
        *,
        operation_id: str | None,
        cancelled: Callable[[], bool] | None,
    ) -> list[tuple[_SafeTreeEntry, str | None]]:
        if not entries:
            return []
        request = b"".join(
            f"{identity.head_sha}:{entry.path}\n".encode("utf-8")
            for entry in entries
        )
        maximum = sum(entry.size for entry in entries) + (len(entries) * 128) + 1024
        raw = _git_bytes(
            identity.root,
            "cat-file",
            "--batch",
            maximum=maximum,
            input_data=request,
            sandbox=self.sandbox,
            operation_id=operation_id,
            component="repository-tools",
            cancelled=cancelled,
        )
        cursor = 0
        result: list[tuple[_SafeTreeEntry, str | None]] = []
        for entry in entries:
            header_end = raw.find(b"\n", cursor)
            if header_end < 0:
                raise RepositoryToolError("Git 일괄 파일 조회 결과가 잘렸습니다.")
            fields = raw[cursor:header_end].split()
            if len(fields) != 3 or fields[1] != b"blob":
                raise RepositoryToolError("Git 일괄 파일 조회 결과가 올바르지 않습니다.")
            try:
                size = int(fields[2])
            except ValueError as exc:
                raise RepositoryToolError(
                    "Git 일괄 파일 크기 결과가 올바르지 않습니다."
                ) from exc
            if size != entry.size:
                raise RepositorySnapshotChanged(
                    "조회 중 Git 스냅샷의 파일 크기가 달라져 작업을 중단했습니다."
                )
            content_start = header_end + 1
            content_end = content_start + size
            if content_end >= len(raw) or raw[content_end : content_end + 1] != b"\n":
                raise RepositoryToolError("Git 일괄 파일 내용이 올바르지 않습니다.")
            data = raw[content_start:content_end]
            cursor = content_end + 1
            if b"\0" in data[:4096]:
                result.append((entry, None))
                continue
            try:
                result.append((entry, data.decode("utf-8")))
            except UnicodeDecodeError:
                result.append((entry, None))
        return result


def _positive_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name}은 양의 정수여야 합니다.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name}은 양의 정수여야 합니다.") from exc
    if parsed < 1:
        raise ValueError(f"{field_name}은 양의 정수여야 합니다.")
    return parsed
