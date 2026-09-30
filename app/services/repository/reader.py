from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from app.services.logging.redaction import SecretRedactor
from app.services.process_tree import ProcessTree, isolated_process_options
from app.services.sandbox import DockerSandbox, DockerSandboxError


class RepositoryAccessError(RuntimeError):
    pass


class RepositoryCancelled(RepositoryAccessError):
    pass


class RepositoryTimeout(RepositoryAccessError):
    pass


class RepositoryIdentityChanged(RepositoryAccessError):
    pass


class RepositoryProtocolSnapshotChanged(RepositoryAccessError):
    pass


class RepositoryPinnedSnapshotUnavailable(RepositoryAccessError):
    pass


@dataclass(frozen=True)
class RepositoryIdentity:
    root: Path
    identity_hash: str
    head_sha: str
    branch: str


@dataclass(frozen=True)
class RepositorySnapshotEntry:
    path: str
    size: int
    object_id: str = ""


@dataclass(frozen=True)
class RepositorySnapshotManifest:
    identity_hash: str
    commit_sha: str
    branch: str
    entries: tuple[RepositorySnapshotEntry, ...]
    exclusions: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class RepositoryReadContext:
    identity_hash: str
    head_sha: str
    branch: str
    tree: tuple[str, ...]
    documents: tuple[tuple[str, str], ...]
    truncated: bool = False
    exclusions: tuple[tuple[str, int], ...] = ()

    @property
    def characters(self) -> int:
        return sum(len(path) + len(content) for path, content in self.documents) + sum(
            len(path) for path in self.tree
        )

    def to_dict(self) -> dict:
        return {
            "source": "approved_committed_git_snapshot",
            "untrusted_repository_data": True,
            "identity_hash": self.identity_hash,
            "head_sha": self.head_sha,
            "branch": self.branch,
            "tree": list(self.tree),
            "documents": [
                {"path": path, "content": content}
                for path, content in self.documents
            ],
            "truncated": self.truncated,
            "exclusions": dict(self.exclusions),
        }


_HEX_SHA = re.compile(r"[0-9a-f]{40,64}")
_QUERY_TOKEN = re.compile(r"[A-Za-z0-9_.-]{3,}|[가-힣]{2,}")
_SENSITIVE_PARTS = {
    ".git",
    ".env",
    ".npmrc",
    ".pypirc",
    ".netrc",
    "credentials",
    "credential",
    "secrets",
    "secret",
    "private",
    "id_rsa",
    "id_ed25519",
}
_SENSITIVE_SUFFIXES = {
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".jks",
    ".keystore",
    ".der",
    ".crt",
    ".cer",
}
_FOUNDATION_NAMES = {
    "readme": 100,
    "readme.md": 100,
    "pyproject.toml": 95,
    "package.json": 95,
    "cargo.toml": 95,
    "go.mod": 95,
    "pom.xml": 95,
    "build.gradle": 90,
    "build.gradle.kts": 90,
    "settings.gradle": 85,
    "settings.gradle.kts": 85,
    "requirements.txt": 85,
    "composer.json": 85,
}


def inspect_repository_identity(
    path: Path,
    *,
    sandbox: DockerSandbox | None = None,
    operation_id: str | None = None,
    component: str = "repository-identity",
    cancelled: Callable[[], bool] | None = None,
) -> RepositoryIdentity:
    try:
        root = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RepositoryAccessError("프로젝트 경로를 확인할 수 없습니다.") from exc
    if not root.is_dir():
        raise RepositoryAccessError("프로젝트 경로가 폴더가 아닙니다.")
    result = _repository_protocol(
        root,
        _protocol_config(root, mode="identity"),
        maximum=64_000,
        sandbox=sandbox,
        operation_id=operation_id,
        component=component,
        cancelled=cancelled,
    )
    return _protocol_identity(root, result)


class SafeRepositoryReader:
    """승인된 저장소의 커밋된 HEAD만 제한된 크기로 읽는다."""

    def __init__(
        self,
        *,
        max_tree_entries: int = 200,
        max_tree_bytes: int = 2_000_000,
        max_files: int = 8,
        max_file_bytes: int = 32_768,
        max_total_characters: int = 12_000,
        redactor: SecretRedactor | None = None,
        sandbox: DockerSandbox | None = None,
    ):
        if min(
            max_tree_entries,
            max_tree_bytes,
            max_files,
            max_file_bytes,
            max_total_characters,
        ) < 1:
            raise ValueError("repository reader limits must be positive")
        self.max_tree_entries = max_tree_entries
        self.max_tree_bytes = max_tree_bytes
        self.max_files = max_files
        self.max_file_bytes = max_file_bytes
        self.max_total_characters = max_total_characters
        self.redactor = redactor or SecretRedactor()
        self.sandbox = sandbox or DockerSandbox()

    @staticmethod
    def should_inspect(query: str) -> bool:
        normalized = query.casefold()
        if re.search(r"(?:란|이란|뜻|개념|무엇|뭐야|뭔가요|없을\s*때|일반적으로)", normalized):
            return False
        if re.search(r"(?:왜|어떻게).*?장기\s*(?:저장소\s*)?분석.*?(?:들어가|등록|분류|처리)", normalized):
            return False
        if re.search(r"(?:이|그|우리|현재|지금|선택한)\s*(?:프로젝트|저장소|레포|코드|파일|구조|함수|클래스)", normalized):
            return True
        if re.search(r"(?:readme|package\.json|pyproject|cargo\.toml|[\w./\\-]+\.(?:py|js|ts|tsx|java|go|rs|md|json|yaml|yml))", normalized):
            return True
        if not re.search(r"(?:읽어|열어|찾아|검색|확인|분석|검토|살펴|조사|점검|보여|봐줘|알려|정리)", normalized):
            return False
        markers = (
            "프로젝트",
            "저장소",
            "레포",
            "코드",
            "파일",
            "폴더",
            "구조",
            "소스",
            "함수",
            "클래스",
            "메서드",
            "구현",
            "버그",
            "리팩터링",
            "동작",
            "readme",
            "package.json",
            "pyproject",
            "cargo.toml",
        )
        return any(marker in normalized for marker in markers)

    def inspect(
        self,
        repository_path: str | Path,
        query: str,
        *,
        expected_identity: str,
        operation_id: str | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> RepositoryReadContext:
        if not expected_identity.strip():
            raise RepositoryAccessError("프로젝트 읽기 승인이 없습니다.")
        try:
            root = Path(repository_path).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise RepositoryAccessError("프로젝트 경로를 확인할 수 없습니다.") from exc
        config = _protocol_config(
            root,
            mode="inspect",
            query=query,
            max_tree_bytes=self.max_tree_bytes,
            max_files=self.max_files,
            max_file_bytes=self.max_file_bytes,
            query_tokens=tuple(item.casefold() for item in _QUERY_TOKEN.findall(query)),
        )
        result = _repository_protocol(
            root,
            config,
            maximum=self.max_tree_bytes + (self.max_files * self.max_file_bytes * 2) + 1_000_000,
            sandbox=self.sandbox,
            operation_id=operation_id,
            component="repository-reader",
            cancelled=cancelled,
        )
        identity = _protocol_identity(root, result)
        if identity.identity_hash != expected_identity:
            raise RepositoryIdentityChanged(
                "저장소 식별값이 승인 당시와 달라 다시 승인이 필요합니다."
            )
        raw_paths = result.get("paths")
        if not isinstance(raw_paths, list) or not all(
            isinstance(path, str) and self._path_is_safe(path) for path in raw_paths
        ):
            raise RepositoryAccessError("Git 트리 조회 결과가 올바르지 않습니다.")
        safe_paths = tuple(raw_paths)
        visible_tree_list: list[str] = []
        tree_characters = 0
        tree_limit = min(4_000, self.max_total_characters // 3)
        tree_limit_reason = ""
        for relative in safe_paths:
            if len(visible_tree_list) >= self.max_tree_entries:
                tree_limit_reason = "tree_entry_limit"
                break
            if tree_characters + len(relative) > tree_limit:
                tree_limit_reason = "tree_character_limit"
                break
            visible_tree_list.append(relative)
            tree_characters += len(relative)
        visible_tree = tuple(visible_tree_list)
        truncated = bool(tree_limit_reason)
        documents: list[tuple[str, str]] = []
        exclusions = _protocol_exclusions(result)
        if tree_limit_reason:
            exclusions[tree_limit_reason] = len(safe_paths) - len(visible_tree)
        used = tree_characters
        raw_blobs = result.get("document_blobs")
        if not isinstance(raw_blobs, list):
            raise RepositoryAccessError("Git 문서 조회 결과가 올바르지 않습니다.")
        seen_documents: set[str] = set()
        for blob in raw_blobs:
            if not isinstance(blob, dict) or not isinstance(blob.get("path"), str) or not isinstance(blob.get("data"), str):
                raise RepositoryAccessError("Git 문서 조회 결과가 올바르지 않습니다.")
            relative = blob["path"]
            if relative not in safe_paths or relative in seen_documents:
                raise RepositoryAccessError("Git 문서 경로가 트리 스냅샷과 일치하지 않습니다.")
            seen_documents.add(relative)
            try:
                data = base64.b64decode(blob["data"], validate=True)
                if len(data) > self.max_file_bytes:
                    exclusions["document_file_size_limit"] = exclusions.get("document_file_size_limit", 0) + 1
                    continue
                if b"\0" in data[:4096]:
                    raise UnicodeDecodeError("utf-8", data, 0, 1, "binary")
                content = self.redactor.text(data.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                exclusions["binary_or_non_utf8"] = exclusions.get("binary_or_non_utf8", 0) + 1
                continue
            remaining = self.max_total_characters - used
            if remaining <= 0:
                truncated = True
                exclusions["context_character_limit"] = exclusions.get("context_character_limit", 0) + 1
                break
            if len(content) > remaining:
                notice = "\n[내용 크기 제한으로 생략됨]"
                content = content[: max(0, remaining - len(notice))] + notice
                truncated = True
                exclusions["context_character_limit"] = exclusions.get("context_character_limit", 0) + 1
            documents.append((relative, content))
            used += len(content)
        truncated = truncated or any(
            exclusions.get(reason, 0) > 0
            for reason in (
                "document_file_size_limit",
                "document_selection_limit",
                "binary_or_non_utf8",
                "context_character_limit",
            )
        )
        return RepositoryReadContext(
            identity_hash=identity.identity_hash,
            head_sha=identity.head_sha,
            branch=identity.branch,
            tree=visible_tree,
            documents=tuple(documents),
            truncated=truncated,
            exclusions=tuple(sorted(exclusions.items())),
        )

    def pinned_manifest(
        self,
        repository_path: str | Path,
        *,
        expected_identity: str,
        commit_sha: str,
        operation_id: str | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> RepositorySnapshotManifest:
        """List every safe tracked blob from one immutable approved commit."""
        if not expected_identity.strip() or not _HEX_SHA.fullmatch(commit_sha):
            raise RepositoryAccessError("고정 저장소 스냅샷 정보가 올바르지 않습니다.")
        try:
            root = Path(repository_path).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise RepositoryAccessError("프로젝트 경로를 확인할 수 없습니다.") from exc
        identity = inspect_repository_identity(
            root,
            sandbox=self.sandbox,
            operation_id=operation_id,
            component="repository-analysis-manifest",
            cancelled=cancelled,
        )
        if identity.identity_hash != expected_identity:
            raise RepositoryIdentityChanged(
                "저장소 식별값이 승인 당시와 달라 다시 승인이 필요합니다."
            )
        try:
            _git_bytes(
                root,
                "cat-file",
                "-e",
                f"{commit_sha}^{{commit}}",
                maximum=128,
                sandbox=self.sandbox,
                operation_id=operation_id,
                component="repository-analysis-manifest",
                cancelled=cancelled,
            )
        except RepositoryAccessError as exc:
            raise RepositoryPinnedSnapshotUnavailable(
                "분석을 시작한 고정 commit object를 더 이상 읽을 수 없습니다."
            ) from exc
        raw = _git_bytes(
            root,
            "ls-tree",
            "-r",
            "-z",
            "-l",
            "--full-tree",
            commit_sha,
            maximum=self.max_tree_bytes,
            sandbox=self.sandbox,
            operation_id=operation_id,
            component="repository-analysis-manifest",
            cancelled=cancelled,
        )
        entries: list[RepositorySnapshotEntry] = []
        exclusions: dict[str, int] = {}
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
                if "\n" in path or "\r" in path or not self._path_is_safe(path):
                    exclusions["unsafe_or_sensitive_path"] = (
                        exclusions.get("unsafe_or_sensitive_path", 0) + 1
                    )
                    continue
                if size < 0:
                    raise ValueError("negative blob size")
                object_id = fields[2].decode("ascii", errors="strict")
                if not _HEX_SHA.fullmatch(object_id):
                    raise ValueError("invalid blob object id")
                entries.append(RepositorySnapshotEntry(path, size, object_id))
        except (UnicodeDecodeError, ValueError) as exc:
            raise RepositoryAccessError(
                "고정 Git 트리의 파일 이름 또는 크기 정보가 올바르지 않습니다."
            ) from exc
        return RepositorySnapshotManifest(
            identity_hash=identity.identity_hash,
            commit_sha=commit_sha,
            branch=identity.branch,
            entries=tuple(sorted(entries, key=lambda item: item.path.casefold())),
            exclusions=tuple(sorted(exclusions.items())),
        )

    def read_pinned_files(
        self,
        repository_path: str | Path,
        manifest: RepositorySnapshotManifest,
        paths: tuple[str, ...],
        *,
        max_total_bytes: int,
        max_file_bytes: int,
        operation_id: str | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[tuple[str, str], ...]:
        """Read only explicitly planned safe paths from the manifest's fixed commit."""
        documents, _excluded = self.read_pinned_files_with_exclusions(
            repository_path, manifest, paths, max_total_bytes=max_total_bytes,
            max_file_bytes=max_file_bytes, operation_id=operation_id,
            cancelled=cancelled,
        )
        return documents

    def read_pinned_files_with_exclusions(
        self,
        repository_path: str | Path,
        manifest: RepositorySnapshotManifest,
        paths: tuple[str, ...],
        *,
        max_total_bytes: int,
        max_file_bytes: int,
        operation_id: str | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
        """Read one batch in one isolated process and report per-file text exclusions."""
        if max_total_bytes < 1 or max_file_bytes < 1:
            raise ValueError("pinned read limits must be positive")
        if not paths or len(set(paths)) != len(paths):
            raise ValueError("pinned read paths must be unique and non-empty")
        try:
            root = Path(repository_path).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise RepositoryAccessError("프로젝트 경로를 확인할 수 없습니다.") from exc
        entry_by_path = {entry.path: entry for entry in manifest.entries}
        selected: list[RepositorySnapshotEntry] = []
        total = 0
        for path in paths:
            entry = entry_by_path.get(path)
            if entry is None or entry.size > max_file_bytes:
                raise RepositoryAccessError("고정 스냅샷에서 안전하게 읽을 수 없는 파일입니다.")
            total += entry.size
            if total > max_total_bytes:
                raise RepositoryAccessError("고정 스냅샷 묶음의 읽기 크기 제한을 초과했습니다.")
            selected.append(entry)
        result = _repository_protocol(
            root,
            {
                "mode": "pinned_read", "identity_root": str(root),
                "expected_identity": manifest.identity_hash,
                "commit_sha": manifest.commit_sha,
                "entries": [
                    {"path": entry.path, "size": entry.size, "object_id": entry.object_id}
                    for entry in selected
                ],
                "max_git_bytes": max_total_bytes + len(selected) * 160 + 1024,
                "policy": {
                    "sensitive_parts": sorted(_SENSITIVE_PARTS),
                    "sensitive_suffixes": sorted(_SENSITIVE_SUFFIXES),
                },
            },
            maximum=max_total_bytes * 2 + len(selected) * 256 + 4096,
            sandbox=self.sandbox, operation_id=operation_id,
            component="repository-analysis-read", cancelled=cancelled,
        )
        if result.get("identity_hash") != manifest.identity_hash:
            raise RepositoryIdentityChanged("저장소 식별값이 분석 시작 당시와 달라 읽기를 중단했습니다.")
        if result.get("commit_sha") != manifest.commit_sha:
            raise RepositoryPinnedSnapshotUnavailable("고정 commit을 읽을 수 없습니다.")
        rows = result.get("results")
        if not isinstance(rows, list) or len(rows) != len(selected):
            raise RepositoryAccessError("고정 스냅샷 묶음 결과가 올바르지 않습니다.")
        documents: list[tuple[str, str]] = []
        excluded: list[tuple[str, str]] = []
        for entry, row in zip(selected, rows):
            if not isinstance(row, dict) or row.get("path") != entry.path:
                raise RepositoryAccessError("고정 스냅샷 파일 순서가 달라졌습니다.")
            if row.get("status") == "excluded":
                excluded.append((entry.path, str(row.get("reason", "unreadable"))))
                continue
            if row.get("status") != "ok":
                raise RepositoryAccessError("고정 스냅샷 파일 결과가 올바르지 않습니다.")
            try:
                data = base64.b64decode(str(row["data"]), validate=True)
                if len(data) != entry.size:
                    raise ValueError("size mismatch")
                documents.append((entry.path, self.redactor.text(data.decode("utf-8"))))
            except (KeyError, ValueError, UnicodeDecodeError) as exc:
                raise RepositoryAccessError("고정 스냅샷 파일 결과가 손상되었습니다.") from exc
        return tuple(documents), tuple(excluded)

    def _read_blob(
        self,
        identity: RepositoryIdentity,
        relative: str,
        *,
        operation_id: str | None,
        cancelled: Callable[[], bool] | None,
    ) -> str | None:
        object_name = f"{identity.head_sha}:{relative}"
        raw_size = _git_text(
            identity.root,
            "cat-file",
            "-s",
            object_name,
            sandbox=self.sandbox,
            operation_id=operation_id,
            component="repository-reader",
            cancelled=cancelled,
        )
        try:
            size = int(raw_size)
        except ValueError as exc:
            raise RepositoryAccessError("Git 파일 크기 결과가 올바르지 않습니다.") from exc
        if size > self.max_file_bytes:
            return None
        data = _git_bytes(
            identity.root,
            "show",
            "--no-ext-diff",
            object_name,
            maximum=self.max_file_bytes,
            sandbox=self.sandbox,
            operation_id=operation_id,
            component="repository-reader",
            cancelled=cancelled,
        )
        if b"\0" in data[:4096]:
            return None
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return None
        return self.redactor.text(text)

    @staticmethod
    def _path_is_safe(relative: str) -> bool:
        if (
            not relative
            or "\\" in relative
            or any(ord(character) < 32 for character in relative)
        ):
            return False
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts:
            return False
        lowered = tuple(part.casefold() for part in path.parts)
        if any(part in _SENSITIVE_PARTS for part in lowered):
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
            or any(
            marker in name for marker in ("credential", "secret", "private_key")
            )
        ):
            return False
        return PurePosixPath(name).suffix.casefold() not in _SENSITIVE_SUFFIXES

    @staticmethod
    def _rank_paths(paths: Iterable[str], query: str) -> list[str]:
        tokens = {item.casefold() for item in _QUERY_TOKEN.findall(query)}
        scored: list[tuple[int, str]] = []
        for relative in paths:
            lowered = relative.casefold()
            name = PurePosixPath(lowered).name
            score = _FOUNDATION_NAMES.get(name, 0)
            score += sum(30 for token in tokens if token in lowered)
            if score:
                scored.append((score, relative))
            elif any(
                marker in query.casefold()
                for marker in ("코드", "소스", "파일", "구조", "프로젝트")
            ) and PurePosixPath(name).suffix in {
                ".py",
                ".js",
                ".ts",
                ".tsx",
                ".jsx",
                ".java",
                ".kt",
                ".rs",
                ".go",
                ".cs",
            }:
                scored.append((10, relative))
        scored.sort(key=lambda item: (-item[0], len(item[1]), item[1].casefold()))
        return [relative for _score, relative in scored]


def _remote_identity(remote: str) -> str:
    value = remote.strip()
    value = re.sub(r"(?i)(https?://)[^/@\s]+@", r"\1", value)
    value = re.sub(r"(?i)(ssh://)[^/@\s]+@", r"\1", value)
    return value.casefold()


def _protocol_config(
    root: Path,
    *,
    mode: str,
    query: str = "",
    requests: list[dict[str, Any]] | None = None,
    max_tree_bytes: int = 2_000_000,
    max_files: int = 8,
    max_file_bytes: int = 32_768,
    max_blob_bytes: int = 262_144,
    max_search_bytes: int = 4_000_000,
    max_search_results: int = 20,
    max_result_characters: int = 12_000,
    max_response_bytes: int = 1_000_000,
    query_tokens: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "mode": mode,
        "identity_root": str(root),
        "query": query,
        "query_tokens": list(query_tokens),
        "requests": requests or [],
        "limits": {
            "max_tree_bytes": max_tree_bytes,
            "max_files": max_files,
            "max_file_bytes": max_file_bytes,
            "max_blob_bytes": max_blob_bytes,
            "max_search_bytes": max_search_bytes,
            "max_search_results": max_search_results,
            "max_result_characters": max_result_characters,
            "max_response_bytes": max_response_bytes,
        },
        "policy": {
            "sensitive_parts": sorted(_SENSITIVE_PARTS),
            "sensitive_suffixes": sorted(_SENSITIVE_SUFFIXES),
            "foundation_names": dict(_FOUNDATION_NAMES),
        },
    }


def _protocol_identity(root: Path, result: dict[str, Any]) -> RepositoryIdentity:
    identity_hash = result.get("identity_hash")
    head = result.get("head_sha")
    branch = result.get("branch")
    if (
        not isinstance(identity_hash, str)
        or not _HEX_SHA.fullmatch(identity_hash)
        or not isinstance(head, str)
        or not _HEX_SHA.fullmatch(head)
        or not isinstance(branch, str)
    ):
        raise RepositoryAccessError("Git 저장소 식별 조회 결과가 올바르지 않습니다.")
    return RepositoryIdentity(root, identity_hash, head, branch)


def _protocol_exclusions(result: dict[str, Any]) -> dict[str, int]:
    raw = result.get("exclusions", {})
    if not isinstance(raw, dict):
        raise RepositoryAccessError("Git 조회 제외 사유가 올바르지 않습니다.")
    exclusions: dict[str, int] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, int) or value < 0:
            raise RepositoryAccessError("Git 조회 제외 사유가 올바르지 않습니다.")
        exclusions[key] = value
    return exclusions


def _repository_protocol(
    root: Path,
    config: dict[str, Any],
    *,
    maximum: int,
    sandbox: DockerSandbox | None,
    operation_id: str | None,
    component: str,
    cancelled: Callable[[], bool] | None,
) -> dict[str, Any]:
    source = Path(__file__).with_name("protocol.py").read_text(encoding="utf-8")
    request = json.dumps(config, ensure_ascii=False, separators=(",", ":"))
    for attempt in range(2):
        try:
            raw = _protocol_bytes(
                root,
                source,
                request,
                maximum=maximum,
                sandbox=sandbox,
                operation_id=operation_id,
                component=component,
                cancelled=cancelled,
                timeout_seconds=(
                    min(60, max(20, len(config.get("entries", [])) // 2))
                    if config.get("mode") == "pinned_read" else
                    60 if config.get("mode") == "tools" else 20
                ),
            )
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RepositoryAccessError("격리 저장소 조회 결과를 해석하지 못했습니다.") from exc
        if attempt == 0 and isinstance(payload, dict) and payload.get("kind") == "request_decode":
            continue
        break
    if not isinstance(payload, dict) or not isinstance(payload.get("ok"), bool):
        raise RepositoryAccessError("격리 저장소 조회 결과가 올바르지 않습니다.")
    if not payload["ok"]:
        message = payload.get("error")
        if not isinstance(message, str) or not message:
            message = "격리 저장소 조회를 완료하지 못했습니다."
        if payload.get("kind") == "snapshot":
            raise RepositoryProtocolSnapshotChanged(message)
        if payload.get("kind") == "identity":
            raise RepositoryIdentityChanged(message)
        if payload.get("kind") == "unavailable":
            raise RepositoryPinnedSnapshotUnavailable(message)
        raise RepositoryAccessError(message)
    result = payload.get("result")
    if not isinstance(result, dict):
        raise RepositoryAccessError("격리 저장소 조회 결과가 올바르지 않습니다.")
    return result


def _protocol_bytes(
    root: Path,
    source: str,
    request: str,
    *,
    maximum: int,
    sandbox: DockerSandbox | None,
    operation_id: str | None,
    component: str,
    cancelled: Callable[[], bool] | None,
    timeout_seconds: int = 20,
) -> bytes:
    sandbox = sandbox or DockerSandbox()
    try:
        process = sandbox.popen(
            root,
            ["python3", "-I", "-S", "-c", source],
            writable_workspace=False,
            interactive=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            operation_id=operation_id,
            component=component,
            **isolated_process_options(),
        )
    except DockerSandboxError as exc:
        raise RepositoryAccessError("격리 컨테이너에서 Git 읽기 작업을 완료하지 못했습니다.") from exc

    stdout = bytearray()
    stderr = bytearray()
    overflow = threading.Event()
    try:
        process_tree = ProcessTree(process)
    except BaseException:
        sandbox.cleanup_process(process, reason="process-tree-initialization-failed")
        raise
    cleanup_reason = "completed"

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
                    process_tree.terminate()
                    return
        finally:
            stream.close()

    assert process.stdout is not None and process.stderr is not None
    assert process.stdin is not None
    writer_error: list[OSError] = []

    def send_request() -> None:
        try:
            process.stdin.write(request.encode("utf-8"))
            process.stdin.close()
        except OSError as exc:
            writer_error.append(exc)
            process.stdin.close()

    request_thread = threading.Thread(target=send_request, daemon=True)
    stdout_thread = threading.Thread(
        target=drain, args=(process.stdout, stdout, maximum), daemon=True
    )
    stderr_thread = threading.Thread(
        target=drain,
        args=(process.stderr, stderr, min(maximum, 65_536)),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    request_thread.start()
    started = time.monotonic()
    try:
        while True:
            try:
                return_code = process.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                if cancelled is not None and cancelled():
                    cleanup_reason = "cancelled"
                    sandbox.note_process_termination(process, reason=cleanup_reason)
                    process_tree.terminate()
                    raise RepositoryCancelled("사용자가 저장소 조회를 중지했습니다.")
                if time.monotonic() - started >= timeout_seconds:
                    cleanup_reason = "timeout"
                    sandbox.note_process_termination(process, reason=cleanup_reason)
                    process_tree.terminate()
                    raise RepositoryTimeout("Git 읽기 작업 시간이 초과되었습니다.")
    except BaseException:
        if process.poll() is None:
            sandbox.note_process_termination(process, reason=cleanup_reason)
            process_tree.terminate()
        raise
    finally:
        process_tree.close()
        sandbox.cleanup_process(process, reason=cleanup_reason)
        request_thread.join(timeout=5)
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
    if request_thread.is_alive() or stdout_thread.is_alive() or stderr_thread.is_alive():
        raise RepositoryAccessError("Git 출력 수집을 안전하게 완료하지 못했습니다.")
    if writer_error:
        raise RepositoryAccessError("격리 저장소 조회 요청을 안전하게 전달하지 못했습니다.") from writer_error[0]
    if overflow.is_set():
        raise RepositoryAccessError("격리 저장소 조회 결과가 안전 크기 제한을 초과했습니다.")
    if return_code != 0:
        detail = bytes(stderr).decode("utf-8", errors="replace").strip()
        raise RepositoryAccessError(f"격리 저장소 조회가 실패했습니다: {detail[:300]}")
    return bytes(stdout)


def _git_text(
    root: Path,
    *arguments: str,
    allow_empty: bool = False,
    sandbox: DockerSandbox | None = None,
    operation_id: str | None = None,
    component: str = "repository-reader",
    cancelled: Callable[[], bool] | None = None,
) -> str:
    data = _git_bytes(
        root,
        *arguments,
        maximum=2_000_000,
        allow_failure=allow_empty,
        sandbox=sandbox,
        operation_id=operation_id,
        component=component,
        cancelled=cancelled,
    )
    value = data.decode("utf-8", errors="replace").strip()
    if not value and not allow_empty:
        raise RepositoryAccessError(
            f"Git 조회 결과가 비어 있습니다: {' '.join(arguments)}"
        )
    return value


def _git_bytes(
    root: Path,
    *arguments: str,
    maximum: int,
    allow_failure: bool = False,
    input_data: bytes | None = None,
    sandbox: DockerSandbox | None = None,
    operation_id: str | None = None,
    component: str = "repository-reader",
    cancelled: Callable[[], bool] | None = None,
) -> bytes:
    sandbox = sandbox or DockerSandbox()
    try:
        process = sandbox.popen(
            root,
            [
                "git",
                "--no-replace-objects",
                "-c",
                "core.pager=cat",
                *arguments,
            ],
            writable_workspace=False,
            interactive=input_data is not None,
            stdin=subprocess.PIPE if input_data is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            operation_id=operation_id,
            component=component,
            **isolated_process_options(),
        )
    except DockerSandboxError as exc:
        raise RepositoryAccessError("격리 컨테이너에서 Git 읽기 작업을 완료하지 못했습니다.") from exc

    stdout = bytearray()
    stderr = bytearray()
    overflow = threading.Event()
    try:
        process_tree = ProcessTree(process)
    except BaseException:
        sandbox.cleanup_process(process, reason="process-tree-initialization-failed")
        raise
    cleanup_reason = "completed"

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
                    process_tree.terminate()
                    return
        finally:
            stream.close()

    assert process.stdout is not None and process.stderr is not None
    stdout_thread = threading.Thread(
        target=drain, args=(process.stdout, stdout, maximum), daemon=True
    )
    stderr_thread = threading.Thread(
        target=drain,
        args=(process.stderr, stderr, min(maximum, 65_536)),
        daemon=True,
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
    started = time.monotonic()
    try:
        while True:
            try:
                return_code = process.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                if cancelled is not None and cancelled():
                    cleanup_reason = "cancelled"
                    sandbox.note_process_termination(process, reason=cleanup_reason)
                    process_tree.terminate()
                    raise RepositoryCancelled("사용자가 저장소 조회를 중지했습니다.")
                if time.monotonic() - started >= 20:
                    cleanup_reason = "timeout"
                    sandbox.note_process_termination(process, reason=cleanup_reason)
                    process_tree.terminate()
                    raise RepositoryTimeout("Git 읽기 작업 시간이 초과되었습니다.")
    except BaseException:
        if process.poll() is None:
            sandbox.note_process_termination(process, reason=cleanup_reason)
            process_tree.terminate()
        raise
    finally:
        process_tree.close()
        sandbox.cleanup_process(process, reason=cleanup_reason)
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
    if stdout_thread.is_alive() or stderr_thread.is_alive():
        raise RepositoryAccessError("Git 출력 수집을 안전하게 완료하지 못했습니다.")
    if overflow.is_set():
        raise RepositoryAccessError("Git 조회 결과가 안전 크기 제한을 초과했습니다.")
    if return_code != 0:
        if allow_failure:
            return b""
        detail = bytes(stderr).decode("utf-8", errors="replace").strip()
        raise RepositoryAccessError(f"Git 읽기 작업이 실패했습니다: {detail[:300]}")
    return bytes(stdout)
