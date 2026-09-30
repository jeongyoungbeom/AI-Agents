from __future__ import annotations

import re
from pathlib import PurePosixPath

from .analysis import RepositoryAnalysisPhase, is_full_repository_audit_request
from .reader import RepositorySnapshotEntry, RepositorySnapshotManifest


_SOURCE_SUFFIXES = frozenset(
    {
        ".c", ".cc", ".cpp", ".cs", ".go", ".java", ".js", ".jsx", ".kt",
        ".kts", ".php", ".py", ".rb", ".rs", ".swift", ".ts", ".tsx",
        ".ps1", ".sh", ".sql", ".html", ".htm", ".vue", ".svelte",
    }
)
_TEST_MARKERS = frozenset({"test", "tests", "spec", "specs", "__tests__"})
_FOUNDATION_NAMES = frozenset(
    {
        "readme", "readme.md", "pyproject.toml", "package.json", "cargo.toml",
        "go.mod", "pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle",
        "settings.gradle.kts", "requirements.txt", "composer.json", "makefile",
    }
)
_RISK_NAMES = frozenset(
    {
        "dockerfile", "docker-compose.yml", "docker-compose.yaml", "compose.yml",
        "compose.yaml", "terraform.tf", "main.tf", "deployment.yaml", "deployment.yml",
    }
)
_BINARY_SUFFIXES = frozenset({".bin", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".zip", ".jar", ".exe", ".dll", ".so", ".pyc", ".db"})


def classify_analysis_path(entry: RepositorySnapshotEntry) -> str:
    path = PurePosixPath(entry.path)
    parts = {part.casefold() for part in path.parts[:-1]}
    name = path.name.casefold()
    suffix = path.suffix.casefold()
    if name in _FOUNDATION_NAMES:
        return "foundation"
    if parts & _TEST_MARKERS or name.startswith("test_") or name.endswith("_test.py"):
        return "tests"
    if name in _RISK_NAMES or {".github", "infra", "deploy", "k8s", "terraform"} & parts:
        return "risk"
    if suffix in _SOURCE_SUFFIXES:
        return "source"
    if suffix in {".json", ".toml", ".yaml", ".yml", ".xml", ".properties", ".ini", ".cfg"}:
        return "configuration"
    if suffix in {".md", ".rst", ".txt"}:
        return "documentation"
    return "other"


def build_repository_analysis_plan(
    manifest: RepositorySnapshotManifest,
    request_text: str,
    *,
    max_files_per_batch: int,
    max_file_bytes: int,
    max_batch_bytes: int = 96 * 1024,
    max_context_bytes: int | None = None,
    adaptive_file_limit: int = 48,
    mode: str | None = None,
) -> dict:
    """Produce a deterministic, purpose-weighted plan without reading file content."""
    limits = (max_files_per_batch, max_file_bytes, max_batch_bytes, adaptive_file_limit)
    if any(limit < 1 for limit in limits) or (max_context_bytes is not None and max_context_bytes < 1):
        raise ValueError("repository analysis plan limits must be positive")
    mode = mode or ("full" if is_full_repository_audit_request(request_text) else "adaptive")
    if mode not in {"full", "adaptive"}:
        raise ValueError("repository analysis mode is invalid")
    tokens = {
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9_.-]{3,}|[가-힣]{2,}", request_text)
    }
    targets = []
    for entry in manifest.entries:
        category = classify_analysis_path(entry)
        reason = ""
        if mode == "full" and PurePosixPath(entry.path).suffix.casefold() in _BINARY_SUFFIXES:
            reason = "binary_extension"
        elif entry.size > max_file_bytes:
            reason = "file_size_limit"
        elif max_context_bytes is not None and entry.size > max_context_bytes:
            reason = "context_size_limit"
        elif entry.size > max_batch_bytes:
            reason = "batch_size_limit"
        targets.append(
            {
                "path": entry.path,
                "size": entry.size,
                "object_id": entry.object_id,
                "category": category,
                "eligible": not reason,
                "selected": False,
                "exclude_reason": reason,
            }
        )

    categories = ("foundation", "configuration", "documentation", "tests", "source", "risk")
    if mode == "full":
        categories += ("other",)
    selected: dict[str, list[dict]] = {category: [] for category in categories}
    eligible = [item for item in targets if item["eligible"]]
    selection_limits: dict[str, int] = {}
    for category in categories:
        candidates = [item for item in eligible if item["category"] == category]
        candidates.sort(
            key=lambda item: (
                -sum(token in item["path"].casefold() for token in tokens),
                len(item["path"]),
                item["path"].casefold(),
            )
        )
        limit = (len(candidates) if mode == "full" else
                 adaptive_file_limit if category == "source" else max_files_per_batch * 3)
        selection_limits[category] = limit
        selected[category] = candidates[:limit]
        for item in selected[category]:
            item["selected"] = True

    not_selected = [
        {
            "path": item["path"],
            "category": item["category"],
            "reason": (
                "selection_limit"
                if item["category"] in selection_limits
                else "category_not_selected"
            ),
        }
        for item in eligible
        if not item["selected"]
    ]

    phase_specs = (
        (
            RepositoryAnalysisPhase.STRUCTURE,
            selected["foundation"] + selected["configuration"] + selected["documentation"],
        ),
        (RepositoryAnalysisPhase.TESTS, selected["tests"]),
        (RepositoryAnalysisPhase.CORE, selected["source"]),
        (RepositoryAnalysisPhase.RISKS, selected["risk"] + selected.get("other", [])),
    )
    batches: list[dict] = []

    def append_batch(phase, chunk: list[dict], total_bytes: int) -> None:
        batches.append(
            {
                "batch_index": len(batches),
                "phase": phase.value,
                "paths": [item["path"] for item in chunk],
                "categories": sorted({str(item["category"]) for item in chunk}),
                "estimated_bytes": total_bytes,
            }
        )

    for phase, items in phase_specs:
        chunk: list[dict] = []
        total_bytes = 0
        for item in items:
            size = int(item["size"])
            if chunk and (
                len(chunk) >= max_files_per_batch
                or total_bytes + size > max_batch_bytes
            ):
                append_batch(phase, chunk, total_bytes)
                chunk = []
                total_bytes = 0
            chunk.append(item)
            total_bytes += size
        if chunk:
            append_batch(phase, chunk, total_bytes)

    batches.append(
        {
            "batch_index": len(batches),
            "phase": RepositoryAnalysisPhase.SYNTHESIS.value,
            "paths": [],
            "categories": [],
            "estimated_bytes": 0,
        }
    )
    excluded = dict(manifest.exclusions)
    for item in targets:
        if item["exclude_reason"]:
            reason = str(item["exclude_reason"])
            excluded[reason] = excluded.get(reason, 0) + 1
    partial_reasons = []
    if excluded.get("context_size_limit", 0):
        partial_reasons.append("CONTEXT_LIMIT")
    if excluded.get("file_size_limit", 0):
        partial_reasons.append("FILE_SIZE_LIMIT")
    if excluded.get("batch_size_limit", 0):
        partial_reasons.append("BATCH_SIZE_LIMIT")
    if any(item["reason"] == "selection_limit" for item in not_selected):
        partial_reasons.append("SELECTION_LIMIT")
    if any(item["reason"] == "category_not_selected" for item in not_selected):
        partial_reasons.append("CATEGORY_NOT_SELECTED")
    return {
        "commit_sha": manifest.commit_sha,
        "identity_hash": manifest.identity_hash,
        "files": targets,
        "batches": batches,
        "not_selected": not_selected,
        "unprocessed_paths": [],
        "partial_reasons": partial_reasons,
        "excluded": dict(sorted(excluded.items())),
        "mode": mode,
    }

