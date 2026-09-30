from __future__ import annotations

import json
import os
import shutil
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.storage import StateStore


@dataclass(frozen=True)
class RetentionPolicy:
    database_ttl_days: int = 30
    artifact_ttl_days: int = 30
    attachment_ttl_days: int = 14
    log_ttl_days: int = 30
    backup_ttl_days: int = 30
    max_artifact_bytes: int = 1_073_741_824
    max_attachment_bytes: int = 536_870_912
    backup_count: int = 7

    def __post_init__(self) -> None:
        for name in (
            "database_ttl_days",
            "artifact_ttl_days",
            "attachment_ttl_days",
            "log_ttl_days",
            "backup_ttl_days",
        ):
            if not 1 <= getattr(self, name) <= 3_650:
                raise ValueError(f"{name} must be between 1 and 3650")
        for name in ("max_artifact_bytes", "max_attachment_bytes"):
            if getattr(self, name) < 1_024:
                raise ValueError(f"{name} must be at least 1024")
        if not 1 <= self.backup_count <= 90:
            raise ValueError("backup_count must be between 1 and 90")

    @classmethod
    def load(cls, path: Path) -> "RetentionPolicy":
        raw = json.loads(path.read_text(encoding="utf-8"))
        values = dict(raw.get("retention", {}))
        return cls(
            database_ttl_days=int(values.get("database_ttl_days", 30)),
            artifact_ttl_days=int(values.get("artifact_ttl_days", 30)),
            attachment_ttl_days=int(values.get("attachment_ttl_days", 14)),
            log_ttl_days=int(values.get("log_ttl_days", 30)),
            backup_ttl_days=int(values.get("backup_ttl_days", 30)),
            max_artifact_bytes=int(values.get("max_artifact_bytes", 1_073_741_824)),
            max_attachment_bytes=int(values.get("max_attachment_bytes", 536_870_912)),
            backup_count=int(values.get("backup_count", 7)),
        )


@dataclass(frozen=True)
class RetentionResult:
    database_rows: dict[str, int]
    files: dict[str, int]
    bytes_freed: int

    def to_dict(self) -> dict:
        return asdict(self)


class RetentionManager:
    """Purges only expired terminal operational data below approved roots."""

    def __init__(self, root: Path, store: StateStore, policy: RetentionPolicy):
        self.root = root.resolve()
        self.store = store
        self.policy = policy
        self.artifacts = self.root / "artifacts"
        self.attachments = self.root / "data" / "attachments"
        self.logs = self.root / "logs"
        self.backups = self.root / "data" / "backups"

    def backup_database(self, *, now: datetime | None = None) -> Path:
        now = now or datetime.now(timezone.utc)
        self.backups.mkdir(parents=True, exist_ok=True)
        destination = self.backups / (
            f"agent-team-{now.strftime('%Y%m%dT%H%M%SZ')}.db"
        )
        return self.store.backup_to(destination)

    def restore_database(self, backup: Path) -> Path:
        backup = backup.resolve()
        self._under(backup, self.backups)
        if not backup.is_file():
            raise ValueError("database backup file does not exist")
        source = sqlite3.connect(backup)
        target = sqlite3.connect(self.store.path)
        try:
            result = source.execute("PRAGMA integrity_check").fetchone()
            if result is None or str(result[0]).lower() != "ok":
                raise ValueError("database backup integrity check failed")
            source.backup(target)
        finally:
            target.close()
            source.close()
        return self.store.path

    def purge(self, *, now: datetime | None = None) -> RetentionResult:
        now = now or datetime.now(timezone.utc)
        rows = self.store.purge_expired_operational_rows(
            self._cutoff(now, self.policy.database_ttl_days)
        )
        for scope, count in self.store.purge_expired_memory_facts(now.isoformat()).items():
            rows[f"{scope}_memories"] = count
        protected_runs = self.store.active_run_ids()
        files: dict[str, int] = {}
        bytes_freed = 0
        for name, root, days, maximum, protected in (
            (
                "artifacts",
                self.artifacts,
                self.policy.artifact_ttl_days,
                self.policy.max_artifact_bytes,
                protected_runs,
            ),
            (
                "attachments",
                self.attachments,
                self.policy.attachment_ttl_days,
                self.policy.max_attachment_bytes,
                set(),
            ),
        ):
            removed, freed = self._purge_tree(
                root,
                older_than=now - timedelta(days=days),
                max_bytes=maximum,
                protected_names=protected,
                directories_only=name == "artifacts",
            )
            files[name] = removed
            bytes_freed += freed
        for name, root, days in (
            ("logs", self.logs, self.policy.log_ttl_days),
            ("backups", self.backups, self.policy.backup_ttl_days),
        ):
            removed, freed = self._purge_tree(
                root,
                older_than=now - timedelta(days=days),
                max_bytes=None,
                protected_names=set(),
                directories_only=False,
            )
            files[name] = removed
            bytes_freed += freed
        removed, freed = self._trim_backup_count()
        files["backup_count"] = removed
        bytes_freed += freed
        return RetentionResult(rows, files, bytes_freed)

    @staticmethod
    def _cutoff(now: datetime, days: int) -> str:
        return (now - timedelta(days=days)).isoformat()

    def _trim_backup_count(self) -> tuple[int, int]:
        if not self.backups.is_dir():
            return 0, 0
        backups = sorted(
            (path for path in self.backups.glob("agent-team-*.db") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return self._remove_paths(backups[self.policy.backup_count :])

    def _purge_tree(
        self,
        root: Path,
        *,
        older_than: datetime,
        max_bytes: int | None,
        protected_names: set[str],
        directories_only: bool,
    ) -> tuple[int, int]:
        if not root.is_dir():
            return 0, 0
        root = root.resolve()
        if directories_only:
            candidates = [
                path
                for path in root.iterdir()
                if path.is_dir() and path.name not in protected_names
            ]
        else:
            candidates = [
                Path(directory) / name
                for directory, _names, names in os.walk(root, followlinks=False)
                for name in names
            ]
        candidates = [path for path in candidates if self._is_safe_child(path, root)]
        old = [
            path
            for path in candidates
            if datetime.fromtimestamp(path.stat().st_mtime, timezone.utc) < older_than
        ]
        removed, freed = self._remove_paths(old)
        if max_bytes is None:
            return removed, freed
        remaining = [path for path in candidates if path not in old and path.exists()]
        total = sum(self._path_size(path) for path in remaining)
        if total <= max_bytes:
            return removed, freed
        for path in sorted(remaining, key=lambda item: item.stat().st_mtime):
            if total <= max_bytes:
                break
            size = self._path_size(path)
            count, reclaimed = self._remove_paths((path,))
            removed += count
            freed += reclaimed
            if count:
                total -= size
        return removed, freed

    def _remove_paths(self, paths) -> tuple[int, int]:
        removed = 0
        bytes_freed = 0
        for path in paths:
            try:
                size = self._path_size(path)
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except OSError:
                continue
            removed += 1
            bytes_freed += size
        return removed, bytes_freed

    @staticmethod
    def _path_size(path: Path) -> int:
        try:
            if path.is_file():
                return path.stat().st_size
            return sum(
                child.stat().st_size
                for child in path.rglob("*")
                if child.is_file() and not child.is_symlink()
            )
        except OSError:
            return 0

    @staticmethod
    def _under(path: Path, root: Path) -> None:
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError("retention path escapes its managed root") from exc

    def _is_safe_child(self, path: Path, root: Path) -> bool:
        try:
            self._under(path.resolve(), root)
        except (OSError, ValueError):
            return False
        return not path.is_symlink()
