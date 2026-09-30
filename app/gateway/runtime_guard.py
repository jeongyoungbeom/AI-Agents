from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO


class GatewayInstanceLock:
    """같은 데이터 폴더에서 Telegram 폴러가 둘 이상 실행되지 않게 막는다."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        self._handle: BinaryIO | None = None

    def __enter__(self) -> "GatewayInstanceLock":
        self.acquire()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.release()

    def acquire(self) -> None:
        if self._handle is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        handle = os.fdopen(descriptor, "r+b", buffering=0)
        try:
            self._lock(handle)
        except OSError as exc:
            handle.close()
            owner = self._owner_hint()
            detail = f" (PID {owner})" if owner else ""
            raise RuntimeError(
                f"대화 게이트웨이가 이미 실행 중입니다{detail}."
            ) from exc
        # 0번 바이트는 OS 잠금 전용으로 유지하고 PID는 그 뒤에 기록한다.
        handle.seek(1)
        handle.write(f"{os.getpid():<20}".encode("ascii"))
        handle.truncate(21)
        handle.flush()
        os.fsync(handle.fileno())
        self._handle = handle

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            self._unlock(handle)
        finally:
            handle.close()

    def _owner_hint(self) -> str:
        try:
            with self.path.open("rb") as handle:
                handle.seek(1)
                value = handle.read(20).decode("ascii", errors="ignore").strip()
                return value if value.isdecimal() else ""
        except OSError:
            return ""

    @staticmethod
    def _lock(handle: BinaryIO) -> None:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            if handle.read(1) == b"":
                handle.seek(0)
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _unlock(handle: BinaryIO) -> None:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            return
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
