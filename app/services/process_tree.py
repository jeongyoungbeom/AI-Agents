from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import threading
import time
from ctypes import wintypes
from typing import Any


def validate_process_tree_support() -> None:
    """세션을 이탈한 후손까지 추적하는 런타임 의존성을 확인한다."""
    try:
        import psutil  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "프로세스 트리 정리에 필요한 psutil이 설치되어 있지 않습니다."
        ) from exc


def isolated_process_options() -> dict[str, Any]:
    """하위 프로세스를 현재 게이트웨이와 분리된 종료 단위로 시작한다."""
    if os.name == "nt":
        # 부모가 첫 자식을 만들기 전에 Job Object에 넣기 위해 일시 정지 상태로 시작한다.
        return {
            "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0) | 0x00000004
        }
    return {"start_new_session": True}


class ProcessTree:
    """Popen 프로세스와 그 하위 프로세스를 하나의 수명 단위로 관리한다."""

    def __init__(self, process: subprocess.Popen[str]):
        self.process = process
        self._lock = threading.RLock()
        self._closed = False
        self._job_handle: int | None = None
        self._root_identity: tuple[int, float] | None = None
        self._process_group_id: int | None = None
        self._descendants_lock = threading.Lock()
        self._known_descendants: dict[tuple[int, float], object] = {}
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        if os.name == "nt":
            try:
                self._job_handle = _assign_windows_kill_job(process)
            except Exception as exc:
                self._abort_suspended_process()
                raise RuntimeError("Windows Job Object 격리에 실패했습니다.") from exc
            is_real_process = isinstance(getattr(process, "pid", None), int) and hasattr(
                process, "_handle"
            )
            if is_real_process and self._job_handle is None:
                self._abort_suspended_process()
                raise RuntimeError("Windows Job Object 격리에 실패했습니다.")
            if not _resume_windows_process(process):
                self._abort_suspended_process()
                raise RuntimeError("격리한 Windows 프로세스를 재개하지 못했습니다.")
        else:
            self._capture_posix_identity()
            self._remember_descendants()
            self._monitor_thread = threading.Thread(
                target=self._monitor_descendants,
                name=f"process-tree-{getattr(process, 'pid', 'test')}",
                daemon=True,
            )
            self._monitor_thread.start()

    def terminate(self, *, grace_seconds: float = 5) -> None:
        with self._lock:
            if self._closed:
                return
            self._stop_monitor()
            if os.name == "nt":
                self._terminate_windows(grace_seconds)
            else:
                self._terminate_posix(grace_seconds)

    def close(self) -> None:
        """남은 하위 프로세스를 정리하고 OS 자원을 해제한다."""
        with self._lock:
            if self._closed:
                return
            try:
                self._stop_monitor()
                if os.name == "nt" and self._job_handle is not None:
                    # KILL_ON_JOB_CLOSE가 정상 종료 뒤 남은 자식까지 정리한다.
                    _close_windows_handle(self._job_handle)
                    self._job_handle = None
                    self._wait_or_kill_direct(2)
                elif os.name == "nt":
                    self._terminate_windows(2)
                else:
                    self._terminate_posix(2)
            finally:
                self._closed = True

    def _terminate_windows(self, grace_seconds: float) -> None:
        if self._job_handle is not None:
            _terminate_windows_job(self._job_handle)
        elif self.process.poll() is None:
            self._taskkill_fallback()
        self._wait_or_kill_direct(grace_seconds)

    def _taskkill_fallback(self) -> None:
        pid = getattr(self.process, "pid", None)
        if not isinstance(pid, int) or pid <= 0:
            self._terminate_direct()
            return
        taskkill = os.path.join(
            os.environ.get("SystemRoot", r"C:\Windows"), "System32", "taskkill.exe"
        )
        try:
            subprocess.run(
                [taskkill, "/PID", str(pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            self._terminate_direct()

    def _terminate_posix(self, grace_seconds: float) -> None:
        descendants = self._remember_descendants()
        self._signal_descendants(descendants, hard=False)
        if not self._signal_owned_group(signal.SIGTERM):
            self._terminate_direct()
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            descendants = self._remember_descendants()
            if not self._owned_group_exists() and not self._any_running(descendants):
                self._wait_direct(0)
                return
            time.sleep(0.02)

        # 그룹을 이탈한 후손도 포함하도록, 각 소유 프로세스를 멈추고 재탐색한다.
        descendants, stopped = self._freeze_and_rescan_descendants()
        try:
            self._signal_descendants(descendants, hard=True)
        finally:
            for process in stopped:
                try:
                    process.resume()
                except Exception:
                    pass
        self._signal_owned_group(signal.SIGKILL)
        self._kill_direct()
        self._wait_direct(2)

    def _capture_posix_identity(self) -> None:
        pid = getattr(self.process, "pid", None)
        if not isinstance(pid, int) or pid <= 0:
            return
        try:
            import psutil
        except ImportError as exc:
            self._terminate_direct()
            self._wait_direct(2)
            raise RuntimeError("POSIX 프로세스 신원을 고정하지 못했습니다.") from exc

        try:
            root = psutil.Process(pid)
            self._root_identity = (pid, float(root.create_time()))
            self._process_group_id = int(os.getpgid(pid))
        except (psutil.Error, OSError) as exc:
            self._terminate_direct()
            self._wait_direct(2)
            raise RuntimeError("POSIX 프로세스 신원을 고정하지 못했습니다.") from exc
        if self._process_group_id != pid:
            self._terminate_direct()
            self._wait_direct(2)
            raise RuntimeError("POSIX 프로세스 그룹 격리에 실패했습니다.")

    def _root_process(self) -> object | None:
        identity = self._root_identity
        if identity is None:
            return None
        try:
            import psutil
        except ImportError:
            return None

        try:
            process = psutil.Process(identity[0])
            if float(process.create_time()) != identity[1]:
                return None
            return process
        except (psutil.Error, OSError):
            return None

    def _remember_descendants(self) -> list[object]:
        try:
            import psutil

            root = self._root_process()
            with self._descendants_lock:
                known = list(self._known_descendants.values())
            roots = ([root] if root is not None else []) + known
            for root in roots:
                try:
                    candidates = root.children(recursive=True)
                except (psutil.Error, OSError):
                    continue
                for child in candidates:
                    try:
                        identity = (int(child.pid), float(child.create_time()))
                    except (psutil.Error, OSError):
                        continue
                    with self._descendants_lock:
                        self._known_descendants[identity] = child
        except (ImportError, OSError):
            pass
        with self._descendants_lock:
            return list(self._known_descendants.values())

    def _freeze_and_rescan_descendants(self) -> tuple[list[object], list[object]]:
        try:
            import psutil
        except ImportError:
            return self._remember_descendants(), []
        with self._descendants_lock:
            roots = list(self._known_descendants.values())
        root = self._root_process()
        if root is not None:
            roots.insert(0, root)
        pending = roots
        seen: set[tuple[int, float]] = set()
        stopped: list[object] = []
        deadline = time.monotonic() + 1
        while pending and time.monotonic() < deadline:
            process = pending.pop(0)
            try:
                identity = (int(process.pid), float(process.create_time()))
                if identity in seen:
                    continue
                seen.add(identity)
                status = process.status()
                if status in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                    continue
                if status != psutil.STATUS_STOPPED:
                    process.suspend()
                    stopped.append(process)
                children = process.children(recursive=False)
            except (psutil.Error, OSError):
                continue
            for child in children:
                try:
                    child_identity = (int(child.pid), float(child.create_time()))
                except (psutil.Error, OSError):
                    continue
                with self._descendants_lock:
                    self._known_descendants[child_identity] = child
                pending.append(child)
        with self._descendants_lock:
            return list(self._known_descendants.values()), stopped

    def _monitor_descendants(self) -> None:
        while not self._monitor_stop.wait(0.02):
            self._remember_descendants()
            if self._root_process() is None:
                return

    def _stop_monitor(self) -> None:
        self._monitor_stop.set()
        thread = self._monitor_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1)

    def _owns_process_group(self) -> bool:
        pgid = self._process_group_id
        if pgid is None:
            return False
        if self._root_process() is not None:
            return True
        with self._descendants_lock:
            descendants = list(self._known_descendants.items())
        for identity, process in descendants:
            try:
                if (
                    process.is_running()
                    and float(process.create_time()) == identity[1]
                    and os.getpgid(identity[0]) == pgid
                ):
                    return True
            except Exception:
                continue
        return False

    def _signal_owned_group(self, selected_signal: int) -> bool:
        pgid = self._process_group_id
        if pgid is None or not self._owns_process_group():
            return False
        try:
            os.killpg(pgid, selected_signal)
            return True
        except ProcessLookupError:
            return False
        except (PermissionError, OSError):
            return False

    def _owned_group_exists(self) -> bool:
        pgid = self._process_group_id
        if pgid is None or not self._owns_process_group():
            return False
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except (PermissionError, OSError):
            return True

    @staticmethod
    def _signal_descendants(descendants: list[object], *, hard: bool) -> None:
        for process in reversed(descendants):
            try:
                if process.is_running():
                    process.kill() if hard else process.terminate()
            except Exception:
                continue

    @staticmethod
    def _any_running(descendants: list[object]) -> bool:
        try:
            import psutil
        except ImportError:
            return False
        for process in descendants:
            try:
                if process.is_running() and process.status() not in (
                    psutil.STATUS_ZOMBIE,
                    psutil.STATUS_DEAD,
                ):
                    return True
            except (psutil.Error, OSError):
                continue
        return False

    def _abort_suspended_process(self) -> None:
        if self._job_handle is not None:
            _close_windows_handle(self._job_handle)
            self._job_handle = None
        self._kill_direct()
        self._wait_direct(2)

    def _wait_or_kill_direct(self, grace_seconds: float) -> None:
        if self.process.poll() is not None:
            return
        try:
            self.process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            self._kill_direct()
            self._wait_direct(2)

    def _terminate_direct(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            self.process.terminate()
        except OSError:
            pass

    def _kill_direct(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            self.process.kill()
        except OSError:
            pass

    def _wait_direct(self, timeout: float) -> None:
        try:
            self.process.wait(timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            pass

    def __enter__(self) -> ProcessTree:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


if os.name == "nt":
    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]


    class _BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]


    class _ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]


def _assign_windows_kill_job(process: subprocess.Popen[str]) -> int | None:
    try:
        process_handle = int(getattr(process, "_handle"))
    except (AttributeError, TypeError, ValueError):
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        return None
    info = _ExtendedLimitInformation()
    info.BasicLimitInformation.LimitFlags = 0x00002000
    configured = kernel32.SetInformationJobObject(
        handle, 9, ctypes.byref(info), ctypes.sizeof(info)
    )
    assigned = configured and kernel32.AssignProcessToJobObject(
        handle, wintypes.HANDLE(process_handle)
    )
    if not assigned:
        kernel32.CloseHandle(handle)
        return None
    return int(handle)


def _resume_windows_process(process: subprocess.Popen[str]) -> bool:
    try:
        process_handle = int(getattr(process, "_handle"))
    except (AttributeError, TypeError, ValueError):
        # 테스트 대역은 실제로 일시 정지되어 있지 않다.
        return True
    try:
        ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
        ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
        ntdll.NtResumeProcess.restype = ctypes.c_long
        return ntdll.NtResumeProcess(wintypes.HANDLE(process_handle)) == 0
    except (AttributeError, OSError):
        return False


def _terminate_windows_job(handle: int) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.TerminateJobObject(wintypes.HANDLE(handle), 1)


def _close_windows_handle(handle: int) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle(wintypes.HANDLE(handle))
