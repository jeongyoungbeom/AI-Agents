from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.config import FoundationConfig
from app.contracts import RoleId, StageContract, TokenUsage
from app.orchestrator import RunStateMachine
from app.pipeline import PipelineCoordinator, PipelineWorker
from app.services.budget import BudgetManager, BudgetPolicy
from app.services.hermes import HermesResult
from app.services.logging.audit import AuditLogger
from app.services.repository import inspect_repository_identity
from app.services.verification import VerificationRunner
from app.storage import ArtifactStore, StateStore


AI_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = AI_ROOT / "test-tmp"


def temporary_directory() -> tempfile.TemporaryDirectory[str]:
    # Docker-owned operational artifacts can be unreadable to the current
    # user. The project-owned test root stays outside those retained artifacts
    # and is cleaned per fixture.
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="pipeline-test-", dir=TEST_ROOT)


def git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    return result.stdout.strip()


class HostGitSandbox:
    """Deterministic test double for the Docker Git/worktree boundary.

    It only translates container workspace paths back to the fixture path; it
    never receives a model prompt or repository-derived command argv.
    """

    def with_image(self, _image: str) -> "HostGitSandbox":
        return self

    def run(self, repository: Path, argv, *, timeout=None, **_kwargs):
        translated = self._argv(Path(repository), argv)
        result = subprocess.run(
            translated,
            cwd=repository,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        if "rev-parse" in translated and "--show-toplevel" in translated:
            return subprocess.CompletedProcess(
                result.args, result.returncode, "/workspace\n", result.stderr
            )
        return result

    def popen(self, repository: Path, argv, **kwargs):
        for name in (
            "writable_workspace",
            "interactive",
            "operation_id",
            "component",
            "additional_mounts",
            "git_metadata",
            "writable_git_metadata",
        ):
            kwargs.pop(name, None)
        translated = self._argv(Path(repository), argv)
        if translated and Path(translated[0]).name.startswith("python") and "-c" in translated:
            translated[1:1] = ["-X", "utf8"]
            source_index = translated.index("-c") + 1
            local_root = repr(str(Path(repository).resolve()))
            translated[source_index] = (
                translated[source_index]
                .replace(
                    "from pathlib import PurePosixPath",
                    "from pathlib import Path, PurePosixPath",
                )
                .replace(
                    'if top != "/workspace":',
                    f"if Path(top).resolve() != Path({local_root}).resolve():",
                )
                .replace('cwd="/workspace"', f"cwd={local_root}")
            )
        return subprocess.Popen(translated, cwd=repository, **kwargs)

    def cleanup_process(self, _process, *, reason: str) -> bool:
        return False

    def note_process_termination(self, _process, *, reason: str) -> None:
        return None

    @staticmethod
    def _argv(repository: Path, argv) -> list[str]:
        result: list[str] = []
        for item in argv:
            if item == "python3":
                result.append(sys.executable)
                continue
            if item.startswith("--git-dir=/ai-agents-gitdir/"):
                continue
            if item == "--work-tree=/workspace":
                continue
            result.append(str(repository) if item == "/workspace" else item)
        return result


def create_repository(root: Path, *, verifier_passes: bool = True) -> Path:
    repository = root / "repository"
    repository.mkdir()
    git(repository, "init")
    git(repository, "config", "user.name", "AI Agents Test")
    git(repository, "config", "user.email", "ai-agents-test@example.invalid")
    expected = "fixed" if verifier_passes else "never"
    (repository / "README.md").write_text("test\n", encoding="utf-8")
    (repository / "verify.py").write_text(
        "from pathlib import Path\n"
        f"assert Path('feature.txt').read_text(encoding='utf-8').strip() == {expected!r}\n",
        encoding="utf-8",
    )
    git(repository, "add", "-A")
    git(repository, "commit", "-m", "initial")
    return repository.resolve()


DEFAULT_IMPLEMENTATION_FINDING = {
    "finding_id": "F-001",
    "severity": "medium",
    "category": "implementation",
    "evidence": "feature.txt가 draft 상태임",
    "required_change": "fixed로 변경",
    "file": "feature.txt",
    "line": 1,
    "status": "open",
}


class FakeRoleRunner:
    def __init__(
        self,
        *,
        reviewer_mutates: bool = False,
        improvement_fixes: bool = True,
        review_responses: list[list[dict]] | None = None,
        development_outputs: list[str] | None = None,
        improvement_outputs: list[str] | None = None,
        development_out_of_scope_write: bool = False,
    ):
        self.reviewer_mutates = reviewer_mutates
        self.improvement_fixes = improvement_fixes
        self.review_responses = review_responses
        self.development_outputs = development_outputs
        self.improvement_outputs = improvement_outputs
        self.development_out_of_scope_write = development_out_of_scope_write
        self.calls: list[tuple[RoleId, bool]] = []
        self._review_calls = 0
        self._development_calls = 0
        self._improvement_calls = 0

    def run(
        self,
        run_id,
        stage_id,
        role_id,
        repository,
        prompt,
        *,
        allow_writes,
        max_output_tokens=None,
        cancelled=None,
        heartbeat=None,
    ):
        self.calls.append((role_id, allow_writes))
        if heartbeat:
            heartbeat()
        repository = Path(repository)
        if role_id == RoleId.DEVELOPMENT:
            value = "draft"
            if self.development_outputs:
                index = min(
                    self._development_calls, len(self.development_outputs) - 1
                )
                value = self.development_outputs[index]
            self._development_calls += 1
            (repository / "feature.txt").write_text(f"{value}\n", encoding="utf-8")
            if self.development_out_of_scope_write:
                (repository / "outside-stage-scope.txt").write_text(
                    "must not be committed\n", encoding="utf-8"
                )
            text = json.dumps({"summary": "기능 초안 구현", "needs_user_input": []})
        elif role_id == RoleId.REVIEW:
            if self.reviewer_mutates:
                (repository / "reviewer-was-here.txt").write_text("bad\n", encoding="utf-8")
            findings = [dict(DEFAULT_IMPLEMENTATION_FINDING)]
            if self.review_responses is not None:
                index = min(self._review_calls, len(self.review_responses) - 1)
                findings = self.review_responses[index] if self.review_responses else []
            self._review_calls += 1
            text = json.dumps(
                {
                    "summary": "수정 필요" if findings else "문제 없음",
                    "findings": findings,
                    "needs_user_input": [],
                }
            )
        else:
            value = "fixed" if self.improvement_fixes else None
            if self.improvement_outputs:
                index = min(
                    self._improvement_calls, len(self.improvement_outputs) - 1
                )
                value = self.improvement_outputs[index]
            self._improvement_calls += 1
            if value is not None:
                (repository / "feature.txt").write_text(f"{value}\n", encoding="utf-8")
            text = json.dumps({"summary": "리뷰 사항 보완", "needs_user_input": []})
        return HermesResult(text, TokenUsage(100, 20), 0.01)


def build_pipeline(
    root: Path,
    repository: Path,
    runner: FakeRoleRunner,
    *,
    stage_count: int = 1,
    sandbox=None,
    run_id: str = "RUN-PIPELINE-TEST",
):
    sandbox = sandbox or HostGitSandbox()
    store = StateStore(root / "state.db")
    artifacts = ArtifactStore(root / "artifacts")
    logger = AuditLogger(root / "artifacts", store)
    machine = RunStateMachine(store, event_sink=logger, approval_phrase="개발 시작해")
    foundation = FoundationConfig.load(AI_ROOT)
    budget = BudgetManager(BudgetPolicy.load(AI_ROOT / "config" / "limits.json"), store)
    coordinator = PipelineCoordinator(
        store,
        machine,
        foundation,
        artifacts,
        logger,
        budget,
        runner,
        VerificationRunner(
            timeout_seconds=30, poll_seconds=0.05, sandbox=sandbox
        ),
        sandbox=sandbox,
    )
    worker = PipelineWorker(store, machine, coordinator, logger, poll_seconds=0.01)
    identity = inspect_repository_identity(repository, sandbox=sandbox)
    state = machine.create_run(
        run_id,
        repository=str(repository),
        repository_identity=identity.identity_hash,
        repository_head_sha=identity.head_sha,
        repository_approved=True,
        objective="테스트 기능 구현",
    )
    contracts = [
        StageContract(
            run_id=state.run_id,
            stage_id=f"stage-{index:03d}",
            objective=f"기능 파일 구현 {index}",
            scope=("feature.txt",),
            acceptance_criteria=("feature.txt 내용이 fixed",),
            verification_commands=("python verify.py",),
        )
        for index in range(1, stage_count + 1)
    ]
    state = machine.register_plan(
        state, {"stages": [contract.to_dict() for contract in contracts]}
    )
    state = machine.request_approval(state)
    state = machine.approve(state, "개발 시작해", "test-user")
    store.bind_conversation(
        "telegram", "chat-1", "test-user", state.run_id, RoleId.DEVELOPMENT.value
    )
    store.approve_repository(
        "telegram",
        "chat-1",
        "test-user",
        str(repository),
        identity.identity_hash,
        (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
    )
    worker.enqueue(state.run_id, "telegram", "chat-1")
    return store, worker, state.run_id
