from __future__ import annotations

import tempfile
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.contracts import RunState
from app.gateway.core import (
    AccessPolicy,
    DialogueRouter,
    GatewayApplication,
    RepositoryInfo,
)
from app.orchestrator import RunStateMachine
from app.services.context import ContextService
from app.services.logging.audit import AuditLogger
from app.services.repository import SafeRepositoryReader, SafeRepositoryToolLayer
from app.storage import ArtifactStore, StateStore


AI_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = AI_ROOT / "artifacts"
TEST_REPOSITORY_IDENTITY = "a" * 64
TEST_REPOSITORY_HEAD = "b" * 40


def future_expiry(hours: int = 1) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def temporary_directory() -> tempfile.TemporaryDirectory[str]:
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="gateway-test-", dir=ARTIFACTS_ROOT)


class FakeRepositoryValidator:
    def __init__(self, path: Path):
        self.path = path.resolve()

    def validate(
        self,
        raw_path: str,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> RepositoryInfo:
        if "invalid" in raw_path:
            raise ValueError("테스트용 잘못된 경로")
        return RepositoryInfo(
            self.path,
            "main",
            TEST_REPOSITORY_IDENTITY,
            TEST_REPOSITORY_HEAD,
        )


def build_application(
    root: Path,
    backend,
    *,
    channel: str = "telegram",
    pipeline_scheduler=None,
    team_backend=None,
    max_auto_agent_replies: int = 4,
    group_parallel_workers: int = 3,
    repository_validator=None,
    repository_reader: SafeRepositoryReader | None = None,
    repository_tools: SafeRepositoryToolLayer | None = None,
    repository_approval_ttl_hours: int = 720,
    pending_project_request_ttl_hours: int = 24,
    budget=None,
    execution_preflight=None,
    repository_analysis_scheduler=None,
):
    store = StateStore(root / "state.db")
    logger = AuditLogger(root / "artifacts", store)
    machine = RunStateMachine(store, event_sink=logger, approval_phrase="개발 시작해")
    router = DialogueRouter(
        store,
        machine,
        ContextService(store),
        ArtifactStore(root / "artifacts"),
        logger,
        backend,
        repository_validator or FakeRepositoryValidator(root / "selected-repository"),
        team_backend=team_backend,
        repository_reader=repository_reader,
        repository_tools=repository_tools,
        repository_approval_ttl_hours=repository_approval_ttl_hours,
        pending_project_request_ttl_hours=pending_project_request_ttl_hours,
        pipeline_scheduler=pipeline_scheduler,
        role_names={
            "development": "빌더",
            "review": "센티널",
            "improvement": "피니셔",
        },
        max_auto_agent_replies=max_auto_agent_replies,
        group_parallel_workers=group_parallel_workers,
        budget=budget,
        execution_preflight=execution_preflight,
        repository_analysis_scheduler=repository_analysis_scheduler,
    )
    application = GatewayApplication(
        store,
        router,
        AccessPolicy(allowed_users=frozenset({"100"})),
        max_processing_attempts=2,
    )
    return store, application
