from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_identifier(value: str, field_name: str) -> str:
    value = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", value):
        raise ValueError(
            f"{field_name} must be 1-80 characters using letters, numbers, dot, dash, or underscore"
        )
    return value


def normalize_stage_scope(value: str) -> str:
    """Return one unambiguous, repository-relative stage write boundary.

    A stage scope is used as an authorization boundary, not just as a planning
    note.  Files are exact paths and directories must end with ``/``.  Keeping
    the grammar deliberately small prevents a generated plan from widening a
    write boundary through a drive path, traversal, or glob syntax.
    """

    raw = value.strip().replace("\\", "/")
    is_directory = raw.endswith("/")
    raw = raw.rstrip("/")
    if not raw:
        raise ValueError("stage scope cannot be the repository root")
    if raw.startswith(("/", "~")) or ":" in raw:
        raise ValueError("stage scope must be a relative repository path")
    if any(character in raw for character in ("\r", "\n", "\0", "*", "?", "[", "]")):
        raise ValueError("stage scope cannot contain control characters or glob syntax")
    parts = raw.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("stage scope cannot contain empty or traversal path segments")
    normalized = "/".join(parts)
    return normalized + "/" if is_directory else normalized


class RoleId(str, Enum):
    DEVELOPMENT = "development"
    REVIEW = "review"
    IMPROVEMENT = "improvement"


class RunPhase(str, Enum):
    DISCUSSING = "DISCUSSING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    DEVELOPING = "DEVELOPING"
    REVIEWING = "REVIEWING"
    IMPROVING = "IMPROVING"
    VERIFYING = "VERIFYING"
    STAGE_COMPLETED = "STAGE_COMPLETED"
    COMPLETED = "COMPLETED"
    PAUSED = "PAUSED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class ConversationSessionState:
    """채널 대화 자체의 상태. 프로젝트와 작업 수명에 종속되지 않는다."""

    channel: str
    conversation_id: str
    user_id: str
    active_role: RoleId = RoleId.DEVELOPMENT
    mode: str = "free_chat"
    session_run_id: str = ""
    active_task_id: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.channel.strip() or not self.conversation_id.strip() or not self.user_id.strip():
            raise ValueError("conversation channel, id, and user are required")
        if self.mode not in {"free_chat", "planning", "executing"}:
            raise ValueError(f"unsupported conversation mode: {self.mode}")
        require_identifier(self.session_run_id, "session_run_id")
        if self.active_task_id:
            require_identifier(self.active_task_id, "active_task_id")


@dataclass(frozen=True)
class ProjectSelectionState:
    """한 대화에서 현재 보고 있는 프로젝트 선택 상태."""

    channel: str
    conversation_id: str
    user_id: str
    repository_path: str
    repository_identity: str = ""
    head_sha: str = ""
    approved: bool = False
    selected_at: str = field(default_factory=utc_now)
    approved_at: str = ""
    approval_expires_at: str = ""

    def __post_init__(self) -> None:
        if not self.channel.strip() or not self.conversation_id.strip() or not self.user_id.strip():
            raise ValueError("project selection owner is required")
        if not self.repository_path.strip():
            raise ValueError("project repository path is required")
        if self.repository_identity and not re.fullmatch(
            r"[0-9a-f]{64}", self.repository_identity
        ):
            raise ValueError("repository identity must be a SHA-256 hex digest")
        if self.head_sha and not re.fullmatch(r"[0-9a-f]{40,64}", self.head_sha):
            raise ValueError("repository head must be a Git object id")
        if self.approved and not all(
            (
                self.repository_identity,
                self.head_sha,
                self.approved_at,
                self.approval_expires_at,
            )
        ):
            raise ValueError(
                "approved project selection requires identity, head, and expiry"
            )


@dataclass(frozen=True)
class TaskDefinitionState:
    """실행 작업에 고정된 목표와 프로젝트 스냅샷."""

    run_id: str
    objective: str = ""
    repository_path: str = ""
    repository_identity: str = ""
    repository_head_sha: str = ""
    repository_approved: bool = False

    def __post_init__(self) -> None:
        require_identifier(self.run_id, "run_id")
        if self.repository_identity and not re.fullmatch(
            r"[0-9a-f]{64}", self.repository_identity
        ):
            raise ValueError("task repository identity must be a SHA-256 hex digest")
        if self.repository_head_sha and not re.fullmatch(
            r"[0-9a-f]{40,64}", self.repository_head_sha
        ):
            raise ValueError("task repository head must be a Git object id")
        if self.repository_approved and not all(
            (
                self.repository_path.strip(),
                self.repository_identity,
                self.repository_head_sha,
            )
        ):
            raise ValueError(
                "approved task repository requires path, identity, and head"
            )


@dataclass(frozen=True)
class PlanPointerState:
    """작업의 현재 계획 버전 포인터. 계획 본문은 revision 원장에 보관한다."""

    run_id: str
    revision: int = 0
    plan_hash: str = ""

    def __post_init__(self) -> None:
        require_identifier(self.run_id, "run_id")
        if self.revision < 0:
            raise ValueError("plan revision cannot be negative")
        if self.plan_hash and not re.fullmatch(r"[0-9a-f]{64}", self.plan_hash):
            raise ValueError("plan_hash must be a SHA-256 hex digest")
        if bool(self.revision) != bool(self.plan_hash):
            raise ValueError("plan revision and hash must be set together")


@dataclass(frozen=True)
class ExecutionApprovalState:
    """현재 계획과 분리해 저장하는 일회 실행 승인 상태."""

    run_id: str
    granted: bool = False
    plan_revision: int = 0
    plan_hash: str = ""
    approved_at: str = ""
    approved_by: str = ""

    def __post_init__(self) -> None:
        require_identifier(self.run_id, "run_id")
        if self.plan_revision < 0:
            raise ValueError("approved plan revision cannot be negative")
        if self.plan_hash and not re.fullmatch(r"[0-9a-f]{64}", self.plan_hash):
            raise ValueError("approved plan hash must be a SHA-256 hex digest")
        metadata = (
            self.plan_revision,
            self.plan_hash,
            self.approved_at,
            self.approved_by.strip(),
        )
        if self.granted and not all(metadata):
            raise ValueError("granted approval requires plan, time, and approver")
        if not self.granted and any(metadata):
            raise ValueError("ungranted approval cannot retain approval metadata")


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated: bool = False

    def __post_init__(self) -> None:
        if min(self.input_tokens, self.output_tokens, self.total_tokens) < 0:
            raise ValueError("token counts cannot be negative")
        calculated = self.input_tokens + self.output_tokens
        if self.total_tokens == 0 and calculated:
            object.__setattr__(self, "total_tokens", calculated)
        elif calculated and self.total_tokens < calculated:
            raise ValueError("total_tokens cannot be lower than input + output tokens")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TokenUsage":
        return cls(
            input_tokens=int(value.get("input_tokens", 0)),
            output_tokens=int(value.get("output_tokens", 0)),
            total_tokens=int(value.get("total_tokens", 0)),
            estimated=bool(value.get("estimated", False)),
        )


@dataclass(frozen=True)
class StageContract:
    run_id: str
    stage_id: str
    objective: str
    scope: tuple[str, ...]
    acceptance_criteria: tuple[str, ...]
    verification_commands: tuple[str, ...]
    non_goals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_identifier(self.run_id, "run_id")
        require_identifier(self.stage_id, "stage_id")
        if not self.objective.strip():
            raise ValueError("objective is required")
        if not self.scope:
            raise ValueError("scope must contain at least one item")
        normalized_scope = tuple(normalize_stage_scope(item) for item in self.scope)
        if len(set(normalized_scope)) != len(normalized_scope):
            raise ValueError("stage scope cannot contain duplicates")
        object.__setattr__(self, "scope", normalized_scope)
        if not self.acceptance_criteria:
            raise ValueError("acceptance_criteria must contain at least one item")
        if not self.verification_commands:
            raise ValueError("verification_commands must contain at least one item")
        for command in self.verification_commands:
            if not command.strip():
                raise ValueError("verification commands cannot be blank")
            if any(character in command for character in ("\r", "\n", "\0")):
                raise ValueError("verification commands must be single-line commands")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StageContract":
        return cls(
            run_id=str(value["run_id"]),
            stage_id=str(value["stage_id"]),
            objective=str(value["objective"]),
            scope=tuple(str(item) for item in value.get("scope", [])),
            acceptance_criteria=tuple(
                str(item) for item in value.get("acceptance_criteria", [])
            ),
            verification_commands=tuple(
                str(item) for item in value.get("verification_commands", [])
            ),
            non_goals=tuple(str(item) for item in value.get("non_goals", [])),
        )


@dataclass(frozen=True)
class ReviewFinding:
    finding_id: str
    severity: str
    evidence: str
    required_change: str
    category: str = "implementation"
    file: str = ""
    line: int | None = None
    status: str = "open"

    def __post_init__(self) -> None:
        require_identifier(self.finding_id, "finding_id")
        if self.severity not in {"critical", "high", "medium", "low"}:
            raise ValueError(f"unsupported finding severity: {self.severity}")
        if self.category not in {"design", "implementation"}:
            raise ValueError(f"unsupported finding category: {self.category}")
        if self.status not in {"open", "fixed", "rejected_with_evidence", "needs_user"}:
            raise ValueError(f"unsupported finding status: {self.status}")
        if not self.evidence.strip() or not self.required_change.strip():
            raise ValueError("review findings require evidence and required_change")
        if self.line is not None and self.line < 1:
            raise ValueError("finding line must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReviewFinding":
        raw_line = value.get("line")
        return cls(
            finding_id=str(value["finding_id"]),
            severity=str(value["severity"]),
            evidence=str(value["evidence"]),
            required_change=str(value["required_change"]),
            category=str(value.get("category", "implementation")),
            file=str(value.get("file", "")),
            line=int(raw_line) if raw_line is not None else None,
            status=str(value.get("status", "open")),
        )


@dataclass(frozen=True)
class AgentHandoff:
    contract: StageContract
    from_role: RoleId
    to_role: RoleId
    summary: str
    base_sha: str = ""
    candidate_sha: str = ""
    changed_files: tuple[str, ...] = ()
    findings: tuple[ReviewFinding, ...] = ()
    verification: dict[str, Any] = field(default_factory=dict)
    usage: TokenUsage = field(default_factory=TokenUsage)
    needs_user_input: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        allowed = {
            (RoleId.DEVELOPMENT, RoleId.REVIEW),
            (RoleId.REVIEW, RoleId.DEVELOPMENT),
            (RoleId.REVIEW, RoleId.IMPROVEMENT),
            (RoleId.IMPROVEMENT, RoleId.REVIEW),
        }
        if (self.from_role, self.to_role) not in allowed:
            raise ValueError("unsupported role handoff")
        if not self.summary.strip():
            raise ValueError("handoff summary is required")
        if self.from_role == RoleId.REVIEW:
            if not self.findings:
                raise ValueError("review handoff requires at least one finding")
            expected = (
                "design"
                if self.to_role == RoleId.DEVELOPMENT
                else "implementation"
            )
            if any(finding.category != expected for finding in self.findings):
                raise ValueError(
                    f"review -> {self.to_role.value} requires only {expected} findings"
                )
        if self.from_role == RoleId.IMPROVEMENT:
            if not self.findings or any(
                finding.category != "implementation" for finding in self.findings
            ):
                raise ValueError(
                    "improvement -> review requires implementation findings"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract.to_dict(),
            "from_role": self.from_role.value,
            "to_role": self.to_role.value,
            "summary": self.summary,
            "base_sha": self.base_sha,
            "candidate_sha": self.candidate_sha,
            "changed_files": list(self.changed_files),
            "findings": [finding.to_dict() for finding in self.findings],
            "verification": self.verification,
            "usage": self.usage.to_dict(),
            "needs_user_input": list(self.needs_user_input),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AgentHandoff":
        return cls(
            contract=StageContract.from_dict(dict(value["contract"])),
            from_role=RoleId(str(value["from_role"])),
            to_role=RoleId(str(value["to_role"])),
            summary=str(value["summary"]),
            base_sha=str(value.get("base_sha", "")),
            candidate_sha=str(value.get("candidate_sha", "")),
            changed_files=tuple(str(item) for item in value.get("changed_files", [])),
            findings=tuple(
                ReviewFinding.from_dict(dict(item)) for item in value.get("findings", [])
            ),
            verification=dict(value.get("verification", {})),
            usage=TokenUsage.from_dict(dict(value.get("usage", {}))),
            needs_user_input=tuple(
                str(item) for item in value.get("needs_user_input", [])
            ),
        )


@dataclass(frozen=True)
class RunState:
    run_id: str
    phase: RunPhase = RunPhase.DISCUSSING
    stage_index: int = 0
    stage_count: int = 1
    repository: str = ""
    repository_identity: str = ""
    repository_head_sha: str = ""
    objective: str = ""
    repository_approved: bool = False
    plan_revision: int = 0
    plan_hash: str = ""
    approval_granted: bool = False
    approved_plan_revision: int = 0
    approved_plan_hash: str = ""
    approved_at: str = ""
    approved_by: str = ""
    resume_phase: RunPhase | None = None
    total_tokens: int = 0
    version: int = 0
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    last_error: str = ""

    def __post_init__(self) -> None:
        require_identifier(self.run_id, "run_id")
        if self.stage_count < 1:
            raise ValueError("stage_count must be at least one")
        if not 0 <= self.stage_index < self.stage_count:
            raise ValueError("stage_index is outside stage_count")
        if min(self.total_tokens, self.version, self.plan_revision, self.approved_plan_revision) < 0:
            raise ValueError("token, version, and revision values cannot be negative")
        TaskDefinitionState(
            run_id=self.run_id,
            objective=self.objective,
            repository_path=self.repository,
            repository_identity=self.repository_identity,
            repository_head_sha=self.repository_head_sha,
            repository_approved=self.repository_approved,
        )
        PlanPointerState(
            run_id=self.run_id,
            revision=self.plan_revision,
            plan_hash=self.plan_hash,
        )
        ExecutionApprovalState(
            run_id=self.run_id,
            granted=self.approval_granted,
            plan_revision=self.approved_plan_revision,
            plan_hash=self.approved_plan_hash,
            approved_at=self.approved_at,
            approved_by=self.approved_by,
        )
        if self.approval_granted and (
            self.approved_plan_revision != self.plan_revision
            or self.approved_plan_hash != self.plan_hash
        ):
            raise ValueError("approval must match the current plan revision and hash")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["phase"] = self.phase.value
        value["resume_phase"] = self.resume_phase.value if self.resume_phase else None
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RunState":
        resume = value.get("resume_phase")
        return cls(
            run_id=str(value["run_id"]),
            phase=RunPhase(str(value.get("phase", RunPhase.DISCUSSING.value))),
            stage_index=int(value.get("stage_index", 0)),
            stage_count=int(value.get("stage_count", 1)),
            repository=str(value.get("repository", "")),
            repository_identity=str(value.get("repository_identity", "")),
            repository_head_sha=str(value.get("repository_head_sha", "")),
            objective=str(value.get("objective", "")),
            repository_approved=bool(value.get("repository_approved", False)),
            plan_revision=int(value.get("plan_revision", 0)),
            plan_hash=str(value.get("plan_hash", "")),
            approval_granted=bool(value.get("approval_granted", False)),
            approved_plan_revision=int(value.get("approved_plan_revision", 0)),
            approved_plan_hash=str(value.get("approved_plan_hash", "")),
            approved_at=str(value.get("approved_at", "")),
            approved_by=str(value.get("approved_by", "")),
            resume_phase=RunPhase(str(resume)) if resume else None,
            total_tokens=int(value.get("total_tokens", 0)),
            version=int(value.get("version", 0)),
            created_at=str(value.get("created_at", utc_now())),
            updated_at=str(value.get("updated_at", utc_now())),
            last_error=str(value.get("last_error", "")),
        )
