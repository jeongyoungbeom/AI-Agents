"""Stable data contracts shared by the gateway, orchestrator, and roles."""

from .models import (
    AgentHandoff,
    ConversationSessionState,
    ExecutionApprovalState,
    PlanPointerState,
    ProjectSelectionState,
    ReviewFinding,
    RoleId,
    RunPhase,
    RunState,
    StageContract,
    TaskDefinitionState,
    TokenUsage,
    normalize_stage_scope,
)

__all__ = [
    "AgentHandoff",
    "ConversationSessionState",
    "ExecutionApprovalState",
    "PlanPointerState",
    "ProjectSelectionState",
    "ReviewFinding",
    "RoleId",
    "RunPhase",
    "RunState",
    "StageContract",
    "TaskDefinitionState",
    "TokenUsage",
    "normalize_stage_scope",
]
