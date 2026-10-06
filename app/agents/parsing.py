from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from app.contracts import ReviewFinding
from app.contracts import RoleId
from app.gateway.core.errors import InvalidAgentResponse
from app.gateway.core.models import (
    AgentCallRequest,
    AgentReply,
    MemoryScope,
    MemoryUpdate,
    ProposedStage,
)
from app.services.repository import RepositoryToolRequest
from app.services.message_intent import TaskIntent


@dataclass(frozen=True)
class RoleSummary:
    summary: str
    needs_user_input: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReviewSummary:
    summary: str
    findings: tuple[ReviewFinding, ...] = ()
    needs_user_input: tuple[str, ...] = ()


def extract_json_object(text: str) -> dict[str, Any]:
    value = text.strip()
    fenced = re.fullmatch(r"```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```", value, re.I)
    if fenced is not None:
        value = fenced.group(1).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise InvalidAgentResponse("에이전트 응답은 JSON 객체 하나여야 합니다.") from exc
    if not isinstance(parsed, dict):
        raise InvalidAgentResponse("에이전트 응답은 JSON 객체 하나여야 합니다.")
    return parsed


def parse_conversation_reply(text: str) -> AgentReply:
    value = extract_json_object(text)
    message = str(value.get("message", "")).strip()
    if not message:
        raise InvalidAgentResponse("대화 응답 message가 비어 있습니다.")
    raw_stages = value.get("stages", [])
    if not isinstance(raw_stages, list):
        raise InvalidAgentResponse("stages는 배열이어야 합니다.")
    stages: list[ProposedStage] = []
    for raw in raw_stages:
        if not isinstance(raw, dict):
            raise InvalidAgentResponse("각 stage는 객체여야 합니다.")
        stages.append(
            ProposedStage(
                objective=str(raw.get("objective", "")),
                scope=_strings(raw.get("scope", []), "scope"),
                acceptance_criteria=_strings(
                    raw.get("acceptance_criteria", []), "acceptance_criteria"
                ),
                verification_commands=_strings(
                    raw.get("verification_commands", []), "verification_commands"
                ),
                non_goals=_strings(raw.get("non_goals", []), "non_goals"),
            )
        )
    return AgentReply(
        text=message,
        stages=tuple(stages),
        decisions=_strings(value.get("decisions", []), "decisions"),
    )


def parse_team_conversation_reply(text: str, speaker: RoleId) -> AgentReply:
    """Parse one free-chat turn without trusting model-supplied ownership fields."""
    value = extract_json_object(text)
    message = str(value.get("message", "")).strip()

    raw_calls = value.get("calls", [])
    if not isinstance(raw_calls, list):
        raise InvalidAgentResponse("calls는 배열이어야 합니다.")
    if len(raw_calls) > 1:
        raise InvalidAgentResponse("한 응답에서는 다른 에이전트를 한 명만 호출할 수 있습니다.")
    calls: list[AgentCallRequest] = []
    for raw in raw_calls:
        if not isinstance(raw, dict):
            raise InvalidAgentResponse("각 call은 객체여야 합니다.")
        try:
            target = RoleId(str(raw.get("to_role", "")))
            calls.append(
                AgentCallRequest(
                    from_role=speaker,
                    to_role=target,
                    purpose=str(raw.get("purpose", "")).strip(),
                    mode=raw.get("mode", "independent"),
                )
            )
        except (TypeError, ValueError) as exc:
            raise InvalidAgentResponse(f"에이전트 호출 형식 오류: {exc}") from exc

    raw_updates = value.get("memory_updates", [])
    if not isinstance(raw_updates, list):
        raise InvalidAgentResponse("memory_updates는 배열이어야 합니다.")
    if len(raw_updates) > 2:
        raise InvalidAgentResponse("한 응답의 기억 갱신은 두 개를 넘을 수 없습니다.")
    updates: list[MemoryUpdate] = []
    for raw in raw_updates:
        if not isinstance(raw, dict):
            raise InvalidAgentResponse("각 memory_update는 객체여야 합니다.")
        try:
            scope = MemoryScope(str(raw.get("scope", "")))
            raw_role = str(raw.get("role_id", "shared")).strip()
            if raw_role in {"", "shared"}:
                memory_role = None
            else:
                memory_role = RoleId(raw_role)
                if memory_role != speaker:
                    raise ValueError("다른 역할의 비공개 기억은 갱신할 수 없습니다.")
            updates.append(
                MemoryUpdate(
                    scope=scope,
                    content=str(raw.get("content", "")).strip(),
                    role_id=memory_role,
                )
            )
        except (TypeError, ValueError) as exc:
            raise InvalidAgentResponse(f"기억 갱신 형식 오류: {exc}") from exc

    raw_tools = value.get("repository_tools", [])
    if not isinstance(raw_tools, list):
        raise InvalidAgentResponse("repository_tools는 배열이어야 합니다.")
    if len(raw_tools) > 3:
        raise InvalidAgentResponse("한 응답의 저장소 도구 요청은 세 개를 넘을 수 없습니다.")
    tools: list[RepositoryToolRequest] = []
    for raw in raw_tools:
        if not isinstance(raw, dict):
            raise InvalidAgentResponse("각 repository_tool은 객체여야 합니다.")
        try:
            tools.append(RepositoryToolRequest.from_dict(raw))
        except (TypeError, ValueError) as exc:
            raise InvalidAgentResponse(f"저장소 도구 요청 형식 오류: {exc}") from exc
    if len(set(tools)) != len(tools):
        raise InvalidAgentResponse("같은 저장소 도구 요청을 중복할 수 없습니다.")
    if tools and (calls or updates):
        raise InvalidAgentResponse(
            "저장소 도구를 요청하는 중간 응답에는 호출이나 기억 갱신을 함께 넣을 수 없습니다."
        )
    # A repository lookup is an internal round trip, not a user-visible reply.
    # Hermes correctly emits an empty message for this case; the router executes
    # the approved lookup and only sends the later final answer to Telegram.
    if not message and not tools:
        raise InvalidAgentResponse("자유 대화 응답 message가 비어 있습니다.")

    intent = value.get("intent", {})
    if (not isinstance(intent, dict)
        or not isinstance(intent.get("write_forbidden", False), bool)
        or not isinstance(intent.get("question_purpose", ""), str)):
        raise InvalidAgentResponse("intent와 write_forbidden 형식이 올바르지 않습니다.")
    try:
        task_intent = TaskIntent(intent.get("task_intent", "answer"))
    except (ValueError, TypeError) as exc:
        raise InvalidAgentResponse("알 수 없는 task_intent입니다.") from exc
    if task_intent != TaskIntent.ANSWER and (tools or calls or updates):
        raise InvalidAgentResponse("모드 전환과 조회·호출·기억 갱신을 함께 요청할 수 없습니다.")
    questions = _strings(value.get("needs_user_input", []), "needs_user_input")
    if len(questions) > 3 or (questions and (tools or calls or updates or task_intent != TaskIntent.ANSWER)):
        raise InvalidAgentResponse("사용자 질문은 세 개까지이며 다른 행동과 함께 요청할 수 없습니다.")

    return AgentReply(
        text=message,
        calls=tuple(calls),
        memory_updates=tuple(updates),
        repository_tools=tuple(tools),
        task_intent=task_intent,
        write_forbidden=intent.get("write_forbidden", False),
        question_purpose=intent.get("question_purpose", "")[:512],
        needs_user_input=questions,
        execution_intent=_execution_intent(value.get("execution_intent")),
    )


def _execution_intent(value):
    if value is None:
        return None
    if (not isinstance(value, dict) or value.get('action') not in
            {'question', 'supplement', 'redirect', 'test_first', 'scope_change'}):
        raise InvalidAgentResponse('execution_intent action 형식이 올바르지 않습니다.')
    paths = value.get('requested_scope', [])
    if not isinstance(paths, list) or len(paths) > 32 or any(not isinstance(p, str) or not p.strip() for p in paths):
        raise InvalidAgentResponse('execution_intent requested_scope 형식이 올바르지 않습니다.')
    clarifies = value.get('clarifies_input_ids', [])
    if (not isinstance(clarifies, list) or len(clarifies) > 32
            or any(type(item) is not int or item <= 0 for item in clarifies)):
        raise InvalidAgentResponse('execution_intent clarifies_input_ids 형식이 올바르지 않습니다.')
    return {'action': value['action'], 'requested_scope': paths, 'clarifies_input_ids': clarifies}


def parse_role_summary(text: str) -> RoleSummary:
    value = extract_json_object(text)
    summary = str(value.get("summary", "")).strip()
    if not summary:
        raise InvalidAgentResponse("역할 응답 summary가 비어 있습니다.")
    return RoleSummary(
        summary=summary,
        needs_user_input=_strings(value.get("needs_user_input", []), "needs_user_input"),
    )


def parse_review_summary(text: str) -> ReviewSummary:
    value = extract_json_object(text)
    summary = str(value.get("summary", "")).strip()
    raw_findings = value.get("findings", [])
    if not summary or not isinstance(raw_findings, list):
        raise InvalidAgentResponse("리뷰 summary/findings 형식이 올바르지 않습니다.")
    findings: list[ReviewFinding] = []
    for raw in raw_findings:
        if not isinstance(raw, dict):
            raise InvalidAgentResponse("각 finding은 객체여야 합니다.")
        if "category" not in raw:
            raise InvalidAgentResponse("리뷰 finding에 category가 필요합니다.")
        try:
            findings.append(ReviewFinding.from_dict(raw))
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidAgentResponse(f"리뷰 finding 형식 오류: {exc}") from exc
    return ReviewSummary(
        summary=summary,
        findings=tuple(findings),
        needs_user_input=_strings(value.get("needs_user_input", []), "needs_user_input"),
    )


def _strings(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise InvalidAgentResponse(f"{field_name}는 배열이어야 합니다.")
    items = tuple(str(item).strip() for item in value)
    if any(not item for item in items):
        raise InvalidAgentResponse(f"{field_name}에 빈 항목이 있습니다.")
    return items
