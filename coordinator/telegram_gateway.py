from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import agent_pipeline as pipeline


STATE_PATH = pipeline.ARTIFACTS_ROOT / "telegram-gateway-state.json"
PLANNING_ROOT = pipeline.ARTIFACTS_ROOT / "telegram-planning"
PLANS_ROOT = pipeline.AI_ROOT / "plans"
TRANSIENT_PHASES = {"PLANNING", "RUNNING", "STOPPING"}
BUSY_PHASES = {"PLANNING", "RUNNING", "STOPPING"}


class GatewayError(RuntimeError):
    pass


def parse_ids(value: str) -> frozenset[int]:
    ids: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            ids.add(int(item))
        except ValueError as exc:
            raise GatewayError(f"Telegram ID must be numeric: {item}") from exc
    return frozenset(ids)


@dataclass(frozen=True)
class BotConfig:
    token: str
    allowed_users: frozenset[int]
    allowed_chats: frozenset[int]

    @classmethod
    def load(cls, env_path: Path = pipeline.ENV_PATH) -> "BotConfig":
        values = pipeline.load_env(env_path)
        return cls(
            token=values.get("TELEGRAM_BOT_TOKEN", "").strip(),
            allowed_users=parse_ids(values.get("TELEGRAM_ALLOWED_USERS", "")),
            allowed_chats=parse_ids(values.get("TELEGRAM_ALLOWED_CHATS", "")),
        )

    def validate(self) -> None:
        if not self.token:
            raise GatewayError(
                "TELEGRAM_BOT_TOKEN is blank in D:\\AI-Agents\\coordinator\\.env"
            )
        if not self.allowed_users:
            raise GatewayError(
                "TELEGRAM_ALLOWED_USERS is blank; at least one numeric user ID is required"
            )


def split_message(text: str, limit: int = 3900) -> list[str]:
    text = text.strip() or "(empty)"
    chunks: list[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    chunks.append(text)
    return chunks


class TelegramAPI:
    def __init__(self, token: str):
        self._base = f"https://api.telegram.org/bot{token}/"

    def call(self, method: str, data: dict[str, Any] | None = None, timeout: int = 40) -> Any:
        encoded = urllib.parse.urlencode(data or {}).encode("utf-8")
        request = urllib.request.Request(self._base + method, data=encoded, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            raise GatewayError(f"Telegram API {method} failed: {type(exc).__name__}") from exc
        if not payload.get("ok"):
            raise GatewayError(
                f"Telegram API {method} rejected the request: "
                f"{payload.get('description', 'unknown error')}"
            )
        return payload.get("result")

    def get_me(self) -> dict[str, Any]:
        return dict(self.call("getMe"))

    def delete_webhook(self) -> None:
        self.call("deleteWebhook", {"drop_pending_updates": "false"})

    def set_commands(self) -> None:
        commands = [
            {"command": "project", "description": "Git 프로젝트 절대경로 지정"},
            {"command": "task", "description": "새 개발 작업 시작"},
            {"command": "plan", "description": "생성된 실행 계획 확인"},
            {"command": "approve", "description": "계획 승인 및 실행"},
            {"command": "revise", "description": "계획 수정 요청"},
            {"command": "status", "description": "현재 상태 확인"},
            {"command": "stop", "description": "안전 중단 요청"},
            {"command": "new", "description": "현재 요청 초기화"},
            {"command": "help", "description": "사용법"},
        ]
        self.call("setMyCommands", {"commands": json.dumps(commands, ensure_ascii=False)})

    def get_updates(self, offset: int) -> list[dict[str, Any]]:
        result = self.call(
            "getUpdates",
            {
                "offset": str(offset),
                "timeout": "25",
                "allowed_updates": json.dumps(["message"]),
            },
            timeout=35,
        )
        return list(result or [])

    def send(self, chat_id: int, text: str) -> None:
        for chunk in split_message(text):
            self.call(
                "sendMessage",
                {
                    "chat_id": str(chat_id),
                    "text": chunk,
                    "disable_web_page_preview": "true",
                },
            )


def default_chat_state() -> dict[str, Any]:
    return {
        "repository": "",
        "phase": "IDLE",
        "task": "",
        "answers": [],
        "pending_questions": [],
        "plan_path": "",
        "run_id": "",
        "last_error": "",
        "updated_at": pipeline.utc_now(),
    }


class GatewayStore:
    def __init__(self, path: Path = STATE_PATH):
        self.path = path
        self.lock = threading.RLock()
        if path.exists():
            self.data = pipeline.read_json(path)
        else:
            self.data = {"version": 1, "last_update_id": 0, "chats": {}}
            self._save()
        self.data.setdefault("chats", {})
        self.data.setdefault("last_update_id", 0)

    def _save(self) -> None:
        pipeline.write_json(self.path, self.data)

    def offset(self) -> int:
        with self.lock:
            return int(self.data.get("last_update_id", 0)) + 1

    def set_update_id(self, update_id: int) -> None:
        with self.lock:
            self.data["last_update_id"] = max(
                int(self.data.get("last_update_id", 0)), int(update_id)
            )
            self._save()

    def chat(self, chat_id: int) -> dict[str, Any]:
        key = str(chat_id)
        with self.lock:
            if key not in self.data["chats"]:
                self.data["chats"][key] = default_chat_state()
                self._save()
            return json.loads(json.dumps(self.data["chats"][key]))

    def update_chat(self, chat_id: int, **changes: Any) -> dict[str, Any]:
        key = str(chat_id)
        with self.lock:
            state = self.data["chats"].setdefault(key, default_chat_state())
            state.update(changes)
            state["updated_at"] = pipeline.utc_now()
            self._save()
            return json.loads(json.dumps(state))

    def recover_interrupted(self) -> None:
        with self.lock:
            changed = False
            for state in self.data["chats"].values():
                if state.get("phase") in TRANSIENT_PHASES:
                    state["phase"] = "INTERRUPTED"
                    state["last_error"] = (
                        "Gateway restarted while work was active. Inspect /status before retrying."
                    )
                    state["updated_at"] = pipeline.utc_now()
                    changed = True
            if changed:
                self._save()


def extract_planning_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("status") in {"ready", "needs_input"}:
            return value
    raise GatewayError("Developer did not return the required planning JSON")


def read_text_limited(path: Path, limit: int = 12000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def validate_verification_command(command: str) -> str:
    command = command.strip()
    if not command:
        raise GatewayError("Verification command cannot be blank")
    if "\n" in command or "\r" in command:
        raise GatewayError("Verification commands must be one line each")
    if re.search(r"(?:&&|\|\||[;|<>`])", command):
        raise GatewayError(
            f"Verification command contains shell chaining or redirection: {command}"
        )
    forbidden = (
        r"\bgit\s+(?:push|merge|reset|clean|checkout|switch|commit|add)\b",
        r"\b(?:rm|rmdir|del|erase|format|shutdown|taskkill)\b",
        r"\b(?:remove-item|stop-process|restart-computer|stop-computer)\b",
    )
    lowered = command.lower()
    if any(re.search(pattern, lowered) for pattern in forbidden):
        raise GatewayError(f"Unsafe verification command was rejected: {command}")
    return command


def build_repository_snapshot(repository: pipeline.GitRepository) -> str:
    tracked = repository.run("ls-files").stdout.splitlines()
    path_listing = "\n".join(tracked[:2500])
    if len(tracked) > 2500:
        path_listing += f"\n... ({len(tracked) - 2500} more tracked files)"

    important_names = {
        "agents.md",
        "readme.md",
        "readme.txt",
        "package.json",
        "pyproject.toml",
        "requirements.txt",
        "cargo.toml",
        "go.mod",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "settings.gradle",
        "settings.gradle.kts",
        "composer.json",
        "gemfile",
    }
    selected: list[str] = []
    for relative in tracked:
        path = Path(relative)
        if path.name.lower() == "agents.md" or (
            len(path.parts) <= 2 and path.name.lower() in important_names
        ):
            selected.append(relative)
        if len(selected) >= 30:
            break

    documents: list[str] = []
    budget = 90000
    for relative in selected:
        candidate = (repository.path / relative).resolve()
        try:
            candidate.relative_to(repository.path)
        except ValueError:
            continue
        content = read_text_limited(candidate)
        block = f"\n--- {relative} ---\n{content}"
        if len(block) > budget:
            break
        documents.append(block)
        budget -= len(block)

    return (
        f"Repository: {repository.path}\n"
        f"Branch: {repository.current_branch()}\n"
        f"Working tree status:\n{repository.status() or '(clean)'}\n\n"
        f"Tracked files:\n{path_listing}\n"
        + "".join(documents)
    )


def make_run_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"TG-{stamp}-{uuid.uuid4().hex[:6]}"


def normalize_plan(
    planning: dict[str, Any], repository: Path, run_id: str
) -> dict[str, Any]:
    stages_raw = planning.get("stages")
    if not isinstance(stages_raw, list) or not stages_raw:
        raise GatewayError("Developer returned a ready plan without stages")
    stages: list[dict[str, Any]] = []
    for index, raw in enumerate(stages_raw, start=1):
        if not isinstance(raw, dict):
            raise GatewayError("Developer returned an invalid stage")
        stage_id = f"stage-{index:03d}"
        criteria = raw.get("acceptance_criteria", [])
        commands = raw.get("verification_commands", [])
        if not isinstance(criteria, list) or not criteria:
            raise GatewayError(f"{stage_id} has no acceptance criteria")
        if not isinstance(commands, list) or not commands:
            raise GatewayError(f"{stage_id} has no verification commands")
        stages.append(
            {
                "id": stage_id,
                "objective": str(raw.get("objective", "")).strip(),
                "scope": list(raw.get("scope", [])),
                "non_goals": list(raw.get("non_goals", [])),
                "acceptance_criteria": [str(item) for item in criteria],
                "verification_commands": [
                    validate_verification_command(str(item)) for item in commands
                ],
            }
        )
    plan = {
        "run_id": run_id,
        "repository": {
            "path": str(repository.resolve()),
            "mode": "in_place",
            "require_clean": True,
        },
        "branch": f"ai/{run_id.lower()}",
        "max_improvement_attempts": 2,
        "summary": str(planning.get("summary", "")).strip(),
        "stages": stages,
    }
    return pipeline.validate_plan(plan)


def render_plan(plan: dict[str, Any]) -> str:
    lines = [
        f"📋 실행 계획: {plan['run_id']}",
        f"프로젝트: {plan['repository']['path']}",
        f"브랜치: {plan['branch']}",
    ]
    if plan.get("summary"):
        lines.append(f"요약: {plan['summary']}")
    for index, stage in enumerate(plan["stages"], start=1):
        lines.append(f"\n{index}. {stage['objective']}")
        for criterion in stage["acceptance_criteria"]:
            lines.append(f"   완료조건: {criterion}")
        for command in stage["verification_commands"]:
            lines.append(f"   검증: {command}")
    lines.append("\n실행하려면 /approve")
    lines.append("수정하려면 /revise 수정할 내용")
    return "\n".join(lines)


class ChatNotifier:
    def __init__(self, api: TelegramAPI, chat_id: int):
        self.api = api
        self.chat_id = chat_id

    def send(self, message: str) -> None:
        try:
            self.api.send(self.chat_id, message)
        except Exception as exc:
            print(
                f"Telegram notification failed without stopping the pipeline: {exc}",
                flush=True,
            )


HELP_TEXT = """
개발 AI 게이트웨이

1. /project C:\\path\\to\\git-repository
2. /task 만들고 싶은 기능
3. 개발 AI가 질문하면 그냥 답장
4. 계획을 확인하고 /approve
5. 개발 → 리뷰 → 보완 → 검증을 단계별로 자동 실행

명령: /project /task /plan /approve /revise /status /stop /new /help
절대경로의 Git 프로젝트만 사용하며 push·merge는 하지 않습니다.
""".strip()


class TelegramGateway:
    def __init__(
        self,
        config: BotConfig,
        settings: pipeline.Settings,
        api: TelegramAPI | None = None,
        store: GatewayStore | None = None,
    ):
        self.config = config
        self.settings = settings
        self.api = api or TelegramAPI(config.token)
        self.store = store or GatewayStore()
        self.jobs: dict[int, threading.Thread] = {}
        self.cancel_events: dict[int, threading.Event] = {}
        self.jobs_lock = threading.RLock()

    def authorized(self, user_id: int, chat_id: int, chat_type: str) -> bool:
        if user_id not in self.config.allowed_users:
            return False
        if chat_type == "private":
            return True
        return chat_id in self.config.allowed_chats

    def send(self, chat_id: int, text: str) -> None:
        self.api.send(chat_id, text)

    def handle_update(self, update: dict[str, Any]) -> None:
        message = update.get("message")
        if not isinstance(message, dict):
            return
        text = message.get("text")
        sender = message.get("from", {})
        chat = message.get("chat", {})
        if not isinstance(text, str):
            return
        user_id = int(sender.get("id", 0))
        chat_id = int(chat.get("id", 0))
        chat_type = str(chat.get("type", ""))
        if not self.authorized(user_id, chat_id, chat_type):
            return
        self.route(chat_id, text.strip())

    def route(self, chat_id: int, text: str) -> None:
        if not text:
            return
        if text.startswith("/"):
            first, _, argument = text.partition(" ")
            command = first.split("@", 1)[0].lower()
            handlers = {
                "/start": self.command_help,
                "/help": self.command_help,
                "/project": self.command_project,
                "/task": self.command_task,
                "/plan": self.command_plan,
                "/approve": self.command_approve,
                "/revise": self.command_revise,
                "/status": self.command_status,
                "/stop": self.command_stop,
                "/new": self.command_new,
            }
            handler = handlers.get(command)
            if handler is None:
                self.send(chat_id, "알 수 없는 명령이야. /help 로 확인해줘.")
                return
            handler(chat_id, argument.strip())
            return

        state = self.store.chat(chat_id)
        if state["phase"] == "NEEDS_INPUT":
            answers = list(state.get("answers", []))
            questions = state.get("pending_questions", [])
            answers.append({"questions": questions, "answer": text})
            self.store.update_chat(chat_id, answers=answers, pending_questions=[])
            self.start_planning(chat_id)
            return
        if state["phase"] in BUSY_PHASES:
            self.send(chat_id, "현재 작업 중이야. /status 또는 /stop 을 사용해줘.")
            return
        if state["phase"] == "AWAITING_APPROVAL":
            self.send(chat_id, "계획 승인은 /approve, 수정은 /revise 내용 으로 해줘.")
            return
        self.command_task(chat_id, text)

    def command_help(self, chat_id: int, _argument: str = "") -> None:
        self.store.chat(chat_id)
        self.send(chat_id, HELP_TEXT)

    def command_project(self, chat_id: int, argument: str) -> None:
        state = self.store.chat(chat_id)
        if state["phase"] in BUSY_PHASES or state["phase"] == "AWAITING_APPROVAL":
            self.send(chat_id, "진행 중인 요청을 /stop 또는 /new 로 정리한 뒤 바꿔줘.")
            return
        if not argument:
            current = state.get("repository") or "(미지정)"
            self.send(chat_id, f"현재 프로젝트: {current}\n\uc0ac용법: /project C:\\path\\to\\repo")
            return
        raw_path = argument.strip().strip('"').strip("'")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            self.send(chat_id, "프로젝트는 절대경로로 입력해줘.")
            return
        try:
            repository = pipeline.GitRepository(path)
            repository.validate()
            branch = repository.current_branch() or "(detached HEAD)"
            dirty = bool(repository.status().strip())
        except (pipeline.PipelineError, OSError) as exc:
            self.send(chat_id, f"프로젝트를 선택할 수 없어: {exc}")
            return
        self.store.update_chat(
            chat_id,
            repository=str(path.resolve()),
            phase="IDLE",
            task="",
            answers=[],
            pending_questions=[],
            plan_path="",
            run_id="",
            last_error="",
        )
        note = "\n⚠️ 현재 미커밋 변경이 있어. 실행 전에 commit 또는 stash이 필요해." if dirty else ""
        self.send(chat_id, f"✅ 프로젝트 선택됨\n{path.resolve()}\n브랜치: {branch}{note}")

    def command_task(self, chat_id: int, argument: str) -> None:
        state = self.store.chat(chat_id)
        if state["phase"] in BUSY_PHASES:
            self.send(chat_id, "이미 작업 중이야. /status 로 확인해줘.")
            return
        if not state.get("repository"):
            self.send(chat_id, "먼저 /project 절대경로 로 Git 프로젝트를 지정해줘.")
            return
        if not argument:
            self.send(chat_id, "사용법: /task 개발할 내용")
            return
        run_id = make_run_id()
        self.store.update_chat(
            chat_id,
            phase="IDLE",
            task=argument,
            answers=[],
            pending_questions=[],
            plan_path="",
            run_id=run_id,
            last_error="",
        )
        self.start_planning(chat_id)

    def start_planning(self, chat_id: int) -> None:
        with self.jobs_lock:
            existing = self.jobs.get(chat_id)
            if existing and existing.is_alive():
                self.send(chat_id, "현재 계획 작업이 진행 중이야.")
                return
            event = threading.Event()
            self.cancel_events[chat_id] = event
            self.store.update_chat(chat_id, phase="PLANNING")
            thread = threading.Thread(
                target=self._planning_worker,
                args=(chat_id, event),
                name=f"telegram-plan-{chat_id}",
                daemon=True,
            )
            self.jobs[chat_id] = thread
            thread.start()
        self.send(chat_id, "🧠 개발 AI가 프로젝트를 분석하고 단계별 계획을 만들고 있어.")

    def _planning_worker(self, chat_id: int, cancel_event: threading.Event) -> None:
        try:
            state = self.store.chat(chat_id)
            repository = pipeline.GitRepository(Path(state["repository"]))
            repository.validate()
            snapshot = build_repository_snapshot(repository)
            prompt = f"""
You are the developer agent in planning-only mode. Do not edit files and do not run tools.
Create a staged implementation plan for the user's request using only the supplied repository snapshot.
Ask questions only when a real product, architecture, data-loss, security, or scope decision cannot be inferred safely.
Do not ask about minor implementation details you can decide from repository conventions.

User request:
{state['task']}

Previous answers:
{json.dumps(state.get('answers', []), ensure_ascii=False)}

Repository snapshot:
{snapshot}

Return only one JSON object. If input is required:
{{"status":"needs_input","questions":["question 1"],"summary":"why it matters"}}

If ready:
{{"status":"ready","summary":"short Korean summary","questions":[],"stages":[{{"objective":"...","scope":["..."],"non_goals":["..."],"acceptance_criteria":["..."],"verification_commands":["..."]}}]}}

Use commands that already fit the detected repository. Every stage must have deterministic verification commands.
Keep stages independently reviewable and reasonably small. Never include push, merge, reset, clean, delete, deployment, or credential commands.
""".strip()
            run_id = state["run_id"]
            output_dir = PLANNING_ROOT / run_id / f"attempt-{len(state.get('answers', [])) + 1:02d}"
            text = pipeline.HermesRunner(self.settings).run(
                "developer",
                prompt,
                repository.path,
                output_dir,
                "planner",
                toolsets_override="safe",
            )
            if cancel_event.is_set():
                self.store.update_chat(chat_id, phase="CANCELLED")
                self.send(chat_id, "⏹ 계획 작업을 취소했어.")
                return
            planning = extract_planning_json(text)
            if planning["status"] == "needs_input":
                questions = planning.get("questions", [])
                if not isinstance(questions, list) or not questions:
                    raise GatewayError("Developer requested input without a question")
                questions = [str(item).strip() for item in questions[:3] if str(item).strip()]
                self.store.update_chat(
                    chat_id,
                    phase="NEEDS_INPUT",
                    pending_questions=questions,
                )
                rendered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, start=1))
                self.send(chat_id, f"❓ 개발 AI의 확인이 필요해.\n{rendered}\n\n그냥 답장하면 계획을 이어갈게.")
                return
            plan = normalize_plan(planning, repository.path, run_id)
            plan_path = PLANS_ROOT / f"{run_id}.json"
            pipeline.write_json(plan_path, plan)
            self.store.update_chat(
                chat_id,
                phase="AWAITING_APPROVAL",
                plan_path=str(plan_path),
                pending_questions=[],
            )
            self.send(chat_id, render_plan(plan))
        except Exception as exc:
            self.store.update_chat(chat_id, phase="FAILED", last_error=str(exc))
            self.send(chat_id, f"❌ 계획 생성 중단: {exc}")

    def command_plan(self, chat_id: int, _argument: str = "") -> None:
        state = self.store.chat(chat_id)
        plan_path = state.get("plan_path")
        if not plan_path:
            self.send(chat_id, "아직 생성된 계획이 없어.")
            return
        try:
            self.send(chat_id, render_plan(pipeline.read_json(Path(plan_path))))
        except pipeline.PipelineError as exc:
            self.send(chat_id, f"계획을 읽을 수 없어: {exc}")

    def command_revise(self, chat_id: int, argument: str) -> None:
        state = self.store.chat(chat_id)
        if state["phase"] != "AWAITING_APPROVAL":
            self.send(chat_id, "수정할 대기 중 계획이 없어.")
            return
        if not argument:
            self.send(chat_id, "사용법: /revise 수정할 내용")
            return
        answers = list(state.get("answers", []))
        answers.append({"revision_request": argument})
        self.store.update_chat(chat_id, answers=answers, plan_path="")
        self.start_planning(chat_id)

    def command_approve(self, chat_id: int, _argument: str = "") -> None:
        state = self.store.chat(chat_id)
        if state["phase"] != "AWAITING_APPROVAL" or not state.get("plan_path"):
            self.send(chat_id, "승인할 대기 중 계획이 없어.")
            return
        try:
            repository = pipeline.GitRepository(Path(state["repository"]))
            repository.validate()
            repository.require_clean()
            repository.require_identity()
        except pipeline.PipelineError as exc:
            self.send(chat_id, f"실행할 수 없어: {exc}")
            return
        with self.jobs_lock:
            existing = self.jobs.get(chat_id)
            if existing and existing.is_alive():
                self.send(chat_id, "이미 작업 중이야.")
                return
            event = threading.Event()
            self.cancel_events[chat_id] = event
            self.store.update_chat(chat_id, phase="RUNNING")
            thread = threading.Thread(
                target=self._run_worker,
                args=(chat_id, Path(state["plan_path"]), event),
                name=f"telegram-run-{chat_id}",
                daemon=True,
            )
            self.jobs[chat_id] = thread
            thread.start()

    def _run_worker(
        self, chat_id: int, plan_path: Path, cancel_event: threading.Event
    ) -> None:
        notifier = ChatNotifier(self.api, chat_id)
        coordinator = pipeline.Coordinator(
            self.settings,
            notifier=notifier,
            cancel_check=cancel_event.is_set,
        )
        try:
            run_dir = coordinator.run(plan_path)
            state = pipeline.read_json(run_dir / "state.json")
            self.store.update_chat(
                chat_id,
                phase="COMPLETED",
                last_error="",
                final_sha=state.get("final_sha", ""),
            )
        except pipeline.PipelineCancelled as exc:
            self.store.update_chat(chat_id, phase="CANCELLED", last_error=str(exc))
        except Exception as exc:
            self.store.update_chat(chat_id, phase="FAILED", last_error=str(exc))

    def command_status(self, chat_id: int, _argument: str = "") -> None:
        state = self.store.chat(chat_id)
        lines = [
            f"상태: {state['phase']}",
            f"프로젝트: {state.get('repository') or '(미지정)'}",
            f"실행 ID: {state.get('run_id') or '(없음)'}",
        ]
        run_id = state.get("run_id")
        if run_id:
            run_state_path = pipeline.ARTIFACTS_ROOT / run_id / "state.json"
            if run_state_path.exists():
                run_state = pipeline.read_json(run_state_path)
                lines.append(f"파이프라인: {run_state.get('status', 'UNKNOWN')}")
                stages = run_state.get("stages", [])
                if stages:
                    last = stages[-1]
                    lines.append(f"현재 단계: {last.get('id')} / {last.get('status')}")
        if state.get("last_error"):
            lines.append(f"최근 오류: {state['last_error']}")
        self.send(chat_id, "\n".join(lines))

    def command_stop(self, chat_id: int, _argument: str = "") -> None:
        state = self.store.chat(chat_id)
        phase = state["phase"]
        if phase in {"PLANNING", "RUNNING", "STOPPING"}:
            event = self.cancel_events.get(chat_id)
            if event:
                event.set()
            self.store.update_chat(chat_id, phase="STOPPING")
            if phase == "RUNNING":
                self.send(chat_id, "⏹ 안전 중단을 예약했어. 현재 단계의 리뷰·보완·검증을 마친 뒤 멈출게.")
            else:
                self.send(chat_id, "⏹ 중단을 요청했어.")
            return
        if phase in {"AWAITING_APPROVAL", "NEEDS_INPUT"}:
            self.store.update_chat(
                chat_id,
                phase="CANCELLED",
                plan_path="",
                pending_questions=[],
            )
            self.send(chat_id, "⏹ 요청을 취소했어.")
            return
        self.send(chat_id, "현재 진행 중인 작업이 없어.")

    def command_new(self, chat_id: int, _argument: str = "") -> None:
        state = self.store.chat(chat_id)
        if state["phase"] in BUSY_PHASES:
            self.send(chat_id, "먼저 /stop 으로 진행 중인 작업을 중단해줘.")
            return
        self.store.update_chat(
            chat_id,
            phase="IDLE",
            task="",
            answers=[],
            pending_questions=[],
            plan_path="",
            run_id="",
            last_error="",
        )
        self.send(chat_id, "새 요청을 받을 준비가 됐어. 프로젝트는 그대로 유지했어.")

    def check(self, online: bool = True) -> str:
        self.config.validate()
        issues = pipeline.HermesRunner(self.settings).preflight(require_auth=True)
        if issues:
            raise GatewayError("Preflight failed: " + "; ".join(issues))
        if online:
            me = self.api.get_me()
            return f"Telegram bot @{me.get('username', '(unknown)')} is ready"
        return "Local configuration is ready"

    def run_forever(self) -> None:
        print(self.check(online=True), flush=True)
        self.store.recover_interrupted()
        self.api.delete_webhook()
        self.api.set_commands()
        print("Telegram gateway started. Press Ctrl+C to stop.", flush=True)
        while True:
            try:
                updates = self.api.get_updates(self.store.offset())
                for update in updates:
                    update_id = int(update.get("update_id", 0))
                    try:
                        self.handle_update(update)
                    except Exception as exc:
                        print(f"Update {update_id} failed: {exc}", flush=True)
                    finally:
                        self.store.set_update_id(update_id)
            except GatewayError as exc:
                print(f"Gateway connection warning: {exc}; retrying in 5 seconds", flush=True)
                time.sleep(5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Telegram gateway for the three-agent pipeline")
    parser.add_argument("command", nargs="?", choices=("run", "check"), default="run")
    parser.add_argument("--offline", action="store_true", help="check local configuration only")
    args = parser.parse_args(argv)
    try:
        config = BotConfig.load()
        settings = pipeline.Settings.load()
        gateway = TelegramGateway(config, settings)
        if args.command == "check":
            print(gateway.check(online=not args.offline))
            return 0
        gateway.run_forever()
    except KeyboardInterrupt:
        print("Telegram gateway stopped.")
        return 0
    except (GatewayError, pipeline.PipelineError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
