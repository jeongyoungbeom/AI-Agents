from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


COORDINATOR_DIR = Path(__file__).resolve().parent
AI_ROOT = COORDINATOR_DIR.parent
SETTINGS_PATH = COORDINATOR_DIR / "config" / "settings.json"
ENV_PATH = COORDINATOR_DIR / ".env"
ARTIFACTS_ROOT = AI_ROOT / "artifacts"
LOCKS_ROOT = ARTIFACTS_ROOT / "locks"
PROFILES = ("developer", "reviewer", "improver")


class PipelineError(RuntimeError):
    pass


class PipelineCancelled(PipelineError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PipelineError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PipelineError(f"Invalid JSON in {path}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def safe_id(value: str, field: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", value):
        raise PipelineError(
            f"{field} must be 1-80 characters using letters, numbers, dot, dash, or underscore"
        )
    return value


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@dataclass(frozen=True)
class Settings:
    hermes_executable: Path
    hermes_home: Path
    provider: str
    model: str
    timeout_seconds: int
    reasoning: dict[str, str]
    toolsets: dict[str, str]

    @classmethod
    def load(cls, path: Path = SETTINGS_PATH) -> "Settings":
        raw = read_json(path)
        return cls(
            hermes_executable=Path(raw["hermes_executable"]),
            hermes_home=Path(raw["hermes_home"]),
            provider=str(raw.get("provider", "openai-codex")),
            model=str(raw.get("model", "")),
            timeout_seconds=int(raw.get("timeout_seconds", 3600)),
            reasoning=dict(raw.get("reasoning", {})),
            toolsets=dict(raw.get("toolsets", {})),
        )


def validate_plan(raw: dict[str, Any], require_repository: bool = True) -> dict[str, Any]:
    run_id = safe_id(str(raw.get("run_id", "")), "run_id")
    repository = raw.get("repository")
    if not isinstance(repository, dict):
        raise PipelineError("repository must be an object")
    repository_path = Path(str(repository.get("path", ""))).expanduser()
    if not str(repository_path):
        raise PipelineError("repository.path is required")
    if repository.get("mode", "in_place") != "in_place":
        raise PipelineError("MVP currently supports repository.mode=in_place only")
    if require_repository and not repository_path.is_dir():
        raise PipelineError(f"Repository directory does not exist: {repository_path}")

    stages = raw.get("stages")
    if not isinstance(stages, list) or not stages:
        raise PipelineError("stages must contain at least one stage")
    seen: set[str] = set()
    default_commands = raw.get("verification_commands", [])
    if not isinstance(default_commands, list):
        raise PipelineError("verification_commands must be a list")
    for stage in stages:
        if not isinstance(stage, dict):
            raise PipelineError("every stage must be an object")
        stage_id = safe_id(str(stage.get("id", "")), "stage.id")
        if stage_id in seen:
            raise PipelineError(f"duplicate stage id: {stage_id}")
        seen.add(stage_id)
        if not str(stage.get("objective", "")).strip():
            raise PipelineError(f"stage {stage_id} requires objective")
        criteria = stage.get("acceptance_criteria", [])
        if not isinstance(criteria, list) or not criteria:
            raise PipelineError(f"stage {stage_id} requires acceptance_criteria")
        commands = stage.get("verification_commands", default_commands)
        if not isinstance(commands, list) or not commands:
            raise PipelineError(f"stage {stage_id} requires verification_commands")
        if not all(isinstance(command, str) and command.strip() for command in commands):
            raise PipelineError(f"stage {stage_id} has an invalid verification command")

    branch = str(raw.get("branch", "")).strip()
    if branch and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,119}", branch):
        raise PipelineError("branch contains unsupported characters")
    raw["run_id"] = run_id
    raw["repository"]["path"] = str(repository_path.resolve())
    raw.setdefault("max_improvement_attempts", 2)
    return raw


class GitRepository:
    def __init__(self, path: Path):
        self.path = path.resolve()

    def run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["git", *args],
            cwd=self.path,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )
        if check and result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise PipelineError(f"git {' '.join(args)} failed: {detail}")
        return result

    def validate(self) -> None:
        result = self.run("rev-parse", "--is-inside-work-tree", check=False)
        if result.returncode != 0 or result.stdout.strip() != "true":
            raise PipelineError(f"Not a Git working tree: {self.path}")

    def status(self) -> str:
        return self.run("status", "--porcelain=v1", "--untracked-files=all").stdout

    def require_clean(self) -> None:
        status = self.status()
        if status.strip():
            raise PipelineError(
                "Repository has existing changes. Commit/stash them or use a separate worktree before running."
            )

    def head(self) -> str:
        return self.run("rev-parse", "HEAD").stdout.strip()

    def current_branch(self) -> str:
        return self.run("branch", "--show-current").stdout.strip()

    def require_identity(self) -> None:
        missing: list[str] = []
        for key in ("user.name", "user.email"):
            result = self.run("config", "--get", key, check=False)
            if result.returncode != 0 or not result.stdout.strip():
                missing.append(key)
        if missing:
            raise PipelineError(
                "Git commit identity is missing: " + ", ".join(missing)
            )

    def require_position(self, expected_branch: str, expected_head: str, actor: str) -> None:
        current_branch = self.current_branch()
        current_head = self.head()
        if current_branch != expected_branch or current_head != expected_head:
            raise PipelineError(
                f"{actor} changed Git position unexpectedly: "
                f"branch {expected_branch}->{current_branch}, HEAD {expected_head}->{current_head}"
            )

    def prepare_branch(self, branch: str) -> None:
        if not branch or self.current_branch() == branch:
            return
        exists = self.run(
            "show-ref", "--verify", "--quiet", f"refs/heads/{branch}", check=False
        ).returncode == 0
        if exists:
            raise PipelineError(
                f"Branch {branch} already exists but is not checked out. Select it manually or choose a new run id."
            )
        self.run("switch", "-c", branch)

    def commit_all(self, message: str) -> tuple[str, bool]:
        self.run("add", "-A")
        changed = self.run("diff", "--cached", "--quiet", check=False).returncode != 0
        if not changed:
            return self.head(), False
        self.run("commit", "-m", message)
        return self.head(), True


class TelegramNotifier:
    def __init__(self, env_path: Path = ENV_PATH):
        values = load_env(env_path)
        self.token = values.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = values.get("TELEGRAM_CHAT_ID", "")

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, message: str) -> None:
        stream = sys.stdout
        encoding = stream.encoding or "utf-8"
        try:
            print(message, file=stream, flush=True)
        except UnicodeEncodeError:
            safe_message = message.encode(encoding, errors="backslashreplace").decode(encoding)
            print(safe_message, file=stream, flush=True)
        if not self.enabled:
            return
        data = urllib.parse.urlencode(
            {"chat_id": self.chat_id, "text": message, "disable_web_page_preview": "true"}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"https://api.telegram.org/bot{self.token}/sendMessage",
            data=data,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                if response.status != 200:
                    raise PipelineError(f"Telegram returned HTTP {response.status}")
        except Exception as exc:
            print(f"Telegram notification failed: {exc}", file=sys.stderr, flush=True)


class HermesRunner:
    def __init__(self, settings: Settings):
        self.settings = settings

    def preflight(self, require_auth: bool = True) -> list[str]:
        issues: list[str] = []
        if not self.settings.hermes_executable.is_file():
            issues.append(f"Hermes executable missing: {self.settings.hermes_executable}")
        for profile in PROFILES:
            profile_home = self.settings.hermes_home / "profiles" / profile
            if not (profile_home / "SOUL.md").is_file():
                issues.append(f"Profile missing or incomplete: {profile}")
        if not self.settings.model:
            issues.append("config/settings.json model is blank")
        auth_path = self.settings.hermes_home / "auth.json"
        if require_auth and (not auth_path.is_file() or auth_path.stat().st_size < 3):
            issues.append(
                "ChatGPT OAuth is not configured. Run scripts\\finish-auth.ps1 interactively."
            )
        return issues

    def run(
        self,
        profile: str,
        prompt: str,
        repository: Path,
        output_dir: Path,
        label: str,
        toolsets_override: str | None = None,
    ) -> str:
        if profile not in PROFILES:
            raise PipelineError(f"Unknown profile: {profile}")
        output_dir.mkdir(parents=True, exist_ok=True)
        usage_file = output_dir / f"{label}-usage.json"
        command = [
            str(self.settings.hermes_executable),
            "-p",
            profile,
            "--in",
            str(repository),
            "--provider",
            self.settings.provider,
            "--reasoning",
            self.settings.reasoning.get(profile, "high"),
            "--toolsets",
            toolsets_override or self.settings.toolsets.get(profile, "coding"),
            "--usage-file",
            str(usage_file),
        ]
        if self.settings.model:
            command.extend(["--model", self.settings.model])
        command.extend(["--oneshot", prompt])
        environment = os.environ.copy()
        environment["HERMES_HOME"] = str(self.settings.hermes_home)
        result = subprocess.run(
            command,
            cwd=repository,
            env=environment,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=self.settings.timeout_seconds,
        )
        (output_dir / f"{label}-stdout.txt").write_text(result.stdout, encoding="utf-8")
        (output_dir / f"{label}-stderr.txt").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[-2000:]
            raise PipelineError(f"Hermes {profile} run failed: {detail}")
        return result.stdout.strip()


def extract_review_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "verdict" in value and "findings" in value:
            verdict = value["verdict"]
            findings = value["findings"]
            if verdict not in {"approved", "changes_requested"}:
                raise PipelineError(f"Reviewer returned invalid verdict: {verdict}")
            if not isinstance(findings, list):
                raise PipelineError("Reviewer findings must be a list")
            return value
    raise PipelineError("Reviewer did not return the required JSON object")


def run_verification(
    commands: list[str], repository: Path, output_dir: Path, timeout_seconds: int
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    success = True
    for index, command in enumerate(commands, start=1):
        started_at = utc_now()
        try:
            result = subprocess.run(
                command,
                cwd=repository,
                shell=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=timeout_seconds,
            )
            record = {
                "command": command,
                "started_at": started_at,
                "finished_at": utc_now(),
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired as exc:
            record = {
                "command": command,
                "started_at": started_at,
                "finished_at": utc_now(),
                "exit_code": -1,
                "stdout": exc.stdout or "",
                "stderr": f"Timed out after {timeout_seconds} seconds",
            }
        results.append(record)
        write_json(output_dir / f"verification-{index:02d}.json", record)
        if record["exit_code"] != 0:
            success = False
            break
    summary = {"success": success, "commands": results, "finished_at": utc_now()}
    write_json(output_dir / "verification.json", summary)
    return summary


@contextmanager
def repository_lock(repository: Path, run_id: str) -> Iterator[None]:
    LOCKS_ROOT.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(str(repository.resolve()).lower().encode("utf-8")).hexdigest()[:20]
    lock_path = LOCKS_ROOT / f"{digest}.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        owner = lock_path.read_text(encoding="utf-8", errors="replace").strip()
        raise PipelineError(f"Repository is already locked: {owner or lock_path}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"run_id": run_id, "pid": os.getpid(), "created_at": utc_now()}))
        yield
    finally:
        lock_path.unlink(missing_ok=True)


class Coordinator:
    def __init__(self, settings: Settings, notifier: Any | None = None, cancel_check=None):
        self.settings = settings
        self.hermes = HermesRunner(settings)
        self.notifier = notifier or TelegramNotifier()
        self.cancel_check = cancel_check or (lambda: False)

    def _raise_if_cancelled(self) -> None:
        if self.cancel_check():
            raise PipelineCancelled("Safe stop requested from Telegram")

    def _save_state(self, run_dir: Path, state: dict[str, Any]) -> None:
        state["updated_at"] = utc_now()
        write_json(run_dir / "state.json", state)

    def run(self, plan_path: Path) -> Path:
        plan = validate_plan(read_json(plan_path))
        issues = self.hermes.preflight(require_auth=True)
        if issues:
            raise PipelineError("Preflight failed:\n- " + "\n- ".join(issues))

        run_id = plan["run_id"]
        repository_path = Path(plan["repository"]["path"])
        repository = GitRepository(repository_path)
        repository.validate()
        run_dir = ARTIFACTS_ROOT / run_id
        if (run_dir / "state.json").exists():
            raise PipelineError(
                f"Run {run_id} already exists. Use a new run_id; existing runs are never overwritten."
            )
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(run_dir / "plan.json", plan)
        state: dict[str, Any] = {
            "run_id": run_id,
            "status": "STARTING",
            "repository": str(repository_path),
            "started_at": utc_now(),
            "stages": [],
        }
        self._save_state(run_dir, state)

        with repository_lock(repository_path, run_id):
            try:
                if plan["repository"].get("require_clean", True):
                    repository.require_clean()
                repository.require_identity()
                repository.prepare_branch(str(plan.get("branch", "")))
                state["starting_sha"] = repository.head()
                state["branch"] = repository.current_branch()
                state["status"] = "RUNNING"
                self._save_state(run_dir, state)
                self.notifier.send(f"🚀 {run_id} 시작 · {repository_path}")

                for position, stage in enumerate(plan["stages"], start=1):
                    self._raise_if_cancelled()
                    self._run_stage(plan, stage, position, repository, run_dir, state)
                    self._raise_if_cancelled()

                state["status"] = "COMPLETED"
                state["final_sha"] = repository.head()
                state["finished_at"] = utc_now()
                self._save_state(run_dir, state)
                self.notifier.send(f"🏁 {run_id} 완료 · {state['final_sha'][:12]}")
                return run_dir
            except PipelineCancelled as exc:
                state["status"] = "CANCELLED"
                state["error"] = str(exc)
                state["finished_at"] = utc_now()
                self._save_state(run_dir, state)
                self.notifier.send(f"⏹ {run_id} 안전 중단 · {exc}")
                raise
            except Exception as exc:
                state["status"] = "FAILED"
                state["error"] = str(exc)
                state["finished_at"] = utc_now()
                self._save_state(run_dir, state)
                self.notifier.send(f"❌ {run_id} 중단 · {exc}")
                raise

    def _run_stage(
        self,
        plan: dict[str, Any],
        stage: dict[str, Any],
        position: int,
        repository: GitRepository,
        run_dir: Path,
        state: dict[str, Any],
    ) -> None:
        stage_id = stage["id"]
        stage_dir = run_dir / "stages" / f"{position:03d}-{stage_id}"
        stage_dir.mkdir(parents=True, exist_ok=False)
        base_sha = repository.head()
        commands = stage.get("verification_commands", plan.get("verification_commands", []))
        contract = {
            **stage,
            "run_id": plan["run_id"],
            "repository": str(repository.path),
            "base_sha": base_sha,
            "verification_commands": commands,
        }
        contract_path = stage_dir / "contract.json"
        write_json(contract_path, contract)
        stage_state: dict[str, Any] = {
            "id": stage_id,
            "status": "DEVELOPING",
            "base_sha": base_sha,
            "started_at": utc_now(),
        }
        state["stages"].append(stage_state)
        self._save_state(run_dir, state)
        self.notifier.send(f"👨‍💻 {stage_id} 개발 시작")

        stage_branch = repository.current_branch()

        developer_prompt = f"""
Work on exactly one approved stage in the current Git repository.
Read the stage contract at: {contract_path}
Inspect repository instructions and relevant code, implement the objective within scope, and run useful focused checks.
Do not switch branches, commit, push, merge, or modify files outside the allowed scope.
Do not claim success without evidence. Finish with a concise summary of changed files and checks run.
""".strip()
        self.hermes.run("developer", developer_prompt, repository.path, stage_dir, "developer")
        repository.require_position(stage_branch, base_sha, "Developer")
        candidate_sha, changed = repository.commit_all(f"agent({stage_id}): implementation")
        stage_state.update(
            {"status": "REVIEWING", "candidate_sha": candidate_sha, "developer_changed": changed}
        )
        self._save_state(run_dir, state)
        self.notifier.send(f"🔍 {stage_id} 리뷰 시작 · {candidate_sha[:12]}")

        clean_before_review = repository.status()
        review_head = repository.head()
        reviewer_prompt = f"""
Perform an independent review only. Do not modify files, Git state, branches, commits, or configuration.
Stage contract: {contract_path}
Review the exact diff from base {base_sha} to candidate {candidate_sha}. Inspect relevant callers and tests.
Return only one JSON object with this shape:
{{"verdict":"approved|changes_requested","summary":"...","findings":[{{"id":"R-001","severity":"critical|high|medium|low","file":"path","line":1,"evidence":"...","required_change":"..."}}]}}
Use an empty findings array when approved. Report correctness, regression, security, and missing-test issues; omit subjective style comments.
""".strip()
        review_text = self.hermes.run(
            "reviewer", reviewer_prompt, repository.path, stage_dir, "reviewer"
        )
        repository.require_position(stage_branch, review_head, "Reviewer")
        if repository.status() != clean_before_review:
            raise PipelineError(
                f"Reviewer modified the repository during {stage_id}; pipeline stopped for inspection."
            )
        review = extract_review_json(review_text)
        review_path = stage_dir / "review.json"
        write_json(review_path, review)
        stage_state["review_verdict"] = review["verdict"]
        stage_state["review_findings"] = len(review["findings"])
        stage_state["status"] = "IMPROVING"
        self._save_state(run_dir, state)
        self.notifier.send(f"🛠 {stage_id} 보완 시작 · 발견 {len(review['findings'])}건")

        max_attempts = int(plan.get("max_improvement_attempts", 2))
        verification: dict[str, Any] | None = None
        final_sha = candidate_sha
        for attempt in range(1, max_attempts + 1):
            attempt_dir = stage_dir / f"improvement-{attempt:02d}"
            failure_note = ""
            if verification is not None:
                failure_note = (
                    f"\nThe previous deterministic verification failed. Read: "
                    f"{stage_dir / 'verification.json'} and fix the demonstrated failure."
                )
            improver_prompt = f"""
Perform the mandatory improvement pass for this stage.
Stage contract: {contract_path}
Independent review: {review_path}
Address every supported finding, make any clearly necessary bounded correction, and run focused checks.
If the review is approved and no correction is needed, leave source unchanged and report no_changes_needed.
Do not switch branches, commit, push, merge, expand scope, or hide a failing check.{failure_note}
Finish with a concise mapping from each finding id to fixed, rejected-with-evidence, or needs-human.
""".strip()
            improvement_head = repository.head()
            self.hermes.run(
                "improver", improver_prompt, repository.path, attempt_dir, "improver"
            )
            repository.require_position(stage_branch, improvement_head, "Improver")
            final_sha, _ = repository.commit_all(f"agent({stage_id}): improvement {attempt}")
            stage_state["status"] = "VERIFYING"
            stage_state["improvement_attempt"] = attempt
            stage_state["final_candidate_sha"] = final_sha
            self._save_state(run_dir, state)
            verification_head = repository.head()
            verification = run_verification(
                commands, repository.path, stage_dir, self.settings.timeout_seconds
            )
            repository.require_position(stage_branch, verification_head, "Verification command")
            if repository.status().strip():
                raise PipelineError(
                    f"Verification commands modified tracked or untracked repository files during {stage_id}"
                )
            if verification["success"]:
                break
            stage_state["status"] = "IMPROVING"
            self._save_state(run_dir, state)

        if verification is None or not verification["success"]:
            raise PipelineError(
                f"Deterministic verification failed for {stage_id} after {max_attempts} attempt(s)"
            )
        stage_state.update(
            {
                "status": "COMPLETED",
                "final_sha": final_sha,
                "verification_passed": True,
                "finished_at": utc_now(),
            }
        )
        self._save_state(run_dir, state)
        self.notifier.send(f"✅ {stage_id} 완료 · {final_sha[:12]}")


def command_preflight(settings: Settings, allow_missing_auth: bool) -> int:
    runner = HermesRunner(settings)
    issues = runner.preflight(require_auth=not allow_missing_auth)
    result = subprocess.run(
        [str(settings.hermes_executable), "--version"],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        env={**os.environ, "HERMES_HOME": str(settings.hermes_home)},
    )
    if result.returncode != 0:
        issues.append((result.stderr or result.stdout).strip())
    if issues:
        print("Preflight failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print(result.stdout.strip())
    print("Profiles: developer, reviewer, improver")
    print("Preflight passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Three-role Hermes development pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--allow-missing-auth", action="store_true")

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("plan", type=Path)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("plan", type=Path)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("run_id")

    subparsers.add_parser("notify-test")
    args = parser.parse_args(argv)

    try:
        settings = Settings.load()
        if args.command == "preflight":
            return command_preflight(settings, args.allow_missing_auth)
        if args.command == "validate":
            plan = validate_plan(read_json(args.plan))
            repository = GitRepository(Path(plan["repository"]["path"]))
            repository.validate()
            print(f"Valid plan: {plan['run_id']} ({len(plan['stages'])} stages)")
            return 0
        if args.command == "run":
            run_dir = Coordinator(settings).run(args.plan)
            print(f"Artifacts: {run_dir}")
            return 0
        if args.command == "status":
            run_id = safe_id(args.run_id, "run_id")
            print(json.dumps(read_json(ARTIFACTS_ROOT / run_id / "state.json"), ensure_ascii=False, indent=2))
            return 0
        if args.command == "notify-test":
            notifier = TelegramNotifier()
            if not notifier.enabled:
                raise PipelineError("Telegram token/chat id are blank")
            notifier.send("✅ AI Agents Telegram monitoring test")
            return 0
    except (PipelineError, KeyError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
