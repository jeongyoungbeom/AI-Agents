from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TelegramSettings:
    token: str
    allowed_users: frozenset[str]
    allowed_chats: frozenset[str]
    private_only: bool
    poll_timeout_seconds: int
    max_message_characters: int

    def validate(self) -> None:
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN이 비어 있습니다.")
        if not self.allowed_users:
            raise ValueError("TELEGRAM_ALLOWED_USERS가 비어 있습니다.")


@dataclass(frozen=True)
class ConversationSettings:
    max_auto_agent_replies: int = 4
    group_parallel_workers: int = 3
    worker_count: int = 2
    worker_poll_seconds: float = 0.2
    lease_seconds: int = 120
    max_job_attempts: int = 2

    def validate(self) -> None:
        if not 1 <= self.max_auto_agent_replies <= 4:
            raise ValueError("max_auto_agent_replies는 1~4여야 합니다.")
        if not 1 <= self.group_parallel_workers <= 3:
            raise ValueError("group_parallel_workers는 1~3이어야 합니다.")
        if not 1 <= self.worker_count <= 8:
            raise ValueError("conversation worker_count는 1~8이어야 합니다.")
        if self.worker_poll_seconds <= 0 or self.lease_seconds < 10:
            raise ValueError("대화 워커 시간 설정을 확인해 주세요.")
        if self.max_job_attempts < 1:
            raise ValueError("conversation max_job_attempts는 1 이상이어야 합니다.")


@dataclass(frozen=True)
class RepositoryAnalysisSettings:
    worker_count: int = 1
    worker_poll_seconds: float = 0.5
    lease_seconds: int = 120
    max_files_per_batch: int = 3
    max_file_bytes: int = 48 * 1024
    max_batch_bytes: int = 96 * 1024
    max_batches: int = 40
    max_query_rounds: int = 80
    max_read_bytes: int = 4 * 1024 * 1024
    max_no_progress: int = 2
    max_elapsed_seconds: int = 60 * 60

    def validate(self) -> None:
        if not 1 <= self.worker_count <= 2:
            raise ValueError("repository analysis worker_count는 1~2여야 합니다.")
        if self.worker_poll_seconds <= 0 or self.lease_seconds < 10:
            raise ValueError("repository analysis 워커 시간 설정을 확인해 주세요.")
        if not 1 <= self.max_files_per_batch <= 8:
            raise ValueError("repository analysis max_files_per_batch는 1~8이어야 합니다.")
        if not 1_024 <= self.max_file_bytes <= 512 * 1024:
            raise ValueError("repository analysis max_file_bytes 범위를 확인해 주세요.")
        if self.max_batch_bytes < self.max_file_bytes:
            raise ValueError("repository analysis max_batch_bytes는 파일 한도 이상이어야 합니다.")
        if self.max_batches < 2 or self.max_query_rounds < 1:
            raise ValueError("repository analysis 반복 한도를 확인해 주세요.")
        if self.max_read_bytes < self.max_batch_bytes or self.max_no_progress < 1:
            raise ValueError("repository analysis 조회 한도를 확인해 주세요.")
        if not 60 <= self.max_elapsed_seconds <= 24 * 60 * 60:
            raise ValueError("repository analysis max_elapsed_seconds 범위를 확인해 주세요.")


@dataclass(frozen=True)
class AttachmentSettings:
    max_bytes: int = 8 * 1024 * 1024
    max_text_characters: int = 24_000

    def validate(self) -> None:
        if not 1_024 <= self.max_bytes <= 20 * 1024 * 1024:
            raise ValueError("attachment max_bytes는 1KB~20MB여야 합니다.")
        if not 1_000 <= self.max_text_characters <= 100_000:
            raise ValueError("attachment max_text_characters는 1,000~100,000이어야 합니다.")


@dataclass(frozen=True)
class LoggingSettings:
    gateway_max_bytes: int = 5 * 1024 * 1024
    gateway_backup_count: int = 14

    def validate(self) -> None:
        if self.gateway_max_bytes < 1024:
            raise ValueError("gateway_max_bytes는 1024 이상이어야 합니다.")
        if not 1 <= self.gateway_backup_count <= 90:
            raise ValueError("gateway_backup_count는 1~90이어야 합니다.")


@dataclass(frozen=True)
class GatewaySettings:
    telegram: TelegramSettings
    conversation: ConversationSettings = ConversationSettings()
    repository_analysis: RepositoryAnalysisSettings = RepositoryAnalysisSettings()
    attachments: AttachmentSettings = AttachmentSettings()
    logging: LoggingSettings = LoggingSettings()
    max_processing_attempts: int = 2
    connection_retry_limit: int = 8
    retry_max_seconds: float = 60.0

    @classmethod
    def load(cls, root: Path) -> "GatewaySettings":
        root = root.resolve()
        raw = _read_json(root / "config" / "channels.json")
        telegram = dict(raw.get("telegram", {}))
        gateway = dict(raw.get("gateway", {}))
        conversation = dict(raw.get("conversation", {}))
        repository_analysis = dict(raw.get("repository_analysis", {}))
        attachments = dict(raw.get("attachments", {}))
        logging = dict(raw.get("logging", {}))
        env = _read_env(root / "config" / "secrets.env")
        for key in (
            "TELEGRAM_BOT_TOKEN",
            "TELEGRAM_ALLOWED_USERS",
            "TELEGRAM_ALLOWED_CHATS",
        ):
            if key in os.environ:
                env[key] = os.environ[key]
        return cls(
            telegram=TelegramSettings(
                token=env.get("TELEGRAM_BOT_TOKEN", "").strip(),
                allowed_users=_split_values(env.get("TELEGRAM_ALLOWED_USERS", "")),
                allowed_chats=_split_values(env.get("TELEGRAM_ALLOWED_CHATS", "")),
                private_only=bool(telegram.get("private_only", True)),
                poll_timeout_seconds=int(telegram.get("poll_timeout_seconds", 25)),
                max_message_characters=int(
                    telegram.get("max_message_characters", 3900)
                ),
            ),
            conversation=ConversationSettings(
                max_auto_agent_replies=int(
                    conversation.get("max_auto_agent_replies", 4)
                ),
                group_parallel_workers=int(
                    conversation.get("group_parallel_workers", 3)
                ),
                worker_count=int(conversation.get("worker_count", 2)),
                worker_poll_seconds=float(
                    conversation.get("worker_poll_seconds", 0.2)
                ),
                lease_seconds=int(conversation.get("lease_seconds", 120)),
                max_job_attempts=int(conversation.get("max_job_attempts", 2)),
            ),
            repository_analysis=RepositoryAnalysisSettings(
                worker_count=int(repository_analysis.get("worker_count", 1)),
                worker_poll_seconds=float(repository_analysis.get("worker_poll_seconds", 0.5)),
                lease_seconds=int(repository_analysis.get("lease_seconds", 120)),
                max_files_per_batch=int(repository_analysis.get("max_files_per_batch", 3)),
                max_file_bytes=int(repository_analysis.get("max_file_bytes", 48 * 1024)),
                max_batch_bytes=int(repository_analysis.get("max_batch_bytes", 96 * 1024)),
                max_batches=int(repository_analysis.get("max_batches", 40)),
                max_query_rounds=int(repository_analysis.get("max_query_rounds", 80)),
                max_read_bytes=int(repository_analysis.get("max_read_bytes", 4 * 1024 * 1024)),
                max_no_progress=int(repository_analysis.get("max_no_progress", 2)),
                max_elapsed_seconds=int(repository_analysis.get("max_elapsed_seconds", 60 * 60)),
            ),
            attachments=AttachmentSettings(
                max_bytes=int(attachments.get("max_bytes", 8 * 1024 * 1024)),
                max_text_characters=int(
                    attachments.get("max_text_characters", 24_000)
                ),
            ),
            logging=LoggingSettings(
                gateway_max_bytes=int(logging.get("gateway_max_bytes", 5 * 1024 * 1024)),
                gateway_backup_count=int(logging.get("gateway_backup_count", 14)),
            ),
            max_processing_attempts=int(
                gateway.get("max_processing_attempts", 2)
            ),
            connection_retry_limit=int(gateway.get("connection_retry_limit", 8)),
            retry_max_seconds=float(gateway.get("retry_max_seconds", 60)),
        )

    def validate(self) -> None:
        self.telegram.validate()
        self.conversation.validate()
        self.repository_analysis.validate()
        self.attachments.validate()
        self.logging.validate()
        if self.max_processing_attempts < 1:
            raise ValueError("max_processing_attempts는 1 이상이어야 합니다.")
        if self.connection_retry_limit < 1 or self.retry_max_seconds <= 0:
            raise ValueError("게이트웨이 연결 재시도 설정은 양수여야 합니다.")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"설정 파일이 없습니다: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"설정 JSON이 올바르지 않습니다: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"설정 최상위 값은 객체여야 합니다: {path}")
    return value


def _read_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _split_values(value: str) -> frozenset[str]:
    return frozenset(item.strip() for item in value.split(",") if item.strip())
