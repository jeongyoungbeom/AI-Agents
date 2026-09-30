"""텔레그램용 첫 번째 대화 어댑터."""

from .adapter import TelegramAdapter
from .client import TelegramAPIError, TelegramClient

__all__ = ["TelegramAPIError", "TelegramAdapter", "TelegramClient"]
