from __future__ import annotations

import re
from typing import Any


REDACTED = "[REDACTED]"


class SecretRedactor:
    """Best-effort local redaction before messages reach persistent logs."""

    _key_name = (
        r"(?:api[_-]?key|token|password|passwd|secret|authorization|"
        r"access[_-]?key|private[_-]?key)"
    )
    _quoted_key_value = re.compile(
        rf"(?i)([\"'][^\"']*{_key_name}[^\"']*[\"']\s*:\s*)"
        r"([\"'])(.*?)(\2)"
    )
    _plain_key_value = re.compile(
        rf"(?i)(\b[A-Za-z0-9_.-]*{_key_name}[A-Za-z0-9_.-]*\b"
        r"\s*[:=]\s*)([^\s,;}}\]]+)"
    )
    _credential_patterns = (
        re.compile(
            r"(?i)\b([a-z][a-z0-9+.-]*://)([^/\s:@]+):([^/\s@]+)@"
        ),
        re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*"),
        re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
        re.compile(r"\b\d{6,}:[A-Za-z0-9_-]{20,}\b"),
        re.compile(r"\b(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{20,}\b"),
        re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
        re.compile(
            r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
        ),
        re.compile(
            r"-----BEGIN [^-\r\n]*PRIVATE KEY-----.*?"
            r"-----END [^-\r\n]*PRIVATE KEY-----",
            re.DOTALL,
        ),
    )

    _sensitive_keys = {
        "api_key",
        "apikey",
        "token",
        "password",
        "passwd",
        "secret",
        "authorization",
        "telegram_bot_token",
        "access_token",
        "refresh_token",
    }

    def text(self, value: str) -> str:
        result = value
        for index, pattern in enumerate(self._credential_patterns):
            if index == 0:
                result = pattern.sub(
                    lambda match: f"{match.group(1)}{REDACTED}@", result
                )
            else:
                result = pattern.sub(REDACTED, result)
        result = self._quoted_key_value.sub(
            lambda match: (
                f"{match.group(1)}{match.group(2)}{REDACTED}{match.group(4)}"
            ),
            result,
        )
        result = self._plain_key_value.sub(
            lambda match: f"{match.group(1)}{REDACTED}",
            result,
        )
        return result

    def value(self, value: Any) -> Any:
        if isinstance(value, dict):
            cleaned: dict[str, Any] = {}
            for key, item in value.items():
                if str(key).lower() in self._sensitive_keys:
                    cleaned[str(key)] = REDACTED
                else:
                    cleaned[str(key)] = self.value(item)
            return cleaned
        if isinstance(value, list):
            return [self.value(item) for item in value]
        if isinstance(value, tuple):
            return [self.value(item) for item in value]
        if isinstance(value, str):
            return self.text(value)
        return value
