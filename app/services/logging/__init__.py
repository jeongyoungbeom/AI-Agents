"""Human-readable and structured audit logging."""

from .redaction import SecretRedactor

__all__ = ["AuditLogger", "GatewayLog", "SecretRedactor"]


def __getattr__(name: str):
    if name == "AuditLogger":
        from .audit import AuditLogger

        return AuditLogger
    if name == "GatewayLog":
        from .gateway import GatewayLog

        return GatewayLog
    raise AttributeError(name)
