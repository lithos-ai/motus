"""Programmatic Python client for Motus agents (local serve and Motus Cloud)."""

from ._models import ChatResult, Interrupt, SessionEvent
from .errors import (
    AgentError,
    AmbiguousInterrupt,
    AuthError,
    BackendUnavailable,
    InterruptNotFound,
    MotusClientError,
    ProtocolError,
    ServerBusy,
    SessionClosed,
    SessionConflict,
    SessionNotFound,
    SessionTimeout,
    SessionUnsupported,
)

__all__ = [
    "AgentError",
    "AmbiguousInterrupt",
    "AuthError",
    "BackendUnavailable",
    "ChatResult",
    "InterruptNotFound",
    "Interrupt",
    "MotusClientError",
    "ProtocolError",
    "ServerBusy",
    "SessionClosed",
    "SessionConflict",
    "SessionEvent",
    "SessionNotFound",
    "SessionTimeout",
    "SessionUnsupported",
]
