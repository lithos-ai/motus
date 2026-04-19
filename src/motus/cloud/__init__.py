"""Programmatic Python client for Motus agents (local serve and Motus Cloud)."""

from ._models import ChatResult, Interrupt, SessionEvent
from .client import Client
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
from .session import Session

__all__ = [
    "AgentError",
    "AmbiguousInterrupt",
    "AuthError",
    "BackendUnavailable",
    "ChatResult",
    "Client",
    "InterruptNotFound",
    "Interrupt",
    "MotusClientError",
    "ProtocolError",
    "ServerBusy",
    "SessionClosed",
    "SessionConflict",
    "Session",
    "SessionEvent",
    "SessionNotFound",
    "SessionTimeout",
    "SessionUnsupported",
]
