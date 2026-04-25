"""Programmatic Python client for Motus agents (local serve and Motus Cloud)."""

from ._models import ChatResult, Interrupt, SessionEvent
from .async_client import AsyncClient, AsyncSession
from .client import Client
from .errors import (
    AgentError,
    AmbiguousInterrupt,
    AuthError,
    BackendUnavailable,
    BadRequest,
    ClientClosed,
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
    "AsyncClient",
    "AsyncSession",
    "AuthError",
    "BackendUnavailable",
    "BadRequest",
    "ChatResult",
    "Client",
    "ClientClosed",
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
