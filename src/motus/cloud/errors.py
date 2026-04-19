"""Typed exception hierarchy for motus.cloud.

Every public method on Client / AsyncClient / Session / AsyncSession
translates transport and protocol failures into subclasses of
MotusClientError. Raw httpx exceptions never leak to user code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx

    from motus.serve.schemas import SessionResponse


class MotusClientError(Exception):
    """Base class for all motus.cloud errors."""

    def __init__(
        self,
        message: str = "",
        *,
        response: "httpx.Response | None" = None,
    ) -> None:
        super().__init__(message)
        self.response = response


class AuthError(MotusClientError):
    """HTTP 401 / 403."""


class SessionNotFound(MotusClientError):
    """HTTP 404 on a session-addressed path."""


class InterruptNotFound(MotusClientError):
    """HTTP 404 returned by POST /sessions/{id}/resume for an unknown interrupt_id."""


class SessionConflict(MotusClientError):
    """HTTP 409 — send-while-running, resume mismatch, or other session-state conflict."""


class ServerBusy(MotusClientError):
    """HTTP 503 — server capacity (max_sessions) reached."""


class BackendUnavailable(MotusClientError):
    """Other 5xx, connect error, or exhausted bounded-retry budget on polling."""


class SessionTimeout(MotusClientError):
    """Client-side turn deadline exceeded while the server session is still running.

    The server session is intentionally NOT deleted when this is raised — the caller
    can reconnect via ``client.get_session(session_id, wait=True)`` using the fields
    attached to this exception.
    """

    def __init__(
        self,
        message: str = "",
        *,
        session_id: str,
        elapsed: float | None = None,
        last_snapshot: "SessionResponse | None" = None,
    ) -> None:
        super().__init__(message)
        self.session_id = session_id
        self.elapsed = elapsed
        self.last_snapshot = last_snapshot


class AgentError(MotusClientError):
    """HTTP 200 response with session.status == 'error' (agent turn failed)."""

    def __init__(self, message: str = "", *, session_id: str | None = None) -> None:
        super().__init__(message)
        self.session_id = session_id


class ProtocolError(MotusClientError):
    """Malformed / non-JSON response body, or valid JSON missing required schema fields."""


class SessionClosed(MotusClientError):
    """Client-side: Session.chat / resume called after close() or context-exit."""


class AmbiguousInterrupt(MotusClientError):
    """ChatResult.resume(value) called when zero or multiple interrupts are pending.

    Use ``client.resume(session_id, interrupt_id, value)`` to disambiguate.
    """


class SessionUnsupported(MotusClientError):
    """HTTP 405 on PUT /sessions/{id} — server was not started with --allow-custom-ids."""


class ClientClosed(MotusClientError):
    """Client-side: a public method was called after ``close()`` / ``aclose()``."""


def attach_response(
    exc: MotusClientError, response: "httpx.Response"
) -> MotusClientError:
    """Return ``exc`` with the originating response attached (for debugging)."""
    exc.response = response
    return exc


__all__ = [
    "AgentError",
    "AmbiguousInterrupt",
    "AuthError",
    "BackendUnavailable",
    "ClientClosed",
    "InterruptNotFound",
    "MotusClientError",
    "ProtocolError",
    "ServerBusy",
    "SessionClosed",
    "SessionConflict",
    "SessionNotFound",
    "SessionTimeout",
    "SessionUnsupported",
    "attach_response",
]


# Silence unused-import warning for Any when not used in runtime;
# kept for future public API additions that need typed attributes.
_ = Any
