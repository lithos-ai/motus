"""Typed response models returned by motus.cloud public APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from motus.serve.schemas import InterruptInfo, SessionResponse, SessionStatus

from .errors import AmbiguousInterrupt

if TYPE_CHECKING:
    from motus.models import ChatMessage


@dataclass(frozen=True)
class Interrupt:
    """A single pending interrupt surfaced on ChatResult.interrupts."""

    id: str
    type: str
    payload: dict

    @classmethod
    def from_info(cls, info: "InterruptInfo") -> "Interrupt":
        return cls(id=info.interrupt_id, type=info.type, payload=dict(info.payload))


@dataclass
class ChatResult:
    """Return value of Client.chat / Session.chat / Client.resume."""

    message: "ChatMessage | None"
    interrupts: list[Interrupt] = field(default_factory=list)
    session_id: str = ""
    status: SessionStatus = SessionStatus.idle
    snapshot: SessionResponse | None = None
    _client: Any = field(default=None, repr=False, compare=False)

    def resume(self, value: Any) -> "ChatResult":
        """Resolve the single pending interrupt with ``value`` and return the next ChatResult.

        Raises AmbiguousInterrupt if zero or more than one interrupt is pending;
        use the parent client's ``resume(session_id, interrupt_id, value)`` in that case.
        """
        if len(self.interrupts) != 1:
            raise AmbiguousInterrupt(
                f"result.resume(value) requires exactly 1 pending interrupt, "
                f"got {len(self.interrupts)}"
            )
        if self._client is None:
            raise AmbiguousInterrupt(
                "ChatResult is detached from its client; use client.resume(...) directly"
            )
        return self._client.resume(self.session_id, self.interrupts[0].id, value)


@dataclass(frozen=True)
class SessionEvent:
    """Coarse session-status event emitted by ``chat_events()``.

    Not token streaming — the server only exposes session-state transitions.
    ``type`` is one of ``"running"``, ``"idle"``, ``"interrupted"``, ``"error"``.
    """

    type: str
    session_id: str
    snapshot: SessionResponse | None = None


__all__ = ["ChatResult", "Interrupt", "SessionEvent"]
