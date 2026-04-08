"""In-memory session store for serve."""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from motus.models import ChatMessage
from motus.serve.interrupt import InterruptMessage

from .schemas import SessionStatus

logger = logging.getLogger("motus.serve")


class SessionLimitReached(Exception):
    """The server's max_sessions limit has been hit."""


class SessionAlreadyExists(Exception):
    """A session with the given ID already exists."""


@dataclass
class Session:
    session_id: str
    status: SessionStatus = SessionStatus.idle
    state: list[ChatMessage] = field(default_factory=list)
    response: ChatMessage | None = None
    error: str | None = None
    last_message_at: float = field(default_factory=time.monotonic)
    running_since: float | None = None
    _task: asyncio.Task | None = field(default=None, repr=False)
    _done: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    pending_interrupts: dict = field(default_factory=dict, repr=False)
    _resume_queue: "asyncio.Queue | None" = field(default=None, repr=False)

    def start_turn(self, task: asyncio.Task) -> None:
        """Transition to running state for a new turn."""
        self._task = task
        self.status = SessionStatus.running
        self.running_since = time.monotonic()
        self.response = None
        self.error = None
        self._done.clear()

    def complete_turn(
        self, response: ChatMessage, new_state: list[ChatMessage]
    ) -> None:
        """Record a successful turn result and transition to idle."""
        self.response = response
        self.state = new_state
        self.status = SessionStatus.idle
        self.running_since = None
        self.error = None
        self.last_message_at = time.monotonic()
        self._done.set()
        self.pending_interrupts.clear()

    def fail_turn(self, error: str) -> None:
        """Record a turn failure and transition to error state."""
        self.response = None
        self.status = SessionStatus.error
        self.running_since = None
        self.error = error
        self.last_message_at = time.monotonic()
        self._done.set()
        self.pending_interrupts.clear()

    def cancel(self) -> None:
        """Cancel any running task and signal waiters."""
        if self._task is not None:
            self._task.cancel()
        self._done.set()
        self.pending_interrupts.clear()

    async def wait(self, timeout: float | None = None) -> None:
        """Wait for the current turn to complete."""
        if self.status != SessionStatus.running:
            return
        if timeout is not None:
            try:
                await asyncio.wait_for(self._done.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass
        else:
            await self._done.wait()

    def interrupt_turn(self, msg: "InterruptMessage") -> None:
        """Transition to interrupted. Drops late deliveries after terminal states."""
        if self.status not in (SessionStatus.running, SessionStatus.interrupted):
            return
        self.status = SessionStatus.interrupted
        self.pending_interrupts[msg.interrupt_id] = msg
        self._done.set()  # wake long-poll waiters

    def submit_resume(self, interrupt_id: str, value: Any) -> None:
        """Forward user's reply to the worker. Raises ValueError on bad state."""
        if self._resume_queue is None:
            raise ValueError("Session not actively waiting for resume")
        if interrupt_id not in self.pending_interrupts:
            raise ValueError(f"Unknown interrupt_id: {interrupt_id}")

        from motus.serve.interrupt import ResumeMessage

        del self.pending_interrupts[interrupt_id]
        self._resume_queue.put_nowait(ResumeMessage(interrupt_id, value))
        if not self.pending_interrupts:
            self.status = SessionStatus.running
            self._done.clear()


class SessionStore:
    """In-memory session store with optional TTL-based expiry."""

    def __init__(self, *, ttl: float = 0, max_sessions: int = 0):
        self._sessions: dict[str, Session] = {}
        self.ttl = ttl
        self.max_sessions = max_sessions

    def __len__(self) -> int:
        return len(self._sessions)

    def create(
        self, state: list[ChatMessage] | None = None, session_id: str | None = None
    ) -> Session:
        if session_id is not None and session_id in self._sessions:
            raise SessionAlreadyExists(f"Session {session_id} already exists")
        if self.max_sessions > 0 and len(self._sessions) >= self.max_sessions:
            raise SessionLimitReached(
                f"Maximum number of sessions ({self.max_sessions}) reached"
            )
        if session_id is None:
            session_id = str(uuid.uuid4())
        session = Session(session_id=session_id, state=state or [])
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def list(self) -> list[Session]:
        return list(self._sessions.values())

    def delete(self, session_id: str) -> bool:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        session.cancel()
        return True

    def sweep(self) -> int:
        """Delete expired idle/error sessions. Skip active sessions (running, interrupted)."""
        if self.ttl <= 0:
            return 0
        now = time.monotonic()
        expired = [
            sid
            for sid, s in self._sessions.items()
            if s.status not in (SessionStatus.running, SessionStatus.interrupted)
            and now - s.last_message_at > self.ttl
        ]
        for sid in expired:
            self._sessions[sid].cancel()
            del self._sessions[sid]
        if expired:
            logger.info(f"Swept {len(expired)} expired session(s)")
        return len(expired)
