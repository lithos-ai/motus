"""In-memory session store for serve."""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from motus.models import ChatMessage
from motus.serve.interrupt import InterruptMessage

from .schemas import InterruptInfo, SessionResponse, SessionStatus

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
    _event_log: list = field(default_factory=list, repr=False)
    _event_epoch: int = field(default=0, repr=False)
    _event_condition: asyncio.Condition = field(
        default_factory=asyncio.Condition, repr=False
    )
    _pending_publishes: set = field(default_factory=set, repr=False)

    def to_response(self) -> SessionResponse:
        """Snapshot the current session as a SessionResponse.

        Used by GET /sessions/{id} and as the base shape for SSE events,
        so terminal SSE frames are a strict superset of the GET body.
        """
        interrupts = None
        if self.status == SessionStatus.interrupted:
            interrupts = [
                InterruptInfo(
                    interrupt_id=iid,
                    type=msg.payload.get("type", "unknown"),
                    payload=msg.payload,
                )
                for iid, msg in self.pending_interrupts.items()
            ]

        return SessionResponse(
            session_id=self.session_id,
            status=self.status,
            response=self.response,
            error=self.error,
            interrupts=interrupts,
        )

    async def publish_event(self, event_name: str, **extras) -> None:
        """Publish an SSE event with standard {session_id, status, ...} envelope.

        Every SSE event carries session_id and status (auto-included via
        to_response()) plus any state-derived fields populated for the
        current status. Per-event extras (e.g. ``message=...``) layer on top.

        The special event name ``"closed"`` is an internal end-of-stream
        signal: the SSE generator detects it and disconnects subscribers
        without forwarding the frame on the wire.

        A ``"message"`` event may carry an optional ``agent_path`` field, a
        list of registered subagent (``AgentTool``) names identifying the
        nested call chain that produced the message. The field is omitted
        for messages from the root agent. Note that the SSE stream is a
        **superset** of the final returned ``list[ChatMessage]``: subagent
        messages (those with a non-empty ``agent_path``) appear on the
        stream but are NOT included in ``done.response`` or in the history
        returned by ``GET /sessions/{id}/messages`` — those reflect only the
        parent's view of the conversation (tool call + tool result string).
        """
        payload = {
            "event": event_name,
            **self.to_response().model_dump(exclude_none=True),
        }
        payload.update({k: v for k, v in extras.items() if v is not None})
        async with self._event_condition:
            self._event_log.append(payload)
            self._event_condition.notify_all()

    def publish_event_soon(self, event_name: str, **extras) -> None:
        """Fire-and-forget version of :meth:`publish_event` for sync callers
        (e.g. interrupt/message callbacks invoked via ``call_soon_threadsafe``,
        or ``cancel()`` from a sync test path).

        No-op if there is no running event loop. Otherwise schedules the
        publish and holds a strong reference to the task until it completes
        so it isn't GC'd mid-flight (asyncio only weakly references tasks).
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        task = asyncio.ensure_future(self.publish_event(event_name, **extras))
        self._pending_publishes.add(task)
        task.add_done_callback(self._pending_publishes.discard)

    def start_turn(self, task: asyncio.Task) -> None:
        """Transition to running state for a new turn."""
        self._task = task
        self.status = SessionStatus.running
        self.running_since = time.monotonic()
        self.response = None
        self.error = None
        self._done.clear()
        self._event_log.clear()
        self._event_epoch += 1

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
        # Tells SSE subscribers the session is gone and they should disconnect.
        # No-op when there is no running loop (e.g. sync test path).
        self.publish_event_soon("closed")

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
