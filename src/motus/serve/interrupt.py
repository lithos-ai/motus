"""Interrupt/resume primitive for motus serve workers.

Provides `interrupt()` — the single entry point for pausing execution
and waiting for user input. Called from tool bodies, guardrails, etc.

Architecture: single duplex multiprocessing.Pipe per worker subprocess.
A daemon thread reads resumes, the main asyncio loop sends interrupts.
Module-level state is safe because each worker is its own subprocess.
"""

from __future__ import annotations

import asyncio
import logging
import pickle
import threading
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any
from uuid import uuid4

logger = logging.getLogger("motus.serve.interrupt")

# Defensive size limit for pickled messages (not a PIPE_BUF constraint).
MAX_MESSAGE_BYTES = 16 * 1024


@dataclass
class InterruptMessage:
    """Worker → parent: request user input."""

    interrupt_id: str
    payload: dict


@dataclass
class ResumeMessage:
    """Parent → worker: user's reply to an interrupt."""

    interrupt_id: str
    value: Any


_conn: Connection | None = None
_pending: dict[str, asyncio.Future] = {}
_loop: asyncio.AbstractEventLoop | None = None
_reader_thread: threading.Thread | None = None


async def interrupt(payload: dict) -> Any:
    """Pause execution, send payload to user, wait for reply.

    Raises RuntimeError if called outside a serve worker subprocess.
    """
    if _conn is None or _loop is None:
        raise RuntimeError("interrupt() called outside motus serve worker subprocess")

    iid = str(uuid4())
    msg = InterruptMessage(interrupt_id=iid, payload=payload)

    size = len(pickle.dumps(msg))
    if size > MAX_MESSAGE_BYTES:
        raise ValueError(
            f"Interrupt message too large: {size} bytes (max {MAX_MESSAGE_BYTES})"
        )

    future = _loop.create_future()
    _pending[iid] = future
    try:
        _conn.send(msg)
        return await future
    finally:
        _pending.pop(iid, None)


def _init_interrupt_channel(conn: Connection) -> None:
    """Initialize the interrupt channel. Must be called once inside asyncio.run."""
    global _conn, _loop, _reader_thread

    _conn = conn
    _loop = asyncio.get_running_loop()

    _reader_thread = threading.Thread(
        target=_pipe_reader_thread,
        name="motus-interrupt-reader",
        daemon=True,
    )
    _reader_thread.start()


def _pipe_reader_thread() -> None:
    """Daemon thread: recv() loop dispatching ResumeMessages to futures.

    Threading contract: this thread ONLY recv()s, main loop ONLY send()s.
    """
    assert _conn is not None
    assert _loop is not None

    while True:
        try:
            msg = _conn.recv()
        except (EOFError, OSError):
            # Pipe closed — poison pending futures so callers don't hang.
            def _poison() -> None:
                for fut in list(_pending.values()):
                    if not fut.done():
                        fut.set_exception(EOFError("Worker pipe closed"))

            try:
                if _loop is not None:
                    _loop.call_soon_threadsafe(_poison)
            except RuntimeError:
                pass  # loop already closed
            return

        if isinstance(msg, ResumeMessage):
            # Default-arg capture avoids loop-variable closure bug.
            def _set_result(iid: str = msg.interrupt_id, val: Any = msg.value) -> None:
                fut = _pending.get(iid)
                if fut is not None and not fut.done():
                    fut.set_result(val)

            try:
                if _loop is not None:
                    _loop.call_soon_threadsafe(_set_result)
            except RuntimeError:
                # Loop already closed (worker exiting). Drop the message.
                return
        else:
            logger.warning(
                "motus-interrupt-reader: unexpected message type %s",
                type(msg).__name__,
            )
