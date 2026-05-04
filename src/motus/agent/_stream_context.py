"""Context-variable plumbing for subagent message attribution.

When an agent runs under the serve harness, the worker installs a forwarding
callback in :data:`_stream_callback`. To make subagent (``AgentTool``) activity
visible on the same SSE stream, we propagate three pieces of ambient state via
:mod:`contextvars`:

- :data:`_stream_callback` holds the single forwarding callback. Every
  ``AgentBase.add_message`` call reads it to emit the new message, so the same
  closure fires at every depth in the agent tree.
- :data:`_agent_path` accumulates the registered tool names along the call
  path (e.g. ``("researcher", "summarizer")``). The forwarding callback reads
  it at emit time and tags the outgoing event with the path.
- :data:`_caller_tagged` is set ``True`` by ``AgentTool`` and by
  ``AgentBase.run_turn`` for the duration of the next ``__call__``,
  signalling the immediate child's ``_execute`` to skip pushing its own
  name onto ``_agent_path``. ``_execute`` resets it to ``False`` for the
  body of ``_run``, so descendants — e.g. a free-standing agent created
  inside a tool body — self-tag normally.

All three default to "no streaming" (``None`` / empty tuple / ``False``), so
non-serve usage is unaffected. Contextvars copy correctly across ``await``,
``asyncio.create_task``, and ``asyncio.to_thread``, so parallel subagent calls
get independent paths without bleeding into siblings.
"""

from collections.abc import Awaitable, Callable
from contextvars import ContextVar

from motus.models import ChatMessage

_stream_callback: ContextVar[Callable[[ChatMessage], Awaitable[None]] | None] = (
    ContextVar("motus_stream_callback", default=None)
)

_agent_path: ContextVar[tuple[str, ...]] = ContextVar(
    "motus_agent_path", default=()
)

_caller_tagged: ContextVar[bool] = ContextVar(
    "motus_caller_tagged", default=False
)
