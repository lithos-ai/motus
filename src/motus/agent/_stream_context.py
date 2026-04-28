"""Context-variable plumbing for subagent message attribution.

When an agent runs under the serve harness, the worker installs a forwarding
``on_message`` callback on the root agent. To make subagent (``AgentTool``)
activity visible on the same SSE stream, we propagate two pieces of ambient
state via :mod:`contextvars`:

- :data:`_stream_callback` holds the single forwarding callback. ``AgentTool``
  reads it to install on each child agent, so the same closure fires at every
  depth in the agent tree.
- :data:`_agent_path` accumulates the registered tool names along the call
  path (e.g. ``("researcher", "summarizer")``). The forwarding callback reads
  it at emit time and tags the outgoing event with the path.

Both default to "no streaming" (``None`` / empty tuple), so non-serve usage is
unaffected. Contextvars copy correctly across ``await``,
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
