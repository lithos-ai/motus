"""Motus runtime — tracing decorators and task-type constants.

The custom runtime abstraction (``AgentRuntime``, ``AgentFuture``, ``hooks``,
``agent_task``, ``TaskPolicy``) has been removed. Tasks are now plain async
functions; use the tracing decorators below for observability. Retries and
timeouts are handled at call sites (e.g. tenacity / ``asyncio.timeout()``).
"""

from ..tracing import (
    export_trace,
    set_session_id,
    setup_tracing,
    shutdown_tracing,
    traced,
    traced_agent_call,
    traced_model_call,
    traced_tool_call,
)
from .types import AGENT_CALL, MODEL_CALL, TASK, TOOL_CALL

__all__ = [
    "AGENT_CALL",
    "MODEL_CALL",
    "TASK",
    "TOOL_CALL",
    "export_trace",
    "set_session_id",
    "setup_tracing",
    "shutdown_tracing",
    "traced",
    "traced_agent_call",
    "traced_model_call",
    "traced_tool_call",
]
