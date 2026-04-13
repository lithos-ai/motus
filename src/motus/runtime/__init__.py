"""Motus runtime — tracing and type constants.

Tasks are plain async functions. Use tracing decorators for observability,
tenacity for retries, and asyncio.timeout() for timeouts.
"""

from .tracing import (
    export_trace,
    get_tracer,
    get_turn_metrics,
    set_session_id,
    setup_tracing,
    shutdown_tracing,
    traced,
    traced_agent_call,
    traced_model_call,
    traced_tool_call,
)
from .types import AGENT_CALL, MODEL_CALL, TOOL_CALL

__all__ = [
    "AGENT_CALL",
    "MODEL_CALL",
    "TOOL_CALL",
    "export_trace",
    "get_tracer",
    "get_turn_metrics",
    "set_session_id",
    "setup_tracing",
    "shutdown_tracing",
    "traced",
    "traced_agent_call",
    "traced_model_call",
    "traced_tool_call",
]
