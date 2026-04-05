"""Anthropic SDK compatibility layer for motus.

Wraps the Anthropic Python SDK's Beta Tool Runner with motus tracing and
a serve-compatible ``ToolRunner`` class.

Usage::

    from motus.anthropic import ToolRunner, beta_async_tool

    @beta_async_tool
    async def get_weather(city: str) -> str:
        \"\"\"Get the weather for a city.\"\"\"
        return f"Sunny in {city}"

    runner = ToolRunner(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=[get_weather],
    )
    # Deploy: motus serve start my_module:runner
"""

from __future__ import annotations

import atexit
import logging
from typing import TYPE_CHECKING, Any

try:
    from anthropic import AsyncAnthropic
    from anthropic.lib.tools import (  # noqa: F401
        BetaAsyncFunctionTool,
        BetaFunctionTool,
        beta_async_tool,
        beta_tool,
    )
except ImportError as exc:
    raise ImportError(
        "motus.anthropic requires the Anthropic SDK with tool runner support. "
        "Install or upgrade with: uv pip install 'anthropic>=0.49.0'"
    ) from exc

from ._motus_runner import (  # noqa: F401
    MotusBetaAsyncStreamingToolRunner,
    MotusBetaAsyncToolRunner,
    MotusBetaStreamingToolRunner,
    MotusBetaToolRunner,
)
from ._motus_tracing import _now_us, build_agent_call_meta

if TYPE_CHECKING:
    from motus.models import ChatMessage

logger = logging.getLogger("AgentTracer")

_tracer = None


def register_tracing() -> None:
    """Create a standalone TraceManager for Anthropic SDK tracing (idempotent)."""
    global _tracer
    if _tracer is not None:
        return

    try:
        from motus.runtime.tracing import TraceManager

        _tracer = TraceManager()
        atexit.register(_auto_export)
        logger.debug("Registered motus tracing for Anthropic SDK")
    except Exception as e:
        logger.warning(f"Could not register motus tracing: {e}")


def get_tracer():
    """Get the TraceManager instance (call register_tracing() first)."""
    return _tracer


def _auto_export():
    """Auto-export traces on process exit (registered via atexit)."""
    if _tracer and _tracer.task_meta and _tracer.config.export_enabled:
        try:
            _tracer.export_trace()
        except Exception:
            pass


def _state_to_anthropic_messages(state: list[ChatMessage]) -> list[dict[str, Any]]:
    """Convert motus ChatMessage state to Anthropic message format."""
    messages: list[dict[str, Any]] = []
    for msg in state:
        if msg.role in ("user", "assistant"):
            messages.append({"role": msg.role, "content": msg.content or ""})
    return messages


class ToolRunner:
    """Motus wrapper for Anthropic SDK tool runners.

    Holds configuration (model, tools, system prompt, etc.) and provides
    a ``run_turn()`` method satisfying the serve agent contract::

        async def run_turn(message, state) -> (response, state)

    Each ``run_turn()`` creates a fresh ``MotusBetaAsyncToolRunner``
    (tool runners are single-use generators that cannot be re-iterated).
    """

    def __init__(
        self,
        *,
        model: str,
        max_tokens: int,
        tools: list,
        system: str | None = None,
        max_iterations: int | None = None,
        **kwargs: Any,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.tools = tools
        self.system = system
        self.max_iterations = max_iterations
        self.extra_params = kwargs

    async def run_turn(
        self, message: ChatMessage, state: list[ChatMessage]
    ) -> tuple[ChatMessage, list[ChatMessage]]:
        """Run a single conversational turn through the Anthropic tool runner.

        Satisfies the serve agent contract:
        ``(ChatMessage, list[ChatMessage]) -> (ChatMessage, list[ChatMessage])``
        """
        from motus.models import ChatMessage as _CM

        register_tracing()

        # Build Anthropic messages from state + current message
        messages = _state_to_anthropic_messages(state)
        messages.append({"role": "user", "content": message.content or ""})

        # Allocate root agent span
        parent_task_id = None
        if _tracer and _tracer.config.is_collecting:
            parent_task_id = _tracer.allocate_external_task_id()
            root_meta = build_agent_call_meta(model=self.model, start_us=_now_us())
            _tracer.ingest_external_span(root_meta, task_id=parent_task_id)

        # Separate runnable tools from raw tool dicts
        from anthropic.lib.tools import (
            BetaAsyncBuiltinFunctionTool,
            BetaAsyncFunctionTool,
        )

        runnable_tools = []
        raw_tools = []
        for tool in self.tools:
            if isinstance(tool, (BetaAsyncFunctionTool, BetaAsyncBuiltinFunctionTool)):
                runnable_tools.append(tool)
            else:
                raw_tools.append(tool)

        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "tools": [t.to_dict() for t in runnable_tools] + raw_tools,
            **self.extra_params,
        }
        if self.system:
            params["system"] = self.system

        client = AsyncAnthropic()

        runner = MotusBetaAsyncToolRunner(
            params=params,
            options={},
            tools=runnable_tools,
            client=client,
            max_iterations=self.max_iterations,
            trace_manager=_tracer,
            parent_task_id=parent_task_id,
        )
        result = await runner.until_done()

        # Extract text response
        text_parts = [
            b.text for b in result.content if getattr(b, "type", None) == "text"
        ]
        response_text = "\n".join(text_parts) or "(no response)"
        response = _CM.assistant_message(content=response_text)

        # Finalize root agent span
        if _tracer and parent_task_id is not None:
            _tracer.update_external_span(parent_task_id, {"end_us": _now_us()})
            _tracer.close()

        return response, state + [message, response]


# Auto-register tracing on import
register_tracing()
