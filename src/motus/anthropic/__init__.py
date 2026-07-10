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

import inspect
import logging
from typing import TYPE_CHECKING, Any

try:
    from anthropic import AsyncAnthropic
    from anthropic.lib.tools import (  # noqa: F401
        BetaAsyncBuiltinFunctionTool,
        BetaAsyncFunctionTool,
        BetaBuiltinFunctionTool,
        BetaFunctionTool,
        beta_async_tool,
        beta_tool,
    )
except ImportError as exc:
    raise ImportError(
        "motus.anthropic requires the Anthropic SDK with tool runner support. "
        "Install or upgrade with: uv pip install 'anthropic>=0.49.0'"
    ) from exc

from opentelemetry import trace

from ._motus_runner import (  # noqa: F401
    MotusBetaAsyncStreamingToolRunner,
    MotusBetaAsyncToolRunner,
    MotusBetaStreamingToolRunner,
    MotusBetaToolRunner,
)
from ._motus_tracing import _now_ns, start_agent_span

_TOOL_TYPES = (
    BetaAsyncFunctionTool,
    BetaAsyncBuiltinFunctionTool,
    BetaFunctionTool,
    BetaBuiltinFunctionTool,
)

if TYPE_CHECKING:
    from motus.models import ChatMessage

logger = logging.getLogger("AgentTracer")


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

        # Build Anthropic messages from state + current message
        messages = _state_to_anthropic_messages(state)
        messages.append({"role": "user", "content": message.content or ""})

        # Start root agent span (ended after run completes)
        parent_span = start_agent_span(model=self.model)

        # Separate runnable tools from raw tool dicts, auto-wrapping
        # plain functions and motus @tool objects so users don't need
        # explicit @beta_async_tool decorators
        from motus.tools.core.function_tool import FunctionTool as _FunctionTool

        runnable_tools = []
        raw_tools = []
        for tool in self.tools:
            if isinstance(tool, _TOOL_TYPES):
                runnable_tools.append(tool)
            elif isinstance(tool, _FunctionTool):
                # Unwrap motus @tool -> extract the raw function
                fn = tool.func
                if inspect.iscoroutinefunction(fn):
                    runnable_tools.append(beta_async_tool(fn))
                else:
                    runnable_tools.append(beta_tool(fn))
            elif callable(tool):
                if inspect.iscoroutinefunction(tool):
                    runnable_tools.append(beta_async_tool(tool))
                else:
                    runnable_tools.append(beta_tool(tool))
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
            parent_span=parent_span,
        )
        result = await runner.until_done()

        # Extract text response
        text_parts = [
            b.text for b in result.content if getattr(b, "type", None) == "text"
        ]
        response_text = "\n".join(text_parts) or "(no response)"
        response = _CM.assistant_message(content=response_text)

        # End root agent span
        parent_span.end(end_time=_now_ns())

        return response, state + [message, response]
