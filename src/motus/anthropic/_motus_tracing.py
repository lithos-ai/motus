"""Span building helpers for Anthropic SDK tool runner tracing.

Creates OTel spans directly using get_tracer() instead of building dicts
for TraceManager.ingest_external_span().
"""

from __future__ import annotations

import time
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace

from motus.runtime.tracing.agent_tracer import (
    ATTR_ERROR,
    ATTR_FUNC,
    ATTR_MODEL_INPUT,
    ATTR_MODEL_NAME,
    ATTR_MODEL_OUTPUT,
    ATTR_TASK_TYPE,
    ATTR_TOOL_INPUT,
    ATTR_TOOL_OUTPUT,
    ATTR_USAGE,
    get_tracer,
    json_attr,
)

# Limit large values to prevent huge SSE payloads
_MAX_VALUE_LEN = 4000


def _now_ns() -> int:
    """Current time in nanoseconds since epoch (OTel native unit)."""
    return int(time.time() * 1_000_000_000)


def _truncate(value: Any, limit: int = _MAX_VALUE_LEN) -> Any:
    """Truncate strings that exceed *limit* characters."""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "...(truncated)"
    return value


def start_agent_span(
    *,
    model: str,
) -> trace.Span:
    """Start a root agent_call span that parents all model/tool spans in a turn.

    Returns the started (but not ended) span. Caller must call span.end()
    when the turn completes.
    """
    tracer = get_tracer()
    span = tracer.start_span(
        f"anthropic_tool_runner({model})",
        attributes={
            ATTR_FUNC: f"anthropic_tool_runner({model})",
            ATTR_TASK_TYPE: "agent_call",
            ATTR_MODEL_NAME: model,
        },
        start_time=_now_ns(),
    )
    return span


def emit_model_span(
    *,
    message: Any,
    model: str,
    input_messages: list | None,
    start_ns: int,
    end_ns: int,
    parent_context: otel_context.Context | None = None,
) -> None:
    """Create and immediately end a model_call span from a BetaMessage."""
    tracer = get_tracer()
    ctx = parent_context or otel_context.get_current()

    span = tracer.start_span(
        model,
        context=ctx,
        attributes={
            ATTR_FUNC: model,
            ATTR_TASK_TYPE: "model_call",
            ATTR_MODEL_NAME: model,
        },
        start_time=start_ns,
    )

    # Usage
    usage_dict: dict[str, Any] = {}
    if message is not None and hasattr(message, "usage") and message.usage:
        usage = message.usage
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        usage_dict = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        cache_creation = getattr(usage, "cache_creation_input_tokens", None)
        cache_read = getattr(usage, "cache_read_input_tokens", None)
        if cache_creation is not None:
            usage_dict["cache_creation_input_tokens"] = cache_creation
        if cache_read is not None:
            usage_dict["cache_read_input_tokens"] = cache_read
        span.set_attribute(ATTR_USAGE, json_attr(usage_dict))

    # Output metadata
    if message is not None and hasattr(message, "content") and message.content:
        output_meta: dict[str, Any] = {"model": model}
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        for block in message.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(_truncate(getattr(block, "text", "")))
            elif btype == "tool_use":
                tool_calls.append(
                    {
                        "name": getattr(block, "name", None),
                        "input": getattr(block, "input", None),
                        "id": getattr(block, "id", None),
                    }
                )
            elif btype == "thinking":
                thinking = getattr(block, "thinking", None)
                if thinking:
                    output_meta["thinking"] = _truncate(thinking)
        if text_parts:
            output_meta["content"] = "\n".join(text_parts)
            output_meta["role"] = "assistant"
        if tool_calls:
            output_meta["tool_calls"] = tool_calls
        if usage_dict:
            output_meta["usage"] = usage_dict
        span.set_attribute(ATTR_MODEL_OUTPUT, json_attr(output_meta))

    # Input metadata
    if input_messages:
        span.set_attribute(ATTR_MODEL_INPUT, json_attr(input_messages))

    span.end(end_time=end_ns)


def emit_tool_span(
    *,
    tool_name: str,
    tool_input: Any,
    tool_output: Any | None = None,
    start_ns: int,
    end_ns: int,
    parent_context: otel_context.Context | None = None,
    error: str | None = None,
) -> None:
    """Create and immediately end a tool_call span."""
    tracer = get_tracer()
    ctx = parent_context or otel_context.get_current()

    span = tracer.start_span(
        tool_name,
        context=ctx,
        attributes={
            ATTR_FUNC: tool_name,
            ATTR_TASK_TYPE: "tool_call",
            ATTR_TOOL_INPUT: json_attr({"name": tool_name, "arguments": tool_input}),
        },
        start_time=start_ns,
    )
    if tool_output is not None:
        span.set_attribute(ATTR_TOOL_OUTPUT, json_attr(_truncate(tool_output)))
    if error:
        span.set_attribute(ATTR_ERROR, error)
        span.set_status(trace.StatusCode.ERROR, error)

    span.end(end_time=end_ns)
