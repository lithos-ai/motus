"""Span building helpers for Anthropic SDK tool runner tracing.

Pure functions that convert Anthropic SDK objects (BetaMessage, tool_use blocks)
into motus task_meta dicts for TraceManager.ingest_external_span().
"""

from __future__ import annotations

import time
from typing import Any

from motus.runtime.types import AGENT_CALL, MODEL_CALL, TOOL_CALL

# Limit large values to prevent huge SSE payloads (matches claude_agent convention)
_MAX_VALUE_LEN = 4000


def _now_us() -> int:
    """Current time in microseconds since epoch."""
    return int(time.time() * 1_000_000)


def _truncate(value: Any, limit: int = _MAX_VALUE_LEN) -> Any:
    """Truncate strings that exceed *limit* characters."""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "...(truncated)"
    return value


def build_agent_call_meta(
    *,
    model: str,
    start_us: int,
) -> dict[str, Any]:
    """Build a root agent_call span that parents all model/tool spans in a turn."""
    return {
        "func": f"anthropic_tool_runner({model})",
        "task_type": AGENT_CALL,
        "parent": None,
        "start_us": start_us,
        "end_us": 0,  # Updated when turn completes
        "model_name": model,
    }


def build_model_call_meta(
    *,
    message: Any,
    model: str,
    input_messages: list | None,
    start_us: int,
    end_us: int,
    parent: int | None,
) -> dict[str, Any]:
    """Build a model_call span from a BetaMessage / ParsedBetaMessage."""
    meta: dict[str, Any] = {
        "func": model,
        "task_type": MODEL_CALL,
        "parent": parent,
        "start_us": start_us,
        "end_us": end_us,
        "model_name": model,
    }

    # Usage
    if message is not None and hasattr(message, "usage") and message.usage:
        usage = message.usage
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        meta["usage"] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        cache_creation = getattr(usage, "cache_creation_input_tokens", None)
        cache_read = getattr(usage, "cache_read_input_tokens", None)
        if cache_creation is not None:
            meta["usage"]["cache_creation_input_tokens"] = cache_creation
        if cache_read is not None:
            meta["usage"]["cache_read_input_tokens"] = cache_read

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
        if meta.get("usage"):
            output_meta["usage"] = meta["usage"]
        meta["model_output_meta"] = output_meta

    # Input metadata
    if input_messages:
        meta["model_input_meta"] = input_messages

    return meta


def build_tool_call_meta(
    *,
    tool_name: str,
    tool_input: Any,
    tool_output: Any | None = None,
    start_us: int,
    end_us: int,
    parent: int | None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build a tool_call span for a single tool invocation."""
    meta: dict[str, Any] = {
        "func": tool_name,
        "task_type": TOOL_CALL,
        "parent": parent,
        "start_us": start_us,
        "end_us": end_us,
        "tool_input_meta": {
            "name": tool_name,
            "arguments": tool_input,
        },
    }
    if tool_output is not None:
        meta["tool_output_meta"] = _truncate(tool_output)
    if error:
        meta["error"] = error
    return meta
