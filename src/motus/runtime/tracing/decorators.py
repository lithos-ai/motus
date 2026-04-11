"""Tracing decorators for agent tasks.

These decorators replace both the old @agent_task hook emission and the
extractor registry.  Each decorator creates an OTel span with the
appropriate attributes for its task type.

Usage:
    @traced_model_call
    async def model_serve_task(client, model, messages, tools, ...):
        ...

    @traced_agent_call
    async def _execute(self, user_prompt=None, **kwargs):
        ...

    @traced_tool_call
    async def _execute(self, **kwargs):
        ...
"""

from __future__ import annotations

import functools
from typing import Any

from opentelemetry import trace

from .agent_tracer import (
    ATTR_AGENT_ID,
    ATTR_CHOSEN_TOOLS,
    ATTR_ERROR,
    ATTR_FUNC,
    ATTR_MODEL_INPUT,
    ATTR_MODEL_NAME,
    ATTR_MODEL_OUTPUT,
    ATTR_TASK_TYPE,
    ATTR_TOOL_INPUT,
    ATTR_TOOL_META,
    ATTR_TOOL_OUTPUT,
    get_tracer,
    json_attr,
)


def _safe_dump(obj: Any) -> Any:
    """Safely convert an object to a JSON-serializable dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return str(obj)


def traced_model_call(func):
    """Trace an LLM model call with input/output metadata."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        tracer = get_tracer()

        # Extract model info from args
        # model_serve_task(client, model, messages, tools, ...)
        model = kwargs.get("model") or (args[1] if len(args) > 1 else "unknown")
        messages = kwargs.get("messages") or (args[2] if len(args) > 2 else [])
        tools = kwargs.get("tools") or (args[3] if len(args) > 3 else None)

        tool_meta = []
        if tools:
            try:
                tool_meta = [
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": tools[name].description,
                            "parameters": tools[name].json_schema,
                        },
                    }
                    for name in tools
                ]
            except Exception:
                pass

        with tracer.start_as_current_span(
            model,
            attributes={
                ATTR_FUNC: func.__name__,
                ATTR_TASK_TYPE: "model_call",
                ATTR_MODEL_NAME: model,
                ATTR_MODEL_INPUT: json_attr([_safe_dump(m) for m in messages]),
                ATTR_TOOL_META: json_attr(tool_meta),
            },
        ) as span:
            try:
                result = await func(*args, **kwargs)

                # Extract output metadata from result
                chosen_tools = []
                if result and hasattr(result, "to_message"):
                    message = result.to_message()
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        chosen_tools = [tc.function.name for tc in message.tool_calls]

                span.set_attribute(ATTR_MODEL_OUTPUT, json_attr(_safe_dump(result)))
                if chosen_tools:
                    span.set_attribute(ATTR_CHOSEN_TOOLS, json_attr(chosen_tools))

                return result
            except Exception as e:
                span.set_attribute(ATTR_ERROR, str(e))
                span.set_status(trace.StatusCode.ERROR, str(e))
                span.record_exception(e)
                raise

    return wrapper


def traced_agent_call(func):
    """Trace an agent execution with agent identity metadata."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        tracer = get_tracer()

        # args[0] is self (the agent instance)
        agent = args[0] if args else None
        agent_name = getattr(agent, "name", None) or (
            agent.__class__.__name__ if agent else func.__name__
        )

        with tracer.start_as_current_span(
            agent_name,
            attributes={
                ATTR_FUNC: func.__name__,
                ATTR_TASK_TYPE: "agent_call",
                ATTR_AGENT_ID: agent_name,
            },
        ) as span:
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                span.set_attribute(ATTR_ERROR, str(e))
                span.set_status(trace.StatusCode.ERROR, str(e))
                span.record_exception(e)
                raise

    return wrapper


def traced_tool_call(func):
    """Trace a tool execution with input/output metadata."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        tracer = get_tracer()

        # args[0] is self (the Tool instance)
        tool = args[0] if args else None
        tool_name = getattr(tool, "name", "unknown")

        with tracer.start_as_current_span(
            tool_name,
            attributes={
                ATTR_FUNC: func.__name__,
                ATTR_TASK_TYPE: "tool_call",
                ATTR_TOOL_INPUT: json_attr({"name": tool_name, "arguments": kwargs}),
            },
        ) as span:
            try:
                result = await func(*args, **kwargs)
                span.set_attribute(ATTR_TOOL_OUTPUT, json_attr(_safe_dump(result)))
                return result
            except Exception as e:
                span.set_attribute(ATTR_ERROR, str(e))
                span.set_status(trace.StatusCode.ERROR, str(e))
                span.record_exception(e)
                raise

    return wrapper


def traced(name: str | None = None, task_type: str = "normal_task"):
    """Generic tracing decorator for any async function."""

    def decorator(func):
        span_name = name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(
                span_name,
                attributes={
                    ATTR_FUNC: func.__name__,
                    ATTR_TASK_TYPE: task_type,
                },
            ) as span:
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_attribute(ATTR_ERROR, str(e))
                    span.set_status(trace.StatusCode.ERROR, str(e))
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator
