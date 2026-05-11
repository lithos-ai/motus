from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .tracing import AgentTrace, write_trace

MOTUS_TRACE_SCHEMA = "motustracing.agent_trace.v1"


def trace_from_task_meta(
    task_meta: dict[Any, dict[str, Any]],
    *,
    trace_id: str,
    agent_instance_id: str | None = None,
    agent_class_id: str | None = None,
    source: dict[str, Any] | None = None,
    tokenizer: dict[str, Any] | None = None,
) -> AgentTrace:
    model_spans = _spans_by_type(task_meta, "model_call")
    if not model_spans:
        raise ValueError("task_meta does not contain any model_call spans")

    tool_spans = _spans_by_type(task_meta, "tool_call")
    agent_class = agent_class_id or _agent_class_id(task_meta) or "agent"
    previous_messages: list[dict[str, Any]] | None = None
    turns = []

    for index, model_span in enumerate(model_spans):
        next_start_us = (
            model_spans[index + 1].get("start_us")
            if index + 1 < len(model_spans)
            else None
        )
        messages = _messages(model_span)
        tools = _tools_for_turn(
            model_span,
            _tool_spans_between(tool_spans, model_span, next_start_us),
        )
        input_delta = _input_delta(previous_messages, messages, index)
        turns.append(
            {
                "turn_index": index,
                "input_delta": input_delta,
                "output_tokens": _output_tokens(model_span),
                "tools": tools,
                "is_terminal": not tools,
            }
        )
        previous_messages = messages

    return AgentTrace.model_validate(
        {
            "schema": MOTUS_TRACE_SCHEMA,
            "trace_id": trace_id,
            "source": source or {},
            "tokenizer": tokenizer or {},
            "agent": {
                "agent_instance_id": agent_instance_id or trace_id,
                "agent_class_id": agent_class,
            },
            "system_prompt": {"text": _system_prompt(_messages(model_spans[0]))},
            "turns": turns,
        }
    )


def write_trace_from_task_meta(
    task_meta: dict[Any, dict[str, Any]],
    path: str | Path,
    *,
    trace_id: str,
    agent_instance_id: str | None = None,
    agent_class_id: str | None = None,
    source: dict[str, Any] | None = None,
    tokenizer: dict[str, Any] | None = None,
) -> AgentTrace:
    trace = trace_from_task_meta(
        task_meta,
        trace_id=trace_id,
        agent_instance_id=agent_instance_id,
        agent_class_id=agent_class_id,
        source=source,
        tokenizer=tokenizer,
    )
    write_trace(trace, path)
    return trace


async def record_agent_run(
    agent: Any,
    user_prompt: str,
    trace_path: str | Path,
    *,
    trace_id: str,
    agent_instance_id: str | None = None,
    agent_class_id: str | None = None,
    source: dict[str, Any] | None = None,
    tokenizer: dict[str, Any] | None = None,
) -> tuple[Any, AgentTrace]:
    from motus.runtime.agent_runtime import get_runtime

    runtime = get_runtime()
    tracer = runtime.scheduler.tracer
    before = set(tracer.task_meta)

    result = await agent(user_prompt)

    run_task_meta = {
        task_id: meta
        for task_id, meta in tracer.task_meta.items()
        if task_id not in before
    }
    trace = write_trace_from_task_meta(
        run_task_meta,
        trace_path,
        trace_id=trace_id,
        agent_instance_id=agent_instance_id,
        agent_class_id=agent_class_id,
        source=source,
        tokenizer=tokenizer,
    )
    return result, trace


def _spans_by_type(
    task_meta: dict[Any, dict[str, Any]],
    task_type: str,
) -> list[dict[str, Any]]:
    spans = [meta for meta in task_meta.values() if meta.get("task_type") == task_type]
    return sorted(spans, key=lambda meta: (meta.get("start_us", 0), meta.get("end_us", 0)))


def _agent_class_id(task_meta: dict[Any, dict[str, Any]]) -> str | None:
    agents = _spans_by_type(task_meta, "agent_call")
    if not agents:
        return None
    return agents[0].get("agent_id")


def _messages(model_span: dict[str, Any]) -> list[dict[str, Any]]:
    messages = model_span.get("model_input_meta") or []
    return [message for message in messages if isinstance(message, dict)]


def _system_prompt(messages: Iterable[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "system":
            return _message_content(message)
    return ""


def _input_delta(
    previous_messages: list[dict[str, Any]] | None,
    messages: list[dict[str, Any]],
    turn_index: int,
) -> dict[str, Any]:
    if previous_messages is None:
        new_messages = [message for message in messages if message.get("role") != "system"]
    elif messages[: len(previous_messages)] == previous_messages:
        new_messages = messages[len(previous_messages) :]
    else:
        new_messages = messages[-1:]

    if previous_messages is not None and new_messages and new_messages[0].get("role") == "assistant":
        new_messages = new_messages[1:]

    content_messages = [
        message
        for message in new_messages
        if message.get("role") in {"user", "tool"} and _message_content(message)
    ]
    text = "\n".join(_message_content(message) for message in content_messages)
    roles = {message.get("role") for message in content_messages}
    kind = "user" if turn_index == 0 or "user" in roles else "tool_result"
    return {"kind": kind, "text": text}


def _message_content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def _output_tokens(model_span: dict[str, Any]) -> int:
    output = model_span.get("model_output_meta") or {}
    usage = output.get("usage") or {}
    tokens = usage.get("completion_tokens")
    return tokens if isinstance(tokens, int) else 0


def _tool_spans_between(
    tool_spans: list[dict[str, Any]],
    model_span: dict[str, Any],
    next_model_start_us: int | float | None,
) -> list[dict[str, Any]]:
    lower = model_span.get("end_us", model_span.get("start_us", 0))
    upper = next_model_start_us if next_model_start_us is not None else float("inf")
    return [
        span
        for span in tool_spans
        if lower <= span.get("start_us", 0) < upper
    ]


def _tools_for_turn(
    model_span: dict[str, Any],
    tool_spans: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    planned_calls = _planned_tool_calls(model_span)
    used_span_indexes: set[int] = set()
    tools = []

    for call in planned_calls:
        span_index = _find_matching_tool_span(call["name"], tool_spans, used_span_indexes)
        span = tool_spans[span_index] if span_index is not None else None
        if span_index is not None:
            used_span_indexes.add(span_index)
        tools.append(
            {
                "name": call["name"],
                "args": call["args"] or _tool_args(span),
                "duration_ms": _duration_ms(span),
            }
        )

    if planned_calls:
        return tools

    for index, span in enumerate(tool_spans):
        if index in used_span_indexes:
            continue
        tool_input = span.get("tool_input_meta") or {}
        tools.append(
            {
                "name": tool_input.get("name", "unknown"),
                "args": _tool_args(span),
                "duration_ms": _duration_ms(span),
            }
        )
    return tools


def _planned_tool_calls(model_span: dict[str, Any]) -> list[dict[str, Any]]:
    output = model_span.get("model_output_meta") or {}
    calls = []
    for call in output.get("tool_calls") or []:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict) or not function.get("name"):
            continue
        calls.append(
            {
                "name": function["name"],
                "args": _parse_tool_args(function.get("arguments")),
            }
        )
    if calls:
        return calls

    return [
        {"name": name, "args": {}}
        for name in model_span.get("chosen_tools") or []
        if isinstance(name, str)
    ]


def _parse_tool_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"arguments": raw}
        return parsed if isinstance(parsed, dict) else {"arguments": parsed}
    return {}


def _find_matching_tool_span(
    name: str,
    tool_spans: list[dict[str, Any]],
    used_span_indexes: set[int],
) -> int | None:
    for index, span in enumerate(tool_spans):
        if index not in used_span_indexes and (span.get("tool_input_meta") or {}).get("name") == name:
            return index
    for index in range(len(tool_spans)):
        if index not in used_span_indexes:
            return index
    return None


def _tool_args(span: dict[str, Any] | None) -> dict[str, Any]:
    if span is None:
        return {}
    tool_input = span.get("tool_input_meta") or {}
    args = tool_input.get("arguments")
    return args if isinstance(args, dict) else {}


def _duration_ms(span: dict[str, Any] | None) -> float | None:
    if span is None:
        return None
    start_us = span.get("start_us")
    end_us = span.get("end_us")
    if not isinstance(start_us, (int, float)) or not isinstance(end_us, (int, float)):
        return None
    return (end_us - start_us) / 1000.0
