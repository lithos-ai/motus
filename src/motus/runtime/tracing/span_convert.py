"""Convert OTel ReadableSpan to the viewer/export dict format.

This replaces trace_to_otel.py — since spans are now native OTel, the
conversion goes OTel → viewer dict (the reverse of the old direction).
"""

import json
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan

from .pricing import TEXT_MODEL_PRICING


def _get_attr(span: ReadableSpan, key: str, default=None):
    """Safely read an attribute from a ReadableSpan."""
    attrs = span.attributes or {}
    return attrs.get(key, default)


def _parse_json_attr(span: ReadableSpan, key: str) -> Any:
    """Parse a JSON-encoded span attribute."""
    val = _get_attr(span, key)
    if not val:
        return None
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return None


def _normalize_model_name(model_name: str) -> str:
    """Strip date suffixes for pricing lookup."""
    if not model_name:
        return model_name
    return model_name.split("-202")[0] if "-202" in model_name else model_name


def readable_span_to_viewer_dict(span: ReadableSpan) -> dict:
    """Convert an OTel ReadableSpan to the viewer dict format.

    This produces the same structure that the JS trace viewer and Jaeger
    exporter expect: {traceId, spanId, parentSpanId, operationName,
    startTime, duration, tags, kind, meta}.
    """
    ctx = span.context
    trace_id = format(ctx.trace_id, "032x") if ctx else "unknown"

    # Use motus task_id_int as span ID if available, else OTel span ID
    task_id_int = _get_attr(span, "motus.task_id_int")
    span_id = (
        f"span-{task_id_int}"
        if task_id_int is not None
        else format(ctx.span_id, "016x")
    )

    # Parent span ID
    parent_task_id = _get_attr(span, "motus.parent_task_id", -1)
    parent_span_id = (
        f"span-{parent_task_id}" if parent_task_id not in (None, -1) else None
    )

    # If no motus parent, try OTel parent
    if parent_span_id is None and span.parent is not None:
        parent_span_id = format(span.parent.span_id, "016x")

    # Timestamps (OTel stores nanoseconds, viewer expects microseconds)
    start_ns = span.start_time or 0
    end_ns = span.end_time or start_ns
    start_us = start_ns // 1000
    duration_us = (end_ns - start_ns) // 1000

    func_name = _get_attr(span, "motus.func", span.name)
    task_type = _get_attr(span, "motus.task_type", "")

    # Build tags dict
    tags: dict[str, Any] = {
        "task.id": task_id_int if task_id_int is not None else span_id,
        "task.func": func_name,
        "task.type": task_type,
    }

    # Model metadata
    model_name = _get_attr(span, "motus.model_name", "")
    model_output = _parse_json_attr(span, "motus.model_output_meta")
    model_input = _parse_json_attr(span, "motus.model_input_meta")
    usage = _parse_json_attr(span, "motus.usage")

    if model_name:
        tags["model.name"] = model_name
    if model_output and isinstance(model_output, dict):
        out_model = model_output.get("model")
        if out_model:
            tags["model.name"] = out_model
        out_usage = model_output.get("usage") or usage
        if isinstance(out_usage, dict):
            tags["model.tokens.total"] = out_usage.get("total_tokens", 0)
            tags["model.tokens.prompt"] = out_usage.get(
                "prompt_tokens", out_usage.get("input_tokens", 0)
            )
            tags["model.tokens.completion"] = out_usage.get(
                "completion_tokens", out_usage.get("output_tokens", 0)
            )
            reasoning = out_usage.get("completion_tokens_details", {})
            if isinstance(reasoning, dict) and reasoning.get("reasoning_tokens"):
                tags["model.tokens.reasoning"] = reasoning["reasoning_tokens"]

            # Cost calculation
            pricing_name = _normalize_model_name(model_name)
            if pricing_name in TEXT_MODEL_PRICING:
                pricing = TEXT_MODEL_PRICING[pricing_name]
                input_k = tags.get("model.tokens.prompt", 0) / 1000
                output_k = tags.get("model.tokens.completion", 0) / 1000
                tags["model.cost_usd"] = round(
                    input_k * pricing.input_per_1k + output_k * pricing.output_per_1k, 5
                )
    elif usage and isinstance(usage, dict):
        tags["model.tokens.total"] = usage.get("total_tokens", 0)
        tags["model.tokens.prompt"] = usage.get("input_tokens", 0)
        tags["model.tokens.completion"] = usage.get("output_tokens", 0)

    # Tool metadata
    tool_meta = _parse_json_attr(span, "motus.tool_meta")
    if tool_meta and isinstance(tool_meta, list):
        tags["tools.available"] = len(tool_meta)
        tool_names = [
            t.get("function", {}).get("name", "")
            for t in tool_meta
            if isinstance(t, dict) and t.get("function")
        ]
        if tool_names:
            tags["tools.names"] = ", ".join(tool_names)

    tool_input = _parse_json_attr(span, "motus.tool_input_meta")
    if tool_input and isinstance(tool_input, dict) and tool_input.get("name"):
        tags["tool.name"] = tool_input["name"]

    # Span kind and display name
    kind = "INTERNAL"
    display_name = func_name
    if task_type in ("model_call",):
        kind = "CLIENT"
        if model_name:
            display_name = _normalize_model_name(model_name)
    elif task_type in ("tool_call",):
        kind = "CLIENT"

    # Reconstruct the rich meta dict for the detail panel
    meta: dict[str, Any] = {"func": func_name, "task_type": task_type}
    if model_name:
        meta["model_name"] = model_name
    if model_output:
        meta["model_output_meta"] = model_output
    if model_input:
        meta["model_input_meta"] = model_input
    if tool_input:
        meta["tool_input_meta"] = tool_input
    tool_output = _parse_json_attr(span, "motus.tool_output_meta")
    if tool_output:
        meta["tool_output_meta"] = tool_output
    if usage:
        meta["usage"] = usage
    error = _get_attr(span, "motus.error")
    if error:
        meta["error"] = error

    return {
        "traceId": trace_id,
        "spanId": span_id,
        "parentSpanId": parent_span_id,
        "operationName": display_name,
        "startTime": start_us,
        "duration": duration_us,
        "tags": tags,
        "logs": [],
        "references": [],
        "kind": kind,
        "meta": meta,
    }
