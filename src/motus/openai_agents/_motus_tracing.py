"""MotusTracingProcessor -- bridges OAI SDK tracing into motus OTel spans.

OAI SDK emits trace/span events via its own TracingProcessor interface
(not OTel). This processor converts each completed OAI span into an OTel
span on the motus tracer with motus.* attributes, so it flows through
our SpanProcessors and appears in the trace viewer, Jaeger export, and
analytics pipeline.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

from agents.tracing.processor_interface import TracingProcessor
from agents.tracing.spans import Span
from agents.tracing.traces import Trace
from opentelemetry import trace

from motus.runtime.tracing.agent_tracer import (
    ATTR_AGENT_ID,
    ATTR_ERROR,
    ATTR_FUNC,
    ATTR_MODEL_INPUT,
    ATTR_MODEL_NAME,
    ATTR_MODEL_OUTPUT,
    ATTR_TASK_TYPE,
    ATTR_TOOL_INPUT,
    ATTR_USAGE,
    get_tracer,
    json_attr,
)

logger = logging.getLogger("AgentTracer")


def _iso_to_ns(iso_str: str | None) -> int:
    """Convert ISO 8601 timestamp to nanoseconds since epoch (OTel native unit)."""
    if not iso_str:
        return 0
    try:
        dt = datetime.datetime.fromisoformat(iso_str)
        return int(dt.timestamp() * 1_000_000_000)
    except (ValueError, TypeError):
        return 0


class MotusTracingProcessor(TracingProcessor):
    """Receives OAI SDK trace/span events and creates OTel spans on the motus tracer.

    Usage::

        from agents import add_trace_processor
        processor = MotusTracingProcessor()
        add_trace_processor(processor)
    """

    def on_trace_start(self, trace_obj: Trace) -> None:
        pass

    def on_trace_end(self, trace_obj: Trace) -> None:
        pass

    def on_span_start(self, span: Span[Any]) -> None:
        pass  # We process on end when all data is available

    def on_span_end(self, span: Span[Any]) -> None:
        """Convert completed OAI span to an OTel span on the motus tracer."""
        data = span.span_data
        exported = data.export() if data else {}
        span_type = data.type if data else "unknown"

        # Map OAI span type -> motus task_type + display name
        task_type, func_name = self._classify(span_type, data)

        tracer = get_tracer()

        # Build motus attributes
        attrs: dict[str, Any] = {
            ATTR_FUNC: func_name,
            ATTR_TASK_TYPE: task_type,
        }

        if span.error:
            error_msg = span.error.get("message", str(span.error))
            attrs[ATTR_ERROR] = error_msg

        # Add type-specific enrichment
        self._enrich_attrs(attrs, span_type, data)

        # Timestamps
        start_ns = _iso_to_ns(span.started_at)
        end_ns = _iso_to_ns(span.ended_at)

        # Create the OTel span. Parent context is handled natively by OTel
        # (no manual _span_id_map needed).
        motus_span = tracer.start_span(
            func_name,
            attributes=attrs,
            start_time=start_ns if start_ns > 0 else None,
        )

        if span.error:
            motus_span.set_status(
                trace.StatusCode.ERROR,
                attrs.get(ATTR_ERROR, ""),
            )

        motus_span.end(end_time=end_ns if end_ns > 0 else None)

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass

    # -- internal helpers --

    @staticmethod
    def _classify(span_type: str, data: Any) -> tuple[str, str]:
        """Return (task_type, func_name) for a given OAI span type."""
        if span_type == "generation":
            model = getattr(data, "model", None) or "llm"
            return "model_call", model
        if span_type == "response":
            # Responses API -- data.response is a Response object
            resp = getattr(data, "response", None)
            model = getattr(resp, "model", None) or "llm"
            return "model_call", model
        if span_type == "function":
            name = getattr(data, "name", None) or "tool"
            return "tool_call", name
        if span_type == "agent":
            name = getattr(data, "name", None) or "agent"
            return "agent_call", name
        if span_type == "handoff":
            to_agent = getattr(data, "to_agent", None) or "?"
            return "handoff", f"handoff->{to_agent}"
        if span_type == "guardrail":
            name = getattr(data, "name", None) or "guardrail"
            return "guardrail", name
        # custom, mcp_tools, etc.
        name = getattr(data, "name", None) or span_type
        return span_type, name

    @staticmethod
    def _enrich_attrs(attrs: dict, span_type: str, data: Any) -> None:
        """Add type-specific motus attributes."""
        if span_type == "generation":
            if hasattr(data, "model") and data.model:
                attrs[ATTR_MODEL_NAME] = data.model
            # Model output
            output_meta: dict[str, Any] = {}
            if hasattr(data, "model") and data.model:
                output_meta["model"] = data.model
            if hasattr(data, "usage") and data.usage:
                output_meta["usage"] = data.usage
            if hasattr(data, "output") and data.output:
                first = data.output[0] if data.output else {}
                output_meta["content"] = first.get("content")
                output_meta["tool_calls"] = first.get("tool_calls")
                output_meta["role"] = first.get("role")
            if output_meta:
                attrs[ATTR_MODEL_OUTPUT] = json_attr(output_meta)
            if hasattr(data, "input") and data.input:
                # OAI SDK tool messages lack 'name' field -- resolve from
                # assistant tool_calls in the same conversation turn
                messages = list(data.input)
                tc_id_to_name: dict[str, str] = {}
                for msg in messages:
                    if msg.get("role") == "assistant":
                        for tc in msg.get("tool_calls") or []:
                            fn = tc.get("function") or {}
                            if tc.get("id") and fn.get("name"):
                                tc_id_to_name[tc["id"]] = fn["name"]
                for msg in messages:
                    if msg.get("role") == "tool" and not msg.get("name"):
                        name = tc_id_to_name.get(msg.get("tool_call_id", ""))
                        if name:
                            msg["name"] = name
                attrs[ATTR_MODEL_INPUT] = json_attr(messages)

        elif span_type == "response":
            # Responses API -- data is ResponseSpanData
            resp = getattr(data, "response", None)
            if resp is not None:
                model = getattr(resp, "model", None)
                if model:
                    attrs[ATTR_MODEL_NAME] = model

                output_meta: dict[str, Any] = {}
                if model:
                    output_meta["model"] = model
                if resp.usage:
                    usage_dict = {
                        "input_tokens": resp.usage.input_tokens,
                        "output_tokens": resp.usage.output_tokens,
                        "total_tokens": resp.usage.total_tokens,
                    }
                    output_meta["usage"] = usage_dict
                    attrs[ATTR_USAGE] = json_attr(usage_dict)
                if resp.output:
                    content_parts = []
                    tool_calls = []
                    for item in resp.output:
                        item_type = getattr(item, "type", None)
                        if item_type == "message":
                            for part in getattr(item, "content", []):
                                if getattr(part, "type", None) == "output_text":
                                    content_parts.append(getattr(part, "text", ""))
                        elif item_type == "function_call":
                            tool_calls.append(
                                {
                                    "name": getattr(item, "name", None),
                                    "arguments": getattr(item, "arguments", None),
                                    "call_id": getattr(item, "call_id", None),
                                }
                            )
                    if content_parts:
                        output_meta["content"] = "\n".join(content_parts)
                        output_meta["role"] = "assistant"
                    if tool_calls:
                        output_meta["tool_calls"] = tool_calls
                if output_meta:
                    attrs[ATTR_MODEL_OUTPUT] = json_attr(output_meta)

            # Input
            raw_input = getattr(data, "input", None)
            if raw_input:
                if isinstance(raw_input, str):
                    attrs[ATTR_MODEL_INPUT] = json_attr(
                        [{"role": "user", "content": raw_input}]
                    )
                elif isinstance(raw_input, list):
                    messages = []
                    for item in raw_input:
                        if isinstance(item, dict):
                            messages.append(item)
                        elif hasattr(item, "model_dump"):
                            messages.append(item.model_dump())
                        else:
                            messages.append({"content": str(item)})
                    attrs[ATTR_MODEL_INPUT] = json_attr(messages)

        elif span_type == "function":
            if hasattr(data, "name") and data.name:
                attrs[ATTR_TOOL_INPUT] = json_attr(
                    {
                        "name": data.name,
                        "arguments": data.input if hasattr(data, "input") else None,
                    }
                )

        elif span_type == "agent":
            name = getattr(data, "name", None)
            if name:
                attrs[ATTR_AGENT_ID] = name
