"""MotusTracingProcessor — bridges OAI SDK tracing into motus TraceManager.

OAI SDK emits trace/span events via TracingProcessor. This processor converts
each completed span into motus task_meta format and calls
TraceManager.ingest_external_span() so it appears in the motus trace viewer,
Jaeger export, and analytics pipeline.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

from agents.tracing.processor_interface import TracingProcessor
from agents.tracing.spans import Span
from agents.tracing.traces import Trace

from motus.runtime.types import MODEL_CALL, TOOL_CALL

logger = logging.getLogger("AgentTracer")


def _iso_to_us(iso_str: str | None) -> int:
    """Convert ISO 8601 timestamp to microseconds since epoch."""
    if not iso_str:
        return 0
    try:
        dt = datetime.datetime.fromisoformat(iso_str)
        return int(dt.timestamp() * 1_000_000)
    except (ValueError, TypeError):
        return 0


class MotusTracingProcessor(TracingProcessor):
    """Receives OAI SDK trace/span events and forwards them to motus TraceManager.

    Usage::

        from agents import add_trace_processor
        processor = MotusTracingProcessor(trace_manager)
        add_trace_processor(processor)
    """

    def __init__(self, trace_manager) -> None:
        self._tm = trace_manager
        # Map OAI span_id → motus task_id_int for parent resolution
        self._span_id_map: dict[str, int] = {}

    def on_trace_start(self, trace: Trace) -> None:
        pass  # TraceManager already has its own trace_id

    def on_trace_end(self, trace: Trace) -> None:
        pass

    def on_span_start(self, span: Span[Any]) -> None:
        if not self._tm.config.is_collecting:
            return
        # Pre-allocate a task_id so child spans can resolve their parent.
        # Child spans end before their parent, so we must register the mapping
        # here (not on_span_end) for parent lookups to succeed.
        self._span_id_map[span.span_id] = self._tm.allocate_external_task_id()

    def on_span_end(self, span: Span[Any]) -> None:
        """Convert completed OAI span to motus task_meta and ingest."""
        if not self._tm.config.is_collecting:
            return

        data = span.span_data
        exported = data.export() if data else {}
        span_type = data.type if data else "unknown"

        # Map OAI span type → motus task_type + display name
        task_type, func_name = self._classify(span_type, data)

        # Resolve parent via span_id mapping (registered in on_span_start)
        parent = self._span_id_map.get(span.parent_id) if span.parent_id else None

        # Use the task_id pre-allocated in on_span_start
        task_id_int = self._span_id_map.get(span.span_id)
        if task_id_int is None:
            return  # Should not happen, but guard

        meta: dict[str, Any] = {
            "func": func_name,
            "task_type": task_type,
            "parent": parent,
            "started_at": span.started_at or "",
            "start_us": _iso_to_us(span.started_at),
            "ended_at": span.ended_at or "",
            "end_us": _iso_to_us(span.ended_at),
            "oai_span_type": span_type,
            "oai_span_data": exported,
        }

        if span.error:
            meta["error"] = span.error.get("message", str(span.error))

        # Add type-specific fields that motus trace viewer / analytics expect
        self._enrich_meta(meta, span_type, data)

        self._tm.ingest_external_span(meta, task_id=task_id_int)

        # Clean up span_id mapping to avoid unbounded growth
        del self._span_id_map[span.span_id]

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass

    # ── internal helpers ──

    @staticmethod
    def _classify(span_type: str, data: Any) -> tuple[str, str]:
        """Return (task_type, func_name) for a given OAI span type."""
        if span_type == "generation":
            model = getattr(data, "model", None) or "llm"
            return MODEL_CALL, model
        if span_type == "response":
            # Responses API — data.response is a Response object
            resp = getattr(data, "response", None)
            model = getattr(resp, "model", None) or "llm"
            return MODEL_CALL, model
        if span_type == "function":
            name = getattr(data, "name", None) or "tool"
            return TOOL_CALL, name
        if span_type == "agent":
            name = getattr(data, "name", None) or "agent"
            return "agent_call", name
        if span_type == "handoff":
            to_agent = getattr(data, "to_agent", None) or "?"
            return "handoff", f"handoff→{to_agent}"
        if span_type == "guardrail":
            name = getattr(data, "name", None) or "guardrail"
            return "guardrail", name
        # custom, mcp_tools, etc.
        name = getattr(data, "name", None) or span_type
        return span_type, name

    @staticmethod
    def _enrich_meta(meta: dict, span_type: str, data: Any) -> None:
        """Add type-specific fields that motus trace viewer / analytics expect."""
        if span_type == "generation":
            if hasattr(data, "model") and data.model:
                meta["model_name"] = data.model
            # model_output_meta: trace viewer expects .content / .tool_calls / .usage
            output_meta: dict[str, Any] = {}
            if hasattr(data, "model") and data.model:
                output_meta["model"] = data.model
            if hasattr(data, "usage") and data.usage:
                output_meta["usage"] = data.usage
            if hasattr(data, "output") and data.output:
                # data.output is list[dict] (e.g. [message.model_dump()])
                # trace viewer looks for .choices[0].message or top-level .content
                first = data.output[0] if data.output else {}
                output_meta["content"] = first.get("content")
                output_meta["tool_calls"] = first.get("tool_calls")
                output_meta["role"] = first.get("role")
            if output_meta:
                meta["model_output_meta"] = output_meta
            if hasattr(data, "input") and data.input:
                # OAI SDK tool messages lack 'name' field — resolve from
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
                meta["model_input_meta"] = messages
        elif span_type == "response":
            # Responses API — data is ResponseSpanData with .response (Response) and .input
            resp = getattr(data, "response", None)
            if resp is not None:
                meta["model_name"] = getattr(resp, "model", None)
                # Extract output from Response.output (list of output items)
                output_meta: dict[str, Any] = {}
                if resp.model:
                    output_meta["model"] = resp.model
                if resp.usage:
                    output_meta["usage"] = {
                        "input_tokens": resp.usage.input_tokens,
                        "output_tokens": resp.usage.output_tokens,
                        "total_tokens": resp.usage.total_tokens,
                    }
                    meta["usage"] = output_meta["usage"]
                # Extract text content and tool calls from Response.output items
                if resp.output:
                    content_parts = []
                    tool_calls = []
                    for item in resp.output:
                        item_type = getattr(item, "type", None)
                        if item_type == "message":
                            # ResponseOutputMessage — has .content list
                            for part in getattr(item, "content", []):
                                if getattr(part, "type", None) == "output_text":
                                    content_parts.append(getattr(part, "text", ""))
                        elif item_type == "function_call":
                            # ResponseFunctionToolCall
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
                    meta["model_output_meta"] = output_meta
            # Extract input (ResponseSpanData.input)
            raw_input = getattr(data, "input", None)
            if raw_input:
                if isinstance(raw_input, str):
                    meta["model_input_meta"] = [{"role": "user", "content": raw_input}]
                elif isinstance(raw_input, list):
                    # list of ResponseInputItemParam dicts or objects
                    messages = []
                    for item in raw_input:
                        if isinstance(item, dict):
                            messages.append(item)
                        elif hasattr(item, "model_dump"):
                            messages.append(item.model_dump())
                        else:
                            messages.append({"content": str(item)})
                    meta["model_input_meta"] = messages
        elif span_type == "function":
            if hasattr(data, "name") and data.name:
                meta["tool_input_meta"] = {
                    "name": data.name,
                    "arguments": data.input if hasattr(data, "input") else None,
                }
