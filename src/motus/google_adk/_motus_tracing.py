"""MotusSpanProcessor — bridges Google ADK OTEL spans into motus TraceManager.

Google ADK emits OpenTelemetry spans for agent invocations, LLM calls, and
tool executions. This SpanProcessor converts each completed span into motus
task_meta format and calls TraceManager.ingest_external_span() so it appears
in the motus trace viewer, Jaeger export, and analytics pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from motus.runtime.types import MODEL_CALL, TOOL_CALL

logger = logging.getLogger("AgentTracer")

# OTEL semconv attribute keys used by Google ADK
_OP_NAME = "gen_ai.operation.name"
_AGENT_NAME = "gen_ai.agent.name"
_MODEL = "gen_ai.request.model"
_TOOL_NAME = "gen_ai.tool.name"
_TOOL_DESCRIPTION = "gen_ai.tool.description"
_TOOL_TYPE = "gen_ai.tool.type"
_TOOL_CALL_ID = "gen_ai.tool.call.id"
_INPUT_TOKENS = "gen_ai.usage.input_tokens"
_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
_FINISH_REASONS = "gen_ai.response.finish_reasons"
_ERROR_TYPE = "error.type"
_LLM_REQUEST = "gcp.vertex.agent.llm_request"
_LLM_RESPONSE = "gcp.vertex.agent.llm_response"
_TOOL_ARGS = "gcp.vertex.agent.tool_call_args"
_TOOL_RESPONSE = "gcp.vertex.agent.tool_response"


def _span_time_us(span, attr: str) -> int:
    """Extract start/end time from an OTEL ReadableSpan as microseconds since epoch."""
    ns = getattr(span, attr, None)
    if ns is None:
        return 0
    return ns // 1000


def _get_attr(span, key: str, default=None):
    """Safely get a span attribute."""
    attrs = getattr(span, "attributes", None) or {}
    return attrs.get(key, default)


def _parse_json_attr(span, key: str) -> dict | list | None:
    """Parse a JSON-encoded span attribute, returning None on failure."""
    val = _get_attr(span, key)
    if not val or val == "{}":
        return None
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return None


class MotusSpanProcessor:
    """Receives OTEL spans from Google ADK and forwards them to motus TraceManager.

    Usage::

        from google.adk.telemetry.setup import OTelHooks, maybe_set_otel_providers

        processor = MotusSpanProcessor(trace_manager)
        maybe_set_otel_providers([OTelHooks(span_processors=[processor])])
    """

    def __init__(self, trace_manager) -> None:
        self._tm = trace_manager

    def on_start(self, span, parent_context=None) -> None:
        pass  # We process on end when all attributes are populated

    def on_end(self, span) -> None:
        """Convert a completed OTEL span to motus task_meta and ingest."""
        if not self._tm.config.is_collecting:
            return

        op = _get_attr(span, _OP_NAME)
        if op is None:
            return  # Not an ADK span

        task_type, func_name = self._classify(span, op)

        meta: dict[str, Any] = {
            "func": func_name,
            "task_type": task_type,
            "parent": None,  # OTEL parent resolution handled below
            "started_at": "",
            "start_us": _span_time_us(span, "start_time"),
            "ended_at": "",
            "end_us": _span_time_us(span, "end_time"),
            "adk_operation": op,
        }

        # Error
        error_type = _get_attr(span, _ERROR_TYPE)
        if error_type:
            meta["error"] = error_type

        self._enrich_meta(meta, span, op)
        self._tm.ingest_external_span(meta)

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    # ── internal helpers ──

    @staticmethod
    def _classify(span, op: str) -> tuple[str, str]:
        """Return (task_type, func_name) for a given ADK span."""
        if op == "invoke_agent":
            name = _get_attr(span, _AGENT_NAME) or "agent"
            return "agent_call", name
        if op == "generate_content":
            model = _get_attr(span, _MODEL) or "llm"
            return MODEL_CALL, model
        if op == "execute_tool":
            name = _get_attr(span, _TOOL_NAME) or "tool"
            return TOOL_CALL, name
        # Fallback
        name = _get_attr(span, _AGENT_NAME) or op
        return op, name

    @staticmethod
    def _enrich_meta(meta: dict, span, op: str) -> None:
        """Add type-specific fields that motus trace viewer / analytics expect."""
        if op == "generate_content":
            model = _get_attr(span, _MODEL)
            if model:
                meta["model_name"] = model

            usage: dict[str, Any] = {}
            input_tokens = _get_attr(span, _INPUT_TOKENS)
            output_tokens = _get_attr(span, _OUTPUT_TOKENS)
            if input_tokens is not None:
                usage["input_tokens"] = input_tokens
            if output_tokens is not None:
                usage["output_tokens"] = output_tokens
            if input_tokens is not None and output_tokens is not None:
                usage["total_tokens"] = input_tokens + output_tokens
            if usage:
                meta["usage"] = usage

            finish_reasons = _get_attr(span, _FINISH_REASONS)
            if finish_reasons:
                meta["finish_reasons"] = finish_reasons

            output_meta: dict[str, Any] = {}
            if model:
                output_meta["model"] = model
            if usage:
                output_meta["usage"] = usage
            llm_response = _parse_json_attr(span, _LLM_RESPONSE)
            if llm_response:
                output_meta["llm_response"] = llm_response
            if output_meta:
                meta["model_output_meta"] = output_meta

            llm_request = _parse_json_attr(span, _LLM_REQUEST)
            if llm_request:
                meta["model_input_meta"] = llm_request

        elif op == "execute_tool":
            tool_name = _get_attr(span, _TOOL_NAME)
            tool_meta: dict[str, Any] = {}
            if tool_name:
                tool_meta["name"] = tool_name
            tool_args = _parse_json_attr(span, _TOOL_ARGS)
            if tool_args:
                tool_meta["arguments"] = tool_args
            if tool_meta:
                meta["tool_input_meta"] = tool_meta

            tool_response = _parse_json_attr(span, _TOOL_RESPONSE)
            if tool_response:
                meta["tool_output_meta"] = tool_response

        elif op == "invoke_agent":
            agent_name = _get_attr(span, _AGENT_NAME)
            if agent_name:
                meta["agent_name"] = agent_name
