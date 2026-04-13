"""MotusSpanProcessor -- bridges Google ADK OTel spans into motus tracing.

Google ADK emits OpenTelemetry spans with gen_ai.* semconv attributes.
This SpanProcessor re-emits each completed ADK span on the motus tracer
with motus.* attributes so it flows through our SpanProcessors
(OfflineSpanCollector, LiveSpanProcessor, CloudSpanProcessor) and
appears in the trace viewer, Jaeger export, and analytics pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor

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

logger = logging.getLogger("AgentTracer")

# OTel semconv attribute keys used by Google ADK
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


def _get_attr(span: ReadableSpan, key: str, default: Any = None) -> Any:
    """Safely get a span attribute."""
    attrs = span.attributes or {}
    return attrs.get(key, default)


def _parse_json_attr(span: ReadableSpan, key: str) -> dict | list | None:
    """Parse a JSON-encoded span attribute, returning None on failure."""
    val = _get_attr(span, key)
    if not val or val == "{}":
        return None
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return None


class MotusSpanProcessor(SpanProcessor):
    """Receives OTel spans from Google ADK and re-emits them on the motus tracer.

    Instead of building dicts and calling ingest_external_span(), this
    creates new OTel spans on the motus tracer with motus.* attributes.
    Those spans then flow through our configured SpanProcessors
    (OfflineSpanCollector, LiveSpanProcessor, CloudSpanProcessor).

    Usage::

        processor = MotusSpanProcessor()
        # Add to ADK's TracerProvider or motus's TracerProvider
        provider.add_span_processor(processor)
    """

    def on_start(
        self, span: ReadableSpan, parent_context: Context | None = None
    ) -> None:
        pass  # We process on end when all attributes are populated

    def on_end(self, span: ReadableSpan) -> None:
        """Re-emit a completed ADK span on the motus tracer with motus attributes."""
        op = _get_attr(span, _OP_NAME)
        if op is None:
            return  # Not an ADK span

        task_type, func_name = self._classify(span, op)
        tracer = get_tracer()

        # Build motus attributes
        attrs: dict[str, Any] = {
            ATTR_FUNC: func_name,
            ATTR_TASK_TYPE: task_type,
        }

        # Error
        error_type = _get_attr(span, _ERROR_TYPE)
        if error_type:
            attrs[ATTR_ERROR] = str(error_type)

        # Type-specific enrichment
        self._enrich_attrs(attrs, span, op)

        # Create a new span on the motus tracer, preserving original timing.
        # Use start_time/end from the ADK span (nanoseconds).
        start_ns = span.start_time or 0
        end_ns = span.end_time or start_ns

        motus_span = tracer.start_span(
            func_name,
            attributes=attrs,
            start_time=start_ns,
        )

        if error_type:
            motus_span.set_status(trace.StatusCode.ERROR, str(error_type))

        motus_span.end(end_time=end_ns)

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    # -- internal helpers --

    @staticmethod
    def _classify(span: ReadableSpan, op: str) -> tuple[str, str]:
        """Return (task_type, func_name) for a given ADK span."""
        if op == "invoke_agent":
            name = _get_attr(span, _AGENT_NAME) or "agent"
            return "agent_call", name
        if op == "generate_content":
            model = _get_attr(span, _MODEL) or "llm"
            return "model_call", model
        if op == "execute_tool":
            name = _get_attr(span, _TOOL_NAME) or "tool"
            return "tool_call", name
        # Fallback
        name = _get_attr(span, _AGENT_NAME) or op
        return op, name

    @staticmethod
    def _enrich_attrs(attrs: dict, span: ReadableSpan, op: str) -> None:
        """Add type-specific motus attributes."""
        if op == "generate_content":
            model = _get_attr(span, _MODEL)
            if model:
                attrs[ATTR_MODEL_NAME] = model

            # Usage
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
                attrs[ATTR_USAGE] = json_attr(usage)

            # Model output
            output_meta: dict[str, Any] = {}
            if model:
                output_meta["model"] = model
            if usage:
                output_meta["usage"] = usage
            llm_response = _parse_json_attr(span, _LLM_RESPONSE)
            if llm_response:
                output_meta["llm_response"] = llm_response
            if output_meta:
                attrs[ATTR_MODEL_OUTPUT] = json_attr(output_meta)

            # Model input
            llm_request = _parse_json_attr(span, _LLM_REQUEST)
            if llm_request:
                attrs[ATTR_MODEL_INPUT] = json_attr(llm_request)

        elif op == "execute_tool":
            tool_name = _get_attr(span, _TOOL_NAME)
            tool_meta: dict[str, Any] = {}
            if tool_name:
                tool_meta["name"] = tool_name
            tool_args = _parse_json_attr(span, _TOOL_ARGS)
            if tool_args:
                tool_meta["arguments"] = tool_args
            if tool_meta:
                attrs[ATTR_TOOL_INPUT] = json_attr(tool_meta)

            tool_response = _parse_json_attr(span, _TOOL_RESPONSE)
            if tool_response:
                attrs[ATTR_TOOL_OUTPUT] = json_attr(tool_response)

        elif op == "invoke_agent":
            agent_name = _get_attr(span, _AGENT_NAME)
            if agent_name:
                attrs[ATTR_FUNC] = agent_name
