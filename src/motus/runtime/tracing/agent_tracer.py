"""Agent tracing system built on OpenTelemetry.

This module provides the TraceManager class which wraps an OTel TracerProvider
to track task lifecycle events (start, end, error) and export traces via
pluggable SpanProcessors (live SSE viewer, cloud upload, offline files).
"""

import atexit
import itertools
import json
import logging
import os
import time
import uuid
from contextvars import ContextVar

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from ..hooks import HookEvent
from ..types import AgentTaskId
from .config import TraceConfig
from .exporters import CloudSpanProcessor, OfflineSpanCollector
from .live_server import LiveSpanProcessor

logging.basicConfig(
    level=getattr(
        logging, os.environ.get("MOTUS_LOG_LEVEL", "WARNING").upper(), logging.WARNING
    ),
    format="%(asctime)s [%(threadName)s][%(filename)s:%(lineno)d] %(message)s",
)
logger = logging.getLogger("AgentTracer")


def _now_us() -> int:
    """Return current time as microseconds since epoch."""
    return time.time_ns() // 1000


# OTel attribute keys for rich metadata.
# Complex dicts are JSON-serialized since OTel attributes must be primitives.
ATTR_TASK_TYPE = "motus.task_type"
ATTR_FUNC = "motus.func"
ATTR_PARENT_TASK_ID = "motus.parent_task_id"
ATTR_MODEL_NAME = "motus.model_name"
ATTR_MODEL_OUTPUT = "motus.model_output_meta"
ATTR_MODEL_INPUT = "motus.model_input_meta"
ATTR_TOOL_INPUT = "motus.tool_input_meta"
ATTR_TOOL_OUTPUT = "motus.tool_output_meta"
ATTR_TOOL_META = "motus.tool_meta"
ATTR_USAGE = "motus.usage"
ATTR_AGENT_ID = "motus.agent_id"
ATTR_ERROR = "motus.error"
ATTR_CANCELLED = "motus.cancelled"
ATTR_TASK_ID_INT = "motus.task_id_int"
ATTR_CHOSEN_TOOLS = "motus.chosen_tools"
ATTR_EXTERNAL = "motus.external"


def _json_attr(value) -> str:
    """Serialize a value to JSON string for use as an OTel attribute."""
    if value is None:
        return ""
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


class TraceManager:
    """Manages task tracing via OpenTelemetry.

    Wraps an OTel TracerProvider and provides the same public API that the
    runtime hooks and framework integrations expect.  Internally, all spans
    are native OTel spans.
    """

    def __init__(self, config: TraceConfig | None = None):
        if config is None:
            config = TraceConfig()

        self.config = config
        self.log_dir = config.log_dir
        self.trace_id = str(uuid.uuid4())

        # Task stack for parent tracking
        self.current_task_stack: ContextVar[tuple[int, ...]] = ContextVar(
            "current_task_stack",
            default=(),
        )

        # OTel provider and tracer
        self._provider = TracerProvider()
        self._tracer = self._provider.get_tracer("motus", "0.2.0")

        # Map task_id_int → active OTel Span (for end/error)
        self._active_spans: dict[int, trace.Span] = {}
        # Lightweight metadata for get_turn_metrics() and analytics
        self._span_meta: dict[int, dict] = {}
        self._analytics_callback = None
        self._finalized = False

        # External span ID counter (thread-safe)
        self._ext_task_counter = itertools.count(1_000_001)

        # Span collector for offline export
        self._collector: OfflineSpanCollector | None = None
        if config.is_collecting:
            self._collector = OfflineSpanCollector()
            self._provider.add_span_processor(self._collector)

        # Live SSE server
        self._live_processor: LiveSpanProcessor | None = None
        if config.online_tracing and config.is_collecting:
            logger.info("Online tracing enabled - traces will update in real-time")
            self._live_processor = LiveSpanProcessor(log_dir=config.log_dir)
            self._provider.add_span_processor(self._live_processor)

        # Cloud live streaming
        self._cloud_processor: CloudSpanProcessor | None = None
        if config.cloud_enabled and config.is_collecting:
            logger.info("Cloud live tracing enabled - spans will stream to cloud")
            self._cloud_processor = CloudSpanProcessor(
                api_url=config.cloud_api_url,
                api_key=config.cloud_api_key,
                trace_name=self.trace_id,
                project=config.project,
                build=config.build,
                session_id=config.session_id,
            )
            self._provider.add_span_processor(self._cloud_processor)
            atexit.register(self._finalize_incomplete_spans)

    # -- Backward-compat aliases --

    @property
    def _cloud_exporter(self):
        """Alias so worker.py can call _cloud_exporter.close()."""
        return self._cloud_processor

    @property
    def task_meta(self) -> dict:
        """Lightweight span metadata for backward compat."""
        return self._span_meta

    @property
    def online_tracing(self) -> bool:
        return self._live_processor is not None

    def close(self) -> None:
        self._provider.shutdown()

    def set_session_id(self, session_id: str) -> None:
        if self._cloud_processor is not None:
            self._cloud_processor.set_session_id(session_id)

    def set_analytics_callback(self, callback):
        self._analytics_callback = callback

    def allocate_external_task_id(self) -> int:
        return next(self._ext_task_counter)

    def get_stack(self) -> tuple[int, ...] | None:
        if not self.config.is_collecting:
            return None
        return self.current_task_stack.get()

    def get_trace_id(self) -> str:
        return self.trace_id

    # ------------------------------------------------------------------
    # Task lifecycle (called from runtime hooks)
    # ------------------------------------------------------------------

    def start_task(
        self,
        task_id: AgentTaskId,
        func_name: str,
        parent_stack: tuple[int, ...] | None = None,
        args: tuple | None = None,
        kwargs: dict | None = None,
        task_type: str | None = None,
    ) -> None:
        if not self.config.is_collecting:
            return

        task_id_int = task_id.id
        stack = (
            parent_stack if parent_stack is not None else self.current_task_stack.get()
        )
        parent_id = stack[-1] if stack else None
        self.current_task_stack.set(stack + (task_id_int,))

        # Extract metadata via extractors
        from .extractors import get_extractor

        extractor = get_extractor(task_type or func_name)
        extra_meta = extractor.extract_start_meta(args or (), kwargs or {})
        extractor.on_task_start(task_id_int, parent_id, args or (), kwargs or {})

        # Set up OTel parent context
        ctx = otel_context.get_current()
        if parent_id is not None and parent_id in self._active_spans:
            parent_span = self._active_spans[parent_id]
            ctx = trace.set_span_in_context(parent_span, ctx)

        span = self._tracer.start_span(
            name=func_name,
            context=ctx,
            attributes={
                ATTR_TASK_ID_INT: task_id_int,
                ATTR_FUNC: func_name,
                ATTR_TASK_TYPE: task_type or "",
                ATTR_PARENT_TASK_ID: parent_id if parent_id is not None else -1,
            },
        )
        _set_meta_attributes(span, extra_meta)

        self._active_spans[task_id_int] = span
        self._span_meta[task_id_int] = {
            "func": func_name,
            "task_type": task_type,
            "parent": parent_id,
            "start_us": _now_us(),
            **extra_meta,
        }

    def end_task(self, task_id: AgentTaskId, result=None) -> None:
        if not self.config.is_collecting:
            return

        task_id_int = task_id.id
        span = self._active_spans.pop(task_id_int, None)
        if span is None:
            logger.warning(f"end_task called for unknown task_id: {task_id_int}")
            return

        meta = self._span_meta.get(task_id_int, {})

        from .extractors import get_extractor

        extractor = get_extractor(meta.get("task_type") or meta.get("func", ""))
        extra_meta = extractor.extract_end_meta(result)
        meta.update(extra_meta)
        meta["end_us"] = _now_us()

        _set_meta_attributes(span, extra_meta)
        span.end()

        stack = self.current_task_stack.get()
        if stack and stack[-1] == task_id_int:
            self.current_task_stack.set(stack[:-1])

        if self._analytics_callback:
            try:
                self._analytics_callback(task_id_int, meta, success=True)
            except Exception as e:
                logger.debug(f"Analytics callback failed: {e}")

    def error_task(self, task_id: AgentTaskId, error: Exception) -> None:
        if not self.config.is_collecting:
            return

        task_id_int = task_id.id
        span = self._active_spans.pop(task_id_int, None)
        if span is None:
            logger.warning(f"error_task called for unknown task_id: {task_id_int}")
            return

        meta = self._span_meta.get(task_id_int, {})
        meta["end_us"] = _now_us()
        meta["error"] = str(error)

        from .extractors import get_extractor

        extractor = get_extractor(meta.get("task_type") or meta.get("func", ""))
        extra_meta = extractor.extract_error_meta(error)
        if extra_meta:
            meta.update(extra_meta)

        span.set_attribute(ATTR_ERROR, str(error))
        span.set_status(trace.StatusCode.ERROR, str(error))
        span.record_exception(error)
        span.end()

        stack = self.current_task_stack.get()
        if stack and stack[-1] == task_id_int:
            self.current_task_stack.set(stack[:-1])

        if self._analytics_callback:
            try:
                self._analytics_callback(task_id_int, meta, success=False)
            except Exception as e:
                logger.debug(f"Analytics callback failed: {e}")

    # ------------------------------------------------------------------
    # Hook event handlers
    # ------------------------------------------------------------------

    def on_task_start(self, event: HookEvent) -> None:
        if not self.config.is_collecting:
            return
        self.start_task(
            event.task_id,
            event.name,
            parent_stack=event.metadata.get("parent_stack"),
            args=event.args,
            kwargs=event.kwargs,
            task_type=event.task_type,
        )

    def on_task_end(self, event: HookEvent) -> None:
        if not self.config.is_collecting:
            return
        self.end_task(event.task_id, event.result)

    def on_task_error(self, event: HookEvent) -> None:
        if not self.config.is_collecting:
            return
        self.error_task(event.task_id, event.error)

    def on_task_cancelled(self, event: HookEvent) -> None:
        if not self.config.is_collecting:
            return

        task_id_int = event.task_id.id
        span = self._active_spans.pop(task_id_int, None)
        if span is None:
            return

        meta = self._span_meta.get(task_id_int, {})
        meta["end_us"] = _now_us()
        meta["cancelled"] = True
        meta["error"] = str(event.error)

        span.set_attribute(ATTR_CANCELLED, True)
        span.set_attribute(ATTR_ERROR, str(event.error))
        span.set_status(trace.StatusCode.ERROR, str(event.error))
        span.end()

        stack = self.current_task_stack.get()
        if stack and stack[-1] == task_id_int:
            self.current_task_stack.set(stack[:-1])

        if self._analytics_callback:
            try:
                self._analytics_callback(task_id_int, meta, success=False)
            except Exception as e:
                logger.debug(f"Analytics callback failed: {e}")

    # ------------------------------------------------------------------
    # External span ingestion (framework integrations)
    # ------------------------------------------------------------------

    def ingest_external_span(self, meta: dict, *, task_id: int | None = None) -> int:
        """Ingest an externally-produced span as a real OTel span."""
        if not self.config.is_collecting:
            return -1

        if task_id is None:
            task_id = self.allocate_external_task_id()

        parent = meta.get("parent")

        # Build parent context
        ctx = otel_context.get_current()
        if parent is not None and parent in self._active_spans:
            ctx = trace.set_span_in_context(self._active_spans[parent], ctx)

        func_name = meta.get("func", "external")
        task_type = meta.get("task_type", "")

        start_us = meta.get("start_us", _now_us())
        end_us = meta.get("end_us") or _now_us()

        span = self._tracer.start_span(
            name=func_name,
            context=ctx,
            start_time=start_us * 1000,
            attributes={
                ATTR_TASK_ID_INT: task_id,
                ATTR_FUNC: func_name,
                ATTR_TASK_TYPE: task_type,
                ATTR_PARENT_TASK_ID: parent if parent is not None else -1,
                ATTR_EXTERNAL: True,
            },
        )
        _set_meta_attributes(span, meta)

        if meta.get("error"):
            span.set_attribute(ATTR_ERROR, str(meta["error"]))
            span.set_status(trace.StatusCode.ERROR, str(meta["error"]))

        span.end(end_time=end_us * 1000)

        self._span_meta[task_id] = meta

        if self._analytics_callback:
            try:
                self._analytics_callback(task_id, meta, success="error" not in meta)
            except Exception as e:
                logger.debug(f"Analytics callback failed: {e}")

        return task_id

    def update_external_span(self, task_id: int, updates: dict) -> None:
        """Update metadata for an already-ingested external span.

        OTel spans are immutable once ended, so this only updates the
        internal metadata dict (used by metrics/analytics).
        """
        if not self.config.is_collecting:
            return
        meta = self._span_meta.get(task_id)
        if meta is None:
            return
        meta.update(updates)

    # ------------------------------------------------------------------
    # Finalization & metrics
    # ------------------------------------------------------------------

    def _finalize_incomplete_spans(self) -> None:
        if self._finalized:
            return
        self._finalized = True

        now = _now_us()
        for task_id_int, span in list(self._active_spans.items()):
            meta = self._span_meta.get(task_id_int, {})
            meta["end_us"] = now
            meta["error"] = meta.get("error") or "Process exited before span completed"
            span.set_attribute(ATTR_ERROR, meta["error"])
            span.set_status(trace.StatusCode.ERROR, meta["error"])
            span.end()
        self._active_spans.clear()

    def get_turn_metrics(self) -> dict:
        """Return aggregate metrics for the current trace."""
        min_start = float("inf")
        max_end = 0
        total_tokens = 0
        has_error = False

        for meta in self._span_meta.values():
            start_us = meta.get("start_us", 0)
            if start_us <= 0:
                continue
            end_us = meta.get("end_us") or start_us
            if start_us < min_start:
                min_start = start_us
            if end_us > max_end:
                max_end = end_us
            if meta.get("error"):
                has_error = True
            output = meta.get("model_output_meta")
            if isinstance(output, dict):
                usage = output.get("usage")
                if isinstance(usage, dict):
                    total_tokens += usage.get("total_tokens", 0)

        total_duration = 0.0
        if max_end > min_start:
            total_duration = (max_end - min_start) / 1_000_000

        return {
            "trace_id": self._cloud_processor.get_trace_id()
            if self._cloud_processor
            else None,
            "total_duration": total_duration,
            "total_tokens": total_tokens,
            "has_error": has_error,
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_trace(self) -> None:
        if not self.config.is_collecting:
            return

        if self._live_processor:
            self._live_processor.broadcast_finish()
            self._live_processor.stop()

        if not self._collector or not self._collector.spans:
            logger.debug("No spans to export.")
            return

        import webbrowser

        from .exporters import export_offline

        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True)

        try:
            export_offline(self._collector.spans, self.log_dir)
            logger.debug(f"Processed {len(self._collector.spans)} spans")
        except Exception as e:
            logger.warning(f"Failed to export traces: {e}")

        if not self.online_tracing:
            html_path = self.log_dir / "trace_viewer.html"
            webbrowser.open(f"file://{html_path.absolute()}")


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _set_meta_attributes(span: trace.Span, meta: dict) -> None:
    """Set extracted metadata as OTel span attributes."""
    if meta.get("model_name"):
        span.set_attribute(ATTR_MODEL_NAME, meta["model_name"])
    if meta.get("model_input_meta"):
        span.set_attribute(ATTR_MODEL_INPUT, _json_attr(meta["model_input_meta"]))
    if meta.get("model_output_meta"):
        span.set_attribute(ATTR_MODEL_OUTPUT, _json_attr(meta["model_output_meta"]))
    if meta.get("tool_input_meta"):
        span.set_attribute(ATTR_TOOL_INPUT, _json_attr(meta["tool_input_meta"]))
    if meta.get("tool_output_meta"):
        span.set_attribute(ATTR_TOOL_OUTPUT, _json_attr(meta["tool_output_meta"]))
    if meta.get("tool_meta"):
        span.set_attribute(ATTR_TOOL_META, _json_attr(meta["tool_meta"]))
    if meta.get("usage"):
        span.set_attribute(ATTR_USAGE, _json_attr(meta["usage"]))
    if meta.get("agent_id") or meta.get("agent_name"):
        span.set_attribute(
            ATTR_AGENT_ID, str(meta.get("agent_id") or meta.get("agent_name", ""))
        )
    if meta.get("chosen_tools"):
        span.set_attribute(ATTR_CHOSEN_TOOLS, _json_attr(meta["chosen_tools"]))
