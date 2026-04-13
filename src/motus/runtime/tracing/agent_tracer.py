"""Agent tracing — thin setup over OpenTelemetry.

Provides setup_tracing() to configure a TracerProvider with the
appropriate SpanProcessors, and get_turn_metrics() for serve workers.
The module-level `tracer` is the standard OTel Tracer used everywhere.
"""

import json
import logging
import os
import webbrowser
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from .config import TraceConfig
from .exporters import OfflineSpanCollector, create_cloud_processor, export_offline
from .live_server import LiveSpanProcessor

logging.basicConfig(
    level=getattr(
        logging, os.environ.get("MOTUS_LOG_LEVEL", "WARNING").upper(), logging.WARNING
    ),
    format="%(asctime)s [%(threadName)s][%(filename)s:%(lineno)d] %(message)s",
)
logger = logging.getLogger("AgentTracer")

# OTel attribute keys for rich metadata.
ATTR_TASK_TYPE = "motus.task_type"
ATTR_FUNC = "motus.func"
ATTR_MODEL_NAME = "motus.model_name"
ATTR_MODEL_OUTPUT = "motus.model_output_meta"
ATTR_MODEL_INPUT = "motus.model_input_meta"
ATTR_TOOL_INPUT = "motus.tool_input_meta"
ATTR_TOOL_OUTPUT = "motus.tool_output_meta"
ATTR_TOOL_META = "motus.tool_meta"
ATTR_USAGE = "motus.usage"
ATTR_AGENT_ID = "motus.agent_id"
ATTR_ERROR = "motus.error"
ATTR_CHOSEN_TOOLS = "motus.chosen_tools"


def json_attr(value) -> str:
    """Serialize a value to JSON string for use as an OTel attribute."""
    if value is None:
        return ""
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# Module-level state (set by setup_tracing)
# ---------------------------------------------------------------------------

_config: TraceConfig | None = None
_collector: OfflineSpanCollector | None = None
_live_processor: LiveSpanProcessor | None = None
_cloud_processor: Any = None


def setup_tracing(config: TraceConfig | None = None) -> trace.Tracer:
    """Configure the OTel TracerProvider with motus SpanProcessors.

    Call once at startup. Returns the tracer to use for creating spans.
    Subsequent calls are no-ops (returns the existing tracer).
    """
    global _config, _collector, _live_processor, _cloud_processor

    # Idempotent — don't reconfigure if already set up
    if _config is not None:
        return trace.get_tracer("motus")

    config = config or TraceConfig()
    _config = config

    # Reuse existing provider if already set (OTel doesn't allow re-setting),
    # otherwise create a new one.
    existing = trace.get_tracer_provider()
    if isinstance(existing, TracerProvider):
        provider = existing
    else:
        provider = TracerProvider()
        trace.set_tracer_provider(provider)

    if config.is_collecting:
        _collector = OfflineSpanCollector()
        provider.add_span_processor(_collector)

    if config.online_tracing and config.is_collecting:
        logger.info("Online tracing enabled - traces will update in real-time")
        _live_processor = LiveSpanProcessor(log_dir=config.log_dir)
        provider.add_span_processor(_live_processor)

    if config.cloud_enabled and config.is_collecting:
        logger.info("Cloud live tracing enabled - spans will stream to cloud")
        otlp_endpoint = config.cloud_api_url.rstrip("/") + "/v1/traces"
        headers = {
            "Authorization": f"Bearer {config.cloud_api_key}",
            "x-motus-project": config.project or "",
            "x-motus-build": config.build or "",
            "x-motus-session": config.session_id or "",
        }
        _cloud_processor = create_cloud_processor(
            endpoint=otlp_endpoint, headers=headers
        )
        provider.add_span_processor(_cloud_processor)

    return trace.get_tracer("motus")


def get_tracer() -> trace.Tracer:
    """Return the motus tracer, auto-initializing if needed."""
    if _config is None:
        return setup_tracing()
    return trace.get_tracer("motus")


# ---------------------------------------------------------------------------
# Serve worker helpers
# ---------------------------------------------------------------------------


def set_session_id(session_id: str) -> None:
    """No-op. Session ID is now set at setup time via OTLP headers."""


def get_turn_metrics() -> dict:
    """Aggregate metrics from the current trace's spans.

    Called by serve workers after an agent turn completes.
    Uses the OfflineSpanCollector's buffered spans.
    """
    if _collector is None or not _collector.spans:
        return {
            "trace_id": None,
            "total_duration": 0.0,
            "total_tokens": 0,
            "has_error": False,
        }

    min_start = float("inf")
    max_end = 0
    total_tokens = 0
    has_error = False

    for span in _collector.spans:
        start_ns = span.start_time or 0
        end_ns = span.end_time or start_ns
        start_us = start_ns // 1000
        end_us = end_ns // 1000

        if start_us <= 0:
            continue
        if start_us < min_start:
            min_start = start_us
        if end_us > max_end:
            max_end = end_us

        attrs = span.attributes or {}
        if attrs.get(ATTR_ERROR):
            has_error = True

        # Parse usage from JSON attribute
        usage_json = attrs.get(ATTR_USAGE) or attrs.get(ATTR_MODEL_OUTPUT)
        if usage_json:
            try:
                data = json.loads(usage_json) if isinstance(usage_json, str) else {}
                usage = data.get("usage", data) if isinstance(data, dict) else {}
                if isinstance(usage, dict):
                    total_tokens += usage.get("total_tokens", 0)
            except (json.JSONDecodeError, TypeError):
                pass

    total_duration = 0.0
    if max_end > min_start:
        total_duration = (max_end - min_start) / 1_000_000

    return {
        "trace_id": None,
        "total_duration": total_duration,
        "total_tokens": total_tokens,
        "has_error": has_error,
    }


def export_trace() -> None:
    """Export collected spans to files and open the viewer."""
    if _config is None or not _config.is_collecting:
        return

    if _live_processor:
        _live_processor.broadcast_finish()
        _live_processor.stop()

    if not _collector or not _collector.spans:
        logger.debug("No spans to export.")
        return

    log_dir = _config.log_dir
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    try:
        export_offline(_collector.spans, log_dir)
        logger.debug(f"Processed {len(_collector.spans)} spans")
    except Exception as e:
        logger.warning(f"Failed to export traces: {e}")

    if not _live_processor:
        html_path = log_dir / "trace_viewer.html"
        webbrowser.open(f"file://{html_path.absolute()}")


def shutdown_tracing() -> None:
    """Flush and shut down all trace processors."""
    global _config, _collector, _live_processor, _cloud_processor

    provider = trace.get_tracer_provider()
    if hasattr(provider, "shutdown"):
        provider.shutdown()

    _config = None
    _collector = None
    _live_processor = None
    _cloud_processor = None


# ---------------------------------------------------------------------------
# Backward-compatible TraceManager shim
# ---------------------------------------------------------------------------
# The old TraceManager maintained its own task_meta dict and hook handlers.
# agent_runtime.py still depends on this interface. This shim preserves
# that interface while the migration to pure OTel spans is completed.
# Framework integrations (anthropic, google_adk, openai_agents) no longer
# use TraceManager — they create OTel spans directly via get_tracer().
# ---------------------------------------------------------------------------

import itertools
import time
import uuid
from contextvars import ContextVar


def _now_us() -> int:
    return int(time.time() * 1_000_000)


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


class TraceManager:
    """Backward-compatible task tracker used by agent_runtime's hook system.

    New framework integrations should use get_tracer() and create OTel spans
    directly instead of calling ingest_external_span().
    """

    def __init__(self, config: TraceConfig | None = None):
        config = config or TraceConfig()
        self.config = config
        self.log_dir = config.log_dir
        self.trace_id = str(uuid.uuid4())

        self.current_task_stack: ContextVar[tuple[int, ...]] = ContextVar(
            "current_task_stack", default=()
        )
        self.task_span_tree: dict[int, list[int]] = {}
        self.task_meta: dict[int, dict] = {}
        self._analytics_callback: Any = None
        self._ext_task_counter = itertools.count(1_000_001)

        # Ensure OTel TracerProvider is set up so hook-generated spans
        # are also captured by OTel processors.
        if config.is_collecting:
            setup_tracing(config)

    @property
    def _cloud_exporter(self):
        """Backward compat for agent_runtime.py shutdown.

        Returns an object with .close() that calls shutdown_tracing().
        """

        class _Closer:
            @staticmethod
            def close():
                shutdown_tracing()

        return _Closer() if _cloud_processor is not None else None

    def export_trace(self) -> None:
        export_trace()

    def close(self) -> None:
        shutdown_tracing()

    def set_session_id(self, session_id: str) -> None:
        set_session_id(session_id)

    def set_analytics_callback(self, callback: Any) -> None:
        self._analytics_callback = callback

    def allocate_external_task_id(self) -> int:
        return next(self._ext_task_counter)

    def get_stack(self) -> tuple[int, ...] | None:
        if not self.config.is_collecting:
            return None
        return self.current_task_stack.get()

    def get_trace_id(self) -> str:
        return self.trace_id

    # -- Hook event handlers (called by agent_runtime) --

    def on_task_start(self, event: Any) -> None:
        if not self.config.is_collecting:
            return
        task_id_int = event.task_id.id
        parent_stack = (
            event.metadata.get("parent_stack") if hasattr(event, "metadata") else None
        )
        stack = (
            parent_stack if parent_stack is not None else self.current_task_stack.get()
        )
        parent = stack[-1] if stack else None

        if parent is not None:
            self.task_span_tree.setdefault(parent, []).append(task_id_int)
        self.current_task_stack.set(stack + (task_id_int,))

        self.task_meta[task_id_int] = {
            "func": event.name,
            "task_type": getattr(event, "task_type", None),
            "parent": parent,
            "started_at": _now_iso(),
            "start_us": _now_us(),
        }

        # Also create an OTel span (started, not yet ended)
        tracer = get_tracer()
        span = tracer.start_span(
            event.name,
            attributes={
                ATTR_FUNC: event.name,
                ATTR_TASK_TYPE: getattr(event, "task_type", None) or "",
            },
        )
        self.task_meta[task_id_int]["_otel_span"] = span

    def on_task_end(self, event: Any) -> None:
        if not self.config.is_collecting:
            return
        task_id_int = event.task_id.id
        if task_id_int not in self.task_meta:
            return

        self.task_meta[task_id_int]["ended_at"] = _now_iso()
        self.task_meta[task_id_int]["end_us"] = _now_us()

        span = self.task_meta[task_id_int].pop("_otel_span", None)
        if span is not None:
            span.end()

        stack = self.current_task_stack.get()
        if stack and stack[-1] == task_id_int:
            self.current_task_stack.set(stack[:-1])

        if self._analytics_callback:
            try:
                self._analytics_callback(
                    task_id_int, self.task_meta[task_id_int], success=True
                )
            except Exception:
                pass

    def on_task_error(self, event: Any) -> None:
        if not self.config.is_collecting:
            return
        task_id_int = event.task_id.id
        if task_id_int not in self.task_meta:
            return

        self.task_meta[task_id_int]["ended_at"] = _now_iso()
        self.task_meta[task_id_int]["end_us"] = _now_us()
        self.task_meta[task_id_int]["error"] = str(getattr(event, "error", ""))

        span = self.task_meta[task_id_int].pop("_otel_span", None)
        if span is not None:
            error_str = str(getattr(event, "error", ""))
            span.set_attribute(ATTR_ERROR, error_str)
            span.set_status(trace.StatusCode.ERROR, error_str)
            span.end()

        stack = self.current_task_stack.get()
        if stack and stack[-1] == task_id_int:
            self.current_task_stack.set(stack[:-1])

        if self._analytics_callback:
            try:
                self._analytics_callback(
                    task_id_int, self.task_meta[task_id_int], success=False
                )
            except Exception:
                pass

    def on_task_cancelled(self, event: Any) -> None:
        if not self.config.is_collecting:
            return
        task_id_int = event.task_id.id
        if task_id_int not in self.task_meta:
            return

        self.task_meta[task_id_int]["ended_at"] = _now_iso()
        self.task_meta[task_id_int]["end_us"] = _now_us()
        self.task_meta[task_id_int]["cancelled"] = True
        self.task_meta[task_id_int]["error"] = str(getattr(event, "error", ""))

        span = self.task_meta[task_id_int].pop("_otel_span", None)
        if span is not None:
            span.set_attribute(ATTR_ERROR, "cancelled")
            span.set_status(trace.StatusCode.ERROR, "cancelled")
            span.end()

        stack = self.current_task_stack.get()
        if stack and stack[-1] == task_id_int:
            self.current_task_stack.set(stack[:-1])

        if self._analytics_callback:
            try:
                self._analytics_callback(
                    task_id_int, self.task_meta[task_id_int], success=False
                )
            except Exception:
                pass

    # -- External span ingestion (legacy interface) --

    def ingest_external_span(self, meta: dict, *, task_id: int | None = None) -> int:
        """Ingest an externally-produced span. Legacy interface -- new code
        should create OTel spans directly via get_tracer().
        """
        if not self.config.is_collecting:
            return -1

        if task_id is None:
            task_id = self.allocate_external_task_id()

        parent = meta.get("parent")
        if parent is not None:
            self.task_span_tree.setdefault(parent, []).append(task_id)

        self.task_meta[task_id] = meta

        if self._analytics_callback:
            try:
                self._analytics_callback(task_id, meta, success="error" not in meta)
            except Exception:
                pass

        return task_id

    def update_external_span(self, task_id: int, updates: dict) -> None:
        """Update an existing external span's metadata."""
        if not self.config.is_collecting:
            return
        meta = self.task_meta.get(task_id)
        if meta is None:
            return
        meta.update(updates)

    def get_turn_metrics(self) -> dict:
        """Return aggregate metrics (delegates to module-level function)."""
        return get_turn_metrics()
