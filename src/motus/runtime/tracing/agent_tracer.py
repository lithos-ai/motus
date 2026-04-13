"""Agent tracing — thin setup over OpenTelemetry.

Provides setup_tracing() to configure a TracerProvider with the
appropriate SpanProcessors, and get_turn_metrics() for serve workers.
The module-level `tracer` is the standard OTel Tracer used everywhere.

Also exposes module-level helpers (allocate_external_task_id,
ingest_external_span, update_external_span, config) used by the
OAI/Anthropic/ADK tracing bridges.
"""

import itertools
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
# External span helpers (used by OAI / Anthropic / ADK tracing bridges)
# ---------------------------------------------------------------------------

_ext_counter = itertools.count(1_000_001)

# Map external task_id -> OTel span + meta dict for later updates
_external_spans: dict[int, dict[str, Any]] = {}


class _ConfigProxy:
    """Lightweight proxy that exposes ``is_collecting`` from module state."""

    @property
    def is_collecting(self) -> bool:
        return _config is not None and _config.is_collecting


config = _ConfigProxy()


def allocate_external_task_id() -> int:
    """Return a unique task_id for an external (non-native) span."""
    return next(_ext_counter)


def ingest_external_span(meta: dict[str, Any], *, task_id: int | None = None) -> None:
    """Create an OTel span from a bridge-supplied meta dict.

    The meta dict uses the same keys as the native motus tracing layer
    (func, task_type, parent, start_us, end_us, model_name, usage, ...).
    """
    if _config is None:
        setup_tracing()

    tracer = trace.get_tracer("motus")
    func_name = meta.get("func", "external")
    task_type = meta.get("task_type", "unknown")

    span = tracer.start_span(func_name)

    # Set standard motus attributes
    span.set_attribute(ATTR_TASK_TYPE, task_type)
    span.set_attribute(ATTR_FUNC, func_name)

    if meta.get("model_name"):
        span.set_attribute(ATTR_MODEL_NAME, meta["model_name"])
    if meta.get("usage"):
        span.set_attribute(ATTR_USAGE, json_attr(meta["usage"]))
    if meta.get("model_output_meta"):
        span.set_attribute(ATTR_MODEL_OUTPUT, json_attr(meta["model_output_meta"]))
    if meta.get("model_input_meta"):
        span.set_attribute(ATTR_MODEL_INPUT, json_attr(meta["model_input_meta"]))
    if meta.get("tool_input_meta"):
        span.set_attribute(ATTR_TOOL_INPUT, json_attr(meta["tool_input_meta"]))
    if meta.get("tool_output_meta"):
        span.set_attribute(ATTR_TOOL_OUTPUT, json_attr(meta["tool_output_meta"]))
    if meta.get("error"):
        span.set_attribute(ATTR_ERROR, str(meta["error"]))

    # End the span immediately (external spans are ingested after completion)
    end_us = meta.get("end_us", 0)
    if end_us > 0:
        span.end(end_time=end_us * 1000)  # OTel expects nanoseconds

    if task_id is not None:
        _external_spans[task_id] = {"span": span, "meta": meta}


def update_external_span(task_id: int, updates: dict[str, Any]) -> None:
    """Update a previously ingested external span (e.g. set end_us)."""
    entry = _external_spans.get(task_id)
    if entry is None:
        return

    span = entry["span"]
    entry["meta"].update(updates)

    if "end_us" in updates and updates["end_us"] > 0:
        span.end(end_time=updates["end_us"] * 1000)
        # Clean up — span is finalized
        _external_spans.pop(task_id, None)
