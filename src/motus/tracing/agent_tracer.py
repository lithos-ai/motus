"""OpenTelemetry setup, module-level state, and the motus span-content
attribute namespace.

Owns the TracerProvider lifecycle (``setup_tracing`` / ``shutdown_tracing``)
and the handles to the configured SpanProcessors (offline collector, live
viewer, cloud exporter). Every tracing consumer — decorators, framework
bridges, runtime task wrappers, span_convert — reads the ``ATTR_*``
constants defined here so span attribute keys agree across producers.

Runtime-specific concepts (motus task IDs, the current-task ContextVar)
live in :mod:`motus.runtime.tracing`, not here — apps that don't use the
motus runtime have no task tree to represent.
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

# OTel attribute keys for span content (what the span is about).
# Keys that encode the motus *runtime* task tree live in motus.runtime.tracing.
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


def get_config() -> TraceConfig:
    """Return the active :class:`TraceConfig`, auto-initializing on first use."""
    global _config
    if _config is None:
        setup_tracing()
    assert _config is not None  # satisfied by setup_tracing
    return _config


def get_collector() -> OfflineSpanCollector | None:
    """Return the active ``OfflineSpanCollector``, if one is configured."""
    return _collector


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
        _cloud_processor = create_cloud_processor(
            api_url=config.cloud_api_url,
            api_key=config.cloud_api_key,
            project=config.project,
            build=config.build,
            session_id=config.session_id,
        )
        provider.add_span_processor(_cloud_processor)

    return trace.get_tracer("motus")


def set_session_id(session_id: str) -> None:
    """No-op. Session ID is now set at setup time via OTLP headers."""


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


def flush_cloud_trace() -> None:
    """Flush the cloud span buffer and mark the current trace complete.

    Called per-turn by ``motus.serve.worker`` so spans reach the cloud API
    even if the worker subprocess is killed immediately after returning.
    """
    if _cloud_processor is not None:
        _cloud_processor.shutdown()


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
