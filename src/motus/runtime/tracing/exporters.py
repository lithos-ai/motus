"""Trace export via OTel SpanProcessors and offline file writers.

SpanProcessors receive ReadableSpan objects from the OTel SDK on span end.
Offline export functions take a list of collected ReadableSpans and write
files (JSON state, HTML viewer, Jaeger JSON).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor

from .span_convert import readable_span_to_viewer_dict

logger = logging.getLogger("AgentTracer")


# ── SpanProcessors ──────────────────────────────────────────────────


class OfflineSpanCollector(SpanProcessor):
    """Collects completed spans in memory for batch export at shutdown."""

    def __init__(self):
        self.spans: list[ReadableSpan] = []

    def on_start(self, span, parent_context: Context | None = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        self.spans.append(span)

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def create_cloud_processor(endpoint: str, headers: dict[str, str] | None = None):
    """Create a BatchSpanProcessor that exports spans via OTLP/HTTP.

    Uses lazy imports so the OTLP exporter dependency is only required
    when cloud tracing is actually enabled.
    """
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
    return BatchSpanProcessor(exporter)


# ── Offline export functions ────────────────────────────────────────


def export_offline(spans: list[ReadableSpan], output_dir: Path) -> None:
    """Export collected spans to JSON state, HTML viewer, and Jaeger JSON."""
    viewer_spans = [readable_span_to_viewer_dict(s) for s in spans]

    # JSON state
    _export_json_state(viewer_spans, output_dir / "tracer_state.json")

    # HTML viewer
    _export_html_viewer(viewer_spans, output_dir / "trace_viewer.html")

    # Jaeger JSON
    _export_jaeger_json(viewer_spans, output_dir / "jaeger_traces.json")


def _export_json_state(viewer_spans: list[dict], output_path: Path) -> None:
    """Export raw span data as JSON."""
    # Convert to task_id-keyed dict for backward compat
    state = {}
    for s in viewer_spans:
        key = s.get("tags", {}).get("task.id", s.get("spanId", ""))
        state[str(key)] = s.get("meta", s)
    with open(output_path, "w") as f:
        json.dump(state, f, indent=4, default=str)
    logger.debug(f"Exported JSON state to {output_path}")


def _export_html_viewer(viewer_spans: list[dict], output_path: Path) -> None:
    """Generate interactive HTML trace viewer."""
    template_dir = Path(__file__).parent / "templates"
    html_template = (template_dir / "trace_viewer.html").read_text()
    css_content = (template_dir / "trace_viewer.css").read_text()
    js_content = (template_dir / "trace_viewer.js").read_text()

    min_time = min((s["startTime"] for s in viewer_spans), default=0)
    max_time = max((s["startTime"] + s["duration"] for s in viewer_spans), default=0)
    total_duration = max_time - min_time

    spans_json = json.dumps(viewer_spans, indent=2, default=str)

    html = html_template.replace("{{CSS_CONTENT}}", css_content)
    html = html.replace("{{JS_CONTENT}}", js_content)
    html = html.replace("{{SPANS_JSON}}", spans_json)
    html = html.replace("{{MIN_TIME}}", str(min_time))
    html = html.replace("{{TOTAL_DURATION}}", str(total_duration))

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Generated trace viewer: {output_path}")


def _export_jaeger_json(viewer_spans: list[dict], output_path: Path) -> None:
    """Export spans in Jaeger-compatible JSON format."""
    if not viewer_spans:
        return

    trace_id = viewer_spans[0].get("traceId", "unknown")

    jaeger_spans = []
    for span in viewer_spans:
        jaeger_span = {
            "traceID": span["traceId"],
            "spanID": span["spanId"],
            "operationName": span["operationName"],
            "references": [],
            "startTime": span["startTime"],
            "duration": span["duration"],
            "tags": [
                {"key": k, "type": "string", "value": str(v)}
                for k, v in span.get("tags", {}).items()
            ],
            "logs": [],
            "processID": "p1",
        }
        if span.get("parentSpanId"):
            jaeger_span["references"].append(
                {
                    "refType": "CHILD_OF",
                    "traceID": span["traceId"],
                    "spanID": span["parentSpanId"],
                }
            )
        jaeger_spans.append(jaeger_span)

    jaeger_data = {
        "data": [
            {
                "traceID": trace_id,
                "spans": jaeger_spans,
                "processes": {"p1": {"serviceName": "agent-runtime", "tags": []}},
            }
        ]
    }

    with open(output_path, "w") as f:
        json.dump(jaeger_data, f, indent=2)
    print(f"Exported Jaeger JSON: {output_path}")
