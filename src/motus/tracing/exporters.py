"""Trace export via OTel SpanProcessors and offline file writers.

SpanProcessors receive ReadableSpan objects from the OTel SDK on span end.
Offline export functions take a list of collected ReadableSpans and write
files (JSON state, HTML viewer, Jaeger JSON).

The cloud exporter speaks motus's own trace API (POST /traces → POST
/traces/{id}/spans → POST /traces/{id}/complete), not OTLP.
"""

from __future__ import annotations

import atexit
import json
import logging
import queue
import threading
import urllib.error
import urllib.request
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


class CloudLiveExporter:
    """Pushes span updates to the motus trace API in real-time.

    Spans are buffered in a queue and flushed every ~1 second by a
    background thread, so :meth:`push_span` never blocks the agent.

    Protocol:
        1. First flush: POST /traces → creates trace record (returns trace_id)
        2. Each flush: POST /traces/{trace_id}/spans → batch of span updates
        3. Close: POST /traces/{trace_id}/complete → status = COMPLETED
    """

    _FLUSH_INTERVAL = 1.0  # seconds

    def __init__(
        self,
        api_url: str,
        api_key: str,
        trace_name: str = "",
        project: str | None = None,
        build: str | None = None,
        session_id: str | None = None,
    ):
        self._api_url = api_url.rstrip("/")
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        self._trace_name = trace_name
        self._project = project
        self._build = build
        self._session_id = session_id
        self._cloud_trace_id: str | None = None
        self._closed = False
        # Build a urllib opener that skips macOS system proxy detection.
        # The default urlopen() calls SCDynamicStoreCopyProxiesWithOptions
        # via _scproxy, which uses CoreFoundation — not fork-safe on macOS.
        # In forkserver workers this causes SIGSEGV.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._queue: queue.Queue[tuple[str, dict]] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def set_session_id(self, session_id: str) -> None:
        """Set session_id after construction (for forkserver compatibility)."""
        self._session_id = session_id

    def get_trace_id(self) -> str | None:
        return self._cloud_trace_id

    def push_span(self, span_id: str, meta: dict) -> None:
        """Queue a span update for background upload. Non-blocking."""
        self._queue.put((span_id, meta))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._thread.join(timeout=10)
        self._mark_complete()

    def _flush_loop(self) -> None:
        while not self._stop.is_set():
            self._flush()
            self._stop.wait(self._FLUSH_INTERVAL)
        self._flush()  # final drain

    def _flush(self) -> None:
        items: list[tuple[str, dict]] = []
        while True:
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if not items:
            return

        self._create_trace_record_if_needed()
        if not self._cloud_trace_id:
            return  # trace creation failed, drop spans

        spans = [{"span_id": sid, **meta} for sid, meta in items]
        try:
            payload = json.dumps({"spans": spans}, default=str).encode()
            req = urllib.request.Request(
                f"{self._api_url}/traces/{self._cloud_trace_id}/spans",
                data=payload,
                headers=self._headers,
                method="POST",
            )
            self._opener.open(req, timeout=30)
            logger.debug(f"Flushed {len(spans)} spans to cloud")
        except (urllib.error.URLError, OSError) as e:
            logger.debug(f"Cloud span flush failed (non-fatal): {e}")

    def _mark_complete(self) -> None:
        if not self._cloud_trace_id:
            return
        try:
            req = urllib.request.Request(
                f"{self._api_url}/traces/{self._cloud_trace_id}/complete",
                data=b"{}",
                headers=self._headers,
                method="POST",
            )
            self._opener.open(req, timeout=30)
            logger.debug(f"Marked cloud trace complete: {self._cloud_trace_id}")
        except (urllib.error.URLError, OSError) as e:
            logger.debug(f"Cloud trace complete failed (non-fatal): {e}")

    def _create_trace_record_if_needed(self) -> None:
        if self._cloud_trace_id:
            return
        try:
            body: dict[str, str] = {"name": self._trace_name}
            if self._build:
                body["build_id"] = self._build
            if self._project:
                body["project_id"] = self._project
            if self._session_id:
                body["session_id"] = self._session_id
            payload = json.dumps(body).encode()
            req = urllib.request.Request(
                f"{self._api_url}/traces",
                data=payload,
                headers=self._headers,
                method="POST",
            )
            with self._opener.open(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            self._cloud_trace_id = data.get("trace_id")
            logger.debug(f"Created cloud trace: {self._cloud_trace_id}")
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            logger.warning(f"Cloud trace creation failed (non-fatal): {e}")


class CloudSpanProcessor(SpanProcessor):
    """Adapts ``CloudLiveExporter`` to OTel's SpanProcessor contract.

    Converts each completed ``ReadableSpan`` to motus's viewer dict format
    via :func:`readable_span_to_viewer_dict`, then pushes it onto the
    exporter's async queue. The exporter's background thread handles
    batching, HTTP retries, and trace-record lifecycle.
    """

    def __init__(self, exporter: CloudLiveExporter):
        self._exporter = exporter

    def on_start(self, span, parent_context: Context | None = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        view = readable_span_to_viewer_dict(span)
        span_id = view.get("spanId") or format(span.context.span_id, "016x")
        self._exporter.push_span(span_id, view)

    def shutdown(self) -> None:
        self._exporter.close()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def create_cloud_processor(
    api_url: str,
    api_key: str,
    project: str | None = None,
    build: str | None = None,
    session_id: str | None = None,
) -> CloudSpanProcessor:
    """Build a SpanProcessor that streams spans to motus's trace API."""
    exporter = CloudLiveExporter(
        api_url=api_url,
        api_key=api_key,
        project=project,
        build=build,
        session_id=session_id,
    )
    return CloudSpanProcessor(exporter)


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
