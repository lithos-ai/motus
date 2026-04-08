"""Trace export strategies.

This module provides a pluggable system for exporting traces to different formats.
Each exporter implements the TraceExporter protocol and can be composed as needed.
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
from typing import Any, Protocol

from .trace_to_otel import (
    convert_to_otel_spans,
    export_jaeger_json,
    generate_html_viewer,
)

logger = logging.getLogger("AgentTracer")


class TraceExporter(Protocol):
    """Protocol for trace exporters."""

    def export(self, task_meta: dict[str, Any], output_dir: Path) -> None:
        """Export trace data to the specified output directory.

        Args:
            task_meta: The task metadata dictionary from TraceManager.
            output_dir: Directory to write output files.
        """
        ...


class JSONStateExporter:
    """Exports raw task metadata as JSON."""

    def __init__(self, filename: str = "tracer_state.json", indent: int = 2):
        self.filename = filename
        self.indent = indent

    def export(self, task_meta: dict[str, Any], output_dir: Path) -> None:
        output_path = output_dir / self.filename
        with open(output_path, "w") as f:
            json.dump(task_meta, f, indent=self.indent)
        logger.debug(f"Exported JSON state to {output_path}")


class HTMLViewerExporter:
    """Exports an interactive HTML trace viewer."""

    def __init__(self, filename: str = "trace_viewer.html", quiet: bool = True):
        self.filename = filename
        self.quiet = quiet

    def export(self, task_meta: dict[str, Any], output_dir: Path) -> None:
        spans = convert_to_otel_spans(task_meta)
        output_path = output_dir / self.filename
        generate_html_viewer(spans, output_path, quiet=self.quiet)
        logger.debug(f"Exported HTML viewer to {output_path}")


class JaegerExporter:
    """Exports traces in Jaeger-compatible JSON format."""

    def __init__(self, filename: str = "jaeger_traces.json", quiet: bool = True):
        self.filename = filename
        self.quiet = quiet

    def export(self, task_meta: dict[str, Any], output_dir: Path) -> None:
        spans = convert_to_otel_spans(task_meta)
        output_path = output_dir / self.filename
        export_jaeger_json(spans, output_path, quiet=self.quiet)
        logger.debug(f"Exported Jaeger JSON to {output_path}")


class CloudLiveExporter:
    """Pushes span updates to the cloud trace API in real-time.

    Spans are buffered in a queue and flushed to the cloud every ~1 second
    by a background thread, so push_span() never blocks the agent.

    Protocol:
        1. First flush: POST /traces → creates trace record (gets trace_id)
        2. Each flush: POST /traces/{trace_id}/spans → batch of span updates
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
        self._queue: queue.Queue[tuple[int, dict]] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._thread.start()
        # Ensure close() runs even if the process crashes or exits unexpectedly
        atexit.register(self.close)

    def set_session_id(self, session_id: str) -> None:
        """Set session_id after construction (for forkserver compatibility)."""
        self._session_id = session_id

    def get_trace_id(self) -> str | None:
        """Public accessor for the cloud trace ID."""
        return self._cloud_trace_id

    def push_span(self, span_id: int, meta: dict) -> None:
        """Queue a span update for background upload. Non-blocking."""
        self._queue.put((span_id, meta.copy()))

    def close(self) -> None:
        """Stop the background thread, flush remaining spans, and mark trace complete."""
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
        items: list[tuple[int, dict]] = []
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

        spans = [{"span_id": str(sid), **meta} for sid, meta in items]
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
        """POST /traces/{id}/complete to transition status to COMPLETED."""
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
        """POST /traces to create the DynamoDB metadata record (once)."""
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


class CompositeExporter:
    """Composes multiple exporters to run in sequence."""

    def __init__(self, exporters: list[TraceExporter]):
        self.exporters = exporters

    def export(self, task_meta: dict[str, Any], output_dir: Path) -> None:
        for exporter in self.exporters:
            exporter.export(task_meta, output_dir)

    def add(self, exporter: TraceExporter) -> "CompositeExporter":
        """Add an exporter and return self for chaining."""
        self.exporters.append(exporter)
        return self


# Pre-configured exporter combinations
def create_offline_exporter() -> CompositeExporter:
    """Create an exporter for offline/batch trace export (local files only)."""
    exporters: list[TraceExporter] = [
        JSONStateExporter(indent=4),
        HTMLViewerExporter(quiet=False),
        JaegerExporter(quiet=False),
    ]
    return CompositeExporter(exporters)
