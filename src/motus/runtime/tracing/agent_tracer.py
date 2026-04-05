"""Agent tracing system for tracking task execution.

This module provides the TraceManager class which tracks task lifecycle events
(start, end, error) and can export traces in various formats for visualization.
"""

import atexit
import datetime
import itertools
import logging
import os
import time
import uuid
import webbrowser
from contextvars import ContextVar

from ..hooks import HookEvent
from ..types import AgentTaskId
from .config import TraceConfig
from .exporters import CloudLiveExporter, create_offline_exporter
from .extractors import get_extractor
from .live_server import LiveTraceServer

logging.basicConfig(
    level=getattr(
        logging, os.environ.get("MOTUS_LOG_LEVEL", "WARNING").upper(), logging.WARNING
    ),
    format="%(asctime)s [%(threadName)s][%(filename)s:%(lineno)d] %(message)s",
)
logger = logging.getLogger("AgentTracer")


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _now_us() -> int:
    """Return current time as microseconds since epoch."""
    return time.time_ns() // 1000


class TraceManager:
    """Manages task tracing and trace export.

    The TraceManager tracks task lifecycle events (start, end, error) and maintains
    metadata about each task. It supports both offline batch export and online
    live-updating trace visualization.

    Usage:
        # Basic usage with defaults
        tracer = TraceManager()

        # Custom configuration
        config = TraceConfig(
            enabled=True,
            log_dir="my_traces",
            online_tracing=True,
        )
        tracer = TraceManager(config=config)

        # Track tasks
        tracer.start_task(task_id, "my_function", args=args, kwargs=kwargs)
        tracer.end_task(task_id, result=result)

        # Export traces
        tracer.export_trace()
    """

    def __init__(
        self,
        config: TraceConfig | None = None,
    ):
        """Initialize the TraceManager.

        Args:
            config: Configuration object. If not provided, uses defaults from
                   environment variables (see TraceConfig for details).
        """
        # Handle config
        if config is None:
            config = TraceConfig()

        self.config = config
        self.log_dir = config.log_dir

        # Generate unique trace ID for this TraceManager instance
        self.trace_id = str(uuid.uuid4())

        # Task tracking state
        self.current_task_stack: ContextVar[tuple[int, ...]] = ContextVar(
            "current_task_stack",
            default=(),
        )
        self.task_span_tree: dict[int, list[int]] = {}  # parent -> [children]
        self.task_meta: dict[int, dict] = {}  # task_id -> info
        self._analytics_callback = None  # Optional callback for analytics

        # External span ID counter (for ingest_external_span), thread-safe
        self._ext_task_counter = itertools.count(1_000_001)

        # Online tracing (local browser SSE)
        self._live_server: LiveTraceServer | None = None
        if config.online_tracing and config.is_collecting:
            logger.info("Online tracing enabled - traces will update in real-time")
            self._live_server = LiveTraceServer(log_dir=config.log_dir)
            self._live_server.start()

        # Cloud live streaming (per-span uploads)
        self._cloud_exporter: CloudLiveExporter | None = None
        self._finalized = False
        if config.cloud_enabled and config.is_collecting:
            logger.info("Cloud live tracing enabled - spans will stream to cloud")
            self._cloud_exporter = CloudLiveExporter(
                api_url=config.cloud_api_url,
                api_key=config.cloud_api_key,
                trace_name=self.trace_id,
                project=config.project,
                build=config.build,
            )
            # Register AFTER CloudLiveExporter's atexit so this runs FIRST (LIFO)
            atexit.register(self._finalize_incomplete_spans)

    @property
    def online_tracing(self) -> bool:
        """Whether online tracing is active."""
        return self._live_server is not None

    def close(self) -> None:
        """Flush and shut down all exporters.

        Must be called explicitly in environments where atexit handlers do not
        run (e.g. multiprocessing subprocesses that exit via os._exit).
        """
        if self._cloud_exporter:
            self._cloud_exporter.close()
        if self._live_server:
            self._live_server.stop()

    def _push_span_if_needed(self, task_id_int: int, meta: dict) -> None:
        """Push a span update to all live sinks (local SSE + cloud)."""
        if self._live_server:
            self._live_server.push_span(task_id_int, meta, self.trace_id)
        if self._cloud_exporter:
            self._cloud_exporter.push_span(task_id_int, meta)

    def set_analytics_callback(self, callback):
        """Set callback for analytics integration. Called when tasks complete."""
        self._analytics_callback = callback

    def allocate_external_task_id(self) -> int:
        """Allocate a unique task ID for external spans (thread-safe)."""
        return next(self._ext_task_counter)

    def get_stack(self) -> tuple[int, ...] | None:
        """Get the current task stack for this context."""
        if not self.config.is_collecting:
            return None
        return self.current_task_stack.get()

    def get_trace_id(self) -> str:
        """Get the trace ID for this TraceManager instance.

        Returns:
            The trace ID (UUID) for this TraceManager. All tasks tracked by this
            TraceManager share this trace_id.
        """
        return self.trace_id

    def start_task(
        self,
        task_id: AgentTaskId,
        func_name: str,
        parent_stack: tuple[int, ...] | None = None,
        args: tuple | None = None,
        kwargs: dict | None = None,
        task_type: str | None = None,
    ) -> None:
        """Record the start of a task.

        Args:
            task_id: The task identifier.
            func_name: Name of the function/task being executed (display name).
            parent_stack: Optional explicit parent stack (for cross-context tracing).
            args: Positional arguments passed to the task.
            kwargs: Keyword arguments passed to the task.
            task_type: Task category for extractor matching (e.g. "tool_call").
        """
        if not self.config.is_collecting:
            return

        task_id_int = task_id.id
        stack = (
            parent_stack if parent_stack is not None else self.current_task_stack.get()
        )
        parent = stack[-1] if stack else None

        # Update span tree
        if parent is not None:
            self.task_span_tree.setdefault(parent, []).append(task_id_int)
        self.current_task_stack.set(stack + (task_id_int,))

        # Try task_type first for extractor, fall back to func_name
        extractor = get_extractor(task_type or func_name)
        extra_meta = extractor.extract_start_meta(args or (), kwargs or {})

        # Allow extractor to perform side effects (e.g., debug logging)
        extractor.on_task_start(task_id_int, parent, args or (), kwargs or {})

        self.task_meta[task_id_int] = {
            "func": func_name,
            "task_type": task_type,
            "parent": parent,
            "started_at": _now_iso(),
            "start_us": _now_us(),
            **extra_meta,
        }

        # Push in-progress span for live visibility
        self._push_span_if_needed(task_id_int, self.task_meta[task_id_int])

    def end_task(self, task_id: AgentTaskId, result: any = None) -> None:
        """Record the successful completion of a task.

        Args:
            task_id: The task identifier.
            result: The result returned by the task.
        """
        if not self.config.is_collecting:
            return

        task_id_int = task_id.id

        # Guard against missing task
        if task_id_int not in self.task_meta:
            logger.warning(f"end_task called for unknown task_id: {task_id_int}")
            return

        self.task_meta[task_id_int]["ended_at"] = _now_iso()
        self.task_meta[task_id_int]["end_us"] = _now_us()

        # Get task-specific end metadata from extractor
        meta = self.task_meta[task_id_int]
        extractor = get_extractor(meta.get("task_type") or meta.get("func"))
        extra_meta = extractor.extract_end_meta(result)
        self.task_meta[task_id_int].update(extra_meta)

        # Pop from stack (conditional: deferred task_end may fire in a
        # different async context where this task is not on the stack)
        stack = self.current_task_stack.get()
        if stack and stack[-1] == task_id_int:
            self.current_task_stack.set(stack[:-1])

        # Notify analytics if callback is set
        if self._analytics_callback:
            try:
                self._analytics_callback(
                    task_id_int, self.task_meta[task_id_int], success=True
                )
            except Exception as e:
                logger.debug(f"Analytics callback failed: {e}")

        # Push updated span via SSE + cloud
        self._push_span_if_needed(task_id_int, self.task_meta[task_id_int])

    def error_task(self, task_id: AgentTaskId, error: Exception) -> None:
        """Record a task that ended with an error.

        Args:
            task_id: The task identifier.
            error: The exception that was raised.
        """
        if not self.config.is_collecting:
            return

        task_id_int = task_id.id

        # Guard against missing task
        if task_id_int not in self.task_meta:
            logger.warning(f"error_task called for unknown task_id: {task_id_int}")
            return

        self.task_meta[task_id_int]["ended_at"] = _now_iso()
        self.task_meta[task_id_int]["end_us"] = _now_us()
        self.task_meta[task_id_int]["error"] = str(error)

        # Get task-specific error metadata from extractor
        meta = self.task_meta[task_id_int]
        extractor = get_extractor(meta.get("task_type") or meta.get("func"))
        extra_meta = extractor.extract_error_meta(error)
        if extra_meta:
            self.task_meta[task_id_int].update(extra_meta)

        # Pop from stack (conditional: deferred task_error may fire in a
        # different async context where this task is not on the stack)
        stack = self.current_task_stack.get()
        if stack and stack[-1] == task_id_int:
            self.current_task_stack.set(stack[:-1])

        # Notify analytics if callback is set
        if self._analytics_callback:
            try:
                self._analytics_callback(
                    task_id_int, self.task_meta[task_id_int], success=False
                )
            except Exception as e:
                logger.debug(f"Analytics callback failed: {e}")

        # Push updated span via SSE + cloud
        self._push_span_if_needed(task_id_int, self.task_meta[task_id_int])

    # Hook event handlers

    def on_task_start(self, event: HookEvent) -> None:
        """Handle task start hook event."""
        if not self.config.is_collecting:
            return
        parent_stack = event.metadata.get("parent_stack")
        self.start_task(
            event.task_id,
            event.name,
            parent_stack=parent_stack,
            args=event.args,
            kwargs=event.kwargs,
            task_type=event.task_type,
        )

    def on_task_end(self, event: HookEvent) -> None:
        """Handle task end hook event."""
        if not self.config.is_collecting:
            return
        self.end_task(event.task_id, event.result)

    def on_task_error(self, event: HookEvent) -> None:
        """Handle task error hook event."""
        if not self.config.is_collecting:
            return
        self.error_task(event.task_id, event.error)

    def on_task_cancelled(self, event: HookEvent) -> None:
        """Handle task cancelled hook event."""
        if not self.config.is_collecting:
            return

        task_id_int = event.task_id.id
        if task_id_int not in self.task_meta:
            return

        self.task_meta[task_id_int]["ended_at"] = _now_iso()
        self.task_meta[task_id_int]["end_us"] = _now_us()
        self.task_meta[task_id_int]["cancelled"] = True
        self.task_meta[task_id_int]["error"] = str(event.error)

        stack = self.current_task_stack.get()
        if stack and stack[-1] == task_id_int:
            self.current_task_stack.set(stack[:-1])

        if self._analytics_callback:
            try:
                self._analytics_callback(
                    task_id_int, self.task_meta[task_id_int], success=False
                )
            except Exception as e:
                logger.debug(f"Analytics callback failed: {e}")

        self._push_span_if_needed(task_id_int, self.task_meta[task_id_int])

    # External span ingestion (for OAI SDK TracingProcessor bridge)

    def ingest_external_span(self, meta: dict, *, task_id: int | None = None) -> int:
        """Ingest an externally-produced span (e.g. from OAI SDK TracingProcessor).

        The caller builds a meta dict with the same field names used internally
        (func, task_type, parent, started_at, start_us, ended_at, end_us, error, ...).
        TraceManager stores it like any other span.

        Args:
            meta: Span metadata dictionary. Must include at least 'func' and 'start_us'.
                  If 'parent' is set it must reference an existing task_id_int.
            task_id: Optional pre-allocated task ID. If None, a new one is assigned.

        Returns:
            The task_id_int for this span.
        """
        if not self.config.is_collecting:
            return -1

        if task_id is None:
            task_id = self.allocate_external_task_id()

        parent = meta.get("parent")
        if parent is not None:
            self.task_span_tree.setdefault(parent, []).append(task_id)

        self.task_meta[task_id] = meta

        self._push_span_if_needed(task_id, meta)

        if self._analytics_callback:
            try:
                self._analytics_callback(task_id, meta, success="error" not in meta)
            except Exception as e:
                logger.debug(f"Analytics callback failed: {e}")

        return task_id

    def update_external_span(self, task_id: int, updates: dict) -> None:
        """Update an existing external span's metadata.

        Merges *updates* into the existing meta dict and pushes to live server.
        No-op if the task_id doesn't exist or tracing is disabled.
        """
        if not self.config.is_collecting:
            return
        meta = self.task_meta.get(task_id)
        if meta is None:
            return
        meta.update(updates)
        self._push_span_if_needed(task_id, meta)

    # Finalization

    def _finalize_incomplete_spans(self) -> None:
        """Mark any open spans with an error message before the cloud exporter flushes.

        Called via atexit (runs before CloudLiveExporter.close due to LIFO order).
        Idempotent.
        """
        if self._finalized:
            return
        self._finalized = True

        if not self._cloud_exporter:
            return

        now = _now_us()
        for task_id_int, meta in self.task_meta.items():
            if meta.get("start_us") and not meta.get("end_us"):
                meta["end_us"] = now
                meta["ended_at"] = _now_iso()
                meta["error"] = (
                    meta.get("error") or "Process exited before span completed"
                )
                self._push_span_if_needed(task_id_int, meta)

    # Export methods

    def export_trace(self) -> None:
        """Export traces to files and open the viewer.

        Exports:
        - JSON state file (tracer_state.json)
        - HTML viewer (trace_viewer.html)
        - Jaeger-compatible JSON (jaeger_traces.json)
        """
        if not self.config.is_collecting:
            logger.debug("Tracer is disabled, skipping export.")
            return

        # Always signal finish to live server so browsers stop reconnecting
        if self._live_server:
            self._live_server.broadcast_finish()
            self._live_server.stop()

        if not self.task_meta:
            logger.debug("No tasks to export.")
            return

        # Ensure output directory exists
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True)

        # Use the offline exporter
        try:
            exporter = create_offline_exporter()
            exporter.export(self.task_meta, self.log_dir)
            logger.debug(f"Processed {len(self.task_meta)} tasks")
        except Exception as e:
            logger.warning(f"Failed to export traces: {e}")

        # Open browser if online tracing isn't already showing it
        if not self.online_tracing:
            html_path = self.log_dir / "trace_viewer.html"
            logger.debug("Done! Opening viewer in browser...")
            webbrowser.open(f"file://{html_path.absolute()}")
