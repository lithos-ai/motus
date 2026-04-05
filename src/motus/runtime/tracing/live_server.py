"""Live trace server with SSE push for real-time trace viewing.

This module provides a LiveTraceServer that maintains span state in memory
and pushes incremental updates to connected browsers via Server-Sent Events.
"""

import http.server
import json
import logging
import queue
import socket
import socketserver
import threading
import time
import webbrowser
from pathlib import Path

from .trace_to_otel import convert_single_span_to_otel

logger = logging.getLogger("AgentTracer")

_EVENT_LOG_MAX = 1000
_CLIENT_QUEUE_MAX = 256
_KEEPALIVE_TIMEOUT = 15  # seconds


def _find_free_port(max_retries: int = 3) -> int:
    """Find a free port on localhost."""
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                s.listen(1)
                port = s.getsockname()[1]
            return port
        except OSError as e:
            last_error = e
            logger.debug(
                f"Failed to find free port (attempt {attempt + 1}/{max_retries}): {e}"
            )
            time.sleep(0.1)
    raise OSError(
        f"Unable to find free port after {max_retries} attempts"
    ) from last_error


class _SSEClient:
    """Represents a connected SSE client with its own event queue."""

    __slots__ = ("queue",)

    def __init__(self):
        self.queue: queue.Queue[str | None] = queue.Queue(maxsize=_CLIENT_QUEUE_MAX)


class LiveTraceServer:
    """Server that pushes trace spans to browsers via SSE.

    Protocol:
        GET /events        → SSE stream (init, span, keepalive events)
        GET /trace_viewer.html → pre-rendered HTML (spans loaded via SSE init)
        GET /*             → 404

    Events:
        init      – full current state on connect (or replay after Last-Event-ID)
        span      – single span upsert (start/end/error)
        keepalive – empty heartbeat every 15s
    """

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir

        self._spans: dict[str, str] = {}  # spanId → serialized JSON string
        self._spans_lock = threading.Lock()

        self._clients: list[_SSEClient] = []
        self._clients_lock = threading.Lock()

        self._event_counter = 0
        self._event_log: list[tuple[int, str]] = []  # (id, serialized SSE data)
        self._event_lock = threading.Lock()

        self._http_server: socketserver.TCPServer | None = None
        self._http_server_port: int | None = None
        self._browser_opened = False

        # Pre-loaded template content (populated in start())
        self._html_content: str | None = None

    @property
    def port(self) -> int | None:
        return self._http_server_port

    @property
    def is_running(self) -> bool:
        return self._http_server is not None

    def start(self, max_retries: int = 5) -> bool:
        """Start the HTTP/SSE server."""
        if self._http_server is not None:
            return True

        # Pre-load template into memory
        try:
            self._html_content = self._build_html()
        except Exception as e:
            logger.warning(f"Failed to load trace viewer template: {e}")
            return False

        server_ref = self  # closure reference for handler

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/events":
                    server_ref._handle_sse(self)
                elif self.path in ("/trace_viewer.html", "/"):
                    self._serve_html()
                else:
                    self.send_error(404)

            def _serve_html(self):
                body = server_ref._html_content.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                pass  # suppress request logging

        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                self._http_server_port = _find_free_port()

                class _QuietTCPServer(socketserver.ThreadingTCPServer):
                    def handle_error(self, request, client_address):
                        # Suppress noisy ConnectionResetError from SSE clients
                        pass

                self._http_server = _QuietTCPServer(
                    ("127.0.0.1", self._http_server_port), _Handler
                )
                self._http_server.daemon_threads = True

                t = threading.Thread(
                    target=self._http_server.serve_forever, daemon=True
                )
                t.start()
                logger.info(f"SSE server started on port {self._http_server_port}")
                return True

            except OSError as e:
                last_error = e
                logger.debug(
                    f"Failed to bind (attempt {attempt + 1}/{max_retries}): {e}"
                )
                self._http_server = None
                self._http_server_port = None
                time.sleep(0.1)

            except Exception as e:
                logger.warning(f"Failed to start SSE server: {e}")
                self._http_server = None
                self._http_server_port = None
                return False

        logger.warning(
            f"Failed to start SSE server after {max_retries} attempts: {last_error}"
        )
        return False

    def broadcast_finish(self) -> None:
        """Broadcast a finish event so browsers stop reconnecting."""
        self._broadcast("finish", "{}")

    def stop(self) -> None:
        """Shut down the server and disconnect all SSE clients."""
        with self._clients_lock:
            for client in self._clients:
                client.queue.put(None)  # sentinel to unblock
            self._clients.clear()

        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server = None
            self._http_server_port = None

    def open_viewer(self) -> str | None:
        """Open the trace viewer in the browser."""
        if self._browser_opened:
            return None

        if self._http_server_port:
            html_url = f"http://localhost:{self._http_server_port}/trace_viewer.html"
        else:
            return None

        try:
            webbrowser.open(html_url)
            self._browser_opened = True
            logger.info(f"Trace viewer opened: {html_url}")
            print(f"\n{'=' * 60}")
            print(f"LIVE TRACING: {html_url}")
            print(f"{'=' * 60}\n")
            return html_url
        except Exception as e:
            logger.warning(f"Failed to open trace viewer: {e}")
            return None

    def push_span(self, task_id_int: int, task_meta_entry: dict, trace_id: str) -> None:
        """Convert a single task to an OTel span and broadcast to all SSE clients."""
        # Open viewer on first span (deferred from start() to avoid racing
        # with other browser tabs that may be loading simultaneously).
        if not self._browser_opened:
            self.open_viewer()

        try:
            span = convert_single_span_to_otel(task_id_int, task_meta_entry, trace_id)
            data = json.dumps(span, default=str)

            with self._spans_lock:
                self._spans[span["spanId"]] = data  # store serialized JSON

            self._broadcast("span", data)
        except Exception as e:
            logger.debug(f"push_span failed for task {task_id_int}: {e}")

    # ── internal ──────────────────────────────────────────────

    def _build_html(self) -> str:
        """Load template files and build the HTML page (spans start empty)."""
        template_dir = Path(__file__).parent / "templates"
        html_template = (template_dir / "trace_viewer.html").read_text(encoding="utf-8")
        css_content = (template_dir / "trace_viewer.css").read_text(encoding="utf-8")
        js_content = (template_dir / "trace_viewer.js").read_text(encoding="utf-8")

        html = html_template.replace("{{CSS_CONTENT}}", css_content)
        html = html.replace("{{JS_CONTENT}}", js_content)
        html = html.replace("{{SPANS_JSON}}", "[]")
        html = html.replace("{{MIN_TIME}}", "0")
        html = html.replace("{{TOTAL_DURATION}}", "0")
        return html

    def _broadcast(self, event_type: str, data: str) -> None:
        """Send an SSE event to all connected clients and append to replay log."""
        with self._event_lock:
            self._event_counter += 1
            event_id = self._event_counter
            frame = f"id: {event_id}\nevent: {event_type}\ndata: {data}\n\n"
            self._event_log.append((event_id, frame))
            if len(self._event_log) > _EVENT_LOG_MAX:
                self._event_log = self._event_log[-_EVENT_LOG_MAX:]

        # Push to all clients (drop slow ones)
        dead: list[_SSEClient] = []
        with self._clients_lock:
            for client in self._clients:
                try:
                    client.queue.put_nowait(frame)
                except queue.Full:
                    dead.append(client)
            for client in dead:
                self._clients.remove(client)

    def _get_init_payload(self) -> str:
        """Build the init event payload (full current state)."""
        with self._spans_lock:
            span_jsons = list(self._spans.values())  # list of JSON strings

        if span_jsons:
            parsed = [json.loads(s) for s in span_jsons]
            min_time = min(s["startTime"] for s in parsed)
            max_time = max(s["startTime"] + s["duration"] for s in parsed)
            total_duration = (max_time - min_time) or 1  # guard against zero
            # Build JSON manually to embed pre-serialized span objects
            spans_array = "[" + ",".join(span_jsons) + "]"
        else:
            min_time = 0
            total_duration = 0
            spans_array = "[]"

        return (
            '{"spans":'
            + spans_array
            + ',"minTime":'
            + str(min_time)
            + ',"totalDuration":'
            + str(total_duration)
            + "}"
        )

    def _handle_sse(self, handler: http.server.BaseHTTPRequestHandler) -> None:
        """Handle an SSE connection: send init/replay then stream events."""
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "keep-alive")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()

        client = _SSEClient()
        with self._clients_lock:
            self._clients.append(client)

        try:
            # Check for Last-Event-ID (reconnect replay)
            last_id_str = handler.headers.get("Last-Event-ID")
            if last_id_str is not None:
                try:
                    last_id = int(last_id_str)
                except ValueError:
                    last_id = 0
                # Replay missed events
                with self._event_lock:
                    replay = [f for eid, f in self._event_log if eid > last_id]
                for f in replay:
                    handler.wfile.write(f.encode("utf-8"))
                handler.wfile.flush()
            else:
                # Fresh connection – send full init
                init_data = self._get_init_payload()
                init_frame = f"event: init\ndata: {init_data}\n\n"
                handler.wfile.write(init_frame.encode("utf-8"))
                handler.wfile.flush()

            # Stream loop
            while True:
                try:
                    frame = client.queue.get(timeout=_KEEPALIVE_TIMEOUT)
                except queue.Empty:
                    # Send keepalive
                    handler.wfile.write(b"event: keepalive\ndata: {}\n\n")
                    handler.wfile.flush()
                    continue

                if frame is None:
                    break  # server shutting down

                handler.wfile.write(frame.encode("utf-8"))
                handler.wfile.flush()

        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # client disconnected
        finally:
            with self._clients_lock:
                if client in self._clients:
                    self._clients.remove(client)
