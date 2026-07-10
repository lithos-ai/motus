"""Live trace server with SSE push — implemented as an OTel SpanProcessor.

Receives ReadableSpan objects from the OTel SDK on span end, converts them
to the viewer dict format, and broadcasts to connected browsers via SSE.
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

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor

from .span_convert import readable_span_to_viewer_dict

logger = logging.getLogger("AgentTracer")

_EVENT_LOG_MAX = 1000
_CLIENT_QUEUE_MAX = 256
_KEEPALIVE_TIMEOUT = 15  # seconds


def _find_free_port(max_retries: int = 3) -> int:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                s.listen(1)
                return s.getsockname()[1]
        except OSError as e:
            last_error = e
            time.sleep(0.1)
    raise OSError(
        f"Unable to find free port after {max_retries} attempts"
    ) from last_error


class _SSEClient:
    __slots__ = ("queue",)

    def __init__(self):
        self.queue: queue.Queue[str | None] = queue.Queue(maxsize=_CLIENT_QUEUE_MAX)


class LiveSpanProcessor(SpanProcessor):
    """OTel SpanProcessor that broadcasts spans to browsers via SSE.

    Also serves the trace viewer HTML page.  Replaces the old
    LiveTraceServer + convert_single_span_to_otel() pipeline.
    """

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir

        self._spans: dict[str, str] = {}  # spanId → serialized JSON
        self._spans_lock = threading.Lock()

        self._clients: list[_SSEClient] = []
        self._clients_lock = threading.Lock()

        self._event_counter = 0
        self._event_log: list[tuple[int, str]] = []
        self._event_lock = threading.Lock()

        self._http_server: socketserver.TCPServer | None = None
        self._http_server_port: int | None = None
        self._browser_opened = False
        self._html_content: str | None = None

        self._start_server()

    # -- SpanProcessor interface --

    def on_start(self, span, parent_context: Context | None = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        """Convert span and broadcast to SSE clients."""
        if not self._browser_opened:
            self._open_viewer()

        try:
            viewer_dict = readable_span_to_viewer_dict(span)
            data = json.dumps(viewer_dict, default=str)

            with self._spans_lock:
                self._spans[viewer_dict["spanId"]] = data

            self._broadcast("span", data)
        except Exception as e:
            logger.debug(f"LiveSpanProcessor.on_end failed: {e}")

    def shutdown(self) -> None:
        self.stop()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    # -- Public API --

    @property
    def port(self) -> int | None:
        return self._http_server_port

    def broadcast_finish(self) -> None:
        self._broadcast("finish", "{}")

    def stop(self) -> None:
        with self._clients_lock:
            for client in self._clients:
                client.queue.put(None)
            self._clients.clear()

        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server = None
            self._http_server_port = None

    # -- Internal --

    def _start_server(self, max_retries: int = 5) -> bool:
        try:
            self._html_content = self._build_html()
        except Exception as e:
            logger.warning(f"Failed to load trace viewer template: {e}")
            return False

        server_ref = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/events":
                    server_ref._handle_sse(self)
                elif self.path in ("/trace_viewer.html", "/"):
                    body = server_ref._html_content.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_error(404)

            def log_message(self, format, *args):
                pass

        for attempt in range(max_retries):
            try:
                self._http_server_port = _find_free_port()

                class _QuietTCPServer(socketserver.ThreadingTCPServer):
                    def handle_error(self, request, client_address):
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
            except OSError:
                self._http_server = None
                self._http_server_port = None
                time.sleep(0.1)
            except Exception as e:
                logger.warning(f"Failed to start SSE server: {e}")
                return False

        return False

    def _open_viewer(self) -> None:
        if self._browser_opened or not self._http_server_port:
            return
        url = f"http://localhost:{self._http_server_port}/trace_viewer.html"
        try:
            webbrowser.open(url)
            self._browser_opened = True
            logger.info(f"Trace viewer opened: {url}")
            print(f"\n{'=' * 60}")
            print(f"LIVE TRACING: {url}")
            print(f"{'=' * 60}\n")
        except Exception as e:
            logger.warning(f"Failed to open trace viewer: {e}")

    def _build_html(self) -> str:
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
        with self._event_lock:
            self._event_counter += 1
            event_id = self._event_counter
            frame = f"id: {event_id}\nevent: {event_type}\ndata: {data}\n\n"
            self._event_log.append((event_id, frame))
            if len(self._event_log) > _EVENT_LOG_MAX:
                self._event_log = self._event_log[-_EVENT_LOG_MAX:]

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
        with self._spans_lock:
            span_jsons = list(self._spans.values())

        if span_jsons:
            parsed = [json.loads(s) for s in span_jsons]
            min_time = min(s["startTime"] for s in parsed)
            max_time = max(s["startTime"] + s["duration"] for s in parsed)
            total_duration = (max_time - min_time) or 1
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
            last_id_str = handler.headers.get("Last-Event-ID")
            if last_id_str is not None:
                try:
                    last_id = int(last_id_str)
                except ValueError:
                    last_id = 0
                with self._event_lock:
                    replay = [f for eid, f in self._event_log if eid > last_id]
                for f in replay:
                    handler.wfile.write(f.encode("utf-8"))
                handler.wfile.flush()
            else:
                init_data = self._get_init_payload()
                init_frame = f"event: init\ndata: {init_data}\n\n"
                handler.wfile.write(init_frame.encode("utf-8"))
                handler.wfile.flush()

            while True:
                try:
                    frame = client.queue.get(timeout=_KEEPALIVE_TIMEOUT)
                except queue.Empty:
                    handler.wfile.write(b"event: keepalive\ndata: {}\n\n")
                    handler.wfile.flush()
                    continue

                if frame is None:
                    break

                handler.wfile.write(frame.encode("utf-8"))
                handler.wfile.flush()

        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with self._clients_lock:
                if client in self._clients:
                    self._clients.remove(client)
