"""chat_events() coarse-status iterator behavior."""

from __future__ import annotations

import httpx
import pytest

from motus.cloud import AsyncClient, Client, SessionTimeout

from .conftest import echo_server

# ---- sync ----


def test_sync_chat_events_yields_running_then_idle(recorder, fresh_env):
    transport = recorder(echo_server())
    with Client(base_url="http://x", transport=transport) as c:
        events = list(c.chat_events("hi"))
    assert [e.type for e in events] == ["running", "idle"]
    assert events[1].snapshot is not None
    assert events[1].snapshot.response.content == "hi"
    # Ephemeral session cleaned up.
    assert any(r.method == "DELETE" for r in recorder.requests)


def test_sync_chat_events_yields_interrupted_terminal_leaves_session_alive(
    recorder, fresh_env
):
    transport = recorder(
        echo_server(
            interrupts=[{"interrupt_id": "i1", "type": "tool_approval", "payload": {}}]
        )
    )
    with Client(base_url="http://x", transport=transport) as c:
        events = list(c.chat_events("hi"))
    assert [e.type for e in events] == ["running", "interrupted"]
    # No DELETE because interrupted is not a clean idle terminal.
    assert all(r.method != "DELETE" for r in recorder.requests)


def test_sync_chat_events_yields_error_terminal(recorder, fresh_env):
    transport = recorder(echo_server(error="boom"))
    with Client(base_url="http://x", transport=transport) as c:
        events = list(c.chat_events("hi"))
    assert [e.type for e in events] == ["running", "error"]
    assert all(r.method != "DELETE" for r in recorder.requests)


def test_sync_chat_events_deadline_propagates(recorder, fresh_env):
    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST" and req.url.path == "/sessions":
            return httpx.Response(201, json={"session_id": "s1", "status": "idle"})
        if req.method == "POST" and req.url.path.endswith("/messages"):
            return httpx.Response(202, json={"session_id": "s1", "status": "running"})
        return httpx.Response(200, json={"session_id": "s1", "status": "running"})

    transport = recorder(handler)
    with Client(
        base_url="http://x",
        transport=transport,
        turn_timeout=0.05,
        server_wait_slice=0.01,
    ) as c:
        gen = c.chat_events("hi")
        assert next(gen).type == "running"
        with pytest.raises(SessionTimeout):
            list(gen)
    # Session left alive (no DELETE on timeout).
    assert all(r.method != "DELETE" for r in recorder.requests)


def test_sync_chat_events_propagates_extra_headers(recorder, fresh_env):
    transport = recorder(echo_server())
    with Client(base_url="http://x", transport=transport) as c:
        list(c.chat_events("hi", extra_headers={"X-Trace": "t"}))
    for r in recorder.requests:
        assert r.headers["x-trace"] == "t"


# ---- async ----


class _AsyncRecorder:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def wrap(self, handler):
        def inner(req):
            self.requests.append(req)
            return handler(req)

        return httpx.MockTransport(inner)


async def test_async_chat_events_yields_running_then_idle(fresh_env):
    rec = _AsyncRecorder()
    async with AsyncClient(base_url="http://x", transport=rec.wrap(echo_server())) as c:
        types: list[str] = []
        async for ev in c.chat_events("hi"):
            types.append(ev.type)
    assert types == ["running", "idle"]
    assert any(r.method == "DELETE" for r in rec.requests)


async def test_async_chat_events_error_leaves_session_alive(fresh_env):
    rec = _AsyncRecorder()
    async with AsyncClient(
        base_url="http://x", transport=rec.wrap(echo_server(error="boom"))
    ) as c:
        types = [ev.type async for ev in c.chat_events("hi")]
    assert types == ["running", "error"]
    assert all(r.method != "DELETE" for r in rec.requests)
