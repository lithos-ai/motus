"""get_session(wait=True, timeout=N) overrides httpx read timeout when N > default."""

from __future__ import annotations

import httpx
import pytest

from motus.cloud import AsyncClient, Client


def _capture_timeout(captured: dict):
    def handler(req: httpx.Request) -> httpx.Response:
        # httpx attaches the effective timeout to the request extensions.
        captured["timeout"] = req.extensions.get("timeout")
        return httpx.Response(
            200, json={"session_id": req.url.path.rsplit("/", 1)[-1], "status": "idle"}
        )

    return handler


def test_sync_get_session_wait_overrides_read_timeout(fresh_env):
    captured: dict = {}
    transport = httpx.MockTransport(_capture_timeout(captured))
    with Client(base_url="http://x", transport=transport) as c:
        c.get_session("s1", wait=True, timeout=300.0)
    t = captured["timeout"]
    assert t is not None
    assert t["read"] >= 300.0


def test_sync_get_session_without_wait_uses_client_default(fresh_env):
    captured: dict = {}
    transport = httpx.MockTransport(_capture_timeout(captured))
    with Client(base_url="http://x", transport=transport) as c:
        c.get_session("s1")
    t = captured["timeout"]
    # Default httpx client timeout is 120s; do not inflate on plain GETs.
    assert t is not None
    assert t["read"] == 120.0


def test_sync_get_session_wait_shorter_than_default_does_not_shrink(fresh_env):
    captured: dict = {}
    transport = httpx.MockTransport(_capture_timeout(captured))
    with Client(base_url="http://x", transport=transport) as c:
        c.get_session("s1", wait=True, timeout=5.0)
    t = captured["timeout"]
    assert t is not None
    # Do NOT go below the client's default read timeout for short waits.
    assert t["read"] >= 120.0


@pytest.mark.asyncio
async def test_async_get_session_wait_overrides_read_timeout(fresh_env):
    captured: dict = {}
    transport = httpx.MockTransport(_capture_timeout(captured))
    async with AsyncClient(base_url="http://x", transport=transport) as c:
        await c.get_session("s1", wait=True, timeout=300.0)
    t = captured["timeout"]
    assert t is not None
    assert t["read"] >= 300.0


def test_sync_poll_loop_uses_override_for_long_wait_slices(fresh_env):
    """server_wait_slice larger than http_timeout.read must NOT trip the
    client — each poll GET has its read deadline raised to cover the wait."""
    poll_timeouts: list[float] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST" and req.url.path == "/sessions":
            return httpx.Response(201, json={"session_id": "s1", "status": "idle"})
        if req.method == "POST" and req.url.path.endswith("/messages"):
            return httpx.Response(202, json={"session_id": "s1", "status": "running"})
        if req.method == "GET" and req.url.path.startswith("/sessions/"):
            t = req.extensions.get("timeout") or {}
            poll_timeouts.append(float(t.get("read") or 0.0))
            return httpx.Response(
                200,
                json={
                    "session_id": "s1",
                    "status": "idle",
                    "response": {"role": "assistant", "content": "ok"},
                },
            )
        if req.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(404)

    with Client(
        base_url="http://x",
        transport=httpx.MockTransport(handler),
        http_timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
        server_wait_slice=200.0,  # deliberately larger than the 30s read
    ) as c:
        c.chat("hi")

    assert poll_timeouts, "poll GET was never invoked"
    # Every poll GET should have a read timeout at least as large as the wait slice.
    assert all(rt >= 200.0 for rt in poll_timeouts), poll_timeouts


@pytest.mark.asyncio
async def test_async_poll_loop_uses_override_for_long_wait_slices(fresh_env):
    poll_timeouts: list[float] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST" and req.url.path == "/sessions":
            return httpx.Response(201, json={"session_id": "s1", "status": "idle"})
        if req.method == "POST" and req.url.path.endswith("/messages"):
            return httpx.Response(202, json={"session_id": "s1", "status": "running"})
        if req.method == "GET" and req.url.path.startswith("/sessions/"):
            t = req.extensions.get("timeout") or {}
            poll_timeouts.append(float(t.get("read") or 0.0))
            return httpx.Response(
                200,
                json={
                    "session_id": "s1",
                    "status": "idle",
                    "response": {"role": "assistant", "content": "ok"},
                },
            )
        if req.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(404)

    async with AsyncClient(
        base_url="http://x",
        transport=httpx.MockTransport(handler),
        http_timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
        server_wait_slice=200.0,
    ) as c:
        await c.chat("hi")

    assert poll_timeouts
    assert all(rt >= 200.0 for rt in poll_timeouts), poll_timeouts
