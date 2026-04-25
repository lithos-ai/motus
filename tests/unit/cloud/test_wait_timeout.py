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


def test_sync_get_session_wait_without_timeout_disables_read_deadline(fresh_env):
    """`wait=True, timeout=None` should NOT keep the default 120s read cap —
    the server-side wait has no upper bound, so the httpx read deadline must
    be disabled (None) for that single call."""
    captured: dict = {}
    transport = httpx.MockTransport(_capture_timeout(captured))
    with Client(base_url="http://x", transport=transport) as c:
        c.get_session("s1", wait=True)
    t = captured["timeout"]
    assert t is not None
    # read=None means "no deadline" in httpx.
    assert t["read"] is None


@pytest.mark.asyncio
async def test_async_get_session_wait_without_timeout_disables_read_deadline(
    fresh_env,
):
    captured: dict = {}
    transport = httpx.MockTransport(_capture_timeout(captured))
    async with AsyncClient(base_url="http://x", transport=transport) as c:
        await c.get_session("s1", wait=True)
    t = captured["timeout"]
    assert t is not None
    assert t["read"] is None


def test_wait_http_timeout_preserves_unbounded_read():
    """If the caller configured read=None (unbounded), wait_http_timeout
    must NOT convert that back to a finite read deadline."""
    from motus.cloud._transport import wait_http_timeout

    unbounded = httpx.Timeout(connect=5.0, read=None, write=10.0, pool=5.0)
    result = wait_http_timeout(unbounded, 300.0)
    assert result.read is None


def test_poll_loop_preserves_injected_unbounded_read(fresh_env):
    """End-to-end: a client with http_timeout.read=None running chat() must
    never impose a finite read deadline on the long-poll GET."""
    captured: list[object] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET" and req.url.path.startswith("/sessions/"):
            captured.append(req.extensions.get("timeout"))
        if req.method == "POST" and req.url.path == "/sessions":
            return httpx.Response(201, json={"session_id": "s1", "status": "idle"})
        if req.method == "POST" and req.url.path.endswith("/messages"):
            return httpx.Response(202, json={"session_id": "s1", "status": "running"})
        if req.method == "GET" and req.url.path.startswith("/sessions/"):
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

    unbounded = httpx.Timeout(connect=5.0, read=None, write=10.0, pool=5.0)
    with Client(
        base_url="http://x",
        transport=httpx.MockTransport(handler),
        http_timeout=unbounded,
    ) as c:
        c.chat("hi")

    # Every poll GET must have used read=None (the injected unbounded value).
    assert captured, "no poll GET was captured"
    for t in captured:
        assert t is not None
        assert t["read"] is None


def test_poll_timeout_respects_turn_timeout(fresh_env, monkeypatch):
    """A stalled poll GET must surface SessionTimeout once the turn deadline
    is exceeded, not BackendUnavailable from retry exhaustion."""
    import motus.cloud._transport as tr
    from motus.cloud import SessionTimeout

    # Synthetic clock: POST happens at t=0, deadline is t=0.5, and the
    # TimeoutException fires when monotonic() returns 1.0 (well past deadline).
    clock = iter([0.0, 0.0, 0.0, 1.0, 1.0])

    def fake_monotonic():
        return next(clock, 1.0)

    monkeypatch.setattr(tr.time, "monotonic", fake_monotonic)

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST" and req.url.path == "/sessions":
            return httpx.Response(201, json={"session_id": "s1", "status": "idle"})
        if req.method == "POST" and req.url.path.endswith("/messages"):
            return httpx.Response(202, json={"session_id": "s1", "status": "running"})
        if req.method == "GET" and req.url.path.startswith("/sessions/"):
            raise httpx.ReadTimeout("stall", request=req)
        if req.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(404)

    with Client(
        base_url="http://x",
        transport=httpx.MockTransport(handler),
        turn_timeout=0.5,
    ) as c:
        with pytest.raises(SessionTimeout):
            c.chat("hi")


@pytest.mark.asyncio
async def test_async_poll_timeout_respects_turn_timeout(fresh_env, monkeypatch):
    import motus.cloud._transport as tr
    from motus.cloud import SessionTimeout

    clock = iter([0.0, 0.0, 0.0, 1.0, 1.0])

    def fake_monotonic():
        return next(clock, 1.0)

    monkeypatch.setattr(tr.time, "monotonic", fake_monotonic)

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST" and req.url.path == "/sessions":
            return httpx.Response(201, json={"session_id": "s1", "status": "idle"})
        if req.method == "POST" and req.url.path.endswith("/messages"):
            return httpx.Response(202, json={"session_id": "s1", "status": "running"})
        if req.method == "GET" and req.url.path.startswith("/sessions/"):
            raise httpx.ReadTimeout("stall", request=req)
        if req.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(404)

    async with AsyncClient(
        base_url="http://x",
        transport=httpx.MockTransport(handler),
        turn_timeout=0.5,
    ) as c:
        with pytest.raises(SessionTimeout):
            await c.chat("hi")


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
