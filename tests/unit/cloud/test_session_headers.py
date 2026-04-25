"""Session-scoped extra_headers propagate through the full session lifecycle."""

from __future__ import annotations

import httpx
import pytest

from motus.cloud import AsyncClient, Client

from .conftest import echo_server


def test_sync_session_extra_headers_on_every_request(recorder, fresh_env):
    transport = recorder(echo_server())
    with Client(base_url="http://x", transport=transport) as c:
        with c.session(extra_headers={"X-Tenant": "acme"}) as s:
            s.chat("hi")
    # Every request (POST /sessions, POST /messages, GET poll, DELETE) carries X-Tenant.
    for r in recorder.requests:
        assert r.headers["x-tenant"] == "acme", (
            f"{r.method} {r.url.path} missing header"
        )
    assert {r.method for r in recorder.requests} >= {"POST", "GET", "DELETE"}


def test_sync_session_per_call_headers_merge_over_session(recorder, fresh_env):
    transport = recorder(echo_server())
    with Client(
        base_url="http://x", transport=transport, extra_headers={"X-Base": "c"}
    ) as c:
        with c.session(extra_headers={"X-Tenant": "acme"}) as s:
            s.chat("hi", extra_headers={"X-Tenant": "override", "X-Trace": "t"})
    # The per-call X-Tenant overrides the session's; X-Trace is injected per-call.
    turn = [r for r in recorder.requests if r.url.path.endswith("/messages")][0]
    assert turn.headers["x-tenant"] == "override"
    assert turn.headers["x-trace"] == "t"
    assert turn.headers["x-base"] == "c"


def test_sync_session_headers_reach_resume(recorder, fresh_env):
    """Session-scoped headers land on POST /resume and the subsequent poll."""
    state = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        m, p = req.method, req.url.path
        if m == "POST" and p == "/sessions":
            return httpx.Response(201, json={"session_id": "s1", "status": "idle"})
        if m == "POST" and p.endswith("/messages"):
            state["n"] = 0
            return httpx.Response(202, json={"session_id": "s1", "status": "running"})
        if m == "POST" and p.endswith("/resume"):
            state["n"] = 99
            return httpx.Response(200, json={"session_id": "s1", "status": "running"})
        if m == "GET" and p.startswith("/sessions/"):
            if state["n"] >= 99:
                return httpx.Response(
                    200,
                    json={
                        "session_id": "s1",
                        "status": "idle",
                        "response": {"role": "assistant", "content": "done"},
                    },
                )
            return httpx.Response(
                200,
                json={
                    "session_id": "s1",
                    "status": "interrupted",
                    "interrupts": [{"interrupt_id": "i1", "type": "t", "payload": {}}],
                },
            )
        if m == "DELETE":
            return httpx.Response(204)
        return httpx.Response(404)

    transport = recorder(handler)
    with Client(base_url="http://x", transport=transport) as c:
        with c.session(extra_headers={"X-Tenant": "acme"}) as s:
            r1 = s.chat("pick")
            assert r1.status.value == "interrupted"
            s.resume("i1", "blue")
    resume_req = [r for r in recorder.requests if r.url.path.endswith("/resume")][0]
    assert resume_req.headers["x-tenant"] == "acme"


@pytest.mark.asyncio
async def test_async_session_extra_headers_on_every_request(fresh_env):
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return echo_server()(req)

    transport = httpx.MockTransport(handler)
    async with AsyncClient(base_url="http://x", transport=transport) as c:
        async with c.session(extra_headers={"X-Tenant": "acme"}) as s:
            await s.chat("hi")
    for r in seen:
        assert r.headers["x-tenant"] == "acme", (
            f"{r.method} {r.url.path} missing header"
        )
    assert {r.method for r in seen} >= {"POST", "GET", "DELETE"}


@pytest.mark.asyncio
async def test_async_session_per_call_override(fresh_env):
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return echo_server()(req)

    transport = httpx.MockTransport(handler)
    async with AsyncClient(base_url="http://x", transport=transport) as c:
        async with c.session(extra_headers={"X-Tenant": "acme"}) as s:
            await s.chat("hi", extra_headers={"X-Tenant": "override"})
    turn = [r for r in seen if r.url.path.endswith("/messages")][0]
    assert turn.headers["x-tenant"] == "override"
