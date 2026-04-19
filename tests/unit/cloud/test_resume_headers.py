"""ChatResult.resume preserves the extra_headers scope that produced the result."""

from __future__ import annotations

import httpx
import pytest

from motus.cloud import AsyncClient, Client


def _interrupted_then_resolved(resume_marker: dict):
    """MockTransport handler: first GET interrupts, then resume + GET resolves."""
    state = {"turn": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        m, p = req.method, req.url.path
        if m == "POST" and p == "/sessions":
            return httpx.Response(201, json={"session_id": "s1", "status": "idle"})
        if m == "POST" and p.endswith("/messages"):
            state["turn"] = 1
            return httpx.Response(202, json={"session_id": "s1", "status": "running"})
        if m == "POST" and p.endswith("/resume"):
            resume_marker["resume_headers"] = dict(req.headers)
            state["turn"] = 99
            return httpx.Response(200, json={"session_id": "s1", "status": "running"})
        if m == "GET" and p.startswith("/sessions/"):
            if state["turn"] >= 99:
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

    return handler


def test_sync_client_chat_result_resume_preserves_per_call_headers(fresh_env):
    captured: dict = {}
    transport = httpx.MockTransport(_interrupted_then_resolved(captured))
    with Client(base_url="http://x", transport=transport) as c:
        first = c.chat("pick", extra_headers={"X-Tenant": "acme"})
        assert first.status.value == "interrupted"
        first.resume("blue")
    assert captured["resume_headers"].get("x-tenant") == "acme"


def test_sync_session_chat_result_resume_preserves_session_headers(fresh_env):
    captured: dict = {}
    transport = httpx.MockTransport(_interrupted_then_resolved(captured))
    with Client(base_url="http://x", transport=transport) as c:
        with c.session(extra_headers={"X-Tenant": "acme"}) as s:
            first = s.chat("pick")
            assert first.status.value == "interrupted"
            first.resume("blue")
    assert captured["resume_headers"].get("x-tenant") == "acme"


def test_sync_session_chat_result_resume_preserves_per_call_over_session(fresh_env):
    captured: dict = {}
    transport = httpx.MockTransport(_interrupted_then_resolved(captured))
    with Client(base_url="http://x", transport=transport) as c:
        with c.session(extra_headers={"X-Tenant": "acme"}) as s:
            first = s.chat("pick", extra_headers={"X-Tenant": "override"})
            first.resume("blue")
    assert captured["resume_headers"].get("x-tenant") == "override"


@pytest.mark.asyncio
async def test_async_client_chat_result_resume_preserves_per_call_headers(fresh_env):
    captured: dict = {}
    transport = httpx.MockTransport(_interrupted_then_resolved(captured))
    async with AsyncClient(base_url="http://x", transport=transport) as c:
        first = await c.chat("pick", extra_headers={"X-Tenant": "acme"})
        assert first.status.value == "interrupted"
        await first.resume("blue")
    assert captured["resume_headers"].get("x-tenant") == "acme"


@pytest.mark.asyncio
async def test_async_session_chat_result_resume_preserves_session_headers(fresh_env):
    captured: dict = {}
    transport = httpx.MockTransport(_interrupted_then_resolved(captured))
    async with AsyncClient(base_url="http://x", transport=transport) as c:
        async with c.session(extra_headers={"X-Tenant": "acme"}) as s:
            first = await s.chat("pick")
            await first.resume("blue")
    assert captured["resume_headers"].get("x-tenant") == "acme"


def test_sync_client_chat_result_resume_deletes_on_clean_idle(fresh_env):
    """Ephemeral session from Client.chat → interrupted → result.resume → idle
    must DELETE the server session (otherwise it leaks)."""
    observed: list[str] = []
    state = {"turn": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        observed.append(f"{req.method} {req.url.path}")
        m, p = req.method, req.url.path
        if m == "POST" and p == "/sessions":
            return httpx.Response(201, json={"session_id": "s1", "status": "idle"})
        if m == "POST" and p.endswith("/messages"):
            return httpx.Response(202, json={"session_id": "s1", "status": "running"})
        if m == "POST" and p.endswith("/resume"):
            state["turn"] = 99
            return httpx.Response(200, json={"session_id": "s1", "status": "running"})
        if m == "GET" and p.startswith("/sessions/"):
            if state["turn"] >= 99:
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

    with Client(base_url="http://x", transport=httpx.MockTransport(handler)) as c:
        first = c.chat("pick")
        assert first.status.value == "interrupted"
        final = first.resume("blue")
        assert final.status.value == "idle"

    assert any(line.startswith("DELETE /sessions/") for line in observed), observed


@pytest.mark.asyncio
async def test_async_client_chat_result_resume_deletes_on_clean_idle(fresh_env):
    observed: list[str] = []
    state = {"turn": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        observed.append(f"{req.method} {req.url.path}")
        m, p = req.method, req.url.path
        if m == "POST" and p == "/sessions":
            return httpx.Response(201, json={"session_id": "s1", "status": "idle"})
        if m == "POST" and p.endswith("/messages"):
            return httpx.Response(202, json={"session_id": "s1", "status": "running"})
        if m == "POST" and p.endswith("/resume"):
            state["turn"] = 99
            return httpx.Response(200, json={"session_id": "s1", "status": "running"})
        if m == "GET" and p.startswith("/sessions/"):
            if state["turn"] >= 99:
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

    async with AsyncClient(
        base_url="http://x", transport=httpx.MockTransport(handler)
    ) as c:
        first = await c.chat("pick")
        assert first.status.value == "interrupted"
        final = await first.resume("blue")
        assert final.status.value == "idle"

    assert any(line.startswith("DELETE /sessions/") for line in observed), observed


def test_sync_session_chat_result_resume_does_not_delete_caller_owned(fresh_env):
    """Session.chat results must NOT delete on resume — Session.close owns that."""
    observed: list[str] = []
    state = {"turn": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        observed.append(f"{req.method} {req.url.path}")
        m, p = req.method, req.url.path
        if m == "POST" and p == "/sessions":
            return httpx.Response(201, json={"session_id": "s1", "status": "idle"})
        if m == "POST" and p.endswith("/messages"):
            return httpx.Response(202, json={"session_id": "s1", "status": "running"})
        if m == "POST" and p.endswith("/resume"):
            state["turn"] = 99
            return httpx.Response(200, json={"session_id": "s1", "status": "running"})
        if m == "GET" and p.startswith("/sessions/"):
            if state["turn"] >= 99:
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

    with Client(base_url="http://x", transport=httpx.MockTransport(handler)) as c:
        with c.session(keep=True) as s:
            first = s.chat("pick")
            assert first.status.value == "interrupted"
            final = first.resume("blue")
            assert final.status.value == "idle"
            # No DELETE from inside the resume chain — Session handles it.
            deletes_before_close = [o for o in observed if o.startswith("DELETE")]
    # keep=True → Session.close also skips DELETE.
    assert deletes_before_close == []
