"""AsyncClient parity tests."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from motus.cloud import AgentError, AsyncClient, AuthError, SessionTimeout
from motus.cloud._transport import USER_AGENT

from .conftest import echo_server

pytestmark = pytest.mark.asyncio


class _AsyncRecorder:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def wrap(self, handler):
        def inner(req: httpx.Request) -> httpx.Response:
            self.requests.append(req)
            return handler(req)

        return httpx.MockTransport(inner)


async def test_async_chat_returns_assistant_message(fresh_env):
    rec = _AsyncRecorder()
    async with AsyncClient(base_url="http://x", transport=rec.wrap(echo_server())) as c:
        r = await c.chat("hi")
    assert r.message.content == "hi"
    deletes = [r for r in rec.requests if r.method == "DELETE"]
    assert len(deletes) == 1


async def test_async_chat_error_leaves_session_alive(fresh_env):
    rec = _AsyncRecorder()
    async with AsyncClient(
        base_url="http://x", transport=rec.wrap(echo_server(error="boom"))
    ) as c:
        with pytest.raises(AgentError):
            await c.chat("hi")
    assert [r for r in rec.requests if r.method == "DELETE"] == []


async def test_async_chat_timeout_leaves_session_alive(fresh_env):
    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST" and req.url.path == "/sessions":
            return httpx.Response(201, json={"session_id": "s1", "status": "idle"})
        if req.method == "POST" and req.url.path.endswith("/messages"):
            return httpx.Response(202, json={"session_id": "s1", "status": "running"})
        return httpx.Response(200, json={"session_id": "s1", "status": "running"})

    rec = _AsyncRecorder()
    async with AsyncClient(
        base_url="http://x",
        transport=rec.wrap(handler),
        turn_timeout=0.05,
        server_wait_slice=0.01,
    ) as c:
        with pytest.raises(SessionTimeout):
            await c.chat("hi")
    assert [r for r in rec.requests if r.method == "DELETE"] == []


async def test_async_session_keep_suppresses_delete(fresh_env):
    rec = _AsyncRecorder()
    async with AsyncClient(base_url="http://x", transport=rec.wrap(echo_server())) as c:
        async with c.session(keep=True) as s:
            r = await s.chat("hi")
            assert r.status.value == "idle"
    assert [r for r in rec.requests if r.method == "DELETE"] == []


async def test_async_session_not_owned_never_deletes(fresh_env, new_uuid):
    rec = _AsyncRecorder()
    async with AsyncClient(base_url="http://x", transport=rec.wrap(echo_server())) as c:
        async with c.session(session_id=new_uuid()) as s:
            await s.chat("hi")
    assert [r for r in rec.requests if r.method == "DELETE"] == []


async def test_async_auth_header_injected(fresh_env):
    rec = _AsyncRecorder()
    async with AsyncClient(
        base_url="http://x", api_key="sk-async", transport=rec.wrap(echo_server())
    ) as c:
        await c.chat("hi")
    for r in rec.requests:
        assert r.headers["authorization"] == "Bearer sk-async"
        assert r.headers["user-agent"] == USER_AGENT


async def test_async_401_maps_to_auth_error(fresh_env):
    rec = _AsyncRecorder()

    def h(req):
        return httpx.Response(401, json={"detail": "nope"})

    async with AsyncClient(base_url="http://x", transport=rec.wrap(h)) as c:
        with pytest.raises(AuthError):
            await c.health()


async def test_async_session_id_with_initial_state_routes_to_put(fresh_env):
    """Async counterpart: passing initial_state with session_id goes direct to PUT."""
    import uuid

    from motus.models import ChatMessage

    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if req.method == "PUT":
            return httpx.Response(
                201,
                json={
                    "session_id": req.url.path.rsplit("/", 1)[-1],
                    "status": "idle",
                },
            )
        if req.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(404)

    sid = str(uuid.uuid4())
    async with AsyncClient(
        base_url="http://x", transport=httpx.MockTransport(handler)
    ) as c:
        async with c.session(
            session_id=sid, initial_state=[ChatMessage.user_message("seed")]
        ) as s:
            assert s.owned
            assert s.session_id == sid

    methods = {(r.method, r.url.path) for r in seen}
    assert ("PUT", f"/sessions/{sid}") in methods
    assert not any(r.method == "GET" and r.url.path == f"/sessions/{sid}" for r in seen)


async def test_asyncio_gather_runs_ten_chats_independently(fresh_env):
    """Ten concurrent chat() calls complete with 10 distinct session IDs."""
    created: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST" and req.url.path == "/sessions":
            sid = f"s-{len(created)}"
            created.append(sid)
            return httpx.Response(201, json={"session_id": sid, "status": "idle"})
        if req.method == "POST" and req.url.path.endswith("/messages"):
            return httpx.Response(202, json={"session_id": "x", "status": "running"})
        if req.method == "GET" and req.url.path.startswith("/sessions/"):
            return httpx.Response(
                200,
                json={
                    "session_id": req.url.path.rsplit("/", 1)[-1],
                    "status": "idle",
                    "response": {"role": "assistant", "content": "ok"},
                },
            )
        if req.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(404)

    rec = _AsyncRecorder()
    async with AsyncClient(base_url="http://x", transport=rec.wrap(handler)) as c:
        results = await asyncio.gather(*(c.chat(f"m{i}") for i in range(10)))
    assert len({r.session_id for r in results}) == 10
