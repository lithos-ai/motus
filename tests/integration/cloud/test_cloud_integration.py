"""End-to-end tests: motus.cloud.Client + AsyncClient against a live AgentServer."""

from __future__ import annotations

import httpx
import pytest

from motus.cloud import AsyncClient, Client

pytestmark = pytest.mark.integration


# ---------- sync ----------


def test_sync_chat_happy_path(echo_server_url):
    with Client(base_url=echo_server_url) as c:
        result = c.chat("hello")
    assert result.status.value == "idle"
    assert result.message is not None
    assert "echo: hello" in (result.message.content or "")

    # Ephemeral session deleted.
    with httpx.Client() as probe:
        r = probe.get(f"{echo_server_url}/sessions/{result.session_id}")
        assert r.status_code == 404


def test_sync_session_multi_turn_with_keep(echo_server_url):
    """--keep semantics: session stays alive; history grows across turns."""
    with Client(base_url=echo_server_url) as c:
        with c.session(keep=True) as s:
            s.chat("one")
            s.chat("two")
            sid = s.session_id
            msgs = c.get_messages(sid)
    # Session survives close because keep=True.
    with httpx.Client() as probe:
        r = probe.get(f"{echo_server_url}/sessions/{sid}")
        assert r.status_code == 200
        # Cleanup so the server doesn't leak for other tests.
        probe.delete(f"{echo_server_url}/sessions/{sid}")
    # History: two user messages + two assistant responses = 4 (plus any seed).
    roles = [m.get("role") for m in msgs]
    assert roles.count("user") == 2
    assert roles.count("assistant") == 2


def test_sync_interrupt_and_resume(approval_server_url):
    with Client(base_url=approval_server_url) as c:
        result = c.chat("please delete file")
        assert result.status.value == "interrupted"
        assert len(result.interrupts) == 1
        final = result.resume({"approved": True})
    assert final.status.value == "idle"
    assert "Deleted" in (final.message.content or "")


# ---------- async ----------


@pytest.mark.asyncio
async def test_async_chat_via_asgi_transport():
    """AsyncClient reaching the FastAPI app via ASGITransport — no sockets."""
    from motus.serve.server import AgentServer

    server = AgentServer(
        agent_fn="tests.integration.cloud._fixtures:echo_agent",
        max_workers=1,
        timeout=30.0,
        ttl=0,
    )
    transport = httpx.ASGITransport(app=server.app)
    async with AsyncClient(base_url="http://test", transport=transport) as c:
        result = await c.chat("hi")
    assert result.status.value == "idle"
    assert "echo: hi" in (result.message.content or "")


@pytest.mark.asyncio
async def test_async_session_multi_turn_pinned(echo_server_url):
    """Live server: AsyncClient.session() keeps state across turns."""
    async with AsyncClient(base_url=echo_server_url) as c:
        async with c.session() as s:
            r1 = await s.chat("one")
            r2 = await s.chat("two")
            assert "echo: one" in (r1.message.content or "")
            assert "echo: two" in (r2.message.content or "")
