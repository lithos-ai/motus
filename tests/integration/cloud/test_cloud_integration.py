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


def test_sync_chat_events_against_live_server(echo_server_url):
    """chat_events yields (running, idle) for a successful turn."""
    with Client(base_url=echo_server_url) as c:
        events = list(c.chat_events("stream hi"))
    types = [e.type for e in events]
    assert types[0] == "running"
    assert types[-1] == "idle"
    assert "echo: stream hi" in (events[-1].snapshot.response.content or "")


def test_sync_session_attach_to_existing_preserves_state(echo_server_url):
    """GET-first attach: pre-create a session, attach via session(session_id=...),
    confirm owned=False, add a turn, detach without DELETE, and verify the
    session is still alive on the server."""
    with Client(base_url=echo_server_url) as c:
        pre = c.create_session()
        sid = pre.session_id
        try:
            with c.session(session_id=sid) as attached:
                assert not attached.owned
                assert attached.session_id == sid
                r = attached.chat("hello from attach")
                assert "echo: hello from attach" in (r.message.content or "")
            # Session should still exist (caller owns it).
            probe = c.get_session(sid)
            assert probe.session_id == sid
        finally:
            c.delete_session(sid)


def test_cli_chat_with_session_flag_attaches_to_existing_session(echo_server_url):
    """CLI `motus serve chat --session <id>` routes through client.session(session_id=...)
    and must work on a stock server (no --allow-custom-ids)."""
    from types import SimpleNamespace

    from motus.serve import cli as cli_mod

    with Client(base_url=echo_server_url) as c:
        pre = c.create_session()
        sid = pre.session_id
        try:
            args = SimpleNamespace(
                url=echo_server_url,
                message="hi via cli",
                session=sid,
                keep=False,
                params=None,
            )
            cli_mod.chat_command(args)
            # CLI owns=False branch must not DELETE our session.
            probe = c.get_session(sid)
            assert probe.session_id == sid
        finally:
            c.delete_session(sid)
