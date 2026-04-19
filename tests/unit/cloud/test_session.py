"""Session ownership + keep flag behavior (AC-4.1)."""

from __future__ import annotations

import logging

import httpx
import pytest

from motus.cloud import Client, SessionClosed, SessionUnsupported

from .conftest import echo_server


def test_owned_session_deletes_on_close(recorder, fresh_env):
    transport = recorder(echo_server())
    with Client(base_url="http://x", transport=transport) as c:
        with c.session() as s:
            s.chat("hi")
    deletes = [r for r in recorder.requests if r.method == "DELETE"]
    assert len(deletes) == 1


def test_owned_session_with_keep_does_not_delete(recorder, fresh_env, caplog):
    caplog.set_level(logging.INFO, logger="motus.cloud")
    transport = recorder(echo_server())
    with Client(base_url="http://x", transport=transport) as c:
        with c.session(keep=True) as s:
            s.chat("hi")
    deletes = [r for r in recorder.requests if r.method == "DELETE"]
    assert deletes == []
    assert any("kept alive" in rec.getMessage() for rec in caplog.records)


def test_session_id_put_409_attaches_as_not_owned(recorder, fresh_env, new_uuid):
    """Default echo_server returns 409 on PUT — the attach-to-existing path."""
    transport = recorder(echo_server(custom_id_mode="conflict"))
    existing = new_uuid()
    with Client(base_url="http://x", transport=transport) as c:
        with c.session(session_id=existing) as s:
            assert not s.owned
            s.chat("hi")
    # No DELETE (caller owns the session).
    assert [r for r in recorder.requests if r.method == "DELETE"] == []


def test_session_id_put_201_creates_owned_custom_id(recorder, fresh_env, new_uuid):
    """Server allows custom IDs → PUT 201 → owned=True → DELETE on close."""
    transport = recorder(echo_server(custom_id_mode="created"))
    sid = new_uuid()
    with Client(base_url="http://x", transport=transport) as c:
        with c.session(session_id=sid) as s:
            assert s.owned
            assert s.session_id == sid
            s.chat("hi")
    deletes = [r for r in recorder.requests if r.method == "DELETE"]
    assert len(deletes) == 1


def test_session_id_put_405_raises_session_unsupported(recorder, fresh_env, new_uuid):
    """Server without --allow-custom-ids returns 405 — raise SessionUnsupported."""
    transport = recorder(echo_server(custom_id_mode="unsupported"))
    with Client(base_url="http://x", transport=transport) as c:
        with pytest.raises(SessionUnsupported):
            c.session(session_id=new_uuid())


def test_attach_with_keep_true_still_does_not_delete(recorder, fresh_env, new_uuid):
    transport = recorder(echo_server(custom_id_mode="conflict"))
    with Client(base_url="http://x", transport=transport) as c:
        with c.session(session_id=new_uuid(), keep=True) as s:
            assert not s.owned
            s.chat("hi")
    assert [r for r in recorder.requests if r.method == "DELETE"] == []


def test_session_id_plus_non_empty_initial_state_rejected(fresh_env, new_uuid):
    from motus.models import ChatMessage

    with Client(
        base_url="http://x",
        transport=httpx.MockTransport(lambda r: httpx.Response(200)),
    ) as c:
        with pytest.raises(ValueError):
            c.session(
                session_id=new_uuid(),
                initial_state=[ChatMessage.user_message("seed")],
            )


def test_chat_after_close_raises(recorder, fresh_env):
    transport = recorder(echo_server())
    with Client(base_url="http://x", transport=transport) as c:
        s = c.session()
        s.close()
        with pytest.raises(SessionClosed):
            s.chat("hi")


def test_second_close_is_noop(recorder, fresh_env):
    transport = recorder(echo_server())
    with Client(base_url="http://x", transport=transport) as c:
        s = c.session()
        s.close()
        s.close()  # must not raise, must not issue second DELETE
    deletes = [r for r in recorder.requests if r.method == "DELETE"]
    assert len(deletes) == 1


def test_close_tolerates_404(recorder, fresh_env):
    def handler(req):
        if req.method == "POST" and req.url.path == "/sessions":
            return httpx.Response(201, json={"session_id": "s1", "status": "idle"})
        if req.method == "DELETE":
            return httpx.Response(404, json={"detail": "Session not found"})
        return httpx.Response(404)

    transport = recorder(handler)
    with Client(base_url="http://x", transport=transport) as c:
        with c.session() as _:
            pass  # close triggers DELETE → 404; must not raise


def test_sync_client_in_running_event_loop_raises():
    """AC-8 negative test: sync Client inside asyncio loop raises RuntimeError."""
    import asyncio

    async def inside_loop():
        c = Client(
            base_url="http://x",
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
        )
        with pytest.raises(RuntimeError):
            c.health()

    asyncio.run(inside_loop())
