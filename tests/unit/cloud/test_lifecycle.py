"""Use-after-close and Session.close propagation (R7 review findings)."""

from __future__ import annotations

import httpx
import pytest

from motus.cloud import (
    AsyncClient,
    BackendUnavailable,
    Client,
    ClientClosed,
)


def test_sync_client_use_after_close_raises_client_closed(fresh_env):
    """Calling any public method after close() must raise typed ClientClosed,
    not the raw httpx RuntimeError."""
    transport = httpx.MockTransport(
        lambda r: httpx.Response(
            200,
            json={
                "status": "ok",
                "max_workers": 1,
                "running_workers": 0,
                "total_sessions": 0,
            },
        )
    )
    with Client(base_url="http://x", transport=transport) as c:
        c.health()  # works
    with pytest.raises(ClientClosed):
        c.health()
    with pytest.raises(ClientClosed):
        c.chat("hi")
    with pytest.raises(ClientClosed):
        c.create_session()


@pytest.mark.asyncio
async def test_async_client_use_after_aclose_raises_client_closed(fresh_env):
    transport = httpx.MockTransport(
        lambda r: httpx.Response(
            200,
            json={
                "status": "ok",
                "max_workers": 1,
                "running_workers": 0,
                "total_sessions": 0,
            },
        )
    )
    async with AsyncClient(base_url="http://x", transport=transport) as c:
        await c.health()
    with pytest.raises(ClientClosed):
        await c.health()
    with pytest.raises(ClientClosed):
        await c.chat("hi")


def test_sync_client_close_is_idempotent():
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={}))
    c = Client(base_url="http://x", transport=transport)
    c.close()
    c.close()  # no-op, no error


def test_sync_session_close_propagates_non_404(recorder, fresh_env):
    """Explicit-cleanup path: a 5xx on DELETE must propagate and leave the
    session re-closable instead of silently turning into a leak."""

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST" and req.url.path == "/sessions":
            return httpx.Response(201, json={"session_id": "s1", "status": "idle"})
        if req.method == "DELETE":
            return httpx.Response(503, json={"detail": "busy"})
        return httpx.Response(404)

    transport = recorder(handler)
    with Client(base_url="http://x", transport=transport) as c:
        sess = c.session()
        from motus.cloud import ServerBusy

        with pytest.raises(ServerBusy):
            sess.close()
        # The handle is still unclosed — caller can retry.
        assert not sess.closed


def test_sync_session_close_tolerates_404_as_before(fresh_env):
    def handler(req):
        if req.method == "POST" and req.url.path == "/sessions":
            return httpx.Response(201, json={"session_id": "s1", "status": "idle"})
        if req.method == "DELETE":
            return httpx.Response(404, json={"detail": "Session not found"})
        return httpx.Response(404)

    with Client(base_url="http://x", transport=httpx.MockTransport(handler)) as c:
        with c.session() as s:
            pass  # exits cleanly, DELETE 404 is silent
        assert s.closed


def test_sync_session_keep_alive_switch_suppresses_delete(recorder, fresh_env):
    """Session.keep_alive() converts an owned session into keep mode on the fly."""

    transport = recorder(
        lambda req: (
            httpx.Response(201, json={"session_id": "s1", "status": "idle"})
            if req.method == "POST" and req.url.path == "/sessions"
            else httpx.Response(204)
        )
    )
    with Client(base_url="http://x", transport=transport) as c:
        with c.session() as s:
            s.keep_alive()
    # DELETE must NOT have been issued.
    assert all(r.method != "DELETE" for r in recorder.requests)


def test_backend_unavailable_still_raised_for_network_failure_on_close(
    recorder, fresh_env
):
    """Propagate BackendUnavailable if the DELETE hits a network error."""

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST" and req.url.path == "/sessions":
            return httpx.Response(201, json={"session_id": "s1", "status": "idle"})
        if req.method == "DELETE":
            raise httpx.ConnectError("boom", request=req)
        return httpx.Response(404)

    transport = recorder(handler)
    with Client(base_url="http://x", transport=transport) as c:
        sess = c.session()
        with pytest.raises(BackendUnavailable):
            sess.close()
        assert not sess.closed
