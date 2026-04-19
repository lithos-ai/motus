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
