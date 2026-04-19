"""ProtocolError is raised whenever the server returns non-JSON or malformed bodies."""

from __future__ import annotations

import httpx
import pytest

from motus.cloud import AsyncClient, Client, ProtocolError


def _always(response: httpx.Response):
    return httpx.MockTransport(lambda req: response)


@pytest.mark.parametrize(
    "method_name,args",
    [
        ("health", ()),
        ("list_sessions", ()),
        ("get_messages", ("s1",)),
        ("get_session", ("s1",)),
    ],
)
def test_sync_public_methods_wrap_non_json_as_protocol_error(
    method_name, args, fresh_env
):
    transport = _always(httpx.Response(200, text="not json"))
    with Client(base_url="http://x", transport=transport) as c:
        with pytest.raises(ProtocolError):
            getattr(c, method_name)(*args)


def test_sync_create_session_rejects_non_json(fresh_env):
    transport = _always(httpx.Response(201, text="not json"))
    with Client(base_url="http://x", transport=transport) as c:
        with pytest.raises(ProtocolError):
            c.create_session()


def test_sync_get_session_rejects_schema_violation(fresh_env):
    transport = _always(httpx.Response(200, json={"oops": 1}))
    with Client(base_url="http://x", transport=transport) as c:
        with pytest.raises(ProtocolError):
            c.get_session("s1")


def test_sync_send_message_rejects_non_json(fresh_env):
    transport = _always(httpx.Response(202, text="not json"))
    with Client(base_url="http://x", transport=transport) as c:
        with pytest.raises(ProtocolError):
            c.send_message("s1", "hi")


pytestmark_async = pytest.mark.asyncio


@pytest.mark.asyncio
async def test_async_public_methods_wrap_non_json_as_protocol_error(fresh_env):
    transport = _always(httpx.Response(200, text="not json"))
    async with AsyncClient(base_url="http://x", transport=transport) as c:
        with pytest.raises(ProtocolError):
            await c.health()
        with pytest.raises(ProtocolError):
            await c.list_sessions()
        with pytest.raises(ProtocolError):
            await c.get_messages("s1")
