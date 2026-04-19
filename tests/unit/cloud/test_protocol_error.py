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


# The following cover P1: every low-level method must raise ProtocolError
# when the body decodes as JSON but has the wrong shape.


def test_sync_health_rejects_schema_violation(fresh_env):
    transport = _always(httpx.Response(200, json={}))  # missing required fields
    with Client(base_url="http://x", transport=transport) as c:
        with pytest.raises(ProtocolError):
            c.health()


def test_sync_list_sessions_rejects_schema_violation(fresh_env):
    # Server returns an object where a list was expected.
    transport = _always(httpx.Response(200, json={"not": "a list"}))
    with Client(base_url="http://x", transport=transport) as c:
        with pytest.raises(ProtocolError):
            c.list_sessions()


def test_sync_get_messages_rejects_schema_violation(fresh_env):
    transport = _always(httpx.Response(200, json={"not": "a list"}))
    with Client(base_url="http://x", transport=transport) as c:
        with pytest.raises(ProtocolError):
            c.get_messages("s1")


def test_sync_send_message_rejects_schema_violation(fresh_env):
    transport = _always(httpx.Response(202, json={"oops": 1}))
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


@pytest.mark.asyncio
async def test_async_public_methods_wrap_schema_violation_as_protocol_error(fresh_env):
    """Every low-level async method must raise ProtocolError when the body
    decodes as JSON but has the wrong shape."""
    async with AsyncClient(
        base_url="http://x", transport=_always(httpx.Response(200, json={}))
    ) as c:
        with pytest.raises(ProtocolError):
            await c.health()
    async with AsyncClient(
        base_url="http://x",
        transport=_always(httpx.Response(200, json={"not": "a list"})),
    ) as c:
        with pytest.raises(ProtocolError):
            await c.list_sessions()
        with pytest.raises(ProtocolError):
            await c.get_messages("s1")
    async with AsyncClient(
        base_url="http://x", transport=_always(httpx.Response(202, json={"oops": 1}))
    ) as c:
        with pytest.raises(ProtocolError):
            await c.send_message("s1", "hi")
