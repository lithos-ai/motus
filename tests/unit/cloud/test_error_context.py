"""Every MotusClientError carries an ErrorContext populated at the transport
choke points (map_status_error / map_transport_error).

One parametrized test per sync/async sweeps across the concrete raise categories
(4xx, 404 on resume, 5xx, transport-level, timeout-on-POST) and asserts
``exc.context`` has session_id/method/url/status_code populated so fan-out
callers can correlate failures without string-parsing exception messages.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from motus.cloud import (
    AsyncClient,
    BackendUnavailable,
    BadRequest,
    Client,
    InterruptNotFound,
    MotusClientError,
    SessionConflict,
)


def _new_sid() -> str:
    return str(uuid.uuid4())


def _make_handler(responses: dict[tuple[str, str], httpx.Response | Exception]):
    def handler(req: httpx.Request) -> httpx.Response:
        key = (req.method, req.url.path)
        if key in responses:
            v = responses[key]
            if isinstance(v, Exception):
                raise v
            return v
        return httpx.Response(404)

    return handler


# ------------------- Response-mapped errors (map_status_error) -------------------


def test_context_on_4xx_send_message(fresh_env):
    sid = _new_sid()
    handler = _make_handler(
        {
            ("POST", f"/sessions/{sid}/messages"): httpx.Response(
                422, json={"detail": "bad"}
            ),
        }
    )
    with Client(base_url="http://x", transport=httpx.MockTransport(handler)) as c:
        with pytest.raises(BadRequest) as ei:
            c.send_message(sid, "hi")
    ctx = ei.value.context
    assert ctx.session_id == sid
    assert ctx.method == "POST"
    assert ctx.url == f"http://x/sessions/{sid}/messages"
    assert ctx.status_code == 422


def test_context_on_5xx_get_session(fresh_env):
    sid = _new_sid()
    handler = _make_handler(
        {
            ("GET", f"/sessions/{sid}"): httpx.Response(
                502, json={"detail": "upstream"}
            ),
        }
    )
    with Client(base_url="http://x", transport=httpx.MockTransport(handler)) as c:
        with pytest.raises(BackendUnavailable) as ei:
            c.get_session(sid)
    ctx = ei.value.context
    assert ctx.session_id == sid
    assert ctx.method == "GET"
    assert ctx.status_code == 502


def test_context_on_resume_404_interrupt_not_found(fresh_env):
    sid = _new_sid()
    iid = "int-1"
    handler = _make_handler(
        {
            ("POST", f"/sessions/{sid}/resume"): httpx.Response(
                404, json={"detail": "interrupt not found"}
            ),
        }
    )
    with Client(base_url="http://x", transport=httpx.MockTransport(handler)) as c:
        with pytest.raises(InterruptNotFound) as ei:
            c.resume(sid, iid, "value")
    ctx = ei.value.context
    assert ctx.session_id == sid
    assert ctx.interrupt_id == iid
    assert ctx.method == "POST"
    assert ctx.status_code == 404


def test_context_on_409_session_conflict(fresh_env):
    sid = _new_sid()
    handler = _make_handler(
        {
            ("POST", f"/sessions/{sid}/messages"): httpx.Response(
                409, json={"detail": "running"}
            ),
        }
    )
    with Client(base_url="http://x", transport=httpx.MockTransport(handler)) as c:
        with pytest.raises(SessionConflict) as ei:
            c.send_message(sid, "hi")
    ctx = ei.value.context
    assert ctx.session_id == sid
    assert ctx.status_code == 409


# ------------------- Transport-level errors (map_transport_error) -------------------


def test_context_on_transport_error(fresh_env):
    sid = _new_sid()

    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connect refused", request=req)

    with Client(base_url="http://x", transport=httpx.MockTransport(handler)) as c:
        with pytest.raises(BackendUnavailable) as ei:
            c.send_message(sid, "hi")
    ctx = ei.value.context
    assert ctx.session_id == sid
    assert ctx.method == "POST"
    assert ctx.url == f"http://x/sessions/{sid}/messages"
    assert ctx.status_code is None


def test_context_on_timeout_error_post(fresh_env):
    sid = _new_sid()

    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow", request=req)

    with Client(base_url="http://x", transport=httpx.MockTransport(handler)) as c:
        with pytest.raises(BackendUnavailable) as ei:
            c.send_message(sid, "hi")
    ctx = ei.value.context
    assert ctx.session_id == sid
    assert ctx.method == "POST"


def test_context_on_transport_error_delete(fresh_env):
    sid = _new_sid()

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "DELETE":
            raise httpx.ReadError("conn reset", request=req)
        return httpx.Response(404)

    with Client(base_url="http://x", transport=httpx.MockTransport(handler)) as c:
        with pytest.raises(BackendUnavailable) as ei:
            c.delete_session(sid)
    ctx = ei.value.context
    assert ctx.session_id == sid
    assert ctx.method == "DELETE"


# ------------------- Async parity -------------------


@pytest.mark.asyncio
async def test_async_context_on_4xx_send_message(fresh_env):
    sid = _new_sid()
    handler = _make_handler(
        {
            ("POST", f"/sessions/{sid}/messages"): httpx.Response(
                422, json={"detail": "bad"}
            ),
        }
    )
    async with AsyncClient(
        base_url="http://x", transport=httpx.MockTransport(handler)
    ) as c:
        with pytest.raises(BadRequest) as ei:
            await c.send_message(sid, "hi")
    ctx = ei.value.context
    assert ctx.session_id == sid
    assert ctx.method == "POST"
    assert ctx.status_code == 422


@pytest.mark.asyncio
async def test_async_context_on_transport_error(fresh_env):
    sid = _new_sid()

    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connect refused", request=req)

    async with AsyncClient(
        base_url="http://x", transport=httpx.MockTransport(handler)
    ) as c:
        with pytest.raises(BackendUnavailable) as ei:
            await c.send_message(sid, "hi")
    ctx = ei.value.context
    assert ctx.session_id == sid
    assert ctx.method == "POST"


# ------------------- Base-class invariant -------------------


def test_empty_context_when_constructed_directly():
    """Any MotusClientError defaults to an empty ErrorContext — never None —
    so callers can do ``exc.context.session_id`` without a guard."""
    e = MotusClientError("boom")
    assert e.context is not None
    assert e.context.session_id is None
    assert e.context.method is None
