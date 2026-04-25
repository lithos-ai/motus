"""Sync Client behavior: chat / session / resume / low-level / auth / cleanup / injection."""

from __future__ import annotations

import httpx
import pytest

from motus.cloud import (
    AgentError,
    AuthError,
    Client,
    SessionConflict,
    SessionTimeout,
    SessionUnsupported,
)
from motus.cloud._transport import USER_AGENT

from .conftest import echo_server

# ---- construction ----


def test_client_requires_base_url():
    with pytest.raises(ValueError):
        Client(base_url="")


def test_client_rejects_both_transport_and_http_client():
    with pytest.raises(ValueError):
        Client(
            base_url="http://x",
            transport=httpx.MockTransport(lambda r: httpx.Response(200)),
            http_client=httpx.Client(),
        )


def test_client_accepts_injected_http_client():
    http = httpx.Client(
        transport=httpx.MockTransport(
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
    )
    c = Client(base_url="http://x", http_client=http)
    assert c.health()["status"] == "ok"


# ---- auth header plumbing ----


def test_user_agent_set_on_every_request(recorder, fresh_env):
    transport = recorder(echo_server())
    with Client(base_url="http://x", transport=transport) as c:
        c.chat("hi")
    for r in recorder.requests:
        assert r.headers["user-agent"] == USER_AGENT


def test_no_authorization_header_when_no_key(recorder, fresh_env):
    transport = recorder(echo_server())
    with Client(base_url="http://x", transport=transport) as c:
        c.chat("hi")
    for r in recorder.requests:
        assert "authorization" not in r.headers


def test_explicit_api_key_used(recorder, fresh_env):
    transport = recorder(echo_server())
    with Client(base_url="http://x", api_key="sk-xyz", transport=transport) as c:
        c.chat("hi")
    for r in recorder.requests:
        assert r.headers["authorization"] == "Bearer sk-xyz"


def test_env_api_key_used(recorder, fresh_env):
    fresh_env.setenv("LITHOSAI_API_KEY", "sk-env")
    transport = recorder(echo_server())
    with Client(base_url="http://x", transport=transport) as c:
        c.chat("hi")
    auths = {r.headers.get("authorization") for r in recorder.requests}
    assert auths == {"Bearer sk-env"}


def test_empty_string_api_key_disables_auth(recorder, fresh_env):
    fresh_env.setenv("LITHOSAI_API_KEY", "sk-env")
    transport = recorder(echo_server())
    with Client(base_url="http://x", api_key="", transport=transport) as c:
        c.chat("hi")
    for r in recorder.requests:
        assert "authorization" not in r.headers


def test_per_call_extra_headers_apply_to_every_request(recorder, fresh_env):
    """Per-call extra_headers override constructor headers on every outbound
    request within a chat() call — POST /sessions, POST /messages, GET poll,
    and the ephemeral-cleanup DELETE all carry the per-call header."""
    transport = recorder(echo_server())
    with Client(
        base_url="http://x",
        extra_headers={"X-Req": "c"},
        transport=transport,
    ) as c:
        c.chat("hi", extra_headers={"X-Req": "p", "X-Trace": "t"})

    for r in recorder.requests:
        assert r.headers["x-req"] == "p"
        assert r.headers["x-trace"] == "t"

    methods = {r.method for r in recorder.requests}
    assert {"POST", "GET", "DELETE"} <= methods


# ---- chat lifecycle ----


def test_chat_clean_idle_deletes_session(recorder, fresh_env):
    transport = recorder(echo_server())
    with Client(base_url="http://x", transport=transport) as c:
        r = c.chat("hi")
    assert r.message.content == "hi"
    methods = recorder.methods_paths()
    assert ("POST", "/sessions") in methods
    delete_count = sum(1 for m, _ in methods if m == "DELETE")
    assert delete_count == 1


def test_chat_interrupted_leaves_session_alive(recorder, fresh_env):
    transport = recorder(
        echo_server(
            interrupts=[
                {
                    "interrupt_id": "i1",
                    "type": "tool_approval",
                    "payload": {"tool_name": "search"},
                }
            ]
        )
    )
    with Client(base_url="http://x", transport=transport) as c:
        r = c.chat("hi")
    assert r.status.value == "interrupted"
    assert len(r.interrupts) == 1
    methods = recorder.methods_paths()
    assert all(m != "DELETE" for m, _ in methods)


def test_chat_error_leaves_session_alive_and_raises(recorder, fresh_env):
    transport = recorder(echo_server(error="boom"))
    with Client(base_url="http://x", transport=transport) as c:
        with pytest.raises(AgentError):
            c.chat("hi")
    methods = recorder.methods_paths()
    assert all(m != "DELETE" for m, _ in methods)


def test_chat_timeout_leaves_session_alive(fresh_env, recorder):
    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST" and req.url.path == "/sessions":
            return httpx.Response(201, json={"session_id": "s1", "status": "idle"})
        if req.method == "POST" and req.url.path.endswith("/messages"):
            return httpx.Response(202, json={"session_id": "s1", "status": "running"})
        return httpx.Response(200, json={"session_id": "s1", "status": "running"})

    transport = recorder(handler)
    with Client(
        base_url="http://x",
        transport=transport,
        turn_timeout=0.05,
        server_wait_slice=0.01,
    ) as c:
        with pytest.raises(SessionTimeout):
            c.chat("hi")
    assert all(m != "DELETE" for m, _ in recorder.methods_paths())


# ---- low-level fidelity ----


def test_send_message_forwards_user_params_role_webhook(recorder, fresh_env):
    transport = recorder(echo_server())
    with Client(base_url="http://x", transport=transport) as c:
        # Create first so we have a session_id to send against.
        sess = c.create_session()
        c.send_message(
            sess.session_id,
            content="tool-result",
            role="tool",
            user_params={"tenant": "acme"},
            webhook={"url": "https://cb", "token": "t", "include_messages": True},
            name="search",
            tool_call_id="tc-1",
        )
    post_msg = [
        r
        for r in recorder.requests
        if r.method == "POST" and r.url.path.endswith("/messages")
    ][0]
    import json

    body = json.loads(post_msg.content)
    assert body["role"] == "tool"
    assert body["content"] == "tool-result"
    assert body["user_params"] == {"tenant": "acme"}
    assert body["webhook"] == {
        "url": "https://cb",
        "token": "t",
        "include_messages": True,
    }
    assert body["name"] == "search"
    assert body["tool_call_id"] == "tc-1"


def test_delete_session_is_idempotent_on_404(recorder, fresh_env):
    def handler(req):
        if req.method == "DELETE":
            return httpx.Response(404, json={"detail": "Session not found"})
        return httpx.Response(200, json={})

    transport = recorder(handler)
    with Client(base_url="http://x", transport=transport) as c:
        c.delete_session("gone")  # must not raise


def test_auth_error_maps_401(recorder, fresh_env):
    def handler(req):
        return httpx.Response(401, json={"detail": "unauthorized"})

    transport = recorder(handler)
    with Client(base_url="http://x", transport=transport) as c:
        with pytest.raises(AuthError):
            c.health()


def test_session_conflict_maps_409(recorder, fresh_env):
    def handler(req):
        if req.method == "POST" and req.url.path == "/sessions":
            return httpx.Response(201, json={"session_id": "s1", "status": "idle"})
        return httpx.Response(409, json={"detail": "running"})

    transport = recorder(handler)
    with Client(base_url="http://x", transport=transport) as c:
        with pytest.raises(SessionConflict):
            c.send_message("s1", "hi")


def test_put_session_unsupported_maps_405(recorder, fresh_env, new_uuid):
    def handler(req):
        if req.method == "PUT":
            return httpx.Response(
                405, json={"detail": "Custom session IDs are not enabled"}
            )
        return httpx.Response(404)

    transport = recorder(handler)
    with Client(base_url="http://x", transport=transport) as c:
        with pytest.raises(SessionUnsupported):
            c.create_session(session_id=new_uuid())


def test_create_session_rejects_non_uuid(fresh_env):
    with Client(
        base_url="http://x",
        transport=httpx.MockTransport(
            lambda r: httpx.Response(201, json={"session_id": "x", "status": "idle"})
        ),
    ) as c:
        with pytest.raises(ValueError):
            c.create_session(session_id="not-a-uuid")


# ---- resume ----


def test_chat_post_failure_deletes_ephemeral_session(recorder, fresh_env):
    """If the initial POST /messages fails, the orphaned session must be
    DELETE'd best-effort before the exception propagates, so the caller
    doesn't end up with a leaked session they have no session_id for."""
    from motus.cloud import BadRequest

    state = {"created_sid": None}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST" and req.url.path == "/sessions":
            sid = "orphan-sid"
            state["created_sid"] = sid
            return httpx.Response(201, json={"session_id": sid, "status": "idle"})
        if req.method == "POST" and req.url.path.endswith("/messages"):
            return httpx.Response(422, json={"detail": "bad role"})
        if req.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(404)

    transport = recorder(handler)
    with Client(base_url="http://x", transport=transport) as c:
        with pytest.raises(BadRequest):
            c.chat("hi", role="bogus")
    deletes = [r for r in recorder.requests if r.method == "DELETE"]
    assert len(deletes) == 1
    assert deletes[0].url.path == f"/sessions/{state['created_sid']}"


def test_chat_events_post_failure_deletes_ephemeral_session(recorder, fresh_env):
    """Same cleanup guarantee for chat_events — a failing initial POST
    must not strand the ephemeral session."""
    from motus.cloud import BadRequest

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST" and req.url.path == "/sessions":
            return httpx.Response(201, json={"session_id": "s1", "status": "idle"})
        if req.method == "POST" and req.url.path.endswith("/messages"):
            return httpx.Response(422, json={"detail": "bad role"})
        if req.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(404)

    transport = recorder(handler)
    with Client(base_url="http://x", transport=transport) as c:
        with pytest.raises(BadRequest):
            # Iterator is cold; consuming it triggers the setup POST which fails.
            list(c.chat_events("hi", role="bogus"))
    deletes = [r for r in recorder.requests if r.method == "DELETE"]
    assert len(deletes) == 1


def test_resume_delegates_back_to_client(recorder, fresh_env):
    state = {"resumed": False}

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        method = req.method
        if method == "POST" and path == "/sessions":
            return httpx.Response(201, json={"session_id": "s1", "status": "idle"})
        if method == "POST" and path.endswith("/messages"):
            return httpx.Response(202, json={"session_id": "s1", "status": "running"})
        if method == "POST" and path.endswith("/resume"):
            state["resumed"] = True
            return httpx.Response(200, json={"session_id": "s1", "status": "running"})
        if method == "GET" and path.startswith("/sessions/"):
            if state["resumed"]:
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
                    "interrupts": [
                        {"interrupt_id": "i1", "type": "user_input", "payload": {}}
                    ],
                },
            )
        return httpx.Response(404)

    transport = recorder(handler)
    with Client(base_url="http://x", transport=transport) as c:
        first = c.chat("pick")
        assert first.status.value == "interrupted"
        next_ = first.resume("blue")
        assert next_.status.value == "idle"
        assert next_.message.content == "done"
