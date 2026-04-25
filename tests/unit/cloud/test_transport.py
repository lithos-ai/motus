"""Transport-core behavior: auth resolution, header building, error mapping, poll loop."""

from __future__ import annotations

import httpx
import pytest

from motus.cloud import (
    AuthError,
    BackendUnavailable,
    BadRequest,
    InterruptNotFound,
    ProtocolError,
    ServerBusy,
    SessionConflict,
    SessionNotFound,
    SessionTimeout,
    SessionUnsupported,
)
from motus.cloud._transport import (
    USER_AGENT,
    build_headers,
    map_status_error,
    parse_session,
    resolve_api_key,
    sync_poll_until_terminal,
    sync_send_and_poll,
    validate_base_url,
)

# ---- resolve_api_key ----


def test_resolve_api_key_explicit_wins_over_env(monkeypatch):
    monkeypatch.setenv("LITHOSAI_API_KEY", "from-env")
    assert resolve_api_key("from-arg") == "from-arg"


def test_resolve_api_key_env_wins_over_file(monkeypatch, tmp_path):
    monkeypatch.setenv("LITHOSAI_API_KEY", "from-env")
    assert resolve_api_key(None) == "from-env"


def test_resolve_api_key_empty_explicit_means_no_key(monkeypatch):
    monkeypatch.setenv("LITHOSAI_API_KEY", "from-env")
    assert resolve_api_key("") is None


def test_resolve_api_key_none_falls_through(monkeypatch):
    monkeypatch.delenv("LITHOSAI_API_KEY", raising=False)
    # credential file may or may not exist; we only assert we didn't crash
    resolve_api_key(None)


# ---- validate_base_url ----


@pytest.mark.parametrize("bad", ["", "localhost:8000", "ftp://x", None])
def test_validate_base_url_rejects(bad):
    with pytest.raises((ValueError, TypeError)):
        validate_base_url(bad)  # type: ignore[arg-type]


def test_validate_base_url_strips_trailing_slash():
    assert validate_base_url("https://api.example.com/") == "https://api.example.com"


# ---- build_headers ----


def test_build_headers_sets_user_agent_and_auth():
    h = build_headers("sk-abc", None, None)
    assert h["User-Agent"] == USER_AGENT
    assert h["Authorization"] == "Bearer sk-abc"


def test_build_headers_omits_authorization_when_no_key():
    h = build_headers(None, None, None)
    assert "Authorization" not in h


def test_build_headers_per_call_overrides_constructor():
    h = build_headers("sk", {"X-Req": "c"}, {"X-Req": "p", "X-Trace": "t"})
    assert h["X-Req"] == "p"
    assert h["X-Trace"] == "t"


# ---- map_status_error ----


def _resp(code: int, body: dict | str = "") -> httpx.Response:
    return httpx.Response(
        status_code=code,
        json=body if isinstance(body, dict) else None,
        text=body if isinstance(body, str) else None,
        request=httpx.Request("GET", "http://x"),
    )


@pytest.mark.parametrize(
    "code,expected",
    [
        (401, AuthError),
        (403, AuthError),
        (404, SessionNotFound),
        (405, SessionUnsupported),
        (409, SessionConflict),
        (503, ServerBusy),
        (500, BackendUnavailable),
        (502, BackendUnavailable),
        # Unmapped 4xx codes are request-validation / client-side problems,
        # not backend outages. They must map to BadRequest — retrying the
        # exact same request won't help.
        (400, BadRequest),
        (422, BadRequest),
        (429, BadRequest),
    ],
)
def test_map_status_error_maps_codes(code, expected):
    exc = map_status_error(_resp(code, {"detail": "x"}))
    assert isinstance(exc, expected)
    assert exc.response is not None and exc.response.status_code == code


def test_map_status_error_resume_404_is_interrupt_not_found():
    exc = map_status_error(
        _resp(404, {"detail": "No interrupt with id X"}), is_resume=True
    )
    assert isinstance(exc, InterruptNotFound)


def test_map_status_error_resume_404_session_gone_is_session_not_found():
    exc = map_status_error(_resp(404, {"detail": "Session not found"}), is_resume=True)
    assert isinstance(exc, SessionNotFound)


# ---- parse_session ----


def test_parse_session_rejects_bad_shape():
    with pytest.raises(ProtocolError):
        parse_session({"not": "valid"})


# ---- sync_send_and_poll ----


def _mock_transport(handler):
    return httpx.MockTransport(handler)


def test_sync_send_and_poll_returns_on_idle():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append((req.method, req.url.path))
        if req.method == "POST" and req.url.path.endswith("/messages"):
            return httpx.Response(202, json={"session_id": "s1", "status": "running"})
        return httpx.Response(
            200,
            json={
                "session_id": "s1",
                "status": "idle",
                "response": {"role": "assistant", "content": "hi"},
            },
        )

    http = httpx.Client(transport=_mock_transport(handler))
    snap = sync_send_and_poll(
        http,
        "http://x",
        "s1",
        {"role": "user", "content": "hello"},
        turn_timeout=None,
        server_wait_slice=1.0,
        read_retry_budget=3,
        headers={"User-Agent": "test"},
    )
    assert snap.status.value == "idle"
    assert snap.response is not None and snap.response.content == "hi"
    assert calls[0] == ("POST", "/sessions/s1/messages")
    assert calls[1] == ("GET", "/sessions/s1")


def test_sync_send_and_poll_raises_session_timeout():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST":
            return httpx.Response(202, json={"session_id": "s1", "status": "running"})
        return httpx.Response(200, json={"session_id": "s1", "status": "running"})

    http = httpx.Client(transport=_mock_transport(handler))
    with pytest.raises(SessionTimeout) as ei:
        sync_send_and_poll(
            http,
            "http://x",
            "s1",
            {"role": "user", "content": "hi"},
            turn_timeout=0.05,
            server_wait_slice=0.01,
            read_retry_budget=3,
            headers={},
        )
    assert ei.value.session_id == "s1"
    assert ei.value.last_snapshot is not None


def test_sync_poll_retries_read_timeout_then_returns():
    state = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] < 2:
            raise httpx.ReadTimeout("poll timeout", request=req)
        return httpx.Response(
            200,
            json={
                "session_id": "s1",
                "status": "idle",
                "response": {"role": "assistant", "content": "ok"},
            },
        )

    http = httpx.Client(transport=_mock_transport(handler))
    snap = sync_poll_until_terminal(
        http,
        "http://x",
        "s1",
        turn_timeout=None,
        server_wait_slice=1.0,
        read_retry_budget=3,
        headers={},
    )
    assert snap.status.value == "idle"
    assert state["n"] == 2


def test_sync_poll_exhausts_retry_budget():
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("poll timeout", request=req)

    http = httpx.Client(transport=_mock_transport(handler))
    with pytest.raises(BackendUnavailable):
        sync_poll_until_terminal(
            http,
            "http://x",
            "s1",
            turn_timeout=None,
            server_wait_slice=1.0,
            read_retry_budget=2,
            headers={},
        )
