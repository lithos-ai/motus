"""motus serve CLI delegates every session/message/resume flow through motus.cloud.Client.

These tests run the CLI's command functions directly with a minimal argparse Namespace
stand-in, and use a MockTransport to record what motus.cloud.Client actually emits over HTTP.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from motus.serve import cli as cli_mod
from tests.unit.cloud.conftest import echo_server


@pytest.fixture
def auto_transport(monkeypatch):
    """Patch motus.serve.cli._make_client to return a Client backed by a MockTransport."""
    handlers: dict[str, httpx.MockTransport] = {}

    def make_transport(handler):
        return httpx.MockTransport(handler)

    def set_transport(handler) -> None:
        from motus.cloud import Client

        def _make(base_url):
            return Client(base_url=base_url, transport=make_transport(handler))

        monkeypatch.setattr(cli_mod, "_make_client", _make)

    handlers["set"] = set_transport  # type: ignore[assignment]
    return handlers


def test_chat_owned_session_deletes_on_exit(auto_transport, capsys, recorder=None):
    seen: list[tuple[str, str]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.method, req.url.path))
        return echo_server()(req)

    auto_transport["set"](handler)

    args = SimpleNamespace(
        url="http://x", message="hi", session=None, keep=False, params=None
    )
    cli_mod.chat_command(args)
    assert ("POST", "/sessions") in seen
    assert any(m == "DELETE" for m, _ in seen)
    # Assistant content printed to stdout.
    assert "hi" in capsys.readouterr().out


def test_chat_keep_suppresses_delete(auto_transport, capsys):
    seen: list[tuple[str, str]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.method, req.url.path))
        return echo_server()(req)

    auto_transport["set"](handler)
    args = SimpleNamespace(
        url="http://x", message="hi", session=None, keep=True, params=None
    )
    cli_mod.chat_command(args)
    assert all(m != "DELETE" for m, _ in seen)
    out = capsys.readouterr().out
    assert "Session:" in out and "use --session to resume" in out


def test_chat_existing_session_never_deletes(auto_transport):
    import uuid

    seen: list[tuple[str, str]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.method, req.url.path))
        return echo_server()(req)

    auto_transport["set"](handler)
    args = SimpleNamespace(
        url="http://x",
        message="hi",
        session=str(uuid.uuid4()),
        keep=False,
        params=None,
    )
    cli_mod.chat_command(args)
    assert ("POST", "/sessions") not in seen  # existing session: no creation
    assert all(m != "DELETE" for m, _ in seen)


def test_chat_interrupt_then_resume(auto_transport, capsys, monkeypatch):
    """Simulate a tool_approval interrupt; user approves; second turn returns idle."""
    state = {"turn": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        m, p = req.method, req.url.path
        if m == "POST" and p == "/sessions":
            return httpx.Response(201, json={"session_id": "s1", "status": "idle"})
        if m == "POST" and p.endswith("/messages"):
            state["turn"] += 1
            return httpx.Response(202, json={"session_id": "s1", "status": "running"})
        if m == "POST" and p.endswith("/resume"):
            state["turn"] = 99
            return httpx.Response(200, json={"session_id": "s1", "status": "running"})
        if m == "GET" and p.startswith("/sessions/"):
            if state["turn"] >= 99:
                return httpx.Response(
                    200,
                    json={
                        "session_id": "s1",
                        "status": "idle",
                        "response": {"role": "assistant", "content": "approved"},
                    },
                )
            return httpx.Response(
                200,
                json={
                    "session_id": "s1",
                    "status": "interrupted",
                    "interrupts": [
                        {
                            "interrupt_id": "i1",
                            "type": "tool_approval",
                            "payload": {"tool_name": "shell", "tool_args": ["ls"]},
                        }
                    ],
                },
            )
        if m == "DELETE":
            return httpx.Response(204)
        return httpx.Response(404)

    auto_transport["set"](handler)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
    args = SimpleNamespace(
        url="http://x", message="run ls", session=None, keep=False, params=None
    )
    cli_mod.chat_command(args)
    assert "approved" in capsys.readouterr().out


def test_health_routes_through_cloud_client(auto_transport, capsys):
    called = {"health": False}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/health":
            called["health"] = True
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "max_workers": 4,
                    "running_workers": 1,
                    "total_sessions": 2,
                },
            )
        return httpx.Response(404)

    auto_transport["set"](handler)
    cli_mod.health_check(SimpleNamespace(url="http://x"))
    assert called["health"]
    out = capsys.readouterr().out
    assert "Status: ok" in out


def test_send_with_wait_polls_until_idle(auto_transport, capsys):
    import uuid

    sid = str(uuid.uuid4())

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST" and req.url.path.endswith("/messages"):
            return httpx.Response(202, json={"session_id": sid, "status": "running"})
        if req.method == "GET" and req.url.path.startswith("/sessions/"):
            return httpx.Response(
                200,
                json={
                    "session_id": sid,
                    "status": "idle",
                    "response": {"role": "assistant", "content": "done"},
                },
            )
        return httpx.Response(404)

    auto_transport["set"](handler)
    args = SimpleNamespace(
        url="http://x",
        id=sid,
        message="hi",
        role="user",
        wait=True,
        timeout=None,
        webhook_url=None,
        webhook_token=None,
        webhook_include_messages=False,
        params=None,
    )
    cli_mod.send_message(args)
    out = capsys.readouterr().out
    assert "done" in out


def test_cli_does_not_construct_httpx_client_directly():
    """Structural guard: cli.py contains no direct httpx.Client(...) construction."""
    import inspect

    src = inspect.getsource(cli_mod)
    # The only `httpx` reference allowed is via imports or type hints; no direct
    # `httpx.Client(...)` or `httpx.AsyncClient(...)` construction.
    assert "httpx.Client(" not in src
    assert "httpx.AsyncClient(" not in src


def test_send_and_wait_helper_removed():
    """Refactor verification: the old private helper must be gone."""
    assert not hasattr(cli_mod, "_send_and_wait")
    assert not hasattr(cli_mod, "_auth_headers")
    assert not hasattr(cli_mod, "_api_call")


def test_chat_unknown_interrupt_type_preserves_session(auto_transport, capsys):
    """CLI must NOT DELETE a session when it can't resolve the interrupt type."""
    observed: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        observed.append(f"{req.method} {req.url.path}")
        m, p = req.method, req.url.path
        if m == "POST" and p == "/sessions":
            return httpx.Response(201, json={"session_id": "s1", "status": "idle"})
        if m == "POST" and p.endswith("/messages"):
            return httpx.Response(202, json={"session_id": "s1", "status": "running"})
        if m == "GET" and p.startswith("/sessions/"):
            return httpx.Response(
                200,
                json={
                    "session_id": "s1",
                    "status": "interrupted",
                    "interrupts": [
                        {
                            "interrupt_id": "i1",
                            "type": "some_future_type",
                            "payload": {"foo": "bar"},
                        }
                    ],
                },
            )
        return httpx.Response(404)

    auto_transport["set"](handler)
    args = SimpleNamespace(
        url="http://x",
        message="run unknown",
        session=None,
        keep=False,
        params=None,
    )
    cli_mod.chat_command(args)
    out = capsys.readouterr().out
    assert "some_future_type" in out
    assert "kept alive" in out.lower()
    # No DELETE was issued — the session is preserved for manual resume.
    assert all(not line.startswith("DELETE") for line in observed)


def test_chat_session_flag_rejects_missing_session(auto_transport, capsys):
    """`motus serve chat --session <typo>` must surface 'session not found', not
    silently create a new one or raise 'custom IDs not enabled'."""
    import uuid

    seen_methods: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen_methods.append(req.method)
        if req.method == "GET" and req.url.path.startswith("/sessions/"):
            return httpx.Response(404, json={"detail": "Session not found"})
        # No other endpoint should be hit for a missing --session.
        return httpx.Response(500, json={"detail": "unexpected"})

    auto_transport["set"](handler)
    args = SimpleNamespace(
        url="http://x",
        message="hi",
        session=str(uuid.uuid4()),
        keep=False,
        params=None,
    )
    with pytest.raises(SystemExit) as ei:
        cli_mod.chat_command(args)
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "not found" in out
    # No POST/PUT/DELETE should have been issued.
    assert set(seen_methods) <= {"GET"}
