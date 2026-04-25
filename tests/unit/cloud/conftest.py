"""Shared fixtures for motus.cloud unit tests."""

from __future__ import annotations

import uuid
from typing import Any, Callable

import httpx
import pytest


@pytest.fixture
def fresh_env(monkeypatch, tmp_path):
    """Clear auth-related env and isolate the credentials file so credential
    tests are deterministic regardless of whether the developer has run
    ``motus login`` on their machine."""
    monkeypatch.delenv("LITHOSAI_API_KEY", raising=False)
    monkeypatch.delenv("LITHOSAI_API_URL", raising=False)
    monkeypatch.setattr(
        "motus.auth.credentials.CREDENTIALS_FILE",
        tmp_path / "credentials.json",
    )
    yield monkeypatch


class Recorder:
    """Records httpx requests made through a MockTransport handler."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, handler: Callable[[httpx.Request], httpx.Response]):
        def wrapped(req: httpx.Request) -> httpx.Response:
            self.requests.append(req)
            return handler(req)

        return httpx.MockTransport(wrapped)

    def methods_paths(self) -> list[tuple[str, str]]:
        return [(r.method, r.url.path) for r in self.requests]


@pytest.fixture
def recorder() -> Recorder:
    return Recorder()


@pytest.fixture
def new_uuid() -> Callable[[], str]:
    return lambda: str(uuid.uuid4())


def echo_server(
    *,
    reply_content: str = "hi",
    interrupts: list[dict] | None = None,
    error: str | None = None,
    extra_get_responses: list[dict] | None = None,
    custom_id_mode: str = "attach",
    pre_existing_session_ids: set[str] | None = None,
) -> Callable[[httpx.Request], httpx.Response]:
    """Build a MockTransport handler that mimics the session/message protocol.

    ``custom_id_mode`` shapes attach-vs-create semantics for PUT /sessions/{id}
    and the non-wait GET /sessions/{id} used by the GET-first attach probe:

        - "attach" (default): GET returns 200 for any id — attach path wins;
          PUT is unreachable for this fixture.
        - "created": GET returns 404 for any id (nothing pre-exists); PUT
          returns 201 — owned custom-ID creation path.
        - "unsupported": GET returns 404 for any id; PUT returns 405 —
          server without --allow-custom-ids.
    """
    state: dict[str, Any] = {"sid": str(uuid.uuid4())}

    def terminal() -> dict:
        if error is not None:
            return {
                "session_id": state["sid"],
                "status": "error",
                "error": error,
            }
        if interrupts:
            return {
                "session_id": state["sid"],
                "status": "interrupted",
                "interrupts": interrupts,
            }
        return {
            "session_id": state["sid"],
            "status": "idle",
            "response": {"role": "assistant", "content": reply_content},
        }

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        method = req.method
        if method == "POST" and path == "/sessions":
            return httpx.Response(
                201, json={"session_id": state["sid"], "status": "idle"}
            )
        if method == "PUT" and path.startswith("/sessions/"):
            sid = path.rsplit("/", 1)[-1]
            if custom_id_mode == "created":
                state["sid"] = sid
                return httpx.Response(201, json={"session_id": sid, "status": "idle"})
            if custom_id_mode == "unsupported":
                return httpx.Response(
                    405, json={"detail": "Custom session IDs are not enabled"}
                )
            # "attach" mode: the GET-first probe already found the session, so
            # PUT is not expected to fire. Treat as a 409 if it does.
            return httpx.Response(409, json={"detail": "Session already exists"})
        if method == "POST" and path.endswith("/messages"):
            return httpx.Response(
                202, json={"session_id": state["sid"], "status": "running"}
            )
        if method == "POST" and path.endswith("/resume"):
            # After resume, session returns to running, then terminal on next GET.
            return httpx.Response(
                200, json={"session_id": state["sid"], "status": "running"}
            )
        if (
            method == "GET"
            and path.startswith("/sessions/")
            and not path.endswith("/messages")
        ):
            params = dict(req.url.params)
            if params.get("wait") != "true" and custom_id_mode in (
                "created",
                "unsupported",
            ):
                # GET-first attach probe against a server that will take the
                # create path: report 404 so the client proceeds to PUT.
                return httpx.Response(404, json={"detail": "Session not found"})
            return httpx.Response(200, json=terminal())
        if method == "GET" and path == "/sessions":
            return httpx.Response(200, json=[])
        if method == "GET" and path == "/health":
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "max_workers": 1,
                    "running_workers": 0,
                    "total_sessions": 0,
                },
            )
        if method == "DELETE" and path.startswith("/sessions/"):
            return httpx.Response(204)
        if method == "GET" and path.endswith("/messages"):
            return httpx.Response(200, json=[])
        return httpx.Response(404)

    return handler
