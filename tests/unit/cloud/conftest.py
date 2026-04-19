"""Shared fixtures for motus.cloud unit tests."""

from __future__ import annotations

import uuid
from typing import Any, Callable

import httpx
import pytest


@pytest.fixture
def fresh_env(monkeypatch):
    """Clear auth-related env so credential tests are deterministic."""
    monkeypatch.delenv("LITHOSAI_API_KEY", raising=False)
    monkeypatch.delenv("LITHOSAI_API_URL", raising=False)
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
    custom_id_mode: str = "conflict",
) -> Callable[[httpx.Request], httpx.Response]:
    """Build a MockTransport handler that mimics the session/message protocol.

    ``custom_id_mode`` shapes how PUT /sessions/{id} behaves:
        - "conflict" (default): return 409 (session already exists — attach path).
        - "created":  return 201 (owned custom-ID creation — PUT path).
        - "unsupported": return 405 (server without --allow-custom-ids).
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
            # Default: conflict (existing session — caller wanted attach)
            state["sid"] = sid
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
