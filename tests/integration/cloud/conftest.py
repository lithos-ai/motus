"""Live AgentServer fixtures for cloud integration tests."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
import uvicorn

from motus.serve.server import AgentServer


def _start_server(import_path: str):
    server = AgentServer(agent_fn=import_path, max_workers=1, timeout=30.0, ttl=0)
    config = uvicorn.Config(server.app, host="127.0.0.1", port=0, log_level="warning")
    uv_server = uvicorn.Server(config)
    thread = threading.Thread(
        target=lambda: asyncio.run(uv_server.serve()), daemon=True
    )
    thread.start()
    for _ in range(50):
        if uv_server.started:
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("Server failed to start")
    port = uv_server.servers[0].sockets[0].getsockname()[1]
    return uv_server, thread, f"http://127.0.0.1:{port}"


@pytest.fixture(scope="module")
def echo_server_url():
    uv_server, thread, url = _start_server(
        "tests.integration.cloud._fixtures:echo_agent"
    )
    yield url
    uv_server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def approval_server_url():
    uv_server, thread, url = _start_server(
        "tests.integration.serve.fixtures.approval_agent:fake_agent"
    )
    yield url
    uv_server.should_exit = True
    thread.join(timeout=5)
