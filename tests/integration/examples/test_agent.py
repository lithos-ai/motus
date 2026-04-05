"""
Integration tests for examples/agent.py — the basic ReAct agent example.

Exercises three interfaces:
  1. Console  — direct agent function call (VCR cassette replay)
  2. Serve    — HTTP API via ASGI transport (no network)
  3. Deploy   — validate server export is an AgentServer

Usage:
    # Run in replay mode (no API keys needed)
    pytest tests/integration/examples/test_examples.py -v

    # Re-record VCR cassette (needs real API keys)
    pytest tests/integration/examples/test_examples.py -v -k console --vcr-record=all
"""

import importlib
from pathlib import Path

import pytest

CASSETTES_DIR = Path(__file__).parent / "cassettes"


def _import_attr(dotted_path: str):
    """Import 'module.path:attribute' and return the attribute."""
    module_path, attr_name = dotted_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


class TestAgentDeploy:
    """Validate that examples.agent exports a valid AgentServer."""

    def test_deploy_export(self):
        from motus.serve import AgentServer

        server = _import_attr("examples.agent:server")
        assert isinstance(server, AgentServer)
        assert hasattr(server, "app")


class TestAgentConsole:
    """Call the agent function directly (console mode) with VCR cassette replay."""

    @pytest.mark.integration
    async def test_console(self, configured_vcr):
        from motus.models import ChatMessage

        agent_fn = _import_attr("examples.agent:agent")
        msg = ChatMessage.user_message(content="Say hello in one sentence.")

        cassette = CASSETTES_DIR / "agent_console.yaml"
        cassette.parent.mkdir(parents=True, exist_ok=True)
        with configured_vcr.use_cassette(str(cassette)):
            response, new_state = await agent_fn(msg, [])

        assert isinstance(response, ChatMessage)
        assert response.role == "assistant"
        assert response.content and len(response.content) > 0
        assert isinstance(new_state, list)
        assert len(new_state) >= 2


class TestAgentServe:
    """Exercise the serve HTTP API via ASGI transport."""

    async def test_serve_health_and_sessions(self):
        from httpx import ASGITransport, AsyncClient

        server = _import_attr("examples.agent:server")

        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            # Health check
            r = await client.get("/health")
            assert r.status_code == 200
            assert r.json()["status"] == "ok"

            # Create session
            r = await client.post("/sessions")
            assert r.status_code == 201
            sid = r.json()["session_id"]
            assert r.json()["status"] == "idle"

            # List sessions
            r = await client.get("/sessions")
            assert r.status_code == 200
            session_ids = [s["session_id"] for s in r.json()]
            assert sid in session_ids

            # Get session
            r = await client.get(f"/sessions/{sid}")
            assert r.status_code == 200
            assert r.json()["session_id"] == sid

            # Get messages (empty)
            r = await client.get(f"/sessions/{sid}/messages")
            assert r.status_code == 200
            assert r.json() == []

            # Delete session
            r = await client.delete(f"/sessions/{sid}")
            assert r.status_code == 204

            # Verify deleted
            r = await client.get(f"/sessions/{sid}")
            assert r.status_code == 404
