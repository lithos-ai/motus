"""Integration tests for HITL interrupt/resume (Tasks 19-25).

These tests start a real AgentServer with a fake agent that calls
interrupt() directly.  They exercise the full pipe mechanism end-to-end.
"""

import asyncio
import threading
import time

import pytest
import requests

from motus.serve.server import AgentServer

# ---------------------------------------------------------------------------
# Running server fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def running_server():
    """Start an AgentServer in a background thread and return its base URL."""
    import_path = "tests.integration.serve.fixtures.approval_agent:fake_agent"

    server = AgentServer(
        agent_fn=import_path,
        max_workers=2,
        timeout=30.0,
        ttl=0,
    )

    import uvicorn

    config = uvicorn.Config(
        server.app,
        host="127.0.0.1",
        port=0,
        log_level="warning",
    )
    uv_server = uvicorn.Server(config)

    thread = threading.Thread(
        target=lambda: asyncio.run(uv_server.serve()), daemon=True
    )
    thread.start()

    # Wait for startup
    for _ in range(50):
        if uv_server.started:
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("Server failed to start")

    port = uv_server.servers[0].sockets[0].getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"

    yield base_url

    uv_server.should_exit = True
    thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _create_session(base_url):
    """Create a new session and return its session_id."""
    resp = requests.post(f"{base_url}/sessions", timeout=5.0)
    assert resp.status_code == 201, f"create_session failed: {resp.text}"
    return resp.json()["session_id"]


def _send_message(base_url, session_id, content):
    """POST a user message to a session."""
    resp = requests.post(
        f"{base_url}/sessions/{session_id}/messages",
        json={"role": "user", "content": content},
        timeout=5.0,
    )
    assert resp.status_code == 202, f"send_message failed: {resp.text}"
    return resp.json()


def _wait_for_interrupted(base_url, session_id, max_wait=15):
    """Poll until session reaches 'interrupted' status."""
    for _ in range(max_wait * 10):  # 100ms intervals
        resp = requests.get(
            f"{base_url}/sessions/{session_id}",
            params={"wait": "true", "timeout": "2"},
            timeout=5.0,
        )
        data = resp.json()
        if data["status"] == "interrupted":
            return data
        if data["status"] in ("error",):
            pytest.fail(f"Expected interrupted, got {data['status']}: {data}")
        time.sleep(0.1)
    pytest.fail("Session never reached interrupted state")


def _wait_for_idle(base_url, session_id, max_wait=15):
    """Poll until session reaches 'idle' status."""
    for _ in range(max_wait * 10):
        resp = requests.get(
            f"{base_url}/sessions/{session_id}",
            params={"wait": "true", "timeout": "2"},
            timeout=5.0,
        )
        data = resp.json()
        if data["status"] == "idle":
            return data
        if data["status"] == "error":
            pytest.fail(f"Expected idle, got error: {data}")
        time.sleep(0.1)
    pytest.fail("Session never reached idle state")


def _resume(base_url, session_id, interrupt_id, value):
    """POST a resume for a pending interrupt."""
    resp = requests.post(
        f"{base_url}/sessions/{session_id}/resume",
        json={"interrupt_id": interrupt_id, "value": value},
        timeout=5.0,
    )
    return resp


# ---------------------------------------------------------------------------
# Task 19: Infrastructure smoke test
# ---------------------------------------------------------------------------


def test_infrastructure_smoke(running_server):
    """Verify the running_server fixture: GET /health returns 200 with status=ok."""
    resp = requests.get(f"{running_server}/health", timeout=5.0)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# Task 20: Tool approval happy path
# ---------------------------------------------------------------------------


def test_tool_approval_happy_path(running_server):
    """POST 'delete' → wait for interrupted → approve → idle with 'Deleted' in response."""
    base_url = running_server
    session_id = _create_session(base_url)

    _send_message(base_url, session_id, "please delete the file")

    data = _wait_for_interrupted(base_url, session_id)
    assert data["status"] == "interrupted"
    interrupts = data["interrupts"]
    assert interrupts is not None and len(interrupts) == 1
    interrupt_info = interrupts[0]
    assert interrupt_info["type"] == "tool_approval"

    interrupt_id = interrupt_info["interrupt_id"]
    resp = _resume(base_url, session_id, interrupt_id, {"approved": True})
    assert resp.status_code == 200

    final = _wait_for_idle(base_url, session_id)
    assert final["status"] == "idle"
    assert final["response"] is not None
    assert "Deleted" in (final["response"].get("content") or "")


# ---------------------------------------------------------------------------
# Task 21: Tool approval rejection
# ---------------------------------------------------------------------------


def test_tool_approval_rejection(running_server):
    """POST 'delete' → interrupted → resume approved=False → 'rejected' in response."""
    base_url = running_server
    session_id = _create_session(base_url)

    _send_message(base_url, session_id, "please delete the file")

    data = _wait_for_interrupted(base_url, session_id)
    interrupts = data["interrupts"]
    interrupt_id = interrupts[0]["interrupt_id"]

    resp = _resume(base_url, session_id, interrupt_id, {"approved": False})
    assert resp.status_code == 200

    final = _wait_for_idle(base_url, session_id)
    assert final["status"] == "idle"
    response_content = final["response"].get("content") or ""
    assert "rejected" in response_content.lower()


# ---------------------------------------------------------------------------
# Task 22: ask_user_question round-trip
# ---------------------------------------------------------------------------


def test_ask_user_question_round_trip(running_server):
    """POST 'ask' → interrupted with type=user_input → resume with answer → 'chose' in response."""
    base_url = running_server
    session_id = _create_session(base_url)

    _send_message(base_url, session_id, "please ask me a question")

    data = _wait_for_interrupted(base_url, session_id)
    assert data["status"] == "interrupted"
    interrupts = data["interrupts"]
    assert interrupts is not None and len(interrupts) == 1
    interrupt_info = interrupts[0]
    assert interrupt_info["type"] == "user_input"

    interrupt_id = interrupt_info["interrupt_id"]
    resp = _resume(base_url, session_id, interrupt_id, "Option A")
    assert resp.status_code == 200

    final = _wait_for_idle(base_url, session_id)
    assert final["status"] == "idle"
    response_content = final["response"].get("content") or ""
    assert "chose" in response_content.lower()


# ---------------------------------------------------------------------------
# Task 23: 409 during interrupted state
# ---------------------------------------------------------------------------


def test_post_messages_409_during_interrupted(running_server):
    """POST 'delete' → interrupted → POST another message → assert 409."""
    base_url = running_server
    session_id = _create_session(base_url)

    _send_message(base_url, session_id, "please delete the file")

    data = _wait_for_interrupted(base_url, session_id)
    interrupt_id = data["interrupts"][0]["interrupt_id"]

    try:
        # Try to send another message while the session is interrupted
        resp = requests.post(
            f"{base_url}/sessions/{session_id}/messages",
            json={"role": "user", "content": "another message"},
            timeout=5.0,
        )
        assert resp.status_code == 409, (
            f"Expected 409 while interrupted, got {resp.status_code}: {resp.text}"
        )
    finally:
        # Clean up: resume to avoid hanging worker
        _resume(base_url, session_id, interrupt_id, {"approved": True})
        _wait_for_idle(base_url, session_id)


# ---------------------------------------------------------------------------
# Task 24: Wrong interrupt_id returns 404
# ---------------------------------------------------------------------------


def test_resume_wrong_interrupt_id_returns_404(running_server):
    """POST 'delete' → interrupted → resume with fake interrupt_id → 404."""
    base_url = running_server
    session_id = _create_session(base_url)

    _send_message(base_url, session_id, "please delete the file")

    data = _wait_for_interrupted(base_url, session_id)
    real_interrupt_id = data["interrupts"][0]["interrupt_id"]

    fake_interrupt_id = "00000000-0000-0000-0000-000000000000"

    try:
        resp = _resume(base_url, session_id, fake_interrupt_id, {"approved": True})
        assert resp.status_code == 404, (
            f"Expected 404 for fake interrupt_id, got {resp.status_code}: {resp.text}"
        )
    finally:
        # Clean up: resume with the real interrupt_id
        _resume(base_url, session_id, real_interrupt_id, {"approved": True})
        _wait_for_idle(base_url, session_id)


# ---------------------------------------------------------------------------
# Task 25: GET shows pending interrupts
# ---------------------------------------------------------------------------


def test_refresh_shows_pending_interrupts(running_server):
    """POST 'delete' → wait for interrupted → GET without wait → interrupts array present."""
    base_url = running_server
    session_id = _create_session(base_url)

    _send_message(base_url, session_id, "please delete the file")

    # Wait for interrupted via long-poll
    _wait_for_interrupted(base_url, session_id)

    try:
        # GET without wait parameter — should immediately return interrupted with interrupts
        resp = requests.get(
            f"{base_url}/sessions/{session_id}",
            timeout=5.0,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "interrupted"
        assert data["interrupts"] is not None
        assert len(data["interrupts"]) > 0
        interrupt_info = data["interrupts"][0]
        assert "interrupt_id" in interrupt_info
        assert "type" in interrupt_info
        assert interrupt_info["type"] == "tool_approval"
    finally:
        # Clean up: resume to avoid hanging worker
        interrupt_id = data["interrupts"][0]["interrupt_id"]
        _resume(base_url, session_id, interrupt_id, {"approved": True})
        _wait_for_idle(base_url, session_id)
