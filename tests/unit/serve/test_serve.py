"""Tests for motus.serve module."""

import asyncio
import os
import time
import uuid

import pytest

from motus.models import ChatMessage
from motus.serve import AgentServer
from motus.serve.schemas import SessionStatus
from motus.serve.session import Session, SessionAlreadyExists, SessionStore
from motus.serve.worker import WorkerExecutor, _resolve_import_path, _validate_result

# Module-level functions for pickling across process boundaries.


def _echo(message, state):
    response = ChatMessage.assistant_message(content=f"echo: {message.content}")
    return response, state + [message, response]


def _add(message, state):
    a, b = map(int, message.content.split())
    response = ChatMessage.assistant_message(content=str(a + b))
    return response, state + [message, response]


async def _async_add(message, state):
    a, b = map(int, message.content.split())
    await asyncio.sleep(0.05)
    response = ChatMessage.assistant_message(content=str(a + b))
    return response, state + [message, response]


def _slow(message, state):
    seconds = float(message.content)
    time.sleep(seconds)
    response = ChatMessage.assistant_message(content=str(os.getpid()))
    return response, state + [message, response]


def _fail(message, state):
    raise ValueError("Intentional error")


def _maybe_fail(message, state):
    if message.content == "fail":
        raise ValueError("Intentional failure")
    response = ChatMessage.assistant_message(content="success")
    return response, state + [message, response]


def _bad_return(message, state):
    """Returns wrong type — not a (response, state) tuple."""
    return "not a tuple pair"


def _conditional_bad(message, state):
    """Returns wrong type when content is 'bad', correct type otherwise."""
    if message.content == "bad":
        return "not a tuple pair"
    response = ChatMessage.assistant_message(content="ok")
    return response, state + [message, response]


def _fail_or_slow(message, state):
    """Fails on 'fail', otherwise sleeps for the given number of seconds."""
    if message.content == "fail":
        raise ValueError("Intentional failure")
    seconds = float(message.content)
    time.sleep(seconds)
    response = ChatMessage.assistant_message(content=str(os.getpid()))
    return response, state + [message, response]


def _import_path(fn):
    return f"{fn.__module__}:{fn.__qualname__}"


async def _poll_until(client, sid, target_status, timeout=10.0, interval=0.2):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = await client.get(f"/sessions/{sid}")
        data = r.json()
        if data["status"] == target_status:
            return data
        await asyncio.sleep(interval)
    pytest.fail(f"Timed out waiting for session to reach '{target_status}'")


# ---------------------------------------------------------------------------
# WorkerExecutor
# ---------------------------------------------------------------------------


class TestWorkerExecutor:
    @pytest.fixture
    def executor(self):
        return WorkerExecutor(max_workers=2)

    async def test_submit_turn_and_get_result(self, executor):
        msg = ChatMessage.user_message("1 2")
        result = await executor.submit_turn(_import_path(_add), msg, [])
        assert result.success
        response, state = result.value
        assert response.content == "3"
        assert len(state) == 2
        assert state[0].role == "user"
        assert state[0].content == "1 2"
        assert state[1].role == "assistant"
        assert state[1].content == "3"

    async def test_async_agent(self, executor):
        msg = ChatMessage.user_message("5 7")
        result = await executor.submit_turn(_import_path(_async_add), msg, [])
        assert result.success
        response, state = result.value
        assert response.content == "12"
        assert len(state) == 2
        assert state[0].role == "user"
        assert state[0].content == "5 7"
        assert state[1].role == "assistant"
        assert state[1].content == "12"

    async def test_parallel_execution(self, executor):
        msg1 = ChatMessage.user_message("1")
        msg2 = ChatMessage.user_message("1")
        t1 = asyncio.create_task(executor.submit_turn(_import_path(_slow), msg1, []))
        t2 = asyncio.create_task(executor.submit_turn(_import_path(_slow), msg2, []))
        r1, r2 = await asyncio.gather(t1, t2)

        assert r1.success and r2.success
        # Different PIDs prove workers ran in different processes (parallel)
        pid1 = int(r1.value[0].content)
        pid2 = int(r2.value[0].content)
        assert pid1 != pid2

    async def test_error_handling(self, executor):
        msg = ChatMessage.user_message("")
        result = await executor.submit_turn(_import_path(_fail), msg, [])
        assert not result.success
        assert "Intentional error" in result.error

    async def test_semaphore_limits_concurrency(self):
        executor = WorkerExecutor(max_workers=1)
        start = time.time()
        msg1 = ChatMessage.user_message("1")
        msg2 = ChatMessage.user_message("1")
        t1 = asyncio.create_task(executor.submit_turn(_import_path(_slow), msg1, []))
        t2 = asyncio.create_task(executor.submit_turn(_import_path(_slow), msg2, []))
        r1, r2 = await asyncio.gather(t1, t2)
        elapsed = time.time() - start

        assert r1.success and r2.success
        # Serialized: should be ~2s, not ~1s.
        assert elapsed >= 1.8

    async def test_state_passthrough_across_turns(self, executor):
        msg1 = ChatMessage.user_message("1 2")
        r1 = await executor.submit_turn(_import_path(_add), msg1, [])
        assert r1.success
        _, state1 = r1.value

        msg2 = ChatMessage.user_message("10 20")
        r2 = await executor.submit_turn(_import_path(_add), msg2, state1)
        assert r2.success
        _, state2 = r2.value

        assert len(state2) == 4
        assert state2[0].role == "user"
        assert state2[0].content == "1 2"
        assert state2[1].role == "assistant"
        assert state2[1].content == "3"
        assert state2[2].role == "user"
        assert state2[2].content == "10 20"
        assert state2[3].role == "assistant"
        assert state2[3].content == "30"

    async def test_running_workers_property(self):
        executor = WorkerExecutor(max_workers=2)
        assert executor.running_workers == 0
        msg = ChatMessage.user_message("1")
        task = asyncio.create_task(executor.submit_turn(_import_path(_slow), msg, []))
        await asyncio.sleep(0.3)  # let worker start
        assert executor.running_workers == 1
        await task
        assert executor.running_workers == 0

    async def test_resilience_after_failures(self, executor):
        # Fail a few times
        for i in range(3):
            msg = ChatMessage.user_message("fail")
            r = await executor.submit_turn(_import_path(_maybe_fail), msg, [])
            assert not r.success

        # Should still work
        msg = ChatMessage.user_message("ok")
        r = await executor.submit_turn(_import_path(_maybe_fail), msg, [])
        assert r.success
        response, state = r.value
        assert response.content == "success"

    async def test_forkserver_background_warming_smoke(self):
        executor = WorkerExecutor(max_workers=1)
        # Smoke-test: the background warming thread doesn't break anything.
        msg = ChatMessage.user_message("1 2")
        r = await executor.submit_turn(_import_path(_add), msg, [])
        assert r.success


# ---------------------------------------------------------------------------
# AgentServer (unit)
# ---------------------------------------------------------------------------


class TestAgentServer:
    def test_import_path_from_callable(self):
        server = AgentServer(_echo, max_workers=1)
        assert server._import_path.endswith("test_serve:_echo")

    def test_import_path_from_string(self):
        server = AgentServer("myapp:my_agent", max_workers=1)
        assert server._import_path == "myapp:my_agent"

    def test_inner_function_rejected(self):
        def outer():
            def inner(message, state):
                pass

            return inner

        with pytest.raises(ValueError, match="module-level function"):
            AgentServer(outer(), max_workers=1)


# ---------------------------------------------------------------------------
# Resolve Import Path
# ---------------------------------------------------------------------------


class TestResolveImportPath:
    def test_valid_path_resolves(self):
        from motus.serve.worker import _resolve_import_path

        fn = _resolve_import_path(f"{_echo.__module__}:{_echo.__qualname__}")
        assert fn is _echo

    def test_missing_colon_raises_value_error(self):
        from motus.serve.worker import _resolve_import_path

        with pytest.raises(ValueError, match="expected 'module:variable'"):
            _resolve_import_path("no_colon_here")

    def test_nonexistent_module_raises(self):
        from motus.serve.worker import _resolve_import_path

        with pytest.raises(ModuleNotFoundError):
            _resolve_import_path("totally_fake_module_xyz:func")

    def test_nonexistent_attribute_raises(self):
        from motus.serve.worker import _resolve_import_path

        with pytest.raises(AttributeError):
            _resolve_import_path(f"{_echo.__module__}:nonexistent_func_xyz")

    def test_non_callable_returns_object(self):
        """_resolve_import_path no longer rejects non-callables (Agent instances
        aren't always callable in the traditional sense). Type checking is
        deferred to _worker_entry."""
        from motus.serve.worker import _resolve_import_path

        obj = _resolve_import_path("os.path:sep")
        assert isinstance(obj, str)


# ---------------------------------------------------------------------------
# _validate_result
# ---------------------------------------------------------------------------


class TestValidateResult:
    def test_valid_tuple(self):
        response = ChatMessage.assistant_message(content="hi")
        state = [ChatMessage.user_message("hello"), response]
        r, s = _validate_result((response, state))
        assert r is response
        assert s is state

    def test_not_tuple_raises(self):
        with pytest.raises(TypeError, match="must return a .response, state. tuple"):
            _validate_result("just a string")

    def test_wrong_length_raises(self):
        with pytest.raises(TypeError, match="must return a .response, state. tuple"):
            _validate_result((1, 2, 3))

    def test_response_not_chat_message_raises(self):
        with pytest.raises(TypeError, match="response must be a ChatMessage"):
            _validate_result(("not a message", []))

    def test_state_not_list_raises(self):
        response = ChatMessage.assistant_message(content="hi")
        with pytest.raises(TypeError, match="state must be a list"):
            _validate_result((response, "not a list"))

    def test_state_contains_non_chat_message_raises(self):
        response = ChatMessage.assistant_message(content="hi")
        with pytest.raises(TypeError, match="state must be a list"):
            _validate_result((response, [response, "not a message"]))


# ---------------------------------------------------------------------------
# HTTP Endpoints
# ---------------------------------------------------------------------------


class TestHTTPEndpoints:
    @pytest.fixture
    def server(self):
        return AgentServer(_add, max_workers=2)

    @pytest.fixture
    async def client(self, server):
        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            yield client

    async def test_health(self, client):
        r = await client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["max_workers"] == 2
        assert data["running_workers"] == 0
        assert data["total_sessions"] == 0

    async def test_create_session(self, client):
        r = await client.post("/sessions")
        assert r.status_code == 201
        data = r.json()
        sid = data["session_id"]
        assert sid
        assert data["status"] == "idle"
        assert data["response"] is None
        assert data["error"] is None
        assert r.headers["location"] == f"/sessions/{sid}"

    async def test_list_sessions(self, client):
        r1 = await client.post("/sessions")
        r2 = await client.post("/sessions")
        sid1, sid2 = r1.json()["session_id"], r2.json()["session_id"]

        r = await client.get("/sessions")
        assert r.status_code == 200
        sessions = r.json()
        assert len(sessions) == 2
        listed_ids = {s["session_id"] for s in sessions}
        assert sid1 in listed_ids
        assert sid2 in listed_ids
        for s in sessions:
            if s["session_id"] in (sid1, sid2):
                assert s["status"] == "idle"
                assert s["total_messages"] == 0

    async def test_get_session(self, client):
        r = await client.post("/sessions")
        sid = r.json()["session_id"]

        r = await client.get(f"/sessions/{sid}")
        assert r.status_code == 200
        data = r.json()
        assert data["session_id"] == sid
        assert data["status"] == "idle"
        assert data["response"] is None

    async def test_delete_session(self, client):
        r = await client.post("/sessions")
        sid = r.json()["session_id"]

        r = await client.delete(f"/sessions/{sid}")
        assert r.status_code == 204

        r = await client.get(f"/sessions/{sid}")
        assert r.status_code == 404

    async def test_send_message_and_poll(self, client):
        # Create session
        r = await client.post("/sessions")
        sid = r.json()["session_id"]

        # Send message
        r = await client.post(f"/sessions/{sid}/messages", json={"content": "3 4"})
        assert r.status_code == 202
        assert r.json()["status"] == "running"
        assert r.json()["session_id"] == sid
        assert r.headers["location"] == f"/sessions/{sid}"

        # Poll until idle
        data = await _poll_until(client, sid, "idle")
        assert data["response"]["role"] == "assistant"
        assert data["response"]["content"] == "7"

    async def test_error_propagation(self):
        from httpx import ASGITransport, AsyncClient

        server = AgentServer(_fail, max_workers=2)
        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            r = await client.post("/sessions")
            sid = r.json()["session_id"]

            r = await client.post(
                f"/sessions/{sid}/messages", json={"content": "anything"}
            )
            assert r.status_code == 202

            data = await _poll_until(client, sid, "error")
            assert "Intentional error" in data["error"]

    async def test_multi_turn_history(self, client):
        r = await client.post("/sessions")
        sid = r.json()["session_id"]

        # Turn 1
        await client.post(f"/sessions/{sid}/messages", json={"content": "1 2"})
        data = await _poll_until(client, sid, "idle")
        assert data["response"]["content"] == "3"

        # Turn 2
        await client.post(f"/sessions/{sid}/messages", json={"content": "10 20"})
        data = await _poll_until(client, sid, "idle")
        assert data["response"]["content"] == "30"

        # Check full state via sub-resource
        r = await client.get(f"/sessions/{sid}/messages")
        assert r.status_code == 200
        state = r.json()
        assert len(state) == 4  # 2 user + 2 assistant
        assert state[0]["role"] == "user"
        assert state[0]["content"] == "1 2"
        assert state[1]["role"] == "assistant"
        assert state[1]["content"] == "3"
        assert state[2]["role"] == "user"
        assert state[2]["content"] == "10 20"
        assert state[3]["role"] == "assistant"
        assert state[3]["content"] == "30"

    async def test_await_complete(self, client):
        r = await client.post("/sessions")
        sid = r.json()["session_id"]

        await client.post(f"/sessions/{sid}/messages", json={"content": "3 4"})

        r = await client.get(f"/sessions/{sid}", params={"wait": "true"})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "idle"
        assert data["response"]["content"] == "7"

    async def test_await_complete_second_turn(self, client):
        r = await client.post("/sessions")
        sid = r.json()["session_id"]

        # Complete turn 1
        await client.post(f"/sessions/{sid}/messages", json={"content": "3 4"})
        await _poll_until(client, sid, "idle")

        # Send turn 2 and immediately await (without polling first)
        await client.post(f"/sessions/{sid}/messages", json={"content": "10 20"})

        r = await client.get(f"/sessions/{sid}", params={"wait": "true"})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "idle"
        assert data["response"]["content"] == "30"

    async def test_await_error(self):
        from httpx import ASGITransport, AsyncClient

        server = AgentServer(_fail, max_workers=2)
        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            r = await client.post("/sessions")
            sid = r.json()["session_id"]

            await client.post(f"/sessions/{sid}/messages", json={"content": "anything"})

            r = await client.get(f"/sessions/{sid}", params={"wait": "true"})
            assert r.status_code == 200
            data = r.json()
            assert data["status"] == "error"
            assert "Intentional error" in data["error"]

    async def test_error_recovery(self):
        from httpx import ASGITransport, AsyncClient

        server = AgentServer(_maybe_fail, max_workers=2)
        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            r = await client.post("/sessions")
            sid = r.json()["session_id"]

            # Trigger an error
            await client.post(f"/sessions/{sid}/messages", json={"content": "fail"})
            data = await _poll_until(client, sid, "error")
            assert "Intentional failure" in data["error"]

            # Recover with a valid message
            await client.post(f"/sessions/{sid}/messages", json={"content": "ok"})
            data = await _poll_until(client, sid, "idle")
            assert data["response"]["content"] == "success"
            assert data["error"] is None

            # Verify state only contains the recovery turn (error left no garbage)
            r = await client.get(f"/sessions/{sid}/messages")
            state = r.json()
            assert len(state) == 2
            assert state[0]["role"] == "user"
            assert state[0]["content"] == "ok"
            assert state[1]["role"] == "assistant"
            assert state[1]["content"] == "success"
            # Failed turn's user message must not appear in state
            assert all(m["content"] != "fail" for m in state)

    async def test_await_already_idle_no_response(self, client):
        r = await client.post("/sessions")
        sid = r.json()["session_id"]

        # Await on a brand-new idle session
        r = await client.get(f"/sessions/{sid}", params={"wait": "true"})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "idle"
        assert data["response"] is None

    async def test_await_already_idle_after_turn(self, client):
        r = await client.post("/sessions")
        sid = r.json()["session_id"]

        # Complete a turn
        await client.post(f"/sessions/{sid}/messages", json={"content": "5 6"})
        await _poll_until(client, sid, "idle")

        # Await after completion — should return last response immediately
        r = await client.get(f"/sessions/{sid}", params={"wait": "true"})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "idle"
        assert data["response"]["content"] == "11"

    async def test_list_sessions_total_messages(self, client):
        r = await client.post("/sessions")
        sid = r.json()["session_id"]

        await client.post(f"/sessions/{sid}/messages", json={"content": "2 3"})
        await _poll_until(client, sid, "idle")

        r = await client.get("/sessions")
        sessions = {s["session_id"]: s for s in r.json()}
        assert sessions[sid]["total_messages"] == 2

    async def test_health_reflects_session_count(self, client):
        r = await client.get("/health")
        assert r.json()["total_sessions"] == 0

        r1 = await client.post("/sessions")
        await client.post("/sessions")
        sid1 = r1.json()["session_id"]

        r = await client.get("/health")
        assert r.json()["total_sessions"] == 2

        await client.delete(f"/sessions/{sid1}")
        r = await client.get("/health")
        assert r.json()["total_sessions"] == 1

    async def test_create_session_with_preloaded_state(self):
        from httpx import ASGITransport, AsyncClient

        server = AgentServer(_add, max_workers=1)
        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            state = [
                {"role": "user", "content": "1 2"},
                {"role": "assistant", "content": "3"},
            ]
            r = await client.post("/sessions", json={"state": state})
            assert r.status_code == 201
            sid = r.json()["session_id"]

            r = await client.get(f"/sessions/{sid}/messages")
            assert r.status_code == 200
            messages = r.json()
            assert len(messages) == 2
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "1 2"
            assert messages[1]["role"] == "assistant"
            assert messages[1]["content"] == "3"

            # Verify the preloaded state is passed to the agent
            await client.post(f"/sessions/{sid}/messages", json={"content": "10 20"})
            data = await _poll_until(client, sid, "idle")
            assert data["response"]["content"] == "30"

            r = await client.get(f"/sessions/{sid}/messages")
            messages = r.json()
            assert len(messages) == 4  # 2 preloaded + 1 user + 1 assistant

    async def test_session_state_empty_on_new_session(self, client):
        r = await client.post("/sessions")
        sid = r.json()["session_id"]

        r = await client.get(f"/sessions/{sid}/messages")
        assert r.status_code == 200
        assert r.json() == []

    async def test_request_body_validation(self, client):
        r = await client.post("/sessions")
        sid = r.json()["session_id"]

        # Invalid role
        r = await client.post(
            f"/sessions/{sid}/messages", json={"role": "invalid", "content": "hi"}
        )
        assert r.status_code == 422

        # No body at all
        r = await client.post(f"/sessions/{sid}/messages")
        assert r.status_code == 422

    async def test_run_turn_bad_return_type(self):
        from httpx import ASGITransport, AsyncClient

        server = AgentServer(_bad_return, max_workers=1)
        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            r = await client.post("/sessions")
            sid = r.json()["session_id"]

            r = await client.post(
                f"/sessions/{sid}/messages", json={"content": "hello"}
            )
            assert r.status_code == 202

            data = await _poll_until(client, sid, "error")
            assert "must return a (response, state) tuple" in data["error"]

    async def test_role_defaults_to_user(self):
        from httpx import ASGITransport, AsyncClient

        server = AgentServer(_echo, max_workers=2)
        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            r = await client.post("/sessions")
            sid = r.json()["session_id"]

            # Send without explicit role — should default to "user"
            r = await client.post(
                f"/sessions/{sid}/messages", json={"content": "hello"}
            )
            assert r.status_code == 202

            data = await _poll_until(client, sid, "idle")
            assert data["response"]["role"] == "assistant"

    async def test_send_message_with_explicit_role(self):
        from httpx import ASGITransport, AsyncClient

        server = AgentServer(_echo, max_workers=1)
        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            r = await client.post("/sessions")
            sid = r.json()["session_id"]

            r = await client.post(
                f"/sessions/{sid}/messages",
                json={"role": "system", "content": "you are helpful"},
            )
            assert r.status_code == 202

            await _poll_until(client, sid, "idle")

            r = await client.get(f"/sessions/{sid}/messages")
            state = r.json()
            assert state[0]["role"] == "system"
            assert state[0]["content"] == "you are helpful"

    async def test_concurrent_different_sessions(self):
        from httpx import ASGITransport, AsyncClient

        server = AgentServer(_slow, max_workers=2)
        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            r1 = await client.post("/sessions")
            r2 = await client.post("/sessions")
            sid1 = r1.json()["session_id"]
            sid2 = r2.json()["session_id"]

            # Send slow messages to both concurrently
            await client.post(f"/sessions/{sid1}/messages", json={"content": "0.5"})
            await client.post(f"/sessions/{sid2}/messages", json={"content": "0.5"})

            # Both should complete
            data1 = await _poll_until(client, sid1, "idle")
            data2 = await _poll_until(client, sid2, "idle")

            # Different PIDs prove different processes (parallel)
            pid1 = int(data1["response"]["content"])
            pid2 = int(data2["response"]["content"])
            assert pid1 != pid2

    async def test_list_sessions_shows_running_status(self):
        from httpx import ASGITransport, AsyncClient

        server = AgentServer(_slow, max_workers=2)
        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            r = await client.post("/sessions")
            sid = r.json()["session_id"]

            # Start a slow turn
            await client.post(f"/sessions/{sid}/messages", json={"content": "2"})
            await _poll_until(client, sid, "running")

            # List sessions while running
            r = await client.get("/sessions")
            sessions = {s["session_id"]: s for s in r.json()}
            assert sessions[sid]["status"] == "running"

            # Wait for completion
            await _poll_until(client, sid, "idle")

    async def test_health_during_processing(self):
        from httpx import ASGITransport, AsyncClient

        server = AgentServer(_slow, max_workers=2)
        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            r = await client.post("/sessions")
            sid = r.json()["session_id"]

            # Start slow message
            await client.post(f"/sessions/{sid}/messages", json={"content": "2"})
            await _poll_until(client, sid, "running")

            # Health should reflect active session
            r = await client.get("/health")
            assert r.json()["total_sessions"] == 1

            # Wait for completion
            await _poll_until(client, sid, "idle")


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.fixture
    def server(self):
        return AgentServer(_slow, max_workers=2)

    @pytest.fixture
    async def client(self, server):
        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            yield client

    async def test_message_to_nonexistent_session(self, client):
        r = await client.post("/sessions/nonexistent/messages", json={"content": "hi"})
        assert r.status_code == 404

    async def test_message_while_running(self, client):
        r = await client.post("/sessions")
        sid = r.json()["session_id"]

        # Start a slow turn
        await client.post(f"/sessions/{sid}/messages", json={"content": "2"})

        # Wait until the session is actually running before sending the conflict
        await _poll_until(client, sid, "running")

        # Try to send another while running
        r = await client.post(f"/sessions/{sid}/messages", json={"content": "1"})
        assert r.status_code == 409

        # Wait for completion so shutdown is clean
        await _poll_until(client, sid, "idle")

    async def test_delete_while_running_clean_completion(self, server, client):
        r = await client.post("/sessions")
        sid = r.json()["session_id"]

        # Start a slow turn
        await client.post(f"/sessions/{sid}/messages", json={"content": "1"})

        # Delete while running
        r = await client.delete(f"/sessions/{sid}")
        assert r.status_code == 204

        # Await all background tasks — should complete without raising
        results = await asyncio.gather(
            *server._background_tasks, return_exceptions=True
        )
        for result in results:
            assert not isinstance(result, Exception)

    async def test_delete_during_execution_no_resurrection(self, server, client):
        r = await client.post("/sessions")
        sid = r.json()["session_id"]

        # Start a slow turn
        await client.post(f"/sessions/{sid}/messages", json={"content": "1"})

        # Delete while running
        r = await client.delete(f"/sessions/{sid}")
        assert r.status_code == 204

        # Wait for all background tasks to finish
        await asyncio.gather(*server._background_tasks, return_exceptions=True)

        # Session must still be gone (no resurrection by the background task)
        r = await client.get(f"/sessions/{sid}")
        assert r.status_code == 404

        # No orphaned sessions
        r = await client.get("/sessions")
        assert len(r.json()) == 0

    async def test_get_nonexistent_session(self, client):
        r = await client.get("/sessions/nope")
        assert r.status_code == 404

    async def test_delete_nonexistent_session(self, client):
        r = await client.delete("/sessions/nope")
        assert r.status_code == 404

    async def test_get_messages_nonexistent_session(self, client):
        r = await client.get("/sessions/nonexistent/messages")
        assert r.status_code == 404

    async def test_wait_nonexistent_session(self, client):
        r = await client.get("/sessions/nonexistent", params={"wait": "true"})
        assert r.status_code == 404

    async def test_delete_error_session(self):
        from httpx import ASGITransport, AsyncClient

        server = AgentServer(_fail, max_workers=1)
        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            r = await client.post("/sessions")
            sid = r.json()["session_id"]

            # Trigger error
            await client.post(f"/sessions/{sid}/messages", json={"content": "x"})
            await _poll_until(client, sid, "error")

            # Delete errored session
            r = await client.delete(f"/sessions/{sid}")
            assert r.status_code == 204

            # Verify gone
            r = await client.get(f"/sessions/{sid}")
            assert r.status_code == 404

    async def test_background_tasks_cleaned_up(self, server, client):
        r = await client.post("/sessions")
        sid = r.json()["session_id"]

        await client.post(f"/sessions/{sid}/messages", json={"content": "0.1"})
        await _poll_until(client, sid, "idle")

        # Give the done callback a moment to fire
        await asyncio.sleep(0.05)

        assert len(server._background_tasks) == 0

    async def test_await_timeout(self, client):
        r = await client.post("/sessions")
        sid = r.json()["session_id"]

        # Start a slow turn (2 seconds)
        await client.post(f"/sessions/{sid}/messages", json={"content": "2"})

        # Await with a short timeout — should return running
        r = await client.get(
            f"/sessions/{sid}", params={"wait": "true", "timeout": "0.1"}
        )
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "running"

        # Let it finish
        await _poll_until(client, sid, "idle")

    async def test_await_session_deleted_during_wait(self, client):
        r = await client.post("/sessions")
        sid = r.json()["session_id"]

        # Start a slow turn
        await client.post(f"/sessions/{sid}/messages", json={"content": "3"})

        async def await_result():
            return await client.get(f"/sessions/{sid}", params={"wait": "true"})

        async def delete_after_delay():
            await asyncio.sleep(0.5)
            await client.delete(f"/sessions/{sid}")

        # Await and delete concurrently — await should get 404
        await_task = asyncio.create_task(await_result())
        delete_task = asyncio.create_task(delete_after_delay())

        r = await asyncio.wait_for(await_task, timeout=5.0)
        await delete_task
        assert r.status_code == 404

    async def test_running_clears_previous_response(self, client):
        r = await client.post("/sessions")
        sid = r.json()["session_id"]

        # Complete a fast turn (0.1s)
        await client.post(f"/sessions/{sid}/messages", json={"content": "0.1"})
        data = await _poll_until(client, sid, "idle")
        assert data["response"] is not None

        # Start a slow turn (2s)
        await client.post(f"/sessions/{sid}/messages", json={"content": "2"})
        await _poll_until(client, sid, "running")

        # Previous response should be cleared while running
        r = await client.get(f"/sessions/{sid}")
        data = r.json()
        assert data["status"] == "running"
        assert data["response"] is None

        # Let it finish
        await _poll_until(client, sid, "idle")

    async def test_running_clears_previous_error(self):
        from httpx import ASGITransport, AsyncClient

        server = AgentServer(_fail_or_slow, max_workers=2)
        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            r = await client.post("/sessions")
            sid = r.json()["session_id"]

            # Trigger an error
            await client.post(f"/sessions/{sid}/messages", json={"content": "fail"})
            data = await _poll_until(client, sid, "error")
            assert data["error"] is not None

            # Start a slow turn (2s)
            await client.post(f"/sessions/{sid}/messages", json={"content": "2"})
            await _poll_until(client, sid, "running")

            # Previous error should be cleared while running
            r = await client.get(f"/sessions/{sid}")
            data = r.json()
            assert data["status"] == "running"
            assert data["error"] is None

            # Let it finish
            await _poll_until(client, sid, "idle")


# ---------------------------------------------------------------------------
# Session Expiry
# ---------------------------------------------------------------------------


class TestSessionTTLIntegration:
    async def test_session_ttl_via_http(self):
        from httpx import ASGITransport, AsyncClient

        server = AgentServer(_echo, ttl=0.3, max_workers=1)
        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            r = await client.post("/sessions")
            sid = r.json()["session_id"]
            assert r.status_code == 201

            # Session exists
            r = await client.get(f"/sessions/{sid}")
            assert r.status_code == 200

            # Wait past TTL
            await asyncio.sleep(0.35)

            # Trigger sweep (sweep_loop requires lifespan context)
            server._sessions.sweep()

            # Session should be gone
            r = await client.get(f"/sessions/{sid}")
            assert r.status_code == 404


class TestSessionExpiry:
    def test_sweep_deletes_expired_idle_sessions(self):
        from motus.serve.session import SessionStore

        store = SessionStore(ttl=0.1)
        s = store.create()
        sid = s.session_id
        time.sleep(0.15)
        count = store.sweep()
        assert count == 1
        assert store.get(sid) is None

    def test_sweep_deletes_expired_error_sessions(self):
        from motus.serve.session import SessionStore

        store = SessionStore(ttl=0.1)
        s = store.create()
        sid = s.session_id
        s.fail_turn("test error")
        time.sleep(0.15)
        count = store.sweep()
        assert count == 1
        assert store.get(sid) is None

    def test_sweep_skips_running_sessions(self):
        from motus.serve.schemas import SessionStatus
        from motus.serve.session import SessionStore

        store = SessionStore(ttl=0.1)
        s = store.create()
        s.status = SessionStatus.running
        time.sleep(0.15)
        count = store.sweep()
        assert count == 0
        assert store.get(s.session_id) is not None

    def test_get_does_not_touch_last_message_at(self):
        from motus.serve.session import SessionStore

        store = SessionStore(ttl=0.5)
        s = store.create()
        initial = s.last_message_at

        time.sleep(0.05)
        store.get(s.session_id)
        assert s.last_message_at == initial

    def test_sweep_signals_done(self):
        """sweep() signals _done before removing sessions, like delete()."""
        from motus.serve.session import SessionStore

        store = SessionStore(ttl=0.1)
        s = store.create()
        done_event = s._done
        time.sleep(0.15)
        count = store.sweep()
        assert count == 1
        assert store.get(s.session_id) is None
        assert done_event.is_set()

    def test_sweep_noop_when_ttl_zero(self):
        from motus.serve.session import SessionStore

        store = SessionStore(ttl=0)
        store.create()
        time.sleep(0.01)
        count = store.sweep()
        assert count == 0
        assert len(store.list()) == 1

    def test_fail_turn_clears_pending_interrupts(self):
        from motus.serve.interrupt import InterruptMessage
        from motus.serve.session import Session

        s = Session(session_id="s1")
        s.status = SessionStatus.interrupted
        s.pending_interrupts = {"i1": InterruptMessage(interrupt_id="i1", payload={})}
        s.fail_turn("worker crashed")
        assert s.pending_interrupts == {}
        assert s.status == SessionStatus.error

    def test_complete_turn_clears_pending_interrupts(self):
        s = Session(session_id="s1")
        s.status = SessionStatus.interrupted
        from motus.serve.interrupt import InterruptMessage

        s.pending_interrupts = {"i1": InterruptMessage(interrupt_id="i1", payload={})}
        resp = ChatMessage.assistant_message(content="done")
        s.complete_turn(resp, [resp])
        assert s.pending_interrupts == {}

    def test_cancel_clears_pending_interrupts(self):
        from motus.serve.interrupt import InterruptMessage

        s = Session(session_id="s1")
        s.status = SessionStatus.interrupted
        s.pending_interrupts = {"i1": InterruptMessage(interrupt_id="i1", payload={})}
        s.cancel()
        assert s.pending_interrupts == {}

    def test_sweep_skips_interrupted_sessions(self):
        """Sessions in interrupted state must not be swept, even if TTL elapsed."""
        from motus.serve.session import SessionStore

        store = SessionStore(ttl=0.01)  # 10ms TTL
        session = store.create()
        session.status = SessionStatus.interrupted
        # Simulate TTL elapsing
        session.last_message_at = time.monotonic() - 1.0  # 1s ago

        store.sweep()

        # Session should still exist
        assert store.get(session.session_id) is not None


# ---------------------------------------------------------------------------
# Agent Timeout
# ---------------------------------------------------------------------------


class TestAgentTimeout:
    async def test_timeout_sets_error(self):
        from httpx import ASGITransport, AsyncClient

        server = AgentServer(_slow, max_workers=1, timeout=3)
        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            r = await client.post("/sessions")
            sid = r.json()["session_id"]

            # Start a turn that takes 10 seconds — longer than the 3s timeout
            await client.post(f"/sessions/{sid}/messages", json={"content": "10"})

            r = await client.get(f"/sessions/{sid}", params={"wait": "true"})
            data = r.json()
            assert data["status"] == "error"
            assert data["error"] == "Agent timed out"

    async def test_timeout_recovery(self):
        """After a timeout error, the session is usable again."""
        from httpx import ASGITransport, AsyncClient

        server = AgentServer(_slow, max_workers=1, timeout=3)
        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            r = await client.post("/sessions")
            sid = r.json()["session_id"]

            # This turn takes 10s but timeout is 3s
            await client.post(f"/sessions/{sid}/messages", json={"content": "10"})
            await _poll_until(client, sid, "error")

            # Wait for background task cleanup
            await asyncio.gather(*server._background_tasks, return_exceptions=True)

            # Session should be recoverable after timeout
            await client.post(f"/sessions/{sid}/messages", json={"content": "0.1"})
            data = await _poll_until(client, sid, "idle")
            assert data["response"] is not None

    async def test_no_timeout_by_default(self):
        """timeout=0 means no limit — slow agent completes normally."""
        from httpx import ASGITransport, AsyncClient

        server = AgentServer(_slow, max_workers=1, timeout=0)
        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            r = await client.post("/sessions")
            sid = r.json()["session_id"]

            await client.post(f"/sessions/{sid}/messages", json={"content": "0.5"})

            r = await client.get(f"/sessions/{sid}", params={"wait": "true"})
            data = r.json()
            assert data["status"] == "idle"


# ---------------------------------------------------------------------------
# SessionStore (direct unit tests)
# ---------------------------------------------------------------------------


class TestSessionStore:
    def test_create_returns_idle_session_with_uuid(self):
        store = SessionStore(ttl=0)
        session = store.create()
        # Valid UUID
        uuid.UUID(session.session_id)
        assert session.status.value == "idle"
        assert session.state == []
        assert session.response is None
        assert session.error is None

    def test_get_returns_none_for_unknown_id(self):
        store = SessionStore(ttl=0)
        assert store.get("nonexistent-id") is None

    def test_delete_returns_true_for_existing(self):
        store = SessionStore(ttl=0)
        session = store.create()
        assert store.delete(session.session_id) is True
        assert store.get(session.session_id) is None

    def test_delete_returns_false_for_nonexistent(self):
        store = SessionStore(ttl=0)
        assert store.delete("nonexistent-id") is False

    def test_delete_sets_done_event(self):
        store = SessionStore(ttl=0)
        session = store.create()
        assert not session._done.is_set()
        store.delete(session.session_id)
        assert session._done.is_set()

    def test_create_multiple_unique_ids(self):
        store = SessionStore(ttl=0)
        ids = {store.create().session_id for _ in range(100)}
        assert len(ids) == 100

    def test_list_returns_all_sessions(self):
        store = SessionStore(ttl=0)
        s1 = store.create()
        s2 = store.create()
        s3 = store.create()
        listed = store.list()
        listed_ids = {s.session_id for s in listed}
        assert listed_ids == {s1.session_id, s2.session_id, s3.session_id}

    def test_max_sessions_enforced(self):
        from motus.serve.session import SessionLimitReached

        store = SessionStore(ttl=0, max_sessions=2)
        store.create()
        store.create()
        with pytest.raises(SessionLimitReached):
            store.create()

    def test_max_sessions_zero_means_unlimited(self):
        store = SessionStore(ttl=0, max_sessions=0)
        for _ in range(50):
            store.create()
        assert len(store.list()) == 50

    def test_max_sessions_delete_frees_slot(self):
        store = SessionStore(ttl=0, max_sessions=2)
        s1 = store.create()
        store.create()
        store.delete(s1.session_id)
        # Should succeed now
        store.create()
        assert len(store.list()) == 2


# ---------------------------------------------------------------------------
# Session limit (HTTP)
# ---------------------------------------------------------------------------


class TestSessionLimit:
    async def test_max_sessions_returns_503(self):
        from httpx import ASGITransport, AsyncClient

        server = AgentServer(_echo, max_workers=1, max_sessions=1)
        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            r = await client.post("/sessions")
            assert r.status_code == 201

            r = await client.post("/sessions")
            assert r.status_code == 503


# ---------------------------------------------------------------------------
# Error behavior (HTTP)
# ---------------------------------------------------------------------------


class TestErrorBehavior:
    @pytest.fixture
    def server(self):
        return AgentServer(_maybe_fail, max_workers=2)

    @pytest.fixture
    async def client(self, server):
        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            yield client

    async def test_error_clears_previous_response(self, client):
        """After success → error, GET session shows error with no stale response."""
        r = await client.post("/sessions")
        sid = r.json()["session_id"]

        # Successful turn
        await client.post(f"/sessions/{sid}/messages", json={"content": "ok"})
        data = await _poll_until(client, sid, "idle")
        assert data["response"]["content"] == "success"
        assert data["error"] is None

        # Failing turn
        await client.post(f"/sessions/{sid}/messages", json={"content": "fail"})
        data = await _poll_until(client, sid, "error")

        # Error is set
        assert "Intentional failure" in data["error"]
        # Stale response is cleared
        assert data["response"] is None

    async def test_state_unchanged_after_error(self, client):
        """Error turn does not modify session state."""
        r = await client.post("/sessions")
        sid = r.json()["session_id"]

        # Successful turn
        await client.post(f"/sessions/{sid}/messages", json={"content": "ok"})
        await _poll_until(client, sid, "idle")

        r = await client.get(f"/sessions/{sid}/messages")
        state_before = r.json()
        assert len(state_before) == 2  # user + assistant

        # Failing turn
        await client.post(f"/sessions/{sid}/messages", json={"content": "fail"})
        await _poll_until(client, sid, "error")

        # State unchanged — the failed turn's user message was NOT added
        r = await client.get(f"/sessions/{sid}/messages")
        state_after = r.json()
        assert state_after == state_before

    async def test_bad_return_recovery(self):
        """Send 'bad' → error, then 'ok' → recovers. Uses _conditional_bad agent."""
        from httpx import ASGITransport, AsyncClient

        server = AgentServer(_conditional_bad, max_workers=2)
        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            r = await client.post("/sessions")
            sid = r.json()["session_id"]

            # Bad return → error
            await client.post(f"/sessions/{sid}/messages", json={"content": "bad"})
            data = await _poll_until(client, sid, "error")
            assert "must return a (response, state) tuple" in data["error"]

            # Good return → recovers
            await client.post(f"/sessions/{sid}/messages", json={"content": "ok"})
            data = await _poll_until(client, sid, "idle")
            assert data["response"]["content"] == "ok"
            assert data["error"] is None

    async def test_bad_return_state_unchanged(self):
        """After bad-return error, session state is not corrupted."""
        from httpx import ASGITransport, AsyncClient

        server = AgentServer(_conditional_bad, max_workers=2)
        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            r = await client.post("/sessions")
            sid = r.json()["session_id"]

            # Good turn first
            await client.post(f"/sessions/{sid}/messages", json={"content": "ok"})
            await _poll_until(client, sid, "idle")

            r = await client.get(f"/sessions/{sid}/messages")
            state_before = r.json()
            assert len(state_before) == 2

            # Bad return → error
            await client.post(f"/sessions/{sid}/messages", json={"content": "bad"})
            await _poll_until(client, sid, "error")

            # State unchanged
            r = await client.get(f"/sessions/{sid}/messages")
            assert r.json() == state_before

    async def test_message_to_error_session_accepted(self, client):
        """Sending to an error session returns 202, not 409."""
        r = await client.post("/sessions")
        sid = r.json()["session_id"]

        await client.post(f"/sessions/{sid}/messages", json={"content": "fail"})
        await _poll_until(client, sid, "error")

        r = await client.post(f"/sessions/{sid}/messages", json={"content": "ok"})
        assert r.status_code == 202

        await _poll_until(client, sid, "idle")

    async def test_await_error_session_not_running(self, client):
        """Await on an error session (not running) returns immediate error response."""
        r = await client.post("/sessions")
        sid = r.json()["session_id"]

        # Trigger error and wait for it to settle
        await client.post(f"/sessions/{sid}/messages", json={"content": "fail"})
        await _poll_until(client, sid, "error")

        # Now await — session is in error status, NOT running
        r = await client.get(f"/sessions/{sid}", params={"wait": "true"})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "error"
        assert "Intentional failure" in data["error"]


# ---------------------------------------------------------------------------
# Delete-before-run edge case
# ---------------------------------------------------------------------------


class TestDeleteBeforeRun:
    async def test_delete_before_run_turn_starts(self):
        """Send message + immediate delete → background task completes cleanly."""
        from httpx import ASGITransport, AsyncClient

        # Use _slow so there's time to delete before the turn finishes
        server = AgentServer(_slow, max_workers=2)
        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            r = await client.post("/sessions")
            sid = r.json()["session_id"]

            # Send a message (slow agent: 1 second)
            await client.post(f"/sessions/{sid}/messages", json={"content": "1"})

            # Immediately delete — before the turn completes
            r = await client.delete(f"/sessions/{sid}")
            assert r.status_code == 204

            # Wait for all background tasks to finish without errors
            results = await asyncio.gather(
                *server._background_tasks, return_exceptions=True
            )
            for result in results:
                assert not isinstance(result, Exception)

            # Session must remain gone (no resurrection)
            r = await client.get(f"/sessions/{sid}")
            assert r.status_code == 404

            # No orphaned sessions
            r = await client.get("/sessions")
            assert len(r.json()) == 0

    async def test_bad_return_plus_concurrent_deletion(self):
        """Bad-return agent + concurrent deletion → no crash."""
        from httpx import ASGITransport, AsyncClient

        server = AgentServer(_bad_return, max_workers=1)
        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            r = await client.post("/sessions")
            sid = r.json()["session_id"]

            await client.post(f"/sessions/{sid}/messages", json={"content": "hello"})
            # Delete while the bad-return error is being processed
            await client.delete(f"/sessions/{sid}")

            # Background task should complete without crashing
            results = await asyncio.gather(
                *server._background_tasks, return_exceptions=True
            )
            for result in results:
                assert not isinstance(result, Exception)


# ---------------------------------------------------------------------------
# _resolve_import_path edge cases
# ---------------------------------------------------------------------------


class TestResolveImportPathEdgeCases:
    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="expected 'module:variable'"):
            _resolve_import_path("")

    def test_multiple_colons_treated_as_module_path(self):
        """rsplit(':', 1) on 'a:b:c' gives module='a:b', attr='c'.
        Importing 'a:b' should fail with ModuleNotFoundError."""
        with pytest.raises(ModuleNotFoundError):
            _resolve_import_path("a:b:c")


# ---------------------------------------------------------------------------
# _adapt_openai_agent
# ---------------------------------------------------------------------------

try:
    import agents  # noqa: F401
except ImportError:
    _agents_available = False
else:
    _agents_available = True


@pytest.mark.skipif(not _agents_available, reason="OpenAI Agents SDK not installed")
class TestAdaptOpenAIAgent:
    @pytest.fixture
    def mock_runner_run(self):
        import os

        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = "fake-key-for-test"

        from unittest.mock import AsyncMock, MagicMock

        import motus.openai_agents as oai_mod

        mock_result = MagicMock()
        mock_result.final_output = "agent reply"
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                oai_mod._OriginalRunner, "run", AsyncMock(return_value=mock_result)
            )
            yield oai_mod._OriginalRunner.run

    def test_is_openai_agent_rejects_non_agents(self):
        from motus.serve.worker import _is_openai_agent

        assert _is_openai_agent("not an agent") is False
        assert _is_openai_agent(42) is False

    def test_returns_callable_for_oai_agent(self):
        from agents import Agent

        from motus.serve.worker import _adapt_openai_agent

        agent = Agent(name="test", instructions="test")
        fn = _adapt_openai_agent(agent)
        assert fn is not None
        assert callable(fn)

    async def test_first_turn_passes_string_input(self, mock_runner_run):
        from agents import Agent

        from motus.serve.worker import _adapt_openai_agent

        agent = Agent(name="test", instructions="test")
        fn = _adapt_openai_agent(agent)

        msg = ChatMessage.user_message("hello")
        await fn(msg, [])

        args = mock_runner_run.call_args[0]
        assert args[1] == "hello"

    async def test_subsequent_turn_passes_full_history(self, mock_runner_run):
        from agents import Agent

        from motus.serve.worker import _adapt_openai_agent

        agent = Agent(name="test", instructions="test")
        fn = _adapt_openai_agent(agent)

        prior = [
            ChatMessage.user_message("first"),
            ChatMessage.assistant_message(content="first reply"),
        ]
        msg = ChatMessage.user_message("second")
        await fn(msg, prior)

        args = mock_runner_run.call_args[0]
        oai_input = args[1]
        assert isinstance(oai_input, list)
        assert len(oai_input) == 3
        assert oai_input[0] == {"role": "user", "content": "first"}
        assert oai_input[1] == {"role": "assistant", "content": "first reply"}
        assert oai_input[2] == {"role": "user", "content": "second"}

    async def test_returns_correct_state(self, mock_runner_run):
        from agents import Agent

        from motus.serve.worker import _adapt_openai_agent

        agent = Agent(name="test", instructions="test")
        fn = _adapt_openai_agent(agent)

        msg = ChatMessage.user_message("hi")
        response, state = await fn(msg, [])

        assert response.role == "assistant"
        assert response.content == "agent reply"
        assert len(state) == 2
        assert state[0] is msg
        assert state[1] is response


# ---------------------------------------------------------------------------
# WorkerExecutor default max_workers
# ---------------------------------------------------------------------------


class TestWorkerExecutorDefaults:
    def test_default_max_workers(self):
        executor = WorkerExecutor()
        assert executor.max_workers >= 1


# ---------------------------------------------------------------------------
# Session state-transition methods
# ---------------------------------------------------------------------------


class TestSessionMethods:
    async def test_start_turn(self):
        session = Session(session_id="test")
        session._done.set()  # simulate previous turn completed
        dummy_task = asyncio.create_task(asyncio.sleep(0))
        session.start_turn(dummy_task)
        assert session.status == SessionStatus.running
        assert session._task is dummy_task
        assert session.running_since is not None
        assert session.last_message_at is not None
        assert not session._done.is_set()
        await dummy_task

    def test_complete_turn(self):
        session = Session(session_id="test", status=SessionStatus.running)
        response = ChatMessage.assistant_message(content="hi")
        new_state = [ChatMessage(role="user", content="hello"), response]
        session.complete_turn(response, new_state)
        assert session.status == SessionStatus.idle
        assert session.response is response
        assert session.state == new_state
        assert session.running_since is None
        assert session.error is None

    def test_complete_turn_clears_previous_error(self):
        session = Session(
            session_id="test", status=SessionStatus.running, error="old error"
        )
        response = ChatMessage.assistant_message(content="hi")
        session.complete_turn(response, [])
        assert session.error is None

    def test_fail_turn(self):
        session = Session(session_id="test", status=SessionStatus.running)
        session.response = ChatMessage.assistant_message(content="old")
        session.fail_turn("something broke")
        assert session.status == SessionStatus.error
        assert session.response is None
        assert session.running_since is None
        assert session.error == "something broke"

    async def test_cancel_with_task(self):
        session = Session(session_id="test")
        task = asyncio.create_task(asyncio.sleep(10))
        session._task = task
        session.cancel()
        assert session._done.is_set()
        # Task enters "cancelling" state; await to complete the cancellation
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert task.cancelled()

    async def test_cancel_does_not_change_status(self):
        session = Session(session_id="test", status=SessionStatus.running)
        dummy = asyncio.create_task(asyncio.sleep(10))
        session._task = dummy
        session.cancel()
        assert (
            session.status == SessionStatus.running
        )  # cancel only signals, doesn't transition
        assert session._done.is_set()
        dummy.cancel()
        try:
            await dummy
        except asyncio.CancelledError:
            pass

    def test_cancel_without_task(self):
        session = Session(session_id="test")
        session.cancel()
        assert session._done.is_set()

    async def test_wait_no_timeout(self):
        session = Session(session_id="test")
        session._done.set()
        await session.wait()  # should return immediately

    async def test_long_turn_not_swept_after_completion(self):
        """A session that just finished a long turn should not be swept."""
        store = SessionStore(ttl=0.2)
        session = store.create()
        dummy_task = asyncio.create_task(asyncio.sleep(10))
        session.start_turn(dummy_task)
        time.sleep(0.25)  # Exceed TTL
        response = ChatMessage.assistant_message(content="done")
        session.complete_turn(response, [])
        # complete_turn refreshed last_message_at, so sweep should not expire it
        assert store.sweep() == 0
        assert store.get(session.session_id) is not None
        dummy_task.cancel()

    async def test_long_turn_not_swept_after_failure(self):
        """A session that just failed a long turn should not be swept."""
        store = SessionStore(ttl=0.2)
        session = store.create()
        dummy_task = asyncio.create_task(asyncio.sleep(10))
        session.start_turn(dummy_task)
        time.sleep(0.25)  # Exceed TTL
        session.fail_turn("error")
        assert store.sweep() == 0
        assert store.get(session.session_id) is not None
        dummy_task.cancel()

    async def test_wait_with_timeout_expires(self):
        session = Session(session_id="test")
        session.status = SessionStatus.running
        session._done.clear()
        start = time.monotonic()
        await session.wait(timeout=0.05)
        elapsed = time.monotonic() - start
        assert elapsed >= 0.04


# ---------------------------------------------------------------------------
# Session interrupt_turn / submit_resume
# ---------------------------------------------------------------------------


def test_session_interrupt_turn_basic():
    import asyncio

    from motus.serve.interrupt import InterruptMessage
    from motus.serve.schemas import SessionStatus
    from motus.serve.session import Session

    async def run():
        s = Session(session_id="s1")
        s.status = SessionStatus.running
        msg = InterruptMessage(interrupt_id="i1", payload={"type": "test"})
        s.interrupt_turn(msg)
        assert s.status == SessionStatus.interrupted
        assert "i1" in s.pending_interrupts
        assert s._done.is_set()

    asyncio.run(run())


def test_session_interrupt_turn_guards_terminal_states():
    """Late on_interrupt delivery after session is failed/completed must not
    resurrect the session."""
    import asyncio

    from motus.serve.interrupt import InterruptMessage
    from motus.serve.schemas import SessionStatus
    from motus.serve.session import Session

    async def run():
        s = Session(session_id="s1")
        s.status = SessionStatus.error
        msg = InterruptMessage(interrupt_id="i1", payload={"type": "test"})
        s.interrupt_turn(msg)
        # Guard: should silently drop the late interrupt
        assert s.status == SessionStatus.error
        assert "i1" not in s.pending_interrupts

    asyncio.run(run())


def test_session_submit_resume_none_queue_raises():
    import asyncio

    import pytest

    from motus.serve.interrupt import InterruptMessage
    from motus.serve.schemas import SessionStatus
    from motus.serve.session import Session

    async def run():
        s = Session(session_id="s1")
        s.status = SessionStatus.interrupted
        s._resume_queue = None  # worker already finished
        s.pending_interrupts = {"i1": InterruptMessage(interrupt_id="i1", payload={})}
        with pytest.raises(ValueError, match="not actively waiting"):
            s.submit_resume("i1", {"approved": True})

    asyncio.run(run())


def test_session_submit_resume_happy_path():
    import asyncio

    from motus.serve.interrupt import InterruptMessage, ResumeMessage
    from motus.serve.schemas import SessionStatus
    from motus.serve.session import Session

    async def run():
        s = Session(session_id="s1")
        s.status = SessionStatus.interrupted
        s._resume_queue = asyncio.Queue()
        s.pending_interrupts = {"i1": InterruptMessage(interrupt_id="i1", payload={})}
        s.submit_resume("i1", {"approved": True})
        assert "i1" not in s.pending_interrupts
        assert s.status == SessionStatus.running
        # Queue should have one ResumeMessage
        rm = s._resume_queue.get_nowait()
        assert isinstance(rm, ResumeMessage)
        assert rm.interrupt_id == "i1"
        assert rm.value == {"approved": True}

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Webhooks
# ---------------------------------------------------------------------------


class TestWebhooks:
    async def test_webhook_called_on_success(self):
        from unittest.mock import AsyncMock, patch

        from httpx import ASGITransport, AsyncClient

        server = AgentServer(_echo, max_workers=1)
        mock_deliver = AsyncMock()
        with patch.object(server, "_deliver_webhook", mock_deliver):
            async with AsyncClient(
                transport=ASGITransport(app=server.app), base_url="http://test"
            ) as client:
                r = await client.post("/sessions")
                sid = r.json()["session_id"]

                await client.post(
                    f"/sessions/{sid}/messages",
                    json={
                        "content": "hello",
                        "webhook": {"url": "https://example.com/hook"},
                    },
                )
                await _poll_until(client, sid, "idle")
                await asyncio.sleep(0.1)

                mock_deliver.assert_called_once()
                session_arg, webhook_arg = mock_deliver.call_args.args
                assert session_arg.session_id == sid
                assert session_arg.status == SessionStatus.idle
                assert session_arg.response.content == "echo: hello"
                assert webhook_arg.url == "https://example.com/hook"
                assert webhook_arg.token is None
                assert webhook_arg.include_messages is False

    async def test_webhook_called_on_error(self):
        from unittest.mock import AsyncMock, patch

        from httpx import ASGITransport, AsyncClient

        server = AgentServer(_fail, max_workers=1)
        mock_deliver = AsyncMock()
        with patch.object(server, "_deliver_webhook", mock_deliver):
            async with AsyncClient(
                transport=ASGITransport(app=server.app), base_url="http://test"
            ) as client:
                r = await client.post("/sessions")
                sid = r.json()["session_id"]

                await client.post(
                    f"/sessions/{sid}/messages",
                    json={
                        "content": "x",
                        "webhook": {"url": "https://example.com/hook"},
                    },
                )
                await _poll_until(client, sid, "error")
                await asyncio.sleep(0.1)

                mock_deliver.assert_called_once()
                session_arg, _ = mock_deliver.call_args.args
                assert session_arg.status == SessionStatus.error
                assert "Intentional error" in session_arg.error

    async def test_webhook_passes_token(self):
        from unittest.mock import AsyncMock, patch

        from httpx import ASGITransport, AsyncClient

        server = AgentServer(_echo, max_workers=1)
        mock_deliver = AsyncMock()
        with patch.object(server, "_deliver_webhook", mock_deliver):
            async with AsyncClient(
                transport=ASGITransport(app=server.app), base_url="http://test"
            ) as client:
                r = await client.post("/sessions")
                sid = r.json()["session_id"]

                await client.post(
                    f"/sessions/{sid}/messages",
                    json={
                        "content": "hello",
                        "webhook": {
                            "url": "https://example.com/hook",
                            "token": "secret-token-123",
                        },
                    },
                )
                await _poll_until(client, sid, "idle")
                await asyncio.sleep(0.1)

                _, webhook_arg = mock_deliver.call_args.args
                assert webhook_arg.token == "secret-token-123"

    async def test_no_webhook_no_delivery(self):
        """Sending a message without a webhook should not trigger delivery."""
        from unittest.mock import AsyncMock, patch

        from httpx import ASGITransport, AsyncClient

        server = AgentServer(_echo, max_workers=1)
        mock_deliver = AsyncMock()
        with patch.object(server, "_deliver_webhook", mock_deliver):
            async with AsyncClient(
                transport=ASGITransport(app=server.app), base_url="http://test"
            ) as client:
                r = await client.post("/sessions")
                sid = r.json()["session_id"]

                await client.post(
                    f"/sessions/{sid}/messages", json={"content": "hello"}
                )
                await _poll_until(client, sid, "idle")
                await asyncio.sleep(0.1)

                mock_deliver.assert_not_called()


class TestDeliverWebhook:
    """Unit tests for _deliver_webhook HTTP behavior."""

    async def test_posts_json_payload(self):
        import json
        from unittest.mock import AsyncMock, MagicMock, patch

        from motus.serve.schemas import WebhookSpec
        from motus.serve.session import Session

        server = AgentServer(_echo, max_workers=1)
        session = Session(session_id="test-sid", status=SessionStatus.idle)
        session.response = ChatMessage.assistant_message(content="hi")
        session.state = [ChatMessage.user_message("hello"), session.response]
        webhook = WebhookSpec(url="https://example.com/hook")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post = AsyncMock(return_value=mock_response)
        with patch("httpx.AsyncClient.post", mock_post):
            await server._deliver_webhook(session, webhook)

        mock_post.assert_called_once()
        kwargs = mock_post.call_args.kwargs
        assert kwargs["headers"]["Content-Type"] == "application/json"
        assert "Authorization" not in kwargs["headers"]
        payload = json.loads(kwargs["content"])
        assert payload["session_id"] == "test-sid"
        assert payload["status"] == "idle"
        assert payload["response"]["content"] == "hi"
        assert "messages" not in payload

    async def test_includes_bearer_token(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        from motus.serve.schemas import WebhookSpec
        from motus.serve.session import Session

        server = AgentServer(_echo, max_workers=1)
        session = Session(session_id="test-sid", status=SessionStatus.idle)
        session.response = ChatMessage.assistant_message(content="hi")
        webhook = WebhookSpec(url="https://example.com/hook", token="my-token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post = AsyncMock(return_value=mock_response)
        with patch("httpx.AsyncClient.post", mock_post):
            await server._deliver_webhook(session, webhook)

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer my-token"

    async def test_includes_messages_when_requested(self):
        import json
        from unittest.mock import AsyncMock, MagicMock, patch

        from motus.serve.schemas import WebhookSpec
        from motus.serve.session import Session

        server = AgentServer(_echo, max_workers=1)
        session = Session(session_id="test-sid", status=SessionStatus.idle)
        session.response = ChatMessage.assistant_message(content="hi")
        session.state = [ChatMessage.user_message("hello"), session.response]
        webhook = WebhookSpec(url="https://example.com/hook", include_messages=True)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post = AsyncMock(return_value=mock_response)
        with patch("httpx.AsyncClient.post", mock_post):
            await server._deliver_webhook(session, webhook)

        payload = json.loads(mock_post.call_args.kwargs["content"])
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][1]["role"] == "assistant"

    async def test_failure_does_not_raise(self):
        from unittest.mock import patch

        from motus.serve.schemas import WebhookSpec
        from motus.serve.session import Session

        server = AgentServer(_echo, max_workers=1)
        session = Session(session_id="test-sid", status=SessionStatus.idle)
        session.response = ChatMessage.assistant_message(content="hi")
        webhook = WebhookSpec(url="https://example.com/hook")

        with patch(
            "httpx.AsyncClient.post", side_effect=Exception("connection refused")
        ):
            # Should not raise
            await server._deliver_webhook(session, webhook)


# ---------------------------------------------------------------------------
# Agent Mode (AgentBase as entrypoint)
# ---------------------------------------------------------------------------

_AGENT_IMPORT_PATH = "tests.unit.serve.mock_agent:echo_agent"


class TestAgentModeWorker:
    """Test WorkerExecutor with an AgentBase instance as entrypoint."""

    @pytest.fixture
    def executor(self):
        return WorkerExecutor(max_workers=2)

    async def test_agent_basic_turn(self, executor):
        msg = ChatMessage.user_message("hello world")
        result = await executor.submit_turn(_AGENT_IMPORT_PATH, msg, [])
        assert result.success
        response, state = result.value
        assert response.content == "echo: hello world"
        assert response.role == "assistant"
        # State should contain user + assistant (no system messages)
        assert len(state) == 2
        assert state[0].role == "user"
        assert state[0].content == "hello world"
        assert state[1].role == "assistant"
        assert state[1].content == "echo: hello world"

    async def test_agent_multi_turn(self, executor):
        # Turn 1
        msg1 = ChatMessage.user_message("first")
        r1 = await executor.submit_turn(_AGENT_IMPORT_PATH, msg1, [])
        assert r1.success
        _, state1 = r1.value

        # Turn 2 — pass state from turn 1
        msg2 = ChatMessage.user_message("second")
        r2 = await executor.submit_turn(_AGENT_IMPORT_PATH, msg2, state1)
        assert r2.success
        response2, state2 = r2.value

        assert response2.content == "echo: second"
        assert len(state2) == 4
        assert state2[0].role == "user"
        assert state2[0].content == "first"
        assert state2[1].role == "assistant"
        assert state2[1].content == "echo: first"
        assert state2[2].role == "user"
        assert state2[2].content == "second"
        assert state2[3].role == "assistant"
        assert state2[3].content == "echo: second"

    async def test_agent_state_has_no_system_messages(self, executor):
        """Agent state should not include system messages to avoid duplication."""
        msg = ChatMessage.user_message("test")
        result = await executor.submit_turn(_AGENT_IMPORT_PATH, msg, [])
        assert result.success
        _, state = result.value
        for m in state:
            assert m.role != "system", "System messages should not leak into state"

    async def test_agent_coexists_with_function_mode(self, executor):
        """Agent mode and function mode work side by side."""
        # Agent mode
        agent_msg = ChatMessage.user_message("agent test")
        agent_result = await executor.submit_turn(_AGENT_IMPORT_PATH, agent_msg, [])
        assert agent_result.success
        assert agent_result.value[0].content == "echo: agent test"

        # Function mode
        fn_msg = ChatMessage.user_message("1 2")
        fn_result = await executor.submit_turn(_import_path(_add), fn_msg, [])
        assert fn_result.success
        assert fn_result.value[0].content == "3"

    async def test_agent_subprocess_isolation(self, executor):
        """Each agent turn runs in a separate process."""
        msg1 = ChatMessage.user_message("a")
        msg2 = ChatMessage.user_message("b")
        t1 = asyncio.create_task(executor.submit_turn(_AGENT_IMPORT_PATH, msg1, []))
        t2 = asyncio.create_task(executor.submit_turn(_AGENT_IMPORT_PATH, msg2, []))
        r1, r2 = await asyncio.gather(t1, t2)
        assert r1.success and r2.success
        # Both succeed independently — fresh agent per subprocess

    async def test_agent_error_propagation(self, executor):
        """Agent _run() raising an exception produces a failed WorkerResult."""
        msg = ChatMessage.user_message("trigger failure")
        result = await executor.submit_turn(
            "tests.unit.serve.mock_agent:failing_agent", msg, []
        )
        assert not result.success
        assert "Intentional agent failure" in result.error

    async def test_non_agent_non_callable_rejected(self, executor):
        """Import path pointing to a plain object produces a failed WorkerResult."""
        msg = ChatMessage.user_message("test")
        result = await executor.submit_turn(
            "tests.unit.serve.mock_agent:not_an_agent", msg, []
        )
        assert not result.success
        assert "expected a ServableAgent, OpenAI Agent, or callable" in result.error


class TestAgentModeResolve:
    """Test that _resolve_import_path and ServableAgent handle Agent instances."""

    def test_resolve_agent_instance(self):
        from motus.serve.protocol import ServableAgent
        from motus.serve.worker import _resolve_import_path

        obj = _resolve_import_path(_AGENT_IMPORT_PATH)
        assert isinstance(obj, ServableAgent)

    def test_resolve_function_still_works(self):
        from motus.serve.protocol import ServableAgent
        from motus.serve.worker import _resolve_import_path

        obj = _resolve_import_path(_import_path(_echo))
        assert not isinstance(obj, ServableAgent)
        assert callable(obj)

    def test_resolve_non_callable_raises(self):
        """Non-agent, non-callable objects are caught in _worker_entry."""
        from motus.serve.worker import _resolve_import_path

        # This will resolve to a string constant or class — just verify it returns
        obj = _resolve_import_path("os:name")
        assert not callable(obj)


# ---------------------------------------------------------------------------
# PUT /sessions/{session_id} — client-specified IDs
# ---------------------------------------------------------------------------


class TestPutSession:
    @pytest.fixture
    def server(self):
        return AgentServer(_add, max_workers=2, allow_custom_ids=True)

    @pytest.fixture
    async def client(self, server):
        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            yield client

    async def test_put_session_creates_with_custom_id(self, client):
        custom_id = str(uuid.uuid4())
        r = await client.put(f"/sessions/{custom_id}")
        assert r.status_code == 201
        data = r.json()
        assert data["session_id"] == custom_id
        assert data["status"] == "idle"
        assert r.headers["location"] == f"/sessions/{custom_id}"

    async def test_put_session_invalid_uuid(self, client):
        r = await client.put("/sessions/not-a-uuid")
        assert r.status_code == 400
        assert "valid UUID" in r.json()["detail"]

    async def test_put_session_duplicate_id(self, client):
        custom_id = str(uuid.uuid4())
        r = await client.put(f"/sessions/{custom_id}")
        assert r.status_code == 201

        r = await client.put(f"/sessions/{custom_id}")
        assert r.status_code == 409
        assert "already exists" in r.json()["detail"]

    async def test_put_session_disabled_by_default(self):
        from httpx import ASGITransport, AsyncClient

        server = AgentServer(_add, max_workers=1)
        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            custom_id = str(uuid.uuid4())
            r = await client.put(f"/sessions/{custom_id}")
            assert r.status_code == 405
            assert "not enabled" in r.json()["detail"]

    async def test_put_session_with_preloaded_state(self, client):
        custom_id = str(uuid.uuid4())
        state = [
            {"role": "user", "content": "1 2"},
            {"role": "assistant", "content": "3"},
        ]
        r = await client.put(f"/sessions/{custom_id}", json={"state": state})
        assert r.status_code == 201

        r = await client.get(f"/sessions/{custom_id}/messages")
        assert r.status_code == 200
        messages = r.json()
        assert len(messages) == 2
        assert messages[0]["content"] == "1 2"
        assert messages[1]["content"] == "3"

    async def test_put_session_max_sessions(self):
        from httpx import ASGITransport, AsyncClient

        server = AgentServer(_add, max_workers=1, max_sessions=1, allow_custom_ids=True)
        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            id1 = str(uuid.uuid4())
            r = await client.put(f"/sessions/{id1}")
            assert r.status_code == 201

            id2 = str(uuid.uuid4())
            r = await client.put(f"/sessions/{id2}")
            assert r.status_code == 503

    async def test_put_session_usable(self, client):
        custom_id = str(uuid.uuid4())
        r = await client.put(f"/sessions/{custom_id}")
        assert r.status_code == 201

        r = await client.post(
            f"/sessions/{custom_id}/messages", json={"content": "3 4"}
        )
        assert r.status_code == 202

        data = await _poll_until(client, custom_id, "idle")
        assert data["response"]["content"] == "7"


class TestSchemas:
    def test_session_status_has_interrupted(self):
        from motus.serve.schemas import SessionStatus

        assert SessionStatus.interrupted.value == "interrupted"

    def test_interrupt_info_model(self):
        from motus.serve.schemas import InterruptInfo

        info = InterruptInfo(
            interrupt_id="abc123",
            type="tool_approval",
            payload={"tool_name": "delete_file", "tool_args": {"path": "/tmp"}},
        )
        assert info.interrupt_id == "abc123"
        assert info.type == "tool_approval"

    def test_resume_request_model(self):
        from motus.serve.schemas import ResumeRequest

        req = ResumeRequest(interrupt_id="abc123", value={"approved": True})
        assert req.interrupt_id == "abc123"
        assert req.value == {"approved": True}

    def test_session_response_has_interrupts_field(self):
        from motus.serve.schemas import SessionResponse, SessionStatus

        resp = SessionResponse(
            session_id="s1",
            status=SessionStatus.idle,
        )
        assert resp.interrupts is None  # default


class TestSessionStoreCustomId:
    def test_create_with_custom_id(self):
        store = SessionStore(ttl=0)
        session = store.create(session_id="custom-id")
        assert session.session_id == "custom-id"
        assert store.get("custom-id") is session

    def test_create_duplicate_raises(self):
        store = SessionStore(ttl=0)
        store.create(session_id="dup-id")
        with pytest.raises(SessionAlreadyExists):
            store.create(session_id="dup-id")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
