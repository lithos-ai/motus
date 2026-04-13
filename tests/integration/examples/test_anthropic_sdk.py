"""Integration tests for the Anthropic SDK example.

Mocks the Anthropic API so no real API key is needed. Exercises:
  1. Console — call ToolRunner.run_turn directly (with tool-calling flow)
  2. Serve  — HTTP session lifecycle via ASGI transport
  3. Tracing — span building helpers and runner-level span ingestion
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

# Fake ANTHROPIC_API_KEY is injected by conftest.py at module level
# so the Anthropic client can be constructed during collection.
import motus.anthropic as anthropic_mod  # noqa: E402
from motus.anthropic._motus_tracing import (  # noqa: E402
    _now_ns,
    emit_model_span,
    emit_tool_span,
    start_agent_span,
)
from motus.models import ChatMessage  # noqa: E402
from motus.runtime.tracing.agent_tracer import (  # noqa: E402
    ATTR_FUNC,
    ATTR_MODEL_NAME,
    ATTR_TASK_TYPE,
    ATTR_TOOL_INPUT,
    ATTR_USAGE,
    setup_tracing,
    shutdown_tracing,
)
from motus.runtime.types import AGENT_CALL, MODEL_CALL, TOOL_CALL  # noqa: E402

# ---------------------------------------------------------------------------
# Mock Anthropic API — simulates tool-call → tool-result → final-answer
# ---------------------------------------------------------------------------


class _Usage:
    """Non-callable usage mock (MagicMock is callable, which breaks _get_last_message)."""

    def __init__(
        self, *, input_tokens, output_tokens, cache_creation=None, cache_read=None
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation
        self.cache_read_input_tokens = cache_read


class _TextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _ToolUseBlock:
    def __init__(self, *, name: str, input: dict, id: str = "toolu_01"):
        self.type = "tool_use"
        self.name = name
        self.input = input
        self.id = id


class _BetaMessage:
    """Non-callable message mock. Must not be callable — the SDK's
    _get_last_message() treats callable _last_message as a deferred getter."""

    def __init__(
        self,
        *,
        content_blocks,
        stop_reason="end_turn",
        model="claude-sonnet-4-20250514",
        input_tokens=100,
        output_tokens=50,
    ):
        self.role = "assistant"
        self.content = content_blocks
        self.usage = _Usage(input_tokens=input_tokens, output_tokens=output_tokens)
        self.model = model
        self.stop_reason = stop_reason
        self.container = None


def _make_beta_message(**kwargs):
    return _BetaMessage(**kwargs)


def _make_text_block(text: str):
    return _TextBlock(text)


def _make_tool_use_block(*, name: str, input: dict, id: str = "toolu_01"):
    return _ToolUseBlock(name=name, input=input, id=id)


def _mock_parse_factory():
    """Return a mock for client.beta.messages.parse that simulates a tool-calling flow.

    First call: model requests get_weather tool
    Second call: model returns final text answer
    """
    call_count = 0

    async def _mock_parse(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_beta_message(
                content_blocks=[
                    _make_tool_use_block(
                        name="get_weather",
                        input={"location": "Paris", "unit": "celsius"},
                        id="toolu_01",
                    ),
                    _make_tool_use_block(
                        name="calculate_sum",
                        input={"a": 15, "b": 27},
                        id="toolu_02",
                    ),
                ],
                stop_reason="tool_use",
            )
        else:
            return _make_beta_message(
                content_blocks=[
                    _make_text_block(
                        "The weather in Paris is 20°C and sunny. Also, 15 + 27 = 42."
                    )
                ],
                stop_reason="end_turn",
            )

    return _mock_parse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_tracer():
    """Shut down the motus runtime so each test gets a fresh tracer."""
    from motus.runtime.agent_runtime import is_initialized, shutdown

    if is_initialized():
        shutdown()
    shutdown_tracing()
    yield
    if is_initialized():
        shutdown()
    shutdown_tracing()


@pytest.fixture
def mock_anthropic():
    """Patch AsyncAnthropic so beta.messages.parse returns mock responses."""
    mock_parse = _mock_parse_factory()
    with patch("motus.anthropic.AsyncAnthropic") as MockClient:
        instance = MockClient.return_value
        instance.beta.messages.parse = AsyncMock(side_effect=mock_parse)
        yield instance


@pytest.fixture
def tool_runner(mock_anthropic):
    """A ToolRunner with mock tools, ready for run_turn."""
    from anthropic.lib.tools import beta_async_tool

    @beta_async_tool
    async def get_weather(location: str, unit: str = "fahrenheit") -> str:
        """Get the current weather.

        Args:
            location: City and state
            unit: Temperature unit
        """
        return '{"temperature": "20°C", "condition": "Sunny"}'

    @beta_async_tool
    async def calculate_sum(a: int, b: int) -> str:
        """Add two numbers.

        Args:
            a: First number
            b: Second number
        """
        return str(a + b)

    return anthropic_mod.ToolRunner(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=[get_weather, calculate_sum],
        system="You are a helpful assistant.",
    )


# ---------------------------------------------------------------------------
# Console — direct run_turn call
# ---------------------------------------------------------------------------


class TestAnthropicSDKConsole:
    @pytest.mark.integration
    async def test_resolve_detects_tool_runner(self):
        """serve worker resolves the example and detects it via run_turn duck-typing."""
        from motus.serve.worker import _resolve_import_path

        obj = _resolve_import_path("examples.anthropic.tools_runner:runner")
        assert hasattr(obj, "run_turn") and callable(obj.run_turn)

    @pytest.mark.integration
    async def test_run_turn_with_tool_calls(self, tool_runner):
        msg = ChatMessage.user_message(
            "What's the weather in Paris? Also, what's 15 + 27?"
        )
        response, state = await tool_runner.run_turn(msg, [])

        assert response.role == "assistant"
        assert "20°C" in response.content
        assert "42" in response.content
        assert len(state) == 2
        assert state[0].role == "user"
        assert state[1].role == "assistant"

    @pytest.mark.integration
    async def test_multi_turn_state(self, tool_runner):
        msg1 = ChatMessage.user_message("What's the weather in Paris?")
        _, state1 = await tool_runner.run_turn(msg1, [])

        msg2 = ChatMessage.user_message("And in London?")
        _, state2 = await tool_runner.run_turn(msg2, state1)

        assert len(state2) == 4
        assert state2[0].content == "What's the weather in Paris?"
        assert state2[2].content == "And in London?"


# ---------------------------------------------------------------------------
# Serve — HTTP session lifecycle via ASGI transport
# ---------------------------------------------------------------------------


def _make_in_process_submit(mock_parse_factory):
    """Return a patched submit_turn that runs in-process with a mocked Anthropic API.

    The real WorkerExecutor spawns a subprocess where test mocks are absent.
    This replacement runs the agent in the test process so mocks take effect.
    """

    async def _in_process_submit(
        self, import_path, message, state, *, timeout=0, session_id=None, **kwargs
    ):
        from motus.serve.worker import (
            WorkerResult,
            _is_openai_agent,
            _resolve_import_path,
            _validate_result,
        )

        mock_parse = mock_parse_factory()
        with patch("motus.anthropic.AsyncAnthropic") as MockClient:
            instance = MockClient.return_value
            instance.beta.messages.parse = AsyncMock(side_effect=mock_parse)

            try:
                agent_or_fn = _resolve_import_path(import_path)
                if hasattr(agent_or_fn, "run_turn") and callable(agent_or_fn.run_turn):
                    result = await agent_or_fn.run_turn(message, state)
                elif _is_openai_agent(agent_or_fn):
                    from motus.serve.worker import _adapt_openai_agent

                    adapted = _adapt_openai_agent(agent_or_fn)
                    result = await adapted(message, state)
                elif callable(agent_or_fn):
                    import inspect

                    if inspect.iscoroutinefunction(agent_or_fn):
                        result = await agent_or_fn(message, state)
                    else:
                        result = agent_or_fn(message, state)
                else:
                    raise TypeError(f"Cannot run {type(agent_or_fn)}")
                response, new_state = _validate_result(result)
                return WorkerResult(success=True, value=(response, new_state))
            except Exception:
                import traceback

                return WorkerResult(success=False, error=traceback.format_exc())

    return _in_process_submit


async def _poll_session(client, sid, *, timeout=5.0, interval=0.05):
    """Poll GET /sessions/{sid} until status is not 'running'."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = await client.get(f"/sessions/{sid}")
        assert r.status_code == 200
        data = r.json()
        if data["status"] != "running":
            return data
        await asyncio.sleep(interval)
    raise TimeoutError(f"Session {sid} still running after {timeout}s")


class TestAnthropicSDKServe:
    async def test_health_and_sessions(self):
        from motus.serve import AgentServer

        server = AgentServer("examples.anthropic.tools_runner:runner")

        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            r = await client.get("/health")
            assert r.status_code == 200
            assert r.json()["status"] == "ok"

            r = await client.post("/sessions")
            assert r.status_code == 201
            sid = r.json()["session_id"]
            assert r.json()["status"] == "idle"

            r = await client.get("/sessions")
            assert r.status_code == 200
            assert any(s["session_id"] == sid for s in r.json())

            r = await client.get(f"/sessions/{sid}/messages")
            assert r.status_code == 200
            assert r.json() == []

            r = await client.delete(f"/sessions/{sid}")
            assert r.status_code == 204

            r = await client.get(f"/sessions/{sid}")
            assert r.status_code == 404

    @pytest.mark.integration
    async def test_tools_runner_message_round_trip(self):
        """Send a message through serve and get a tool-augmented response."""
        from motus.serve import AgentServer
        from motus.serve.worker import WorkerExecutor

        server = AgentServer("examples.anthropic.tools_runner:runner")

        from httpx import ASGITransport, AsyncClient

        with patch.object(
            WorkerExecutor,
            "submit_turn",
            _make_in_process_submit(_mock_parse_factory),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=server.app), base_url="http://test"
            ) as client:
                r = await client.post("/sessions")
                sid = r.json()["session_id"]

                r = await client.post(
                    f"/sessions/{sid}/messages",
                    json={"content": "What's the weather in Paris?"},
                )
                assert r.status_code == 202

                data = await _poll_session(client, sid)
                assert data["status"] == "idle"
                assert data["response"] is not None
                assert "20°C" in data["response"]["content"]

                r = await client.get(f"/sessions/{sid}/messages")
                assert r.status_code == 200
                messages = r.json()
                assert len(messages) == 2
                assert messages[0]["role"] == "user"
                assert messages[1]["role"] == "assistant"

    @pytest.mark.integration
    async def test_search_tool_message_round_trip(self):
        """Send a message through serve with the search_tool example."""
        from motus.serve import AgentServer
        from motus.serve.worker import WorkerExecutor

        server = AgentServer("examples.anthropic.search_tool:runner")

        from httpx import ASGITransport, AsyncClient

        with patch.object(
            WorkerExecutor,
            "submit_turn",
            _make_in_process_submit(_mock_search_tool_parse_factory),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=server.app), base_url="http://test"
            ) as client:
                r = await client.post("/sessions")
                sid = r.json()["session_id"]

                r = await client.post(
                    f"/sessions/{sid}/messages",
                    json={"content": "What is the weather in SF?"},
                )
                assert r.status_code == 202

                data = await _poll_session(client, sid)
                assert data["status"] == "idle"
                assert data["response"] is not None
                assert "San Francisco" in data["response"]["content"]

    @pytest.mark.integration
    async def test_mcp_runner_message_round_trip(self):
        """Send a message through serve with a ToolRunner using MCP-style tools.

        Uses a custom in-process submit that builds a ToolRunner with a
        FakeMCPTool, since the real MCP example needs external dependencies.
        """
        from motus.serve import AgentServer
        from motus.serve.worker import WorkerExecutor, WorkerResult, _validate_result

        # Custom submit that builds a ToolRunner with a fake MCP tool
        async def _mcp_submit(
            self, import_path, message, state, *, timeout=0, session_id=None, **kwargs
        ):
            mock_parse = _mock_mcp_parse_factory()
            with patch("motus.anthropic.AsyncAnthropic") as MockClient:
                instance = MockClient.return_value
                instance.beta.messages.parse = AsyncMock(side_effect=mock_parse)

                fake_tool = _FakeMCPTool(
                    name="list_directory",
                    description="List files in a directory",
                    input_schema={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                )
                runner = anthropic_mod.ToolRunner(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1024,
                    tools=[fake_tool],
                )
                try:
                    result = await runner.run_turn(message, state)
                    response, new_state = _validate_result(result)
                    return WorkerResult(success=True, value=(response, new_state))
                except Exception:
                    import traceback

                    return WorkerResult(success=False, error=traceback.format_exc())

        # Use tools_runner path for AgentServer (it just needs a valid import)
        server = AgentServer("examples.anthropic.tools_runner:runner")

        from httpx import ASGITransport, AsyncClient

        with patch.object(WorkerExecutor, "submit_turn", _mcp_submit):
            async with AsyncClient(
                transport=ASGITransport(app=server.app), base_url="http://test"
            ) as client:
                r = await client.post("/sessions")
                sid = r.json()["session_id"]

                r = await client.post(
                    f"/sessions/{sid}/messages",
                    json={"content": "List the files in /tmp"},
                )
                assert r.status_code == 202

                data = await _poll_session(client, sid)
                assert data["status"] == "idle"
                assert data["response"] is not None
                assert "readme.txt" in data["response"]["content"]


# ---------------------------------------------------------------------------
# Tracing helpers — build_*_meta functions
# ---------------------------------------------------------------------------


class TestTracingHelpers:
    def test_now_ns_returns_positive_int(self):
        ns = _now_ns()
        assert isinstance(ns, int)
        assert ns > 0

    def test_start_agent_span(self):
        setup_tracing()
        span = start_agent_span(model="claude-sonnet-4-20250514")
        assert span is not None
        # Span should be recording (not yet ended)
        assert span.is_recording()
        span.end()

    def test_emit_model_span(self):
        setup_tracing()
        message = _make_beta_message(
            content_blocks=[_make_text_block("Hello!")],
            input_tokens=50,
            output_tokens=20,
        )
        # Should not raise
        emit_model_span(
            message=message,
            model="claude-sonnet-4-20250514",
            input_messages=[{"role": "user", "content": "Hi"}],
            start_ns=1000,
            end_ns=2000,
        )

    def test_emit_model_span_with_tool_use(self):
        setup_tracing()
        message = _make_beta_message(
            content_blocks=[
                _make_tool_use_block(name="get_weather", input={"city": "Paris"})
            ],
        )
        # Should not raise
        emit_model_span(
            message=message,
            model="claude-sonnet-4-20250514",
            input_messages=None,
            start_ns=1000,
            end_ns=2000,
        )

    def test_emit_model_span_none_message(self):
        setup_tracing()
        # Should not raise with None message
        emit_model_span(
            message=None,
            model="claude-sonnet-4-20250514",
            input_messages=None,
            start_ns=1000,
            end_ns=2000,
        )

    def test_emit_tool_span(self):
        setup_tracing()
        # Should not raise
        emit_tool_span(
            tool_name="get_weather",
            tool_input={"city": "Paris"},
            tool_output='{"temp": "20°C"}',
            start_ns=1000,
            end_ns=1500,
        )

    def test_emit_tool_span_with_error(self):
        setup_tracing()
        # Should not raise
        emit_tool_span(
            tool_name="get_weather",
            tool_input={"city": "Paris"},
            start_ns=1000,
            end_ns=1500,
            error="ConnectionError",
        )


# ---------------------------------------------------------------------------
# Runner tracing — model_call and tool_call span ingestion
# ---------------------------------------------------------------------------


class TestRunnerTracingIntegration:
    """Verify the full tracing pipeline: ToolRunner.run_turn → OTel spans → collector."""

    def _get_collected_spans(self):
        """Return spans from the OfflineSpanCollector."""
        import json

        import motus.runtime.tracing.agent_tracer as _at

        spans = _at._collector.spans if _at._collector else []
        result = []
        for s in spans:
            attrs = dict(s.attributes or {})
            # Parse JSON-encoded attributes back to dicts for easier assertions
            meta = {
                "task_type": attrs.get(ATTR_TASK_TYPE),
                "func": attrs.get(ATTR_FUNC),
            }
            if ATTR_USAGE in attrs:
                try:
                    meta["usage"] = json.loads(attrs[ATTR_USAGE])
                except (json.JSONDecodeError, TypeError):
                    pass
            if ATTR_TOOL_INPUT in attrs:
                try:
                    meta["tool_input_meta"] = json.loads(attrs[ATTR_TOOL_INPUT])
                except (json.JSONDecodeError, TypeError):
                    pass
            if ATTR_MODEL_NAME in attrs:
                meta["model_name"] = attrs[ATTR_MODEL_NAME]
            result.append(meta)
        return result

    @pytest.mark.integration
    async def test_run_turn_produces_trace_spans(self, tool_runner):
        msg = ChatMessage.user_message("What's the weather in Paris?")
        await tool_runner.run_turn(msg, [])

        tracer = anthropic_mod.get_tracer()
        assert tracer is not None

        span_metas = self._get_collected_spans()
        assert len(span_metas) >= 3  # agent + model + tool(s)

        types = {m["task_type"] for m in span_metas}
        assert AGENT_CALL in types
        assert MODEL_CALL in types
        assert TOOL_CALL in types

    @pytest.mark.integration
    async def test_model_call_has_usage(self, tool_runner):
        msg = ChatMessage.user_message("What's the weather in Paris?")
        await tool_runner.run_turn(msg, [])

        span_metas = self._get_collected_spans()
        model_metas = [m for m in span_metas if m["task_type"] == MODEL_CALL]
        assert len(model_metas) >= 1
        for meta in model_metas:
            assert "usage" in meta
            assert meta["usage"]["input_tokens"] == 100
            assert meta["usage"]["output_tokens"] == 50

    @pytest.mark.integration
    async def test_tool_call_has_input_meta(self, tool_runner):
        msg = ChatMessage.user_message("What's the weather in Paris?")
        await tool_runner.run_turn(msg, [])

        span_metas = self._get_collected_spans()
        tool_metas = [m for m in span_metas if m["task_type"] == TOOL_CALL]
        assert len(tool_metas) >= 2

        tool_names = {m["func"] for m in tool_metas}
        assert "get_weather" in tool_names
        assert "calculate_sum" in tool_names

        for meta in tool_metas:
            assert "tool_input_meta" in meta
            assert "name" in meta["tool_input_meta"]

    @pytest.mark.integration
    async def test_serve_run_turn_with_tracing(self, mock_anthropic):
        """serve's duck-typed run_turn path produces real trace spans."""
        from motus.serve.worker import _resolve_import_path

        obj = _resolve_import_path("examples.anthropic.tools_runner:runner")
        assert hasattr(obj, "run_turn")

        msg = ChatMessage.user_message("What's the weather in Paris?")
        response, state = await obj.run_turn(msg, [])

        assert response.role == "assistant"
        assert response.content and len(response.content) > 0

        tracer = anthropic_mod.get_tracer()
        assert tracer is not None

        span_metas = self._get_collected_spans()
        assert len(span_metas) >= 3

        types = {m["task_type"] for m in span_metas}
        assert AGENT_CALL in types
        assert MODEL_CALL in types
        assert TOOL_CALL in types


# ---------------------------------------------------------------------------
# Search tool example — deferred loading with tool-search beta
# ---------------------------------------------------------------------------


def _mock_search_tool_parse_factory():
    """Mock parse for the search-tool flow.

    Turn 1: model calls search_available_tools
    Turn 2: model calls get_weather (discovered via search)
    Turn 3: model returns final text answer
    """
    call_count = 0

    async def _mock_parse(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_beta_message(
                content_blocks=[
                    _make_tool_use_block(
                        name="search_available_tools",
                        input={"keyword": "weather"},
                        id="toolu_search_01",
                    ),
                ],
                stop_reason="tool_use",
            )
        elif call_count == 2:
            return _make_beta_message(
                content_blocks=[
                    _make_tool_use_block(
                        name="get_weather",
                        input={"location": "San Francisco, CA", "units": "c"},
                        id="toolu_weather_01",
                    ),
                ],
                stop_reason="tool_use",
            )
        else:
            return _make_beta_message(
                content_blocks=[
                    _make_text_block("The weather in San Francisco is 20°C and sunny.")
                ],
                stop_reason="end_turn",
            )

    return _mock_parse


@pytest.fixture
def mock_anthropic_search():
    """Patch AsyncAnthropic for the search-tool flow."""
    mock_parse = _mock_search_tool_parse_factory()
    with patch("motus.anthropic.AsyncAnthropic") as MockClient:
        instance = MockClient.return_value
        instance.beta.messages.parse = AsyncMock(side_effect=mock_parse)
        yield instance


class TestSearchToolExample:
    @pytest.mark.integration
    async def test_resolve_detects_search_tool_runner(self):
        """serve worker resolves the search_tool example."""
        from motus.serve.worker import _resolve_import_path

        obj = _resolve_import_path("examples.anthropic.search_tool:runner")
        assert hasattr(obj, "run_turn") and callable(obj.run_turn)

    @pytest.mark.integration
    async def test_search_tool_run_turn(self, mock_anthropic_search):
        """Search tool flow: search → discover → call → answer."""
        from examples.anthropic.search_tool import runner

        msg = ChatMessage.user_message("What is the weather in San Francisco?")
        response, state = await runner.run_turn(msg, [])

        assert response.role == "assistant"
        assert "San Francisco" in response.content
        assert len(state) == 2

    @pytest.mark.integration
    async def test_search_tool_betas_passed(self, mock_anthropic_search):
        """Verify betas param is forwarded to the API call."""
        from examples.anthropic.search_tool import runner

        msg = ChatMessage.user_message("What is the weather in SF?")
        await runner.run_turn(msg, [])

        # The mock was called — check that betas was in the kwargs
        call_kwargs = mock_anthropic_search.beta.messages.parse.call_args_list[0].kwargs
        assert "betas" in call_kwargs
        assert "tool-search-tool-2025-10-19" in call_kwargs["betas"]

    @pytest.mark.integration
    async def test_search_tool_tracing(self, mock_anthropic_search):
        """Search tool flow produces trace spans for all tool calls."""

        import motus.runtime.tracing.agent_tracer as _at
        from examples.anthropic.search_tool import runner

        msg = ChatMessage.user_message("What is the weather in SF?")
        await runner.run_turn(msg, [])

        tracer = anthropic_mod.get_tracer()
        assert tracer is not None

        spans = _at._collector.spans if _at._collector else []
        tool_spans = [
            s for s in spans if (s.attributes or {}).get(ATTR_TASK_TYPE) == TOOL_CALL
        ]
        tool_names = {(s.attributes or {}).get(ATTR_FUNC) for s in tool_spans}
        assert "search_available_tools" in tool_names
        assert "get_weather" in tool_names


# ---------------------------------------------------------------------------
# MCP tool runner example — import and structure validation
# ---------------------------------------------------------------------------


class _FakeMCPTool:
    """Simulates a BetaAsyncBuiltinFunctionTool as produced by async_mcp_tool.

    This lets us test the ToolRunner ↔ MCP-style tool integration without
    requiring the ``mcp`` package or an actual MCP server.
    """

    def __init__(self, *, name: str, description: str, input_schema: dict):
        self._name = name
        self._description = description
        self._input_schema = input_schema

    @property
    def name(self) -> str:
        return self._name

    def to_dict(self) -> dict:
        return {
            "name": self._name,
            "description": self._description,
            "input_schema": self._input_schema,
        }

    async def call(self, input: object) -> str:
        return "files: readme.txt, data.csv"


# Register as a virtual subclass so isinstance() checks pass in ToolRunner
from anthropic.lib.tools import BetaAsyncBuiltinFunctionTool

BetaAsyncBuiltinFunctionTool.register(_FakeMCPTool)


def _mock_mcp_parse_factory():
    """Mock parse for the MCP tool runner flow.

    Turn 1: model calls list_directory tool
    Turn 2: model returns final text answer
    """
    call_count = 0

    async def _mock_parse(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_beta_message(
                content_blocks=[
                    _make_tool_use_block(
                        name="list_directory",
                        input={"path": "/tmp"},
                        id="toolu_mcp_01",
                    ),
                ],
                stop_reason="tool_use",
            )
        else:
            return _make_beta_message(
                content_blocks=[
                    _make_text_block("The files in /tmp are: readme.txt, data.csv")
                ],
                stop_reason="end_turn",
            )

    return _mock_parse
