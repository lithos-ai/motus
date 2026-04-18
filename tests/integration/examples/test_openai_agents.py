"""Integration tests for the OpenAI Agents SDK hello_world example.

Exercises:
  1. Console — serve worker wraps the OAI Agent with _adapt_openai_agent
  2. Serve  — HTTP session lifecycle via ASGI transport
  3. Tracing — MotusTracingProcessor creates OTel spans on the motus tracer
"""

import os
from unittest.mock import MagicMock

import pytest

pytest.importorskip("agents")

# Ensure a fake key exists so the agent module can be imported.
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "fake-key-for-test"

import motus.openai_agents as oai_mod  # noqa: E402
from motus.models import ChatMessage  # noqa: E402
from motus.openai_agents import Runner  # noqa: E402
from motus.openai_agents._motus_tracing import (  # noqa: E402
    MotusTracingProcessor,
    _iso_to_ns,
)
from motus.runtime.types import MODEL_CALL, TOOL_CALL  # noqa: E402
from motus.tracing.agent_tracer import (  # noqa: E402
    ATTR_AGENT_ID,
    ATTR_ERROR,
    ATTR_FUNC,
    ATTR_MODEL_INPUT,
    ATTR_MODEL_NAME,
    ATTR_MODEL_OUTPUT,
    ATTR_TASK_TYPE,
    ATTR_TOOL_INPUT,
    ATTR_USAGE,
    setup_tracing,
    shutdown_tracing,
)

# ---------------------------------------------------------------------------
# Helpers — build synthetic OAI SDK span objects for processor-level tests
# ---------------------------------------------------------------------------


def _make_span(
    *,
    span_id: str,
    parent_id: str | None = None,
    span_type: str = "agent",
    started_at: str = "2026-03-24T12:00:00+00:00",
    ended_at: str = "2026-03-24T12:00:01+00:00",
    error: dict | None = None,
    **data_attrs,
):
    """Build a synthetic OAI SDK Span for processor-level tests."""
    span_data = MagicMock()
    span_data.type = span_type
    for k, v in data_attrs.items():
        setattr(span_data, k, v)
    span_data.export.return_value = {"type": span_type, **data_attrs}

    span = MagicMock()
    span.span_id = span_id
    span.parent_id = parent_id
    span.span_data = span_data
    span.started_at = started_at
    span.ended_at = ended_at
    span.error = error
    return span


def _make_oai_runner_mock():
    """Create _OriginalRunner.run mock that emits real OAI SDK trace spans.

    When Runner.run calls _OriginalRunner.run, this mock fires real OAI SDK
    tracing context managers (trace + agent_span + generation_span) so the
    registered MotusTracingProcessor receives genuine span events.
    """
    from agents.tracing import agent_span, generation_span, trace

    mock_result = MagicMock()
    mock_result.final_output = "The Western Roman Empire fell in 476 AD."

    async def _fake_run(agent, input, **kwargs):
        with trace(workflow_name=agent.name):
            with agent_span(name=agent.name):
                with generation_span(
                    model="test-model",
                    input=[{"role": "user", "content": str(input)}],
                ):
                    pass  # Span auto-completes on exit
        return mock_result

    return _fake_run, mock_result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_tracer():
    """Shut down the motus runtime so each test gets a fresh tracer."""
    from motus.runtime.agent_runtime import is_initialized, shutdown

    oai_mod._processor_registered = False
    if is_initialized():
        shutdown()
    shutdown_tracing()
    yield
    oai_mod._processor_registered = False
    if is_initialized():
        shutdown()
    shutdown_tracing()


@pytest.fixture
def processor():
    setup_tracing()
    return MotusTracingProcessor()


@pytest.fixture
def oai_runner_with_spans(monkeypatch):
    """Patch _OriginalRunner.run with a mock that emits real OAI SDK spans."""
    fake_run, mock_result = _make_oai_runner_mock()
    monkeypatch.setattr(oai_mod._OriginalRunner, "run", fake_run)
    return mock_result


# ---------------------------------------------------------------------------
# Console — serve worker auto-wraps the OAI Agent
# ---------------------------------------------------------------------------


class TestOpenAIAgentsConsole:
    @pytest.mark.integration
    async def test_resolve_wraps_oai_agent(self):
        """serve worker detects an OAI Agent and adapts it to a run_turn function."""
        import inspect

        from motus.serve.worker import (
            _adapt_openai_agent,
            _is_openai_agent,
            _resolve_import_path,
        )

        obj = _resolve_import_path("examples.openai_agents.tools:agent")
        assert _is_openai_agent(obj)
        fn = _adapt_openai_agent(obj)
        assert callable(fn)
        assert inspect.iscoroutinefunction(fn)

    @pytest.mark.integration
    async def test_run_turn(self, oai_runner_with_spans):
        from motus.serve.worker import _adapt_openai_agent, _resolve_import_path

        obj = _resolve_import_path("examples.openai_agents.tools:agent")
        run_turn = _adapt_openai_agent(obj)

        msg = ChatMessage.user_message("When did the Roman Empire fall?")
        response, state = await run_turn(msg, [])

        assert response.role == "assistant"
        assert response.content and len(response.content) > 0
        assert len(state) == 2
        assert state[0].role == "user"
        assert state[1].role == "assistant"

    @pytest.mark.integration
    async def test_multi_turn_state(self, oai_runner_with_spans):
        from motus.serve.worker import _adapt_openai_agent, _resolve_import_path

        obj = _resolve_import_path("examples.openai_agents.tools:agent")
        run_turn = _adapt_openai_agent(obj)

        msg1 = ChatMessage.user_message("When did the Roman Empire fall?")
        _, state1 = await run_turn(msg1, [])

        msg2 = ChatMessage.user_message("What caused it?")
        _, state2 = await run_turn(msg2, state1)

        assert len(state2) == 4
        assert state2[0].content == "When did the Roman Empire fall?"
        assert state2[2].content == "What caused it?"


# ---------------------------------------------------------------------------
# Serve — HTTP session lifecycle via ASGI transport
# ---------------------------------------------------------------------------


class TestOpenAIAgentsServe:
    async def test_health_and_sessions(self):
        from motus.serve import AgentServer

        server = AgentServer("examples.openai_agents.tools:agent")

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


# ---------------------------------------------------------------------------
# MotusTracingProcessor — span ingestion
# ---------------------------------------------------------------------------


class TestTracingProcessorSpanIngestion:
    """Test that MotusTracingProcessor correctly creates OTel spans on the motus tracer."""

    def _get_last_span_attrs(self):
        """Return attributes dict from the last collected span."""
        import json

        import motus.tracing.agent_tracer as _at

        spans = _at._collector.spans if _at._collector else []
        assert len(spans) >= 1, "Expected at least one collected span"
        attrs = dict(spans[-1].attributes or {})
        # Parse JSON-encoded attributes back to dicts
        meta = {"task_type": attrs.get(ATTR_TASK_TYPE), "func": attrs.get(ATTR_FUNC)}
        if ATTR_MODEL_NAME in attrs:
            meta["model_name"] = attrs[ATTR_MODEL_NAME]
        if ATTR_ERROR in attrs:
            meta["error"] = attrs[ATTR_ERROR]
        if ATTR_AGENT_ID in attrs:
            meta["agent_id"] = attrs[ATTR_AGENT_ID]
        for key, meta_key in [
            (ATTR_MODEL_OUTPUT, "model_output_meta"),
            (ATTR_MODEL_INPUT, "model_input_meta"),
            (ATTR_TOOL_INPUT, "tool_input_meta"),
            (ATTR_USAGE, "usage"),
        ]:
            if key in attrs:
                try:
                    meta[meta_key] = json.loads(attrs[key])
                except (json.JSONDecodeError, TypeError):
                    meta[meta_key] = attrs[key]
        return meta

    def test_agent_span(self, processor):
        """An 'agent' span is ingested with task_type='agent_call'."""
        span = _make_span(span_id="s1", span_type="agent", name="History Tutor")

        processor.on_span_start(span)
        processor.on_span_end(span)

        meta = self._get_last_span_attrs()
        assert meta["task_type"] == "agent_call"
        assert meta["func"] == "History Tutor"

    def test_generation_span(self, processor):
        """A 'generation' span is ingested as MODEL_CALL with model name."""
        span = _make_span(
            span_id="s2",
            span_type="generation",
            model="gpt-4o",
            input=[{"role": "user", "content": "Hello"}],
            output=[{"role": "assistant", "content": "Hi there"}],
            usage={"input_tokens": 10, "output_tokens": 5},
        )

        processor.on_span_start(span)
        processor.on_span_end(span)

        meta = self._get_last_span_attrs()
        assert meta["task_type"] == MODEL_CALL
        assert meta["func"] == "gpt-4o"
        assert meta["model_name"] == "gpt-4o"
        assert "model_output_meta" in meta
        assert meta["model_output_meta"]["content"] == "Hi there"
        assert "model_input_meta" in meta

    def test_function_span(self, processor):
        """A 'function' span is ingested as TOOL_CALL."""
        span = _make_span(
            span_id="s3",
            span_type="function",
            name="get_weather",
            input='{"city": "Rome"}',
        )

        processor.on_span_start(span)
        processor.on_span_end(span)

        meta = self._get_last_span_attrs()
        assert meta["task_type"] == TOOL_CALL
        assert meta["func"] == "get_weather"
        assert meta["tool_input_meta"]["name"] == "get_weather"

    def test_response_span(self, processor):
        """A 'response' span (Responses API) is ingested as MODEL_CALL."""
        response_obj = MagicMock()
        response_obj.model = "gpt-4o-mini"
        response_obj.usage = MagicMock(
            input_tokens=20, output_tokens=10, total_tokens=30
        )
        text_part = MagicMock(type="output_text", text="The answer is 42.")
        msg_item = MagicMock(type="message", content=[text_part])
        response_obj.output = [msg_item]

        span = _make_span(span_id="s4", span_type="response", response=response_obj)

        processor.on_span_start(span)
        processor.on_span_end(span)

        meta = self._get_last_span_attrs()
        assert meta["task_type"] == MODEL_CALL
        assert meta["func"] == "gpt-4o-mini"
        assert meta["model_name"] == "gpt-4o-mini"
        assert meta["usage"]["total_tokens"] == 30
        assert meta["model_output_meta"]["content"] == "The answer is 42."

    def test_error_span(self, processor):
        """A span with an error records the error in meta."""
        span = _make_span(
            span_id="s5",
            span_type="generation",
            model="gpt-4o",
            error={"message": "Rate limit exceeded"},
        )

        processor.on_span_start(span)
        processor.on_span_end(span)

        meta = self._get_last_span_attrs()
        assert meta["error"] == "Rate limit exceeded"

    def test_handoff_span(self, processor):
        span = _make_span(span_id="s6", span_type="handoff", to_agent="Math Tutor")

        processor.on_span_start(span)
        processor.on_span_end(span)

        meta = self._get_last_span_attrs()
        assert meta["task_type"] == "handoff"
        assert "Math Tutor" in meta["func"]

    def test_guardrail_span(self, processor):
        span = _make_span(span_id="s7", span_type="guardrail", name="content_filter")

        processor.on_span_start(span)
        processor.on_span_end(span)

        meta = self._get_last_span_attrs()
        assert meta["task_type"] == "guardrail"
        assert meta["func"] == "content_filter"


# ---------------------------------------------------------------------------
# Parent-child span tree
# ---------------------------------------------------------------------------


class TestSpanParentResolution:
    """Test that processor creates OTel spans for parent and child spans."""

    def test_parent_and_child_spans_created(self, processor):
        """Both parent and child OAI spans produce OTel spans."""
        import motus.tracing.agent_tracer as _at

        parent = _make_span(span_id="p1", span_type="agent", name="Tutor")
        child = _make_span(
            span_id="c1", parent_id="p1", span_type="generation", model="gpt-4o"
        )

        processor.on_span_start(parent)
        processor.on_span_start(child)
        processor.on_span_end(child)
        processor.on_span_end(parent)

        spans = _at._collector.spans if _at._collector else []
        assert len(spans) >= 2
        task_types = {(s.attributes or {}).get(ATTR_TASK_TYPE) for s in spans}
        assert "agent_call" in task_types
        assert MODEL_CALL in task_types

    def test_multiple_children_created(self, processor):
        """Multiple child spans are all created as OTel spans."""
        import motus.tracing.agent_tracer as _at

        parent = _make_span(span_id="p2", span_type="agent", name="Tutor")
        child1 = _make_span(
            span_id="c2a", parent_id="p2", span_type="generation", model="gpt-4o"
        )
        child2 = _make_span(
            span_id="c2b", parent_id="p2", span_type="function", name="lookup"
        )

        processor.on_span_start(parent)
        processor.on_span_start(child1)
        processor.on_span_start(child2)
        processor.on_span_end(child1)
        processor.on_span_end(child2)
        processor.on_span_end(parent)

        spans = _at._collector.spans if _at._collector else []
        assert len(spans) >= 3
        func_names = {(s.attributes or {}).get(ATTR_FUNC) for s in spans}
        assert "Tutor" in func_names
        assert "gpt-4o" in func_names
        assert "lookup" in func_names

    def test_root_span_created(self, processor):
        """A root span (no parent_id) is created as an OTel span."""
        import motus.tracing.agent_tracer as _at

        span = _make_span(span_id="root", span_type="agent", name="Root")

        processor.on_span_start(span)
        processor.on_span_end(span)

        spans = _at._collector.spans if _at._collector else []
        assert len(spans) >= 1
        last = spans[-1]
        assert (last.attributes or {}).get(ATTR_TASK_TYPE) == "agent_call"
        assert (last.attributes or {}).get(ATTR_FUNC) == "Root"


# ---------------------------------------------------------------------------
# Span ID cleanup
# ---------------------------------------------------------------------------


class TestSpanIdCleanup:
    def test_span_processed_on_end(self, processor):
        """Spans are processed on on_span_end (no internal state to clean up)."""
        import motus.tracing.agent_tracer as _at

        initial_count = len(_at._collector.spans) if _at._collector else 0

        span = _make_span(span_id="cleanup1", span_type="agent", name="Test")

        processor.on_span_start(span)
        # on_span_start is a no-op in the new processor
        current_count = len(_at._collector.spans) if _at._collector else 0
        assert current_count == initial_count

        processor.on_span_end(span)
        # on_span_end should create an OTel span
        final_count = len(_at._collector.spans) if _at._collector else 0
        assert final_count == initial_count + 1


# ---------------------------------------------------------------------------
# _iso_to_ns helper
# ---------------------------------------------------------------------------


class TestIsoToNs:
    def test_valid_iso(self):
        ns = _iso_to_ns("2026-03-24T12:00:00+00:00")
        assert ns > 0

    def test_none_returns_zero(self):
        assert _iso_to_ns(None) == 0

    def test_empty_string_returns_zero(self):
        assert _iso_to_ns("") == 0

    def test_invalid_string_returns_zero(self):
        assert _iso_to_ns("not-a-date") == 0


# ---------------------------------------------------------------------------
# End-to-end: Runner.run → real OAI SDK spans → MotusTracingProcessor
# ---------------------------------------------------------------------------


class TestRunnerTracingIntegration:
    """Verify the full tracing pipeline using real OAI SDK tracing internals.

    _OriginalRunner.run is replaced with a mock that emits real OAI SDK
    trace/span context managers. This exercises:
      register_tracing() → set_trace_processors([MotusTracingProcessor])
      → OAI SDK span lifecycle → on_span_start/on_span_end callbacks
      → OTel spans on motus tracer
    """

    def _get_collected_span_types(self):
        """Return set of task_type values from collected OTel spans."""
        import motus.tracing.agent_tracer as _at

        spans = _at._collector.spans if _at._collector else []
        return {(s.attributes or {}).get(ATTR_TASK_TYPE) for s in spans}

    @pytest.mark.integration
    async def test_run_produces_trace_spans(self, oai_runner_with_spans):
        """Runner.run emits OAI SDK spans that land as OTel spans."""
        from agents import Agent

        agent = Agent(
            name="History Tutor",
            instructions="You answer history questions clearly and concisely.",
        )
        result = await Runner.run(agent, "When did the Roman Empire fall?")

        assert result.final_output and len(result.final_output) > 0

        import motus.tracing.agent_tracer as _at

        spans = _at._collector.spans if _at._collector else []
        assert len(spans) >= 2

        types = self._get_collected_span_types()
        assert "agent_call" in types
        assert MODEL_CALL in types

    @pytest.mark.integration
    async def test_serve_run_turn_with_tracing(self, oai_runner_with_spans):
        """serve's _adapt_openai_agent path produces real trace spans."""
        from motus.serve.worker import _adapt_openai_agent, _resolve_import_path

        obj = _resolve_import_path("examples.openai_agents.tools:agent")
        run_turn = _adapt_openai_agent(obj)

        msg = ChatMessage.user_message("When did the Roman Empire fall?")
        response, state = await run_turn(msg, [])

        assert response.role == "assistant"
        assert response.content and len(response.content) > 0

        import motus.tracing.agent_tracer as _at

        spans = _at._collector.spans if _at._collector else []
        assert len(spans) >= 2

        types = self._get_collected_span_types()
        assert "agent_call" in types
        assert MODEL_CALL in types
