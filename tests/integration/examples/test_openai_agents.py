"""Integration tests for the OpenAI Agents SDK hello_world example.

Exercises:
  1. Console — serve worker wraps the OAI Agent with _adapt_openai_agent
  2. Serve  — HTTP session lifecycle via ASGI transport
  3. Tracing — MotusTracingProcessor ingests real OAI SDK spans into TraceManager
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
    _iso_to_us,
)
from motus.runtime.tracing.agent_tracer import TraceManager  # noqa: E402
from motus.runtime.types import MODEL_CALL, TOOL_CALL  # noqa: E402

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
    """Reset module-level _tracer so each test gets a fresh TraceManager."""
    saved = oai_mod._tracer
    oai_mod._tracer = None
    yield
    oai_mod._tracer = saved


@pytest.fixture
def trace_manager():
    """A real TraceManager with collection enabled."""
    return TraceManager()


@pytest.fixture
def processor(trace_manager):
    return MotusTracingProcessor(trace_manager)


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
    """Test that MotusTracingProcessor correctly ingests spans into TraceManager."""

    def test_agent_span(self, processor, trace_manager):
        """An 'agent' span is ingested with task_type='agent_call'."""
        span = _make_span(span_id="s1", span_type="agent", name="History Tutor")

        processor.on_span_start(span)
        processor.on_span_end(span)

        assert len(trace_manager.task_meta) == 1
        meta = next(iter(trace_manager.task_meta.values()))
        assert meta["task_type"] == "agent_call"
        assert meta["func"] == "History Tutor"

    def test_generation_span(self, processor, trace_manager):
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

        meta = next(iter(trace_manager.task_meta.values()))
        assert meta["task_type"] == MODEL_CALL
        assert meta["func"] == "gpt-4o"
        assert meta["model_name"] == "gpt-4o"
        assert "model_output_meta" in meta
        assert meta["model_output_meta"]["content"] == "Hi there"
        assert "model_input_meta" in meta

    def test_function_span(self, processor, trace_manager):
        """A 'function' span is ingested as TOOL_CALL."""
        span = _make_span(
            span_id="s3",
            span_type="function",
            name="get_weather",
            input='{"city": "Rome"}',
        )

        processor.on_span_start(span)
        processor.on_span_end(span)

        meta = next(iter(trace_manager.task_meta.values()))
        assert meta["task_type"] == TOOL_CALL
        assert meta["func"] == "get_weather"
        assert meta["tool_input_meta"]["name"] == "get_weather"

    def test_response_span(self, processor, trace_manager):
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

        meta = next(iter(trace_manager.task_meta.values()))
        assert meta["task_type"] == MODEL_CALL
        assert meta["func"] == "gpt-4o-mini"
        assert meta["model_name"] == "gpt-4o-mini"
        assert meta["usage"]["total_tokens"] == 30
        assert meta["model_output_meta"]["content"] == "The answer is 42."

    def test_error_span(self, processor, trace_manager):
        """A span with an error records the error in meta."""
        span = _make_span(
            span_id="s5",
            span_type="generation",
            model="gpt-4o",
            error={"message": "Rate limit exceeded"},
        )

        processor.on_span_start(span)
        processor.on_span_end(span)

        meta = next(iter(trace_manager.task_meta.values()))
        assert meta["error"] == "Rate limit exceeded"

    def test_handoff_span(self, processor, trace_manager):
        span = _make_span(span_id="s6", span_type="handoff", to_agent="Math Tutor")

        processor.on_span_start(span)
        processor.on_span_end(span)

        meta = next(iter(trace_manager.task_meta.values()))
        assert meta["task_type"] == "handoff"
        assert "Math Tutor" in meta["func"]

    def test_guardrail_span(self, processor, trace_manager):
        span = _make_span(span_id="s7", span_type="guardrail", name="content_filter")

        processor.on_span_start(span)
        processor.on_span_end(span)

        meta = next(iter(trace_manager.task_meta.values()))
        assert meta["task_type"] == "guardrail"
        assert meta["func"] == "content_filter"


# ---------------------------------------------------------------------------
# Parent-child span tree
# ---------------------------------------------------------------------------


class TestSpanParentResolution:
    """Test that parent-child relationships are correctly resolved."""

    def test_child_links_to_parent(self, processor, trace_manager):
        """A child span's parent field resolves to the parent's task_id."""
        parent = _make_span(span_id="p1", span_type="agent", name="Tutor")
        child = _make_span(
            span_id="c1", parent_id="p1", span_type="generation", model="gpt-4o"
        )

        processor.on_span_start(parent)
        processor.on_span_start(child)
        processor.on_span_end(child)
        processor.on_span_end(parent)

        assert len(trace_manager.task_meta) == 2
        child_meta = [
            m for m in trace_manager.task_meta.values() if m["task_type"] == MODEL_CALL
        ][0]
        parent_tid = [
            tid
            for tid, m in trace_manager.task_meta.items()
            if m["task_type"] == "agent_call"
        ][0]
        assert child_meta["parent"] == parent_tid

    def test_span_tree_populated(self, processor, trace_manager):
        """TraceManager.task_span_tree has parent -> [children] entries."""
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

        parent_tid = [
            tid for tid, m in trace_manager.task_meta.items() if m["func"] == "Tutor"
        ][0]
        assert parent_tid in trace_manager.task_span_tree
        assert len(trace_manager.task_span_tree[parent_tid]) == 2

    def test_orphan_span_has_no_parent(self, processor, trace_manager):
        """A root span (no parent_id) has parent=None in meta."""
        span = _make_span(span_id="root", span_type="agent", name="Root")

        processor.on_span_start(span)
        processor.on_span_end(span)

        meta = next(iter(trace_manager.task_meta.values()))
        assert meta["parent"] is None


# ---------------------------------------------------------------------------
# Span ID cleanup
# ---------------------------------------------------------------------------


class TestSpanIdCleanup:
    def test_span_id_removed_after_end(self, processor, trace_manager):
        """Span IDs are cleaned up from the internal map after on_span_end."""
        span = _make_span(span_id="cleanup1", span_type="agent", name="Test")

        processor.on_span_start(span)
        assert "cleanup1" in processor._span_id_map

        processor.on_span_end(span)
        assert "cleanup1" not in processor._span_id_map


# ---------------------------------------------------------------------------
# _iso_to_us helper
# ---------------------------------------------------------------------------


class TestIsoToUs:
    def test_valid_iso(self):
        us = _iso_to_us("2026-03-24T12:00:00+00:00")
        assert us > 0

    def test_none_returns_zero(self):
        assert _iso_to_us(None) == 0

    def test_empty_string_returns_zero(self):
        assert _iso_to_us("") == 0

    def test_invalid_string_returns_zero(self):
        assert _iso_to_us("not-a-date") == 0


# ---------------------------------------------------------------------------
# End-to-end: Runner.run → real OAI SDK spans → MotusTracingProcessor
# ---------------------------------------------------------------------------


class TestRunnerTracingIntegration:
    """Verify the full tracing pipeline using real OAI SDK tracing internals.

    _OriginalRunner.run is replaced with a mock that emits real OAI SDK
    trace/span context managers. This exercises:
      register_tracing() → set_trace_processors([MotusTracingProcessor])
      → OAI SDK span lifecycle → on_span_start/on_span_end callbacks
      → TraceManager.ingest_external_span()
    """

    @pytest.mark.integration
    async def test_run_produces_trace_spans(self, oai_runner_with_spans):
        """Runner.run emits OAI SDK spans that land in TraceManager."""
        from agents import Agent

        agent = Agent(
            name="History Tutor",
            instructions="You answer history questions clearly and concisely.",
        )
        result = await Runner.run(agent, "When did the Roman Empire fall?")

        assert result.final_output and len(result.final_output) > 0

        tracer = oai_mod.get_tracer()
        assert tracer is not None
        assert isinstance(tracer, TraceManager)
        assert len(tracer.task_meta) >= 2

        types = {m["task_type"] for m in tracer.task_meta.values()}
        assert "agent_call" in types
        assert MODEL_CALL in types

        # Model call should be a child of the agent span
        agent_tid = next(
            tid for tid, m in tracer.task_meta.items() if m["task_type"] == "agent_call"
        )
        model_metas = [
            m for m in tracer.task_meta.values() if m["task_type"] == MODEL_CALL
        ]
        assert all(m["parent"] == agent_tid for m in model_metas)

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

        tracer = oai_mod.get_tracer()
        assert tracer is not None
        assert isinstance(tracer, TraceManager)
        assert len(tracer.task_meta) >= 2

        types = {m["task_type"] for m in tracer.task_meta.values()}
        assert "agent_call" in types
        assert MODEL_CALL in types
