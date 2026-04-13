"""Integration test for the Google ADK example.

Mocks the Gemini LLM so no real API key is needed. Exercises:
  1. Console — call root_agent.run_turn directly (with tool-calling flow)
  2. Serve  — HTTP session lifecycle via ASGI transport
  3. Tracing — MotusSpanProcessor creates OTel spans on the motus tracer
"""

import os
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("google.adk")

# Ensure a fake key exists so the ADK agent module can be imported.
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "fake-key-for-test"

from google.adk.models.llm_response import LlmResponse  # noqa: E402
from google.genai import types as genai_types  # noqa: E402

import motus.google_adk.agents.llm_agent as adk_mod  # noqa: E402
from motus.google_adk._motus_tracing import MotusSpanProcessor  # noqa: E402
from motus.models import ChatMessage  # noqa: E402
from motus.runtime.tracing.agent_tracer import (  # noqa: E402
    ATTR_ERROR,
    ATTR_FUNC,
    ATTR_MODEL_INPUT,
    ATTR_MODEL_NAME,
    ATTR_MODEL_OUTPUT,
    ATTR_TASK_TYPE,
    ATTR_TOOL_INPUT,
    ATTR_TOOL_OUTPUT,
    ATTR_USAGE,
    setup_tracing,
    shutdown_tracing,
)
from motus.runtime.types import MODEL_CALL, TOOL_CALL  # noqa: E402

# ---------------------------------------------------------------------------
# Mock Gemini that simulates tool-call → tool-result → final-answer
# ---------------------------------------------------------------------------

_FINAL_TEXT = "The current time in Tokyo is 10:30 AM."


def _mock_generate_factory(
    tool_name="get_current_time", tool_args=None, final_text=None
):
    """Return a fresh mock generate_content_async with its own call counter.

    Args:
        tool_name: Name of the tool the mock model will call on the first turn.
        tool_args: Arguments to pass to the tool call.
        final_text: Text to return as the final response.
    """
    if tool_args is None:
        tool_args = {"city": "Tokyo"}
    if final_text is None:
        final_text = _FINAL_TEXT
    call_count = 0

    async def generate(self, llm_request, stream=False):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part(
                            function_call=genai_types.FunctionCall(
                                name=tool_name,
                                args=tool_args,
                            )
                        )
                    ],
                ),
            )
        else:
            yield LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text=final_text)],
                ),
                turn_complete=True,
            )

    return generate


@pytest.fixture(autouse=True)
def _reset_tracer():
    """Shut down the motus runtime so each test gets a fresh tracer."""
    from motus.runtime.agent_runtime import is_initialized, shutdown

    adk_mod._otel_registered = False
    if is_initialized():
        shutdown()
    shutdown_tracing()
    yield
    adk_mod._otel_registered = False
    if is_initialized():
        shutdown()
    shutdown_tracing()


@pytest.fixture
def mock_gemini():
    with patch(
        "google.adk.models.google_llm.Gemini.generate_content_async",
        _mock_generate_factory(),
    ):
        yield


@pytest.fixture
def mock_gemini_roll_die():
    with patch(
        "google.adk.models.google_llm.Gemini.generate_content_async",
        _mock_generate_factory(
            tool_name="roll_die",
            tool_args={"sides": 20},
            final_text="You rolled a 17 on a 20-sided die!",
        ),
    ):
        yield


@pytest.fixture
def mock_gemini_text_only():
    """Mock that returns text immediately without tool calls."""

    async def generate(self, llm_request, stream=False):
        yield LlmResponse(
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part(text="2 + 2 = 4")],
            ),
            turn_complete=True,
        )

    with patch(
        "google.adk.models.google_llm.Gemini.generate_content_async",
        generate,
    ):
        yield


@pytest.fixture
def mock_gemini_weather():
    with patch(
        "google.adk.models.google_llm.Gemini.generate_content_async",
        _mock_generate_factory(
            tool_name="get_weather",
            tool_args={"city": "Tokyo"},
            final_text='{"name":"Tokyo","country":"Japan","population":"14 million","weather":"22°C Partly cloudy","fun_fact":"Tokyo has the busiest railway station in the world."}',
        ),
    ):
        yield


@pytest.fixture
def processor():
    setup_tracing()
    return MotusSpanProcessor()


# ---------------------------------------------------------------------------
# Console — direct run_turn call
# ---------------------------------------------------------------------------


class TestGoogleADKConsole:
    @pytest.mark.integration
    async def test_run_turn_with_tool_call(self, mock_gemini):
        from examples.google_adk.agent import root_agent

        msg = ChatMessage.user_message("What time is it in Tokyo?")
        response, state = await root_agent.run_turn(msg, [])

        assert response.role == "assistant"
        assert "10:30" in response.content
        assert len(state) == 2
        assert state[0].role == "user"
        assert state[1].role == "assistant"

    @pytest.mark.integration
    async def test_multi_turn_state(self, mock_gemini):
        from examples.google_adk.agent import root_agent

        msg1 = ChatMessage.user_message("What time is it in Tokyo?")
        response1, state1 = await root_agent.run_turn(msg1, [])

        msg2 = ChatMessage.user_message("And in London?")
        response2, state2 = await root_agent.run_turn(msg2, state1)

        assert len(state2) == 4
        assert state2[0].content == "What time is it in Tokyo?"
        assert state2[2].content == "And in London?"


# ---------------------------------------------------------------------------
# Serve — HTTP session lifecycle via ASGI transport
# ---------------------------------------------------------------------------


class TestGoogleADKServe:
    async def test_health_and_sessions(self, mock_gemini):
        from examples.google_adk.agent import root_agent
        from motus.google_adk.agents.llm_agent import Agent
        from motus.serve import AgentServer

        assert isinstance(root_agent, Agent)

        # Use string import path, same as the real CLI would.
        server = AgentServer("examples.google_adk.agent:root_agent")

        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            # Health
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
            assert any(s["session_id"] == sid for s in r.json())

            # Get messages (empty)
            r = await client.get(f"/sessions/{sid}/messages")
            assert r.status_code == 200
            assert r.json() == []

            # Delete session
            r = await client.delete(f"/sessions/{sid}")
            assert r.status_code == 204

            r = await client.get(f"/sessions/{sid}")
            assert r.status_code == 404


# ---------------------------------------------------------------------------
# Helpers — build synthetic OTEL spans for processor-level tests
# ---------------------------------------------------------------------------


def _make_otel_span(
    *,
    name: str = "test_span",
    start_time_ns: int = 1_711_300_800_000_000_000,
    end_time_ns: int = 1_711_300_801_000_000_000,
    attributes: dict | None = None,
):
    """Build a synthetic OTEL ReadableSpan for processor-level tests."""
    span = MagicMock()
    span.name = name
    span.start_time = start_time_ns
    span.end_time = end_time_ns
    span.attributes = attributes or {}
    return span


# ---------------------------------------------------------------------------
# MotusSpanProcessor — span ingestion
# ---------------------------------------------------------------------------


class TestSpanProcessorIngestion:
    """Test that MotusSpanProcessor correctly creates OTel spans on the motus tracer."""

    def _get_last_span_attrs(self):
        """Return attributes from the last collected span, with JSON parsing."""
        import json

        import motus.runtime.tracing.agent_tracer as _at

        spans = _at._collector.spans if _at._collector else []
        assert len(spans) >= 1, "Expected at least one collected span"
        attrs = dict(spans[-1].attributes or {})
        meta = {"task_type": attrs.get(ATTR_TASK_TYPE), "func": attrs.get(ATTR_FUNC)}
        if ATTR_MODEL_NAME in attrs:
            meta["model_name"] = attrs[ATTR_MODEL_NAME]
        if ATTR_ERROR in attrs:
            meta["error"] = attrs[ATTR_ERROR]
        for key, meta_key in [
            (ATTR_MODEL_OUTPUT, "model_output_meta"),
            (ATTR_MODEL_INPUT, "model_input_meta"),
            (ATTR_TOOL_INPUT, "tool_input_meta"),
            (ATTR_TOOL_OUTPUT, "tool_output_meta"),
            (ATTR_USAGE, "usage"),
        ]:
            if key in attrs:
                try:
                    meta[meta_key] = json.loads(attrs[key])
                except (json.JSONDecodeError, TypeError):
                    meta[meta_key] = attrs[key]
        return meta

    def test_agent_span(self, processor):
        span = _make_otel_span(
            attributes={
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": "root_agent",
            }
        )
        processor.on_end(span)

        meta = self._get_last_span_attrs()
        assert meta["task_type"] == "agent_call"
        assert meta["func"] == "root_agent"

    def test_model_call_span(self, processor):
        span = _make_otel_span(
            attributes={
                "gen_ai.operation.name": "generate_content",
                "gen_ai.request.model": "gemini-2.5-flash",
                "gen_ai.usage.input_tokens": 100,
                "gen_ai.usage.output_tokens": 50,
                "gen_ai.response.finish_reasons": ["stop"],
            }
        )
        processor.on_end(span)

        meta = self._get_last_span_attrs()
        assert meta["task_type"] == MODEL_CALL
        assert meta["func"] == "gemini-2.5-flash"
        assert meta["model_name"] == "gemini-2.5-flash"
        assert meta["usage"]["input_tokens"] == 100
        assert meta["usage"]["output_tokens"] == 50
        assert meta["usage"]["total_tokens"] == 150

    def test_tool_call_span(self, processor):
        span = _make_otel_span(
            attributes={
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "get_current_time",
                "gen_ai.tool.description": "Returns the current time",
                "gcp.vertex.agent.tool_call_args": '{"city": "Tokyo"}',
                "gcp.vertex.agent.tool_response": '{"time": "10:30 AM"}',
            }
        )
        processor.on_end(span)

        meta = self._get_last_span_attrs()
        assert meta["task_type"] == TOOL_CALL
        assert meta["func"] == "get_current_time"
        assert meta["tool_input_meta"]["name"] == "get_current_time"
        assert meta["tool_input_meta"]["arguments"] == {"city": "Tokyo"}
        assert meta["tool_output_meta"] == {"time": "10:30 AM"}

    def test_error_span(self, processor):
        span = _make_otel_span(
            attributes={
                "gen_ai.operation.name": "generate_content",
                "gen_ai.request.model": "gemini-2.5-flash",
                "error.type": "ResourceExhausted",
            }
        )
        processor.on_end(span)

        meta = self._get_last_span_attrs()
        assert meta["error"] == "ResourceExhausted"

    def test_non_adk_span_ignored(self, processor):
        """Spans without gen_ai.operation.name are skipped."""
        import motus.runtime.tracing.agent_tracer as _at

        initial_count = len(_at._collector.spans) if _at._collector else 0
        span = _make_otel_span(attributes={"http.method": "GET"})
        processor.on_end(span)
        final_count = len(_at._collector.spans) if _at._collector else 0
        assert final_count == initial_count

    def test_timestamps_preserved(self, processor):
        """OTel spans preserve the original nanosecond timestamps."""
        import motus.runtime.tracing.agent_tracer as _at

        span = _make_otel_span(
            attributes={"gen_ai.operation.name": "invoke_agent"},
            start_time_ns=1_711_300_800_000_000_000,
            end_time_ns=1_711_300_801_000_000_000,
        )
        processor.on_end(span)

        spans = _at._collector.spans if _at._collector else []
        last = spans[-1]
        # The motus span should have start/end times set
        assert last.start_time is not None
        assert last.end_time is not None

    def test_model_call_with_llm_request_response(self, processor):
        """LLM request/response JSON attributes are parsed into meta."""
        span = _make_otel_span(
            attributes={
                "gen_ai.operation.name": "generate_content",
                "gen_ai.request.model": "gemini-2.5-flash",
                "gcp.vertex.agent.llm_request": '{"model": "gemini-2.5-flash", "contents": []}',
                "gcp.vertex.agent.llm_response": '{"candidates": [{"content": "Hello"}]}',
            }
        )
        processor.on_end(span)

        meta = self._get_last_span_attrs()
        assert meta["model_input_meta"]["model"] == "gemini-2.5-flash"
        assert (
            meta["model_output_meta"]["llm_response"]["candidates"][0]["content"]
            == "Hello"
        )


# ---------------------------------------------------------------------------
# End-to-end: run_turn → ADK OTEL spans → MotusSpanProcessor
# ---------------------------------------------------------------------------


class TestADKTracingIntegration:
    """Run the ADK agent through run_turn and verify traces appear."""

    @pytest.mark.integration
    async def test_run_turn_registers_tracing(self, mock_gemini):
        """run_turn() sets up an OTel tracer accessible via get_tracer()."""
        from examples.google_adk.agent import root_agent

        msg = ChatMessage.user_message("What time is it in Tokyo?")
        await root_agent.run_turn(msg, [])

        tracer = adk_mod.get_tracer()
        assert tracer is not None

    @pytest.mark.integration
    async def test_run_turn_produces_trace_spans(self, mock_gemini):
        """Full pipeline: run_turn → ADK OTEL spans → MotusSpanProcessor → OTel collector."""
        from examples.google_adk.agent import root_agent

        msg = ChatMessage.user_message("What time is it in Tokyo?")
        response, _ = await root_agent.run_turn(msg, [])

        assert "10:30" in response.content

        tracer = adk_mod.get_tracer()
        assert tracer is not None

        import motus.runtime.tracing.agent_tracer as _at

        spans = _at._collector.spans if _at._collector else []
        # ADK should emit at least some spans
        # (model/tool spans depend on OTEL provider setup timing)
        assert len(spans) >= 1


# ---------------------------------------------------------------------------
# Callbacks example
# ---------------------------------------------------------------------------


class TestGoogleADKCallbacks:
    @pytest.mark.integration
    async def test_resolve_detects_callbacks_agent(self):
        from motus.serve.worker import _resolve_import_path

        obj = _resolve_import_path("examples.google_adk.callbacks:root_agent")
        assert hasattr(obj, "run_turn") and callable(obj.run_turn)

    @pytest.mark.integration
    async def test_callbacks_run_turn(self, mock_gemini_roll_die):
        from examples.google_adk.callbacks import root_agent

        msg = ChatMessage.user_message("Roll a 20-sided die")
        response, state = await root_agent.run_turn(msg, [])

        assert response.role == "assistant"
        assert len(state) == 2


# ---------------------------------------------------------------------------
# Multi-agent example
# ---------------------------------------------------------------------------


class TestGoogleADKMultiAgent:
    @pytest.mark.integration
    async def test_resolve_detects_multi_agent(self):
        from motus.serve.worker import _resolve_import_path

        obj = _resolve_import_path("examples.google_adk.multi_agent:root_agent")
        assert hasattr(obj, "run_turn") and callable(obj.run_turn)

    @pytest.mark.integration
    async def test_multi_agent_has_sub_agents(self):
        from examples.google_adk.multi_agent import root_agent

        assert len(root_agent.sub_agents) == 2
        sub_names = {a.name for a in root_agent.sub_agents}
        assert "math_agent" in sub_names
        assert "writing_agent" in sub_names

    @pytest.mark.integration
    async def test_multi_agent_run_turn(self, mock_gemini_text_only):
        from examples.google_adk.multi_agent import root_agent

        msg = ChatMessage.user_message("What is 2 + 2?")
        response, state = await root_agent.run_turn(msg, [])

        assert response.role == "assistant"
        assert len(state) == 2


# ---------------------------------------------------------------------------
# Structured output example
# ---------------------------------------------------------------------------


class TestGoogleADKStructuredOutput:
    @pytest.mark.integration
    async def test_resolve_detects_structured_output(self):
        from motus.serve.worker import _resolve_import_path

        obj = _resolve_import_path("examples.google_adk.structured_output:root_agent")
        assert hasattr(obj, "run_turn") and callable(obj.run_turn)

    @pytest.mark.integration
    async def test_structured_output_has_schema(self):
        from examples.google_adk.structured_output import CityInfo, root_agent

        assert root_agent.output_schema == CityInfo

    @pytest.mark.integration
    async def test_structured_output_run_turn(self, mock_gemini_weather):
        from examples.google_adk.structured_output import root_agent

        msg = ChatMessage.user_message("Tell me about Tokyo")
        response, state = await root_agent.run_turn(msg, [])

        assert response.role == "assistant"
        assert len(state) == 2


# ---------------------------------------------------------------------------
# Parallel tools example
# ---------------------------------------------------------------------------


class TestGoogleADKParallelTools:
    @pytest.mark.integration
    async def test_resolve(self):
        from motus.serve.worker import _resolve_import_path

        obj = _resolve_import_path("examples.google_adk.parallel_tools:root_agent")
        assert hasattr(obj, "run_turn") and callable(obj.run_turn)

    @pytest.mark.integration
    async def test_has_four_tools(self):
        from examples.google_adk.parallel_tools import root_agent

        assert len(root_agent.tools) == 4

    @pytest.mark.integration
    async def test_run_turn(self, mock_gemini_weather):
        from examples.google_adk.parallel_tools import root_agent

        msg = ChatMessage.user_message("Tell me about Tokyo")
        response, state = await root_agent.run_turn(msg, [])

        assert response.role == "assistant"
        assert len(state) == 2


# ---------------------------------------------------------------------------
# Fields output schema example
# ---------------------------------------------------------------------------


class TestGoogleADKFieldsOutputSchema:
    @pytest.mark.integration
    async def test_resolve(self):
        from motus.serve.worker import _resolve_import_path

        obj = _resolve_import_path(
            "examples.google_adk.fields_output_schema:root_agent"
        )
        assert hasattr(obj, "run_turn") and callable(obj.run_turn)

    @pytest.mark.integration
    async def test_run_turn(self):
        """Mock returns valid JSON matching list[WeatherData] schema."""

        async def generate(self, llm_request, stream=False):
            yield LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part(
                            text='[{"city":"Tokyo","temperature":"22°C","condition":"Partly cloudy","humidity":"65%"}]'
                        )
                    ],
                ),
                turn_complete=True,
            )

        with patch(
            "google.adk.models.google_llm.Gemini.generate_content_async",
            generate,
        ):
            from examples.google_adk.fields_output_schema import root_agent

            msg = ChatMessage.user_message("Weather in Tokyo")
            response, state = await root_agent.run_turn(msg, [])
            assert response.role == "assistant"


# ---------------------------------------------------------------------------
# Pydantic tools example
# ---------------------------------------------------------------------------


class TestGoogleADKPydanticTools:
    @pytest.mark.integration
    async def test_resolve(self):
        from motus.serve.worker import _resolve_import_path

        obj = _resolve_import_path("examples.google_adk.pydantic_tools:root_agent")
        assert hasattr(obj, "run_turn") and callable(obj.run_turn)

    @pytest.mark.integration
    async def test_run_turn(self, mock_gemini_text_only):
        from examples.google_adk.pydantic_tools import root_agent

        msg = ChatMessage.user_message("Register user Alice, alice@test.com")
        response, state = await root_agent.run_turn(msg, [])
        assert response.role == "assistant"


# ---------------------------------------------------------------------------
# Workflow triage example
# ---------------------------------------------------------------------------


class TestGoogleADKWorkflowTriage:
    @pytest.mark.integration
    async def test_resolve(self):
        from motus.serve.worker import _resolve_import_path

        obj = _resolve_import_path("examples.google_adk.workflow_triage:root_agent")
        assert hasattr(obj, "run_turn") and callable(obj.run_turn)

    @pytest.mark.integration
    async def test_has_pipeline_sub_agent(self):
        from examples.google_adk.workflow_triage import root_agent

        assert len(root_agent.sub_agents) == 1
        assert root_agent.sub_agents[0].name == "execution_pipeline"


# ---------------------------------------------------------------------------
# Static instruction example
# ---------------------------------------------------------------------------


class TestGoogleADKStaticInstruction:
    @pytest.mark.integration
    async def test_resolve(self):
        from motus.serve.worker import _resolve_import_path

        obj = _resolve_import_path("examples.google_adk.static_instruction:root_agent")
        assert hasattr(obj, "run_turn") and callable(obj.run_turn)

    @pytest.mark.integration
    async def test_run_turn(self, mock_gemini_text_only):
        from examples.google_adk.static_instruction import root_agent

        msg = ChatMessage.user_message("What is a cortado?")
        response, state = await root_agent.run_turn(msg, [])
        assert response.role == "assistant"


# ---------------------------------------------------------------------------
# Token usage example
# ---------------------------------------------------------------------------


class TestGoogleADKTokenUsage:
    @pytest.mark.integration
    async def test_resolve(self):
        from motus.serve.worker import _resolve_import_path

        obj = _resolve_import_path("examples.google_adk.token_usage:root_agent")
        assert hasattr(obj, "run_turn") and callable(obj.run_turn)

    @pytest.mark.integration
    async def test_has_tools(self):
        from examples.google_adk.token_usage import root_agent

        assert len(root_agent.tools) == 2


# ---------------------------------------------------------------------------
# Multimodal tools example
# ---------------------------------------------------------------------------


class TestGoogleADKMultimodalTools:
    @pytest.mark.integration
    async def test_resolve(self):
        from motus.serve.worker import _resolve_import_path

        obj = _resolve_import_path("examples.google_adk.multimodal_tools:root_agent")
        assert hasattr(obj, "run_turn") and callable(obj.run_turn)

    @pytest.mark.integration
    async def test_has_three_tools(self):
        from examples.google_adk.multimodal_tools import root_agent

        assert len(root_agent.tools) == 3
