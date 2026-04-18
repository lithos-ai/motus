"""Tests for motus.openai_agents.Runner.

Verifies that the motus Runner wraps the OAI SDK Runner with:
  1. Tracing — MotusTracingProcessor is registered on the OAI SDK
  2. Provider injection — MotusOpenAIProvider replaces the default
  3. Tool wrapping — agent tools are wrapped for future motus hooks
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("agents")

_HAD_KEY = "OPENAI_API_KEY" in os.environ
if not _HAD_KEY:
    os.environ["OPENAI_API_KEY"] = "fake-key-for-test"

from agents import Agent  # noqa: E402
from agents import RunConfig as _RunConfig  # noqa: E402

import motus.openai_agents as oai_mod  # noqa: E402

if not _HAD_KEY:
    del os.environ["OPENAI_API_KEY"]
from motus.openai_agents import Runner  # noqa: E402
from motus.openai_agents._motus_provider import MotusOpenAIProvider  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_tracer():
    """Shut down the motus runtime and reset processor flag so each test is fresh."""
    from motus.runtime.agent_runtime import is_initialized, shutdown

    oai_mod._processor_registered = False
    if is_initialized():
        shutdown()
    yield
    oai_mod._processor_registered = False
    if is_initialized():
        shutdown()


@pytest.fixture
def mock_oai_runner():
    mock_result = MagicMock()
    mock_result.final_output = "mock response"
    with patch.object(
        oai_mod._OriginalRunner, "run", new_callable=AsyncMock, return_value=mock_result
    ) as mock_run:
        yield mock_run


@pytest.fixture
def agent():
    return Agent(name="test", instructions="test")


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------


class TestTracing:
    async def test_ensure_tracing_installs_processor(self):
        """_ensure_tracing() gets runtime tracer and sets OAI trace processors."""
        with patch("agents.set_trace_processors") as mock_set:
            oai_mod._ensure_tracing()

            mock_set.assert_called_once()
            processors = mock_set.call_args[0][0]
            assert len(processors) == 1
            assert isinstance(processors[0], oai_mod.MotusTracingProcessor)

    async def test_ensure_tracing_is_idempotent(self):
        """Calling _ensure_tracing() twice doesn't install a second processor."""
        with patch("agents.set_trace_processors") as mock_set:
            oai_mod._ensure_tracing()
            oai_mod._ensure_tracing()

            assert mock_set.call_count == 1

    async def test_runner_calls_ensure_tracing(self, agent, mock_oai_runner):
        """Runner.run() triggers _ensure_tracing()."""
        with patch.object(
            oai_mod, "_ensure_tracing", wraps=oai_mod._ensure_tracing
        ) as spy:
            await Runner.run(agent, "hello")
            spy.assert_called_once()


# ---------------------------------------------------------------------------
# Provider injection
# ---------------------------------------------------------------------------


class TestProviderInjection:
    async def test_default_config_uses_motus_provider(self, agent, mock_oai_runner):
        """When no run_config is passed, Runner injects MotusOpenAIProvider."""
        await Runner.run(agent, "hello")

        _, kwargs = mock_oai_runner.call_args
        config = kwargs["run_config"]
        assert isinstance(config.model_provider, MotusOpenAIProvider)

    async def test_explicit_motus_provider_preserved(self, agent, mock_oai_runner):
        """If the user already passes a MotusOpenAIProvider, it's kept as-is."""
        provider = MotusOpenAIProvider()
        config = _RunConfig(model_provider=provider)

        await Runner.run(agent, "hello", run_config=config)

        _, kwargs = mock_oai_runner.call_args
        assert kwargs["run_config"].model_provider is provider

    async def test_none_config_gets_provider(self, agent, mock_oai_runner):
        """Passing run_config=None is equivalent to omitting it."""
        await Runner.run(agent, "hello", run_config=None)

        _, kwargs = mock_oai_runner.call_args
        assert isinstance(kwargs["run_config"].model_provider, MotusOpenAIProvider)

    async def test_default_openai_provider_upgraded(self, agent, mock_oai_runner):
        """A default OpenAIProvider in run_config is upgraded to MotusOpenAIProvider."""
        from agents.models.openai_provider import OpenAIProvider

        provider = OpenAIProvider()
        config = _RunConfig(model_provider=provider)

        await Runner.run(agent, "hello", run_config=config)

        _, kwargs = mock_oai_runner.call_args
        assert isinstance(kwargs["run_config"].model_provider, MotusOpenAIProvider)

    def test_motus_provider_uses_responses_by_default(self):
        """MotusOpenAIProvider inherits the SDK default (Responses API).

        Both /v1/chat/completions and /v1/responses are supported.
        The SDK default is use_responses=True, producing OpenAIResponsesModel.
        """
        import os

        os.environ.setdefault("OPENAI_API_KEY", "fake-key-for-test")

        provider = MotusOpenAIProvider()
        model = provider.get_model("gpt-4o")
        from motus.openai_agents._motus_model import MotusResponsesModel

        assert isinstance(model, MotusResponsesModel)

    def test_motus_provider_chat_completions_when_requested(self):
        """Users can still force chat/completions with use_responses=False."""
        import os

        os.environ.setdefault("OPENAI_API_KEY", "fake-key-for-test")
        from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

        provider = MotusOpenAIProvider(use_responses=False)
        model = provider.get_model("gpt-4o")
        assert isinstance(model, OpenAIChatCompletionsModel)


# ---------------------------------------------------------------------------
# Tool wrapping
# ---------------------------------------------------------------------------


class TestToolWrapping:
    async def test_tools_wrapped(self, mock_oai_runner):
        """Runner.run() wraps agent tools for motus interception."""
        tool = MagicMock()
        tool.on_invoke_tool = AsyncMock(return_value="result")
        tool.on_invoke_tool._motus_wrapped = False

        agent = Agent(name="test", instructions="test", tools=[tool])

        await Runner.run(agent, "hello")

        # Tool's on_invoke_tool should now be wrapped
        assert getattr(agent.tools[0].on_invoke_tool, "_motus_wrapped", False)

    async def test_tools_not_double_wrapped(self, mock_oai_runner):
        """Calling Runner.run() twice doesn't double-wrap tools."""
        tool = MagicMock()
        tool.on_invoke_tool = AsyncMock(return_value="result")
        tool.on_invoke_tool._motus_wrapped = False

        agent = Agent(name="test", instructions="test", tools=[tool])

        await Runner.run(agent, "hello")
        first_wrapper = agent.tools[0].on_invoke_tool

        await Runner.run(agent, "hello again")
        assert agent.tools[0].on_invoke_tool is first_wrapper


# ---------------------------------------------------------------------------
# Delegation to original Runner
# ---------------------------------------------------------------------------


class TestDelegation:
    async def test_run_delegates_to_original(self, agent, mock_oai_runner):
        """Runner.run() calls the original OAI Runner.run() with the same agent and input."""
        await Runner.run(agent, "hello world")

        mock_oai_runner.assert_awaited_once()
        args = mock_oai_runner.call_args[0]
        assert args[0] is agent
        assert args[1] == "hello world"

    async def test_run_returns_original_result(self, agent, mock_oai_runner):
        """Runner.run() returns the result from the original Runner."""
        result = await Runner.run(agent, "hello")
        assert result.final_output == "mock response"

    async def test_kwargs_forwarded(self, agent, mock_oai_runner):
        """Extra kwargs (max_turns, hooks, etc.) are forwarded."""
        await Runner.run(agent, "hello", max_turns=5)

        _, kwargs = mock_oai_runner.call_args
        assert kwargs["max_turns"] == 5
