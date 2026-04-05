"""Tests for AgentTool — wrapping an agent as a tool."""

import json

import pytest

from motus.agent.base_agent import AgentBase
from motus.tools import AgentTool, normalize_tools
from motus.tools.core.agent_tool import _DefaultInput

# ---------------------------------------------------------------------------
# Minimal concrete agent for testing
# ---------------------------------------------------------------------------


class _EchoAgent(AgentBase[str]):
    """Minimal agent that echoes the user prompt."""

    def __init__(self, *, echo_prefix: str = "echo", **kwargs):
        self._echo_prefix = echo_prefix
        super().__init__(**kwargs)

    async def _run(self, user_prompt=None, **kwargs) -> str:
        return f"{self._echo_prefix}: {user_prompt}"


def _make_echo_agent(**overrides) -> _EchoAgent:
    """Helper to create a minimal _EchoAgent with a mock client."""
    from unittest.mock import Mock

    defaults = dict(
        client=Mock(),
        model_name="mock",
        name="echo_agent",
        system_prompt="You echo.",
    )
    defaults.update(overrides)
    return _EchoAgent(**defaults)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestAgentToolConstruction:
    def test_rejects_non_agent(self):
        with pytest.raises(TypeError, match="AgentTool requires an AgentBase"):
            AgentTool("not an agent")

    def test_default_schema(self):
        agent = _make_echo_agent()
        tool = AgentTool(agent)
        assert tool.json_schema == _DefaultInput.model_json_schema()
        assert "request" in tool.json_schema["properties"]

    def test_name_and_description_from_agent(self):
        agent = _make_echo_agent(name="researcher")
        tool = AgentTool(agent)
        assert tool.name == "researcher"
        assert tool.description == "Delegate to sub-agent: researcher"

    def test_custom_name_and_description(self):
        agent = _make_echo_agent()
        tool = AgentTool(agent, name="custom", description="Custom desc")
        assert tool.name == "custom"
        assert tool.description == "Custom desc"

    def test_schema_is_always_default_input(self):
        agent = _make_echo_agent()
        tool = AgentTool(agent)
        assert tool.json_schema == _DefaultInput.model_json_schema()
        assert list(tool.json_schema["properties"].keys()) == ["request"]


# ---------------------------------------------------------------------------
# Invocation
# ---------------------------------------------------------------------------


class TestAgentToolInvocation:
    def test_basic_call(self):
        agent = _make_echo_agent()
        tool = AgentTool(agent)
        result = tool(json.dumps({"request": "hello world"})).af_result()
        assert "echo: hello world" in result

    def test_stateless_does_not_mutate_template(self):
        """Stateless mode must fork — template agent's state stays unchanged."""
        state_log = []

        class _MutatingAgent(AgentBase[str]):
            async def _run(self, user_prompt=None, **kwargs) -> str:
                await self.add_user_message(user_prompt or "")
                state_log.append(len(self.memory.messages))
                return "ok"

        from unittest.mock import Mock

        agent = _MutatingAgent(client=Mock(), model_name="m", name="mut")
        original_count = len(agent.memory.messages)
        tool = AgentTool(agent, stateful=False)
        tool(json.dumps({"request": "test"})).af_result()
        # Fork was used — template agent's memory is untouched
        assert len(agent.memory.messages) == original_count
        # The forked agent did mutate its own memory
        assert state_log == [1]

    def test_stateful_reuses_agent(self):
        """Stateful mode uses the same agent instance (no fork)."""
        call_log = []

        class _TrackingAgent(AgentBase[str]):
            async def _run(self, user_prompt=None, **kwargs) -> str:
                call_log.append(id(self))
                return "ok"

        from unittest.mock import Mock

        agent = _TrackingAgent(client=Mock(), model_name="m", name="tracked")
        tool = AgentTool(agent, stateful=True)
        tool(json.dumps({"request": "a"})).af_result()
        tool(json.dumps({"request": "b"})).af_result()
        # Both calls should use the exact same agent instance
        assert len(call_log) == 2
        assert call_log[0] == call_log[1] == id(agent)

    def test_output_extractor(self):
        agent = _make_echo_agent()
        tool = AgentTool(
            agent,
            output_extractor=lambda result: result.upper(),
        )
        result = tool(json.dumps({"request": "hi"})).af_result()
        assert "ECHO: HI" in result

    def test_max_steps_override(self):
        """max_steps override applies to forked agent, not the template."""
        observed = []

        class _StepsAgent(AgentBase[str]):
            async def _run(self, user_prompt=None, **kwargs) -> str:
                observed.append(self.max_steps)
                return "ok"

        from unittest.mock import Mock

        agent = _StepsAgent(client=Mock(), model_name="m", name="steps")
        assert agent.max_steps is None
        tool = AgentTool(agent, max_steps=3)
        tool(json.dumps({"request": "test"})).af_result()
        # Template agent unchanged
        assert agent.max_steps is None
        # Forked agent received the override
        assert observed == [3]

    def test_error_in_agent_returns_error_json(self):
        """Tool._execute catches exceptions from _invoke and returns error JSON."""

        class _FailAgent(AgentBase[str]):
            async def _run(self, user_prompt=None, **kwargs) -> str:
                raise RuntimeError("boom")

        from unittest.mock import Mock

        agent = _FailAgent(client=Mock(), model_name="m", name="fail")
        tool = AgentTool(agent)
        result = tool(json.dumps({"request": "go"})).af_result()
        parsed = json.loads(result)
        assert "error" in parsed
        assert "boom" in parsed["error"]


# ---------------------------------------------------------------------------
# AgentBase.as_tool()
# ---------------------------------------------------------------------------


class TestAsToolMethod:
    def test_as_tool_returns_agent_tool(self):
        agent = _make_echo_agent()
        tool = agent.as_tool(name="helper", description="A helper")
        assert isinstance(tool, AgentTool)
        assert tool.name == "helper"
        assert tool.description == "A helper"

    def test_as_tool_with_defaults(self):
        agent = _make_echo_agent(name="mybot")
        tool = agent.as_tool()
        assert tool.name == "mybot"

    def test_as_tool_call(self):
        agent = _make_echo_agent()
        tool = agent.as_tool()
        result = tool(json.dumps({"request": "ping"})).af_result()
        assert "echo: ping" in result


# ---------------------------------------------------------------------------
# normalize_tools integration
# ---------------------------------------------------------------------------


class TestNormalizeToolsAgent:
    def test_agent_in_list(self):
        agent = _make_echo_agent(name="research")
        tools = normalize_tools([agent])
        assert "research" in tools

    def test_agent_as_single_item(self):
        agent = _make_echo_agent(name="research")
        tools = normalize_tools(agent)
        assert "research" in tools

    def test_agent_mixed_with_functions(self):
        agent = _make_echo_agent(name="research")

        async def ping() -> str:
            return "pong"

        tools = normalize_tools([agent, ping])
        assert "research" in tools
        assert "ping" in tools

    def test_duplicate_agent_name_raises(self):
        a1 = _make_echo_agent(name="same")
        a2 = _make_echo_agent(name="same")
        with pytest.raises(ValueError, match="Duplicate tool name"):
            normalize_tools([a1, a2])
