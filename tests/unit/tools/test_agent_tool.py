"""Tests for AgentTool — wrapping an agent as a tool."""

import asyncio
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
        result = asyncio.run(tool(json.dumps({"request": "hello world"})))
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
        asyncio.run(tool(json.dumps({"request": "test"})))
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
        asyncio.run(tool(json.dumps({"request": "a"})))
        asyncio.run(tool(json.dumps({"request": "b"})))
        # Both calls should use the exact same agent instance
        assert len(call_log) == 2
        assert call_log[0] == call_log[1] == id(agent)

    def test_output_extractor(self):
        agent = _make_echo_agent()
        tool = AgentTool(
            agent,
            output_extractor=lambda result: result.upper(),
        )
        result = asyncio.run(tool(json.dumps({"request": "hi"})))
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
        asyncio.run(tool(json.dumps({"request": "test"})))
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
        result = asyncio.run(tool(json.dumps({"request": "go"})))
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
        result = asyncio.run(tool(json.dumps({"request": "ping"})))
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


# ---------------------------------------------------------------------------
# Streaming attribution: AgentTool pushes the registered tool name onto the
# _agent_path contextvar so the child agent's messages are tagged with the
# call chain that produced them.
# ---------------------------------------------------------------------------


class _StreamingAgent(AgentBase[str]):
    """Adds a single assistant message during _run, then returns."""

    async def _run(self, user_prompt=None, **kwargs) -> str:
        if user_prompt:
            await self.add_user_message(user_prompt)
        await self.add_assistant_message(f"reply from {self.name}")
        return f"reply from {self.name}"


class _AwaitingStreamingAgent(AgentBase[str]):
    """Adds a marker message after a yield to the loop (interleaving aid)."""

    async def _run(self, user_prompt=None, **kwargs) -> str:
        import asyncio

        await asyncio.sleep(0)
        await self.add_assistant_message(f"reply from {self.name}")
        await asyncio.sleep(0)
        return f"reply from {self.name}"


class _NestedAgent(AgentBase[str]):
    """Calls a single subagent tool, then adds its own message."""

    async def _run(self, user_prompt=None, **kwargs) -> str:
        for tool in self._tools.values():
            await tool._invoke(request=user_prompt or "")
        await self.add_assistant_message(f"outer from {self.name}")
        return f"outer from {self.name}"


class TestAgentToolStreaming:
    """Verify that AgentTool propagates streaming callbacks and pushes the
    correct agent_path onto the contextvar around the child agent's run."""

    async def test_subagent_messages_carry_path(self):
        """Single-level subagent: messages tagged with [tool_name]."""
        from unittest.mock import Mock

        from motus.agent._stream_context import _agent_path, _stream_callback

        received: list[tuple[str, tuple[str, ...]]] = []

        async def cb(msg):
            received.append((msg.content, _agent_path.get()))

        token = _stream_callback.set(cb)
        try:
            inner = _StreamingAgent(client=Mock(), model_name="m", name="inner")
            tool = inner.as_tool(name="research")
            await tool._invoke(request="hi")
        finally:
            _stream_callback.reset(token)

        assert received == [
            ("hi", ("research",)),
            ("reply from inner", ("research",)),
        ]

    async def test_nested_subagents_accumulate_path(self):
        """Subagent inside subagent: innermost path is [outer_tool, inner_tool]."""
        from unittest.mock import Mock

        from motus.agent._stream_context import _agent_path, _stream_callback

        received: list[tuple[str, tuple[str, ...]]] = []

        async def cb(msg):
            received.append((msg.content, _agent_path.get()))

        token = _stream_callback.set(cb)
        try:
            leaf = _StreamingAgent(client=Mock(), model_name="m", name="leaf")
            outer = _NestedAgent(
                client=Mock(),
                model_name="m",
                name="outer",
                tools=[leaf.as_tool(name="leaf_tool")],
            )
            outer_tool = outer.as_tool(name="branch_tool")
            await outer_tool._invoke(request="go")
        finally:
            _stream_callback.reset(token)

        contents = {(c, p) for c, p in received}
        # Inner agent's reply runs under both path components
        assert ("reply from leaf", ("branch_tool", "leaf_tool")) in contents
        # Outer's own reply runs under just its tool name
        assert ("outer from outer", ("branch_tool",)) in contents

    async def test_parallel_subagents_have_independent_paths(self):
        """Two AgentTools fired under asyncio.gather get independent paths.

        Relies on contextvars being copied per-Task by asyncio.gather, which
        is the entire reason we chose contextvars over a list mutated in place.
        """
        import asyncio
        from unittest.mock import Mock

        from motus.agent._stream_context import _agent_path, _stream_callback

        received: list[tuple[str, tuple[str, ...]]] = []

        async def cb(msg):
            received.append((msg.content, _agent_path.get()))

        token = _stream_callback.set(cb)
        try:
            a = _AwaitingStreamingAgent(client=Mock(), model_name="m", name="a")
            b = _AwaitingStreamingAgent(client=Mock(), model_name="m", name="b")
            await asyncio.gather(
                a.as_tool(name="agent_a")._invoke(request="x"),
                b.as_tool(name="agent_b")._invoke(request="y"),
            )
        finally:
            _stream_callback.reset(token)

        contents = {(c, p) for c, p in received}
        assert ("reply from a", ("agent_a",)) in contents
        assert ("reply from b", ("agent_b",)) in contents
        # No cross-bleeding
        assert ("reply from a", ("agent_b",)) not in contents
        assert ("reply from b", ("agent_a",)) not in contents

    async def test_stateful_subagent_attributes_path(self):
        """as_tool(stateful=True) still attributes correctly (no fork involved)."""
        from unittest.mock import Mock

        from motus.agent._stream_context import _agent_path, _stream_callback

        received: list[tuple[str, ...]] = []

        async def cb(msg):
            received.append(_agent_path.get())

        token = _stream_callback.set(cb)
        try:
            inner = _StreamingAgent(client=Mock(), model_name="m", name="inner")
            tool = inner.as_tool(name="stateful_tool", stateful=True)
            await tool._invoke(request="hi")
        finally:
            _stream_callback.reset(token)

        assert ("stateful_tool",) in received

    async def test_no_stream_callback_means_no_emission(self):
        """Without an ambient callback, AgentTool runs cleanly and emits nothing."""
        from unittest.mock import Mock

        from motus.agent._stream_context import _stream_callback

        # Default: no callback set in this test's task context.
        assert _stream_callback.get() is None

        inner = _StreamingAgent(client=Mock(), model_name="m", name="inner")
        tool = inner.as_tool(name="research")
        # Should not error. add_message reads _stream_callback (None here),
        # so nothing escapes.
        result = await tool._invoke(request="hi")
        assert result == "reply from inner"

    async def test_path_is_reset_after_invocation(self):
        """The contextvar token must reset the path even on the success path."""
        from unittest.mock import Mock

        from motus.agent._stream_context import _agent_path, _stream_callback

        async def cb(msg):  # pragma: no cover — never called in this test
            pass

        cb_token = _stream_callback.set(cb)
        try:
            assert _agent_path.get() == ()
            inner = _StreamingAgent(client=Mock(), model_name="m", name="x")
            await inner.as_tool(name="t")._invoke(request="hi")
            # Path must have been popped on the way out
            assert _agent_path.get() == ()
        finally:
            _stream_callback.reset(cb_token)
