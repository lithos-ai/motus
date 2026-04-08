"""Tests for the guardrail system."""

import asyncio
import json

import pytest
from pydantic import BaseModel

from motus.guardrails import (
    GuardrailTripped,
    InputGuardrailTripped,
    OutputGuardrailTripped,
    ToolInputGuardrailTripped,
    ToolOutputGuardrailTripped,
    _execute_guardrail,
    _execute_structured_guardrail,
    _execute_tool_input_guardrail,
    _execute_tool_output_guardrail,
    run_guardrails,
    run_structured_output_guardrails,
    run_tool_input_guardrails,
    run_tool_output_guardrails,
)
from motus.runtime.agent_task import agent_task
from motus.tools.core.function_tool import FunctionTool

# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    def test_input_guardrail_is_guardrail_tripped(self):
        assert isinstance(InputGuardrailTripped("x"), GuardrailTripped)

    def test_output_guardrail_is_guardrail_tripped(self):
        assert isinstance(OutputGuardrailTripped("x"), GuardrailTripped)

    def test_tool_input_guardrail_is_guardrail_tripped(self):
        assert isinstance(ToolInputGuardrailTripped("x"), GuardrailTripped)

    def test_tool_output_guardrail_is_guardrail_tripped(self):
        assert isinstance(ToolOutputGuardrailTripped("x"), GuardrailTripped)

    def test_message_attribute(self):
        e = InputGuardrailTripped("blocked!")
        assert e.message == "blocked!"
        assert str(e) == "blocked!"

    def test_default_message(self):
        e = GuardrailTripped()
        assert e.message == "Guardrail tripped"


# ---------------------------------------------------------------------------
# _execute_guardrail — agent guardrails (no agent)
# ---------------------------------------------------------------------------


class TestExecuteGuardrailToolStyle:
    """Agent guardrails with no agent receive only (value,)."""

    async def test_async_guardrail_passthrough(self):
        async def passthrough(value: str):
            pass  # returns None

        result = await _execute_guardrail(passthrough, "hello")
        assert result is None

    async def test_async_guardrail_modify(self):
        async def upper(value: str) -> str:
            return value.upper()

        result = await _execute_guardrail(upper, "hello")
        assert result == "HELLO"

    async def test_sync_guardrail_passthrough(self):
        def passthrough(value: str):
            pass

        result = await _execute_guardrail(passthrough, "hello")
        assert result is None

    async def test_sync_guardrail_modify(self):
        def upper(value: str) -> str:
            return value.upper()

        result = await _execute_guardrail(upper, "hello")
        assert result == "HELLO"

    async def test_async_guardrail_raises(self):
        async def blocker(value: str):
            raise InputGuardrailTripped("blocked")

        with pytest.raises(InputGuardrailTripped, match="blocked"):
            await _execute_guardrail(blocker, "hello")

    async def test_sync_guardrail_raises(self):
        def blocker(value: str):
            raise InputGuardrailTripped("blocked")

        with pytest.raises(InputGuardrailTripped, match="blocked"):
            await _execute_guardrail(blocker, "hello")


# ---------------------------------------------------------------------------
# _execute_guardrail — agent guardrails (with agent)
# ---------------------------------------------------------------------------


class TestExecuteGuardrailAgentStyle:
    """Agent guardrails receive (value, agent)."""

    async def test_async_agent_guardrail_receives_agent(self):
        received = {}

        async def check(value: str, agent):
            received["value"] = value
            received["agent"] = agent

        sentinel = object()
        await _execute_guardrail(check, "hello", agent=sentinel)
        assert received["value"] == "hello"
        assert received["agent"] is sentinel

    async def test_sync_agent_guardrail_receives_agent(self):
        received = {}

        def check(value: str, agent):
            received["value"] = value
            received["agent"] = agent

        sentinel = object()
        await _execute_guardrail(check, "hello", agent=sentinel)
        assert received["value"] == "hello"
        assert received["agent"] is sentinel

    async def test_agent_guardrail_can_modify(self):
        def add_agent_name(value: str, agent) -> str:
            return f"[{agent.name}] {value}"

        class FakeAgent:
            name = "TestBot"

        result = await _execute_guardrail(add_agent_name, "hello", agent=FakeAgent())
        assert result == "[TestBot] hello"

    async def test_agent_guardrail_can_raise(self):
        def block_if_disabled(value: str, agent):
            if not agent.enabled:
                raise InputGuardrailTripped("Agent disabled")

        class FakeAgent:
            enabled = False

        with pytest.raises(InputGuardrailTripped, match="Agent disabled"):
            await _execute_guardrail(block_if_disabled, "hello", agent=FakeAgent())

    async def test_single_param_guardrail_in_agent_context(self):
        """A guardrail declaring only (value) works even when agent is passed."""

        def value_only(value: str) -> str:
            return value.upper()

        result = await _execute_guardrail(value_only, "hello", agent=object())
        assert result == "HELLO"

    async def test_single_param_guardrail_raises_in_agent_context(self):
        """Single-param guardrail can raise even in agent context."""

        def blocker(value: str):
            raise InputGuardrailTripped("blocked")

        with pytest.raises(InputGuardrailTripped, match="blocked"):
            await _execute_guardrail(blocker, "hello", agent=object())


# ---------------------------------------------------------------------------
# run_guardrails — sequential chaining (agent style, no agent)
# ---------------------------------------------------------------------------


class TestRunGuardrails:
    """run_guardrails is async — wrap in an @agent_task to run on the runtime event loop."""

    @agent_task
    async def _run_in_task(self, guardrails, value, **kwargs):
        return await run_guardrails(guardrails, value, **kwargs)

    def test_empty_guardrails(self):
        result = self._run_in_task([], "hello").af_result()
        assert result == "hello"

    def test_single_passthrough(self):
        def noop(v: str):
            pass

        result = self._run_in_task([noop], "hello").af_result()
        assert result == "hello"

    def test_single_modify(self):
        def upper(v: str) -> str:
            return v.upper()

        result = self._run_in_task([upper], "hello").af_result()
        assert result == "HELLO"

    def test_chain_modifiers(self):
        def add_prefix(v: str) -> str:
            return "prefix_" + v

        def add_suffix(v: str) -> str:
            return v + "_suffix"

        result = self._run_in_task([add_prefix, add_suffix], "hello").af_result()
        assert result == "prefix_hello_suffix"

    def test_chain_with_passthrough(self):
        def noop(v: str):
            pass

        def upper(v: str) -> str:
            return v.upper()

        result = self._run_in_task([noop, upper, noop], "hello").af_result()
        assert result == "HELLO"

    def test_tripwire_stops_chain(self):
        call_log = []

        def first(v: str):
            call_log.append("first")

        def blocker(v: str):
            call_log.append("blocker")
            raise InputGuardrailTripped("stop")

        def never_called(v: str):
            call_log.append("never")

        with pytest.raises(InputGuardrailTripped, match="stop"):
            self._run_in_task([first, blocker, never_called], "hello").af_result()

        assert call_log == ["first", "blocker"]

    def test_async_guardrail_in_chain(self):
        async def async_upper(v: str) -> str:
            return v.upper()

        def sync_prefix(v: str) -> str:
            return "pre_" + v

        result = self._run_in_task([async_upper, sync_prefix], "hello").af_result()
        assert result == "pre_HELLO"


# ---------------------------------------------------------------------------
# run_guardrails — with agent (agent guardrail style)
# ---------------------------------------------------------------------------


class TestRunGuardrailsWithAgent:
    """Agent guardrails receive (value, agent) — verify via run_guardrails."""

    @agent_task
    async def _run_in_task(self, guardrails, value, **kwargs):
        return await run_guardrails(guardrails, value, **kwargs)

    def test_agent_passed_to_guardrails(self):
        received_agents = []

        def check(v: str, agent):
            received_agents.append(agent)

        sentinel = object()
        self._run_in_task([check], "hello", agent=sentinel).af_result()
        assert len(received_agents) == 1
        assert received_agents[0] is sentinel

    def test_agent_guardrail_chain_modifies(self):
        def add_agent_info(v: str, agent) -> str:
            return f"[{agent.name}] {v}"

        def upper(v: str, agent) -> str:
            return v.upper()

        class FakeAgent:
            name = "Bot"

        result = self._run_in_task(
            [add_agent_info, upper], "hello", agent=FakeAgent()
        ).af_result()
        assert result == "[BOT] HELLO"

    def test_agent_guardrail_tripwire(self):
        def block_if_disabled(v: str, agent):
            if not agent.enabled:
                raise InputGuardrailTripped("Disabled")

        class FakeAgent:
            enabled = False

        with pytest.raises(InputGuardrailTripped, match="Disabled"):
            self._run_in_task(
                [block_if_disabled], "hello", agent=FakeAgent()
            ).af_result()

    def test_mixed_single_and_two_param_guardrails(self):
        """Mix of (value) and (value, agent) guardrails in a chain with agent."""

        def upper_only(v: str) -> str:
            return v.upper()

        def add_agent_name(v: str, agent) -> str:
            return f"[{agent.name}] {v}"

        class FakeAgent:
            name = "Bot"

        result = self._run_in_task(
            [upper_only, add_agent_name], "hello", agent=FakeAgent()
        ).af_result()
        assert result == "[Bot] HELLO"


# ---------------------------------------------------------------------------
# run_guardrails — parallel mode
# ---------------------------------------------------------------------------


class TestRunGuardrailsParallel:
    """Parallel mode: all guardrails run on the original value, only tripwires matter."""

    @agent_task
    async def _run_in_task(self, guardrails, value, **kwargs):
        return await run_guardrails(guardrails, value, **kwargs)

    def test_parallel_passthrough(self):
        call_log = []

        def g1(v: str):
            call_log.append("g1")

        def g2(v: str):
            call_log.append("g2")

        result = self._run_in_task([g1, g2], "hello", parallel=True).af_result()
        assert result == "hello"
        assert set(call_log) == {"g1", "g2"}

    def test_parallel_tripwire(self):
        def ok(v: str):
            pass

        def blocker(v: str):
            raise InputGuardrailTripped("blocked")

        with pytest.raises(InputGuardrailTripped, match="blocked"):
            self._run_in_task([ok, blocker], "hello", parallel=True).af_result()

    def test_parallel_modifications_ignored(self):
        """In parallel mode, modifications are discarded — original value returned."""

        def upper(v: str) -> str:
            return v.upper()

        def prefix(v: str) -> str:
            return "pre_" + v

        result = self._run_in_task([upper, prefix], "hello", parallel=True).af_result()
        assert result == "hello"

    def test_parallel_with_agent(self):
        received_agents = []

        def check(v: str, agent):
            received_agents.append(agent)

        sentinel = object()
        result = self._run_in_task(
            [check, check], "hello", agent=sentinel, parallel=True
        ).af_result()
        assert result == "hello"
        assert len(received_agents) == 2
        assert all(a is sentinel for a in received_agents)


# ---------------------------------------------------------------------------
# _execute_tool_input_guardrail
# ---------------------------------------------------------------------------


class TestExecuteToolInputGuardrail:
    async def test_async_passthrough(self):
        async def check(query: str):
            pass

        result = await _execute_tool_input_guardrail(check, {"query": "SELECT"})
        assert result is None

    async def test_sync_passthrough(self):
        def check(query: str):
            pass

        result = await _execute_tool_input_guardrail(check, {"query": "SELECT"})
        assert result is None

    async def test_async_modify(self):
        async def redact(token: str) -> dict:
            return {"token": "[REDACTED]"}

        result = await _execute_tool_input_guardrail(
            redact, {"token": "sk-123", "url": "x"}
        )
        assert result == {"token": "[REDACTED]"}

    async def test_sync_modify(self):
        def redact(token: str) -> dict:
            return {"token": "[REDACTED]"}

        result = await _execute_tool_input_guardrail(
            redact, {"token": "sk-123", "url": "x"}
        )
        assert result == {"token": "[REDACTED]"}

    async def test_async_raises(self):
        async def block(query: str):
            raise ToolInputGuardrailTripped("blocked")

        with pytest.raises(ToolInputGuardrailTripped, match="blocked"):
            await _execute_tool_input_guardrail(block, {"query": "DROP"})

    async def test_sync_raises(self):
        def block(query: str):
            raise ToolInputGuardrailTripped("blocked")

        with pytest.raises(ToolInputGuardrailTripped, match="blocked"):
            await _execute_tool_input_guardrail(block, {"query": "DROP"})

    async def test_partial_param_matching(self):
        """Guardrail only sees the params it declares."""
        received = {}

        def check(query: str):
            received["query"] = query

        await _execute_tool_input_guardrail(check, {"query": "SELECT", "limit": 10})
        assert received == {"query": "SELECT"}

    async def test_skipped_when_required_param_missing(self):
        """Guardrail is skipped (returns None) when its required param is absent."""
        called = []

        def check(max_results: int):
            called.append(True)

        result = await _execute_tool_input_guardrail(check, {"query": "SELECT"})
        assert result is None
        assert called == []

    async def test_runs_when_optional_param_missing(self):
        """Guardrail runs if only optional params are missing."""
        received = {}

        def check(query: str, limit: int = 10):
            received["query"] = query
            received["limit"] = limit

        await _execute_tool_input_guardrail(check, {"query": "SELECT"})
        assert received == {"query": "SELECT", "limit": 10}


# ---------------------------------------------------------------------------
# _execute_tool_output_guardrail
# ---------------------------------------------------------------------------


class TestExecuteToolOutputGuardrail:
    async def test_async_passthrough(self):
        async def check(value: str):
            pass

        result = await _execute_tool_output_guardrail(check, "hello")
        assert result is None

    async def test_sync_passthrough(self):
        def check(value: str):
            pass

        result = await _execute_tool_output_guardrail(check, "hello")
        assert result is None

    async def test_async_modify(self):
        async def upper(value: str) -> str:
            return value.upper()

        result = await _execute_tool_output_guardrail(upper, "hello")
        assert result == "HELLO"

    async def test_sync_modify(self):
        def upper(value: str) -> str:
            return value.upper()

        result = await _execute_tool_output_guardrail(upper, "hello")
        assert result == "HELLO"

    async def test_async_raises(self):
        async def block(value: str):
            raise ToolOutputGuardrailTripped("blocked")

        with pytest.raises(ToolOutputGuardrailTripped, match="blocked"):
            await _execute_tool_output_guardrail(block, "hello")

    async def test_sync_raises(self):
        def block(value: str):
            raise ToolOutputGuardrailTripped("blocked")

        with pytest.raises(ToolOutputGuardrailTripped, match="blocked"):
            await _execute_tool_output_guardrail(block, "hello")


# ---------------------------------------------------------------------------
# run_tool_input_guardrails
# ---------------------------------------------------------------------------


class TestRunToolInputGuardrails:
    async def test_empty(self):
        result = await run_tool_input_guardrails([], {"query": "SELECT"})
        assert result == {"query": "SELECT"}

    async def test_passthrough(self):
        def check(query: str):
            pass

        result = await run_tool_input_guardrails([check], {"query": "SELECT"})
        assert result == {"query": "SELECT"}

    async def test_single_modify(self):
        def redact(token: str) -> dict:
            return {"token": "[REDACTED]"}

        result = await run_tool_input_guardrails(
            [redact], {"token": "sk-123", "url": "x"}
        )
        assert result == {"token": "[REDACTED]", "url": "x"}

    async def test_chain_merges(self):
        def upper_query(query: str) -> dict:
            return {"query": query.upper()}

        def add_limit(limit: int) -> dict:
            return {"limit": limit + 10}

        result = await run_tool_input_guardrails(
            [upper_query, add_limit],
            {"query": "select", "limit": 5},
        )
        assert result == {"query": "SELECT", "limit": 15}

    async def test_tripwire_stops_chain(self):
        call_log = []

        def first(query: str):
            call_log.append("first")

        def blocker(query: str):
            call_log.append("blocker")
            raise ToolInputGuardrailTripped("stop")

        def never(query: str):
            call_log.append("never")

        with pytest.raises(ToolInputGuardrailTripped, match="stop"):
            await run_tool_input_guardrails([first, blocker, never], {"query": "DROP"})
        assert call_log == ["first", "blocker"]

    async def test_mixed_sync_async(self):
        async def async_guard(query: str) -> dict:
            return {"query": query.upper()}

        def sync_guard(limit: int) -> dict:
            return {"limit": limit * 2}

        result = await run_tool_input_guardrails(
            [async_guard, sync_guard],
            {"query": "select", "limit": 5},
        )
        assert result == {"query": "SELECT", "limit": 10}


# ---------------------------------------------------------------------------
# run_tool_output_guardrails
# ---------------------------------------------------------------------------


class TestRunToolOutputGuardrails:
    async def test_empty(self):
        result = await run_tool_output_guardrails([], "hello")
        assert result == "hello"

    async def test_passthrough(self):
        def check(value: str):
            pass

        result = await run_tool_output_guardrails([check], "hello")
        assert result == "hello"

    async def test_single_modify(self):
        def upper(value: str) -> str:
            return value.upper()

        result = await run_tool_output_guardrails([upper], "hello")
        assert result == "HELLO"

    async def test_chain(self):
        def upper(value: str) -> str:
            return value.upper()

        def prefix(value: str) -> str:
            return "pre_" + value

        result = await run_tool_output_guardrails([upper, prefix], "hello")
        assert result == "pre_HELLO"

    async def test_tripwire(self):
        def block(value: str):
            raise ToolOutputGuardrailTripped("blocked")

        with pytest.raises(ToolOutputGuardrailTripped, match="blocked"):
            await run_tool_output_guardrails([block], "hello")


# ---------------------------------------------------------------------------
# FunctionTool with guardrails (typed signatures)
# ---------------------------------------------------------------------------


class TestFunctionToolGuardrails:
    def test_tool_input_guardrail_blocks(self):
        def block_secret(text: str):
            if "secret" in text:
                raise ToolInputGuardrailTripped("No secrets!")

        async def my_tool(text: str) -> str:
            return f"processed: {text}"

        ft = FunctionTool(my_tool, input_guardrails=[block_secret])
        result = json.loads(ft(json.dumps({"text": "my secret data"})).af_result())
        assert "error" in result
        assert "No secrets" in result["error"]

    def test_tool_input_guardrail_passes(self):
        def block_secret(text: str):
            if "secret" in text:
                raise ToolInputGuardrailTripped("No secrets!")

        async def my_tool(text: str) -> str:
            return f"processed: {text}"

        ft = FunctionTool(my_tool, input_guardrails=[block_secret])
        result = ft(json.dumps({"text": "safe data"})).af_result()
        assert "processed: safe data" in result

    def test_tool_input_guardrail_modifies(self):
        def redact(text: str) -> dict:
            return {"text": text.replace("secret", "[REDACTED]")}

        async def my_tool(text: str) -> str:
            return f"got: {text}"

        ft = FunctionTool(my_tool, input_guardrails=[redact])
        result = ft(json.dumps({"text": "my secret"})).af_result()
        assert "[REDACTED]" in result
        assert "secret" not in result or "[REDACTED]" in result

    def test_tool_input_guardrail_partial_update(self):
        """Guardrail modifies only one param, others preserved."""

        def redact_token(token: str) -> dict:
            return {"token": "[REDACTED]"}

        async def my_tool(url: str, token: str) -> str:
            return f"called {url} with {token}"

        ft = FunctionTool(my_tool, input_guardrails=[redact_token])
        result = ft(
            json.dumps({"url": "http://api.com", "token": "sk-123"})
        ).af_result()
        assert "http://api.com" in result
        assert "[REDACTED]" in result
        assert "sk-123" not in result

    def test_tool_output_guardrail_blocks(self):
        def block_pii(result: str):
            if "password" in result:
                raise ToolOutputGuardrailTripped("PII detected!")

        async def my_tool(text: str) -> str:
            return "password=hunter2"

        ft = FunctionTool(my_tool, output_guardrails=[block_pii])
        result = json.loads(ft(json.dumps({"text": "show"})).af_result())
        assert "error" in result
        assert "PII detected" in result["error"]

    def test_tool_output_guardrail_modifies(self):
        def redact_output(result: str) -> str:
            return result.replace("hunter2", "***")

        async def my_tool(text: str) -> str:
            return "password=hunter2"

        ft = FunctionTool(my_tool, output_guardrails=[redact_output])
        result = ft(json.dumps({"text": "show"})).af_result()
        assert "hunter2" not in result
        assert "***" in result

    def test_tool_no_guardrails(self):
        async def my_tool(text: str) -> str:
            return f"echo: {text}"

        ft = FunctionTool(my_tool)
        result = ft(json.dumps({"text": "hello"})).af_result()
        assert "echo: hello" in result


# ---------------------------------------------------------------------------
# @tool decorator with guardrails
# ---------------------------------------------------------------------------


class TestToolDecoratorGuardrails:
    def test_decorator_sets_guardrail_attrs(self):
        from motus.tools.core.decorators import tool

        def my_guard(v: str):
            pass

        @tool(input_guardrails=[my_guard])
        async def my_func(x: str) -> str:
            return x

        assert hasattr(my_func, "__tool_input_guardrails__")
        assert my_func.__tool_input_guardrails__ == [my_guard]

    def test_decorator_output_guardrail_attrs(self):
        from motus.tools.core.decorators import tool

        def my_guard(v: str):
            pass

        @tool(output_guardrails=[my_guard])
        async def my_func(x: str) -> str:
            return x

        assert hasattr(my_func, "__tool_output_guardrails__")
        assert my_func.__tool_output_guardrails__ == [my_guard]


# ---------------------------------------------------------------------------
# normalize_tools propagation
# ---------------------------------------------------------------------------


class TestNormalizeGuardrails:
    def test_normalize_callable_propagates_guardrails(self):
        from motus.tools.core.normalize import normalize_tools

        def my_guard(v: str):
            pass

        async def my_func(x: str) -> str:
            """A test tool."""
            return x

        my_func.__tool_input_guardrails__ = [my_guard]
        my_func.__tool_output_guardrails__ = [my_guard]

        tools = normalize_tools([my_func])
        ft = tools["my_func"]
        assert isinstance(ft, FunctionTool)
        assert ft._input_guardrails == [my_guard]
        assert ft._output_guardrails == [my_guard]

    def test_normalize_dict_propagates_guardrails(self):
        from motus.tools.core.normalize import normalize_tools

        def my_guard(v: str):
            pass

        async def my_func(x: str) -> str:
            """A test tool."""
            return x

        my_func.__tool_input_guardrails__ = [my_guard]

        tools = normalize_tools({"my_tool": my_func})
        ft = tools["my_tool"]
        assert isinstance(ft, FunctionTool)
        assert ft._input_guardrails == [my_guard]

    def test_tools_class_guardrails_apply_to_all_methods(self):
        from motus.tools.core.decorators import tools as tools_decorator
        from motus.tools.core.normalize import normalize_tools

        def class_guard(v: str):
            pass

        @tools_decorator(
            input_guardrails=[class_guard], output_guardrails=[class_guard]
        )
        class MyTools:
            async def foo(self, x: str) -> str:
                """Tool foo."""
                return x

            async def bar(self, x: str) -> str:
                """Tool bar."""
                return x

        result = normalize_tools(MyTools())
        assert result["foo"]._input_guardrails == [class_guard]
        assert result["foo"]._output_guardrails == [class_guard]
        assert result["bar"]._input_guardrails == [class_guard]
        assert result["bar"]._output_guardrails == [class_guard]

    def test_tools_method_guardrails_override_class(self):
        from motus.tools.core.decorators import tool as tool_decorator
        from motus.tools.core.decorators import tools as tools_decorator
        from motus.tools.core.normalize import normalize_tools

        def class_guard(v: str):
            pass

        def method_guard(v: str):
            pass

        @tools_decorator(input_guardrails=[class_guard])
        class MyTools:
            @tool_decorator(input_guardrails=[method_guard])
            async def foo(self, x: str) -> str:
                """Tool foo."""
                return x

            async def bar(self, x: str) -> str:
                """Tool bar."""
                return x

        result = normalize_tools(MyTools())
        # method-level overrides class-level
        assert result["foo"]._input_guardrails == [method_guard]
        # bar inherits class-level
        assert result["bar"]._input_guardrails == [class_guard]

    def test_tools_method_empty_list_disables_class_guardrails(self):
        """@tool(input_guardrails=[]) explicitly disables class-level guardrails."""
        from motus.tools.core.decorators import tool as tool_decorator
        from motus.tools.core.decorators import tools as tools_decorator
        from motus.tools.core.normalize import normalize_tools

        def class_guard(v: str):
            pass

        @tools_decorator(input_guardrails=[class_guard])
        class MyTools:
            @tool_decorator(input_guardrails=[])
            async def foo(self, x: str) -> str:
                """Tool foo."""
                return x

            async def bar(self, x: str) -> str:
                """Tool bar."""
                return x

        result = normalize_tools(MyTools())
        # foo explicitly set [] — should NOT inherit class-level
        assert result["foo"]._input_guardrails == []
        # bar inherits class-level
        assert result["bar"]._input_guardrails == [class_guard]

    def test_function_tool_empty_list_overrides_func_attr(self):
        """FunctionTool(func, input_guardrails=[]) disables func-level guardrails."""

        def my_guard(v: str):
            pass

        async def my_func(x: str) -> str:
            """A tool."""
            return x

        my_func.__tool_input_guardrails__ = [my_guard]

        ft = FunctionTool(my_func, input_guardrails=[])
        assert ft._input_guardrails == []


# ---------------------------------------------------------------------------
# Structured output guardrails (BaseModel agent results)
# ---------------------------------------------------------------------------


class SampleResult(BaseModel):
    score: float
    summary: str
    raw_data: str


class TestExecuteStructuredGuardrail:
    async def test_passthrough(self):
        def check(score: float):
            pass

        result = await _execute_structured_guardrail(
            check, {"score": 0.9, "summary": "ok", "raw_data": "x"}
        )
        assert result is None

    async def test_modify(self):
        def redact(raw_data: str) -> dict:
            return {"raw_data": "[REDACTED]"}

        result = await _execute_structured_guardrail(
            redact, {"score": 0.9, "summary": "ok", "raw_data": "secret"}
        )
        assert result == {"raw_data": "[REDACTED]"}

    async def test_raises(self):
        def validate(score: float):
            if score < 0:
                raise OutputGuardrailTripped("Score must be >= 0")

        with pytest.raises(OutputGuardrailTripped, match="Score must be >= 0"):
            await _execute_structured_guardrail(
                validate, {"score": -1.0, "summary": "bad"}
            )

    async def test_receives_agent(self):
        received = {}

        def check(score: float, agent):
            received["score"] = score
            received["agent"] = agent

        sentinel = object()
        await _execute_structured_guardrail(
            check, {"score": 0.5, "summary": "ok"}, agent=sentinel
        )
        assert received["score"] == 0.5
        assert received["agent"] is sentinel

    async def test_agent_not_matched_as_field(self):
        """'agent' param comes from the agent instance, not from kwargs."""
        received = {}

        def check(agent):
            received["agent"] = agent

        sentinel = object()
        await _execute_structured_guardrail(
            check, {"agent": "should_be_ignored", "score": 1.0}, agent=sentinel
        )
        assert received["agent"] is sentinel

    async def test_async_modify(self):
        async def redact(raw_data: str) -> dict:
            return {"raw_data": "[REDACTED]"}

        result = await _execute_structured_guardrail(
            redact, {"score": 0.9, "raw_data": "secret"}
        )
        assert result == {"raw_data": "[REDACTED]"}

    async def test_skipped_when_required_field_missing(self):
        """Guardrail is skipped when its required field is absent from kwargs."""
        called = []

        def check(missing_field: str):
            called.append(True)

        result = await _execute_structured_guardrail(
            check, {"score": 0.9, "summary": "ok"}
        )
        assert result is None
        assert called == []

    async def test_runs_when_optional_field_missing(self):
        """Guardrail runs if only optional fields are missing."""
        received = {}

        def check(score: float, extra: str = "default"):
            received["score"] = score
            received["extra"] = extra

        await _execute_structured_guardrail(check, {"score": 0.9, "summary": "ok"})
        assert received == {"score": 0.9, "extra": "default"}

    async def test_skipped_when_required_field_missing_with_agent(self):
        """Skip applies even when agent is available."""
        called = []

        def check(missing_field: str, agent):
            called.append(True)

        result = await _execute_structured_guardrail(
            check, {"score": 0.9}, agent=object()
        )
        assert result is None
        assert called == []


class TestRunStructuredOutputGuardrails:
    @agent_task
    async def _run(self, guardrails, kwargs, **kw):
        return await run_structured_output_guardrails(guardrails, kwargs, **kw)

    def test_empty(self):
        kwargs = {"score": 0.9, "summary": "ok"}
        result = self._run([], kwargs).af_result()
        assert result == kwargs

    def test_passthrough(self):
        def check(score: float):
            pass

        result = self._run([check], {"score": 0.9, "summary": "ok"}).af_result()
        assert result == {"score": 0.9, "summary": "ok"}

    def test_single_modify(self):
        def redact(raw_data: str) -> dict:
            return {"raw_data": "[REDACTED]"}

        result = self._run(
            [redact], {"score": 0.9, "raw_data": "secret", "summary": "ok"}
        ).af_result()
        assert result["raw_data"] == "[REDACTED]"
        assert result["score"] == 0.9

    def test_chain(self):
        def clamp_score(score: float) -> dict:
            return {"score": min(max(score, 0.0), 1.0)}

        def redact(raw_data: str) -> dict:
            return {"raw_data": "[REDACTED]"}

        result = self._run(
            [clamp_score, redact],
            {"score": 5.0, "raw_data": "secret", "summary": "ok"},
        ).af_result()
        assert result["score"] == 1.0
        assert result["raw_data"] == "[REDACTED]"
        assert result["summary"] == "ok"

    def test_tripwire(self):
        def validate(score: float):
            if score < 0:
                raise OutputGuardrailTripped("bad score")

        with pytest.raises(OutputGuardrailTripped, match="bad score"):
            self._run([validate], {"score": -1.0}).af_result()

    def test_with_agent(self):
        received = []

        def check(score: float, agent):
            received.append(agent)

        sentinel = object()
        self._run([check], {"score": 0.9}, agent=sentinel).af_result()
        assert len(received) == 1
        assert received[0] is sentinel

    def test_reconstruct_base_model(self):
        """Verify the full flow: BaseModel → dict → guardrails → BaseModel."""

        def redact(raw_data: str) -> dict:
            return {"raw_data": "[REDACTED]"}

        original = SampleResult(score=0.9, summary="ok", raw_data="secret")
        updated_kwargs = self._run([redact], original.model_dump()).af_result()
        rebuilt = SampleResult.model_validate(updated_kwargs)
        assert rebuilt.raw_data == "[REDACTED]"
        assert rebuilt.score == 0.9
        assert rebuilt.summary == "ok"


# ---------------------------------------------------------------------------
# _execute_tool_input_guardrail — VAR_KEYWORD support
# ---------------------------------------------------------------------------


def test_var_keyword_guardrail_receives_all_kwargs():
    """A `**kwargs` guardrail should receive every kwarg passed to the tool."""
    captured = {}

    async def universal_guardrail(**kwargs):
        captured.update(kwargs)

    kwargs = {"path": "/tmp/foo", "force": True}
    asyncio.run(_execute_tool_input_guardrail(universal_guardrail, kwargs))
    assert captured == {"path": "/tmp/foo", "force": True}


def test_var_keyword_guardrail_with_mixed_required_param():
    """A `(path, **kwargs)` guardrail should be skipped if `path` is missing."""

    async def mixed_guardrail(path: str, **kwargs):
        raise AssertionError("should have been skipped")

    # kwargs missing required 'path'
    result = asyncio.run(
        _execute_tool_input_guardrail(mixed_guardrail, {"force": True})
    )
    assert result is None  # skipped


def test_var_keyword_guardrail_with_mixed_required_param_present():
    """A `(path, **kwargs)` guardrail should be called when path is present."""
    seen = {}

    async def mixed_guardrail(path: str, **kwargs):
        seen["path"] = path
        seen["rest"] = kwargs

    asyncio.run(
        _execute_tool_input_guardrail(
            mixed_guardrail, {"path": "/tmp/foo", "force": True}
        )
    )
    assert seen["path"] == "/tmp/foo"
    # Python binds 'path' to its named parameter; remaining kwargs go to **kwargs
    assert seen["rest"] == {"force": True}


def test_empty_signature_guardrail_still_called():
    """An empty-signature guardrail should be called with no args."""
    called = []

    async def empty_guardrail():
        called.append(True)

    asyncio.run(_execute_tool_input_guardrail(empty_guardrail, {"a": 1}))
    assert called == [True]


def test_tool_rejected_is_subclass_of_tool_input_guardrail_tripped():
    from motus.guardrails import ToolInputGuardrailTripped, ToolRejected

    assert issubclass(ToolRejected, ToolInputGuardrailTripped)
    exc = ToolRejected("user said no")
    assert str(exc) == "user said no"
