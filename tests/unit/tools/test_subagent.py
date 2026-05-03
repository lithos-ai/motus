"""Tests for ``tools.builtins.subagent`` — the synchronous ``task`` tool
used by ``CodingAgent`` to dispatch work to specialized subagents.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from motus.agent.templates.subagents import (
    DEFAULT_SUBAGENTS,
    EXPLORE,
    GENERAL_PURPOSE,
    MODEL_ALIASES,
    PLAN,
    READ_ONLY_TOOLS,
    SubAgentSpec,
    resolve_model,
)
from motus.tools.builtins.subagent import make_task_tool

# ---------------------------------------------------------------------------
# Spec module
# ---------------------------------------------------------------------------


class TestSubagentSpecs:
    def test_default_specs_present(self):
        assert "general-purpose" in DEFAULT_SUBAGENTS
        assert "Explore" in DEFAULT_SUBAGENTS
        assert "Plan" in DEFAULT_SUBAGENTS

    def test_explore_is_read_only(self):
        assert EXPLORE.allowed_tools is not None
        assert EXPLORE.allowed_tools == READ_ONLY_TOOLS
        # Spot check forbidden tools
        for forbidden in ("write_file", "edit_file", "bash"):
            assert forbidden not in EXPLORE.allowed_tools

    def test_plan_is_read_only(self):
        assert PLAN.allowed_tools is not None
        assert PLAN.allowed_tools == READ_ONLY_TOOLS

    def test_general_purpose_has_no_tool_filter(self):
        assert GENERAL_PURPOSE.allowed_tools is None

    def test_extra_prompts_describe_role(self):
        assert "Explore sub-agent" in EXPLORE.extra_prompt
        assert "Plan sub-agent" in PLAN.extra_prompt
        assert "sub-agent" in GENERAL_PURPOSE.extra_prompt.lower()


class TestModelAliases:
    def test_known_aliases_map_correctly(self):
        assert MODEL_ALIASES["haiku"].startswith("claude-haiku")
        assert MODEL_ALIASES["sonnet"].startswith("claude-sonnet")
        assert MODEL_ALIASES["opus"].startswith("claude-opus")

    def test_resolve_none_inherits_parent(self):
        assert resolve_model(None, "parent-model") == "parent-model"

    def test_resolve_alias(self):
        assert resolve_model("haiku", "ignored") == MODEL_ALIASES["haiku"]

    def test_resolve_unknown_string_passes_through(self):
        # Allows callers to pass full model IDs directly.
        assert resolve_model("claude-opus-4-7", "fallback") == "claude-opus-4-7"


# ---------------------------------------------------------------------------
# make_task_tool
# ---------------------------------------------------------------------------


SPECS = {
    "general-purpose": GENERAL_PURPOSE,
    "Explore": EXPLORE,
    "Plan": PLAN,
}


class TestMakeTaskTool:
    @pytest.mark.asyncio
    async def test_unknown_subagent_type_returns_friendly_error(self):
        factory = MagicMock()
        task = make_task_tool(
            subagent_factory=factory,
            subagent_specs=SPECS,
            parent_model="claude-haiku-4-5",
        )
        out = await task(
            prompt="anything",
            subagent_type="UnknownType",
        )
        assert "Unknown subagent_type" in out
        assert "general-purpose" in out
        # Factory must not be called when type is unknown.
        factory.assert_not_called()

    @pytest.mark.asyncio
    async def test_factory_called_with_resolved_model(self):
        captured: dict = {}

        def factory(spec_name, model_id):
            captured["spec_name"] = spec_name
            captured["model_id"] = model_id
            sub = AsyncMock()
            sub.return_value = "ok"
            return sub

        task = make_task_tool(
            subagent_factory=factory,
            subagent_specs=SPECS,
            parent_model="parent-m",
        )

        # No model alias → inherits parent
        await task(
            prompt="self-contained prompt",
            subagent_type="Explore",
        )
        assert captured["spec_name"] == "Explore"
        assert captured["model_id"] == "parent-m"

        # Alias is resolved
        await task(
            prompt="p",
            subagent_type="Explore",
            model="haiku",
        )
        assert captured["model_id"] == MODEL_ALIASES["haiku"]

    @pytest.mark.asyncio
    async def test_returns_subagent_final_message(self):
        sub = AsyncMock()
        sub.return_value = "Sub said this."

        def factory(spec_name, model_id):
            return sub

        task = make_task_tool(
            subagent_factory=factory,
            subagent_specs=SPECS,
            parent_model="m",
        )
        out = await task(
            prompt="p",
            subagent_type="general-purpose",
        )
        assert out == "Sub said this."
        sub.assert_awaited_once_with("p")

    @pytest.mark.asyncio
    async def test_subagent_exception_returned_as_string(self):
        sub = AsyncMock(side_effect=RuntimeError("boom"))

        def factory(spec_name, model_id):
            return sub

        task = make_task_tool(
            subagent_factory=factory,
            subagent_specs=SPECS,
            parent_model="m",
        )
        out = await task(
            prompt="p",
            subagent_type="Explore",
        )
        assert "failed" in out.lower()
        assert "boom" in out

    @pytest.mark.asyncio
    async def test_factory_construction_failure_returned_as_string(self):
        def factory(spec_name, model_id):
            raise ValueError("bad config")

        task = make_task_tool(
            subagent_factory=factory,
            subagent_specs=SPECS,
            parent_model="m",
        )
        out = await task(
            prompt="p",
            subagent_type="Explore",
        )
        assert "Failed to construct" in out
        assert "bad config" in out

    @pytest.mark.asyncio
    async def test_non_string_result_coerced_to_string(self):
        sub = AsyncMock()
        sub.return_value = {"key": "value"}

        def factory(spec_name, model_id):
            return sub

        task = make_task_tool(
            subagent_factory=factory,
            subagent_specs=SPECS,
            parent_model="m",
        )
        out = await task(
            prompt="p",
            subagent_type="general-purpose",
        )
        # str(dict) rendering — the contract is "string result"
        assert isinstance(out, str)
        assert "key" in out


# ---------------------------------------------------------------------------
# Custom specs
# ---------------------------------------------------------------------------


class TestCustomSpecs:
    @pytest.mark.asyncio
    async def test_custom_spec_works_through_make_task_tool(self):
        custom = SubAgentSpec(
            name="Reviewer",
            description="Code reviewer.",
            extra_prompt="You review code.",
            allowed_tools=frozenset({"read_file"}),
        )
        captured: dict = {}

        def factory(spec_name, model_id):
            captured["spec_name"] = spec_name
            sub = AsyncMock()
            sub.return_value = "reviewed"
            return sub

        task = make_task_tool(
            subagent_factory=factory,
            subagent_specs={"Reviewer": custom},
            parent_model="m",
        )
        out = await task(
            prompt="p",
            subagent_type="Reviewer",
        )
        assert captured["spec_name"] == "Reviewer"
        assert out == "reviewed"
