"""Tests for ``tools.builtins.plan_mode`` — the ``enter_plan_mode`` /
``exit_plan_mode`` signal tools and the ``PLAN_MODE_TOOLS`` allowlist.
"""

from __future__ import annotations

import pytest

from motus.tools.builtins.plan_mode import (
    PLAN_MODE_TOOLS,
    make_plan_mode_tools,
)


class TestPlanModeAllowlist:
    def test_includes_read_only_investigation_tools(self):
        assert "read_file" in PLAN_MODE_TOOLS
        assert "glob_search" in PLAN_MODE_TOOLS
        assert "grep_search" in PLAN_MODE_TOOLS
        assert "to_do" in PLAN_MODE_TOOLS
        assert "web_fetch" in PLAN_MODE_TOOLS
        assert "web_search" in PLAN_MODE_TOOLS

    def test_excludes_write_and_shell_tools(self):
        for forbidden in ("write_file", "edit_file", "bash", "task"):
            assert forbidden not in PLAN_MODE_TOOLS

    def test_includes_exit_plan_mode_so_agent_can_leave(self):
        assert "exit_plan_mode" in PLAN_MODE_TOOLS

    def test_excludes_enter_plan_mode_redundant_in_plan_mode(self):
        # Already in plan mode — re-entering is meaningless.
        assert "enter_plan_mode" not in PLAN_MODE_TOOLS


class TestEnterPlanModeTool:
    @pytest.mark.asyncio
    async def test_calls_enter_fn_on_first_call(self):
        state = {"in": False}

        def enter_fn():
            if state["in"]:
                return False
            state["in"] = True
            return True

        def exit_fn():
            state["in"] = False
            return True

        enter, _ = make_plan_mode_tools(enter_fn, exit_fn)
        out = await enter()
        assert state["in"] is True
        assert "Entered plan mode" in out

    @pytest.mark.asyncio
    async def test_returns_idempotent_message_when_already_active(self):
        def enter_fn():
            return False  # already in plan mode

        def exit_fn():
            return True

        enter, _ = make_plan_mode_tools(enter_fn, exit_fn)
        out = await enter()
        assert "Already in plan mode" in out


class TestExitPlanModeTool:
    @pytest.mark.asyncio
    async def test_returns_plan_in_result_when_toggled(self):
        def enter_fn():
            return True

        def exit_fn():
            return True

        _, exit_ = make_plan_mode_tools(enter_fn, exit_fn)
        out = await exit_(plan="1. Step one\n2. Step two")
        assert "Plan submitted" in out
        assert "Step one" in out
        assert "Step two" in out

    @pytest.mark.asyncio
    async def test_friendly_error_when_not_in_plan_mode(self):
        def enter_fn():
            return True

        def exit_fn():
            return False  # not in plan mode

        _, exit_ = make_plan_mode_tools(enter_fn, exit_fn)
        out = await exit_(plan="anything")
        assert "Not in plan mode" in out


class TestEnterExitToggleSequence:
    @pytest.mark.asyncio
    async def test_full_cycle(self):
        state = {"in": False}

        def enter_fn():
            if state["in"]:
                return False
            state["in"] = True
            return True

        def exit_fn():
            if not state["in"]:
                return False
            state["in"] = False
            return True

        enter, exit_ = make_plan_mode_tools(enter_fn, exit_fn)

        # Enter
        out = await enter()
        assert state["in"] is True
        assert "Entered plan mode" in out

        # Re-enter is a no-op
        out = await enter()
        assert "Already in plan mode" in out

        # Exit
        out = await exit_(plan="my plan")
        assert state["in"] is False
        assert "my plan" in out

        # Exiting again returns the friendly error
        out = await exit_(plan="ignored")
        assert "Not in plan mode" in out
