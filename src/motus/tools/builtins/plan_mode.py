"""Plan-mode signal tools — ``enter_plan_mode`` and ``exit_plan_mode``.

Plan mode is a read-only investigation phase the agent can switch into
before tackling a non-trivial change. While in plan mode, the active
tool set is filtered to a read-only subset (no ``write_file``,
``edit_file``, ``bash``, or sub-agent dispatch). The agent
investigates, drafts a plan, and calls ``exit_plan_mode(plan=...)``;
that submits the plan and restores the full tool set.

These tools are signal tools — they mutate the parent
:class:`CodingAgent`'s mode flag and tool set via the ``enter_fn`` /
``exit_fn`` callbacks supplied by :func:`make_plan_mode_tools`.
"""

from __future__ import annotations

from typing import Callable

from pydantic import ConfigDict, Field

from ..core import InputSchema
from ..core.decorators import tool

# The default read-only tool name allowlist applied when entering plan
# mode. The CodingAgent intersects this with its own ``_all_tools``
# snapshot, so missing tools are silently skipped.
PLAN_MODE_TOOLS: frozenset[str] = frozenset(
    {
        "read_file",
        "glob_search",
        "grep_search",
        "to_do",
        "web_fetch",
        "web_search",
        # exit_plan_mode must remain available so the agent can leave.
        "exit_plan_mode",
    }
)


class EnterPlanModeInput(InputSchema):
    model_config = ConfigDict(extra="forbid")


class ExitPlanModeInput(InputSchema):
    plan: str = Field(
        description=(
            "The completed plan as a markdown-formatted string. Should include the "
            "goal, a numbered step-by-step list of changes, the specific files "
            "that need to change, and any tradeoffs or risks the user should "
            "weigh in on."
        ),
    )

    model_config = ConfigDict(extra="forbid")


def make_plan_mode_tools(
    enter_fn: Callable[[], bool],
    exit_fn: Callable[[], bool],
):
    """Create ``enter_plan_mode`` and ``exit_plan_mode`` tools.

    Args:
        enter_fn: Called by ``enter_plan_mode``. Should toggle the agent
            into plan mode and return ``True`` if the toggle happened
            (or ``False`` if already in plan mode).
        exit_fn: Called by ``exit_plan_mode``. Should toggle the agent
            out of plan mode and return ``True`` on success.

    Returns:
        A tuple ``(enter_plan_mode, exit_plan_mode)`` of tool functions.
    """

    @tool(schema=EnterPlanModeInput)
    async def enter_plan_mode() -> str:
        """Enter plan mode — switch to read-only tools while you draft a plan.

        Use plan mode when the work is substantial enough that you should
        investigate and design before making changes — e.g. refactors that
        touch multiple files, new features with non-trivial structure, or
        changes whose blast radius isn't obvious.

        While in plan mode, only investigation tools are available
        (``read_file``, ``glob_search``, ``grep_search``, ``web_fetch``,
        ``web_search``, ``to_do``). ``write_file``, ``edit_file``,
        ``bash``, and the ``task`` sub-agent dispatcher are blocked
        until you call ``exit_plan_mode``.

        Don't use plan mode for: simple bug fixes, single-file changes,
        questions that just need an answer, or pure-research tasks where
        no implementation will follow.
        """
        toggled = enter_fn()
        if not toggled:
            return (
                "Already in plan mode — investigate and call exit_plan_mode when ready."
            )
        return (
            "Entered plan mode. Only read-only tools are available. "
            "Investigate the codebase, then call "
            "``exit_plan_mode(plan=...)`` with your finished plan."
        )

    @tool(schema=ExitPlanModeInput)
    async def exit_plan_mode(plan: str) -> str:
        """Exit plan mode by submitting a finished plan.

        The plan is returned as the tool result and the full tool set is
        restored, so your next turn can begin executing the plan (or
        wait for user direction). Only call this when your plan is
        finished and unambiguous.

        Don't use ``exit_plan_mode`` to ask "is this plan okay?" — just
        submit the plan; the parent / user can redirect from there.
        """
        toggled = exit_fn()
        if not toggled:
            return (
                "Not in plan mode — exit_plan_mode is only valid after enter_plan_mode."
            )
        return f"Plan submitted; full tool set restored.\n\n{plan}"

    return enter_plan_mode, exit_plan_mode
