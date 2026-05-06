"""``task`` tool — synchronous subagent dispatch.

The parent agent invokes this tool to delegate a self-contained task to
a specialized subagent. Subagent types are defined in
:mod:`motus.agent.templates.subagents`. The subagent runs to completion
and its final assistant message is returned to the parent as the tool
result.

This is a synchronous (foreground) task tool — there is no
``run_in_background`` mode. Background subagents will land later when
the runtime grows the notification primitives needed to surface
completion events back to the parent loop without disrupting it.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional

from pydantic import ConfigDict, Field

from ..core import InputSchema
from ..core.decorators import tool


class TaskInput(InputSchema):
    prompt: str = Field(
        description=(
            "Detailed, self-contained instructions for the subagent. The subagent "
            "starts with no context from this conversation — describe the goal, "
            "relevant background, and what form the response should take."
        ),
    )
    subagent_type: str = Field(
        description="Which specialized subagent to invoke. See the available types listed in this tool's description.",
    )
    model: Optional[Literal["haiku", "sonnet", "opus"]] = Field(
        default=None,
        description=(
            "Optional model override for the subagent. When omitted, the "
            "subagent inherits the parent's model."
        ),
    )

    model_config = ConfigDict(extra="forbid")


# Type alias — a subagent factory takes (spec_name, model_id) and
# returns a fully-configured agent ready to be called. The agent must
# accept ``await agent(prompt)`` and return a string result.
SubAgentFactory = Callable[[str, str], Any]


def make_task_tool(
    *,
    subagent_factory: SubAgentFactory,
    subagent_specs: dict,
    parent_model: str,
):
    """Create a ``task`` tool bound to a subagent factory.

    Args:
        subagent_factory: Callable invoked once per task call to produce
            a fresh subagent. Signature ``(spec_name, model_id) ->
            agent``. The returned agent is awaited with the user prompt
            and its final string answer is used as the tool result.
        subagent_specs: Map of spec name to :class:`SubAgentSpec`. Used
            to validate ``subagent_type`` and to render the available
            types into the tool's description.
        parent_model: The parent agent's model ID. Used to resolve the
            default when ``task(model=None)`` is passed.

    Returns:
        A tool function decorated with ``@tool``.
    """
    from motus.agent.templates.subagents import resolve_model

    spec_listing = "\n".join(
        f"- ``{name}`` — {spec.description}" for name, spec in subagent_specs.items()
    )

    docstring = (
        "Launch a specialized sub-agent to handle a self-contained task autonomously.\n\n"
        "Use this when the work would benefit from being done in an isolated context "
        "(e.g. wide-ranging codebase exploration, implementation planning, or a "
        "multi-step task you want to keep out of your own context window). The "
        "subagent runs to completion and returns its final answer; the parent agent "
        "(you) then continues with that result.\n\n"
        "Available subagent types:\n"
        f"{spec_listing}\n\n"
        "Notes:\n"
        "- The ``prompt`` must be self-contained — the subagent starts with no view "
        "into this conversation.\n"
        "- The subagent has its own fresh memory; it cannot see your messages or "
        "tool results.\n"
        "- Subagents cannot launch further subagents.\n"
    )

    @tool(schema=TaskInput, description=docstring)
    async def task(
        prompt: str,
        subagent_type: str,
        model: str | None = None,
    ) -> str:
        spec = subagent_specs.get(subagent_type)
        if spec is None:
            available = ", ".join(subagent_specs.keys())
            return (
                f"Unknown subagent_type {subagent_type!r}. Available types: {available}"
            )

        resolved = resolve_model(model, parent_model)
        try:
            sub = subagent_factory(spec.name, resolved)
        except Exception as e:
            return f"Failed to construct {subagent_type!r} subagent: {e}"

        try:
            result = await sub(prompt)
        except Exception as e:
            return f"Subagent {subagent_type!r} failed: {e}"

        return result if isinstance(result, str) else str(result)

    return task
