"""Built-in subagent specifications for the ``task`` tool.

Each spec defines a specialization on top of the base coding-agent
prompt: an extra prompt fragment (the "what does this subagent do"
preamble) and an optional tool name allowlist used to filter the tool
set after the subagent is constructed.

The shape mirrors Claude Code's three default subagent types:
``general-purpose``, ``Explore``, and ``Plan``.
"""

from __future__ import annotations

from dataclasses import dataclass

# Aliases the parent agent can pass via ``task(model=...)``. Resolved to
# concrete model IDs at subagent construction time. When omitted, the
# subagent inherits the parent's model.
MODEL_ALIASES: dict[str, str] = {
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-7",
}


# Read-only tool subset used by Explore and Plan. These subagents must
# not modify the workspace — they investigate or design and report back.
READ_ONLY_TOOLS: frozenset[str] = frozenset(
    {
        "read_file",
        "glob_search",
        "grep_search",
        "to_do",
        "web_fetch",
        "web_search",
    }
)


@dataclass(frozen=True)
class SubAgentSpec:
    """A subagent type definition.

    Attributes:
        name: Identifier the parent passes via ``task(subagent_type=...)``.
        description: Shown to the parent (in the ``task`` tool's docstring)
            so the model knows when to use this type.
        extra_prompt: Appended to the base coding-agent prompt as
            ``system_prompt_extra``. Should specify scope, output format,
            and any constraints particular to this subagent.
        allowed_tools: Optional name allowlist. Tools whose names aren't
            in the set are filtered out after construction. ``None``
            means "all tools the parent had".
    """

    name: str
    description: str
    extra_prompt: str
    allowed_tools: frozenset[str] | None = None


GENERAL_PURPOSE = SubAgentSpec(
    name="general-purpose",
    description=(
        "General-purpose agent for researching complex questions, searching for "
        "code, and executing multi-step tasks. When you're searching for a keyword "
        "or file and aren't confident you'll find the right match in the first few "
        "tries, use this agent to perform the search for you."
    ),
    extra_prompt=(
        "## You are a sub-agent\n\n"
        "You were invoked by a parent agent to handle a specific task autonomously. "
        "The parent agent will use your final response as the result of this task.\n\n"
        "- The prompt you received is self-contained — don't ask clarifying questions; "
        "make reasonable assumptions and proceed.\n"
        "- Be focused. Do the work the parent asked for; don't expand scope.\n"
        "- Return your result as your final assistant message. The parent reads only "
        "that final message — anything you say earlier is invisible to it."
    ),
)


EXPLORE = SubAgentSpec(
    name="Explore",
    description=(
        "Fast agent specialized for exploring codebases. Use this when you need to "
        "quickly find files by patterns (e.g. 'src/components/**/*.tsx'), search "
        "code for keywords (e.g. 'API endpoints'), or answer questions about the "
        "codebase (e.g. 'how do API endpoints work?'). Cannot modify files — use "
        "this for investigation, not implementation."
    ),
    extra_prompt=(
        "## You are an Explore sub-agent\n\n"
        "Your job is to investigate the codebase per the user's request and return "
        "your findings. You CANNOT modify files; only read-only tools are available.\n\n"
        "Don't try to solve the user's underlying problem — gather information and "
        "report back. The parent agent will decide what to do next.\n\n"
        "Output format for your final message:\n"
        "- A brief summary (1-3 paragraphs) of what you found.\n"
        "- Key locations as `file_path:line_number` references so the parent can "
        "navigate directly.\n"
        "- Suggestions for follow-up investigation if relevant.\n\n"
        "Don't speculate beyond what the code shows. If something is unclear, "
        "say so explicitly rather than guessing."
    ),
    allowed_tools=READ_ONLY_TOOLS,
)


PLAN = SubAgentSpec(
    name="Plan",
    description=(
        "Software architect agent for designing implementation plans. Use this when "
        "you need to plan the implementation strategy for a task. Returns a "
        "step-by-step plan, identifies critical files, and considers architectural "
        "trade-offs. Cannot modify files — use this for planning, not execution."
    ),
    extra_prompt=(
        "## You are a Plan sub-agent\n\n"
        "Your job is to design an implementation plan for the user's request. You "
        "CANNOT modify files; only read-only tools are available. Investigate first, "
        "then return a plan as your final message.\n\n"
        "Output format for your final message:\n"
        "1. **Goal** — one-line restatement of what's being built.\n"
        "2. **Plan** — a numbered step-by-step list of changes.\n"
        "3. **Files** — the specific files that will need to change, with "
        "`file_path:line_number` anchors when relevant.\n"
        "4. **Tradeoffs / risks** — architectural choices the parent should be "
        "aware of, plus any edge cases or risks.\n\n"
        "Don't write code. The parent agent will execute the plan."
    ),
    allowed_tools=READ_ONLY_TOOLS,
)


DEFAULT_SUBAGENTS: dict[str, SubAgentSpec] = {
    GENERAL_PURPOSE.name: GENERAL_PURPOSE,
    EXPLORE.name: EXPLORE,
    PLAN.name: PLAN,
}


def resolve_model(alias_or_none: str | None, parent_model: str) -> str:
    """Resolve a ``task(model=...)`` argument to a concrete model ID.

    Returns the parent's model when *alias_or_none* is None. Looks up
    aliases in :data:`MODEL_ALIASES`; falls back to passing the raw
    string through if it's not a known alias (so callers can also pass
    full model IDs).
    """
    if not alias_or_none:
        return parent_model
    return MODEL_ALIASES.get(alias_or_none, alias_or_none)
