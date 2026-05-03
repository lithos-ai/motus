from __future__ import annotations

from pathlib import Path

from ..core import Sandbox
from .bash import make_bash_tool
from .file import make_file_tools
from .plan_mode import PLAN_MODE_TOOLS, make_plan_mode_tools
from .search import make_search_tools
from .skill import make_skill_tool
from .subagent import make_task_tool
from .to_do import to_do
from .web import make_web_fetch, make_web_search

__all__ = [
    "BuiltinTools",
    "PLAN_MODE_TOOLS",
    "builtin_tools",
    "make_plan_mode_tools",
    "make_task_tool",
    "make_web_fetch",
    "make_web_search",
]


class BuiltinTools:
    """Collection of framework builtin tools bound to a sandbox.

    Supports both attribute access and iteration::

        bt = builtin_tools(sandbox)

        # Iterate all tools (for Agent / normalize_tools)
        agent = Agent(tools=[*bt, other_tool])

        # Access individual tools for customisation
        from motus.tools import tool
        tool(bt.bash, description="Run safely", schema=SafeBash)
    """

    __slots__ = (
        "bash",
        "read_file",
        "write_file",
        "edit_file",
        "glob_search",
        "grep_search",
        "to_do",
        "load_skill",
    )

    def __init__(self, sandbox: Sandbox, skills_dir: str | Path | None = None) -> None:
        self.bash = make_bash_tool(sandbox)
        self.read_file, self.write_file, self.edit_file = make_file_tools(sandbox)
        self.glob_search, self.grep_search = make_search_tools(sandbox)
        self.to_do = to_do
        self.load_skill = (
            make_skill_tool(skills_dir) if skills_dir is not None else None
        )

    def __iter__(self):
        for attr in self.__slots__:
            val = getattr(self, attr)
            if val is not None:
                yield val

    def __len__(self) -> int:
        return sum(1 for attr in self.__slots__ if getattr(self, attr) is not None)


def builtin_tools(
    sandbox: Sandbox | None = None,
    skills_dir: str | Path | None = None,
) -> BuiltinTools:
    """Create the standard set of builtin tools bound to *sandbox*.

    When *sandbox* is ``None``, a :class:`~motus.tools.LocalShell` is
    used so the tools work locally out of the box.  Pass an explicit
    sandbox (from a provider, ``DockerSandbox``, etc.) to run in a
    different environment.

    When *skills_dir* is provided, a ``load_skill`` tool is included
    that lets the agent load on-demand instructions from skill
    directories under that path.

    Usage::

        # Zero-config — runs locally
        bt = builtin_tools()

        # With skills
        bt = builtin_tools(skills_dir="path/to/skills/")

        # Explicit sandbox (e.g. from a provider)
        bt = builtin_tools(sandbox)

        # Customise individual tools
        tool(bt.bash, description="Run safely", input_guardrails=[...])

        # Pass to agent
        agent = Agent(tools=[*bt, other_tool])

    Returns a :class:`BuiltinTools` instance that is both iterable
    (``[*bt]``, ``normalize_tools(bt)``) and supports attribute access
    (``bt.bash``, ``bt.read_file``, …) for per-tool customisation via
    :func:`tool`.
    """
    if sandbox is None:
        from ..providers.local import LocalShell

        sandbox = LocalShell()

    return BuiltinTools(sandbox, skills_dir=skills_dir)
