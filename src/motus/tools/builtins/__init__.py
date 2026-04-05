from __future__ import annotations

from ..core import Sandbox
from .bash import BashInput, make_bash_tool
from .file import EditInput, ReadInput, WriteInput, make_file_tools
from .search import GlobInput, GrepInput, make_search_tools
from .to_do import TodoInput, TodoItem, to_do

__all__ = [
    "BashInput",
    "BuiltinTools",
    "EditInput",
    "GlobInput",
    "GrepInput",
    "ReadInput",
    "TodoInput",
    "TodoItem",
    "WriteInput",
    "builtin_tools",
    "make_bash_tool",
    "make_file_tools",
    "make_search_tools",
    "to_do",
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
    )

    def __init__(self, sandbox: Sandbox) -> None:
        self.bash = make_bash_tool(sandbox)
        self.read_file, self.write_file, self.edit_file = make_file_tools(sandbox)
        self.glob_search, self.grep_search = make_search_tools(sandbox)
        self.to_do = to_do

    def __iter__(self):
        for attr in self.__slots__:
            yield getattr(self, attr)

    def __len__(self) -> int:
        return len(self.__slots__)


def builtin_tools(sandbox: Sandbox | None = None) -> BuiltinTools:
    """Create the standard set of builtin tools bound to *sandbox*.

    When *sandbox* is ``None``, a :class:`~motus.tools.LocalShell` is
    used so the tools work locally out of the box.  Pass an explicit
    sandbox (from a provider, ``DockerSandbox``, etc.) to run in a
    different environment.

    Usage::

        # Zero-config — runs locally
        bt = builtin_tools()

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

    return BuiltinTools(sandbox)
