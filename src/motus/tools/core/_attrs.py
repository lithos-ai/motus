"""Tool attribute utilities — shared by function_tool, normalize, and decorators.

This module has NO dependencies on function_tool.py or normalize.py,
so it can be imported freely without circular import issues.
"""

from inspect import ismethod
from typing import Any, Awaitable, Callable


def _get_tool_attr(obj: Callable[..., Awaitable], attr: str) -> Any | None:
    """Read a tool attribute, falling back to __func__ for bound methods."""
    if hasattr(obj, attr):
        return getattr(obj, attr)
    if ismethod(obj):
        fn = getattr(obj, "__func__", None)
        if fn is not None and hasattr(fn, attr):
            return getattr(fn, attr)
    return None


def resolve_tool_name(func: Any) -> str | None:
    """Canonical tool name resolution from a callable.

    Priority: __tool_name__ (incl. __func__ penetration) → __name__.
    Returns None only when neither attribute exists.
    """
    return _get_tool_attr(func, "__tool_name__") or getattr(func, "__name__", None)
