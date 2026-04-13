from inspect import isclass, ismethod
from typing import Any, Awaitable, Callable, Container, List

from pydantic import BaseModel


def _set_tool_attr(obj: Callable[..., Awaitable], attr: str, value: Any) -> None:
    target = obj.__func__ if ismethod(obj) else obj
    setattr(target, attr, value)


def _merge_tool_options(existing: dict | None, updates: dict) -> dict:
    merged = dict(existing) if existing else {}
    for key, value in updates.items():
        if value is not None:
            merged[key] = value
    return merged


def tool(
    obj: Any | None = None,
    *,
    schema: dict | type[BaseModel] | None = None,
    name: str | None = None,
    description: str | None = None,
    requires_approval: bool = False,
    input_guardrails: list[Callable] | None = None,
    output_guardrails: list[Callable] | None = None,
    on_start: Callable | List[Callable] | None = None,
    on_end: Callable | List[Callable] | None = None,
    on_error: Callable | List[Callable] | None = None,
) -> Callable[..., Awaitable]:
    """Decorator that attaches tool metadata to a callable.

    For ``input_guardrails`` and ``output_guardrails`` usage, see
    ``examples/omni/gpuos_omni_demo.py``.

    Works both as a decorator and as a post-hoc patcher::

        # As decorator
        @tool(schema=MyInput)
        async def my_tool(...): ...

        # As patcher on an existing tool
        tool(existing_tool, description="New desc", schema=NewInput)
    """

    def wrap(target: Any) -> Callable[..., Awaitable]:
        if isclass(target):
            raise TypeError("tool() only supports callables; use tools() for classes")

        # If target is already a Tool instance (e.g. MCPTool), set attrs directly.
        from .tool import Tool as _ToolABC

        if isinstance(target, _ToolABC):
            if name:
                target.name = name
            if description is not None:
                target.description = description
            if schema is not None:
                target.json_schema = schema
            if input_guardrails is not None:
                target._input_guardrails = input_guardrails
            if output_guardrails is not None:
                target._output_guardrails = output_guardrails
            tool_name = target.name
        else:
            if name:
                _set_tool_attr(target, "__tool_name__", name)
            if description is not None:
                _set_tool_attr(target, "__doc__", description)
            if schema:
                _set_tool_attr(target, "__tool_schema__", schema)
            _set_tool_attr(target, "__tool_requires_approval__", requires_approval)
            if input_guardrails is not None:
                _set_tool_attr(target, "__tool_input_guardrails__", input_guardrails)
            if output_guardrails is not None:
                _set_tool_attr(target, "__tool_output_guardrails__", output_guardrails)
            tool_name = getattr(target, "__tool_name__", None)

        return target

    return wrap if obj is None else wrap(obj)


def tools(
    obj: Any | None = None,
    *,
    prefix: str = "",
    include_private: bool = False,
    method_schemas: dict[str, dict | type[BaseModel]] | None = None,
    method_aliases: dict[str, str] | None = None,
    allowlist: Container[str] | None = None,
    blocklist: Container[str] | None = None,
    input_guardrails: list[Callable] | None = None,
    output_guardrails: list[Callable] | None = None,
) -> Any:
    """Decorator that attaches tool-collection metadata to a class or instance."""
    options = {
        "prefix": prefix,
        "include_private": include_private,
        "method_schemas": method_schemas,
        "method_aliases": method_aliases,
        "allowlist": allowlist,
        "blocklist": blocklist,
        "input_guardrails": input_guardrails,
        "output_guardrails": output_guardrails,
    }

    def wrap(target: Any) -> Any:
        existing = getattr(target, "__tool_options__", None)
        target.__tool_options__ = _merge_tool_options(existing, options)
        return target

    return wrap if obj is None else wrap(obj)
