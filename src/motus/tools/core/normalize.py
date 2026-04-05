import logging
from collections.abc import Mapping
from inspect import isclass
from typing import Any, Awaitable, Callable, Container, Iterable

from pydantic import BaseModel

from ._attrs import _get_tool_attr, resolve_tool_name
from .function_tool import FunctionTool
from .mcp_tool import MCPSession
from .tool import DictTools, Tool, Tools


def _is_agent_base(item: Any) -> bool:
    """Check if *item* is an AgentBase instance (lazy import to avoid cycles)."""
    try:
        from motus.agent.base_agent import AgentBase

        return isinstance(item, AgentBase)
    except ImportError:
        return False


def _get_tool_options(obj: Any) -> dict:
    options = getattr(obj, "__tool_options__", None)
    if options is None and not isclass(obj):
        options = getattr(obj.__class__, "__tool_options__", None)
    return options or {}


def _normalize_callable(func: Callable[..., Awaitable], tools: dict[str, Tool]) -> None:
    name = resolve_tool_name(func) or getattr(func, "name", None)
    if not name:
        raise ValueError("Tool is missing a name")
    if name in tools:
        raise ValueError(f"Duplicate tool name: {name}")
    tools[name] = FunctionTool(func, name=name)


def _normalize_mcp_session(session: MCPSession, tools: dict[str, Tool]) -> None:
    """Pull MCPTool instances from a connected MCPSession into *tools*."""
    options = _get_tool_options(session)
    prefix = options.get("prefix", "")
    allowlist = options.get("allowlist")
    blocklist = options.get("blocklist")
    class_input_guardrails = options.get("input_guardrails", [])
    class_output_guardrails = options.get("output_guardrails", [])
    allow = set(allowlist) if allowlist else None
    block = set(blocklist) if blocklist else None

    for name, mcp_tool in session._tools.items():
        if allow and name not in allow:
            continue
        if block and name in block:
            continue
        tool_name = f"{prefix}{name}" if prefix else name
        if tool_name in tools:
            raise ValueError(
                f"Duplicate tool name in this agent's tool set: {tool_name}. "
                f"Please consider adding a prefix to the tool name."
            )
        if class_input_guardrails and not mcp_tool._input_guardrails:
            mcp_tool._input_guardrails = list(class_input_guardrails)
        if class_output_guardrails and not mcp_tool._output_guardrails:
            mcp_tool._output_guardrails = list(class_output_guardrails)
        tools[tool_name] = mcp_tool


def _normalize_instance(instance: Any, tools: dict[str, Tool]) -> None:
    """Normalize a @tools-decorated class instance into a Tools mapping."""
    options = _get_tool_options(instance)
    prefix = options.get("prefix", "")
    include_private = options.get("include_private", False)
    method_schemas = options.get("method_schemas")
    method_aliases = options.get("method_aliases")
    allowlist = options.get("allowlist")
    blocklist = options.get("blocklist")
    class_input_guardrails = options.get("input_guardrails", [])
    class_output_guardrails = options.get("output_guardrails", [])

    for method in tools_from(
        instance,
        prefix=prefix,
        include_private=include_private,
        method_schemas=method_schemas,
        method_aliases=method_aliases,
        allowlist=allowlist,
        blocklist=blocklist,
    ):
        if isinstance(method, Tool):
            tool_name = f"{prefix}{method.name}" if prefix else method.name
            if tool_name in tools:
                raise ValueError(
                    f"Duplicate tool name in this agent's tool set: {tool_name}. "
                    f"Please consider adding a prefix to the tool name."
                )
            # Apply class-level guardrails if the tool has none
            if class_input_guardrails and not method._input_guardrails:
                method._input_guardrails = list(class_input_guardrails)
            if class_output_guardrails and not method._output_guardrails:
                method._output_guardrails = list(class_output_guardrails)
            tools[tool_name] = method
            continue

        method_name = (
            getattr(method, "__name__", None)
            or getattr(method, "__func__", None).__name__
        )
        schema = _get_tool_attr(method, "__tool_schema__") or (
            method_schemas.get(method_name) if method_schemas else None
        )
        base_name = (
            _get_tool_attr(method, "__tool_name__")
            or (method_aliases.get(method_name) if method_aliases else None)
            or method_name
        )
        tool_name = f"{prefix}{base_name}" if prefix else base_name
        if tool_name in tools:
            raise ValueError(
                f"Duplicate tool name in this agent's tool set: {tool_name}. Please consider adding a prefix to the tool name, directly specifying a different tool name, using method_aliases or block duplicates."
            )
        # FunctionTool reads method-level attrs itself; pass class-level as fallback
        method_input_guardrails = _get_tool_attr(method, "__tool_input_guardrails__")
        method_output_guardrails = _get_tool_attr(method, "__tool_output_guardrails__")
        tools[tool_name] = FunctionTool(
            method,
            name=tool_name,
            schema=schema,
            input_guardrails=(
                method_input_guardrails
                if method_input_guardrails is not None
                else (class_input_guardrails or None)
            ),
            output_guardrails=(
                method_output_guardrails
                if method_output_guardrails is not None
                else (class_output_guardrails or None)
            ),
        )


def tools_from(
    obj: Any,
    *,
    prefix: str = "",
    include_private: bool = False,
    method_schemas: dict[str, dict | type[BaseModel]] | None = None,
    method_aliases: dict[str, str] | None = None,
    allowlist: Container[str] | None = None,
    blocklist: Container[str] | None = None,
) -> list[Callable[..., Awaitable]]:
    """Collect tool-like callables from an instance's methods."""
    if isclass(obj):
        raise TypeError(
            "tools_from() requires an instance; pass an instance instead of a class"
        )

    # MCPSession: return MCPTool instances if connected, empty if not
    if isinstance(obj, MCPSession):
        if obj._session is None:
            return []
        allow = set(allowlist) if allowlist else None
        block = set(blocklist) if blocklist else None
        result = []
        for name, mcp_tool in obj._tools.items():
            if allow and name not in allow:
                continue
            if block and name in block:
                continue
            result.append(mcp_tool)
        return result

    instance = obj
    options = _get_tool_options(instance)
    if prefix == "":
        prefix = options.get("prefix", "")
    if include_private is False:
        include_private = options.get("include_private", False)
    method_schemas = method_schemas or options.get("method_schemas")
    method_aliases = method_aliases or options.get("method_aliases")
    allowlist = allowlist or options.get("allowlist")
    blocklist = blocklist or options.get("blocklist")
    allow = set(allowlist) if allowlist else None
    block = set(blocklist) if blocklist else None
    if allow and block:
        logging.warning(
            "Both allowlist and blocklist provided; allowlist takes precedence."
        )
    tools: list[Callable[..., Awaitable]] = []
    # Walk MRO to collect all method names (child overrides take precedence).
    # Unlike dir(), this excludes instance attributes and preserves MRO order.
    seen: set[str] = set()
    class_methods: list[str] = []
    for klass in type(instance).__mro__:
        for name in vars(klass):
            if name not in seen:
                seen.add(name)
                class_methods.append(name)
    if allow:
        missing = allow - set(class_methods)
        if missing:
            logging.warning(
                "Allowlist names not found on %s: %s",
                instance.__class__.__name__,
                ", ".join(sorted(missing)),
            )
        names = [name for name in class_methods if name in allow]
    else:
        names = class_methods
        if block:
            missing = block - set(class_methods)
            if missing:
                logging.warning(
                    "Blocklist names not found on %s: %s",
                    instance.__class__.__name__,
                    ", ".join(sorted(missing)),
                )
            names = [name for name in names if name not in block]
    if method_aliases:
        missing_aliases = set(method_aliases) - set(class_methods)
        if missing_aliases:
            logging.warning(
                "Method aliases not found on %s: %s",
                instance.__class__.__name__,
                ", ".join(sorted(missing_aliases)),
            )
    for name in names:
        if not include_private and name.startswith("_"):
            continue
        value = getattr(instance, name)
        if not callable(value):
            continue
        tools.append(value)
    return tools


def normalize_tools(
    tools: Tools | Iterable[Any] | Tool | Callable[..., Awaitable] | Any | None,
) -> DictTools:
    """Normalize tools, callables, classes, or instances into a Tools mapping."""
    if tools is None:
        return DictTools({})
    if isinstance(tools, Mapping):
        normalized: dict[str, Tool] = {}
        for key, tool in tools.items():
            if not isinstance(tool, Tool) and callable(tool):
                tool = FunctionTool(tool, name=key)
            name = getattr(tool, "name", None) or key
            if name in normalized:
                raise ValueError(f"Duplicate tool name: {name}")
            tool.name = name
            normalized[name] = tool
        return DictTools(normalized)
    normalized: dict[str, Tool] = {}
    owned_sessions: list[MCPSession] = []
    items = (
        [tools]
        if (
            not isinstance(tools, Iterable)  # single Tool / non-iterable
            or isinstance(tools, (str, bytes, dict))  # not a tool list
            or isclass(tools)  # @tools-decorated class
            or _get_tool_options(tools)  # @tools-decorated instance
            or isinstance(tools, MCPSession)  # MCP session (iterable itself)
            or _is_agent_base(tools)  # AgentBase instance (callable)
            or callable(tools)  # bare function
        )
        else tools  # list/tuple of tools
    )
    for item in items:
        if isinstance(item, Tools):
            for name, t in item.items():
                if name in normalized:
                    raise ValueError(f"Duplicate tool name: {name}")
                normalized[name] = t
            owned_sessions.extend(getattr(item, "_owned_sessions", []))
            continue
        if isinstance(item, MCPSession):
            if item._session is not None:
                # Already connected (user-managed) → register tools now
                _normalize_mcp_session(item, normalized)
            else:
                # Not connected (Agent-managed) → defer to lazy connect
                owned_sessions.append(item)
            continue
        if isinstance(item, Tool):
            name = getattr(item, "name", None)
            if not name:
                raise ValueError("Tool is missing a name")
            if name in normalized:
                raise ValueError(f"Duplicate tool name: {name}")
            normalized[name] = item
            continue
        if _is_agent_base(item):
            agent_tool = item.as_tool()
            if agent_tool.name in normalized:
                raise ValueError(f"Duplicate tool name: {agent_tool.name}")
            normalized[agent_tool.name] = agent_tool
            continue
        if _get_tool_options(item):
            _normalize_instance(item, normalized)
            continue
        if callable(item):
            _normalize_callable(item, normalized)
            continue
        raise TypeError(
            f"Cannot use {type(item).__name__} ({item!r}) as a tool. "
            f"Expected a Tool instance, a callable, or a @tools-decorated class instance."
        )
    return DictTools(normalized, owned_sessions=owned_sessions)
