import asyncio
import builtins
import inspect
import json
import logging
import types
import typing
from asyncio import iscoroutinefunction
from collections import abc
from dataclasses import asdict, is_dataclass
from inspect import getdoc, isclass, isfunction, ismethod
from typing import (
    Any,
    Awaitable,
    Callable,
    get_args,
    get_origin,
    get_type_hints,
)
from typing import is_typeddict as _is_typeddict

import jsonref
from pydantic import BaseModel

from ._attrs import _get_tool_attr, resolve_tool_name
from .tool import Tool


def _clean_json_schema(schema: dict) -> dict:
    schema_no_refs = jsonref.replace_refs(schema, proxies=False)

    def clean(node):
        if isinstance(node, dict):
            # Handle AnyOf merging
            # If there is anyOf, merge the objects and remove the anyOf key
            if "anyOf" in node:
                non_null = [x for x in node["anyOf"] if x.get("type") != "null"]
                if len(non_null) == 1:
                    node.update(non_null[0])
                    del node["anyOf"]
            # clean title and $defs
            node.pop("title", None)
            node.pop("$defs", None)
            # clean default
            if "default" in node and node["default"] is None:
                del node["default"]
            # clean children
            for value in list(node.values()):
                clean(value)
        elif isinstance(node, list):
            for item in node:
                clean(item)
        return node

    return clean(schema_no_refs)


class InputSchema(BaseModel):
    """
    The reason we need this is because the model_json_schema method in BaseModel
    returns a schema that is too redundant. We want to replace the model_json_schema method
    to clean it up.

    In the future, we may consider customizing the schema based on the type of the model.
    """

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        raw_schema = super().model_json_schema(*args, **kwargs)
        return _clean_json_schema(raw_schema)


def json_schema(t: type) -> dict:
    """Convert a Python type to a JSON schema.

    We're only supporting the types we need for LLM tool calling.
    """
    origin = get_origin(t)
    if origin in (typing.NotRequired, typing.Required):
        return json_schema(get_args(t)[0])
    match t:
        case builtins.str:
            return {"type": "string"}
        case builtins.int:
            return {"type": "integer"}
        case builtins.float:
            return {"type": "number"}
        case builtins.bool:
            return {"type": "boolean"}
        case _ if t is types.NoneType or isinstance(t, type(None)):
            return {"type": "null"}
        case _ if isclass(t) and issubclass(t, BaseModel):
            return t.model_json_schema()
        case _ if _is_typeddict(t):
            hints = get_type_hints(t, include_extras=True)
            properties = {k: json_schema(v) for k, v in hints.items()}
            required_keys = getattr(t, "__required_keys__", None)
            if required_keys is None:
                required_keys = (
                    set(hints.keys()) if getattr(t, "__total__", True) else set()
                )
            schema = {"type": "object", "properties": properties}
            if required_keys:
                schema["required"] = sorted(required_keys)
            return schema
        case _ if isclass(t):
            d = {
                "type": "object",
                "properties": {
                    k: json_schema(v)
                    for k, v in get_type_hints(t, include_extras=True).items()
                },
            }
            doc = getdoc(t)
            if doc:
                d["description"] = doc
            return d
        case _:
            pass  # Fall through to handle generic types
    match get_origin(t), get_args(t):
        case typing.Annotated, (inner_type, str(annotation)):
            logging.debug(f"Found annotation: {annotation} on type {inner_type}")
            return {**json_schema(inner_type), "description": annotation}
        case typing.Optional, (inner_type,):
            return {"anyOf": [json_schema(inner_type), {"type": "null"}]}
        case typing.Union | types.UnionType, args:
            return {"anyOf": [json_schema(arg) for arg in args]}
        case builtins.list | typing.List | builtins.list, (item_type,):
            return {"type": "array", "items": json_schema(item_type)}
        case typing.Tuple | builtins.tuple, (item_type, builtins.Ellipsis):
            # tuple[int, ...] -> homogeneous variable-length tuple
            return {"type": "array", "items": json_schema(item_type)}
        case typing.Tuple | builtins.tuple, args if args:
            # tuple[int, str, bool] -> fixed-length tuple with specific types
            return {"type": "array", "prefixItems": [json_schema(arg) for arg in args]}
        case builtins.dict | typing.Dict | abc.Mapping, (key_type, value_type):
            assert key_type is str, (
                f"Type {t} isn't supported: only string keys are supported"
            )
            return {
                "type": "object",
                "propertyNames": json_schema(key_type),
                "additionalProperties": json_schema(value_type),
            }
    raise ValueError(f"Type {t} isn't supported")


def from_dict(t: type, v: dict) -> Any:
    """Convert a dictionary to an instance of the given type."""
    if isclass(t) and issubclass(t, BaseModel):
        return t.model_validate(v)
    if is_dataclass(t):
        hints = get_type_hints(t)
        return t(**{k: from_dict(hints[k], v[k]) for k in hints if k in v})
    return v


def _coerce(t: type, v: Any) -> Any:
    """Coerce a raw JSON value to the declared Python type."""
    origin = get_origin(t)
    if origin is typing.Annotated:
        t = get_args(t)[0]
        origin = get_origin(t)
    if origin in (typing.Union, types.UnionType):
        args = [arg for arg in get_args(t) if arg is not types.NoneType]
        if len(args) == 1:
            if v is None:
                return None
            return _coerce(args[0], v)
    if isclass(t) and issubclass(t, BaseModel):
        return t.model_validate(v)
    if is_dataclass(t):
        hints = get_type_hints(t)
        return t(**{k: _coerce(hints[k], v[k]) for k in hints if k in v})
    if origin in (typing.List, builtins.list):
        item_type = get_args(t)[0]
        return [_coerce(item_type, item) for item in v]
    return v


class Parameters(json.JSONDecoder):
    """Decode JSON arguments and expose a JSON Schema for them."""

    def __init__(self, required: set[str] | None = None, **kwargs) -> None:
        super().__init__()
        self.schema = {
            "type": "object",
            "properties": {k: json_schema(kwargs[k]) for k in kwargs},
        }
        if required:
            self.schema["required"] = sorted(required)
        self.kwargs = kwargs

    def coerce(self, data: dict) -> dict:
        """Type-coerce a raw dict using declared parameter types."""
        return {k: _coerce(self.kwargs[k], data[k]) for k in self.kwargs if k in data}

    def decode(self, s: str) -> dict:
        """Decode a JSON string to a dictionary."""
        # Handle None or empty string arguments (for tools with no parameters)
        if not s or s.strip() == "":
            data = {}
        else:
            data = super().decode(s)

        return self.coerce(data)


class ReturnType(json.JSONEncoder):
    """Encode return values and expose a JSON Schema for them."""

    def __init__(self, t: type) -> None:
        super().__init__()
        self.schema = json_schema(t)

    def default(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        return obj


class FunctionTool(Tool):
    """Wrap an async callable as a Tool with schema/serialization."""

    def __init__(
        self,
        func: Callable[..., Awaitable],
        name: str | None = None,
        schema: dict | type[BaseModel] | None = None,
        input_guardrails: list[Callable] | None = None,
        output_guardrails: list[Callable] | None = None,
        requires_approval: bool | None = None,
    ) -> None:
        # Support callable objects.  Keep functools.partial (and other
        # callables wrapping async functions) as-is so that
        # iscoroutinefunction() still detects them correctly.
        if isfunction(func) or ismethod(func) or iscoroutinefunction(func):
            self.func = func
        else:
            self.func = func.__call__

        tool_name = name or resolve_tool_name(func) or self.func.__name__
        # Store tool name on function for introspection/tracing (if possible)
        try:
            self.func.__tool_name__ = tool_name
        except (AttributeError, TypeError):
            # Methods and some callables don't support setting attributes
            # For methods, try setting on the underlying function
            if ismethod(self.func):
                try:
                    self.func.__func__.__tool_name__ = tool_name
                except (AttributeError, TypeError):
                    pass  # Can't set attribute, will fall back to __name__

        description = getdoc(func) or None
        hints = get_type_hints(self.func, include_extras=True)
        signature = inspect.signature(self.func)
        ignored = {"self", "cls"}
        parameters = {
            pname: hints[pname]
            for pname, param in signature.parameters.items()
            if pname in hints
            and pname not in ignored
            and not pname.startswith("_")  # ignore private parameters
            and param.kind  # ignore parameters with no kind
            # ignore parameters with varargs and varkw
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        }
        required = {
            pname
            for pname, param in signature.parameters.items()
            if pname in parameters and param.default is inspect._empty
        }
        self.params = Parameters(required=required, **parameters)
        self.schema_model: type[BaseModel] | None = None
        if schema is None:
            schema = _get_tool_attr(func, "__tool_schema__")
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            self.schema_model = schema
            schema = schema.model_json_schema()
        tool_json_schema = schema if schema is not None else self.params.schema
        if "return" in hints:
            self.ret_type = ReturnType(hints["return"])
        else:
            self.ret_type = ReturnType(type(None))

        # Resolve guardrails: explicit > function attribute > empty
        if input_guardrails is None:
            input_guardrails = _get_tool_attr(func, "__tool_input_guardrails__") or []
        if output_guardrails is None:
            output_guardrails = _get_tool_attr(func, "__tool_output_guardrails__") or []

        # Resolve requires_approval: explicit True/False wins; None means "inherit
        # from decorator-set attribute".
        if requires_approval is None:
            requires_approval = bool(
                _get_tool_attr(func, "__tool_requires_approval__") or False
            )

        super().__init__(
            name=tool_name,
            description=description,
            json_schema=tool_json_schema,
            input_guardrails=input_guardrails,
            output_guardrails=output_guardrails,
            requires_approval=requires_approval,
        )

    def __call__(self, args: str):
        """Parse JSON args with type coercion, then delegate to _execute."""
        if self.schema_model is not None:
            payload = json.loads(args)
            if not isinstance(payload, dict):
                raise ValueError("Tool arguments must be a JSON object")
            kwargs = self.schema_model.model_validate(payload).model_dump()
        else:
            kwargs = self.params.decode(args)
        return self._execute(**kwargs)

    async def _invoke(self, **kwargs) -> Any:
        """Call the wrapped Python function (sync or async)."""
        if iscoroutinefunction(self.func):
            return await self.func(**kwargs)
        else:
            return await asyncio.to_thread(self.func, **kwargs)

    def _serialize(self, result: Any) -> str:
        """Encode using ReturnType (handles dataclasses, BaseModel, etc.)."""
        try:
            return self.ret_type.encode(result)
        except (TypeError, ValueError) as exc:
            return json.dumps(
                {
                    "error": f"Non-serializable tool result: {exc}",
                    "value": str(result),
                }
            )
