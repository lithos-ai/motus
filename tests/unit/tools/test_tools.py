import json
import unittest
from dataclasses import dataclass, is_dataclass
from unittest import mock

from pydantic import BaseModel, Field

from motus.tools import (
    DictTools,
    FunctionTool,
    InputSchema,
    normalize_tools,
    tool,
    tools,
    tools_from,
)


class TestFunctionTool(unittest.IsolatedAsyncioTestCase):
    async def test_schema_includes_docstring(self):
        async def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        tool = FunctionTool(add)

        self.assertEqual(tool.description, "Add two numbers.")
        self.assertEqual(
            tool.json_schema,
            {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        )

    async def test_call_serializes_primitives(self):
        async def add(a: int, b: int) -> int:
            return a + b

        tool = FunctionTool(add)

        result = await tool(json.dumps({"a": 1, "b": 2}))

        self.assertEqual(result, "3")

    async def test_call_serializes_mappings(self):
        async def build(label: str, count: int) -> dict[str, object]:
            return {"label": label, "count": count}

        tool = FunctionTool(build)

        result = await tool(json.dumps({"label": "ok", "count": 2}))

        self.assertEqual(json.loads(result), {"label": "ok", "count": 2})

    async def test_call_handles_nested_dataclasses(self):
        @dataclass
        class Child:
            name: str

        @dataclass
        class Parent:
            child: Child
            children: list[Child]

        async def echo(parent: Parent) -> Parent:
            self.assertTrue(is_dataclass(parent))
            self.assertTrue(is_dataclass(parent.child))
            self.assertTrue(
                all(is_dataclass(child) for child in parent.children), parent.children
            )
            return parent

        tool = FunctionTool(echo)

        result = await tool(
            json.dumps(
                {
                    "parent": {
                        "child": {"name": "root"},
                        "children": [{"name": "a"}, {"name": "b"}],
                    }
                }
            )
        )

        self.assertEqual(
            json.loads(result),
            {
                "child": {"name": "root"},
                "children": [{"name": "a"}, {"name": "b"}],
            },
        )

    async def test_call_handles_nested_basemodels(self):
        class Inner(BaseModel):
            name: str

        class Outer(BaseModel):
            child: Inner
            children: list[Inner]
            meta: dict[str, Inner]

        async def echo(payload: Outer) -> Outer:
            self.assertIsInstance(payload, Outer)
            self.assertIsInstance(payload.child, Inner)
            self.assertTrue(all(isinstance(child, Inner) for child in payload.children))
            self.assertTrue(
                all(isinstance(child, Inner) for child in payload.meta.values())
            )
            return payload

        tool = FunctionTool(echo)

        result = await tool(
            json.dumps(
                {
                    "payload": {
                        "child": {"name": "root"},
                        "children": [{"name": "a"}, {"name": "b"}],
                        "meta": {"x": {"name": "x"}},
                    }
                }
            )
        )

        self.assertEqual(
            json.loads(result),
            {
                "child": {"name": "root"},
                "children": [{"name": "a"}, {"name": "b"}],
                "meta": {"x": {"name": "x"}},
            },
        )


class TestToolWrappers(unittest.IsolatedAsyncioTestCase):
    async def test_class_tool_factory_supports_constructor_args(self):
        class ListFilesInput(InputSchema):
            path: str = Field(description="Root path")
            limit: int | None = Field(default=None, description="Limit results")

        @tools(prefix="sand_", method_schemas={"list_files": ListFilesInput})
        class Sandbox:
            def __init__(self, root_dir: str):
                self.root_dir = root_dir

            async def list_files(
                self, path: str, limit: int | None = None
            ) -> list[str]:
                return [self.root_dir, path, str(limit)]

            async def echo(self, value: str) -> str:
                return value

        tool_list = Sandbox("/tmp")
        tool_map = normalize_tools(tool_list)
        self.assertIn("sand_list_files", tool_map)
        self.assertIn("sand_echo", tool_map)

        list_schema = tool_map["sand_list_files"].json_schema
        self.assertEqual(list_schema["type"], "object")
        self.assertNotIn("title", list_schema)
        self.assertEqual(list_schema["properties"]["path"]["type"], "string")
        self.assertEqual(list_schema["properties"]["path"]["description"], "Root path")
        self.assertEqual(list_schema["properties"]["limit"]["type"], "integer")
        self.assertNotIn("default", list_schema["properties"]["limit"])
        self.assertEqual(list_schema["required"], ["path"])

        echo_schema = tool_map["sand_echo"].json_schema
        self.assertEqual(echo_schema["required"], ["value"])

    async def test_tools_from_requires_instance(self):
        class Dummy:
            async def ping(self) -> str:
                return "pong"

        with self.assertRaises(TypeError):
            tools_from(Dummy)

    async def test_tool_schema_override_for_function(self):
        class GreetInput(InputSchema):
            name: str = Field(description="Person name")

        async def greet(name: str) -> str:
            return name

        greet = tool(schema=GreetInput)(greet)
        tool_map = normalize_tools(greet)
        schema = tool_map["greet"].json_schema
        self.assertEqual(schema["properties"]["name"]["description"], "Person name")

    async def test_pydantic_schema_drives_argument_parsing(self):
        class CountInput(InputSchema):
            count: int

        async def add_one(count: int) -> int:
            return count + 1

        add_one = tool(schema=CountInput)(add_one)
        tool_map = normalize_tools(add_one)
        result = await tool_map["add_one"](json.dumps({"count": "2"}))
        self.assertEqual(result, "3")

    async def test_normalize_tools_requires_names(self):
        class NoNameTool:
            async def __call__(self, args: str) -> str:
                return args

        with self.assertRaises(ValueError):
            normalize_tools([NoNameTool()])

    async def test_normalize_tools_accepts_single_tool(self):
        class NamedTool:
            name = "named"

            async def __call__(self, args: str) -> str:
                return args

        tool_map = normalize_tools(NamedTool())
        self.assertEqual(set(tool_map.keys()), {"named"})

    async def test_normalize_tools_accepts_single_callable(self):
        async def ping() -> str:
            return "pong"

        tool_map = normalize_tools(ping)
        self.assertIn("ping", tool_map)

    async def test_normalize_tools_accepts_dict_tools(self):
        async def ping(args: str) -> str:
            return "pong"

        inner_tool = FunctionTool(ping, name="ping")
        dict_tools = DictTools({"ping": inner_tool})
        tool_map = normalize_tools(dict_tools)
        self.assertEqual(set(tool_map.keys()), {"ping"})

    async def test_normalize_tools_accepts_tools_in_list(self):
        async def ping(args: str) -> str:
            return "pong"

        async def pong() -> str:
            return "ping"

        inner_tool = FunctionTool(ping, name="ping")
        dict_tools = DictTools({"ping": inner_tool})
        tool_map = normalize_tools([dict_tools, pong])
        self.assertEqual(set(tool_map.keys()), {"ping", "pong"})

    async def test_tools_method_aliases(self):
        class Example:
            async def read(self) -> str:
                return "read"

            async def write(self) -> str:
                return "write"

        Example = tools(method_aliases={"read": "load"})(Example)
        tool_map = normalize_tools(Example())
        self.assertEqual(set(tool_map.keys()), {"load", "write"})

    async def test_tool_decorator_on_method_sets_metadata(self):
        class Example:
            @tool(name="custom_read", schema={"type": "object"})
            async def read(self) -> str:
                return "read"

        instance = Example()
        instance = tools(instance)
        tool_map = normalize_tools(instance)
        self.assertIn("custom_read", tool_map)

    async def test_tool_allowlist_and_blocklist(self):
        class Example:
            async def a(self) -> str:
                return "a"

            async def b(self) -> str:
                return "b"

            async def c(self) -> str:
                return "c"

        with mock.patch("motus.tools.core.normalize.logging.warning") as warn:
            Example = tools(allowlist={"a", "c", "missing"})(Example)
            tool_map = normalize_tools(Example())
            self.assertEqual(set(tool_map.keys()), {"a", "c"})
            warn.assert_called()

        Example = tools(blocklist={"b"})(Example)
        tool_map = normalize_tools(Example())
        self.assertEqual(set(tool_map.keys()), {"a", "c"})

        with mock.patch("motus.tools.core.normalize.logging.warning") as warn:
            Example = tools(allowlist={"b"}, blocklist={"b", "c"})(Example)
            tool_map = normalize_tools(Example())
            self.assertEqual(set(tool_map.keys()), {"b"})
            warn.assert_called()


def test_tool_decorator_sets_requires_approval_attribute():
    from motus.tools import tool

    @tool(requires_approval=True)
    def delete_file(path: str):
        """Delete a file."""
        import os

        os.remove(path)

    assert getattr(delete_file, "__tool_requires_approval__", False) is True


def test_tool_decorator_default_requires_approval_false():
    from motus.tools import tool

    @tool
    def read_file(path: str):
        """Read a file."""
        return open(path).read()

    assert getattr(read_file, "__tool_requires_approval__", False) is False


def test_tool_with_requires_approval_injects_guardrail():
    """Tool.__init__ with requires_approval=True should insert an approval
    guardrail at position 0 of _input_guardrails."""
    from motus.tools.core.tool import Tool

    class DummyTool(Tool):
        async def _invoke(self, **kwargs):
            return "ok"

    t = DummyTool(
        name="dummy",
        description="test",
        json_schema={},
        requires_approval=True,
    )
    assert t.requires_approval is True
    assert len(t._input_guardrails) == 1
    # guardrail is a closure function
    assert callable(t._input_guardrails[0])
    assert t._input_guardrails[0].__name__ == "_builtin_approval_guardrail"


def test_tool_without_requires_approval_no_guardrail_injected():
    from motus.tools.core.tool import Tool

    class DummyTool(Tool):
        async def _invoke(self, **kwargs):
            return "ok"

    t = DummyTool(name="dummy", description="test", json_schema={})
    assert t.requires_approval is False
    assert t._input_guardrails == []


def test_function_tool_forwards_requires_approval_from_decorator():
    from motus.tools import tool
    from motus.tools.core.function_tool import FunctionTool

    @tool(requires_approval=True)
    def delete_file(path: str):
        """Delete a file."""

    ft = FunctionTool(delete_file)
    assert ft.requires_approval is True
    # approval guardrail was injected at position 0
    assert len(ft._input_guardrails) >= 1


def test_function_tool_explicit_requires_approval_overrides_decorator():
    from motus.tools import tool
    from motus.tools.core.function_tool import FunctionTool

    @tool(requires_approval=True)
    def delete_file(path: str):
        """Delete a file."""

    # Explicit False should override the decorator's True
    ft = FunctionTool(delete_file, requires_approval=False)
    assert ft.requires_approval is False
    assert ft._input_guardrails == []
