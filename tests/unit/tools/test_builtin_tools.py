import json
import os
import tempfile
import unittest
from inspect import getdoc

from pydantic import ConfigDict, Field

from motus.tools import InputSchema, LocalShell, builtin_tools, normalize_tools, tool
from motus.tools.builtins import BuiltinTools
from motus.tools.builtins._helpers import (
    add_line_numbers,
    truncate_line,
    truncate_output,
)
from motus.tools.core.function_tool import FunctionTool

# All tests use an explicit LocalShell so they don't depend on Docker /
# the default provider chain.
_SHELL = LocalShell()


def _bt() -> BuiltinTools:
    return builtin_tools(_SHELL)


class TestBuiltinTools(unittest.TestCase):
    """Tests for the builtin_tools() factory function."""

    # --- basics ---

    def test_returns_builtin_tools_object(self):
        self.assertIsInstance(_bt(), BuiltinTools)

    def test_len_is_seven(self):
        self.assertEqual(len(_bt()), 7)

    def test_iter_returns_seven_tools(self):
        names = [t.__name__ for t in _bt()]
        self.assertEqual(
            names,
            [
                "bash",
                "read_file",
                "write_file",
                "edit_file",
                "glob_search",
                "grep_search",
                "to_do",
            ],
        )

    def test_attribute_access(self):
        bt = _bt()
        self.assertEqual(bt.bash.__name__, "bash")
        self.assertEqual(bt.read_file.__name__, "read_file")
        self.assertEqual(bt.grep_search.__name__, "grep_search")

    def test_normalize_tools_compatible(self):
        bt = _bt()
        normalized = normalize_tools(bt)
        self.assertEqual(len(normalized), 7)
        self.assertIn("bash", normalized)
        self.assertIn("read_file", normalized)

    def test_spread_into_list(self):
        self.assertEqual(len([*_bt()]), 7)

    # --- tool() as patcher: description ---

    def test_tool_patch_description(self):
        bt = _bt()
        tool(bt.bash, description="Run stuff")
        self.assertEqual(getdoc(bt.bash), "Run stuff")

    # --- tool() as patcher: schema ---

    def test_tool_patch_schema(self):
        class MinimalBash(InputSchema):
            command: str = Field(description="cmd")
            model_config = ConfigDict(extra="forbid")

        bt = _bt()
        tool(bt.bash, schema=MinimalBash)
        ft = FunctionTool(bt.bash)
        self.assertIs(ft.schema_model, MinimalBash)
        self.assertIn("command", ft.json_schema["properties"])
        self.assertNotIn("timeout", ft.json_schema["properties"])

    # --- tool() as patcher: guardrails ---

    def test_tool_patch_input_guardrails(self):
        def block_rm(kwargs):
            if "rm " in kwargs.get("command", ""):
                raise ValueError("rm not allowed")
            return kwargs

        bt = _bt()
        tool(bt.bash, input_guardrails=[block_rm])
        ft = FunctionTool(bt.bash)
        self.assertEqual(ft._input_guardrails, [block_rm])

    def test_tool_patch_output_guardrails(self):
        def redact(result):
            return result.replace("secret", "***")

        bt = _bt()
        tool(bt.read_file, output_guardrails=[redact])
        ft = FunctionTool(bt.read_file)
        self.assertEqual(ft._output_guardrails, [redact])

    # --- tool() as patcher: name ---

    def test_tool_patch_name(self):
        bt = _bt()
        tool(bt.bash, name="shell")
        ft = FunctionTool(bt.bash)
        self.assertEqual(ft.name, "shell")

    # --- tool() as patcher: hooks ---

    def test_tool_patch_hooks(self):
        bt = _bt()
        tool(bt.bash, on_start=lambda ctx: None)
        # Just verify no error; hook registration is global.

    # --- subset selection via attributes ---

    def test_pick_subset(self):
        bt = _bt()
        subset = [bt.bash, bt.read_file]
        normalized = normalize_tools(subset)
        self.assertEqual(len(normalized), 2)
        self.assertIn("bash", normalized)
        self.assertIn("read_file", normalized)

    # --- rich docstrings ---

    def test_bash_has_rich_docstring(self):
        bt = _bt()
        doc = getdoc(bt.bash)
        self.assertIn("IMPORTANT", doc)
        self.assertIn("truncated", doc)

    def test_read_file_has_rich_docstring(self):
        bt = _bt()
        doc = getdoc(bt.read_file)
        self.assertIn("line numbers", doc)
        self.assertIn("2000", doc)

    def test_edit_file_has_rich_docstring(self):
        bt = _bt()
        doc = getdoc(bt.edit_file)
        self.assertIn("IMPORTANT", doc)
        self.assertIn("line numbers", doc)

    def test_glob_search_has_rich_docstring(self):
        bt = _bt()
        doc = getdoc(bt.glob_search)
        self.assertIn("**", doc)
        self.assertIn("grep_search", doc)

    def test_grep_search_has_rich_docstring(self):
        bt = _bt()
        doc = getdoc(bt.grep_search)
        self.assertIn("files_with_matches", doc)
        self.assertIn("glob_search", doc)


class TestHelpers(unittest.TestCase):
    """Tests for the _helpers module."""

    def test_truncate_output_under_limit(self):
        text = "short text"
        self.assertEqual(truncate_output(text), text)

    def test_truncate_output_over_limit(self):
        text = "x" * 35000
        result = truncate_output(text, limit=30000)
        self.assertTrue(result.startswith("x" * 30000))
        self.assertIn("truncated", result)
        self.assertIn("35000", result)

    def test_truncate_output_exact_limit(self):
        text = "x" * 30000
        self.assertEqual(truncate_output(text, limit=30000), text)

    def test_truncate_line_under_limit(self):
        line = "short line"
        self.assertEqual(truncate_line(line), line)

    def test_truncate_line_over_limit(self):
        line = "x" * 3000
        result = truncate_line(line, limit=2000)
        self.assertEqual(len(result), 2003)  # 2000 + "..."
        self.assertTrue(result.endswith("..."))

    def test_add_line_numbers_basic(self):
        text = "first\nsecond\nthird\n"
        result = add_line_numbers(text)
        self.assertIn("     1\tfirst", result)
        self.assertIn("     2\tsecond", result)
        self.assertIn("     3\tthird", result)

    def test_add_line_numbers_with_offset(self):
        text = "line10\nline11\n"
        result = add_line_numbers(text, start=10)
        self.assertIn("    10\tline10", result)
        self.assertIn("    11\tline11", result)

    def test_add_line_numbers_empty(self):
        result = add_line_numbers("")
        self.assertEqual(result, "")

    def test_add_line_numbers_truncates_long_lines(self):
        text = "x" * 3000 + "\n"
        result = add_line_numbers(text)
        # Line should be truncated + "..."
        self.assertIn("...", result)


class TestBashTool(unittest.IsolatedAsyncioTestCase):
    """Tests for the bash tool improvements."""

    async def test_bash_runs_command(self):
        bt = _bt()
        ft = FunctionTool(bt.bash)
        result = await ft('{"command": "echo hello"}')
        self.assertIn("hello", result)

    async def test_bash_timeout_returns_error(self):
        bt = _bt()
        ft = FunctionTool(bt.bash)
        # 1 second timeout with a 10 second sleep
        result = await ft('{"command": "sleep 10", "timeout": 1000}')
        self.assertIn("timed out", result)

    async def test_bash_truncates_long_output(self):
        bt = _bt()
        ft = FunctionTool(bt.bash)
        # Generate output larger than 30000 chars
        result = await ft('{"command": "python3 -c \\"print(\'x\' * 40000)\\""}')
        self.assertIn("truncated", result)


class TestReadFileTool(unittest.IsolatedAsyncioTestCase):
    """Tests for the read_file tool improvements."""

    async def test_read_file_has_line_numbers(self):
        bt = _bt()
        ft = FunctionTool(bt.read_file)
        # Read this test file itself
        path = os.path.abspath(__file__)
        raw = await ft(f'{{"file_path": "{path}"}}')
        result = json.loads(raw)  # FunctionTool JSON-encodes the result
        # Should contain line numbers in cat -n format
        self.assertRegex(result, r"\s+1\t")

    async def test_read_file_respects_offset_and_limit(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for i in range(1, 101):
                f.write(f"line {i}\n")
            path = f.name

        try:
            bt = _bt()
            ft = FunctionTool(bt.read_file)
            raw = await ft(f'{{"file_path": "{path}", "offset": 10, "limit": 5}}')
            result = json.loads(raw)  # FunctionTool JSON-encodes the result
            # Should have lines 10-14
            self.assertIn("    10\t", result)
            self.assertIn("    14\t", result)
            self.assertNotIn("    15\t", result)
            self.assertNotIn("     9\t", result)
        finally:
            os.unlink(path)

    async def test_read_file_nonexistent_returns_error(self):
        bt = _bt()
        ft = FunctionTool(bt.read_file)
        result = await ft('{"file_path": "/nonexistent/file.txt"}')
        # Should contain some error indication
        self.assertTrue(len(result) > 0)


class TestGlobSearchTool(unittest.IsolatedAsyncioTestCase):
    """Tests for the glob_search tool improvements."""

    async def test_glob_simple_pattern(self):
        bt = _bt()
        ft = FunctionTool(bt.glob_search)
        # Search for this test file
        path = os.path.dirname(os.path.abspath(__file__))
        result = await ft(f'{{"pattern": "test_builtin_tools.py", "path": "{path}"}}')
        self.assertIn("test_builtin_tools.py", result)

    async def test_glob_recursive_pattern(self):
        bt = _bt()
        ft = FunctionTool(bt.glob_search)
        # Search for Python files recursively
        path = os.path.dirname(os.path.abspath(__file__))
        result = await ft(f'{{"pattern": "**/*.py", "path": "{path}"}}')
        self.assertIn(".py", result)


class TestGrepSearchTool(unittest.IsolatedAsyncioTestCase):
    """Tests for the grep_search tool improvements."""

    async def test_grep_type_parameter(self):
        bt = _bt()
        ft = FunctionTool(bt.grep_search)
        path = os.path.dirname(os.path.abspath(__file__))
        result = await ft(
            f'{{"pattern": "import", "type": "py", "path": "{path}", "output_mode": "files_with_matches"}}'
        )
        # Should find .py files only
        if result.strip():
            for line in result.strip().split("\n"):
                self.assertIn(".py", line)

    async def test_grep_offset_and_head_limit(self):
        bt = _bt()
        ft = FunctionTool(bt.grep_search)
        path = os.path.dirname(os.path.abspath(__file__))
        # Get first 3 results, skipping the first 1
        result = await ft(
            f'{{"pattern": "def ", "type": "py", "path": "{path}", '
            f'"output_mode": "content", "offset": 1, "head_limit": 3}}'
        )
        lines = [l for l in result.strip().split("\n") if l.strip()]
        self.assertLessEqual(len(lines), 3)
