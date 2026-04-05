"""Unit tests for MCPSession, MCPTool, normalize_tools MCP paths, and get_mcp."""

import unittest
from unittest import mock

import mcp
import mcp.types

from motus.tools import (
    MCPSession,
    MCPTool,
    get_mcp,
    normalize_tools,
    tool,
    tools,
    tools_from,
)
from motus.tools.core.tool import DictTools


def _make_mcp_tool(name: str, description: str = "", schema: dict | None = None):
    """Create a fake mcp.types.Tool for testing."""
    return mcp.types.Tool(
        name=name,
        description=description,
        inputSchema=schema or {"type": "object", "properties": {}},
    )


def _make_connected_session(tool_names: list[str]) -> MCPSession:
    """Create an MCPSession that looks connected with fake tools."""
    session = MCPSession(url="http://fake")
    session._session = mock.MagicMock()  # non-None → "connected"
    mcp_tools = [_make_mcp_tool(n) for n in tool_names]
    session._populate_tools(mcp_tools)
    return session


class TestMCPSessionConstruction(unittest.TestCase):
    def test_requires_url_or_command_or_sandbox(self):
        with self.assertRaises(ValueError):
            MCPSession()

    def test_url_constructor(self):
        session = MCPSession(url="http://localhost:8080")
        self.assertEqual(session._url, "http://localhost:8080")
        self.assertIsNone(session._server_params)
        self.assertIsNone(session._session)

    def test_command_constructor(self):
        session = MCPSession(command="npx", args=["-y", "server"])
        self.assertIsNone(session._url)
        self.assertIsNotNone(session._server_params)
        self.assertEqual(session._server_params.command, "npx")
        self.assertEqual(session._server_params.args, ["-y", "server"])

    def test_sandbox_constructor(self):
        sb = mock.MagicMock()
        session = MCPSession(
            sandbox=sb, sandbox_command=["npx", "server"], sandbox_port=8080
        )
        self.assertIs(session._sandbox, sb)
        self.assertEqual(session._sandbox_command, ["npx", "server"])
        self.assertEqual(session._sandbox_port, 8080)
        self.assertIsNone(session._url)
        self.assertIsNone(session._server_params)

    def test_sandbox_requires_command_and_port(self):
        sb = mock.MagicMock()
        with self.assertRaises(ValueError):
            MCPSession(sandbox=sb)
        with self.assertRaises(ValueError):
            MCPSession(sandbox=sb, sandbox_command=["cmd"])
        with self.assertRaises(ValueError):
            MCPSession(sandbox=sb, sandbox_port=8080)

    def test_mutually_exclusive_modes(self):
        sb = mock.MagicMock()
        with self.assertRaises(ValueError):
            MCPSession(url="http://fake", command="cmd")
        with self.assertRaises(ValueError):
            MCPSession(
                url="http://fake",
                sandbox=sb,
                sandbox_command=["cmd"],
                sandbox_port=8080,
            )

    def test_always_truthy(self):
        session = MCPSession(url="http://fake")
        self.assertTrue(bool(session))

    def test_empty_before_connect(self):
        session = MCPSession(url="http://fake")
        self.assertEqual(len(session), 0)
        self.assertEqual(list(session), [])


class TestMCPSessionPopulateTools(unittest.TestCase):
    def test_populate_creates_mcptools_and_attrs(self):
        session = _make_connected_session(["read_file", "write_file"])

        self.assertEqual(len(session), 2)
        self.assertIn("read_file", session)
        self.assertIn("write_file", session)
        self.assertIsInstance(session["read_file"], MCPTool)
        self.assertIsInstance(session.read_file, MCPTool)
        self.assertIs(session["read_file"], session.read_file)

    def test_reserved_name_gets_prefix(self):
        # "close" is a method on MCPSession — should get mcptool_ prefix
        session = _make_connected_session(["close", "normal_tool"])

        self.assertIn("close", session._tools)
        self.assertIsInstance(session.mcptool_close, MCPTool)
        self.assertIsInstance(session.normal_tool, MCPTool)

    def test_all_builtins_are_reserved(self):
        """Any name in dir(MCPSession) should be prefixed."""
        session = MCPSession(url="http://fake")
        reserved_candidates = ["close", "aclose", "connect", "connect_stdio"]
        for name in reserved_candidates:
            self.assertIn(name, dir(session))


class TestMCPSessionClose(unittest.TestCase):
    def test_close_calls_on_close_once(self):
        called = []
        session = MCPSession(url="http://fake", on_close=lambda s: called.append(s))

        session.close()
        session.close()  # second call should be no-op

        self.assertEqual(len(called), 1)
        self.assertIs(called[0], session)

    def test_close_discards_async_on_close(self):
        """Sync close should not raise on async on_close callback."""

        async def async_cb(s):
            pass

        session = MCPSession(url="http://fake", on_close=async_cb)
        session.close()  # should not raise


class TestMCPSessionAclose(unittest.IsolatedAsyncioTestCase):
    async def test_aclose_awaits_async_on_close(self):
        called = []

        async def async_cb(s):
            called.append(s)

        session = MCPSession(url="http://fake", on_close=async_cb)
        await session.aclose()

        self.assertEqual(len(called), 1)
        self.assertIsNone(session._session)

    async def test_aclose_handles_sync_on_close(self):
        called = []
        session = MCPSession(url="http://fake", on_close=lambda s: called.append(s))
        await session.aclose()
        self.assertEqual(len(called), 1)


class TestNormalizeToolsMCP(unittest.TestCase):
    def test_connected_session_registers_tools(self):
        session = _make_connected_session(["alpha", "beta"])

        dt = normalize_tools([session])

        self.assertIn("alpha", dt)
        self.assertIn("beta", dt)
        self.assertEqual(len(dt._owned_sessions), 0)

    def test_unconnected_session_defers_to_owned(self):
        session = MCPSession(url="http://fake")

        dt = normalize_tools([session])

        # No tools registered yet
        self.assertEqual(len(dt), 0)
        self.assertEqual(len(dt._owned_sessions), 1)
        self.assertIs(dt._owned_sessions[0], session)

    def test_mixed_connected_and_unconnected(self):
        connected = _make_connected_session(["tool_a"])
        unconnected = MCPSession(command="npx", args=["server"])

        dt = normalize_tools([connected, unconnected])

        self.assertIn("tool_a", dt)
        self.assertEqual(len(dt._owned_sessions), 1)
        self.assertIs(dt._owned_sessions[0], unconnected)

    def test_single_session_not_in_list(self):
        session = MCPSession(url="http://fake")
        dt = normalize_tools(session)
        self.assertEqual(len(dt._owned_sessions), 1)

    def test_connected_session_with_prefix(self):
        session = _make_connected_session(["read", "write"])
        session = tools(session, prefix="fs_")

        dt = normalize_tools([session])

        self.assertIn("fs_read", dt)
        self.assertIn("fs_write", dt)
        self.assertNotIn("read", dt)

    def test_connected_session_with_blocklist(self):
        session = _make_connected_session(["read", "write", "delete"])
        session = tools(session, blocklist={"delete"})

        dt = normalize_tools([session])

        self.assertIn("read", dt)
        self.assertIn("write", dt)
        self.assertNotIn("delete", dt)

    def test_connected_session_with_allowlist(self):
        session = _make_connected_session(["read", "write", "delete"])
        session = tools(session, allowlist={"read"})

        dt = normalize_tools([session])

        self.assertIn("read", dt)
        self.assertNotIn("write", dt)
        self.assertNotIn("delete", dt)


class TestToolsFrom(unittest.TestCase):
    def test_connected_session_returns_tools(self):
        session = _make_connected_session(["a", "b"])
        result = tools_from(session)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(isinstance(t, MCPTool) for t in result))

    def test_unconnected_session_returns_empty(self):
        session = MCPSession(url="http://fake")
        result = tools_from(session)
        self.assertEqual(result, [])

    def test_tools_from_with_blocklist(self):
        session = _make_connected_session(["a", "b", "c"])
        result = tools_from(session, blocklist={"b"})
        names = [t.name for t in result]
        self.assertIn("a", names)
        self.assertNotIn("b", names)
        self.assertIn("c", names)


class TestDictToolsConnectMcpSessions(unittest.IsolatedAsyncioTestCase):
    async def test_connect_owned_sessions(self):
        """_connect_mcp_sessions connects unconnected sessions and registers tools."""
        session = MCPSession(url="http://fake")

        # Mock _connect_coro to simulate connection
        async def fake_connect():
            session._session = mock.MagicMock()
            mcp_tools = [_make_mcp_tool("tool_x"), _make_mcp_tool("tool_y")]
            session._populate_tools(mcp_tools)

        session._connect_coro = fake_connect

        dt = normalize_tools([session])
        self.assertEqual(len(dt), 0)  # no tools yet

        await dt._connect_mcp_sessions()

        self.assertIn("tool_x", dt)
        self.assertIn("tool_y", dt)
        self.assertEqual(len(dt), 2)

    async def test_connect_skips_already_connected(self):
        """Sessions that are already connected should not be re-connected."""
        session = _make_connected_session(["existing"])

        call_count = 0
        original_connect = session._connect_coro

        async def counting_connect():
            nonlocal call_count
            call_count += 1
            await original_connect()

        session._connect_coro = counting_connect

        dt = DictTools(dict(session._tools), owned_sessions=[session])
        await dt._connect_mcp_sessions()

        self.assertEqual(call_count, 0)

    async def test_no_owned_sessions_is_noop(self):
        dt = DictTools({})
        await dt._connect_mcp_sessions()  # should not raise

    async def test_close_closes_owned_sessions(self):
        session = MCPSession(url="http://fake")
        close_called = []
        session.close = lambda: close_called.append(True)

        dt = DictTools({}, owned_sessions=[session])
        dt.close()

        self.assertEqual(len(close_called), 1)


class TestToolDecoratorOnMCPTool(unittest.TestCase):
    def test_tool_decorator_sets_name(self):
        session = _make_connected_session(["original_name"])
        mcp_tool = session["original_name"]

        result = tool(name="renamed")(mcp_tool)

        self.assertIs(result, mcp_tool)
        self.assertEqual(mcp_tool.name, "renamed")

    def test_tool_decorator_sets_schema(self):
        session = _make_connected_session(["my_tool"])
        mcp_tool = session["my_tool"]
        new_schema = {"type": "object", "properties": {"x": {"type": "integer"}}}

        tool(schema=new_schema)(mcp_tool)

        self.assertEqual(mcp_tool.json_schema, new_schema)

    def test_tool_decorator_sets_guardrails(self):
        session = _make_connected_session(["my_tool"])
        mcp_tool = session["my_tool"]

        def input_guard(x: str = ""):
            pass

        def output_guard(result):
            pass

        tool(input_guardrails=[input_guard], output_guardrails=[output_guard])(mcp_tool)

        self.assertEqual(mcp_tool._input_guardrails, [input_guard])
        self.assertEqual(mcp_tool._output_guardrails, [output_guard])


class TestGetMcp(unittest.TestCase):
    def test_url_returns_session(self):
        session = get_mcp(url="http://localhost:8080")
        self.assertIsInstance(session, MCPSession)
        self.assertEqual(session._url, "http://localhost:8080")

    def test_command_returns_session(self):
        session = get_mcp(command="npx", args=["-y", "server"])
        self.assertIsInstance(session, MCPSession)
        self.assertEqual(session._server_params.command, "npx")

    def test_image_creates_sandbox_session(self):
        with mock.patch("motus.tools.get_sandbox") as mock_sandbox:
            sb = mock.MagicMock()
            mock_sandbox.return_value = sb
            session = get_mcp(
                image="node:20",
                command="npx",
                args=["@playwright/mcp", "--port", "8080"],
            )

        mock_sandbox.assert_called_once_with(
            image="node:20", env=None, ports={8080: None}
        )
        self.assertIs(session._sandbox, sb)
        self.assertEqual(
            session._sandbox_command, ["npx", "@playwright/mcp", "--port", "8080"]
        )
        self.assertIsNone(session._sandbox_env)
        self.assertEqual(session._sandbox_port, 8080)
        # No docker exec — server_params should be None.
        self.assertIsNone(session._server_params)

    def test_image_custom_port_with_env(self):
        with mock.patch("motus.tools.get_sandbox") as mock_sandbox:
            sb = mock.MagicMock()
            mock_sandbox.return_value = sb
            session = get_mcp(
                image="node:20",
                command="npx",
                args=["server", "streamableHttp"],
                env={"PORT": "3000"},
                port=3000,
            )

        mock_sandbox.assert_called_once_with(
            image="node:20", env={"PORT": "3000"}, ports={3000: None}
        )
        self.assertEqual(session._sandbox_command, ["npx", "server", "streamableHttp"])
        self.assertEqual(session._sandbox_env, {"PORT": "3000"})
        self.assertEqual(session._sandbox_port, 3000)

    def test_sandbox_reuses_existing(self):
        sb = mock.MagicMock()
        session = get_mcp(
            sandbox=sb, command="node", args=["server.js", "--port", "8080"]
        )

        self.assertIs(session._sandbox, sb)
        self.assertEqual(
            session._sandbox_command, ["node", "server.js", "--port", "8080"]
        )
        self.assertIsNone(session._sandbox_env)
        self.assertEqual(session._sandbox_port, 8080)

    def test_user_sandbox_not_closed_on_session_close(self):
        """User-provided sandbox should NOT be auto-closed."""
        sb = mock.MagicMock()
        session = get_mcp(sandbox=sb, command="node")

        session.close()
        sb.close.assert_not_called()

    def test_image_sandbox_closed_on_session_close(self):
        """Auto-created sandbox (via image=) should be closed."""
        with mock.patch("motus.tools.get_sandbox") as mock_gs:
            sb = mock.MagicMock()
            mock_gs.return_value = sb
            session = get_mcp(image="node:20", command="node")

        session.close()
        sb.close.assert_called_once()

    def test_no_url_or_command_raises(self):
        with self.assertRaises(ValueError):
            get_mcp()
