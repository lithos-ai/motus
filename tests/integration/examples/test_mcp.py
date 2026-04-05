"""
Tests for MCP examples — patterns 1-5 with a mock MCP session.

Patches get_mcp() to return a fake MCPSession with filesystem-like tools,
and patches make_client() to return a mock LLM. This validates the Motus
integration logic (lazy connect, context manager, prefix, blocklist,
guardrails) without needing npx, Docker, or API keys.

Patterns 6 (sandbox) and 7 (remote) are skipped — they need Docker and
Jina API keys respectively.

Usage:
    pytest tests/integration/examples/test_mcp.py -v
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import mcp.types
import pytest

from motus.models.base import ChatCompletion
from motus.tools.core.mcp_tool import MCPSession

# ---------------------------------------------------------------------------
# Fake MCP session
# ---------------------------------------------------------------------------

FAKE_TOOLS = [
    mcp.types.Tool(
        name="list_directory",
        description="List files in a directory",
        inputSchema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    ),
    mcp.types.Tool(
        name="read_file",
        description="Read a file",
        inputSchema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    ),
    mcp.types.Tool(
        name="write_file",
        description="Write a file",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    ),
    mcp.types.Tool(
        name="create_directory",
        description="Create a directory",
        inputSchema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    ),
    mcp.types.Tool(
        name="move_file",
        description="Move a file",
        inputSchema={
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "destination": {"type": "string"},
            },
            "required": ["source", "destination"],
        },
    ),
    mcp.types.Tool(
        name="edit_file",
        description="Edit a file",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    ),
]


def _make_fake_session(connected: bool = True) -> MCPSession:
    """Create an MCPSession with fake tools, bypassing real connection."""
    session = object.__new__(MCPSession)
    # Minimal attrs that MCPSession needs
    session._url = "http://fake"
    session._server_params = None
    session._sandbox = None
    session._sandbox_command = None
    session._sandbox_env = None
    session._sandbox_port = None
    session._sandbox_path = "/mcp"
    session._on_close = None
    session._tools = {}
    session._session = MagicMock() if connected else None
    session._shutdown_event = None
    session._lifecycle_task = None
    session._connect_lock = __import__("asyncio").Lock()
    session._http_client = None
    session._owns_http_client = False
    session._http_kwargs = {}

    if connected:
        session._populate_tools(FAKE_TOOLS)

    # Mock _ensure_connected so lazy patterns work
    async def ensure():
        if session._session is None:
            session._session = MagicMock()
            session._populate_tools(FAKE_TOOLS)

    session._ensure_connected = ensure

    # Mock call_tool on the inner session
    async def call_tool(name, args):
        text = json.dumps({"tool": name, "args": args, "result": "ok"})
        return MagicMock(content=[MagicMock(text=text)])

    if session._session:
        session._session.call_tool = call_tool

    return session


def _make_mock_completion(content="I used the tools successfully.", tool_calls=None):
    return ChatCompletion(
        id="mock",
        model="mock",
        content=content,
        tool_calls=tool_calls,
        usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_llm():
    """Patch OpenAIChatClient.create to return a terminal response (no tool calls)."""
    with patch(
        "motus.models.openai_client.OpenAIChatClient.create",
        new_callable=AsyncMock,
        return_value=_make_mock_completion(),
    ):
        yield


@pytest.fixture(autouse=True)
def _fake_api_keys():
    """Ensure API key env vars exist so client constructors don't fail."""
    import os

    keys = ("OPENROUTER_API_KEY", "OPENAI_API_KEY")
    saved = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ.setdefault(k, "fake-key")
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLazyConnect:
    """Pattern 1: Agent manages MCP lifecycle (lazy connect)."""

    async def test_lazy_session_passed_to_agent(self):
        """Unconnected session is accepted by ReActAgent and tools resolve."""
        from motus.agent import ReActAgent
        from motus.models import OpenAIChatClient

        session = _make_fake_session(connected=False)
        assert session._session is None

        client = OpenAIChatClient(api_key="fake", base_url="https://fake")
        agent = ReActAgent(
            client=client,
            model_name="mock",
            system_prompt="test",
            tools=[session],
        )

        # Agent stores unconnected session; tools resolve after connect
        result = await agent("list files")
        assert result is not None


class TestContextManager:
    """Pattern 2: User manages MCP lifecycle (async with)."""

    async def test_context_manager_connects_session(self):
        """async with connects the session before agent creation."""
        from motus.agent import ReActAgent
        from motus.models import OpenAIChatClient

        session = _make_fake_session(connected=False)

        # Simulate async with
        session._session = MagicMock()
        session._populate_tools(FAKE_TOOLS)

        async def call_tool(name, args):
            text = json.dumps({"tool": name, "args": args})
            return MagicMock(content=[MagicMock(text=text)])

        session._session.call_tool = call_tool

        client = OpenAIChatClient(api_key="fake", base_url="https://fake")
        agent = ReActAgent(
            client=client,
            model_name="mock",
            system_prompt="test",
            tools=[session],
        )

        # Tools should be registered immediately since session is connected
        assert len(agent.tools) >= len(FAKE_TOOLS)
        result = await agent("list files")
        assert result is not None


class TestGuardrails:
    """Pattern 5: Prefix, blocklist, and input guardrails."""

    async def test_prefix_applied(self):
        """tools(session, prefix='fs_') prefixes all tool names."""
        from motus.tools import tools

        session = _make_fake_session(connected=True)
        wrapped = tools(session, prefix="fs_")

        from motus.tools import normalize_tools

        resolved = normalize_tools(wrapped)
        tool_names = list(resolved)
        assert all(name.startswith("fs_") for name in tool_names)
        assert "fs_list_directory" in tool_names

    async def test_blocklist_removes_tools(self):
        """tools(session, blocklist={...}) removes specified tools."""
        from motus.tools import tools

        session = _make_fake_session(connected=True)
        blocked = {"write_file", "create_directory", "move_file", "edit_file"}
        wrapped = tools(session, blocklist=blocked)

        from motus.tools import normalize_tools

        resolved = normalize_tools(wrapped)
        tool_names = set(resolved)
        assert tool_names & blocked == set(), (
            f"Blocked tools still present: {tool_names & blocked}"
        )
        assert "list_directory" in tool_names
        assert "read_file" in tool_names

    async def test_prefix_and_blocklist_combined(self):
        """Prefix and blocklist work together."""
        from motus.tools import normalize_tools, tools

        session = _make_fake_session(connected=True)
        wrapped = tools(
            session,
            prefix="fs_",
            blocklist={"write_file", "create_directory", "move_file", "edit_file"},
        )

        resolved = normalize_tools(wrapped)
        tool_names = list(resolved)
        assert "fs_list_directory" in tool_names
        assert "fs_read_file" in tool_names
        assert "fs_write_file" not in tool_names

    async def test_input_guardrail_blocks(self):
        """Input guardrail raises on forbidden paths."""
        import os

        from motus.tools import normalize_tools, tools

        def block_underscore_files(path: str = ""):
            if path:
                basename = os.path.basename(path)
                if basename.startswith("_"):
                    raise ValueError(f"Access denied: {basename}")

        session = _make_fake_session(connected=True)
        wrapped = tools(session, input_guardrails=[block_underscore_files])

        resolved = normalize_tools(wrapped)
        read_tool = resolved["read_file"]

        # Allowed path — should succeed
        result = read_tool(json.dumps({"path": "README.md"})).af_result()
        assert "error" not in result.lower()

        # Blocked path — guardrail error returned to agent
        result = read_tool(json.dumps({"path": "_secret.py"})).af_result()
        assert "error" in result.lower() or "denied" in result.lower()

    async def test_guardrails_with_agent(self):
        """Full pattern: prefix + blocklist + guardrails through ReActAgent."""
        from motus.agent import ReActAgent
        from motus.models import OpenAIChatClient
        from motus.tools import tools

        session = _make_fake_session(connected=True)
        wrapped = tools(
            session,
            prefix="fs_",
            blocklist={"write_file", "create_directory", "move_file", "edit_file"},
        )

        client = OpenAIChatClient(api_key="fake", base_url="https://fake")
        agent = ReActAgent(
            client=client,
            model_name="mock",
            system_prompt="test",
            tools=wrapped,
        )

        tool_names = list(agent.tools)
        assert "fs_list_directory" in tool_names
        assert "fs_write_file" not in tool_names

        result = await agent("list files")
        assert result is not None
