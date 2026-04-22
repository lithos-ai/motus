"""MCP tools over HTTP with motus `session()` + Anthropic's `async_mcp_tool`.

Adapted from `anthropics/anthropic-sdk-python` `examples/mcp_tool_runner.py`,
substituting motus's `session()` helper for the upstream stdio client. The
MCP connection is held by this process, so `create_auth` (invoked inside
`session()`) plugs straight into the Streamable HTTP transport —
`ConsoleAuth` locally, `DaprAuth` when a daprd sidecar is present.

Non-deployable: `async_mcp_tool` binds each tool to a live `ClientSession`,
whose context-manager lifecycle doesn't match `motus serve`'s stateless
per-turn model. See `examples/mcp_tools.py` for motus-native MCP patterns
that *are* deployable.

Requires: ``pip install anthropic[mcp]``.
"""

import asyncio
import os

import rich
from anthropic import AsyncAnthropic
from anthropic.lib.tools.mcp import async_mcp_tool
from mcp import ClientSession

from motus.mcp.client.streamable_http import streamable_http_client

MCP_SERVER = os.getenv("MCP_SERVER")

client = AsyncAnthropic()


async def main() -> None:
    async with streamable_http_client(MCP_SERVER) as (read, write, _):
        async with ClientSession(read, write) as mcp_client:
            await mcp_client.initialize()

            # List available tools from the MCP server and convert them
            tools_result = await mcp_client.list_tools()
            tools = [async_mcp_tool(t, mcp_client) for t in tools_result.tools]

            print(f"Connected to MCP server with {len(tools)} tools:")
            for tool in tools:
                print(f"  - {tool.name}")
            print()

            # Run a conversation with tool_runner()
            runner = client.beta.messages.tool_runner(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1024,
                tools=tools,
                messages=[{"role": "user", "content": "List the files in /tmp"}],
            )
            async for message in runner:
                rich.print(message)


asyncio.run(main())
