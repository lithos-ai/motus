"""MCP tools over HTTP with motus `create_auth`.

Adapted from `openai/openai-agents-python`
`examples/mcp/streamablehttp_remote_example/main.py`.

Point `MCP_SERVER_URL` at any MCP server that speaks the Streamable HTTP
transport. `create_auth(MCP_SERVER_URL)` returns a `ConsoleAuth` locally
(OAuth + PKCE with a browser, or a `getpass` bearer-token prompt) and a
`DaprAuth` when deployed (bearer fetched from the Dapr secret store).
"""

import asyncio

from motus.openai_agents import Agent, Runner, gen_trace_id, trace
from motus.openai_agents.mcp import MCPServerStreamableHttp


async def main():
    async with MCPServerStreamableHttp(
        name="DeepWiki MCP Streamable HTTP Server",
        params={
            "url": "https://mcp.linear.app/mcp",
            # Allow more time for remote tool responses.
            "timeout": 15,
            "sse_read_timeout": 300,
        },
        # Retry slow/unstable remote calls a couple of times.
        max_retry_attempts=2,
        retry_backoff_seconds_base=2.0,
        client_session_timeout_seconds=15,
    ) as server:
        agent = Agent(
            name="DeepWiki Assistant",
            instructions="Use the tools to respond to user requests.",
            mcp_servers=[server],
        )

        trace_id = gen_trace_id()
        with trace(workflow_name="DeepWiki Streamable HTTP Example", trace_id=trace_id):
            print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}\n")
            result = await Runner.run(
                agent,
                "For the repository openai/codex, tell me the primary programming language.",
            )
            print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
