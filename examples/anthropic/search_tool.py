"""Tool search with deferred loading — demonstrates the tool-search beta.

Defines a weather tool with ``defer_loading=True`` so the model doesn't
receive its full schema upfront. A ``search_available_tools`` meta-tool
lets the model discover and load tools on demand via ``tool_reference``
blocks.

Requires the ``tool-search-tool-2025-10-19`` beta.

Deploy: motus serve start examples.anthropic.search_tool:runner
"""

import asyncio
import json
from typing import Any

from anthropic.lib.tools import BetaAsyncFunctionTool, BetaFunctionToolResultType

from motus.anthropic import ToolRunner, beta_async_tool


@beta_async_tool(defer_loading=True)
async def get_weather(location: str, units: str = "c") -> str:
    """Lookup the weather for a given city.

    Args:
        location: The city and state, e.g. San Francisco, CA
        units: Unit for the output, either 'c' for celsius or 'f' for fahrenheit
    """
    if units == "c":
        return json.dumps(
            {"location": location, "temperature": "20°C", "condition": "Sunny"}
        )
    return json.dumps(
        {"location": location, "temperature": "68°F", "condition": "Sunny"}
    )


def make_tool_searcher(
    tools: list[BetaAsyncFunctionTool[Any]],
) -> BetaAsyncFunctionTool[Any]:
    """Create a meta-tool that searches available tools by keyword."""

    @beta_async_tool
    async def search_available_tools(*, keyword: str) -> BetaFunctionToolResultType:
        """Search for useful tools using a query string.

        Args:
            keyword: A keyword to search for in tool definitions
        """
        results = []
        for tool in tools:
            if keyword.lower() in json.dumps(tool.to_dict()).lower():
                results.append({"type": "tool_reference", "tool_name": tool.name})
        return results

    return search_available_tools


deferred_tools: list[BetaAsyncFunctionTool[Any]] = [get_weather]

runner = ToolRunner(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[*deferred_tools, make_tool_searcher(deferred_tools)],
    system="You are a helpful assistant. Use search_available_tools to find tools.",
    betas=["tool-search-tool-2025-10-19"],
)

# Deploy: motus serve start examples.anthropic.search_tool:runner


async def main():
    from motus.models import ChatMessage

    msg = ChatMessage.user_message("What is the weather in San Francisco?")
    response, _state = await runner.run_turn(msg, [])
    print(response.content)


if __name__ == "__main__":
    asyncio.run(main())
