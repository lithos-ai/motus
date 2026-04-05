import asyncio
import json

from motus.anthropic import ToolRunner, beta_async_tool


@beta_async_tool
async def get_weather(location: str, unit: str = "fahrenheit") -> str:
    """Get the current weather in a given location.

    Args:
        location: The city and state, e.g. San Francisco, CA
        unit: Temperature unit, either 'celsius' or 'fahrenheit'
    """
    return json.dumps({"temperature": "20°C", "condition": "Sunny"})


@beta_async_tool
async def calculate_sum(a: int, b: int) -> str:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number
    """
    return str(a + b)


runner = ToolRunner(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[get_weather, calculate_sum],
    system="You are a helpful assistant.",
)

# Deploy: motus serve start examples.anthropic.tools_runner:runner


async def main():
    from motus.models import ChatMessage

    msg = ChatMessage.user_message(
        "What's the weather like in Paris? Also, what's 15 + 27?"
    )
    response, state = await runner.run_turn(msg, [])
    print(response.content)


if __name__ == "__main__":
    asyncio.run(main())
