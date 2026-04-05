import asyncio

from motus.tools import FunctionTool, Tool


async def question(prompt: str) -> str:
    """Elicit input from the user.

    Args:
        prompt: The prompt to show to the user.
    """
    return await asyncio.to_thread(input, prompt)


def elicitation_tool() -> Tool:
    return FunctionTool(question)
