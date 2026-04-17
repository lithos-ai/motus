"""Basic ReAct agent — console, local server, or cloud deployment.

Console:  python agent.py
Serve:    motus serve start agent:agent
Deploy:   motus deploy agent:agent
"""

import asyncio

from motus.agent import ReActAgent
from motus.models import AnthropicChatClient, ChatMessage

agent = ReActAgent(
    client=AnthropicChatClient(),
    model_name="claude-haiku-4-5",
    system_prompt="You are a helpful assistant.",
)


async def session():
    messages = []
    while True:
        user_input = input("User: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting session.")
            break
        response, messages = await agent.run_turn(
            ChatMessage.user_message(content=user_input), messages
        )
        print(f"Agent: {response.content}")


if __name__ == "__main__":
    asyncio.run(session())
