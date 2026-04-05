"""Basic ReAct agent — console, local server, or cloud deployment.

Console:  python examples/agent.py
Serve:    motus serve start examples.agent:server
Deploy:   motus deploy examples.agent:server
"""

import asyncio

from motus.agent import ReActAgent
from motus.models import ChatMessage, OpenAIChatClient
from motus.serve import AgentServer

AGENT = ReActAgent(
    client=OpenAIChatClient(),
    model_name="anthropic/claude-opus-4.6",
    system_prompt="You are a helpful assistant.",
)


async def agent(message, state):
    """Stateless agent function — matches the motus.serve contract."""
    for msg in state:
        await AGENT.add_message(msg)
    result = await AGENT(message.content)
    response = ChatMessage.assistant_message(content=result)
    return response, state + [message, response]


server = AgentServer(agent)


async def session():
    messages = []
    while True:
        user_input = input("User: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting session.")
            break
        response, messages = await agent(
            ChatMessage.user_message(content=user_input), messages
        )
        print(f"Agent: {response.content}")


if __name__ == "__main__":
    asyncio.run(session())
