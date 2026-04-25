"""Module-level fake agents for cloud integration tests (need importable paths)."""

from __future__ import annotations


async def echo_agent(message, state):
    from motus.models import ChatMessage

    response = ChatMessage.assistant_message(content=f"echo: {message.content}")
    return response, state + [message, response]
