"""ServableAgent protocol — the serve agent contract.

Any object with a ``run_turn`` method matching this signature can be
served via ``motus serve start``.  Implementations include:

- ``motus.agent.base_agent.AgentBase``
- ``motus.google_adk.agents.llm_agent.Agent``
- ``motus.anthropic.ToolRunner``
- Any user-defined class with a conforming ``run_turn``
"""

from __future__ import annotations

from typing import runtime_checkable

from typing_extensions import Protocol

from motus.models import ChatMessage


@runtime_checkable
class ServableAgent(Protocol):
    """Structural protocol for agents that can be served via serve.

    Implementations must provide an async ``run_turn`` method that accepts
    a user message and conversation state, and returns a response with
    updated state.
    """

    async def run_turn(
        self, message: ChatMessage, state: list[ChatMessage]
    ) -> tuple[ChatMessage, list[ChatMessage]]: ...
