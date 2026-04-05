# =============================================================================
# Custom Function Entrypoint
# =============================================================================
#
# Use this when you have custom logic that doesn't use AgentBase.
# You write a function with signature fn(message, state) -> (response, state)
# and manage conversation state yourself.
#
# For AgentBase users, the agent-first entrypoint (see agent_entrypoint.py)
# handles all of this automatically.
#
# -----------------------------------------------------------------------------
# Local testing
# -----------------------------------------------------------------------------
#
#   motus serve start examples.serving.custom_entrypoint:agent --port 8000
#   motus serve chat http://localhost:8000 "Hello"
#
# -----------------------------------------------------------------------------
# Cloud deployment
# -----------------------------------------------------------------------------
#
#   motus deploy <project-id> examples.serving.custom_entrypoint:agent
#
# =============================================================================

from motus.models import ChatMessage


async def agent(message: ChatMessage, state: list[ChatMessage]):
    """Custom entrypoint — echo with turn counter from state."""

    turn = len([m for m in state if m.role == "user"]) + 1
    response = ChatMessage.assistant_message(
        content=f"[turn {turn}] echo: {message.content}"
    )

    return response, state + [message, response]
