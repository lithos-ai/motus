# =============================================================================
# Agent-First Entrypoint Example
# =============================================================================
#
# This example demonstrates the agent-first entrypoint for `motus serve` and
# `motus deploy`. Instead of writing a wrapper function with manual state
# management, you define an AgentBase instance at module level and let the
# framework handle everything.
#
# Without OPENROUTER_API_KEY, it falls back to a built-in echo client so you
# can test the serving infrastructure without spending API credits.
#
# -----------------------------------------------------------------------------
# Local testing
# -----------------------------------------------------------------------------
#
#   motus serve start examples.serving.agent_entrypoint:agent --port 8000
#
# Then interact:
#
#   motus serve chat http://localhost:8000 "Hello"
#
# -----------------------------------------------------------------------------
# Cloud deployment
# -----------------------------------------------------------------------------
#
#   motus deploy <project-id> examples.serving.agent_entrypoint:agent
#
# =============================================================================

import os

from motus.agent.base_agent import AgentBase
from motus.models.base import BaseChatClient, ChatCompletion


class EchoChatClient(BaseChatClient):
    """Mock client that echoes the last user message. No API key needed."""

    async def create(
        self, model, messages, tools=None, include_reasoning=True, **kwargs
    ):
        last_user = ""
        for msg in reversed(messages):
            if msg.role == "user":
                last_user = msg.content
                break
        return ChatCompletion(
            id="mock",
            model=model,
            content=f"echo: {last_user}",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    async def parse(self, model, messages, response_format, **kwargs):
        raise NotImplementedError


if os.environ.get("OPENROUTER_API_KEY"):
    from motus.agent import ReActAgent
    from motus.models import OpenRouterChatClient

    agent = ReActAgent(
        client=OpenRouterChatClient(),
        model_name="anthropic/claude-3-5-sonnet",
        system_prompt="You are a helpful assistant.",
    )
else:

    class EchoAgent(AgentBase):
        """Minimal agent that echoes user input."""

        async def _run(self, user_prompt=None, **kwargs):
            if user_prompt:
                await self.add_user_message(user_prompt)
            messages = self.get_context()
            completion = await self._client.create(self._model_name, messages)
            await self.add_assistant_message(completion.content)
            return completion.content

    agent = EchoAgent(
        client=EchoChatClient(),
        model_name="echo",
        system_prompt="You are an echo bot.",
    )
