"""Mock agent for testing Agent-mode serving.

This module defines module-level Agent instances that the worker subprocess
can import via import path (e.g., 'tests.unit.serve.mock_agent:echo_agent').
"""

from motus.agent.base_agent import AgentBase
from motus.models.base import BaseChatClient, ChatCompletion


class MockChatClient(BaseChatClient):
    """A chat client that echoes the last user message."""

    async def create(
        self, model, messages, tools=None, include_reasoning=True, **kwargs
    ):
        last_user = ""
        for msg in reversed(messages):
            if msg.role == "user":
                last_user = msg.content
                break
        return ChatCompletion(
            id="mock-completion",
            model=model,
            content=f"echo: {last_user}",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    async def parse(self, model, messages, response_format, **kwargs):
        raise NotImplementedError


class MockAgent(AgentBase):
    """Minimal concrete AgentBase subclass for testing."""

    async def _run(self, user_prompt=None, **kwargs):
        if user_prompt:
            await self.add_user_message(user_prompt)

        messages = self.get_context()
        completion = await self._client.create(self._model_name, messages)

        await self.add_assistant_message(completion.content)

        return completion.content


class FailingAgent(AgentBase):
    """Agent that always raises an error."""

    async def _run(self, user_prompt=None, **kwargs):
        raise RuntimeError("Intentional agent failure")


# Module-level instances for subprocess import
echo_agent = MockAgent(
    client=MockChatClient(),
    model_name="mock",
    name="echo_agent",
)

failing_agent = FailingAgent(
    client=MockChatClient(),
    model_name="mock",
    name="failing_agent",
)

# Non-agent, non-callable object for testing type rejection
not_an_agent = "I am just a string"
