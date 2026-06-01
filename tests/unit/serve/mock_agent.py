"""Mock agent for testing Agent-mode serving.

This module defines module-level Agent instances that the worker subprocess
can import via import path (e.g., 'tests.unit.serve.mock_agent:echo_agent').
"""

from motus.agent.base_agent import AgentBase
from motus.models import ChatMessage
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


class StreamingMockAgent(AgentBase):
    """Adds a marker assistant message during _run, then returns it.

    Used by SSE streaming tests as a subagent so we can assert that its
    intermediate messages bubble up to the parent's stream tagged with the
    correct agent_path.
    """

    async def _run(self, user_prompt=None, **kwargs):
        if user_prompt:
            await self.add_user_message(user_prompt)
        await self.add_assistant_message(f"reply from {self.name}")
        return f"reply from {self.name}"


class ParentWithSubagent(AgentBase):
    """Parent that calls each registered tool once, then emits its own marker.

    Bypasses the LLM loop and invokes ``tool._invoke`` directly so the test
    has deterministic ordering and content.
    """

    async def _run(self, user_prompt=None, **kwargs):
        if user_prompt:
            await self.add_user_message(user_prompt)
        if self._tools:
            for tool in self._tools.values():
                await tool._invoke(request=user_prompt or "")
        await self.add_assistant_message("parent done")
        return "parent done"


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

# Subagent + parent fixture for SSE subagent attribution tests.
_subagent_inner = StreamingMockAgent(
    client=MockChatClient(),
    model_name="mock",
    name="inner",
)

parent_with_subagent = ParentWithSubagent(
    client=MockChatClient(),
    model_name="mock",
    name="parent",
    tools=[_subagent_inner.as_tool(name="inner_tool")],
)


async def function_with_internal_agent(message: ChatMessage, state: list[ChatMessage]):
    """Plain serve entrypoint that creates an AgentBase internally."""
    internal = StreamingMockAgent(
        client=MockChatClient(),
        model_name="mock",
        name="internal",
    )
    content = await internal(message.content or "")
    response = ChatMessage.assistant_message(content=content)
    return response, state + [message, response]


# Non-agent, non-callable object for testing type rejection
not_an_agent = "I am just a string"
