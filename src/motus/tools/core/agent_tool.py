import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from .tool import Tool


class _DefaultInput(BaseModel):
    request: str = Field(description="The request or task for the agent")


class AgentTool(Tool):
    """Wraps an AgentBase instance as a Tool.

    Usage::

        research = ReActAgent(client=client, model_name="...", tools=[search])
        orchestrator = ReActAgent(
            tools=[research.as_tool(name="research", description="Deep research")],
        )
    """

    def __init__(
        self,
        agent: Any,
        *,
        name: str | None = None,
        description: str | None = None,
        output_extractor: Callable[[Any], Any] | None = None,
        stateful: bool = False,
        max_steps: int | None = None,
        input_guardrails: list | None = None,
        output_guardrails: list | None = None,
    ):
        from motus.agent.base_agent import AgentBase

        if not isinstance(agent, AgentBase):
            raise TypeError(
                f"AgentTool requires an AgentBase instance, got {type(agent).__name__}"
            )

        self._agent = agent
        self._stateful = stateful
        self._output_extractor = output_extractor
        self._max_steps_override = max_steps

        super().__init__(
            name=name or agent.name,
            description=description or f"Delegate to sub-agent: {agent.name}",
            json_schema=_DefaultInput.model_json_schema(),
            input_guardrails=input_guardrails,
            output_guardrails=output_guardrails,
        )

    async def _invoke(self, **kwargs) -> str:
        """Call the wrapped agent. Invoked by Tool._execute (has @agent_task + guardrails)."""
        request = _DefaultInput.model_validate(kwargs).request

        # Stateless: fork to avoid mutating the template agent
        agent = self._agent if self._stateful else self._agent.fork()

        # Override max_steps if specified
        if self._max_steps_override is not None:
            agent.max_steps = self._max_steps_override

        # Call the agent
        result = await agent(request)

        # Extract output if extractor provided
        if self._output_extractor is not None:
            result = self._output_extractor(result)

        return result if isinstance(result, str) else json.dumps(result)
