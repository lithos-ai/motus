import os

from dotenv import load_dotenv
from harbor.agents.base import AgentContext, BaseAgent, BaseEnvironment

from motus.models import AnthropicChatClient, OpenAIChatClient

from .actus_bot import ActusBot

load_dotenv()


class ActusAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "actusbot"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        """
        Run commands to setup the agent & its tools.
        """

        if os.getenv("OPENROUTER_API_KEY") and os.getenv("OPENROUTER_BASE_URL"):
            client = OpenAIChatClient(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url=os.getenv("OPENROUTER_BASE_URL"),
            )
            model = "anthropic/claude-opus-4-7"
        elif os.getenv("ANTHROPIC_API_KEY"):
            client = AnthropicChatClient(api_key=os.getenv("ANTHROPIC_API_KEY"))
            model = "claude-opus-4-7"
        else:
            raise ValueError("No valid API key found for OpenRouter or Anthropic.")

        self.agent = ActusBot(client=client, model_name=model, environment=environment)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """
        Runs the agent in the environment. Be sure to populate the context with the
        results of the agent execution. Ideally, populate the context as the agent
        executes in case of a timeout or other error.
        Args:
            instruction: The task instruction.
            environment: The environment in which to complete the task.
            context: The context to populate with the results of the agent execution.
        """
        output = await self.agent(instruction)
        context.metadata = {"output": output}
