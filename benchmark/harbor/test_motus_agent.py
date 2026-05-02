import os

from dotenv import load_dotenv
from harbor.agents.base import AgentContext, BaseAgent, BaseEnvironment

from motus.models import AnthropicChatClient, OpenAIChatClient, SelfHostedChatClient

from .motus_agent import MotusAgent

load_dotenv()


class MotusHarborAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "motusagent"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        """
        Run commands to setup the agent & its tools.

        Backend selection (first match wins):
            1. SELF_HOSTED_BASE_URL -> local sglang/vllm via SelfHostedChatClient.
               Optional: SELF_HOSTED_API_KEY (default "EMPTY"),
                         SELF_HOSTED_MODEL  (default: first model from /v1/models),
                         SELF_HOSTED_ENGINE in {"sglang", "vllm", "auto"}.
            2. OPENROUTER_API_KEY + OPENROUTER_BASE_URL -> OpenRouter (claude-opus-4-7).
            3. ANTHROPIC_API_KEY -> direct Anthropic (claude-opus-4-7).
        """

        if os.getenv("SELF_HOSTED_BASE_URL"):
            client = SelfHostedChatClient(
                base_url=os.environ["SELF_HOSTED_BASE_URL"],
                api_key=os.getenv("SELF_HOSTED_API_KEY", "EMPTY"),
                engine=os.getenv("SELF_HOSTED_ENGINE", "auto"),  # type: ignore[arg-type]
            )
            model = os.getenv("SELF_HOSTED_MODEL") or await client.resolve_model()
        elif os.getenv("OPENROUTER_API_KEY") and os.getenv("OPENROUTER_BASE_URL"):
            client = OpenAIChatClient(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url=os.getenv("OPENROUTER_BASE_URL"),
            )
            model = "anthropic/claude-opus-4-7"
        elif os.getenv("ANTHROPIC_API_KEY"):
            client = AnthropicChatClient(api_key=os.getenv("ANTHROPIC_API_KEY"))
            model = "claude-opus-4-7"
        else:
            raise ValueError(
                "No backend configured. Set SELF_HOSTED_BASE_URL (sglang/vllm), "
                "OPENROUTER_API_KEY+OPENROUTER_BASE_URL, or ANTHROPIC_API_KEY."
            )

        self.agent = MotusAgent(client=client, model_name=model, environment=environment)

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
