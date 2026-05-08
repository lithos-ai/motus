"""
Simplest flat ReAct agent for the harbor benchmark (SWE-bench / terminal-bench).

Same backend selection and tool surface as ``test_motus_agent.MotusHarborAgent``,
but instantiates ``ReActAgent`` directly — no MotusAgent wrapper, no
``launch_sub_agent`` tool, no orchestrator-vs-executor split. The system prompt
is the same prompt motus's executor sub-agent uses (``EXECUTOR_PROMPT``), which
is already written for a single flat agent driving ``sandbox_sh`` end-to-end.
"""
import os

from dotenv import load_dotenv
from harbor.agents.base import AgentContext, BaseAgent, BaseEnvironment

from motus.agent import ReActAgent
from motus.models import AnthropicChatClient, OpenAIChatClient, SelfHostedChatClient

from motus_agent.agent_prompts import EXECUTOR_PROMPT
from motus_agent.tools import build_harbor_tools

load_dotenv()


class SimpleReactHarborAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "simple_react_agent"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        """
        Backend selection (first match wins) — same precedence as MotusHarborAgent:
            1. SELF_HOSTED_BASE_URL -> local sglang/vllm via SelfHostedChatClient.
               Optional: SELF_HOSTED_API_KEY (default "EMPTY"),
                         SELF_HOSTED_MODEL  (default: first model from /v1/models),
                         SELF_HOSTED_ENGINE in {"sglang", "vllm", "auto"}.
            2. OPENROUTER_API_KEY + OPENROUTER_BASE_URL -> OpenRouter.
            3. ANTHROPIC_API_KEY -> direct Anthropic.
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

        self.agent = ReActAgent(
            client=client,
            model_name=model,
            system_prompt=EXECUTOR_PROMPT,
            tools=build_harbor_tools(environment),
            max_steps=500,
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        output = await self.agent(instruction)
        context.metadata = {"output": output}
