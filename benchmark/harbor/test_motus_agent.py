"""Harbor adapter for the MotusBench-style SWE-bench coding agent.

This keeps the historical import path ``test_motus_agent:MotusHarborAgent``
used by local eval scripts while mirroring
``motus_bench.adapters.coding_agent:CodingAgentBaseline``. For local MTP eval,
set ``SELF_HOSTED_BASE_URL`` to the logging proxy in front of SGLang.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from motus.agent.templates import CodingAgent
from motus.models import AnthropicChatClient, CachePolicy, SelfHostedChatClient
from motus.models.openrouter_client import OpenRouterChatClient
from motus.runtime import get_runtime, is_initialized
from motus.tools.providers.docker import DockerSandbox

load_dotenv()

logger = logging.getLogger("motus_bench_style_harbor_agent")

_DEFAULT_TIMEOUT = 600.0

_anthropic_client: AnthropicChatClient | None = None
_openrouter_client: OpenRouterChatClient | None = None
_self_hosted_client: SelfHostedChatClient | None = None


def _task_name_from_session(session_id: str) -> str:
    return session_id.split("__")[0]


async def _get_client_and_model() -> tuple[Any, str]:
    """Select provider client and model, with self-hosted SGLang first."""
    global _anthropic_client, _openrouter_client, _self_hosted_client

    if os.getenv("SELF_HOSTED_BASE_URL"):
        if _self_hosted_client is None:
            _self_hosted_client = SelfHostedChatClient(
                base_url=os.environ["SELF_HOSTED_BASE_URL"],
                api_key=os.getenv("SELF_HOSTED_API_KEY", "EMPTY"),
                engine=os.getenv("SELF_HOSTED_ENGINE", "auto"),  # type: ignore[arg-type]
                timeout=float(os.getenv("SELF_HOSTED_TIMEOUT", "600")),
            )
        model = os.getenv("SELF_HOSTED_MODEL") or os.getenv("SWE_MODEL")
        if not model:
            model = await _self_hosted_client.resolve_model()
        return _self_hosted_client, model

    model = os.environ.get("SWE_MODEL", "deepseek/deepseek-v4-flash")
    if model.startswith("claude"):
        if _anthropic_client is None:
            _anthropic_client = AnthropicChatClient()
        return _anthropic_client, model

    if _openrouter_client is None:
        _openrouter_client = OpenRouterChatClient()
    return _openrouter_client, model


class MotusHarborAgent(BaseAgent):
    """Harbor BaseAgent that runs Motus's CodingAgent template."""

    @staticmethod
    def name() -> str:
        return "motusagent"

    def version(self) -> str | None:
        return "0.2.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        project_name = environment.session_id.lower().replace(".", "-")
        self._container_name = f"{project_name}-main-1"
        self._task_name = _task_name_from_session(environment.session_id)
        self._session_id = environment.session_id
        logger.info("Container: %s, task: %s", self._container_name, self._task_name)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        client, model = await _get_client_and_model()
        timeout = float(os.environ.get("SWE_TIMEOUT", _DEFAULT_TIMEOUT))
        sandbox = DockerSandbox.connect(self._container_name)
        cache_policy = CachePolicy.AUTO if model.startswith("claude") else CachePolicy.NONE

        agent = CodingAgent(
            client=client,
            model_name=model,
            sandbox=sandbox,
            project_root=None,
            enable_web=True,
            enable_subagents=True,
            enable_plan_mode=True,
            cache_policy=cache_policy,
            timeout=timeout,
        )

        output = ""
        try:
            result = await agent(instruction)
            output = str(result)
        except BaseException:
            logger.warning("Agent interrupted", exc_info=True)
            output = "Agent interrupted"
            raise
        finally:
            usage = getattr(agent, "usage", {}) or {}
            cost = getattr(agent, "cost", None)

            context.n_input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
            context.n_output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
            context.n_cache_tokens = usage.get("cache_read_input_tokens")
            context.cost_usd = cost
            context.metadata = {
                "output": output,
                "usage": usage,
                "cost_usd": cost,
                "model": model,
                "harness": "coding_agent",
                "source": "motus_bench_style",
                "container": self._container_name,
                "task": self._task_name,
                "session_id": self._session_id,
            }

            if is_initialized():
                try:
                    get_runtime().export_trace()
                except Exception:
                    logger.warning("Failed to export trace", exc_info=True)


CodingAgentBaseline = MotusHarborAgent
