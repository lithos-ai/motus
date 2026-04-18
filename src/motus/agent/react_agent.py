"""
ReActAgent - A simple ReAct (Reasoning + Acting) agent implementation.
"""

import logging
import time
from typing import Any, Awaitable, Callable, Optional

from pydantic import BaseModel

from motus.agent.tasks import model_serve_task
from motus.models.base import CachePolicy, ReasoningConfig

from .base_agent import AgentBase

logger = logging.getLogger(__name__)


class ReActAgent(AgentBase[str]):
    """
    A ReAct (Reasoning + Acting) agent that iteratively reasons and acts.

    See ``examples/agent.py`` for a minimal example, and ``apps/deep_research/``
    and ``examples/omni/`` for more advanced usage with memory, tools, and sandboxes.

    The agent follows the ReAct pattern:
    1. Receive user input
    2. Generate reasoning and optionally call tools
    3. If tools are called, execute them and continue reasoning
    4. Return final response when no more tool calls are needed

    This provides a simple but effective agentic loop for tool-using LLMs.
    """

    def __init__(
        self,
        client: Any,
        model_name: str,
        name: str | None = None,
        system_prompt: Optional[str] = None,
        tools: Optional[Any] = None,
        response_format: Optional[type[BaseModel]] = None,
        max_steps: int = 20,
        timeout: Optional[float] = None,
        memory_type: str = "basic",
        memory: Optional[Any] = None,
        input_guardrails: Optional[list] = None,
        output_guardrails: Optional[list] = None,
        reasoning: ReasoningConfig = ReasoningConfig.auto(),
        cache_policy: CachePolicy | str = CachePolicy.AUTO,
        step_callback: Optional[
            Callable[[Optional[str], list[dict]], Awaitable[None]]
        ] = None,
    ) -> None:
        """
        Initialize the ReactAgent.

        Args:
            client: The chat client to use for LLM calls
            model_name: The model identifier (e.g., "gpt-4", "claude-3-opus")
            system_prompt: Optional system prompt to set agent behavior
            tools: Optional tools (can be Tools, dict, list, or callable)
            max_steps: Maximum number of reasoning steps (default: 20)
            timeout: Optional timeout in seconds. When exceeded, the agent
                stops gracefully after the current step, preserving the
                execution trace. None means no timeout.
            memory_type: Memory type - "basic" (default) or "compact"
            memory: Optional Memory instance for conversation management
            cache_policy: Prompt caching strategy. "none" (default), "static"
                (cache system prompt + tools), or "auto" (static + conversation
                turn prefix). Only effective with providers that support caching
                (e.g. Anthropic).
            step_callback: Optional async callback(content, tool_calls) called
                after each LLM completion for streaming intermediate state.
        """
        self.step_callback = step_callback
        self._timeout = timeout
        self._cache_policy = CachePolicy(cache_policy)
        self._usage: dict[str, int] = {}
        super().__init__(
            client=client,
            model_name=model_name,
            name=name,
            system_prompt=system_prompt,
            tools=tools,
            response_format=response_format,
            max_steps=max_steps,
            memory_type=memory_type,
            memory=memory,
            input_guardrails=input_guardrails,
            output_guardrails=output_guardrails,
            reasoning=reasoning,
        )

    def _fork_kwargs(self) -> dict:
        kwargs = super()._fork_kwargs()
        kwargs.update(
            {
                "timeout": self._timeout,
                "cache_policy": self._cache_policy,
                "step_callback": self.step_callback,
            }
        )
        return kwargs

    async def _run(
        self,
        user_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Execute the ReAct loop.

        The agent will:
        1. Add the user message to memory (if provided)
        2. Loop until max_steps or completion:
           a. Get conversation context
           b. Call the LLM with available tools
           c. Add assistant response to memory
           d. If tool calls exist, execute them and add results to memory
           e. If no tool calls, return the assistant's response

        Args:
            user_prompt: Optional user message to process
            **kwargs: Additional arguments (unused)

        Returns:
            The final assistant response as a string
        """
        logger.info(f"ReactAgent starting with prompt: {user_prompt}")
        logger.info(f"ReactAgent starting with kwargs: {kwargs}")

        _response_format = kwargs.get("response_format", self._response_format)

        # Add user message if provided
        if user_prompt:
            await self.add_user_message(user_prompt)

        completion = None
        step = 0
        _start_time = time.monotonic()
        while self._max_steps is None or step < self._max_steps:
            # Check timeout before starting a new step
            if self._timeout is not None:
                elapsed = time.monotonic() - _start_time
                if elapsed >= self._timeout:
                    logger.warning(
                        f"ReactAgent timed out after {elapsed:.1f}s "
                        f"(limit: {self._timeout}s, steps completed: {step})"
                    )
                    raise TimeoutError(
                        f"Agent timed out after {elapsed:.1f}s "
                        f"(limit: {self._timeout}s, steps completed: {step})"
                    )

            step += 1
            logger.debug(f"ReactAgent step {step}/{self._max_steps or '∞'}")

            # Get current conversation context
            messages = self.get_context()

            # Call the LLM
            completion = await model_serve_task(
                client=self._client,
                model=self._model_name,
                messages=messages,
                tools=self._tools,
                response_format=_response_format,
                reasoning=self._reasoning,
                cache_policy=self._cache_policy,
            )

            # Accumulate usage
            for k, v in completion.usage.items():
                if isinstance(v, (int, float)):
                    self._usage[k] = self._usage.get(k, 0) + v

            # Log context window utilization
            ctx = self.context_window_usage
            logger.info(
                f"Step {step}: context {ctx['percent']} "
                f"({ctx['estimated_tokens']}/{ctx['threshold']} tokens)"
            )

            # Convert completion to message and add to memory
            assistant_msg = completion.to_message()
            await self.add_message(assistant_msg)

            # Check if there are tool calls to execute
            if assistant_msg.tool_calls:
                # Notify listener of intermediate state (for streaming UIs).
                # Only fires on intermediate steps (with tool calls), not the
                # final response — the caller handles final output separately.
                if self.step_callback:
                    tool_calls = [
                        {"name": tc.function.name, "arguments": tc.function.arguments}
                        for tc in assistant_msg.tool_calls
                    ]
                    await self.step_callback(assistant_msg.content, tool_calls)
                logger.info(
                    f"Executing {len(assistant_msg.tool_calls)} tool call(s): "
                    f"{[tc.function.name for tc in assistant_msg.tool_calls]}"
                )

                # Execute all tool calls (scheduled in parallel by runtime)
                tool_futures = [
                    self.tools[call.function.name](call.function.arguments)
                    for call in assistant_msg.tool_calls
                ]

                # Await results and add to memory
                for tool_future, tool_call in zip(
                    tool_futures, assistant_msg.tool_calls
                ):
                    await self.add_tool_message(
                        content=await tool_future,
                        tool_call_id=tool_call.id,
                        name=tool_call.function.name,
                    )
            else:
                # No tool calls - agent has finished reasoning
                logger.info("ReactAgent completed - no more tool calls")
                if _response_format:
                    return completion.parsed
                return assistant_msg.content or ""

        # Max steps reached
        raise RuntimeError(
            f"ReactAgent reached max steps ({self._max_steps}) without completing"
        )

    @property
    def usage(self) -> dict[str, int]:
        """Accumulated token usage across all LLM calls in this agent run."""
        return dict(self._usage)

    @property
    def cost(self) -> float | None:
        """Accumulated cost in USD across all LLM calls, or None if pricing unavailable."""
        from motus.models.pricing import calculate_cost

        return calculate_cost(self._model_name, self._usage)

    @property
    def context_window_usage(self) -> dict:
        """Current context window utilization.

        Returns dict with:
          - estimated_tokens: current working memory token count
          - threshold: compaction threshold (or model context limit)
          - ratio: utilization as a fraction (0.0 to 1.0+)
          - percent: utilization as a percentage string (e.g., "42%")
        """
        from motus.memory.model_limits import (
            estimate_compaction_threshold,
            get_model_limits,
        )

        estimated = self._memory.estimate_working_memory_tokens()

        # Try to get the compaction threshold (what triggers compaction)
        threshold = None
        if hasattr(self._memory, "_token_threshold") and self._memory._token_threshold:
            threshold = self._memory._token_threshold
        if threshold is None:
            threshold = estimate_compaction_threshold(self._model_name)
        if threshold is None:
            # Fall back to raw context window
            limits = get_model_limits(self._model_name)
            threshold = limits.context_window if limits else 128_000

        ratio = estimated / threshold if threshold > 0 else 0.0
        return {
            "estimated_tokens": estimated,
            "threshold": threshold,
            "ratio": round(ratio, 3),
            "percent": f"{ratio:.0%}",
        }

    def get_execution_trace(self) -> dict:
        """Get execution trace for this agent's run.

        Returns the memory trace enriched with:
        - events: All memory events (messages and compactions)
        - usage: Accumulated token counts
        - model: Model identifier
        - cost_usd: Calculated cost (if pricing available)

        Returns:
            Dictionary with execution trace from memory
        """
        trace = self.memory.get_memory_trace()
        trace["usage"] = self.usage
        trace["model"] = self._model_name
        cost = self.cost
        if cost is not None:
            trace["cost_usd"] = cost
        return trace
