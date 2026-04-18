"""
Anthropic chat client implementation.
"""

import json
from typing import Optional, Type

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from .base import (
    BaseChatClient,
    CachePolicy,
    ChatCompletion,
    ChatMessage,
    FunctionCall,
    ReasoningConfig,
    ToolCall,
    ToolDefinition,
)


class AnthropicChatClient(BaseChatClient):
    """
    Anthropic implementation of BaseChatClient.

    Wraps AsyncAnthropic to provide the unified interface.

    Args:
        api_key: API key for authentication (sk-ant-api...)
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        if api_key:
            self._client = AsyncAnthropic(api_key=api_key, **kwargs)
        else:
            # Default: let SDK read from ANTHROPIC_API_KEY env var
            self._client = AsyncAnthropic(**kwargs)

    def _convert_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[Optional[str], list[dict]]:
        """
        Convert ChatMessage list to Anthropic format.

        Returns (system_prompt, messages) tuple since Anthropic
        handles system prompts separately.
        """
        system_prompt = None
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                # Anthropic uses a separate system parameter
                system_prompt = msg.content

            elif msg.role == "user":
                if msg.base64_image:
                    content_parts = [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": msg.base64_image,
                            },
                        },
                    ]
                    if msg.content:
                        content_parts.append({"type": "text", "text": msg.content})
                    anthropic_messages.append(
                        {"role": "user", "content": content_parts}
                    )
                else:
                    anthropic_messages.append({"role": "user", "content": msg.content})

            elif msg.role == "assistant":
                if msg.reasoning_details or msg.tool_calls:
                    # Use content blocks format for thinking and/or tool calls
                    content_blocks = []
                    # Thinking blocks must come first
                    if msg.reasoning_details:
                        for detail in msg.reasoning_details:
                            if detail.get("type") == "thinking":
                                content_blocks.append(detail)
                    if msg.content:
                        content_blocks.append({"type": "text", "text": msg.content})
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            content_blocks.append(
                                {
                                    "type": "tool_use",
                                    "id": tc.id,
                                    "name": tc.function.name,
                                    "input": json.loads(tc.function.arguments),
                                }
                            )
                    anthropic_messages.append(
                        {"role": "assistant", "content": content_blocks}
                    )
                else:
                    anthropic_messages.append(
                        {"role": "assistant", "content": msg.content}
                    )

            elif msg.role == "tool":
                # Anthropic expects tool results as user messages with tool_result blocks
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content,
                            }
                        ],
                    }
                )

        return system_prompt, anthropic_messages

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        """Convert ToolDefinition list to Anthropic format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    def _parse_response(self, response, model: str) -> ChatCompletion:
        """Convert Anthropic response to ChatCompletion."""
        content = None
        tool_calls = None
        reasoning = None
        reasoning_details = None

        for block in response.content:
            if block.type == "thinking":
                reasoning = block.thinking
                if reasoning_details is None:
                    reasoning_details = []
                reasoning_details.append(
                    {
                        "type": "thinking",
                        "thinking": block.thinking,
                        "signature": block.signature,
                    }
                )
            elif block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        function=FunctionCall(
                            name=block.name,
                            arguments=json.dumps(block.input),
                        ),
                    )
                )

        finish_reason = "tool_calls" if response.stop_reason == "tool_use" else "stop"

        return ChatCompletion(
            id=response.id,
            model=model,
            content=content,
            reasoning=reasoning,
            reasoning_details=reasoning_details,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens
                + response.usage.output_tokens,
                "cache_creation_input_tokens": getattr(
                    response.usage, "cache_creation_input_tokens", 0
                )
                or 0,
                "cache_read_input_tokens": getattr(
                    response.usage, "cache_read_input_tokens", 0
                )
                or 0,
            },
        )

    @staticmethod
    def _apply_reasoning(
        request_kwargs: dict,
        model: str,
        reasoning: ReasoningConfig,
    ) -> None:
        """Apply reasoning/thinking configuration to request_kwargs in-place."""
        if "thinking" in request_kwargs:
            return  # already explicitly set by caller

        thinking_param = reasoning.to_anthropic_param(
            model, request_kwargs["max_tokens"]
        )
        if thinking_param:
            request_kwargs["thinking"] = thinking_param
        effort = reasoning.to_anthropic_effort(model)
        if effort:
            request_kwargs.setdefault("output_config", {})["effort"] = effort

    def _apply_cache_policy(
        self,
        request_kwargs: dict,
        cache_policy: CachePolicy,
    ) -> None:
        """Apply cache_control breakpoints to request_kwargs in-place.

        Anthropic prompt caching is prefix-based with up to 4 breakpoints.
        Placement order in the request: system → tools → messages.

        STATIC: cache system prompt + last tool definition.
        AUTO:   STATIC + second-to-last user/tool_result message (conversation
                turn prefix), so prior turns are read from cache each step.
        """
        if cache_policy == CachePolicy.NONE:
            return

        if cache_policy == CachePolicy.AUTO_1H:
            cache_control = {"type": "ephemeral", "ttl": "1h"}
        else:
            cache_control = {"type": "ephemeral"}  # 5-min TTL

        # --- System prompt breakpoint ---
        system = request_kwargs.get("system")
        if system is not None:
            if isinstance(system, str):
                # Convert to content-block format so we can attach cache_control
                request_kwargs["system"] = [
                    {"type": "text", "text": system, "cache_control": cache_control}
                ]
            elif isinstance(system, list) and system:
                # Already content blocks — tag the last one
                system[-1]["cache_control"] = cache_control

        # --- Last tool breakpoint ---
        tools = request_kwargs.get("tools")
        if tools:
            tools[-1]["cache_control"] = cache_control

        if cache_policy == CachePolicy.STATIC:
            return

        # --- AUTO / AUTO_1H: conversation turn prefix breakpoint ---
        # Find the second-to-last user/tool_result message and tag it.
        # This makes the entire prefix up to that point a cache read on
        # the next step, since only the latest turn is new.
        msgs = request_kwargs.get("messages", [])
        if len(msgs) < 2:
            return

        # Walk backwards to find the second-to-last user-role message
        # (user messages include tool_result blocks in Anthropic format)
        user_indices = [i for i, m in enumerate(msgs) if m.get("role") == "user"]
        if len(user_indices) >= 2:
            target_idx = user_indices[-2]
            target_msg = msgs[target_idx]
            content = target_msg.get("content")
            if isinstance(content, str):
                # Convert to block format
                msgs[target_idx] = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": cache_control,
                        }
                    ],
                }
            elif isinstance(content, list) and content:
                content[-1]["cache_control"] = cache_control

    async def create(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: Optional[list[ToolDefinition]] = None,
        reasoning: ReasoningConfig = ReasoningConfig.auto(),
        cache_policy: CachePolicy = CachePolicy.AUTO,
        **kwargs,
    ) -> ChatCompletion:
        """Create a chat completion using Anthropic API.

        Uses streaming internally to avoid 10-minute timeout for long requests.
        """
        system_prompt, anthropic_messages = self._convert_messages(messages)

        request_kwargs = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.pop("max_tokens", 64000),
            **kwargs,
        }

        # Apply reasoning/thinking configuration
        self._apply_reasoning(request_kwargs, model, reasoning)

        if system_prompt:
            request_kwargs["system"] = system_prompt
        if tools:
            request_kwargs["tools"] = self._convert_tools(tools)

        # Apply prompt caching breakpoints
        self._apply_cache_policy(request_kwargs, cache_policy)

        # Use streaming to avoid 10-minute timeout for long requests
        async with self._client.messages.stream(**request_kwargs) as stream:
            response = await stream.get_final_message()
        return self._parse_response(response, model)

    async def parse(
        self,
        model: str,
        messages: list[ChatMessage],
        response_format: Type[BaseModel],
        tools: Optional[list[ToolDefinition]] = None,
        reasoning: ReasoningConfig = ReasoningConfig.auto(),
        cache_policy: CachePolicy = CachePolicy.AUTO,
        **kwargs,
    ) -> ChatCompletion:
        """
        Create a chat completion with structured output parsing.

        Note: Anthropic doesn't have native structured output like OpenAI's
        response_format. We instruct the model to output JSON and parse it.
        """
        # Build schema instruction
        schema = response_format.model_json_schema()
        schema_instruction = (
            f"\n\nYou must respond with valid JSON that matches this schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            f"Respond ONLY with the JSON object, no other text or markdown formatting."
        )

        # Clone messages and append schema instruction to last user message
        messages_copy = list(messages)
        if messages_copy and messages_copy[-1].role == "user":
            last_msg = messages_copy[-1]
            messages_copy[-1] = ChatMessage.user_message(
                (last_msg.content or "") + schema_instruction
            )
        else:
            messages_copy.append(ChatMessage.user_message(schema_instruction))

        system_prompt, anthropic_messages = self._convert_messages(messages_copy)

        request_kwargs = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.pop("max_tokens", 64000),
            **kwargs,
        }

        # Apply reasoning/thinking configuration
        self._apply_reasoning(request_kwargs, model, reasoning)

        if system_prompt:
            request_kwargs["system"] = system_prompt
        if tools:
            request_kwargs["tools"] = self._convert_tools(tools)

        # Apply prompt caching breakpoints
        self._apply_cache_policy(request_kwargs, cache_policy)

        # Use streaming to avoid 10-minute timeout for long requests
        async with self._client.messages.stream(**request_kwargs) as stream:
            response = await stream.get_final_message()

        # Parse the response
        completion = self._parse_response(response, model)

        # Try to parse JSON from content
        if completion.content:
            try:
                content = completion.content
                # Handle markdown code blocks
                if "```json" in content:
                    start = content.find("```json") + 7
                    end = content.find("```", start)
                    content = content[start:end].strip()
                elif "```" in content:
                    start = content.find("```") + 3
                    end = content.find("```", start)
                    content = content[start:end].strip()

                completion.parsed = response_format.model_validate_json(content)
            except Exception:
                # If parsing fails, try raw content
                try:
                    completion.parsed = response_format.model_validate_json(
                        completion.content
                    )
                except Exception:
                    completion.parsed = None

        return completion
