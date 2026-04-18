"""
OpenAI chat client implementation.
"""

import os
from typing import Any, Optional, Type

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel

from .base import (
    BaseChatClient,
    ChatCompletion,
    ChatMessage,
    FunctionCall,
    ReasoningConfig,
    ToolCall,
    ToolDefinition,
)


class OpenAIChatClient(BaseChatClient):
    """
    OpenAI implementation of BaseChatClient.

    Wraps AsyncOpenAI to provide the unified interface.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        **kwargs: Any,
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            base_url: Optional custom base URL
            http_client: Optional httpx.AsyncClient for custom transport
                        (useful for recording/replay in tests)
            **kwargs: Additional arguments passed to AsyncOpenAI
        """
        self._client = AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            base_url=base_url,
            http_client=http_client,
            **kwargs,
        )

    @property
    def images(self):
        return self._client.images

    async def generate_image_base64(self, model: str, prompt: str) -> str:
        """
        Generate an image via OpenAI Images API and return base64 data.
        """
        response = await self._client.images.generate(model=model, prompt=prompt)
        image = response.data[0] if response.data else None
        if image and image.b64_json:
            return image.b64_json
        raise ValueError("No image data found in response.")

    def _convert_messages(self, messages: list[ChatMessage]) -> list[dict]:
        """Convert ChatMessage list to OpenAI format."""
        openai_messages = []
        for msg in messages:
            if msg.role == "system":
                openai_messages.append({"role": "system", "content": msg.content})
            elif msg.role == "user":
                if msg.base64_image:
                    content_parts = []
                    if msg.content:
                        content_parts.append({"type": "text", "text": msg.content})
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{msg.base64_image}",
                            },
                        }
                    )
                    openai_messages.append({"role": "user", "content": content_parts})
                else:
                    openai_messages.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                assistant_msg = {"role": "assistant", "content": msg.content}
                if msg.tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                openai_messages.append(assistant_msg)
            elif msg.role == "tool":
                openai_messages.append(
                    {
                        "role": "tool",
                        "content": msg.content,
                        "tool_call_id": msg.tool_call_id,
                    }
                )
        return openai_messages

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        """Convert ToolDefinition list to OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    **({"strict": tool.strict} if tool.strict is not None else {}),
                },
            }
            for tool in tools
        ]

    @staticmethod
    def _extract_usage(response) -> dict:
        """Extract usage dict from OpenAI response, including reasoning and cache tokens."""
        if not response.usage:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        # Extract cached prompt tokens (OpenAI reports these in prompt_tokens_details)
        prompt_details = getattr(response.usage, "prompt_tokens_details", None)
        if prompt_details:
            cached = getattr(prompt_details, "cached_tokens", None)
            if cached:
                # Map to Anthropic-compatible field names for unified cost calculation
                usage["cache_read_input_tokens"] = cached
                # Uncached input = total prompt - cached
                usage["prompt_tokens"] = response.usage.prompt_tokens - cached
                usage["cache_creation_input_tokens"] = 0
        # Extract reasoning tokens
        details = getattr(response.usage, "completion_tokens_details", None)
        if details:
            reasoning_tokens = getattr(details, "reasoning_tokens", None)
            if reasoning_tokens:
                usage["completion_tokens_details"] = {
                    "reasoning_tokens": reasoning_tokens,
                }
        return usage

    def _parse_response(self, response, model: str) -> ChatCompletion:
        """Convert OpenAI response to ChatCompletion."""
        choice = response.choices[0]
        message = choice.message

        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    function=FunctionCall(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    ),
                )
                for tc in message.tool_calls
            ]

        return ChatCompletion(
            id=response.id,
            model=model,
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=self._extract_usage(response),
        )

    async def create(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: Optional[list[ToolDefinition]] = None,
        reasoning: ReasoningConfig = ReasoningConfig.auto(),
        **kwargs,
    ) -> ChatCompletion:
        """Create a chat completion using OpenAI API.

        Uses streaming internally to avoid gateway timeouts on long requests.
        """
        openai_messages = self._convert_messages(messages)

        request_kwargs = {
            "model": model,
            "messages": openai_messages,
            **kwargs,
        }

        if tools:
            request_kwargs["tools"] = self._convert_tools(tools)

        # Stream internally to avoid gateway timeouts on long requests.
        # Bypass beta.chat.completions.stream() which requires strict tools.
        from openai._types import NOT_GIVEN
        from openai.lib.streaming.chat import AsyncChatCompletionStreamManager

        request_kwargs["stream"] = True
        request_kwargs["stream_options"] = {"include_usage": True}
        api_request = self._client.chat.completions.create(**request_kwargs)
        mgr = AsyncChatCompletionStreamManager(
            api_request, response_format=NOT_GIVEN, input_tools=NOT_GIVEN
        )
        async with mgr as stream:
            response = await stream.get_final_completion()

        return self._parse_response(response, model)

    async def parse(
        self,
        model: str,
        messages: list[ChatMessage],
        response_format: Type[BaseModel],
        tools: Optional[list[ToolDefinition]] = None,
        reasoning: ReasoningConfig = ReasoningConfig.auto(),
        **kwargs,
    ) -> ChatCompletion:
        """Create a chat completion with structured output parsing."""
        openai_messages = self._convert_messages(messages)

        request_kwargs = {
            "model": model,
            "messages": openai_messages,
            "response_format": response_format,
            **kwargs,
        }

        if tools:
            request_kwargs["tools"] = self._convert_tools(tools)

        response = await self._client.beta.chat.completions.parse(**request_kwargs)

        choice = response.choices[0]
        message = choice.message

        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    function=FunctionCall(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    ),
                )
                for tc in message.tool_calls
            ]

        return ChatCompletion(
            id=response.id,
            model=model,
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            parsed=getattr(message, "parsed", None),
            usage=self._extract_usage(response),
        )
