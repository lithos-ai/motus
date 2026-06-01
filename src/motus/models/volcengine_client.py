"""
VolcEngine chat client implementation.
"""

import os
from typing import Any, Optional, Type

from pydantic import BaseModel

try:
    from volcenginesdkarkruntime import AsyncArk
except ImportError:
    AsyncArk = None

from .base import (
    BaseChatClient,
    ChatCompletion,
    ChatMessage,
    FunctionCall,
    ToolCall,
    ToolDefinition,
)


class VolcEngineChatClient(BaseChatClient):
    """
    VolcEngine implementation of BaseChatClient.

    Wraps Ark to provide the unified interface.

    Requires the ``volcengine`` optional dependency::

        pip install lithosai-motus[volcengine]

    Args:
        api_key: VolcEngine API key. If not provided, will look for
                 VOLCENGINE_API_KEY in environment variables.
        base_url: VolcEngine API base URL.
                  (default: https://ark.cn-beijing.volces.com/api/v3)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        if AsyncArk is None:
            raise ImportError(
                "volcenginesdkarkruntime is not installed. "
                "Install it with: pip install lithosai-motus[volcengine]"
            )
        if not base_url:
            base_url = os.getenv(
                "VOLCENGINE_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"
            )
        self._client = AsyncArk(
            api_key=api_key or os.getenv("VOLCENGINE_API_KEY", ""),
            base_url=base_url,
            **kwargs,
        )

    def _convert_messages(self, messages: list[ChatMessage]) -> list[dict]:
        """Convert ChatMessage list to Ark (OpenAI-compatible) format."""
        ark_messages = []
        for msg in messages:
            if msg.role == "system":
                ark_messages.append({"role": "system", "content": msg.content})

            elif msg.role == "user":
                ark_messages.append({"role": "user", "content": msg.content})

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
                ark_messages.append(assistant_msg)

            elif msg.role == "tool":
                ark_messages.append(
                    {
                        "role": "tool",
                        "content": msg.content,
                        "tool_call_id": msg.tool_call_id,
                    }
                )

        return ark_messages

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    @staticmethod
    def _extract_usage(response) -> dict:
        usage = getattr(response, "usage", response)
        if not usage:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        result = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        if prompt_details:
            cached = getattr(prompt_details, "cached_tokens", None)
            if cached:
                result["cache_read_input_tokens"] = cached
                result["prompt_tokens"] = usage.prompt_tokens - cached
                result["cache_creation_input_tokens"] = 0
        details = getattr(usage, "completion_tokens_details", None)
        if details:
            reasoning_tokens = getattr(details, "reasoning_tokens", None)
            if reasoning_tokens:
                result["completion_tokens_details"] = {
                    "reasoning_tokens": reasoning_tokens,
                }
        return result

    def _build_tool_calls_from_stream(
        self, tool_calls_map: dict[int, dict]
    ) -> Optional[list[ToolCall]]:
        """Build ToolCall list from aggregated streaming tool_calls_map."""
        if not tool_calls_map:
            return None
        return [
            ToolCall(
                id=tool_call["id"],
                function=FunctionCall(
                    name=tool_call["name"],
                    arguments=tool_call["arguments"],
                ),
            )
            for _, tool_call in sorted(tool_calls_map.items())
        ]

    def _build_tool_calls_from_response(self, tool_calls) -> Optional[list[ToolCall]]:
        """Build ToolCall list from a non-streaming response message."""
        if not tool_calls:
            return None
        return [
            ToolCall(
                id=tool_call.id,
                function=FunctionCall(
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                ),
            )
            for tool_call in tool_calls
        ]

    async def create(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: Optional[list[ToolDefinition]] = None,
        **kwargs,
    ) -> ChatCompletion:
        """Create a chat completion using Volcengine Ark API (manual streaming)."""
        ark_messages = self._convert_messages(messages)

        # Pop stream/stream_options from kwargs to prevent callers from
        # accidentally overriding them and breaking the streaming aggregation.
        kwargs.pop("stream", None)
        kwargs.pop("stream_options", None)

        request_kwargs = {
            **kwargs,
            "model": model,
            "messages": ark_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if tools:
            request_kwargs["tools"] = self._convert_tools(tools)

        content_parts: list[str] = []
        tool_calls_map: dict[int, dict] = {}
        finish_reason = "stop"
        response_id = ""
        usage_data = None

        stream = await self._client.chat.completions.create(**request_kwargs)
        async for chunk in stream:
            if chunk.id:
                response_id = chunk.id

            if getattr(chunk, "usage", None):
                usage_data = chunk.usage

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason

            delta = choice.delta
            if delta.content:
                content_parts.append(delta.content)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc_delta.id:
                        tool_calls_map[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_map[idx]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls_map[idx]["arguments"] += (
                                tc_delta.function.arguments
                            )

        return ChatCompletion(
            id=response_id,
            model=model,
            content="".join(content_parts) or None,
            tool_calls=self._build_tool_calls_from_stream(tool_calls_map),
            finish_reason=finish_reason or "stop",
            parsed=None,
            usage=self._extract_usage(usage_data) if usage_data else {},
        )

    async def parse(
        self,
        model: str,
        messages: list[ChatMessage],
        response_format: Type[BaseModel],
        tools: Optional[list[ToolDefinition]] = None,
        **kwargs,
    ) -> ChatCompletion:
        """Create a chat completion with structured output using Volcengine Ark API.

        Uses json_schema response format with manual Pydantic parsing,
        since Ark does not provide a native structured output parse endpoint.
        """
        ark_messages = self._convert_messages(messages)

        request_kwargs = {
            "model": model,
            "messages": ark_messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema": response_format.model_json_schema(),
                    "strict": True,
                },
            },
            **kwargs,
        }

        if tools:
            request_kwargs["tools"] = self._convert_tools(tools)

        response = await self._client.chat.completions.create(**request_kwargs)

        choice = response.choices[0]
        message = choice.message

        parsed = None
        if message.content:
            try:
                parsed = response_format.model_validate_json(message.content)
            except Exception:
                parsed = None

        return ChatCompletion(
            id=response.id,
            model=model,
            content=message.content,
            tool_calls=self._build_tool_calls_from_response(message.tool_calls),
            finish_reason=choice.finish_reason or "stop",
            parsed=parsed,
            usage=self._extract_usage(response),
        )
