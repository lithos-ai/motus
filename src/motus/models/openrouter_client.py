"""
OpenRouter chat client implementation.
"""

import asyncio
import base64
import os
from typing import Optional, Type
from urllib.request import urlopen

import httpx
from pydantic import BaseModel

from motus.runtime.agent_task import agent_task

from .base import ChatCompletion, ChatMessage, ReasoningConfig, ToolDefinition
from .openai_client import OpenAIChatClient


class OpenRouterChatClient(OpenAIChatClient):
    """
    OpenRouter implementation of OpenAIChatClient.

    Uses OpenAI-compatible endpoints with OpenRouter's base URL.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        **kwargs,
    ):
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key
            base_url: OpenRouter API base URL (default: https://openrouter.ai/api/v1)
            http_client: Optional httpx.AsyncClient for custom transport
                        (useful for recording/replay in tests)
            **kwargs: Additional arguments passed to AsyncOpenAI
        """
        if not base_url:
            base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        super().__init__(
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            base_url=base_url,
            http_client=http_client,
            **kwargs,
        )

    def _convert_messages(self, messages: list[ChatMessage]) -> list[dict]:
        """Convert ChatMessage list to OpenRouter format.

        Extends OpenAI format by including reasoning_details on assistant
        messages for multi-turn reasoning round-tripping.
        """
        openai_messages = super()._convert_messages(messages)

        for msg, oai_msg in zip(messages, openai_messages):
            if msg.role == "assistant" and msg.reasoning_details:
                oai_msg["reasoning_details"] = msg.reasoning_details

        return openai_messages

    def _parse_response(self, response, model: str) -> ChatCompletion:
        """Convert OpenRouter response to ChatCompletion.

        Extends OpenAI parsing by extracting reasoning and reasoning_details
        from the response message's model_extra (OpenRouter-specific fields).
        """
        completion = super()._parse_response(response, model)

        message = response.choices[0].message
        reasoning = getattr(message, "reasoning", None)
        reasoning_details = getattr(message, "reasoning_details", None)

        if reasoning:
            completion.reasoning = reasoning
        if reasoning_details:
            completion.reasoning_details = reasoning_details

        return completion

    @staticmethod
    def _build_reasoning_body(reasoning: ReasoningConfig, max_tokens: int) -> dict:
        """Build OpenRouter reasoning extra_body from ReasoningConfig.

        OpenRouter uses a unified reasoning object but different models
        prefer different controls:
        - Effort-based (MiniMax, OpenAI, Grok): {"effort": "high"}
        - Budget-based (Anthropic, Gemini, Qwen): {"max_tokens": N}
        """
        if not reasoning.enabled:
            return {}
        reasoning_param = {}
        if reasoning.effort:
            reasoning_param["effort"] = reasoning.effort
        elif reasoning.budget_tokens:
            reasoning_param["max_tokens"] = reasoning.budget_tokens
        else:
            # Default: use effort "high" as the most portable option
            reasoning_param["effort"] = "high"
        return {"reasoning": reasoning_param}

    async def create(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: Optional[list[ToolDefinition]] = None,
        reasoning: ReasoningConfig = ReasoningConfig.auto(),
        **kwargs,
    ) -> ChatCompletion:
        """Create a chat completion using OpenRouter API."""
        if reasoning.enabled:
            extra_body = kwargs.pop("extra_body", {})
            extra_body.update(
                self._build_reasoning_body(reasoning, kwargs.get("max_tokens", 64000))
            )
            kwargs["extra_body"] = extra_body
        return await super().create(
            model=model, messages=messages, tools=tools, **kwargs
        )

    async def parse(
        self,
        model: str,
        messages: list[ChatMessage],
        response_format: Type[BaseModel],
        tools: Optional[list[ToolDefinition]] = None,
        reasoning: ReasoningConfig = ReasoningConfig.auto(),
        **kwargs,
    ) -> ChatCompletion:
        """Create a chat completion with structured output parsing using OpenRouter API."""
        if reasoning.enabled:
            extra_body = kwargs.pop("extra_body", {})
            extra_body.update(
                self._build_reasoning_body(reasoning, kwargs.get("max_tokens", 64000))
            )
            kwargs["extra_body"] = extra_body
        return await super().parse(
            model=model,
            messages=messages,
            response_format=response_format,
            tools=tools,
            **kwargs,
        )

    @agent_task
    async def generate_image_base64(self, model: str, prompt: str) -> str:
        """
        Generate an image via chat completions and return base64 data.

        OpenRouter returns images via chat/completions with image modalities.
        """
        response = await self._client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
            modalities=["image", "text"],
        )
        b64, url = self._extract_image_payload(response)
        if b64:
            return b64
        if url:
            return await self._download_image_base64(url)
        raise ValueError("No image data found in response.")

    @staticmethod
    def _extract_image_payload(response) -> tuple[str | None, str | None]:
        message = response.choices[0].message
        content = message.content

        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "output_image":
                    b64 = part.get("b64_json")
                    if b64:
                        return b64, None
                if isinstance(part, dict) and part.get("type") == "image":
                    image_url = part.get("image_url") or {}
                    url = image_url.get("url")
                    if url:
                        if url.startswith("data:image/") and ";base64," in url:
                            return OpenRouterChatClient._extract_data_url_base64(
                                url
                            ), None
                        return None, url

        images = getattr(message, "images", None)
        if images is None:
            extra = getattr(message, "model_extra", None) or getattr(
                message, "__pydantic_extra__", None
            )
            if isinstance(extra, dict):
                images = extra.get("images")

        if isinstance(images, list):
            for item in images:
                if not isinstance(item, dict):
                    continue
                image_url = item.get("image_url") or {}
                b64 = image_url.get("b64_json") or item.get("b64_json")
                if b64:
                    return b64, None
                url = image_url.get("url")
                if url:
                    if url.startswith("data:image/") and ";base64," in url:
                        return OpenRouterChatClient._extract_data_url_base64(url), None
                    return None, url

        return None, None

    @staticmethod
    def _extract_data_url_base64(url: str) -> str:
        return url.split("base64,", 1)[1]

    @staticmethod
    async def _download_image_base64(url: str) -> str:
        if url.startswith("data:image/") and ";base64," in url:
            return OpenRouterChatClient._extract_data_url_base64(url)

        def _read() -> bytes:
            with urlopen(url) as response:
                return response.read()

        data = await asyncio.to_thread(_read)
        return base64.b64encode(data).decode("utf-8")
