from __future__ import annotations

import json
from typing import Any, Optional, Type

from pydantic import BaseModel

from motus.models.base import ChatMessage, ReasoningConfig, ToolDefinition
from motus.models.openai_client import OpenAIChatClient


class MarsOpenAIChatClient(OpenAIChatClient):
    """OpenAI-compatible chat client that attaches Mars request metadata."""

    def __init__(
        self,
        *,
        agent_instance_id: str | None = None,
        agent_class_id: str | None = None,
        is_last_step: bool | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.agent_instance_id = agent_instance_id
        self.agent_class_id = agent_class_id
        self.is_last_step = is_last_step

    def _with_mars_extra_body(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        request_kwargs = dict(kwargs)
        extra_body = dict(request_kwargs.get("extra_body") or {})
        defaults = {
            "agent_instance_id": self.agent_instance_id,
            "agent_class_id": self.agent_class_id,
            "is_last_step": self.is_last_step,
        }
        for key, value in defaults.items():
            if value is not None and key not in extra_body:
                extra_body[key] = value
        if extra_body:
            request_kwargs["extra_body"] = extra_body
            # Also surface the Mars metadata as request headers so it survives an
            # OpenAI-compatible *gateway* in front of the engine (the gateway's typed
            # ChatCompletionRequest drops unknown body fields; it refolds these headers
            # back into the body before forwarding). Direct-to-engine still uses the
            # body. `X-SMG-Agent-Instance-ID` is also the gateway's routing key.
            mars_keys = ("agent_instance_id", "agent_class_id", "is_last_step", "agent_replay")
            mars_meta = {k: extra_body[k] for k in mars_keys if k in extra_body}
            if mars_meta:
                extra_headers = dict(request_kwargs.get("extra_headers") or {})
                extra_headers.setdefault("X-SMG-Mars-Meta", json.dumps(mars_meta))
                aid = mars_meta.get("agent_instance_id")
                if aid is not None:
                    extra_headers.setdefault("X-SMG-Agent-Instance-ID", str(aid))
                request_kwargs["extra_headers"] = extra_headers
        return request_kwargs

    async def create(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: Optional[list[ToolDefinition]] = None,
        reasoning: ReasoningConfig = ReasoningConfig.auto(),
        **kwargs,
    ):
        return await super().create(
            model=model,
            messages=messages,
            tools=tools,
            reasoning=reasoning,
            **self._with_mars_extra_body(kwargs),
        )

    async def create_non_streaming(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: Optional[list[ToolDefinition]] = None,
        reasoning: ReasoningConfig = ReasoningConfig.auto(),
        **kwargs,
    ):
        openai_messages = self._convert_messages(messages)
        request_kwargs = {
            "model": model,
            "messages": openai_messages,
            **kwargs,
        }
        if tools:
            request_kwargs["tools"] = self._convert_tools(tools)

        response = await self._client.chat.completions.create(
            **self._with_mars_extra_body(request_kwargs)
        )
        return self._parse_response(response, model)

    async def parse(
        self,
        model: str,
        messages: list[ChatMessage],
        response_format: Type[BaseModel],
        tools: Optional[list[ToolDefinition]] = None,
        reasoning: ReasoningConfig = ReasoningConfig.auto(),
        **kwargs,
    ):
        return await super().parse(
            model=model,
            messages=messages,
            response_format=response_format,
            tools=tools,
            reasoning=reasoning,
            **self._with_mars_extra_body(kwargs),
        )
