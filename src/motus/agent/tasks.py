import copy
import logging
from typing import Optional, Type, overload

from pydantic import BaseModel

from ..models import (
    BaseChatClient,
    CachePolicy,
    ChatCompletion,
    ChatMessage,
    ReasoningConfig,
    ToolDefinition,
)
from ..runtime.agent_task import agent_task
from ..runtime.types import MODEL_CALL
from ..tools.core.tool import Tools

logger = logging.getLogger("Tasks")


def _ensure_strict_schema(schema: dict) -> dict:
    # In strict tool-parsing mode, all properties must be required and
    # additionalProperties must be false. We preserve optional semantics by
    # converting non-required fields to allow null.
    strict_schema = copy.deepcopy(schema)
    if not isinstance(strict_schema, dict):
        return strict_schema

    properties = strict_schema.get("properties")
    required = set(strict_schema.get("required", []) or [])
    if isinstance(properties, dict):
        for name, prop in properties.items():
            if name in required or not isinstance(prop, dict):
                continue
            prop_type = prop.get("type")
            if isinstance(prop_type, str):
                if prop_type != "null":
                    prop["type"] = [prop_type, "null"]
            elif isinstance(prop_type, list):
                if "null" not in prop_type:
                    prop["type"] = [*prop_type, "null"]
            elif "anyOf" in prop and isinstance(prop["anyOf"], list):
                prop["anyOf"].append({"type": "null"})

        strict_schema["required"] = list(properties.keys())
        strict_schema["additionalProperties"] = False
    return strict_schema


def tools_to_definitions(
    tools: Optional[Tools],
    *,
    strict: bool | None = None,
) -> Optional[list[ToolDefinition]]:
    """Convert Tools dict to ToolDefinition list.

    When strict parsing is enabled, we normalize schemas to satisfy the
    provider's strict requirements.
    """
    if tools is None:
        return None
    definitions: list[ToolDefinition] = []
    for name, tool in tools.items():
        parameters = tool.json_schema
        if strict:
            parameters = _ensure_strict_schema(parameters)
        definitions.append(
            ToolDefinition(
                name=name,
                description=tool.description or "",
                parameters=parameters,
                strict=strict,
            )
        )
    return definitions


@overload
async def model_serve_task(
    client: BaseChatClient,
    model: str,
    messages: list[ChatMessage],
    tools: Tools = None,
    response_format: None = None,
    reasoning: ReasoningConfig = ReasoningConfig.auto(),
    cache_policy: CachePolicy = CachePolicy.AUTO,
) -> ChatCompletion: ...


@overload
async def model_serve_task(
    client: BaseChatClient,
    model: str,
    messages: list[ChatMessage],
    tools: Optional[Tools],
    response_format: Type[BaseModel],
    reasoning: ReasoningConfig = ReasoningConfig.auto(),
    cache_policy: CachePolicy = CachePolicy.AUTO,
) -> ChatCompletion: ...


@agent_task(task_type=MODEL_CALL)
async def model_serve_task(
    client: BaseChatClient,
    model: str,
    messages: list[ChatMessage],
    tools: Optional[Tools] = None,
    response_format: Optional[Type[BaseModel]] = None,
    reasoning: ReasoningConfig = ReasoningConfig.auto(),
    cache_policy: CachePolicy = CachePolicy.AUTO,
) -> ChatCompletion:
    """
    Execute a model completion using the unified chat client interface.

    Args:
        client: BaseChatClient instance (OpenAI, Anthropic, etc.)
        model: Model identifier
        messages: Conversation history as ChatMessage list
        tools: Optional tools dict
        response_format: Optional Pydantic model for structured output
        reasoning: Reasoning/thinking configuration
        cache_policy: Prompt caching strategy (passed through to client)

    Returns:
        ChatCompletion with the model's response
    """
    logger.info("Begin Model serve task")

    tool_definitions = tools_to_definitions(
        tools,
        strict=True if response_format is not None else None,
    )

    # Only pass cache_policy to clients that support it (e.g. Anthropic).
    # Other clients would forward it via **kwargs to their underlying SDK.
    from motus.models.anthropic_client import AnthropicChatClient

    cache_kwargs = {}
    if cache_policy != CachePolicy.NONE and isinstance(client, AnthropicChatClient):
        cache_kwargs["cache_policy"] = cache_policy

    if response_format is not None and issubclass(response_format, BaseModel):
        completion = await client.parse(
            model=model,
            messages=messages,
            tools=tool_definitions,
            response_format=response_format,
            reasoning=reasoning,
            **cache_kwargs,
        )
    else:
        completion = await client.create(
            model=model,
            messages=messages,
            tools=tool_definitions,
            reasoning=reasoning,
            **cache_kwargs,
        )

    logger.info("Finish Model serve task")
    return completion
