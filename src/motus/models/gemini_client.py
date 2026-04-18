"""
Google Gemini chat client implementation.

Uses the Google Gen AI SDK (google-genai).
Documentation: https://googleapis.github.io/python-genai/
"""

import base64
import json
import uuid
from typing import Optional, Type

from google import genai
from google.genai import types
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


class GeminiChatClient(BaseChatClient):
    """
    Google Gemini implementation of BaseChatClient.

    Wraps the Google Gen AI SDK to provide the unified interface.

    Args:
        api_key: Gemini API key. If not provided, reads from GEMINI_API_KEY env var.
        vertexai: If True, use Vertex AI instead of Gemini Developer API.
        project: Google Cloud project ID (required for Vertex AI).
        location: Google Cloud location (required for Vertex AI).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        vertexai: bool = False,
        project: Optional[str] = None,
        location: Optional[str] = None,
        **kwargs,
    ):
        if vertexai:
            self._client = genai.Client(
                vertexai=True,
                project=project,
                location=location or "us-central1",
                **kwargs,
            )
        else:
            self._client = genai.Client(api_key=api_key, **kwargs)

        # Access async client through .aio property
        self._async_client = self._client.aio

    def _convert_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[Optional[str], list[types.Content]]:
        """
        Convert ChatMessage list to Gemini format.

        Returns (system_instruction, contents) tuple since Gemini
        handles system instructions separately.
        """
        system_instruction = None
        contents = []

        for msg in messages:
            if msg.role == "system":
                # Gemini uses separate system instruction
                system_instruction = msg.content

            elif msg.role == "user":
                if msg.base64_image:
                    parts = []
                    if msg.content:
                        parts.append(types.Part.from_text(text=msg.content))
                    parts.append(
                        types.Part.from_bytes(
                            data=base64.b64decode(msg.base64_image),
                            mime_type="image/png",
                        )
                    )
                    contents.append(types.Content(role="user", parts=parts))
                else:
                    contents.append(
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=msg.content or "")],
                        )
                    )

            elif msg.role == "assistant":
                # Gemini uses "model" role instead of "assistant"
                parts = []
                if msg.content:
                    parts.append(types.Part.from_text(text=msg.content))

                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        # Parse arguments from JSON string
                        args = json.loads(tc.function.arguments)
                        parts.append(
                            types.Part.from_function_call(
                                name=tc.function.name,
                                args=args,
                            )
                        )

                contents.append(types.Content(role="model", parts=parts))

            elif msg.role == "tool":
                # Tool results in Gemini are function responses
                # Try to parse content as JSON, fall back to string wrapper
                try:
                    result = json.loads(msg.content)
                except (json.JSONDecodeError, TypeError):
                    result = {"result": msg.content}

                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=msg.name or "unknown",
                                response=result,
                            )
                        ],
                    )
                )

        return system_instruction, contents

    def _convert_tools(
        self, tools: list[ToolDefinition]
    ) -> list[types.FunctionDeclaration]:
        """Convert ToolDefinition list to Gemini FunctionDeclaration format."""
        declarations = []
        for tool in tools:
            declarations.append(
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                )
            )
        return declarations

    def _parse_response(self, response, model: str) -> ChatCompletion:
        """Convert Gemini response to ChatCompletion."""
        content = None
        tool_calls = None

        # Get the first candidate's content
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]

            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    content = part.text
                elif hasattr(part, "function_call") and part.function_call:
                    if tool_calls is None:
                        tool_calls = []
                    fc = part.function_call
                    tool_calls.append(
                        ToolCall(
                            id=f"call_{uuid.uuid4().hex[:24]}",
                            function=FunctionCall(
                                name=fc.name,
                                arguments=json.dumps(dict(fc.args) if fc.args else {}),
                            ),
                        )
                    )

        # Determine finish reason
        finish_reason = "stop"
        if tool_calls:
            finish_reason = "tool_calls"
        elif response.candidates and response.candidates[0].finish_reason:
            # Map Gemini finish reasons to our format
            gemini_reason = str(response.candidates[0].finish_reason)
            if "STOP" in gemini_reason:
                finish_reason = "stop"
            elif "MAX_TOKENS" in gemini_reason:
                finish_reason = "length"
            elif "SAFETY" in gemini_reason:
                finish_reason = "content_filter"

        # Extract usage information
        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "prompt_tokens": getattr(
                    response.usage_metadata, "prompt_token_count", 0
                ),
                "completion_tokens": getattr(
                    response.usage_metadata, "candidates_token_count", 0
                ),
                "total_tokens": getattr(
                    response.usage_metadata, "total_token_count", 0
                ),
            }

        return ChatCompletion(
            id=f"gemini-{uuid.uuid4().hex[:16]}",
            model=model,
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )

    async def create(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: Optional[list[ToolDefinition]] = None,
        reasoning: ReasoningConfig = ReasoningConfig.auto(),
        **kwargs,
    ) -> ChatCompletion:
        """Create a chat completion using Gemini API."""
        system_instruction, contents = self._convert_messages(messages)

        # Build config
        config_kwargs = {}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tools:
            config_kwargs["tools"] = [
                types.Tool(function_declarations=self._convert_tools(tools))
            ]

        # Handle max_tokens -> max_output_tokens
        if "max_tokens" in kwargs:
            config_kwargs["max_output_tokens"] = kwargs.pop("max_tokens")

        # Merge any additional config options
        config_kwargs.update(kwargs)

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        response = await self._async_client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
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
    ) -> ChatCompletion:
        """
        Create a chat completion with structured output parsing.

        Note: Gemini doesn't have native structured output like OpenAI's
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

        system_instruction, contents = self._convert_messages(messages_copy)

        # Build config
        config_kwargs = {}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tools:
            config_kwargs["tools"] = [
                types.Tool(function_declarations=self._convert_tools(tools))
            ]

        # Handle max_tokens -> max_output_tokens
        if "max_tokens" in kwargs:
            config_kwargs["max_output_tokens"] = kwargs.pop("max_tokens")

        config_kwargs.update(kwargs)

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        response = await self._async_client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

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
