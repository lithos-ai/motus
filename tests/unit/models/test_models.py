"""Tests for the restored provider client API in ``motus.models``."""

import io
import json
import os
import unittest
from base64 import b64decode

import pytest
from pydantic import BaseModel, Field

from motus.models import (
    AnthropicChatClient,
    ChatCompletion,
    ChatMessage,
    FunctionCall,
    OpenAIChatClient,
    OpenRouterChatClient,
    ToolCall,
    ToolDefinition,
    VolcEngineChatClient,
)


class WeatherResponse(BaseModel):
    """Structured response for weather information."""

    location: str = Field(description="The location name")
    temperature: float = Field(description="Temperature in Fahrenheit")
    conditions: str = Field(description="Weather conditions")


def weather_tool_definition() -> ToolDefinition:
    return ToolDefinition(
        name="get_weather",
        description="Get the weather for a city.",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
            },
            "required": ["location"],
            "additionalProperties": False,
        },
    )


class TestChatMessageHelpers(unittest.TestCase):
    """Tests for the common chat message abstractions."""

    def test_helper_constructors(self):
        system = ChatMessage.system_message("system prompt")
        user = ChatMessage.user_message("hello")
        assistant = ChatMessage.assistant_message("hi")
        tool = ChatMessage.tool_message(
            "weather output", tool_call_id="call_1", name="get_weather"
        )

        self.assertEqual(system.role, "system")
        self.assertEqual(user.role, "user")
        self.assertEqual(assistant.role, "assistant")
        self.assertEqual(tool.role, "tool")
        self.assertEqual(tool.tool_call_id, "call_1")
        self.assertEqual(tool.name, "get_weather")

    def test_from_completion_with_tool_calls(self):
        completion = ChatCompletion(
            id="completion_1",
            model="test-model",
            content="Calling tool",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=FunctionCall(
                        name="get_weather", arguments='{"location":"Miami"}'
                    ),
                )
            ],
        )

        message = ChatMessage.from_completion(completion)

        self.assertEqual(message.role, "assistant")
        self.assertEqual(message.content, "Calling tool")
        self.assertEqual(message.tool_calls[0].function.name, "get_weather")


@pytest.mark.slow
class TestOpenAIChatClient(unittest.IsolatedAsyncioTestCase):
    """Tests for the OpenAI-backed chat client."""

    def setUp(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            self.skipTest("OPENAI_API_KEY not set")
        self.client = OpenAIChatClient(api_key=self.api_key)
        self.model = "gpt-4o-mini"

    async def test_create_basic_completion(self):
        messages = [
            ChatMessage.system_message("You are a helpful assistant."),
            ChatMessage.user_message("Say 'Hello, World!' and nothing else."),
        ]

        response = await self.client.create(
            model=self.model,
            messages=messages,
        )

        self.assertIsInstance(response, ChatCompletion)
        self.assertIsNotNone(response.content)
        self.assertIn("Hello", response.content)

    async def test_create_tool_call(self):
        messages = [
            ChatMessage.user_message(
                "Call get_weather for San Francisco and do not answer directly."
            )
        ]

        response = await self.client.create(
            model=self.model,
            messages=messages,
            tools=[weather_tool_definition()],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )

        self.assertIsInstance(response, ChatCompletion)
        self.assertIsNotNone(response.tool_calls)
        self.assertGreater(len(response.tool_calls), 0)
        tool_call = response.tool_calls[0]
        self.assertIsInstance(tool_call, ToolCall)
        self.assertEqual(tool_call.function.name, "get_weather")

        args = json.loads(tool_call.function.arguments)
        self.assertIn("San Francisco", args["location"])

    async def test_parse_structured_output(self):
        messages = [
            ChatMessage.user_message(
                content="Tell me about the weather in Miami. Return a structured response with location, temperature (in Fahrenheit), and conditions.",
            )
        ]

        response = await self.client.parse(
            model=self.model,
            messages=messages,
            response_format=WeatherResponse,
        )

        self.assertIsInstance(response, ChatCompletion)
        weather_data = response.parsed
        self.assertIsInstance(weather_data, WeatherResponse)
        self.assertIsNotNone(weather_data.location)
        self.assertIsNotNone(weather_data.temperature)
        self.assertIsNotNone(weather_data.conditions)

    async def test_multi_turn_conversation(self):
        messages = [
            ChatMessage.system_message("You are a helpful assistant."),
            ChatMessage.user_message("My name is Alice."),
        ]

        response1 = await self.client.create(model=self.model, messages=messages)
        self.assertIsInstance(response1, ChatCompletion)

        messages.append(response1.to_message())
        messages.append(ChatMessage.user_message("What is my name?"))

        response2 = await self.client.create(model=self.model, messages=messages)
        self.assertIsInstance(response2, ChatCompletion)
        self.assertIsNotNone(response2.content)
        self.assertIn("Alice", response2.content)

    async def test_complete_tool_workflow(self):
        messages = [
            ChatMessage.user_message("Use get_weather to answer for New York."),
        ]

        first = await self.client.create(
            model=self.model,
            messages=messages,
            tools=[weather_tool_definition()],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )

        self.assertIsNotNone(first.tool_calls)
        tool_call = first.tool_calls[0]

        messages.append(first.to_message())
        messages.append(
            ChatMessage.tool_message(
                content="The weather in New York is sunny and 72°F",
                tool_call_id=tool_call.id,
                name=tool_call.function.name,
            )
        )

        final = await self.client.create(
            model=self.model,
            messages=messages,
            tools=[weather_tool_definition()],
        )

        self.assertIsInstance(final, ChatCompletion)
        self.assertIsNotNone(final.content)
        self.assertIn("New York", final.content)


@pytest.mark.slow
class TestVolcEngineChatClient(unittest.IsolatedAsyncioTestCase):
    """Tests for the VolcEngine-backed chat client."""

    def setUp(self):
        self.api_key = os.getenv("VOLCENGINE_API_KEY")
        if not self.api_key:
            self.skipTest("VOLCENGINE_API_KEY not set")
        self.client = VolcEngineChatClient(api_key=self.api_key)
        self.model = os.getenv("VOLCENGINE_MODEL")
        if not self.model:
            self.skipTest("VOLCENGINE_MODEL not set in environment variables")

    async def test_create_basic_completion(self):
        messages = [
            ChatMessage.system_message("You are a helpful assistant."),
            ChatMessage.user_message("Say 'Hello, World!' and nothing else."),
        ]

        response = await self.client.create(
            model=self.model,
            messages=messages,
        )

        self.assertIsInstance(response, ChatCompletion)
        self.assertIsNotNone(response.content)
        self.assertIn("Hello", response.content)

    async def test_create_tool_call(self):
        messages = [
            ChatMessage.user_message(
                "Call get_weather for San Francisco and do not answer directly."
            )
        ]

        response = await self.client.create(
            model=self.model,
            messages=messages,
            tools=[weather_tool_definition()],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )

        self.assertIsInstance(response, ChatCompletion)
        self.assertIsNotNone(response.tool_calls)
        self.assertGreater(len(response.tool_calls), 0)
        tool_call = response.tool_calls[0]
        self.assertIsInstance(tool_call, ToolCall)
        self.assertEqual(tool_call.function.name, "get_weather")

        args = json.loads(tool_call.function.arguments)
        self.assertIn("San Francisco", args["location"])

    async def test_parse_structured_output(self):
        messages = [
            ChatMessage.user_message(
                content="Tell me about the weather in Miami. Return a structured response with location, temperature (in Fahrenheit), and conditions.",
            )
        ]

        response = await self.client.parse(
            model=self.model,
            messages=messages,
            response_format=WeatherResponse,
        )

        self.assertIsInstance(response, ChatCompletion)
        weather_data = response.parsed
        self.assertIsInstance(weather_data, WeatherResponse)
        self.assertIsNotNone(weather_data.location)
        self.assertIsNotNone(weather_data.temperature)
        self.assertIsNotNone(weather_data.conditions)

    async def test_multi_turn_conversation(self):
        messages = [
            ChatMessage.system_message("You are a helpful assistant."),
            ChatMessage.user_message("My name is Alice."),
        ]

        response1 = await self.client.create(model=self.model, messages=messages)
        self.assertIsInstance(response1, ChatCompletion)

        messages.append(response1.to_message())
        messages.append(ChatMessage.user_message("What is my name?"))

        response2 = await self.client.create(model=self.model, messages=messages)
        self.assertIsInstance(response2, ChatCompletion)
        self.assertIsNotNone(response2.content)
        self.assertIn("Alice", response2.content)

    async def test_complete_tool_workflow(self):
        messages = [
            ChatMessage.user_message("Use get_weather to answer for New York."),
        ]

        first = await self.client.create(
            model=self.model,
            messages=messages,
            tools=[weather_tool_definition()],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )

        self.assertIsNotNone(first.tool_calls)
        tool_call = first.tool_calls[0]

        messages.append(first.to_message())
        messages.append(
            ChatMessage.tool_message(
                content="The weather in New York is sunny and 72°F",
                tool_call_id=tool_call.id,
                name=tool_call.function.name,
            )
        )

        final = await self.client.create(
            model=self.model,
            messages=messages,
            tools=[weather_tool_definition()],
        )

        self.assertIsInstance(final, ChatCompletion)
        self.assertIsNotNone(final.content)
        self.assertIn("New York", final.content)


@pytest.mark.slow
class TestAnthropicChatClient(unittest.IsolatedAsyncioTestCase):
    """Tests for the Anthropic-backed chat client."""

    def setUp(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            self.skipTest("ANTHROPIC_API_KEY not set")
        self.client = AnthropicChatClient(api_key=self.api_key)
        self.model = "claude-haiku-4-5"

    async def test_basic_completion(self):
        messages = [ChatMessage.user_message("Say 'Hello, World!' and nothing else.")]

        response = await self.client.create(
            model=self.model,
            messages=messages,
        )

        self.assertIsInstance(response, ChatCompletion)
        self.assertIsNotNone(response.content)
        self.assertIn("Hello", response.content)

    async def test_system_message(self):
        messages = [
            ChatMessage.system_message(
                "You are a pirate. Always respond in pirate speak."
            ),
            ChatMessage.user_message("Hello, how are you?"),
        ]

        response = await self.client.create(model=self.model, messages=messages)

        self.assertIsInstance(response, ChatCompletion)
        self.assertIsNotNone(response.content)
        content_lower = response.content.lower()
        pirate_indicators = ["arr", "ye", "me hearty", "ahoy", "matey"]
        has_pirate_speak = any(
            indicator in content_lower for indicator in pirate_indicators
        )
        self.assertTrue(
            has_pirate_speak,
            f"Expected pirate speak in: {response.content}",
        )


@pytest.mark.slow
class TestOpenRouterChatClient(unittest.IsolatedAsyncioTestCase):
    """Tests for the OpenRouter-backed chat client."""

    def setUp(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            self.skipTest("OPENROUTER_API_KEY not set")
        self.client = OpenRouterChatClient(api_key=self.api_key)
        self.model = "moonshotai/kimi-k2.5"

    async def test_basic_completion(self):
        messages = [
            ChatMessage.user_message(
                content="Respond with exactly: 'OpenRouter test successful'",
            )
        ]

        response = await self.client.create(
            model=self.model,
            messages=messages,
        )

        self.assertIsInstance(response, ChatCompletion)
        self.assertIsNotNone(response.content)
        self.assertIn("OpenRouter", response.content)

    async def test_conversation_context(self):
        messages = [
            ChatMessage.user_message("Remember this number: 42"),
        ]

        response1 = await self.client.create(model=self.model, messages=messages)
        self.assertIsInstance(response1, ChatCompletion)

        messages.append(response1.to_message())
        messages.append(
            ChatMessage.user_message("What number did I ask you to remember?")
        )

        response2 = await self.client.create(model=self.model, messages=messages)
        self.assertIsInstance(response2, ChatCompletion)
        self.assertIsNotNone(response2.content)
        self.assertIn("42", response2.content)


@pytest.mark.slow
class TestGenerateImage(unittest.IsolatedAsyncioTestCase):
    """Tests for OpenAI image generation via the restored client API."""

    def setUp(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_key:
            self.skipTest("OPENAI_API_KEY not set")
        self.client = OpenAIChatClient(api_key=self.openai_key)
        self.model = "gpt-image-1"

    async def _generate_image_bytes(self, prompt: str) -> bytes:
        try:
            image_b64 = await self.client.generate_image_base64(
                model=self.model, prompt=prompt
            )
        except Exception as exc:
            error_text = str(exc)
            if "openrouter.ai" in error_text or "Not Found | OpenRouter" in error_text:
                self.skipTest(
                    "Configured OPENAI-compatible endpoint does not support the OpenAI Images API"
                )
            raise

        return b64decode(image_b64)

    async def test_generate_image_base64(self):
        prompt = "A cute robot playing with a ball"
        image_bytes = await self._generate_image_bytes(prompt)

        png_signature = b"\x89PNG\r\n\x1a\n"
        self.assertGreater(len(image_bytes), 0)
        self.assertTrue(image_bytes.startswith(png_signature))

    async def test_generate_image_returns_decodable_image(self):
        prompt = "A simple red circle"
        image_bytes = await self._generate_image_bytes(prompt)

        self.assertTrue(image_bytes.startswith(b"\x89PNG\r\n\x1a\n"), "Not a valid PNG")

        stream = io.BytesIO(image_bytes)
        stream.read(8)
        width, height = None, None
        while True:
            chunk_len_bytes = stream.read(4)
            if len(chunk_len_bytes) < 4:
                break
            chunk_len = int.from_bytes(chunk_len_bytes, "big")
            chunk_type = stream.read(4).decode("ascii")
            if chunk_type == "IHDR":
                ihdr_data = stream.read(chunk_len)
                width = int.from_bytes(ihdr_data[:4], "big")
                height = int.from_bytes(ihdr_data[4:8], "big")
                break
            stream.read(chunk_len)
            stream.read(4)

        self.assertIsNotNone(width, "IHDR chunk not found in PNG")
        self.assertGreater(width, 0, "Invalid image width")
        self.assertGreater(height, 0, "Invalid image height")


if __name__ == "__main__":
    unittest.main()
