"""
Regression tests for tool crash, Anthropic tool-result image loss,
and Gemini empty-response crash fixes.
"""

import unittest
from types import SimpleNamespace

from motus.models.anthropic_client import AnthropicChatClient
from motus.models.base import ChatMessage
from motus.models.gemini_client import GeminiChatClient
from motus.runtime import init, is_initialized, resolve, shutdown
from motus.tools.core.function_tool import FunctionTool


class TestFunctionToolMalformedJSON(unittest.TestCase):
    """Bug 1: FunctionTool.__call__ crashes on malformed JSON instead of
    returning a graceful error the LLM can learn from.

    Before the fix, malformed JSON would raise an unhandled exception that
    propagates through the task graph and kills the agent loop. The base
    Tool.__call__ has this handled, but FunctionTool overrides it without
    the safety net.
    """

    def setup_method(self, method=None):
        if is_initialized():
            shutdown()

    def teardown_method(self, method=None):
        if is_initialized():
            shutdown()

    def test_malformed_json_returns_error_not_crash(self):
        """Sending garbage JSON should return an error string, not crash."""
        init()

        async def greet(name: str) -> str:
            """Say hello."""
            return f"Hello, {name}!"

        tool = FunctionTool(greet)

        # This used to crash with json.JSONDecodeError
        future = tool("{not valid json at all")
        result = resolve(future, timeout=5)

        # Should be a JSON error message, not a crash
        assert isinstance(result, str)
        assert "Invalid tool arguments" in result or "error" in result.lower()

    def test_invalid_schema_returns_error_not_crash(self):
        """Sending valid JSON that doesn't match the schema should not crash."""
        init()

        async def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        tool = FunctionTool(add)

        # Valid JSON but wrong types — should return error, not crash
        future = tool('{"a": "not_a_number", "b": "also_not"}')
        result = resolve(future, timeout=5)

        # Should complete (possibly with coercion or error), NOT crash
        assert isinstance(result, str)

    def test_empty_string_args_handled(self):
        """Empty string arguments should not crash."""
        init()

        async def no_args() -> str:
            """No arguments needed."""
            return "ok"

        tool = FunctionTool(no_args)

        future = tool("")
        result = resolve(future, timeout=5)
        assert isinstance(result, str)

    def test_none_like_args_handled(self):
        """Null-like args from LLMs should be handled gracefully."""
        init()

        async def no_args() -> str:
            """No arguments needed."""
            return "ok"

        tool = FunctionTool(no_args)

        future = tool("null")
        result = resolve(future, timeout=5)
        assert isinstance(result, str)


class TestAnthropicToolResultImage(unittest.TestCase):
    """Bug 2: Anthropic client silently drops base64_image on tool messages.

    When a tool returns an image (e.g., a browser screenshot), the
    base64_image field was completely ignored during message conversion,
    losing critical visual context.
    """

    def test_tool_message_without_image_unchanged(self):
        """Tool messages without images should still work as before."""
        client = AnthropicChatClient.__new__(AnthropicChatClient)
        messages = [
            ChatMessage.tool_message(
                content="result text",
                tool_call_id="tool_123",
                name="my_tool",
            )
        ]
        _, converted = client._convert_messages(messages)

        assert len(converted) == 1
        msg = converted[0]
        assert msg["role"] == "user"
        assert msg["content"][0]["type"] == "tool_result"
        assert msg["content"][0]["content"] == "result text"

    def test_tool_message_with_image_includes_both(self):
        """Tool messages with base64_image should include the image block."""
        client = AnthropicChatClient.__new__(AnthropicChatClient)
        messages = [
            ChatMessage.tool_message(
                content="Screenshot captured",
                tool_call_id="tool_456",
                name="browser_screenshot",
                base64_image="iVBORw0KGgoAAAANSUhEUg==",
            )
        ]
        _, converted = client._convert_messages(messages)

        assert len(converted) == 1
        msg = converted[0]
        assert msg["role"] == "user"

        tool_result = msg["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "tool_456"

        # Content should be a list with text and image
        content = tool_result["content"]
        assert isinstance(content, list), (
            f"Expected list for tool result with image, got {type(content)}"
        )
        assert len(content) == 2

        text_part = content[0]
        assert text_part["type"] == "text"
        assert text_part["text"] == "Screenshot captured"

        image_part = content[1]
        assert image_part["type"] == "image"
        assert image_part["source"]["type"] == "base64"
        assert image_part["source"]["data"] == "iVBORw0KGgoAAAANSUhEUg=="


class TestGeminiEmptyResponse(unittest.TestCase):
    """Bug 3: Gemini client crashes with AttributeError when the response
    has no content (e.g., safety filter triggered).

    candidate.content can be None, but the code accessed
    candidate.content.parts without checking.
    """

    def test_safety_filtered_response_no_crash(self):
        """A response with candidate.content=None should not crash."""
        client = GeminiChatClient.__new__(GeminiChatClient)

        # Simulate a safety-filtered response where content is None
        candidate = SimpleNamespace(
            content=None,
            finish_reason="SAFETY",
        )
        response = SimpleNamespace(
            candidates=[candidate],
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                candidates_token_count=0,
                total_token_count=10,
            ),
        )

        # This used to crash with: AttributeError: 'NoneType' object has no attribute 'parts'
        completion = client._parse_response(response, "gemini-2.5-flash")

        assert completion.content is None
        assert completion.tool_calls is None
        assert completion.finish_reason == "content_filter"

    def test_empty_parts_no_crash(self):
        """A response with candidate.content.parts=None should not crash."""
        client = GeminiChatClient.__new__(GeminiChatClient)

        candidate = SimpleNamespace(
            content=SimpleNamespace(parts=None),
            finish_reason="STOP",
        )
        response = SimpleNamespace(
            candidates=[candidate],
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                candidates_token_count=0,
                total_token_count=10,
            ),
        )

        completion = client._parse_response(response, "gemini-2.5-flash")
        assert completion.content is None
        assert completion.finish_reason == "stop"

    def test_empty_candidates_no_crash(self):
        """A response with no candidates should not crash."""
        client = GeminiChatClient.__new__(GeminiChatClient)

        response = SimpleNamespace(
            candidates=[],
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                candidates_token_count=0,
                total_token_count=10,
            ),
        )

        completion = client._parse_response(response, "gemini-2.5-flash")
        assert completion.content is None
        assert completion.tool_calls is None

    def test_normal_text_response_still_works(self):
        """Normal responses should still parse correctly after the fix."""
        client = GeminiChatClient.__new__(GeminiChatClient)

        text_part = SimpleNamespace(
            text="Hello, world!",
            function_call=None,
        )
        candidate = SimpleNamespace(
            content=SimpleNamespace(parts=[text_part]),
            finish_reason="STOP",
        )
        response = SimpleNamespace(
            candidates=[candidate],
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                candidates_token_count=5,
                total_token_count=15,
            ),
        )

        completion = client._parse_response(response, "gemini-2.5-flash")
        assert completion.content == "Hello, world!"
        assert completion.finish_reason == "stop"
