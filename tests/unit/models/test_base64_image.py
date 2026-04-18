"""Tests for base64_image support in LLM client message conversion.

Each provider has its own format for multimodal content — these tests
verify that a ChatMessage with base64_image is converted correctly
and that plain text messages are left unchanged.
"""

import base64
import unittest
from unittest.mock import patch

from motus.models import ChatMessage

# A tiny 1x1 red PNG for testing (valid image, 68 bytes)
_RED_PIXEL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03"
    b"\x00\x01\x00\x05\xfe\xd4\x00\x00\x00\x00IEND\xaeB`\x82"
)
_B64_IMAGE = base64.b64encode(_RED_PIXEL_PNG).decode()


class TestAnthropicBase64Image(unittest.TestCase):
    """Anthropic user messages with base64_image should use content blocks."""

    def _make_client(self):
        with patch("motus.models.anthropic_client.AsyncAnthropic"):
            from motus.models.anthropic_client import AnthropicChatClient

            return AnthropicChatClient(api_key="fake-key")

    def test_user_message_with_image(self):
        client = self._make_client()
        msgs = [ChatMessage.user_message("What is this?", base64_image=_B64_IMAGE)]

        _, converted = client._convert_messages(msgs)

        self.assertEqual(len(converted), 1)
        msg = converted[0]
        self.assertEqual(msg["role"], "user")
        # Should be a list of content blocks, not a plain string
        self.assertIsInstance(msg["content"], list)

        # Expect image block first, then text
        image_block = msg["content"][0]
        self.assertEqual(image_block["type"], "image")
        self.assertEqual(image_block["source"]["type"], "base64")
        self.assertEqual(image_block["source"]["media_type"], "image/png")
        self.assertEqual(image_block["source"]["data"], _B64_IMAGE)

        text_block = msg["content"][1]
        self.assertEqual(text_block["type"], "text")
        self.assertEqual(text_block["text"], "What is this?")

    def test_user_message_with_image_no_text(self):
        client = self._make_client()
        msgs = [ChatMessage(role="user", base64_image=_B64_IMAGE)]

        _, converted = client._convert_messages(msgs)

        msg = converted[0]
        # Only the image block, no text block appended
        self.assertEqual(len(msg["content"]), 1)
        self.assertEqual(msg["content"][0]["type"], "image")

    def test_user_message_without_image_unchanged(self):
        client = self._make_client()
        msgs = [ChatMessage.user_message("Hello")]

        _, converted = client._convert_messages(msgs)

        msg = converted[0]
        # Plain string, not content-blocks format
        self.assertEqual(msg["content"], "Hello")


class TestOpenAIBase64Image(unittest.TestCase):
    """OpenAI user messages with base64_image should use content array."""

    def _make_client(self):
        from motus.models.openai_client import OpenAIChatClient

        return OpenAIChatClient(api_key="fake-key")

    def test_user_message_with_image(self):
        client = self._make_client()
        msgs = [
            ChatMessage.user_message("Describe this image", base64_image=_B64_IMAGE)
        ]

        converted = client._convert_messages(msgs)

        self.assertEqual(len(converted), 1)
        msg = converted[0]
        self.assertEqual(msg["role"], "user")
        self.assertIsInstance(msg["content"], list)

        # Text part first, then image
        text_part = msg["content"][0]
        self.assertEqual(text_part["type"], "text")
        self.assertEqual(text_part["text"], "Describe this image")

        image_part = msg["content"][1]
        self.assertEqual(image_part["type"], "image_url")
        url = image_part["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/png;base64,"))
        self.assertIn(_B64_IMAGE, url)

    def test_user_message_with_image_no_text(self):
        client = self._make_client()
        msgs = [ChatMessage(role="user", base64_image=_B64_IMAGE)]

        converted = client._convert_messages(msgs)

        msg = converted[0]
        # Image-only: just the image_url part
        self.assertEqual(len(msg["content"]), 1)
        self.assertEqual(msg["content"][0]["type"], "image_url")

    def test_user_message_without_image_unchanged(self):
        client = self._make_client()
        msgs = [ChatMessage.user_message("No image here")]

        converted = client._convert_messages(msgs)

        msg = converted[0]
        self.assertEqual(msg["content"], "No image here")


class TestGeminiBase64Image(unittest.TestCase):
    """Gemini user messages with base64_image should use inline data parts."""

    def _make_client(self):
        with patch("motus.models.gemini_client.genai.Client"):
            from motus.models.gemini_client import GeminiChatClient

            return GeminiChatClient(api_key="fake-key")

    def test_user_message_with_image(self):
        client = self._make_client()
        msgs = [ChatMessage.user_message("What do you see?", base64_image=_B64_IMAGE)]

        _, contents = client._convert_messages(msgs)

        self.assertEqual(len(contents), 1)
        content = contents[0]
        self.assertEqual(content.role, "user")

        # Should have text part + image part
        parts = content.parts
        self.assertEqual(len(parts), 2)
        # First part is text
        self.assertIn("What do you see?", parts[0].text)

    def test_user_message_with_image_no_text(self):
        client = self._make_client()
        msgs = [ChatMessage(role="user", base64_image=_B64_IMAGE)]

        _, contents = client._convert_messages(msgs)

        content = contents[0]
        # Image-only: just the bytes part
        self.assertEqual(len(content.parts), 1)

    def test_user_message_without_image_unchanged(self):
        client = self._make_client()
        msgs = [ChatMessage.user_message("Plain text")]

        _, contents = client._convert_messages(msgs)

        content = contents[0]
        self.assertEqual(len(content.parts), 1)
        self.assertEqual(content.parts[0].text, "Plain text")


if __name__ == "__main__":
    unittest.main()
