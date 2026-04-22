import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from motus.tools.providers.exa.tool_provider import (
    ExaSearch,
    ExaSearchResult,
    ExaSearchTool,
    _build_contents,
    _derive_description,
    _parse_result,
)


def _mock_sdk_response(results: list[dict]) -> SimpleNamespace:
    """Build a minimal fake exa-py SearchResponse."""
    items = [SimpleNamespace(**r) for r in results]
    # SimpleNamespace lacks model_dump; the provider falls back to vars().
    return SimpleNamespace(results=items)


class TestExaParsing(unittest.TestCase):
    def test_build_contents_defaults_to_snippet_highlights(self):
        self.assertEqual(_build_contents("snippet"), {"highlights": True})

    def test_build_contents_full_returns_all_three(self):
        contents = _build_contents("full")
        self.assertIn("highlights", contents)
        self.assertIn("text", contents)
        self.assertIn("summary", contents)

    def test_build_contents_text_only(self):
        self.assertEqual(_build_contents("text"), {"text": True})

    def test_build_contents_summary_only(self):
        self.assertEqual(_build_contents("summary"), {"summary": {}})

    def test_description_prefers_summary(self):
        desc = _derive_description(
            highlights=["h1"], text="long text", summary="a summary"
        )
        self.assertEqual(desc, "a summary")

    def test_description_falls_back_to_highlights(self):
        desc = _derive_description(
            highlights=["first", "second"], text="full text", summary=None
        )
        self.assertEqual(desc, "first … second")

    def test_description_falls_back_to_text_snippet(self):
        long_text = "x" * 1000
        desc = _derive_description(highlights=[], text=long_text, summary=None)
        self.assertEqual(len(desc), 500)

    def test_description_empty_when_no_content(self):
        self.assertEqual(_derive_description([], None, None), "")

    def test_parse_result_with_all_fields(self):
        raw = {
            "title": "Hello",
            "url": "https://example.com",
            "publishedDate": "2025-01-01",
            "author": "Ada",
            "score": 0.9,
            "highlights": ["snippet a", "snippet b"],
            "text": "full text",
            "summary": "short summary",
        }
        result = _parse_result(raw)
        self.assertIsInstance(result, ExaSearchResult)
        self.assertEqual(result.title, "Hello")
        self.assertEqual(result.description, "short summary")
        self.assertEqual(result.highlights, ["snippet a", "snippet b"])
        self.assertEqual(result.published_date, "2025-01-01")

    def test_parse_result_handles_missing_content(self):
        raw = {"title": "T", "url": "https://example.com"}
        result = _parse_result(raw)
        self.assertEqual(result.description, "")
        self.assertEqual(result.highlights, [])
        self.assertIsNone(result.text)
        self.assertIsNone(result.summary)

    def test_parse_result_handles_structured_summary(self):
        raw = {
            "title": "T",
            "url": "https://example.com",
            "summary": {"text": "nested summary"},
        }
        self.assertEqual(_parse_result(raw).summary, "nested summary")


class TestExaSearch(unittest.IsolatedAsyncioTestCase):
    async def test_search_invokes_sdk_with_expected_args(self):
        mock_client = MagicMock()
        mock_client.search_and_contents.return_value = _mock_sdk_response(
            [
                {
                    "title": "Paper",
                    "url": "https://arxiv.org/abs/1",
                    "highlights": ["key finding"],
                    "publishedDate": "2024-06-01",
                }
            ]
        )

        search = ExaSearch("key-abc", client=mock_client)
        results = await search(
            "mixture of experts",
            k=3,
            type="neural",
            category="research paper",
            include_domains=["arxiv.org"],
            content_mode="snippet",
        )

        mock_client.search_and_contents.assert_called_once()
        call_args, call_kwargs = mock_client.search_and_contents.call_args
        self.assertEqual(call_args[0], "mixture of experts")
        self.assertEqual(call_kwargs["num_results"], 3)
        self.assertEqual(call_kwargs["type"], "neural")
        self.assertEqual(call_kwargs["category"], "research paper")
        self.assertEqual(call_kwargs["include_domains"], ["arxiv.org"])
        # snippet mode -> only highlights
        self.assertTrue(call_kwargs.get("highlights"))
        self.assertNotIn("text", call_kwargs)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Paper")
        self.assertEqual(results[0].description, "key finding")

    async def test_full_content_mode_requests_highlights_text_summary(self):
        mock_client = MagicMock()
        mock_client.search_and_contents.return_value = _mock_sdk_response([])

        search = ExaSearch("key", client=mock_client)
        await search("q", content_mode="full")

        _, kwargs = mock_client.search_and_contents.call_args
        self.assertTrue(kwargs.get("highlights"))
        self.assertTrue(kwargs.get("text"))
        self.assertEqual(kwargs.get("summary"), {})

    async def test_result_falls_back_when_highlights_missing(self):
        mock_client = MagicMock()
        mock_client.search_and_contents.return_value = _mock_sdk_response(
            [
                {
                    "title": "Article",
                    "url": "https://example.com",
                    "text": "a" * 50,
                }
            ]
        )

        search = ExaSearch("key", client=mock_client)
        results = await search("q", content_mode="text")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].description, "a" * 50)
        self.assertEqual(results[0].highlights, [])


class TestExaClientIntegrationHeader(unittest.TestCase):
    def test_client_receives_integration_header(self):
        """Verify the x-exa-integration tracking header is set on the SDK client."""
        fake_headers: dict = {}
        fake_client = MagicMock()
        fake_client.headers = fake_headers

        with patch("exa_py.Exa", return_value=fake_client) as mock_exa:
            search = ExaSearch("my-key")
            client = search._get_client()

        mock_exa.assert_called_once_with(api_key="my-key")
        self.assertIs(client, fake_client)
        self.assertEqual(fake_headers.get("x-exa-integration"), "motus")


class TestExaSearchTool(unittest.TestCase):
    def test_tool_raises_when_api_key_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                ExaSearchTool()

    def test_tool_uses_explicit_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            tool = ExaSearchTool(api_key="explicit")
            self.assertEqual(tool.name, "ExaSearchTool")

    def test_tool_reads_env_var(self):
        with patch.dict(os.environ, {"EXA_API_KEY": "env-key"}, clear=True):
            tool = ExaSearchTool()
            self.assertEqual(tool.name, "ExaSearchTool")


if __name__ == "__main__":
    unittest.main()
