import unittest
from unittest.mock import AsyncMock

import httpx

from motus.tools.providers.brave.tool_provider import WebSearch


def _mock_brave_response(payload: dict, query: str = "test") -> httpx.Response:
    """Create a mock httpx.Response for Brave Search API."""
    return httpx.Response(
        status_code=200,
        json=payload,
        request=httpx.Request(
            "GET",
            f"https://api.search.brave.com/res/v1/web/search?q={query}",
        ),
    )


class TestWebSearch(unittest.IsolatedAsyncioTestCase):
    async def test_search_builds_request_and_parses_results(self):
        payload = {
            "web": {
                "results": [
                    {
                        "title": "Hello &amp; World",
                        "url": "https://example.com",
                        "description": "Desc &lt;b&gt;bold&lt;/b&gt;",
                        "page_age": "1d",
                    },
                    {
                        "title": "NoAge",
                        "url": "https://example.com/2",
                        "description": "Plain text",
                    },
                ]
            }
        }

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(
            return_value=_mock_brave_response(payload, "hello%20world")
        )

        search = WebSearch("token-123", http_client=mock_client)
        raw_results = await search("hello world")

        self.assertEqual(len(raw_results), 2)
        self.assertEqual(raw_results[0]["title"], "Hello &amp; World")
        self.assertEqual(raw_results[0]["page_age"], "1d")
        self.assertNotIn("page_age", raw_results[1])

        # Verify correct URL and auth header
        mock_client.get.assert_called_once()
        (call_url,) = mock_client.get.call_args[0]
        self.assertIn("q=hello%20world", call_url)
        self.assertEqual(
            mock_client.get.call_args[1]["headers"]["X-Subscription-Token"],
            "token-123",
        )

    async def test_search_limits_results_with_k_parameter(self):
        payload = {
            "web": {
                "results": [
                    {
                        "title": f"Result {i}",
                        "url": f"https://example.com/{i}",
                        "description": f"Description {i}",
                    }
                    for i in range(10)
                ]
            }
        }

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(
            return_value=_mock_brave_response(payload, "test%20query")
        )

        search = WebSearch("token-123", http_client=mock_client)

        raw_results = await search("test query")
        self.assertEqual(len(raw_results), 5)

        raw_results = await search("test query", k=3)
        self.assertEqual(len(raw_results), 3)

        raw_results = await search("test query", k=20)
        self.assertEqual(len(raw_results), 10)
