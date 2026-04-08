import os
import urllib.parse
from typing import Optional

import httpx

from ...core import FunctionTool
from ...core.function_tool import InputSchema

BASE_URL = os.getenv("BRAVE_BASE_URL", "https://api.search.brave.com/res/v1/web/search")


class WebSearchInputSchema(InputSchema):
    query: str
    k: int = 5


class WebSearch:
    def __init__(
        self, api_key: str, http_client: Optional[httpx.AsyncClient] = None
    ) -> None:
        self.api_key = api_key
        self._http_client = http_client
        self._owns_client = False

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create httpx client."""
        if self._http_client is not None:
            return self._http_client

        # Create new client
        self._http_client = httpx.AsyncClient()
        self._owns_client = True
        return self._http_client

    async def __call__(self, query: str, k: int = 5) -> list[dict]:
        """Search the web using Brave Search API and return relevant results.

        Use for finding current information, researching topics, or locating URLs.
        Returns a list of search results with title, url, and description.

        Args:
            query: Search query string. Be specific for better results.
                   Example: "Python asyncio tutorial 2024" or "arxiv 2405.05751"
            k: Maximum number of results to return. Defaults to 5.

        Returns:
            List of dicts with keys: title, url, description, age (optional).
        """
        encoded_query = "q=" + urllib.parse.quote(query)
        url = f"{BASE_URL}?{encoded_query}"
        headers = {"X-Subscription-Token": self.api_key}

        client = await self._get_client()
        response = await client.get(url, headers=headers)
        response.raise_for_status()

        payload = response.json()
        results = payload["web"]["results"]
        return results[:k]


class WebSearchTool(FunctionTool):
    def __init__(self, api_key: str) -> None:
        super().__init__(
            func=WebSearch(api_key).__call__,
            name="WebSearchTool",
            schema=WebSearchInputSchema,
        )
