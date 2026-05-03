"""Tests for ``tools.builtins.web`` — the LLM-extracted ``web_fetch`` and
Brave-backed ``web_search`` builtins used by ``CodingAgent``.

External I/O is mocked at the ``httpx.AsyncClient`` and
``model_serve_task`` boundary; the tests exercise our request shaping,
response parsing, error paths, and graceful no-op behaviour.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from motus.tools.builtins.web import (
    _build_brave_query,
    _format_brave_results,
    make_web_fetch,
    make_web_search,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resp(
    status_code: int = 200,
    text: str | None = None,
    json_payload: Any = None,
    content_type: str = "text/html",
) -> httpx.Response:
    headers = {"content-type": content_type}
    kwargs: dict[str, Any] = {
        "status_code": status_code,
        "headers": headers,
        "request": httpx.Request("GET", "https://example.com"),
    }
    if json_payload is not None:
        kwargs["json"] = json_payload
    elif text is not None:
        kwargs["text"] = text
    return httpx.Response(**kwargs)


# ---------------------------------------------------------------------------
# Brave query builder
# ---------------------------------------------------------------------------


class TestBraveQueryBuilder:
    def test_plain_query(self):
        assert _build_brave_query("hello world", None, None) == "hello world"

    def test_allowed_domains_appended_as_site_filter(self):
        out = _build_brave_query("foo", ["a.com", "b.com"], None)
        assert "site:a.com OR site:b.com" in out
        assert out.startswith("foo")

    def test_blocked_domains_appended_with_negation(self):
        out = _build_brave_query("foo", None, ["bad.com", "noisy.io"])
        assert "-site:bad.com" in out
        assert "-site:noisy.io" in out

    def test_both_allowed_and_blocked(self):
        out = _build_brave_query("foo", ["good.com"], ["bad.com"])
        assert "site:good.com" in out and "-site:bad.com" in out


# ---------------------------------------------------------------------------
# Brave result formatter
# ---------------------------------------------------------------------------


class TestBraveResultFormatter:
    def test_empty_results(self):
        out = _format_brave_results("nothing", [])
        assert "No results" in out
        assert "'nothing'" in out

    def test_renders_markdown_links(self):
        results = [
            {
                "title": "Python Docs",
                "url": "https://docs.python.org",
                "description": "Official documentation",
            }
        ]
        out = _format_brave_results("python", results)
        assert "[Python Docs](https://docs.python.org)" in out
        assert "Official documentation" in out

    def test_strips_brave_html_tags_from_description(self):
        results = [
            {
                "title": "T",
                "url": "https://x.com",
                "description": "Hello <strong>bold</strong> world",
            }
        ]
        out = _format_brave_results("q", results)
        assert "<strong>" not in out
        assert "</strong>" not in out
        assert "Hello bold world" in out

    def test_handles_missing_fields(self):
        out = _format_brave_results("q", [{"url": "https://x.com"}])
        assert "(untitled)" in out


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------


class TestWebSearchTool:
    @pytest.mark.asyncio
    async def test_no_api_key_returns_friendly_error(self, monkeypatch):
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        search = make_web_search(api_key=None)
        out = await search(query="anything")
        assert "BRAVE_API_KEY" in out

    @pytest.mark.asyncio
    async def test_calls_brave_with_correct_headers(self):
        payload = {
            "web": {
                "results": [
                    {
                        "title": "T",
                        "url": "https://x.com",
                        "description": "D",
                    }
                ]
            }
        }
        mock_resp = _resp(json_payload=payload, content_type="application/json")

        async def _fake_get(self, url, params=None, headers=None, **_):
            assert url.endswith("/web/search")
            assert params["q"] == "py docs"
            assert headers["X-Subscription-Token"] == "key-abc"
            return mock_resp

        with patch.object(httpx.AsyncClient, "get", _fake_get):
            search = make_web_search(api_key="key-abc")
            out = await search(query="py docs")

        assert "[T](https://x.com)" in out

    @pytest.mark.asyncio
    async def test_brave_http_error_returned_as_string(self):
        async def _fake_get(self, url, params=None, headers=None, **_):
            r = _resp(status_code=429, json_payload={}, content_type="application/json")
            raise httpx.HTTPStatusError("rate limited", request=r.request, response=r)

        with patch.object(httpx.AsyncClient, "get", _fake_get):
            search = make_web_search(api_key="key-abc")
            out = await search(query="q")

        assert "429" in out
        assert "Error" in out

    @pytest.mark.asyncio
    async def test_allowed_and_blocked_domains_threaded_into_query(self):
        captured: dict = {}

        async def _fake_get(self, url, params=None, headers=None, **_):
            captured["q"] = params["q"]
            return _resp(
                json_payload={"web": {"results": []}},
                content_type="application/json",
            )

        with patch.object(httpx.AsyncClient, "get", _fake_get):
            search = make_web_search(api_key="key-abc")
            await search(
                query="asyncio",
                allowed_domains=["docs.python.org"],
                blocked_domains=["w3schools.com"],
            )

        assert "site:docs.python.org" in captured["q"]
        assert "-site:w3schools.com" in captured["q"]


# ---------------------------------------------------------------------------
# web_fetch
# ---------------------------------------------------------------------------


HTML_PAGE = """
<html><head><title>Demo</title></head>
<body>
  <nav>nav junk</nav>
  <article>
    <h1>Real Content</h1>
    <p>The answer is 42.</p>
  </article>
  <footer>footer junk</footer>
</body></html>
"""


def _make_fake_completion(content: str):
    """Mock motus.models.base.ChatCompletion-shaped object."""
    c = MagicMock()
    c.content = content
    return c


class TestWebFetchTool:
    @pytest.mark.asyncio
    async def test_fetches_extracts_and_calls_extractor(self):
        async def _fake_get(self, url, **_):
            return _resp(text=HTML_PAGE, content_type="text/html")

        async def _fake_serve(client, model, messages, **_):
            # Verify the extractor sees URL + page content + the user prompt
            user_msg = messages[-1].content
            assert "https://example.com" in user_msg
            assert "Real Content" in user_msg
            assert "answer to life" in user_msg
            return _make_fake_completion("42")

        client = MagicMock(name="client")
        with (
            patch.object(httpx.AsyncClient, "get", _fake_get),
            patch("motus.agent.tasks.model_serve_task", _fake_serve),
        ):
            fetch = make_web_fetch(client=client, model_name="m")
            out = await fetch(
                url="https://example.com",
                prompt="What is the answer to life?",
            )

        assert out == "42"

    @pytest.mark.asyncio
    async def test_http_error_returned_as_string(self):
        async def _fake_get(self, url, **_):
            r = _resp(status_code=404, text="nope")
            raise httpx.HTTPStatusError("not found", request=r.request, response=r)

        client = MagicMock(name="client")
        with patch.object(httpx.AsyncClient, "get", _fake_get):
            fetch = make_web_fetch(client=client, model_name="m")
            out = await fetch(url="https://example.com", prompt="anything")

        assert "404" in out
        assert "Error" in out

    @pytest.mark.asyncio
    async def test_non_html_content_passed_through(self):
        async def _fake_get(self, url, **_):
            return _resp(
                text=json.dumps({"x": 1}),
                content_type="application/json",
            )

        captured: dict = {}

        async def _fake_serve(client, model, messages, **_):
            captured["page"] = messages[-1].content
            return _make_fake_completion("ok")

        client = MagicMock(name="client")
        with (
            patch.object(httpx.AsyncClient, "get", _fake_get),
            patch("motus.agent.tasks.model_serve_task", _fake_serve),
        ):
            fetch = make_web_fetch(client=client, model_name="m")
            await fetch(url="https://api.example.com/x", prompt="what is x")

        # Non-HTML responses should pass through raw, not be markdownified.
        assert '"x": 1' in captured["page"]

    @pytest.mark.asyncio
    async def test_uses_extraction_model_override(self):
        captured: dict = {}

        async def _fake_get(self, url, **_):
            return _resp(text=HTML_PAGE, content_type="text/html")

        async def _fake_serve(client, model, messages, **_):
            captured["model"] = model
            return _make_fake_completion("ok")

        client = MagicMock(name="client")
        with (
            patch.object(httpx.AsyncClient, "get", _fake_get),
            patch("motus.agent.tasks.model_serve_task", _fake_serve),
        ):
            fetch = make_web_fetch(
                client=client,
                model_name="claude-sonnet-4-6",
                extraction_model="claude-haiku-4-5",
            )
            await fetch(url="https://x.com", prompt="p")

        assert captured["model"] == "claude-haiku-4-5"
