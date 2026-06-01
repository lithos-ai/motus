"""Web tools — ``web_fetch`` (LLM-extracted) and ``web_search`` (Brave-backed).

Both tools are factory-built so the agent that owns them can supply its
own chat client (for ``web_fetch`` extraction) and API key (for
``web_search``). They mirror the schemas of Claude Code's WebFetch and
WebSearch.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from pydantic import ConfigDict, Field

from ...models import ChatMessage
from ..core import InputSchema
from ..core.decorators import tool
from ._helpers import truncate_output

logger = logging.getLogger(__name__)

# Brave Search API endpoint. Honors ``BRAVE_BASE_URL`` so deployments
# (e.g. the Motus cloud model proxy) can swap in their own forwarder
# that handles authentication on behalf of the agent.
BRAVE_SEARCH_URL = os.getenv(
    "BRAVE_BASE_URL", "https://api.search.brave.com/res/v1/web/search"
)

# A polite default UA. Some sites block headless requests with no UA.
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; MotusAgent/1.0; +https://lithosai.com)"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class WebFetchInput(InputSchema):
    url: str = Field(
        description="Fully-formed URL to fetch content from (must include scheme).",
    )
    prompt: str = Field(
        description="Instructions describing what information to extract from the page.",
    )

    model_config = ConfigDict(extra="forbid")


class WebSearchInput(InputSchema):
    query: str = Field(
        description="Search query to execute. Be specific for better results.",
    )
    allowed_domains: list[str] | None = Field(
        default=None,
        description="If set, only include results from these domains (e.g. ['python.org']).",
    )
    blocked_domains: list[str] | None = Field(
        default=None,
        description="If set, exclude results from these domains.",
    )

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# web_fetch — LLM-based extraction
# ---------------------------------------------------------------------------


def _html_to_markdown(html: str) -> str:
    """Extract main article content from raw HTML and convert to markdown.

    Falls back to the raw HTML→markdown pipeline if readability can't find
    a main article.
    """
    from markdownify import markdownify
    from readability import Document

    try:
        article_html = Document(html).summary()
    except Exception:
        article_html = html

    return markdownify(article_html, heading_style="ATX")


def make_web_fetch(
    client: Any,
    model_name: str,
    *,
    extraction_model: str | None = None,
    timeout_s: float = 30.0,
):
    """Create a ``web_fetch`` tool bound to *client* for content extraction.

    Args:
        client: Chat client used to run the extraction call. Same shape as
            the agent's main client (e.g. ``AnthropicChatClient``).
        model_name: The agent's main model — used as the default
            extraction model when ``extraction_model`` isn't set.
        extraction_model: Optional override for the extraction model. Use
            a small fast model (e.g. ``claude-haiku-4-5``) to keep costs
            low.
        timeout_s: HTTP timeout for the page fetch.

    Returns:
        A tool function decorated with ``@tool``.
    """
    extractor = extraction_model or model_name

    @tool(schema=WebFetchInput)
    async def web_fetch(url: str, prompt: str) -> str:
        """Fetch a web page and use a small LLM to extract the requested information.

        Returns the extractor's answer, not the raw page content. The
        page is fetched with redirects followed, the main article is
        extracted via readability, converted to markdown, and passed to
        a small model along with the user's extraction prompt.

        Usage notes:
          - Use this for retrieving and summarizing information from
            known URLs (docs, blog posts, GitHub READMEs, etc.).
          - The ``prompt`` argument should describe what to extract,
            e.g. "What does this page say about X?" or "List the API
            endpoints documented here." A specific prompt yields a
            focused answer; a vague prompt yields a vague answer.
          - For *finding* URLs from a query, use ``web_search`` first.
        """
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=timeout_s,
                headers={"User-Agent": DEFAULT_USER_AGENT},
            ) as http:
                resp = await http.get(url)
                resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            return f"Error fetching {url}: HTTP {e.response.status_code}"
        except httpx.HTTPError as e:
            return f"Error fetching {url}: {e}"

        content_type = resp.headers.get("content-type", "").lower()
        if "html" in content_type:
            try:
                content = _html_to_markdown(resp.text)
            except Exception as e:
                logger.warning(f"web_fetch: HTML→markdown failed for {url}: {e}")
                content = resp.text
        else:
            content = resp.text

        from motus.agent.tasks import model_serve_task
        from motus.models import ReasoningConfig

        try:
            completion = await model_serve_task(
                client=client,
                model=extractor,
                messages=[
                    ChatMessage.system_message(
                        "You are a web extraction assistant. Given a fetched web "
                        "page (in markdown) and a user request, extract the "
                        "relevant information concisely. Don't add commentary "
                        "beyond what the user asked for. If the page doesn't "
                        "contain the requested information, say so."
                    ),
                    ChatMessage.user_message(
                        f"URL: {url}\n\n<page>\n{content}\n</page>\n\nRequest: {prompt}"
                    ),
                ],
                reasoning=ReasoningConfig.disabled(),
            )
        except Exception as e:
            return f"Error extracting from {url}: {e}"

        return completion.content or "(empty extraction)"

    return web_fetch


# ---------------------------------------------------------------------------
# web_search — Brave-backed
# ---------------------------------------------------------------------------


def _build_brave_query(
    query: str,
    allowed_domains: list[str] | None,
    blocked_domains: list[str] | None,
) -> str:
    parts = [query]
    if allowed_domains:
        parts.append(" OR ".join(f"site:{d}" for d in allowed_domains))
    if blocked_domains:
        parts.extend(f"-site:{d}" for d in blocked_domains)
    return " ".join(parts)


def _format_brave_results(query: str, results: list[dict]) -> str:
    if not results:
        return f"No results for: {query!r}"
    lines = [f"Web search results for: {query!r}", ""]
    for r in results:
        title = r.get("title") or "(untitled)"
        url = r.get("url") or ""
        desc = r.get("description") or ""
        # Strip basic HTML tags Brave injects in the description.
        desc = (
            desc.replace("<strong>", "")
            .replace("</strong>", "")
            .replace("\n", " ")
            .strip()
        )
        lines.append(f"- [{title}]({url})")
        if desc:
            lines.append(f"  {desc}")
    return "\n".join(lines)


def make_web_search(
    api_key: str | None = None,
    *,
    k: int = 5,
    timeout_s: float = 15.0,
):
    """Create a ``web_search`` tool backed by Brave Search.

    Args:
        api_key: Brave API key. Defaults to the ``BRAVE_API_KEY`` env var.
            If neither is set, the returned tool returns an
            informative error message rather than failing at construction
            time.
        k: Number of results to return.
        timeout_s: HTTP timeout for the search request.

    Returns:
        A tool function decorated with ``@tool``.
    """
    resolved_key = api_key or os.getenv("BRAVE_API_KEY") or ""

    @tool(schema=WebSearchInput)
    async def web_search(
        query: str,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> str:
        """Search the web with Brave Search and return ranked results.

        Use this to find current information, locate URLs, or research
        topics. Returns titles, URLs, and snippets — call ``web_fetch``
        on a returned URL for fuller content.

        Usage notes:
          - ``query`` should be specific. Add years, names, or keywords
            to narrow scope.
          - Use ``allowed_domains`` to constrain results (e.g.
            ``["docs.python.org"]``).
          - Use ``blocked_domains`` to exclude noisy sources.
        """
        if not resolved_key:
            return (
                "web_search is not configured: set the BRAVE_API_KEY "
                "environment variable to enable web search."
            )

        full_query = _build_brave_query(query, allowed_domains, blocked_domains)
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as http:
                resp = await http.get(
                    BRAVE_SEARCH_URL,
                    params={"q": full_query, "count": k},
                    headers={"X-Subscription-Token": resolved_key},
                )
                resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = e.response.text[:500] if e.response.text else ""
            return f"Error from Brave Search: HTTP {e.response.status_code}" + (
                f" — {body}" if body else ""
            )
        except httpx.HTTPError as e:
            return f"Error contacting Brave Search: {e}"

        try:
            payload = resp.json()
            results = payload.get("web", {}).get("results", [])[:k]
        except Exception as e:
            return f"Error parsing Brave Search response: {e}"

        return truncate_output(_format_brave_results(query, results))

    return web_search
