import asyncio
import os
from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import Field

from ...core import FunctionTool
from ...core.function_tool import InputSchema

SearchType = Literal[
    "auto", "neural", "fast", "instant", "deep-lite", "deep", "deep-reasoning"
]
Category = Literal[
    "company",
    "research paper",
    "news",
    "personal site",
    "financial report",
    "people",
]


class ExaSearchInputSchema(InputSchema):
    query: str = Field(
        description="Natural-language search query. Be specific for better results. "
        'Example: "latest research on mixture-of-experts models".',
    )
    k: int = Field(
        default=5,
        description="Maximum number of results to return (1-100).",
    )
    type: Optional[SearchType] = Field(
        default=None,
        description='Search type: "auto" (default), "neural", "fast", "instant", '
        '"deep-lite", "deep", or "deep-reasoning". Leave unset to let Exa choose.',
    )
    category: Optional[Category] = Field(
        default=None,
        description='Restrict results to a content category: "company", "research paper", '
        '"news", "personal site", "financial report", or "people".',
    )
    include_domains: Optional[list[str]] = Field(
        default=None,
        description='Only return results from these domains (e.g. ["arxiv.org", "nature.com"]).',
    )
    exclude_domains: Optional[list[str]] = Field(
        default=None,
        description="Exclude results from these domains.",
    )
    start_published_date: Optional[str] = Field(
        default=None,
        description='Only return content published on/after this date (ISO 8601, e.g. "2024-01-01").',
    )
    end_published_date: Optional[str] = Field(
        default=None,
        description="Only return content published on/before this date (ISO 8601).",
    )
    content_mode: Literal["snippet", "text", "summary", "full"] = Field(
        default="snippet",
        description='Content to retrieve for each result. "snippet" (default) returns short '
        'highlights, "text" returns the full page text, "summary" returns an LLM-generated '
        'summary, "full" returns highlights, text, and summary.',
    )


@dataclass
class ExaSearchResult:
    title: str
    url: str
    description: str
    published_date: Optional[str] = None
    author: Optional[str] = None
    score: Optional[float] = None
    highlights: list[str] = field(default_factory=list)
    text: Optional[str] = None
    summary: Optional[str] = None


def _build_contents(mode: str) -> dict:
    if mode == "text":
        return {"text": True}
    if mode == "summary":
        return {"summary": {}}
    if mode == "full":
        return {"highlights": True, "text": True, "summary": {}}
    # snippet (default)
    return {"highlights": True}


def _coerce_text(value) -> Optional[str]:
    """Extract a plain string from a text/summary field that may be a str or structured dict."""
    if value is None or value is False:
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, dict):
        for key in ("text", "summary", "value", "content"):
            if key in value and isinstance(value[key], str):
                return value[key] or None
    return None


def _coerce_highlights(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [h for h in value if isinstance(h, str)]
    return []


def _derive_description(
    highlights: list[str], text: Optional[str], summary: Optional[str]
) -> str:
    """Cascade through available content to build a short snippet/description."""
    if summary:
        return summary
    if highlights:
        return " … ".join(highlights)
    if text:
        return text[:500]
    return ""


def _parse_result(raw: dict) -> ExaSearchResult:
    highlights = _coerce_highlights(raw.get("highlights"))
    text = _coerce_text(raw.get("text"))
    summary = _coerce_text(raw.get("summary"))
    return ExaSearchResult(
        title=raw.get("title") or "",
        url=raw.get("url") or "",
        description=_derive_description(highlights, text, summary),
        published_date=raw.get("publishedDate") or raw.get("published_date"),
        author=raw.get("author"),
        score=raw.get("score"),
        highlights=highlights,
        text=text,
        summary=summary,
    )


class ExaSearch:
    def __init__(self, api_key: str, client: Optional[object] = None) -> None:
        self.api_key = api_key
        self._client = client

    def _get_client(self):
        """Lazily create the Exa SDK client and attach the integration header."""
        if self._client is not None:
            return self._client

        from exa_py import Exa

        client = Exa(api_key=self.api_key)
        # Attribution header so Exa can track usage originating from Motus.
        try:
            client.headers["x-exa-integration"] = "motus"
        except (AttributeError, TypeError):
            pass
        self._client = client
        return client

    def _search_sync(
        self,
        query: str,
        k: int,
        search_type: Optional[str],
        category: Optional[str],
        include_domains: Optional[list[str]],
        exclude_domains: Optional[list[str]],
        start_published_date: Optional[str],
        end_published_date: Optional[str],
        contents: dict,
    ) -> list[dict]:
        client = self._get_client()
        kwargs: dict = {"num_results": k}
        if search_type:
            kwargs["type"] = search_type
        if category:
            kwargs["category"] = category
        if include_domains:
            kwargs["include_domains"] = include_domains
        if exclude_domains:
            kwargs["exclude_domains"] = exclude_domains
        if start_published_date:
            kwargs["start_published_date"] = start_published_date
        if end_published_date:
            kwargs["end_published_date"] = end_published_date

        response = client.search_and_contents(query, **contents, **kwargs)

        raw_results = []
        for item in getattr(response, "results", []):
            if hasattr(item, "model_dump"):
                raw_results.append(item.model_dump())
            elif hasattr(item, "__dict__"):
                raw_results.append(
                    {k: v for k, v in vars(item).items() if not k.startswith("_")}
                )
            elif isinstance(item, dict):
                raw_results.append(item)
        return raw_results

    async def __call__(
        self,
        query: str,
        k: int = 5,
        type: Optional[str] = None,
        category: Optional[str] = None,
        include_domains: Optional[list[str]] = None,
        exclude_domains: Optional[list[str]] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        content_mode: str = "snippet",
    ) -> list[ExaSearchResult]:
        """Search the web with Exa's AI-powered search and return ranked results with content.

        Use for finding up-to-date information, research papers, news, company profiles,
        and other web content. Exa combines neural search with content retrieval, so each
        result comes back with a snippet (or full text/summary, depending on content_mode).

        Args:
            query: Natural-language search query. Be specific for better results.
            k: Maximum number of results to return (default 5, up to 100).
            type: Search strategy ("auto", "neural", "fast", "instant", "deep-lite",
                "deep", "deep-reasoning"). Leave unset to let Exa pick.
            category: Restrict to a category ("company", "research paper", "news",
                "personal site", "financial report", "people").
            include_domains: Only return results from these domains.
            exclude_domains: Exclude results from these domains.
            start_published_date: ISO 8601 date; only content published on/after.
            end_published_date: ISO 8601 date; only content published on/before.
            content_mode: "snippet" (highlights, default), "text" (full page text),
                "summary" (LLM summary), or "full" (all three).

        Returns:
            List of ExaSearchResult with title, url, description, and optional
            highlights/text/summary depending on content_mode.
        """
        contents = _build_contents(content_mode)
        raw_results = await asyncio.to_thread(
            self._search_sync,
            query,
            k,
            type,
            category,
            include_domains,
            exclude_domains,
            start_published_date,
            end_published_date,
            contents,
        )
        return [_parse_result(r) for r in raw_results]


class ExaSearchTool(FunctionTool):
    def __init__(self, api_key: Optional[str] = None) -> None:
        resolved_key = api_key or os.getenv("EXA_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Exa API key is required. Pass api_key=... or set EXA_API_KEY."
            )
        super().__init__(
            func=ExaSearch(resolved_key).__call__,
            name="ExaSearchTool",
            schema=ExaSearchInputSchema,
        )
