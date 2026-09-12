"""Read-only web search tool for the read-side tool registry."""

from __future__ import annotations

import asyncio
import html
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Protocol
from urllib.parse import urlparse
from xml.etree import ElementTree

import httpx
from tavily import TavilyClient


WEB_SEARCH_MAX_SOURCES = 6
WEB_SEARCH_MAX_TOTAL_CHARS = 12000
WEB_SEARCH_TIMEOUT_SECONDS = 8.0
TAVILY_MAX_RESULTS = 5


@dataclass(frozen=True)
class WebSearchSource:
    source_id: str
    title: str
    url: str
    published_at: str | None
    relevant_text: str


@dataclass(frozen=True)
class WebSearchResult:
    status: str
    query: str
    freshness: str
    searched_at: str
    sources: list[WebSearchSource]
    diagnostics: dict[str, Any] = field(default_factory=dict)


class WebSearchProvider(Protocol):
    async def search(self, *, query: str, freshness: str = "none", max_sources: int = WEB_SEARCH_MAX_SOURCES) -> WebSearchResult:
        """Return compact normalized web evidence for a query."""


class DuckDuckGoWebSearchProvider:
    """Low-dependency HTML search provider with compact result extraction."""

    def __init__(self, *, timeout_seconds: float = WEB_SEARCH_TIMEOUT_SECONDS) -> None:
        self.timeout_seconds = timeout_seconds

    async def search(self, *, query: str, freshness: str = "none", max_sources: int = WEB_SEARCH_MAX_SOURCES) -> WebSearchResult:
        started = time.perf_counter()
        params = {"q": query}
        freshness_param = _freshness_param(freshness)
        if freshness_param:
            params["df"] = freshness_param
        async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
            response = await client.get("https://duckduckgo.com/html/", params=params)
            response.raise_for_status()
        parsed = _DuckDuckGoHTMLParser()
        parsed.feed(response.text)
        sources = _compact_sources(parsed.results, max_sources)
        if not sources and _looks_newsy(query, freshness):
            sources = await self._news_sources(query, max_sources)
        if not sources:
            sources = await self._instant_answer_sources(query, max_sources)
        return WebSearchResult(
            status="OK" if sources else "NO_RESULTS",
            query=query,
            freshness=freshness,
            searched_at=datetime.now(timezone.utc).isoformat(),
            sources=sources,
            diagnostics={"provider": "duckduckgo_html", "search_ms": (time.perf_counter() - started) * 1000},
        )

    async def _news_sources(self, query: str, max_sources: int) -> list[WebSearchSource]:
        async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
            response = await client.get(
                "https://news.google.com/rss/search",
                params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
            )
            response.raise_for_status()
        root = ElementTree.fromstring(response.text)
        results = []
        for item in root.findall("./channel/item"):
            title = _clean_text(item.findtext("title") or "")
            url = _clean_text(item.findtext("link") or "")
            description = _clean_text(re.sub(r"<[^>]+>", " ", item.findtext("description") or ""))
            published = _clean_text(item.findtext("pubDate") or "")
            if title and url:
                snippet = description or title
                if published:
                    snippet = f"{published}. {snippet}"
                results.append({"title": title, "url": url, "snippet": snippet})
            if len(results) >= max_sources:
                break
        return _compact_sources(results, max_sources)

    async def _instant_answer_sources(self, query: str, max_sources: int) -> list[WebSearchSource]:
        async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
            response = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            )
            response.raise_for_status()
        data = response.json()
        results: list[dict[str, str]] = []
        abstract = _clean_text(data.get("AbstractText") or "")
        if abstract:
            results.append(
                {
                    "title": _clean_text(data.get("Heading") or query),
                    "url": _clean_text(data.get("AbstractURL") or ""),
                    "snippet": abstract,
                }
            )
        for topic in data.get("RelatedTopics") or []:
            if "Topics" in topic:
                iterable = topic.get("Topics") or []
            else:
                iterable = [topic]
            for item in iterable:
                text = _clean_text(item.get("Text") or "")
                url = _clean_text(item.get("FirstURL") or "")
                if text and url:
                    results.append({"title": text.split(" - ")[0][:120], "url": url, "snippet": text})
                if len(results) >= max_sources:
                    break
            if len(results) >= max_sources:
                break
        return _compact_sources(results, max_sources)


class TavilyWebSearchProvider:
    """Tavily-backed search provider for reliable low-latency web evidence."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout_seconds: float = WEB_SEARCH_TIMEOUT_SECONDS,
        client: TavilyClient | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.timeout_seconds = timeout_seconds
        self.client = client

    async def search(self, *, query: str, freshness: str = "none", max_sources: int = WEB_SEARCH_MAX_SOURCES) -> WebSearchResult:
        started = time.perf_counter()
        if not self.api_key and self.client is None:
            return self._provider_error(query, freshness, started, "missing TAVILY_API_KEY")

        try:
            data = await self._search_once(query=query, freshness=freshness, max_sources=max_sources)
        except Exception as first_exc:
            if not _is_transient_tavily_error(first_exc):
                return self._provider_error(query, freshness, started, first_exc)
            try:
                data = await self._search_once(query=query, freshness=freshness, max_sources=max_sources)
            except Exception as second_exc:
                return self._provider_error(query, freshness, started, second_exc, retried=True)

        sources = _compact_tavily_sources(data.get("results") or [], min(max_sources, TAVILY_MAX_RESULTS))
        return WebSearchResult(
            status="OK" if sources else "NO_RESULTS",
            query=query,
            freshness=freshness,
            searched_at=datetime.now(timezone.utc).isoformat(),
            sources=sources,
            diagnostics={
                "provider": "tavily",
                "search_ms": (time.perf_counter() - started) * 1000,
                "request_settings": _tavily_request_settings(freshness, min(max_sources, TAVILY_MAX_RESULTS)),
            },
        )

    async def _search_once(self, *, query: str, freshness: str, max_sources: int) -> dict[str, Any]:
        client = self.client or TavilyClient(api_key=self.api_key)
        settings = _tavily_request_settings(freshness, min(max_sources, TAVILY_MAX_RESULTS))
        return await asyncio.to_thread(client.search, query, timeout=self.timeout_seconds, **settings)

    def _provider_error(
        self,
        query: str,
        freshness: str,
        started: float,
        error: Any,
        *,
        retried: bool = False,
    ) -> WebSearchResult:
        return WebSearchResult(
            status="PROVIDER_ERROR",
            query=query,
            freshness=freshness,
            searched_at=datetime.now(timezone.utc).isoformat(),
            sources=[],
            diagnostics={
                "provider": "tavily",
                "search_ms": (time.perf_counter() - started) * 1000,
                "request_settings": _tavily_request_settings(freshness, TAVILY_MAX_RESULTS),
                "retried": retried,
                "error": _safe_error_text(error),
            },
        )


class WebSearchTool:
    """Synchronous read-path facade around an async web provider."""

    def __init__(self, provider: WebSearchProvider | None = None) -> None:
        self.provider = provider or _default_provider()

    def search(self, *, query: str, freshness: str = "none") -> WebSearchResult:
        if not query.strip():
            raise ValueError("web search query must not be empty")
        if freshness not in {"none", "day", "week", "month", "year"}:
            raise ValueError("freshness must be one of none, day, week, month, year")
        return asyncio.run(self.provider.search(query=query.strip(), freshness=freshness, max_sources=WEB_SEARCH_MAX_SOURCES))


def search_web(*, query: str, freshness: str = "none", tool: WebSearchTool | None = None) -> WebSearchResult:
    return (tool or WebSearchTool()).search(query=query, freshness=freshness)


class _DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._in_title = False
        self._in_snippet = False
        self._current_href: str | None = None
        self._current_title: list[str] = []
        self._current_snippet: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        class_name = attrs_dict.get("class") or ""
        if tag == "a" and "result__a" in class_name:
            self._in_title = True
            self._current_href = attrs_dict.get("href")
            self._current_title = []
        if "result__snippet" in class_name:
            self._in_snippet = True
            self._current_snippet = []

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._current_title.append(data)
        if self._in_snippet:
            self._current_snippet.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_title:
            self._in_title = False
            title = _clean_text(" ".join(self._current_title))
            if title and self._current_href:
                self.results.append({"title": title, "url": self._current_href, "snippet": ""})
        if self._in_snippet and tag in {"a", "div"}:
            self._in_snippet = False
            snippet = _clean_text(" ".join(self._current_snippet))
            if snippet and self.results:
                self.results[-1]["snippet"] = snippet


def _compact_sources(results: list[dict[str, str]], max_sources: int) -> list[WebSearchSource]:
    sources: list[WebSearchSource] = []
    used_chars = 0
    for result in results:
        title = _clean_text(result.get("title") or "")
        url = _clean_url(result.get("url") or "")
        text = _clean_text(result.get("snippet") or "")
        if not title or not url or not text:
            continue
        remaining = WEB_SEARCH_MAX_TOTAL_CHARS - used_chars
        if remaining <= 0:
            break
        relevant_text = text[: min(1200, remaining)]
        sources.append(
            WebSearchSource(
                source_id=f"W{len(sources) + 1}",
                title=title,
                url=url,
                published_at=_extract_date(text),
                relevant_text=relevant_text,
            )
        )
        used_chars += len(relevant_text)
        if len(sources) >= max_sources:
            break
    return sources


def _compact_tavily_sources(results: list[dict[str, Any]], max_sources: int) -> list[WebSearchSource]:
    sources: list[WebSearchSource] = []
    used_chars = 0
    for result in results:
        title = _clean_text(result.get("title") or "")
        url = _clean_url(result.get("url") or "")
        text = _clean_text(result.get("content") or "")
        if not title or not url or not text:
            continue
        remaining = WEB_SEARCH_MAX_TOTAL_CHARS - used_chars
        if remaining <= 0:
            break
        relevant_text = text[: min(1200, remaining)]
        sources.append(
            WebSearchSource(
                source_id=f"W{len(sources) + 1}",
                title=title,
                url=url,
                published_at=_clean_text(result.get("published_date") or "") or _extract_date(text),
                relevant_text=relevant_text,
            )
        )
        used_chars += len(relevant_text)
        if len(sources) >= max_sources:
            break
    return sources


def _freshness_param(freshness: str) -> str | None:
    return {"day": "d", "week": "w", "month": "m", "year": "y"}.get(freshness)


def _tavily_time_range(freshness: str) -> str | None:
    return None if freshness == "none" else freshness


def _tavily_request_settings(freshness: str, max_results: int) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "search_depth": "basic",
        "max_results": min(max_results, TAVILY_MAX_RESULTS),
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
        "include_published_date": True,
        "auto_parameters": False,
    }
    time_range = _tavily_time_range(freshness)
    if time_range:
        settings["time_range"] = time_range
    return settings


def _default_provider() -> WebSearchProvider:
    provider = os.getenv("KIVI_WEB_PROVIDER", "tavily").strip().casefold()
    if provider == "duckduckgo":
        return DuckDuckGoWebSearchProvider()
    if provider != "tavily":
        raise ValueError("KIVI_WEB_PROVIDER must be one of tavily, duckduckgo")
    return TavilyWebSearchProvider()


def _is_transient_tavily_error(exc: Exception) -> bool:
    text = str(exc).casefold()
    return any(marker in text for marker in ["429", "timeout", "timed out", "temporarily", "500", "502", "503", "504"])


def _safe_error_text(error: Any) -> str:
    return " ".join(str(error).split())[:300]


def _looks_newsy(query: str, freshness: str) -> bool:
    lowered = query.casefold()
    return freshness in {"day", "week", "month"} or any(word in lowered for word in ["latest", "news", "today", "recent"])


def _clean_url(url: str) -> str:
    value = html.unescape(url).strip()
    if value.startswith("//duckduckgo.com/l/?"):
        match = re.search(r"[?&]uddg=([^&]+)", value)
        if match:
            from urllib.parse import unquote

            value = unquote(match.group(1))
    if value.startswith("//"):
        value = "https:" + value
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        return ""
    return value


def _clean_text(text: str) -> str:
    return " ".join(html.unescape(text).split())


def _extract_date(text: str) -> str | None:
    match = re.search(r"\b(20\d{2}-\d{2}-\d{2}|20\d{2})\b", text)
    return match.group(1) if match else None
