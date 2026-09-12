from __future__ import annotations

import asyncio

from kivi_memory.read_orchestrator.web_search import TavilyWebSearchProvider, WebSearchTool


class FakeTavilyClient:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response if response is not None else {"results": []}
        self.error = error
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        if self.error is not None:
            raise self.error
        return self.response


def test_tavily_provider_uses_low_latency_settings_and_normalizes_sources() -> None:
    client = FakeTavilyClient(
        {
            "results": [
                {
                    "title": "NVIDIA update",
                    "url": "https://example.com/nvidia",
                    "content": "NVIDIA announced a new AI infrastructure update.",
                    "published_date": "2026-09-11",
                }
            ]
        }
    )
    provider = TavilyWebSearchProvider(api_key="test-key", client=client)

    result = asyncio.run(provider.search(query="latest NVIDIA news", freshness="week"))

    assert result.status == "OK"
    assert result.sources[0].source_id == "W1"
    assert result.sources[0].title == "NVIDIA update"
    assert result.sources[0].published_at == "2026-09-11"
    assert client.calls[0]["search_depth"] == "basic"
    assert client.calls[0]["max_results"] == 5
    assert client.calls[0]["include_answer"] is False
    assert client.calls[0]["include_raw_content"] is False
    assert client.calls[0]["include_images"] is False
    assert client.calls[0]["include_published_date"] is True
    assert client.calls[0]["time_range"] == "week"


def test_tavily_provider_reports_no_results() -> None:
    provider = TavilyWebSearchProvider(api_key="test-key", client=FakeTavilyClient({"results": []}))

    result = asyncio.run(provider.search(query="unlikely query", freshness="none"))

    assert result.status == "NO_RESULTS"
    assert result.sources == []


def test_tavily_provider_reports_provider_error_without_fallback() -> None:
    provider = TavilyWebSearchProvider(api_key="test-key", client=FakeTavilyClient(error=RuntimeError("401 auth failed")))

    result = asyncio.run(provider.search(query="latest NVIDIA news", freshness="day"))

    assert result.status == "PROVIDER_ERROR"
    assert result.sources == []
    assert result.diagnostics["provider"] == "tavily"


def test_web_search_tool_defaults_to_tavily_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("KIVI_WEB_PROVIDER", "tavily")
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    assert isinstance(WebSearchTool().provider, TavilyWebSearchProvider)
