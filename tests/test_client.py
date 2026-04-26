from __future__ import annotations

import httpx
import pytest

from client import langchain_chat


def test_ensure_server_is_reachable_raises_runtime_error_on_health_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_connect_error(*args: object, **kwargs: object) -> None:
        request = httpx.Request("GET", "http://127.0.0.1:8080/health")
        raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.setattr(httpx.Client, "get", _raise_connect_error)

    with pytest.raises(RuntimeError, match="not reachable"):
        langchain_chat._ensure_server_is_reachable("http://127.0.0.1:8080", "mcp-physics")


def test_resolve_api_key_prefers_generic_llm_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "generic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    assert langchain_chat._resolve_api_key("openai") == "generic-key"


def test_resolve_api_key_keeps_openai_backward_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-openai-key")

    assert langchain_chat._resolve_api_key("openai") == "legacy-openai-key"


def test_toolset_options_match_expected_modes() -> None:
    labels = {option["label"] for option in langchain_chat.TOOLSET_OPTIONS.values()}
    assert labels == {
        "none",
        "mcp-physics",
        "websearch",
        "mcp-physics + websearch",
        "physik-postdoc",
        "physik-postdoc + websearch",
    }
    assert langchain_chat.TOOLSET_OPTIONS["2"]["tools"] == ["websearch"]
    assert "websearch" in langchain_chat.TOOLSET_OPTIONS["5"]["tools"]


def test_search_google_news_rss_parses_items(monkeypatch: pytest.MonkeyPatch) -> None:
    rss = """
    <rss><channel>
      <item>
        <title>Hamburg News</title>
        <link>https://example.org/news1</link>
        <description>Kurzmeldung</description>
        <pubDate>Sun, 26 Apr 2026 10:00:00 GMT</pubDate>
      </item>
    </channel></rss>
    """

    def _fake_get(*args: object, **kwargs: object) -> httpx.Response:
        request = httpx.Request("GET", "https://news.google.com/rss/search")
        return httpx.Response(status_code=200, text=rss, request=request)

    monkeypatch.setattr(httpx.Client, "get", _fake_get)

    results = langchain_chat._search_google_news_rss("Hamburg", max_results=5)

    assert len(results) == 1
    assert results[0]["title"] == "Hamburg News"
    assert results[0]["url"] == "https://example.org/news1"


def test_web_search_prefers_google_news_results(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_news(query: str, max_results: int) -> list[dict[str, str]]:
        return [{"title": "N", "url": "https://example.org", "snippet": "S"}]

    monkeypatch.setattr(langchain_chat, "_search_google_news_rss", _fake_news)

    payload = langchain_chat._web_search("News Hamburg", max_results=3)

    assert payload["ok"] is True
    assert payload["results"][0]["title"] == "N"
