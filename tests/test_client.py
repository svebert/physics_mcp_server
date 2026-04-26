from __future__ import annotations

import sys
import types

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


def test_build_mcp_tools_wraps_session_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyClient:
        def __init__(self, _server_map: dict[str, dict[str, str]]) -> None:
            self.closed = False

        async def get_tools(self) -> list[object]:
            raise RuntimeError("Session terminated")

        async def aclose(self) -> None:
            self.closed = True

    module = types.ModuleType("langchain_mcp_adapters.client")
    module.MultiServerMCPClient = DummyClient
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters.client", module)

    with pytest.raises(RuntimeError, match="Failed to initialize MCP tools"):
        import asyncio

        asyncio.run(langchain_chat._build_mcp_tools(["postdoc"]))
