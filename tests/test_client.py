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
    monkeypatch.setattr(
        langchain_chat.importlib.metadata,
        "version",
        lambda name: "0.3.84" if name == "langchain-core" else "0.2.1",
    )

    with pytest.raises(RuntimeError, match="Failed to initialize MCP tools"):
        import asyncio

        asyncio.run(langchain_chat._build_mcp_tools(["postdoc"]))


def test_validate_langchain_mcp_versions_rejects_old_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        langchain_chat.importlib.metadata,
        "version",
        lambda name: "0.3.84" if name == "langchain-core" else "0.1.14",
    )

    with pytest.raises(RuntimeError, match="too old"):
        langchain_chat._validate_langchain_mcp_versions()


def test_main_replaces_unresolved_tool_call_response_after_loop_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _LoopingResponse:
        tool_calls = [{"id": "call-1", "name": "missing_tool", "args": {"value": 3}}]
        content = ""

    class _FinalResponse:
        tool_calls: list[dict[str, object]] = []
        content = "Zweite Antwort"

    class _FakeModel:
        def __init__(self) -> None:
            self._invoke_count = 0

        def invoke(self, messages):
            self._invoke_count += 1
            if self._invoke_count == 4:
                assert not getattr(messages[-2], "tool_calls", None)
                return _FinalResponse()
            return _LoopingResponse()

    class _NoopTool:
        name = "websearch"

    questions = iter(["Erste Frage", "Zweite Frage", "/exit"])
    monkeypatch.setattr(langchain_chat, "_build_chat_model", lambda: _FakeModel())
    monkeypatch.setattr(langchain_chat, "_build_web_search_tool", lambda: _NoopTool())
    monkeypatch.setattr(langchain_chat, "_select_toolset_interactive", lambda: "0")
    monkeypatch.setattr(langchain_chat, "_read_question", lambda: next(questions))
    monkeypatch.setattr(langchain_chat, "MAX_TOOL_ROUNDS", 8)

    langchain_chat.main()
