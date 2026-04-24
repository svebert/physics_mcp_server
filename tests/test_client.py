from __future__ import annotations

import httpx
import pytest

from client import langchain_chat


def test_invoke_server_tool_returns_structured_error_on_connect_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_connect_error(*args: object, **kwargs: object) -> None:
        request = httpx.Request("POST", "http://127.0.0.1:8080/dev/tools/solve_beam_case")
        raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.setattr(httpx.Client, "post", _raise_connect_error)
    outcome = langchain_chat.invoke_server_tool("solve_beam_case", {"case": "cantilever"})

    assert outcome["ok"] is False
    assert outcome["error"]["status_code"] is None
    assert "Cannot reach physics MCP server" in outcome["error"]["details"]


def test_ensure_server_is_reachable_raises_runtime_error_on_health_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_connect_error(*args: object, **kwargs: object) -> None:
        request = httpx.Request("GET", "http://127.0.0.1:8080/health")
        raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.setattr(httpx.Client, "get", _raise_connect_error)

    with pytest.raises(RuntimeError, match="not reachable"):
        langchain_chat._ensure_server_is_reachable()


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
