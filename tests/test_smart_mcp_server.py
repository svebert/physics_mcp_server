from __future__ import annotations

import asyncio
import importlib

from fastapi.testclient import TestClient

smart_module = importlib.import_module("smart_mcp_server.app")


def test_health_smoke() -> None:
    client = TestClient(smart_module.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["service"] == "physics-postdoc-mcp"


def test_ask_endpoint_uses_agent(monkeypatch) -> None:
    async def _fake_agent(question: str, enable_web_search: bool = True) -> str:
        assert "Träger" in question
        assert enable_web_search is False
        return "Berechnete Antwort"

    monkeypatch.setattr(smart_module, "run_postdoc_agent", _fake_agent)

    client = TestClient(smart_module.app)
    response = client.post("/v1/ask", json={"question": "Träger mit Punktlast", "enable_web_search": False})
    assert response.status_code == 200
    assert response.json()["answer"] == "Berechnete Antwort"


def test_mcp_endpoint_is_not_redirected() -> None:
    mount_routes = [route for route in smart_module.app.routes if getattr(route, "path", None) == ""]
    assert mount_routes
    mounted_app = mount_routes[0].app
    mounted_paths = [route.path for route in mounted_app.routes]
    assert "/mcp" in mounted_paths


def test_run_postdoc_agent_uses_async_tool_invocation(monkeypatch) -> None:
    class _ToolCallingResponse:
        tool_calls = [{"id": "call-1", "name": "ask_physics", "args": {"value": 3}}]
        content = ""

    class _FinalResponse:
        tool_calls: list[dict[str, object]] = []
        content = "Fertig"

    class _FakeModel:
        def __init__(self) -> None:
            self._call_count = 0

        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, _messages):
            self._call_count += 1
            if self._call_count == 1:
                return _ToolCallingResponse()
            return _FinalResponse()

    class _AsyncOnlyTool:
        name = "ask_physics"

        async def ainvoke(self, args):
            return {"ok": True, "echo": args}

        def invoke(self, _args):  # pragma: no cover - should never be called
            raise AssertionError("sync invoke must not be used")

    class _FakeClient:
        async def aclose(self):
            return None

    monkeypatch.setattr(smart_module, "_build_chat_model", lambda: _FakeModel())

    async def _fake_build_physics_mcp_tools():
        return _FakeClient(), [_AsyncOnlyTool()]

    monkeypatch.setattr(smart_module, "_build_physics_mcp_tools", _fake_build_physics_mcp_tools)
    monkeypatch.setattr(smart_module, "_build_tools", lambda enable_web_search: [])

    answer = asyncio.run(smart_module.run_postdoc_agent("Testfrage", enable_web_search=False))
    assert answer == "Fertig"


def test_local_web_search_tool_has_description() -> None:
    tools = smart_module._build_tools(enable_web_search=True)
    web_tools = [tool for tool in tools if getattr(tool, "name", "") == "web_search"]
    assert web_tools
    assert getattr(web_tools[0], "description", "")


def test_run_postdoc_agent_stops_repeated_tool_rounds(monkeypatch) -> None:
    class _LoopingResponse:
        tool_calls = [{"id": "call-1", "name": "ask_physics", "args": {"value": 3}}]
        content = ""

    class _FakeModel:
        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, _messages):
            return _LoopingResponse()

    class _AsyncTool:
        name = "ask_physics"

        async def ainvoke(self, _args):
            return {"ok": True}

    class _FakeClient:
        async def aclose(self):
            return None

    monkeypatch.setattr(smart_module, "_build_chat_model", lambda: _FakeModel())
    monkeypatch.setattr(smart_module, "MAX_TOOL_ROUNDS", 8)

    async def _fake_build_physics_mcp_tools():
        return _FakeClient(), [_AsyncTool()]

    monkeypatch.setattr(smart_module, "_build_physics_mcp_tools", _fake_build_physics_mcp_tools)
    monkeypatch.setattr(smart_module, "_build_tools", lambda enable_web_search: [])

    answer = asyncio.run(smart_module.run_postdoc_agent("Testfrage", enable_web_search=False))
    assert "wiederholt" in answer
