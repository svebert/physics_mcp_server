from __future__ import annotations

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
