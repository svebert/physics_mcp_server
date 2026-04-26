from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mcp_server.middleware import RateLimitHook


def test_debug_logs_request_and_response_payloads() -> None:
    app = FastAPI()
    hook = RateLimitHook(logger_name="physics-mcp")
    hook._logger.setLevel("DEBUG")
    debug_messages: list[str] = []

    def _capture(message: str, *args) -> None:
        debug_messages.append(message % args)

    hook._logger.debug = _capture  # type: ignore[method-assign]
    app.middleware("http")(hook)

    @app.post("/echo")
    def echo(payload: dict[str, str]) -> dict[str, str]:
        return {"received": payload["value"]}

    response = TestClient(app).post("/echo", json={"value": "hello"})

    assert response.status_code == 200
    assert any('request_payload={"value": "hello"}' in message for message in debug_messages)
    assert any('response_payload={"received": "hello"}' in message for message in debug_messages)
