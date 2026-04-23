from __future__ import annotations

import logging
import os
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from mcp.server.fastmcp import FastMCP
from pydantic import ValidationError

from physics_core import ASSUMPTIONS_V1, BeamInput, get_supported_cases, solve_beam_case
from mcp_server.logging_config import configure_logging
from mcp_server.middleware import RateLimitHook

configure_logging()
logger = logging.getLogger("physics-mcp")

mcp = FastMCP("physics-mcp")


@mcp.tool()
def solve_beam_case_tool(payload: dict[str, Any]) -> dict[str, Any]:
    """Solve one beam case from physics_core v0.1."""
    data = BeamInput.model_validate(payload)
    result = solve_beam_case(data)
    return result.model_dump()


@mcp.tool()
def get_supported_cases_tool() -> list[str]:
    """Return supported case identifiers for v0.1 beam solver."""
    return get_supported_cases()


@mcp.tool()
def get_model_assumptions_tool() -> list[str]:
    """Return physics assumptions and model limits."""
    return ASSUMPTIONS_V1


app = FastAPI(title="physics-mcp", version="0.1.0")
rate_limit_hook = RateLimitHook()
app.middleware("http")(rate_limit_hook)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "physics-mcp", "version": "0.1.0"}


@app.post("/dev/tools/{tool_name}")
def dev_tool(tool_name: str, payload: dict[str, Any] | None = None) -> Any:
    payload = payload or {}
    try:
        if tool_name == "solve_beam_case":
            return solve_beam_case_tool(payload)
        if tool_name == "get_supported_cases":
            return get_supported_cases_tool()
        if tool_name == "get_model_assumptions":
            return get_model_assumptions_tool()
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise HTTPException(status_code=404, detail=f"Unknown tool {tool_name}")


app.mount("/mcp", mcp.streamable_http_app())


def run() -> None:
    host = os.getenv("PHYSICS_MCP_HOST", "0.0.0.0")
    port = int(os.getenv("PHYSICS_MCP_PORT", "8080"))
    logger.info("Starting physics-mcp server", extra={"host": host, "port": port})
    uvicorn.run("mcp_server.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    run()
