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


def _normalize_solve_beam_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept common shorthand keys from LLM tool-calls and map to API schema."""
    normalized = dict(payload)
    aliases = {
        "l": "length_m",
        "length": "length_m",
        "span_m": "length_m",
        "e": "youngs_modulus_pa",
        "youngs_modulus": "youngs_modulus_pa",
        "youngs_modulus_gpa": "youngs_modulus_pa",
        "i": "second_moment_m4",
        "second_moment": "second_moment_m4",
        "f": "point_load_n",
        "p": "point_load_n",
        "point_load": "point_load_n",
        "point_load_kn": "point_load_n",
        "x": "point_load_position_m",
        "a": "point_load_position_m",
        "load_position_m": "point_load_position_m",
        "udl": "udl_n_per_m",
        "q": "udl_n_per_m",
        "udl_kn_per_m": "udl_n_per_m",
    }
    for source, target in aliases.items():
        if source in normalized and target not in normalized:
            normalized[target] = normalized[source]

    if "youngs_modulus_gpa" in normalized and "youngs_modulus_pa" in normalized:
        normalized["youngs_modulus_pa"] = float(normalized["youngs_modulus_pa"]) * 1e9
    if "point_load_kn" in normalized and "point_load_n" in normalized:
        normalized["point_load_n"] = float(normalized["point_load_n"]) * 1e3
    if "udl_kn_per_m" in normalized and "udl_n_per_m" in normalized:
        normalized["udl_n_per_m"] = float(normalized["udl_n_per_m"]) * 1e3

    if "case" not in normalized:
        if "point_load_n" in normalized:
            normalized["case"] = "simply_supported_point"
        elif "udl_n_per_m" in normalized:
            normalized["case"] = "simply_supported_udl"
    return normalized


@mcp.tool()
def solve_beam_case_tool(payload: dict[str, Any]) -> dict[str, Any]:
    """Solve one beam case from physics_core v0.1."""
    data = BeamInput.model_validate(_normalize_solve_beam_payload(payload))
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
