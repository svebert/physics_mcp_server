from __future__ import annotations

import logging
import os
import unicodedata
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


def _normalize_key(raw_key: Any) -> str:
    text = unicodedata.normalize("NFKD", str(raw_key)).encode("ascii", "ignore").decode("ascii")
    normalized = "".join(ch if ch.isalnum() else "_" for ch in text.lower())
    return normalized.strip("_")


def _normalize_case_alias(value: str) -> str:
    normalized = _normalize_key(value)
    case_aliases = {
        "simply_supported_point": "simply_supported_point",
        "simply_supported_udl": "simply_supported_udl",
        "cantilever_point": "cantilever_point",
        "cantilever_udl": "cantilever_udl",
        # German
        "einfach_gelagert_einzellast": "simply_supported_point",
        "einfach_gelagert_punktlast": "simply_supported_point",
        "einfach_gelagert_streckenlast": "simply_supported_udl",
        "kragtrager_einzellast": "cantilever_point",
        "kragtrager_punktlast": "cantilever_point",
        "kragtrager_streckenlast": "cantilever_udl",
        # Spanish
        "viga_apoyada_carga_puntual": "simply_supported_point",
        "viga_apoyada_carga_distribuida": "simply_supported_udl",
        "voladizo_carga_puntual": "cantilever_point",
        "voladizo_carga_distribuida": "cantilever_udl",
        # French
        "appui_simple_charge_ponctuelle": "simply_supported_point",
        "appui_simple_charge_repartie": "simply_supported_udl",
        "console_charge_ponctuelle": "cantilever_point",
        "console_charge_repartie": "cantilever_udl",
    }
    if normalized in case_aliases:
        return case_aliases[normalized]

    if "einfach" in normalized and "gelag" in normalized and ("punkt" in normalized or "einzel" in normalized):
        return "simply_supported_point"
    if "einfach" in normalized and "gelag" in normalized and ("strecken" in normalized or "udl" in normalized):
        return "simply_supported_udl"
    if ("krag" in normalized or "cantilever" in normalized) and ("punkt" in normalized or "point" in normalized):
        return "cantilever_point"
    if ("krag" in normalized or "cantilever" in normalized) and ("strecken" in normalized or "udl" in normalized):
        return "cantilever_udl"
    return value


def _normalize_solve_beam_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept shorthand, multilingual, and nested keys from LLM tool-calls and map to API schema."""
    normalized = dict(payload)

    # Common shape from some clients: {"input": {...}} or {"arguments": {...}}
    nested = normalized.get("input") or normalized.get("arguments")
    if isinstance(nested, dict):
        normalized = {**nested, **{k: v for k, v in normalized.items() if k not in {"input", "arguments"}}}

    alias_map: dict[str, str] = {
        "l": "length_m",
        "length": "length_m",
        "lange": "length_m",
        "laenge": "length_m",
        "span": "length_m",
        "span_m": "length_m",
        "beam_length": "length_m",
        "e": "youngs_modulus_pa",
        "youngs_modulus": "youngs_modulus_pa",
        "elastic_modulus": "youngs_modulus_pa",
        "elastizitatsmodul": "youngs_modulus_pa",
        "elastizitaetsmodul": "youngs_modulus_pa",
        "youngs_modulus_gpa": "youngs_modulus_pa",
        "i": "second_moment_m4",
        "second_moment": "second_moment_m4",
        "area_moment": "second_moment_m4",
        "flachentragheitsmoment": "second_moment_m4",
        "flaechentraegheitsmoment": "second_moment_m4",
        "f": "point_load_n",
        "p": "point_load_n",
        "point_load": "point_load_n",
        "point_load_kn": "point_load_n",
        "punktlast": "point_load_n",
        "einzellast": "point_load_n",
        "x": "point_load_position_m",
        "a": "point_load_position_m",
        "load_position": "point_load_position_m",
        "load_position_m": "point_load_position_m",
        "position": "point_load_position_m",
        "lastposition": "point_load_position_m",
        "udl": "udl_n_per_m",
        "q": "udl_n_per_m",
        "udl_kn_per_m": "udl_n_per_m",
        "streckenlast": "udl_n_per_m",
        "distributed_load": "udl_n_per_m",
        "case": "case",
        "lastfall": "case",
        "typ": "case",
        "beam_case": "case",
        "samples": "samples",
        "stutzstellen": "samples",
    }

    for key in list(normalized.keys()):
        canonical_key = _normalize_key(key)
        target = alias_map.get(canonical_key)
        if target and target not in normalized:
            normalized[target] = normalized[key]

    if "case" in normalized and isinstance(normalized["case"], str):
        normalized["case"] = _normalize_case_alias(normalized["case"])

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
        normalized = _normalize_solve_beam_payload(payload) if tool_name == "solve_beam_case" else payload
        logger.warning(
            "Validation error in dev tool call",
            extra={"tool_name": tool_name, "payload": payload, "normalized_payload": normalized, "errors": exc.errors()},
        )
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Invalid tool payload",
                "tool": tool_name,
                "errors": exc.errors(),
                "received_keys": sorted(list(payload.keys())),
                "normalized_keys": sorted(list(normalized.keys())),
                "hint": "Use one of case=[simply_supported_point, simply_supported_udl, cantilever_point, cantilever_udl] and SI units.",
            },
        ) from exc
    except ValueError as exc:
        logger.warning("Domain validation error in dev tool call", extra={"tool_name": tool_name, "payload": payload, "error": str(exc)})
        raise HTTPException(status_code=422, detail={"message": str(exc), "tool": tool_name}) from exc

    raise HTTPException(status_code=404, detail=f"Unknown tool {tool_name}")


app.mount("/mcp", mcp.streamable_http_app())


def run() -> None:
    host = os.getenv("PHYSICS_MCP_HOST", "0.0.0.0")
    port = int(os.getenv("PHYSICS_MCP_PORT", "8080"))
    logger.info("Starting physics-mcp server", extra={"host": host, "port": port})
    uvicorn.run("mcp_server.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    run()
