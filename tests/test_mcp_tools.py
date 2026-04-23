from __future__ import annotations
from fastapi.testclient import TestClient
from mcp_server.app import (
    app,

    get_model_assumptions_tool,
    get_supported_cases_tool,
    solve_beam_case_tool,
)


def test_supported_cases_tool() -> None:
    cases = get_supported_cases_tool()
    assert "simply_supported_point" in cases
    assert "cantilever_udl" in cases


def test_assumptions_tool() -> None:
    assumptions = get_model_assumptions_tool()
    assert any("Euler-Bernoulli" in item for item in assumptions)


def test_solve_tool_integration() -> None:
    payload = {
        "case": "simply_supported_udl",
        "length_m": 5.0,
        "udl_n_per_m": 1000.0,
        "youngs_modulus_pa": 200e9,
        "second_moment_m4": 8e-6,
    }
    result = solve_beam_case_tool(payload)
    assert result["case"] == "simply_supported_udl"
    assert result["max_bending_moment_nm"] > 0
    assert len(result["x_m"]) == len(result["moment_nm"]) == len(result["deflection_m"])

def test_dev_tool_returns_422_on_invalid_payload() -> None:
    client = TestClient(app)
    response = client.post("/dev/tools/solve_beam_case", json={"case": "simply_supported_point"})
    assert response.status_code == 422


def test_dev_tool_accepts_shorthand_payload_and_infers_case() -> None:
    client = TestClient(app)
    response = client.post(
        "/dev/tools/solve_beam_case",
        json={"l": 6.0, "e": 210e9, "i": 8.5e-6, "point_load_kn": 12.0},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["case"] == "simply_supported_point"
    assert body["reactions_n"]["left"] == 6000.0
    assert body["reactions_n"]["right"] == 6000.0
