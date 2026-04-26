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


def test_solve_tool_accepts_flat_keyword_arguments() -> None:
    result = solve_beam_case_tool(
        case="simply_supported_point",
        length_m=6.0,
        youngs_modulus_pa=210e9,
        second_moment_m4=8.5e-6,
        point_load_n=12_000.0,
        point_load_position_m=3.0,
    )
    assert result["case"] == "simply_supported_point"
    assert result["reactions_n"]["left"] == 6000.0
    assert result["reactions_n"]["right"] == 6000.0



def test_solve_tool_accepts_kwargs_wrapper_argument() -> None:
    result = solve_beam_case_tool(
        kwargs={
            "case": "simply_supported_point",
            "length_m": 6.0,
            "youngs_modulus_pa": 210e9,
            "second_moment_m4": 8.5e-6,
            "point_load_n": 12_000.0,
            "point_load_position_m": 3.0,
        }
    )
    assert result["case"] == "simply_supported_point"
    assert result["reactions_n"]["left"] == 6000.0


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


def test_dev_tool_accepts_german_payload_keys_and_case_alias() -> None:
    client = TestClient(app)
    response = client.post(
        "/dev/tools/solve_beam_case",
        json={
            "lastfall": "einfach_gelagert_punktlast",
            "laenge": 6.0,
            "elastizitaetsmodul": 210e9,
            "flaechentraegheitsmoment": 8.5e-6,
            "punktlast": 12_000.0,
            "lastposition": 3.0,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["case"] == "simply_supported_point"
    assert body["max_bending_moment_nm"] == 18_000.0


def test_dev_tool_accepts_nested_input_payload() -> None:
    client = TestClient(app)
    response = client.post(
        "/dev/tools/solve_beam_case",
        json={
            "input": {
                "case": "simply_supported_udl",
                "length": 5.0,
                "udl": 1000.0,
                "e": 200e9,
                "i": 8e-6,
            }
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["case"] == "simply_supported_udl"


def test_dev_tool_422_contains_debuggable_error_detail() -> None:
    client = TestClient(app)
    response = client.post("/dev/tools/solve_beam_case", json={"foo": "bar"})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["message"] == "Invalid tool payload"
    assert "received_keys" in detail
    assert "normalized_keys" in detail



def test_dev_tool_accepts_nested_kwargs_payload() -> None:
    client = TestClient(app)
    response = client.post(
        "/dev/tools/solve_beam_case",
        json={
            "kwargs": {
                "case": "simply_supported_point",
                "length_m": 6.0,
                "youngs_modulus_pa": 210e9,
                "second_moment_m4": 8.5e-6,
                "point_load_n": 12_000.0,
                "point_load_position_m": 3.0,
            }
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["case"] == "simply_supported_point"
    assert body["reactions_n"]["left"] == 6000.0


def test_dev_tool_accepts_nested_payload_payload() -> None:
    client = TestClient(app)
    response = client.post(
        "/dev/tools/solve_beam_case",
        json={
            "payload": {
                "case": "simply_supported_point",
                "length_m": 6.0,
                "youngs_modulus_pa": 210e9,
                "second_moment_m4": 8.5e-6,
                "point_load_n": 12_000.0,
                "point_load_position_m": 3.0,
            }
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["case"] == "simply_supported_point"
    assert body["reactions_n"]["right"] == 6000.0

def test_mcp_endpoint_is_not_redirected() -> None:
    mount_routes = [route for route in app.routes if getattr(route, "path", None) == ""]
    assert mount_routes
    mounted_app = mount_routes[0].app
    mounted_paths = [route.path for route in mounted_app.routes]
    assert "/mcp" in mounted_paths
