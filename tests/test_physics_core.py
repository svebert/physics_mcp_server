from __future__ import annotations

import math

from physics_core.models import BeamCase, BeamInput
from physics_core.solver import solve_beam_case


def test_simply_supported_point_center_load_golden() -> None:
    data = BeamInput(
        case=BeamCase.SIMPLY_SUPPORTED_POINT,
        length_m=4.0,
        point_load_n=10_000.0,
        point_load_position_m=2.0,
        youngs_modulus_pa=200e9,
        second_moment_m4=8e-6,
    )
    out = solve_beam_case(data)

    assert math.isclose(out.reactions_n["left"], 5_000.0, rel_tol=1e-6)
    assert math.isclose(out.reactions_n["right"], 5_000.0, rel_tol=1e-6)
    assert math.isclose(out.max_bending_moment_nm, 10_000.0, rel_tol=1e-3)
    assert math.isclose(abs(out.max_deflection_m), 0.0083333333, rel_tol=0.02)


def test_cantilever_udl_golden() -> None:
    data = BeamInput(
        case=BeamCase.CANTILEVER_UDL,
        length_m=2.0,
        udl_n_per_m=2000.0,
        youngs_modulus_pa=210e9,
        second_moment_m4=5e-6,
    )
    out = solve_beam_case(data)

    assert math.isclose(out.max_bending_moment_nm, 4000.0, rel_tol=1e-6)
    expected_deflection = (2000 * (2**4)) / (8 * 210e9 * 5e-6)
    assert math.isclose(abs(out.max_deflection_m), expected_deflection, rel_tol=0.02)


def test_simply_supported_point_max_moment_analytic_off_grid_position() -> None:
    data = BeamInput(
        case=BeamCase.SIMPLY_SUPPORTED_POINT,
        length_m=10.0,
        point_load_n=1000.0,
        point_load_position_m=3.33,
        samples=11,
        youngs_modulus_pa=200e9,
        second_moment_m4=8e-6,
    )
    out = solve_beam_case(data)

    expected_max_moment = (1000.0 * 3.33 * (10.0 - 3.33)) / 10.0
    sampled_peak = max(abs(v) for v in out.moment_nm)

    assert math.isclose(out.max_bending_moment_nm, expected_max_moment, rel_tol=1e-9)
    assert out.max_bending_moment_nm > sampled_peak
