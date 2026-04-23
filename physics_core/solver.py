from __future__ import annotations

from typing import Callable

from physics_core.assumptions import ASSUMPTIONS_V1
from physics_core.models import BeamCase, BeamInput, BeamResult


def _linspace(length_m: float, samples: int) -> list[float]:
    step = length_m / (samples - 1)
    return [i * step for i in range(samples)]


def _solve_simply_supported_point(inp: BeamInput) -> BeamResult:
    l = inp.length_m
    p = float(inp.point_load_n)
    a = float(inp.point_load_position_m)
    b = l - a
    e = inp.youngs_modulus_pa
    i = inp.second_moment_m4

    ra = p * b / l
    rb = p * a / l
    x = _linspace(l, inp.samples)

    shear = [ra if xi < a else ra - p for xi in x]
    moment = [ra * xi if xi <= a else ra * xi - p * (xi - a) for xi in x]

    def y(xi: float) -> float:
        if xi <= a:
            return -(p * b * xi * (l**2 - b**2 - xi**2)) / (6 * l * e * i)
        z = l - xi
        return -(p * a * z * (l**2 - a**2 - z**2)) / (6 * l * e * i)

    deflection = [y(xi) for xi in x]
    max_m = (p * a * b) / l
    max_d = min(deflection, key=lambda v: v)

    return BeamResult(
        case=inp.case,
        reactions_n={"left": ra, "right": rb},
        max_bending_moment_nm=max_m,
        max_deflection_m=max_d,
        x_m=x,
        shear_n=shear,
        moment_nm=moment,
        deflection_m=deflection,
        warnings=["Sign convention: downward deflection is negative."],
        assumptions=ASSUMPTIONS_V1,
    )


def _solve_simply_supported_udl(inp: BeamInput) -> BeamResult:
    l = inp.length_m
    w = float(inp.udl_n_per_m)
    e = inp.youngs_modulus_pa
    i = inp.second_moment_m4
    x = _linspace(l, inp.samples)

    ra = rb = w * l / 2
    shear = [ra - w * xi for xi in x]
    moment = [ra * xi - (w * xi**2) / 2 for xi in x]
    deflection = [-(w * xi * (l**3 - 2 * l * xi**2 + xi**3)) / (24 * e * i) for xi in x]

    max_m = w * l**2 / 8
    max_d = min(deflection, key=lambda v: v)

    return BeamResult(
        case=inp.case,
        reactions_n={"left": ra, "right": rb},
        max_bending_moment_nm=max_m,
        max_deflection_m=max_d,
        x_m=x,
        shear_n=shear,
        moment_nm=moment,
        deflection_m=deflection,
        warnings=["UDL is assumed to act over the full span in v0.1."],
        assumptions=ASSUMPTIONS_V1,
    )


def _solve_cantilever_point(inp: BeamInput) -> BeamResult:
    l = inp.length_m
    p = float(inp.point_load_n)
    e = inp.youngs_modulus_pa
    i = inp.second_moment_m4
    x = _linspace(l, inp.samples)

    shear = [-p for _ in x]
    moment = [-p * (l - xi) for xi in x]
    deflection = [-(p * xi**2 * (3 * l - xi)) / (6 * e * i) for xi in x]

    return BeamResult(
        case=inp.case,
        reactions_n={"fixed_vertical": p, "fixed_moment": p * l},
        max_bending_moment_nm=p * l,
        max_deflection_m=min(deflection, key=lambda v: v),
        x_m=x,
        shear_n=shear,
        moment_nm=moment,
        deflection_m=deflection,
        warnings=["Point load location is fixed at free tip for v0.1."],
        assumptions=ASSUMPTIONS_V1,
    )


def _solve_cantilever_udl(inp: BeamInput) -> BeamResult:
    l = inp.length_m
    w = float(inp.udl_n_per_m)
    e = inp.youngs_modulus_pa
    i = inp.second_moment_m4
    x = _linspace(l, inp.samples)

    shear = [-w * (l - xi) for xi in x]
    moment = [-(w * (l - xi) ** 2) / 2 for xi in x]
    deflection = [-(w * xi**2 * (6 * l**2 - 4 * l * xi + xi**2)) / (24 * e * i) for xi in x]

    return BeamResult(
        case=inp.case,
        reactions_n={"fixed_vertical": w * l, "fixed_moment": w * l**2 / 2},
        max_bending_moment_nm=w * l**2 / 2,
        max_deflection_m=min(deflection, key=lambda v: v),
        x_m=x,
        shear_n=shear,
        moment_nm=moment,
        deflection_m=deflection,
        warnings=["UDL is assumed to act over full cantilever length."],
        assumptions=ASSUMPTIONS_V1,
    )


SOLVERS: dict[BeamCase, Callable[[BeamInput], BeamResult]] = {
    BeamCase.SIMPLY_SUPPORTED_POINT: _solve_simply_supported_point,
    BeamCase.SIMPLY_SUPPORTED_UDL: _solve_simply_supported_udl,
    BeamCase.CANTILEVER_POINT: _solve_cantilever_point,
    BeamCase.CANTILEVER_UDL: _solve_cantilever_udl,
}


def solve_beam_case(data: BeamInput) -> BeamResult:
    solver = SOLVERS[data.case]
    return solver(data)


def get_supported_cases() -> list[str]:
    return [case.value for case in BeamCase]
