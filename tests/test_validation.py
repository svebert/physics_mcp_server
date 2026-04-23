from __future__ import annotations

import pytest
from pydantic import ValidationError

from physics_core.models import BeamCase, BeamInput


def test_negative_length_rejected() -> None:
    with pytest.raises(ValidationError):
        BeamInput(
            case=BeamCase.CANTILEVER_UDL,
            length_m=-1.0,
            udl_n_per_m=1.0,
            youngs_modulus_pa=1.0,
            second_moment_m4=1.0,
        )


def test_missing_point_load_value_rejected() -> None:
    with pytest.raises(ValidationError):
        BeamInput(
            case=BeamCase.SIMPLY_SUPPORTED_POINT,
            length_m=2.0,
            point_load_position_m=1.0,
            youngs_modulus_pa=1.0,
            second_moment_m4=1.0,
        )




def test_simply_supported_point_defaults_to_midspan_when_position_missing() -> None:
    payload = {
        "case": "simply_supported_point",
        "length_m": 6.0,
        "point_load_n": 12_000.0,
        "youngs_modulus_pa": 210e9,
        "second_moment_m4": 8.5e-6,
    }
    beam = BeamInput.model_validate(payload)
    assert beam.point_load_position_m == 3.0

def test_cantilever_point_requires_tip_load_in_v01() -> None:
    with pytest.raises(ValidationError):
        BeamInput(
            case=BeamCase.CANTILEVER_POINT,
            length_m=2.0,
            point_load_n=100.0,
            point_load_position_m=1.0,
            youngs_modulus_pa=1.0,
            second_moment_m4=1.0,
        )


def test_case_alias_cantilever_maps_to_point_case() -> None:
    payload = {
        "case": "cantilever",
        "length_m": 2.0,
        "point_load_n": 100.0,
        "youngs_modulus_pa": 1.0,
        "second_moment_m4": 1.0,
    }
    beam = BeamInput.model_validate(payload)
    assert beam.case == BeamCase.CANTILEVER_POINT
    assert beam.point_load_position_m == beam.length_m


def test_case_alias_simply_supported_maps_to_point_case() -> None:
    payload = {
        "case": "simply_supported",
        "length_m": 2.0,
        "point_load_n": 100.0,
        "point_load_position_m": 1.0,
        "youngs_modulus_pa": 1.0,
        "second_moment_m4": 1.0,
    }
    beam = BeamInput.model_validate(payload)
    assert beam.case == BeamCase.SIMPLY_SUPPORTED_POINT
