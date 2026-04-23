from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class BeamCase(str, Enum):
    SIMPLY_SUPPORTED_POINT = "simply_supported_point"
    SIMPLY_SUPPORTED_UDL = "simply_supported_udl"
    CANTILEVER_POINT = "cantilever_point"
    CANTILEVER_UDL = "cantilever_udl"


class BeamInput(BaseModel):
    case: BeamCase
    length_m: float = Field(..., gt=0)
    youngs_modulus_pa: float = Field(..., gt=0)
    second_moment_m4: float = Field(..., gt=0)
    point_load_n: float | None = Field(default=None, gt=0)
    point_load_position_m: float | None = Field(default=None, ge=0)
    udl_n_per_m: float | None = Field(default=None, gt=0)
    samples: int = Field(default=41, ge=11, le=401)

    @field_validator("case", mode="before")
    @classmethod
    def normalize_case_aliases(cls, value: BeamCase | str) -> BeamCase | str:
        if isinstance(value, str):
            aliases = {
                "cantilever": BeamCase.CANTILEVER_POINT.value,
                "simply_supported": BeamCase.SIMPLY_SUPPORTED_POINT.value,
            }
            return aliases.get(value.strip().lower(), value)
        return value

    @model_validator(mode="after")
    def validate_case_loads(self) -> "BeamInput":
        if "point" in self.case:
            if self.point_load_n is None:
                raise ValueError("point_load_n is required for point-load cases")
            if self.point_load_position_m is None:
                if self.case == BeamCase.CANTILEVER_POINT:
                    self.point_load_position_m = self.length_m
                else:
                    raise ValueError("point_load_position_m is required for simply supported point-load case")
            if not (0 <= self.point_load_position_m <= self.length_m):
                raise ValueError("point_load_position_m must be between 0 and length_m")
            if self.case == BeamCase.CANTILEVER_POINT and self.point_load_position_m != self.length_m:
                raise ValueError("v0.1 cantilever point load supports only tip load at x=length_m")
        if "udl" in self.case and self.udl_n_per_m is None:
            raise ValueError("udl_n_per_m is required for UDL cases")
        return self


class BeamResult(BaseModel):
    case: BeamCase
    reactions_n: dict[str, float]
    max_bending_moment_nm: float
    max_deflection_m: float
    x_m: list[float]
    shear_n: list[float]
    moment_nm: list[float]
    deflection_m: list[float]
    warnings: list[str]
    assumptions: list[str]
