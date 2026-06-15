"""Base execution config for xtrax run module."""

from __future__ import annotations

from typing import Any

import equinox as eqx

from xtrax.stages.boundaries import AxisBoundary
from xtrax.tiling import AxisSpec, CarrySpec


class RunSpec(eqx.Module):
    """Base execution config. aminx.run.RunSpec (eqx.Module) extends this."""

    seed: int
    axes: list[AxisSpec]
    carry_specs: list[CarrySpec] = eqx.field(default_factory=list)
    boundaries: list[AxisBoundary] | None = None
