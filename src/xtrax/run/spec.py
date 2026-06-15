"""Base execution config for xtrax run module."""

from __future__ import annotations

from typing import Any

import equinox as eqx

from xtrax.stages.boundaries import AxisBoundary
from xtrax.tiling import AxisSpec


class RunSpec(eqx.Module):
    """Base execution config. aminx.run.RunSpec (eqx.Module) extends this."""

    seed: int
    axes: list[AxisSpec]
    carry_specs: dict[str, Any]
    boundaries: list[AxisBoundary] | None
