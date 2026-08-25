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
    # Static aux data (not a pytree leaf): jitted code receiving a RunSpec-bearing
    # pytree re-traces per distinct run_id; see spec 260824 caveat.
    run_id: str | None = eqx.field(default=None, static=True)

    @classmethod
    def from_spec(cls, spec: Any) -> Any:
        """Create RunSpec from a specification object.

        Identity for already-built RunSpec; subclasses override to build from RunSpecification.

        Args:
            spec: A RunSpec instance or specification object.

        Returns:
            The spec unchanged (identity function at base class level).
        """
        return spec
