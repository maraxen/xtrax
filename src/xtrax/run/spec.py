"""Base execution config for xtrax run module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx

from xtrax.stages.boundaries import AxisBoundary
from xtrax.tiling import AxisSpec, CarrySpec


class RunSpec(eqx.Module):
    """Base execution config. aminx.run.RunSpec (eqx.Module) extends this.

    ``output_root``, ``device_count``, ``precision`` and ``shard_lineage`` are
    optional execution settings (debt #2523). Together with :class:`~xtrax.run.sink.SinkSpec`
    they are the output contract: ``output_root`` is where a derived sink writes
    when the caller does not pass an explicit directory. All four are static, so
    a jitted caller re-traces when they change, and none of them is a pytree leaf.
    ``shard_lineage`` is an ordered tuple of shard identifiers from the root of
    the lineage to this run (``None`` when the run is not a shard).
    """

    seed: int
    axes: list[AxisSpec]
    carry_specs: list[CarrySpec] = eqx.field(default_factory=list)
    boundaries: list[AxisBoundary] | None = None
    # Static aux data (not a pytree leaf): jitted code receiving a RunSpec-bearing
    # pytree re-traces per distinct run_id; see spec 260824 caveat.
    run_id: str | None = eqx.field(default=None, static=True)
    output_root: Path | None = eqx.field(default=None, static=True)
    device_count: int | None = eqx.field(default=None, static=True)
    precision: str | None = eqx.field(default=None, static=True)
    shard_lineage: tuple[str, ...] | None = eqx.field(default=None, static=True)

    def __check_init__(self) -> None:
        if self.device_count is not None and (
            isinstance(self.device_count, bool)
            or not isinstance(self.device_count, int)
            or self.device_count < 1
        ):
            msg = f"RunSpec.device_count must be a positive int, got {self.device_count!r}"
            raise ValueError(msg)
        if self.precision is not None and not isinstance(self.precision, str):
            msg = f"RunSpec.precision must be a str or None, got {type(self.precision).__name__}"
            raise TypeError(msg)
        if self.output_root is not None and not isinstance(self.output_root, Path):
            msg = (
                "RunSpec.output_root must be a pathlib.Path or None, "
                f"got {type(self.output_root).__name__}"
            )
            raise TypeError(msg)
        if self.shard_lineage is not None and (
            not isinstance(self.shard_lineage, tuple)
            or not all(isinstance(part, str) for part in self.shard_lineage)
        ):
            msg = "RunSpec.shard_lineage must be a tuple of str or None"
            raise TypeError(msg)

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
