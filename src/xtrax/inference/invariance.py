"""Check a per-axis varying-input declaration against jaxpr data dependence.

The declaration says which named inputs vary along a mapped axis. Every other
named input is invariant along that axis. A value is invariant when it reads
no varying input. Invariant inputs may combine freely with varying ones
(weights times per-element data is the common case); what the jaxpr can
check is a claim that specific outputs are invariant, which is the claim a
hoist relies on.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax

from xtrax.inference.cse import trace_input_dependence
from xtrax.inference.errors import InputInvarianceError
from xtrax.tiling.plan import AxisSpec

__all__ = [
    "AxisInvariance",
    "InvarianceReport",
    "verify_axis_invariance",
]


@dataclass(frozen=True)
class AxisInvariance:
    """Invariant outputs and intermediates along one mapped axis.

    Attributes:
        axis: Axis name (``AxisSpec.name``).
        varying_inputs: Inputs declared to vary along this axis.
        invariant_intermediates: Primitives that produce a value reading no
            varying input, in jaxpr-walker order, duplicates removed.
        invariant_outputs: Indices of outputs that read no varying input.
    """

    axis: str
    varying_inputs: tuple[str, ...]
    invariant_intermediates: tuple[str, ...]
    invariant_outputs: tuple[int, ...]


@dataclass(frozen=True)
class InvarianceReport:
    """Per-axis invariance for one traced function."""

    by_axis: tuple[AxisInvariance, ...]

    def axis(self, name: str) -> AxisInvariance:
        """Return the report for ``name``.

        Raises:
            KeyError: No entry uses that axis name. The message lists the
                axes that are present.
        """
        for item in self.by_axis:
            if item.axis == name:
                return item
        known = ", ".join(repr(item.axis) for item in self.by_axis) or "(none)"
        raise KeyError(f"no invariance report for axis {name!r}; known axes: {known}")


def verify_axis_invariance(
    fn: Any,
    example_inputs: Mapping[str, Any],
    specs: Sequence[AxisSpec],
    claimed_invariant_outputs: Mapping[str, Sequence[int]] | None = None,
) -> InvarianceReport:
    """Trace ``fn``, report what is invariant per axis, and check output claims.

    ``example_inputs`` is ordered like ``fn``'s positional arguments. Tracing
    uses ``jax.make_jaxpr`` and :func:`xtrax.inference.cse.trace_input_dependence`,
    which walks callees inside ``checkpoint`` / ``custom_jvp`` / ``jit``.

    Args:
        fn: Pure JAX function. Positional parameters match ``example_inputs``.
        example_inputs: Name to example array, one entry per positional argument.
        specs: Axis specs whose ``varying_inputs`` declare the varying names.
            Every other example input is invariant along that axis.
        claimed_invariant_outputs: Optional map from axis name to the indices
            of ``fn``'s flattened outputs the caller claims are invariant along
            that axis (for example, outputs it intends to hoist).

    Returns:
        The outputs and intermediates that read no varying input, per axis.

    Raises:
        ValueError: A ``varying_inputs`` entry is not a key of ``example_inputs``,
            a claim names an axis absent from ``specs``, or a claimed output
            index is out of range. The message names the offending value.
        InputInvarianceError: A claimed-invariant output reads an input that
            varies along that axis. The error names the axis, the output index,
            and the varying input.
    """
    names = tuple(example_inputs)
    name_index = {name: index for index, name in enumerate(names)}
    for spec in specs:
        for name in spec.varying_inputs:
            if name not in name_index:
                known = ", ".join(repr(n) for n in names) or "(none)"
                raise ValueError(
                    f"unknown input {name!r} on axis {spec.name!r}; known inputs: {known}"
                )
    claims = dict(claimed_invariant_outputs or {})
    spec_names = [spec.name for spec in specs]
    for axis_name in claims:
        if axis_name not in spec_names:
            known = ", ".join(repr(n) for n in spec_names) or "(none)"
            raise ValueError(f"claim names unknown axis {axis_name!r}; known axes: {known}")

    closed = jax.make_jaxpr(fn)(*example_inputs.values())
    traced = trace_input_dependence(closed)
    n_outputs = len(traced.outputs)
    reports: list[AxisInvariance] = []
    for spec in specs:
        varying_names = tuple(spec.varying_inputs)
        varying_idx = frozenset(name_index[name] for name in varying_names)
        for output_index in claims.get(spec.name, ()):
            if not 0 <= output_index < n_outputs:
                raise ValueError(
                    f"claimed output {output_index} on axis {spec.name!r} is out of range; "
                    f"fn has {n_outputs} outputs"
                )
            reads = traced.outputs[output_index] & varying_idx
            if reads:
                raise InputInvarianceError(spec.name, output_index, names[min(reads)])
        intermediates = tuple(
            dict.fromkeys(
                primitive for primitive, dep in traced.intermediates if not (dep & varying_idx)
            )
        )
        outputs = tuple(
            index for index, dep in enumerate(traced.outputs) if not (dep & varying_idx)
        )
        reports.append(
            AxisInvariance(
                axis=spec.name,
                varying_inputs=varying_names,
                invariant_intermediates=intermediates,
                invariant_outputs=outputs,
            )
        )
    return InvarianceReport(by_axis=tuple(reports))
