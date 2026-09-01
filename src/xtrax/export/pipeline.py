"""End-to-end export: plan -> traceable callable -> StableHLO -> artifact.

The logic lives here rather than in ``__init__.py`` because the coverage config
omits every ``*/__init__.py``; putting it there would make the package's
coverage gate measure nothing.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax

from xtrax.export.compile import CompileResult, compile_for_target
from xtrax.export.composer import build_traceable_callable
from xtrax.export.parity import ParityResult, verify_native_parity
from xtrax.export.safety import validate_export_safe
from xtrax.export.spirv import SpirvValidationResult
from xtrax.export.targets import NATIVE, WASM32, Target, VerificationLevel
from xtrax.stages.boundaries import AxisBoundary

__all__ = ["ExportResult", "export_pipeline"]


@dataclass(frozen=True)
class ExportResult:
    """One target's compiled artifact and whatever was established about it.

    Attributes:
        target: The target this was compiled for.
        path: The compiled artifact on disk. Execution and parity always go
            through this; a test wanting the real executed array calls
            ``run_native_vmfb(result.path, *concrete_inputs)``.
        vmfb_bytes: The artifact's bytes, read once from ``path``. Convenience
            only.
        size_bytes: Size of the artifact.
        spirv_bytes: Extracted SPIR-V keyed by executable name, or None.
        verification_level: How far this target's artifact is verified.
        verified: EXECUTED -> parity passed; VALIDATED -> validation passed;
            CODEGEN_ONLY -> always False, since nothing beyond compilation was
            established. Read ``verification_level`` to tell that apart from a
            genuine failure.
        parity: The parity comparison, for EXECUTED targets only.
        spirv_validation: The shader validation, for VALIDATED targets only.
        diagnostics: Notes worth surfacing, e.g. a StableHLO downgrade.
    """

    target: Target
    path: Path
    vmfb_bytes: bytes
    size_bytes: int
    spirv_bytes: dict[str, bytes] | None
    verification_level: VerificationLevel
    verified: bool
    parity: ParityResult | None
    spirv_validation: SpirvValidationResult | None
    diagnostics: tuple[str, ...]


class _StrippedSink:
    """A no-op standing in for a materializing sink during export.

    Preserves the original sink's ``ordered`` flag. That matters: the executor's
    branch selection reads ``boundary.sink.ordered``, so replacing a sink with
    bare None flips an ordered SafeMap axis off the ``jax.lax.map`` path onto
    ``safe_map(..., batch_size=...)`` -- a different lowering, and one that
    raises when the axis's cardinality is not divisible by the batch size. A
    working configuration would then crash at export.

    Calling it does nothing and returns None, so no io_callback reaches the
    trace, which is the whole point of stripping.
    """

    __slots__ = ("ordered",)

    def __init__(self, ordered: bool) -> None:
        self.ordered = ordered

    def __call__(self, x: Any) -> None:
        """Discard ``x``; the exported program carries the values as output."""
        return None


def _boundaries_for_export(
    axis_boundaries: Mapping[str, AxisBoundary] | None,
) -> Mapping[str, AxisBoundary] | None:
    """Return a view of ``axis_boundaries`` with materializing sinks stripped.

    Every axis that does not materialize is passed through by identity, not
    reconstructed, so callers keep object identity for untouched axes.

    Args:
        axis_boundaries: The caller's boundaries, or None.

    Returns:
        The same mapping when nothing materializes, else a new mapping where each
        materializing axis's ``sink`` is a ``_StrippedSink``.
    """
    if not axis_boundaries:
        return axis_boundaries

    result: dict[str, AxisBoundary] = {}
    changed = False
    for name, boundary in axis_boundaries.items():
        if getattr(boundary, "materialize", False) and boundary.sink is not None:
            # dataclasses.replace, not a field-by-field rebuild: it preserves any
            # field added later and does not downcast an AxisBoundary subclass.
            # eqx.tree_at cannot be used -- every field is static, i.e. aux_data
            # rather than a pytree leaf.
            stripped = _StrippedSink(bool(getattr(boundary.sink, "ordered", False)))
            result[name] = dataclasses.replace(boundary, sink=stripped)
            changed = True
        else:
            result[name] = boundary
    return result if changed else axis_boundaries


def _verified_for(
    level: VerificationLevel,
    parity: ParityResult | None,
    validation: SpirvValidationResult | None,
) -> bool:
    """Resolve ``ExportResult.verified`` for one verification level."""
    if level is VerificationLevel.EXECUTED:
        return bool(parity is not None and parity.passed)
    if level is VerificationLevel.VALIDATED:
        return bool(validation is not None and validation.valid)
    return False


def _diagnostics_for(compiled: CompileResult) -> tuple[str, ...]:
    """Collect notes worth surfacing from one compile."""
    notes: list[str] = []
    if compiled.downgraded_stablehlo:
        notes.append(
            f"{compiled.target.name}: StableHLO downgraded to a portable artifact "
            f"before IREE accepted it"
        )
    return tuple(notes)


def export_pipeline(
    fn: Callable[..., Any],
    plan: Any,
    abstract_inputs: Sequence[Any],
    concrete_inputs: Sequence[Any] | None = None,
    *,
    axis_boundaries: Mapping[str, AxisBoundary] | None = None,
    targets: Sequence[Target] = (NATIVE, WASM32),
    request_features: frozenset[str] = frozenset(),
    scan_init: Any = None,
    reference_fn: Callable[[Sequence[Any]], Any] | None = None,
) -> dict[str, ExportResult]:
    """Export a planned pipeline to one artifact per target.

    All-or-nothing across ``targets``: they are processed in order, and the first
    target-level exception aborts the whole call. No partial dict is returned.

    Args:
        fn: Per-element function, or a Scan transition.
        plan: A BatchPlan.
        abstract_inputs: Abstract inputs to trace with.
        concrete_inputs: Concrete inputs, required if any target is EXECUTED.
        axis_boundaries: Name-keyed boundaries. Axes declared ``materialize=True``
            have their sink stripped before tracing.
        targets: Targets to compile for.
        request_features: Device features unlocking a target's optional dtypes.
        scan_init: Initial carry for a Scan axis.
        reference_fn: An independently-computed oracle over ``concrete_inputs``,
            required if any target is EXECUTED. Must not be built from the
            callable under test -- see ``verify_native_parity``.

    Returns:
        A dict keyed by target name.

    Raises:
        ValueError: If an EXECUTED target is requested without
            ``concrete_inputs`` or without ``reference_fn``.
        PlanTopologyError: Propagated from the safety gate.
        DtypeNotSupportedError: If a leaf's dtype is rejected by a target.
        CompileError: If IREE rejects the module, or is not installed.
    """
    targets = tuple(targets)
    executed = [t for t in targets if t.verification_level is VerificationLevel.EXECUTED]

    if executed and concrete_inputs is None:
        names = ", ".join(t.name for t in executed)
        msg = (
            f"concrete_inputs is required because target(s) {names} are EXECUTED "
            f"and must be run to be verified."
        )
        raise ValueError(msg)
    if executed and reference_fn is None:
        names = ", ".join(t.name for t in executed)
        msg = (
            f"reference_fn is required because target(s) {names} are EXECUTED. "
            f"It must be an independently-computed oracle over concrete_inputs -- "
            f"passing jax.jit(build_traceable_callable(...)) compares the composed "
            f"callable against itself and verifies nothing."
        )
        raise ValueError(msg)

    decisions = list(plan.decisions)
    results: dict[str, ExportResult] = {}

    for target in targets:
        validate_export_safe(
            decisions,
            axis_boundaries or {},
            abstract_inputs,
            fn,
            target,
            request_features=request_features,
        )

        # Strip only after the gate has confirmed every materializing axis is
        # well-formed. The composer never sees `materialize` itself.
        boundaries = _boundaries_for_export(axis_boundaries)
        callable_ = build_traceable_callable(fn, plan, boundaries, scan_init=scan_init)
        exported = jax.export.export(jax.jit(callable_))(*abstract_inputs)
        compiled = compile_for_target(exported.mlir_module(), target)

        parity: ParityResult | None = None
        if target.verification_level is VerificationLevel.EXECUTED:
            assert concrete_inputs is not None  # noqa: S101 - guarded above
            assert reference_fn is not None  # noqa: S101 - guarded above
            parity = verify_native_parity(
                reference_fn(concrete_inputs),
                compiled.path,
                concrete_inputs,
            )

        results[target.name] = ExportResult(
            target=target,
            path=compiled.path,
            vmfb_bytes=compiled.path.read_bytes(),
            size_bytes=compiled.size_bytes,
            spirv_bytes=compiled.spirv_bytes,
            verification_level=target.verification_level,
            verified=_verified_for(target.verification_level, parity, None),
            parity=parity,
            spirv_validation=None,
            diagnostics=_diagnostics_for(compiled),
        )

    return results
