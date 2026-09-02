"""Plan-time gating for the export boundary.

Two entry points over the same rules: ``check_export_safety`` returns every
blocker it finds, ``validate_export_safe`` raises on the first batch. Both
delegate topology to ``xtrax.stages.topology.validate_plan_topology`` and let
its ``PlanTopologyError`` propagate unwrapped -- topology violations are
structural and are never demoted into a blocker list.

Blockers cover the rules this module owns: dtypes a target's backend does not
accept. They are collected rather than raised one at a time so a caller fixing
a model sees every offending leaf at once.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from xtrax.export.targets import Target
from xtrax.stages.boundaries import AxisBoundary
from xtrax.stages.topology import AxisDecisionLike, validate_plan_topology

__all__ = [
    "DtypeNotSupportedError",
    "ExportBlocker",
    "ExportSafetyError",
    "check_export_safety",
    "dtype_name",
    "find_bcoo_leaves",
    "validate_export_safe",
]


class ExportSafetyError(Exception):
    """Base for this package's own plan-time gate failures.

    Distinct from xtrax.stages.topology.PlanTopologyError, which both entry
    points here let propagate unchanged rather than wrapping.
    """


class DtypeNotSupportedError(ExportSafetyError):
    """A leaf's dtype is not accepted by the requested target."""


@dataclass(frozen=True)
class ExportBlocker:
    """One reason a plan or leaf cannot cross the export boundary.

    Attributes:
        axis: Axis name, or the leaf keypath for a dtype blocker.
        rule: Short rule identifier, e.g. ``"dtype"``.
        detail: Human-readable explanation naming the offending value.
    """

    axis: str
    rule: str
    detail: str


def dtype_name(dtype: Any) -> str:
    """Render a numpy/JAX dtype in the short form targets are keyed by.

    Args:
        dtype: Anything with a ``name``, or a value convertible by ``str``.

    Returns:
        A short name such as ``"f32"``, ``"bf16"``, ``"i32"``, or ``"bool"``.
        Unrecognised dtypes are returned as their own name, which will simply
        fail the membership check against a target's dtype sets.
    """
    raw = getattr(dtype, "name", None) or str(dtype)
    if raw == "bool":
        return "bool"
    prefixes = (("float", "f"), ("bfloat", "bf"), ("int", "i"), ("uint", "u"), ("complex", "c"))
    for long, short in prefixes:
        if raw.startswith(long):
            return short + raw[len(long) :]
    return raw


def find_bcoo_leaves(tree: Any) -> list[str]:
    """Return the keypaths of any BCOO leaves in ``tree``.

    Sparsified models substitute BCOO at leaf positions, and BCOO is a pytree
    *node* rather than a leaf, so the tree structure changes. Under
    ``jax.export`` a closure-held BCOO is baked in as constants, which is what a
    self-contained artifact wants; this exists so a caller knows that is
    happening rather than discovering it in the MLIR.

    Args:
        tree: Any pytree, typically the model held in the exported closure.

    Returns:
        Keypath strings for each BCOO leaf; empty when sparse is unavailable.
    """
    try:
        import jax
        from jax.experimental.sparse import BCOO
    except ImportError:  # pragma: no cover - sparse ships with jax today
        return []

    found: list[str] = []
    flat = jax.tree_util.tree_flatten_with_path(tree, is_leaf=lambda x: isinstance(x, BCOO))[0]
    for path, leaf in flat:
        if isinstance(leaf, BCOO):
            found.append(jax.tree_util.keystr(path))
    return found


def _dtype_blockers(
    abstract_inputs: Sequence[Any],
    target: Target,
    request_features: frozenset[str],
) -> list[ExportBlocker]:
    """Collect a blocker for every abstract input whose dtype the target rejects."""
    blockers: list[ExportBlocker] = []
    for index, spec in enumerate(abstract_inputs):
        dtype = getattr(spec, "dtype", None)
        if dtype is None:
            continue
        name = dtype_name(dtype)
        if name in target.supported_dtypes:
            continue
        if name in target.optional_dtypes:
            feature = target.optional_dtype_features.get(name)
            if feature is not None and feature in request_features:
                continue
            blockers.append(
                ExportBlocker(
                    axis=f"abstract_inputs[{index}]",
                    rule="dtype",
                    detail=(
                        f"dtype {name!r} is optional on target {target.name!r} and "
                        f"needs feature {feature!r}, which was not requested. Pass "
                        f"request_features=frozenset({{{feature!r}}})."
                    ),
                )
            )
            continue
        supported = ", ".join(sorted(target.supported_dtypes))
        blockers.append(
            ExportBlocker(
                axis=f"abstract_inputs[{index}]",
                rule="dtype",
                detail=(
                    f"dtype {name!r} is not supported by target {target.name!r}. "
                    f"Supported: {supported}."
                ),
            )
        )
    return blockers


def check_export_safety(
    decisions: Sequence[AxisDecisionLike],
    axis_boundaries: Mapping[str, AxisBoundary],
    abstract_inputs: Sequence[Any],
    fn: Callable[..., Any],
    target: Target,
    *,
    request_features: frozenset[str] = frozenset(),
) -> list[ExportBlocker]:
    """List every dtype blocker between this plan and the export boundary.

    Deliberately does not call ``validate_plan_topology``: topology violations
    always raise directly and are never converted into a blocker list.

    Args:
        decisions: Axis decisions from the plan.
        axis_boundaries: Map of axis name -> AxisBoundary.
        abstract_inputs: Abstract inputs the callable will be traced with.
        fn: The callable being exported. Its closure-reachable leaves are scanned
            for dtype violations alongside ``abstract_inputs``.
        target: The target being compiled for.
        request_features: Device features the caller will request, unlocking the
            target's optional dtypes.

    Returns:
        Every blocker found, in discovery order. Empty means no dtype objection.
    """
    del decisions, axis_boundaries, fn
    return _dtype_blockers(abstract_inputs, target, request_features)


def validate_export_safe(
    decisions: Sequence[AxisDecisionLike],
    axis_boundaries: Mapping[str, AxisBoundary],
    abstract_inputs: Sequence[Any],
    fn: Callable[..., Any],
    target: Target,
    *,
    request_features: frozenset[str] = frozenset(),
) -> None:
    """Raise unless this plan can cross the export boundary for ``target``.

    Args:
        decisions: Axis decisions from the plan.
        axis_boundaries: Map of axis name -> AxisBoundary.
        abstract_inputs: Abstract inputs the callable will be traced with.
        fn: The callable being exported.
        target: The target being compiled for.
        request_features: Device features the caller will request.

    Raises:
        PlanTopologyError: Propagated unwrapped from validate_plan_topology --
            including its MaterializeFuseConflictError,
            MaterializeWithoutSinkError, and MultipleMaterializeAxesError
            subclasses.
        DtypeNotSupportedError: If any leaf's dtype is rejected by the target,
            naming every offending leaf rather than only the first.
    """
    validate_plan_topology(decisions, axis_boundaries, export_safe=True)

    blockers = check_export_safety(
        decisions,
        axis_boundaries,
        abstract_inputs,
        fn,
        target,
        request_features=request_features,
    )
    if blockers:
        detail = "\n".join(f"  - {b.axis}: {b.detail}" for b in blockers)
        msg = f"{len(blockers)} dtype blocker(s) for target {target.name!r}:\n{detail}"
        raise DtypeNotSupportedError(msg)
