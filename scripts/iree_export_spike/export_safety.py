"""Host-side, pre-trace gate deciding whether a BatchPlan is IREE-exportable.

Every check here is decidable *before* any JAX tracing, from static data only:
``type(strategy).__name__`` and ``AxisBoundary``'s three ``eqx.field(static=True)``
slots. That is the same duck-typed style ``xtrax.stages.topology`` already uses
(``topology.py:134``), so this works on foreign plan objects too.

If promoted into ``src/``, this becomes ``validate_plan_topology(export_safe=True)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# Strategies whose lowering is pure XLA control flow: vmap / lax.map / lax.scan /
# in-trace gather-scatter. All export cleanly.
EXPORTABLE_STRATEGIES = frozenset({"Vmap", "SafeMap", "Scan", "DedupGather"})

# Bucket is a *plan descriptor*, not a device transform -- it pads on the host with
# NumPy precisely because JAX has no dynamic shapes inside jit. It is the mechanism
# that GIVES us static shapes, so it belongs above the boundary, not inside it.
HOST_TIER_STRATEGIES = frozenset({"Bucket"})

# WhileCarry lowers to stablehlo.while fine, but its trip count is data-dependent and
# unbounded at compile time -- unacceptable for a latency-bounded serverless artifact.
# Convert to Scan with a static length (see strategy.fixed_step_count_cond).
UNBOUNDED_STRATEGIES = frozenset({"WhileCarry"})


class ExportUnsafeError(Exception):
    """Raised when a plan cannot be exported to a single traceable artifact."""


@dataclass(frozen=True)
class ExportBlocker:
    """One reason an axis cannot cross the export boundary."""

    axis: str
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.rule}] axis {self.axis!r}: {self.detail}"


def _axis_name(decision: Any) -> str:
    spec = getattr(decision, "spec", None)
    return str(getattr(spec, "name", "<unknown>"))


def check_plan_export_safety(
    decisions: Sequence[Any],
    axis_boundaries: Mapping[str, Any] | None = None,
) -> list[ExportBlocker]:
    """Return every reason this plan cannot be exported. Empty list means exportable.

    Args:
        decisions: ``BatchPlan.decisions`` (or any sequence of objects exposing
            ``.spec.name`` and ``.strategy``).
        axis_boundaries: Name-keyed boundaries, e.g. from
            ``xtrax.stages.topology.axis_boundaries_by_name``.

    Returns:
        A list of ``ExportBlocker``. Order follows ``decisions``.
    """
    boundaries = axis_boundaries or {}
    blockers: list[ExportBlocker] = []

    for decision in decisions:
        name = _axis_name(decision)
        strategy_name = type(getattr(decision, "strategy", None)).__name__

        # Rule 1: strategy must lower to pure XLA control flow.
        if strategy_name in HOST_TIER_STRATEGIES:
            blockers.append(
                ExportBlocker(
                    axis=name,
                    rule="strategy",
                    detail=(
                        f"{strategy_name} is host-tier by construction (NumPy padding "
                        "before the jit boundary). Run select_bucket()/bucketize() in "
                        "host glue and export one artifact per bucket boundary."
                    ),
                )
            )
        elif strategy_name in UNBOUNDED_STRATEGIES:
            blockers.append(
                ExportBlocker(
                    axis=name,
                    rule="strategy",
                    detail=(
                        f"{strategy_name} has a data-dependent, compile-time-unbounded "
                        "trip count. Convert to Scan with a static length."
                    ),
                )
            )
        elif strategy_name not in EXPORTABLE_STRATEGIES:
            blockers.append(
                ExportBlocker(
                    axis=name,
                    rule="strategy",
                    detail=(
                        f"unrecognised strategy {strategy_name!r}; not known to be exportable."
                    ),
                )
            )

        # Rule 2: fuse-only boundaries. Tap/Sink are contractually io_callback-backed.
        boundary = boundaries.get(name)
        if boundary is not None:
            for slot in ("tap", "sink"):
                if getattr(boundary, slot, None) is not None:
                    blockers.append(
                        ExportBlocker(
                            axis=name,
                            rule="boundary",
                            detail=(
                                f"{slot} is set; {slot.capitalize()} implementations "
                                "must use io_callback internally, which pierces the "
                                "export boundary. Only fuse survives export."
                            ),
                        )
                    )

    return blockers


def assert_plan_export_safe(
    decisions: Sequence[Any],
    axis_boundaries: Mapping[str, Any] | None = None,
) -> None:
    """Raise ``ExportUnsafeError`` listing every blocker, or return None."""
    blockers = check_plan_export_safety(decisions, axis_boundaries)
    if blockers:
        joined = "\n  ".join(str(b) for b in blockers)
        raise ExportUnsafeError(f"plan is not IREE-exportable:\n  {joined}")


def find_bcoo_leaves(model: Any) -> list[str]:
    """Return key-paths of any BCOO leaves in ``model``.

    Sparsified models substitute ``BCOO`` at leaf positions, and BCOO is a pytree
    *node*, not a leaf -- so the tree structure changes. Under ``jax.export`` a
    closure-held BCOO is baked in as constants (which is what we want for a
    self-contained artifact); this helper exists so the caller *knows* that is
    happening rather than discovering it in the MLIR.

    Returns an empty list when ``jax.experimental.sparse`` is unavailable.
    """
    try:
        import jax
        from jax.experimental.sparse import BCOO
    except ImportError:  # pragma: no cover - sparse ships with jax today
        return []

    found: list[str] = []
    flat = jax.tree_util.tree_flatten_with_path(model, is_leaf=lambda x: isinstance(x, BCOO))[0]
    for path, leaf in flat:
        if isinstance(leaf, BCOO):
            found.append(jax.tree_util.keystr(path))
    return found


__all__ = [
    "EXPORTABLE_STRATEGIES",
    "HOST_TIER_STRATEGIES",
    "UNBOUNDED_STRATEGIES",
    "ExportBlocker",
    "ExportUnsafeError",
    "assert_plan_export_safe",
    "check_plan_export_safety",
    "find_bcoo_leaves",
]
