"""Loop-body cost that scales with the loop's own extent (debt #1983).

A ``lax.scan``/``lax.while_loop`` over ``L`` steps whose body recomputes an
``(L, ...)`` tensor and keeps one row of it does ``O(L^2)`` work where ``O(L)``
suffices -- and nothing notices: correctness tests pass, small-``L`` wall clocks
look fine, and XLA's ``cost_analysis`` counts a ``while`` body once, so a naive
total hides the trip count. (Found the hard way: an aminx autoregressive sampler
ran 25-40x its reference per draw.)

Two primitives, both static (trace only -- nothing is compiled or executed):

- :func:`loop_bodies` -- every ``scan``/``while`` body reachable from ``fn``, at
  any depth, with the work of ONE iteration and the trip count (``scan`` length;
  ``None`` for ``while``, whose trip count is data-dependent).
- :func:`extent_scaling_report` -- traces at ``extent`` and ``2 * extent``, pairs
  loop bodies by structural path, and flags every body whose per-iteration work
  grows with the extent. A body whose work is independent of the extent has a
  ratio near 1.0; a body that re-does full-extent work every step has a ratio
  near 2.0.

"Work" is an explicit proxy, not a FLOP-exact model: ``2*M*N*K`` per
``dot_general`` plus one unit per output element of every other equation, with
nested loop bodies multiplied by their trip count (``while``: counted once) and
``cond`` taking its most expensive branch. It is meant for *ratios between two
extents of the same program*, where the proxy's constant factors cancel.

jax is imported lazily, inside the functions, so importing this module (and
``xtrax.profiling``) stays free of jax -- matching the package's leaf contract.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

#: Default per-iteration work ratio (at 2x extent vs 1x) at or above which a loop
#: body is flagged. Extent-independent bodies sit near 1.0 and full-extent
#: recompute near 2.0; 1.5 splits them with margin for padding/bookkeeping ops.
DEFAULT_RATIO_THRESHOLD = 1.5

_LOOP_BODY_PARAM = {"scan": "jaxpr", "while": "body_jaxpr"}

#: In-place buffer updates, costed by their UPDATE operand (index into invars).
_IN_PLACE_UPDATE_OPERAND = {
    "dynamic_update_slice": 1,
    "scatter": 2,
    "scatter-add": 2,
    "scatter_add": 2,
    "scatter-mul": 2,
    "scatter_mul": 2,
    "scatter-min": 2,
    "scatter_min": 2,
    "scatter-max": 2,
    "scatter_max": 2,
}


@dataclass(frozen=True, slots=True)
class LoopBody:
    """One ``scan``/``while`` body found in a traced program."""

    path: str
    primitive: str
    trip_count: int | None
    iteration_work: int
    max_dot_output_elements: int

    @property
    def total_work(self) -> int | None:
        """Work of the whole loop.

        Returns:
            ``iteration_work * trip_count``, or ``None`` for ``while`` (its trip
            count is data-dependent).
        """
        return None if self.trip_count is None else self.iteration_work * self.trip_count


@dataclass(frozen=True, slots=True)
class ScalingFinding:
    """A loop body compared across two extents of the same program."""

    path: str
    primitive: str
    work_at_extent: int
    work_at_double_extent: int
    ratio: float
    flagged: bool


@dataclass(frozen=True, slots=True)
class ExtentScalingReport:
    """Every loop body of one program, compared at ``extent`` and ``2 * extent``."""

    extent: int
    threshold: float
    findings: tuple[ScalingFinding, ...]

    @property
    def flagged(self) -> tuple[ScalingFinding, ...]:
        """Findings whose ratio reached ``threshold``.

        Returns:
            The flagged findings, in the report's (outermost-first) order.
        """
        return tuple(f for f in self.findings if f.flagged)


class LoopStructureMismatchError(ValueError):
    """The program's loop structure differs between the two traced extents."""


def _inner_jaxprs(value: Any) -> Iterator[Any]:  # noqa: ANN401 -- jax internals
    """Yield every (open) Jaxpr reachable from an eqn param value."""
    stack = [value]
    while stack:
        v = stack.pop()
        if hasattr(v, "eqns"):
            yield v
        elif hasattr(v, "jaxpr") and hasattr(v.jaxpr, "eqns"):
            yield v.jaxpr
        elif isinstance(v, (tuple, list)):
            stack.extend(v)


def _numel(aval: Any) -> int:  # noqa: ANN401
    shape = getattr(aval, "shape", ())
    return math.prod(int(d) for d in shape)


def _dot_work(eqn: Any) -> tuple[int, int]:  # noqa: ANN401
    """(2 * output elements * contracted size, output elements) for a dot_general."""
    (lhs_contract, _), _ = eqn.params["dimension_numbers"]
    lhs_shape = eqn.invars[0].aval.shape
    contracted = math.prod(int(lhs_shape[i]) for i in lhs_contract)
    out = _numel(eqn.outvars[0].aval)
    return 2 * out * contracted, out


def _jaxpr_work(jaxpr: Any, path: str, found: list[LoopBody]) -> tuple[int, int]:  # noqa: ANN401
    """Work of one execution of ``jaxpr``; appends every loop body found to ``found``.

    Returns ``(work, max_dot_output_elements)``.
    """
    work = 0
    max_dot = 0
    for idx, eqn in enumerate(jaxpr.eqns):
        name = eqn.primitive.name
        here = f"{path}/{idx}:{name}"
        if name in _LOOP_BODY_PARAM:
            body = next(_inner_jaxprs(eqn.params[_LOOP_BODY_PARAM[name]]))
            before = len(found)
            body_work, body_dot = _jaxpr_work(body, here, found)
            trips = int(eqn.params["length"]) if name == "scan" else None
            found.insert(before, LoopBody(here, name, trips, body_work, body_dot))
            work += body_work * (trips if trips is not None else 1)
            max_dot = max(max_dot, body_dot)
        elif name == "dot_general":
            dot_work, out = _dot_work(eqn)
            work += dot_work
            max_dot = max(max_dot, out)
        elif name in _IN_PLACE_UPDATE_OPERAND:
            # XLA updates the carried buffer in place: the cost is the update,
            # not the full-extent output aval (else every incremental body that
            # writes its row into an (L, ...) buffer would look O(L) per step).
            work += _numel(eqn.invars[_IN_PLACE_UPDATE_OPERAND[name]].aval)
        else:
            subs = [j for v in eqn.params.values() for j in _inner_jaxprs(v)]
            if subs:
                # cond: most expensive branch; pjit / custom_* / remat: the one callee.
                branch_costs = [
                    _jaxpr_work(sub, f"{here}[{i}]", found) for i, sub in enumerate(subs)
                ]
                work += max(c[0] for c in branch_costs)
                max_dot = max(max_dot, *(c[1] for c in branch_costs))
            else:
                work += sum(_numel(v.aval) for v in eqn.outvars)
    return work, max_dot


def loop_bodies(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> list[LoopBody]:  # noqa: ANN401
    """Trace ``fn(*args, **kwargs)`` and report every ``scan``/``while`` body.

    Args:
        fn: The program to inspect (traced with ``jax.make_jaxpr``, never executed).
        *args: Positional arguments to trace ``fn`` with.
        **kwargs: Keyword arguments to trace ``fn`` with.

    Returns:
        One :class:`LoopBody` per loop, outermost first; a nested loop follows the
        loop that contains it.
    """
    import jax

    closed = jax.make_jaxpr(fn)(*args, **kwargs)
    found: list[LoopBody] = []
    _jaxpr_work(closed.jaxpr, "", found)
    return found


def extent_scaling_report(
    fn: Callable[..., Any],
    make_args: Callable[[int], tuple[Any, ...]],
    extent: int,
    *,
    threshold: float = DEFAULT_RATIO_THRESHOLD,
) -> ExtentScalingReport:
    """Flag loop bodies whose per-iteration work grows with the loop's extent.

    Args:
        fn: The program to inspect (traced, never executed).
        make_args: Builds ``fn``'s positional arguments for a given extent --
            e.g. ``lambda n: (jnp.zeros((n, d)),)`` for a sequence of length ``n``.
        extent: Base extent; the program is also traced at ``2 * extent``.
        threshold: Per-iteration work ratio (2x vs 1x) at or above which a body
            is flagged.

    Returns:
        An :class:`ExtentScalingReport` with one finding per loop body.

    Raises:
        ValueError: If ``extent < 1``.
        LoopStructureMismatchError: If the two traces do not contain the same
            loops at the same structural paths (their bodies cannot be paired).
    """
    if extent < 1:
        msg = f"extent must be >= 1, got {extent}"
        raise ValueError(msg)
    base = loop_bodies(fn, *make_args(extent))
    doubled = loop_bodies(fn, *make_args(2 * extent))
    base_paths = [b.path for b in base]
    doubled_paths = [b.path for b in doubled]
    if base_paths != doubled_paths:
        msg = (
            f"loop structure differs between extent={extent} and extent={2 * extent}: "
            f"{base_paths} vs {doubled_paths}; per-iteration work cannot be paired"
        )
        raise LoopStructureMismatchError(msg)
    findings = []
    for small, large in zip(base, doubled, strict=True):
        ratio = large.iteration_work / small.iteration_work if small.iteration_work else math.inf
        findings.append(
            ScalingFinding(
                path=small.path,
                primitive=small.primitive,
                work_at_extent=small.iteration_work,
                work_at_double_extent=large.iteration_work,
                ratio=ratio,
                flagged=ratio >= threshold,
            )
        )
    return ExtentScalingReport(extent=extent, threshold=threshold, findings=tuple(findings))
