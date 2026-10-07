"""Loop-body cost that scales with the loop's own extent (debt #1983).

A ``lax.scan``/``lax.while_loop`` over ``L`` steps whose body recomputes an
``(L, ...)`` tensor and keeps one row of it does ``O(L^2)`` work where ``O(L)``
suffices -- and nothing notices: correctness tests pass, small-``L`` wall clocks
look fine, and XLA's ``cost_analysis`` counts a ``while`` body once, so a naive
total hides the trip count. (Found the hard way: an aminx autoregressive sampler
ran 25-40x its reference per draw.)

Two primitives. ``loop_bodies`` only traces. ``extent_scaling_report`` traces
both extents and, when a ``while`` has no static trip count, runs the program
once per extent under ``jax.disable_jit`` to count that loop's iterations:

- :func:`loop_bodies` -- every ``scan``/``while`` body reachable from ``fn``, at
  any depth, with the work of ONE iteration and the trip count (``scan`` length;
  ``None`` for ``while``, whose trip count is data-dependent).
- :func:`extent_scaling_report` -- traces at ``extent`` and ``2 * extent``, pairs
  loop bodies by structural path, and flags a body only when BOTH its
  per-iteration work AND its trip count grow with the extent. A body whose work
  is independent of the extent has a work ratio near 1.0; a body that re-does
  full-extent work every step has a ratio near 2.0. A loop whose trip count is
  constant or sub-linear (``jnp.searchsorted`` lowers to a scan of
  ``ceil(log2(n))`` steps) is not the O(L^2) shape, even when each step's work
  grows because the step is batched over the extent.

"Work" is an explicit proxy, not a FLOP-exact model: ``2*M*N*K`` per
``dot_general`` plus one unit per output element of every other equation, with
nested loop bodies multiplied by their trip count (``while``: counted once) and
``cond`` taking its most expensive branch. It is meant for *ratios between two
extents of the same program*, where the proxy's constant factors cancel.

jax is imported lazily, inside the functions, so importing this module (and
``xtrax.profiling``) stays free of jax -- matching the package's leaf contract.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from xtrax.profiling.jaxpr import sub_jaxprs

#: Default ratio (at 2x extent vs 1x) at or above which per-iteration work, and
#: separately trip count, count as growing with the extent. Extent-independent
#: work sits near 1.0 and full-extent recompute near 2.0; a linear trip count
#: doubles and a log-n trip count (binary search) barely moves. 1.5 splits both
#: with margin for padding and bookkeeping ops.
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
    trip_count_at_extent: int | None = None
    trip_count_at_double_extent: int | None = None


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
            body = next(sub_jaxprs(eqn.params[_LOOP_BODY_PARAM[name]]))
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
            subs = list(sub_jaxprs(eqn.params))
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


def _trip_growth_scales(small: int | None, large: int | None, threshold: float) -> bool:
    """Whether the trip count grows with the extent.

    A missing count (a ``while`` that could not be measured) is treated as
    scaling: the O(L^2) sampler is a ``while`` whose trips equal the extent,
    and dropping an unmeasured one would hide that true positive. A known
    count scales only when ``large / small`` reaches ``threshold``.
    """
    if small is None or large is None:
        return True
    if small <= 0:
        return large > small
    return large / small >= threshold


def _measure_while_trips(fn: Callable[..., Any], args: tuple[Any, ...]) -> list[int]:
    """Run ``fn(*args)`` eagerly and count each ``while_loop``'s iterations.

    The first invocation of each ``(cond, body)`` code pair is kept, in the
    order the loops are entered (outermost first, matching :func:`loop_bodies`).
    Later invocations of the same pair -- a ``while`` inside a ``scan`` -- are
    the same loop, not a new one.

    The count replaces ``jax.lax.while_loop`` on the ``jax.lax`` module for the
    duration of the call. A name bound earlier (``from jax.lax import
    while_loop``) still calls the original and is not counted. The replacement
    is process-global and not thread-safe: another thread that calls
    ``jax.lax.while_loop`` during the measurement runs the wrapper, or misses
    the loop if the original is restored underneath it.
    """
    import jax

    real = jax.lax.while_loop
    first: dict[object, int] = {}
    order: list[object] = []

    def wrapped(cond_fun, body_fun, init_val):  # noqa: ANN001 -- lax's own signature
        key = (
            getattr(cond_fun, "__code__", id(cond_fun)),
            getattr(body_fun, "__code__", id(body_fun)),
        )
        fresh = key not in first
        if fresh:
            order.append(key)
            first[key] = 0
        trips = 0

        def body_count(state):  # noqa: ANN001, ANN202
            nonlocal trips
            trips += 1
            return body_fun(state)

        result = real(cond_fun, body_count, init_val)
        if fresh:
            first[key] = trips
        return result

    jax.lax.while_loop = wrapped
    try:
        with jax.disable_jit():
            fn(*args)
    finally:
        jax.lax.while_loop = real
    return [first[key] for key in order]


def _trip_counts_at(
    fn: Callable[..., Any], args: tuple[Any, ...], bodies: list[LoopBody]
) -> list[int | None]:
    """Trip count aligned with ``bodies``.

    ``scan`` lengths come from the trace. ``while`` lengths are measured. If
    the eager run fails or does not line up with the traced ``while`` bodies,
    those counts stay ``None`` and :func:`_trip_growth_scales` keeps the old
    "assume linear" bias so a true positive is not dropped.
    """
    if not any(body.trip_count is None for body in bodies):
        return [body.trip_count for body in bodies]
    try:
        measured: list[int | None] = list(_measure_while_trips(fn, args))
    except Exception:  # noqa: BLE001 - measurement is a refinement, not a gate
        measured = []
    n_while = sum(body.primitive == "while" for body in bodies)
    if len(measured) != n_while:
        measured = [None] * n_while
    out: list[int | None] = []
    index = 0
    for body in bodies:
        if body.trip_count is not None:
            out.append(body.trip_count)
        else:
            out.append(measured[index])
            index += 1
    return out


def extent_scaling_report(
    fn: Callable[..., Any],
    make_args: Callable[[int], tuple[Any, ...]],
    extent: int,
    *,
    threshold: float = DEFAULT_RATIO_THRESHOLD,
) -> ExtentScalingReport:
    """Flag loop bodies whose work and trip count both grow with the extent.

    Args:
        fn: The program to inspect. Traced at both extents. A ``while`` whose
            trip count is not static is also run, once per extent, to count
            iterations.
        make_args: Builds ``fn``'s positional arguments for a given extent --
            e.g. ``lambda n: (jnp.zeros((n, d)),)`` for a sequence of length ``n``.
        extent: Base extent; the program is also traced at ``2 * extent``.
        threshold: Ratio (2x vs 1x) at or above which per-iteration work and
            trip count each count as growing. A body is flagged only when both
            do. Log-n and constant trip counts stay under it.

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
    base_args = make_args(extent)
    double_args = make_args(2 * extent)
    base = loop_bodies(fn, *base_args)
    doubled = loop_bodies(fn, *double_args)
    base_paths = [b.path for b in base]
    doubled_paths = [b.path for b in doubled]
    if base_paths != doubled_paths:
        msg = (
            f"loop structure differs between extent={extent} and extent={2 * extent}: "
            f"{base_paths} vs {doubled_paths}; per-iteration work cannot be paired"
        )
        raise LoopStructureMismatchError(msg)
    base_trips = _trip_counts_at(fn, base_args, base)
    doubled_trips = _trip_counts_at(fn, double_args, doubled)
    findings = []
    for small, large, small_trips, large_trips in zip(
        base, doubled, base_trips, doubled_trips, strict=True
    ):
        ratio = large.iteration_work / small.iteration_work if small.iteration_work else math.inf
        trips_scale = _trip_growth_scales(small_trips, large_trips, threshold)
        findings.append(
            ScalingFinding(
                path=small.path,
                primitive=small.primitive,
                work_at_extent=small.iteration_work,
                work_at_double_extent=large.iteration_work,
                ratio=ratio,
                flagged=ratio >= threshold and trips_scale,
                trip_count_at_extent=small_trips,
                trip_count_at_double_extent=large_trips,
            )
        )
    return ExtentScalingReport(extent=extent, threshold=threshold, findings=tuple(findings))
