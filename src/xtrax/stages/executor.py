"""Two-tier boundary executor (T1-04, #3051): the first code that runs AxisBoundary ops.

Fork-10 resolution (`.praxia/docs/specs/260702_design-2174-next-slices-minimal-composit.md`):
ordered `Tap`/`Sink` fire **inside** the per-axis tiling iterator body (per-step, one call per
`Vmap`/`SafeMap`/`Scan` element or batch), because a run-layer wrapper only ever sees the fully
stacked output and physically cannot deliver per-step order. `Fuse` fires once, **after** the
axis's iteration completes, over the assembled stacked output -- never per-step, and never over a
`Scan`'s carry (there is no code path here that ever passes a carry to `fuse`).

`Tap`/`Sink` implementations own their own `io_callback` call (see
`xtrax.stages.boundaries.Tap`/`Sink`) -- this module's job is narrower and purely mechanical: call
`boundary.tap`/`boundary.sink` at the right per-step position, and `boundary.fuse` at the right
post-iteration position. It never touches `xtrax.stages._callback.io_callback` directly.

Scope: this executes ONE axis at a time (`execute_map_axis` for `Vmap`/`SafeMap`,
`execute_scan_axis` for `Scan`). Nested composition (e.g. vmap-of-scan) is not built or
certified here -- each function returns an ordinary array/callable result, so nesting composes
naturally by calling one of these as the `fn` of an enclosing axis, but proving that nesting
preserves ordering under jit's reordering latitude is T1-05's job (the nested-ordering stress
harness), not this module's.

## The real performance cost of `ordered=True` (verified against JAX's own JEP-10657 design doc
and External Callbacks docs -- not folklore)

Every `io_callback` call, ordered or not, is a device<->host<->device round trip; JAX's own docs
warn this is "significant overhead" on GPU/TPU regardless of ordering. `ordered=True` adds a
further, structural cost on top of that baseline: it works by threading a **token** as a genuine
XLA data dependency between consecutive ordered calls (JEP-10657) -- each call's token output
feeds the next call's token input. That dependency is what *creates* the ordering guarantee, and
it also means XLA cannot reorder, overlap, or pipeline those calls with anything else. Under a
`Scan` with N steps, an ordered `Sink`/`Tap` becomes N *strictly serialized* host round trips with
no compiler freedom to hide that latency behind other device work.

`ordered=True` is also flatly incompatible with `Vmap` -- JAX raises `ValueError: Cannot vmap
ordered IO callback` if you try (this executor and `validate_plan_topology` both reject that
combination before JAX ever gets the chance to raise it).

**`SafeMap` + `ordered=True` is a steeper cliff than it first looks -- verified empirically, not
just from docs.** `jax.lax.map(..., batch_size=B)` batches internally via `jax.vmap` for ANY B >=
1 (confirmed directly: even `batch_size=1` raises the same "Cannot vmap ordered IO callback"
error `Vmap` does). Only `jax.lax.map` called with NO `batch_size` -- a pure `jax.lax.scan`
underneath -- tolerates `ordered=True`. There is no partially-batched middle ground. Consequence:
**an ordered `SafeMap` axis silently ignores its configured `batch_size` and runs one element at
a time**, whenever ordering is requested. This is not a corner case triggered by an unlucky
cardinality -- it is unconditional, for every ordered `SafeMap` axis, at any `batch_size`. This is
architecturally identical in cost to a fully sequential `Scan` over the same data. If you need
both ordering AND real batching throughput, there isn't one today; consider whether `Scan` (which
makes the sequential cost explicit in the strategy choice) is a more honest fit than a `SafeMap`
whose configured `batch_size` will be silently overridden.

**Practical guidance:** ordering is a real, structural cost, not a knob to leave on by default.
Only set `ordered=True` on a `Tap`/`Sink` when correctness genuinely depends on host-observed
order (e.g. writing a sequential log). If only one axis in a pipeline truly needs ordering, keep
`ordered=False` on every other axis's boundary ops so XLA retains scheduling freedom there. There
is no batch-size lever to pull once ordering is on for a `SafeMap` axis -- see above.
"""

from collections.abc import Callable
from typing import Any

import jax

from xtrax.stages.boundaries import AxisBoundary
from xtrax.tiling.strategy import SafeMap, Vmap
from xtrax.transforms.map import safe_map
from xtrax.transforms.scan import safe_scan


class ExecutorError(Exception):
    """Raised when the executor is asked to run a combination it cannot safely execute.

    Distinct from `xtrax.stages.topology.PlanTopologyError`: that fires at plan-construction
    time over a whole set of axis decisions; this fires at execution time, scoped to a single
    axis, and includes the `Vmap`+ordered case as defense-in-depth (in case a caller invokes
    this module directly without running `validate_plan_topology` first).
    """


def _has_ordered_op(boundary: AxisBoundary | None) -> bool:
    """True if `boundary` has a tap or sink whose `.ordered` attribute is True."""
    if boundary is None:
        return False
    tap_ordered = boundary.tap is not None and getattr(boundary.tap, "ordered", False)
    sink_ordered = boundary.sink is not None and getattr(boundary.sink, "ordered", False)
    return bool(tap_ordered or sink_ordered)


def _wrap_step(fn: Callable[[Any], Any], boundary: AxisBoundary | None) -> Callable[[Any], Any]:
    """Wrap `fn` so its per-step output passes through `boundary.tap`/`boundary.sink`.

    `tap`'s return value replaces the step output (continues downstream / gets stacked);
    `sink`'s return value is discarded, matching its `T -> None` contract.
    """
    if boundary is None or (boundary.tap is None and boundary.sink is None):
        return fn

    def wrapped(x: Any) -> Any:
        y = fn(x)
        if boundary.tap is not None:
            y = boundary.tap(y)
        if boundary.sink is not None:
            boundary.sink(y)
        return y

    return wrapped


def _apply_fuse(ys: Any, boundary: AxisBoundary | None) -> Any:
    """Apply `boundary.fuse` once to the fully stacked `ys`, if set. Never called per-step."""
    if boundary is not None and boundary.fuse is not None:
        return boundary.fuse(ys)
    return ys


def execute_map_axis(
    fn: Callable[[Any], Any],
    xs: Any,
    strategy: Vmap | SafeMap,
    boundary: AxisBoundary | None = None,
) -> Any:
    """Execute one `Vmap`/`SafeMap` axis, firing `tap`/`sink` per step and `fuse` once after.

    Args:
        fn: Per-element function to apply.
        xs: Input pytree; the leading axis is iterated.
        strategy: `Vmap` or `SafeMap` -- selects the iteration strategy.
        boundary: Optional `AxisBoundary` for this axis.

    Returns:
        The (optionally fused) stacked output.

    Raises:
        ExecutorError: `Vmap` with an ordered tap/sink -- JAX cannot lower this at all
            (`ValueError: Cannot vmap ordered IO callback`).

    Note:
        A `SafeMap` axis with an ordered tap/sink does NOT raise, but silently ignores
        `strategy.batch_size` and runs one element at a time -- see the module docstring
        ("`SafeMap` + `ordered=True` is a steeper cliff than it first looks").
    """
    wrapped = _wrap_step(fn, boundary)

    if isinstance(strategy, Vmap):
        if _has_ordered_op(boundary):
            msg = (
                "Vmap strategy cannot host an ordered Tap/Sink: jax.vmap raises "
                "'Cannot vmap ordered IO callback' for any io_callback(ordered=True). "
                "Use SafeMap or Scan for axes with ordered boundary ops. (This should "
                "already be caught by validate_plan_topology before reaching the "
                "executor; this is a defense-in-depth check.)"
            )
            raise ExecutorError(msg)
        ys = jax.vmap(wrapped)(xs)
        return _apply_fuse(ys, boundary)

    if isinstance(strategy, SafeMap):
        if _has_ordered_op(boundary):
            # jax.lax.map(..., batch_size=B) always batches via jax.vmap internally,
            # for ANY B >= 1 (verified empirically -- even batch_size=1 raises "Cannot
            # vmap ordered IO callback"; only batch_size=None, a pure sequential
            # jax.lax.map == jax.lax.scan, tolerates ordered=True). There is no
            # partially-batched path that preserves ordering, so strategy.batch_size
            # is NOT honored here: an ordered SafeMap axis always runs one element at
            # a time, regardless of the configured batch_size. This is a real,
            # possibly large performance cliff relative to the unordered path -- see
            # the module docstring; it is not a corner case, it is unconditional.
            ys = jax.lax.map(wrapped, xs)
        else:
            ys = safe_map(wrapped, xs, batch_size=strategy.batch_size)
        return _apply_fuse(ys, boundary)

    msg = f"execute_map_axis: unsupported strategy type {type(strategy)}"
    raise TypeError(msg)


def execute_scan_axis(
    fn: Callable[[Any, Any], tuple[Any, Any]],
    init: Any,
    xs: Any,
    boundary: AxisBoundary | None = None,
) -> tuple[Any, Any]:
    """Execute one `Scan` axis, firing `tap`/`sink` per step and `fuse` once after.

    Args:
        fn: Transition function `(carry, x) -> (carry, y)`.
        init: Initial carry.
        xs: Input pytree to scan over.
        boundary: Optional `AxisBoundary` for this axis.

    Returns:
        `(final_carry, ys)` -- `ys` is optionally fused; `fuse` never receives `final_carry`.
    """

    def wrapped_transition(carry: Any, x: Any) -> tuple[Any, Any]:
        carry, y = fn(carry, x)
        if boundary is not None:
            if boundary.tap is not None:
                y = boundary.tap(y)
            if boundary.sink is not None:
                boundary.sink(y)
        return carry, y

    final_carry, ys = safe_scan(wrapped_transition, init, xs)
    return final_carry, _apply_fuse(ys, boundary)


__all__ = [
    "ExecutorError",
    "execute_map_axis",
    "execute_scan_axis",
]
