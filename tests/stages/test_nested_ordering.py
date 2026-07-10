"""T1-05 (#3056): nested-ordering stress harness certifying T1-04's executor under real nesting.

Covers scan-of-scan (positive, unconditionally supported) and vmap-of-scan (both the
correct/recommended batched-shape recipe, and the literal-jax.vmap composition that
JAX itself rejects) -- see xtrax.stages.executor's module docstring, "Nesting:
vmap-of-scan", for the full explanation this harness certifies.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tests.stages.test_executor import HostRecordSink
from xtrax.stages._callback import io_callback
from xtrax.stages.boundaries import AxisBoundary
from xtrax.stages.executor import ExecutorError, execute_map_axis, execute_scan_axis
from xtrax.tiling.strategy import Vmap

N_TRIALS = 20


class PairRecordSink:
    """Like HostRecordSink, but records a 2-element (outer, inner) index pair per call."""

    def __init__(self, records: list[tuple[int, int]], *, ordered: bool) -> None:
        self.records = records
        self.ordered = ordered

    def __call__(self, x: jax.Array) -> None:
        def _write(v: jax.Array) -> jax.Array:
            self.records.append((int(v[0]), int(v[1])))
            return jnp.zeros((2,), dtype=jnp.int32)

        io_callback(_write, jax.ShapeDtypeStruct((2,), jnp.int32), x, ordered=self.ordered)
        return None


class BatchedHostRecordSink:
    """Records a whole (B,)-shaped batch of values per call -- for the batched-shape
    vmap-of-scan recipe, where each ordered call carries all B lanes' values together.
    Shape-agnostic (reads x.shape/x.dtype at call time) since B varies per trial.
    """

    def __init__(self, records: list[list[int]], *, ordered: bool) -> None:
        self.records = records
        self.ordered = ordered

    def __call__(self, x: jax.Array) -> None:
        def _write(v: jax.Array) -> jax.Array:
            self.records.append([int(e) for e in v])
            return jnp.zeros(v.shape, dtype=v.dtype)

        io_callback(_write, jax.ShapeDtypeStruct(x.shape, x.dtype), x, ordered=self.ordered)
        return None


class TestScanOfScanPreservesOrder:
    """Positive certification: no vmap anywhere, outer scan wraps inner scan."""

    def test_preserves_full_flattened_order_under_stress(self) -> None:
        for trial in range(N_TRIALS):
            records: list[tuple[int, int]] = []
            boundary = AxisBoundary(sink=PairRecordSink(records, ordered=True))
            outer_n = 2 + (trial % 3)
            inner_n = 2 + (trial % 4)

            def outer_transition(
                carry: jax.Array,
                outer_x: jax.Array,
                inner_n: int = inner_n,
                boundary: AxisBoundary = boundary,
            ) -> tuple[jax.Array, jax.Array]:
                def inner_transition(
                    inner_carry: jax.Array, inner_x: jax.Array
                ) -> tuple[jax.Array, jax.Array]:
                    return inner_carry + inner_x, jnp.array([outer_x, inner_x])

                final_inner_carry, _ = execute_scan_axis(
                    inner_transition, 0, jnp.arange(inner_n), boundary
                )
                return carry + final_inner_carry, final_inner_carry

            run = jax.jit(
                lambda xs, outer_transition=outer_transition: execute_scan_axis(
                    outer_transition, 0, xs, None
                ),
                donate_argnums=(0,),
            )
            xs = jnp.arange(outer_n)
            _final, ys = run(xs)
            jax.block_until_ready(ys)

            expected = [(o, i) for o in range(outer_n) for i in range(inner_n)]
            assert records == expected, (
                f"trial {trial} (outer_n={outer_n}, inner_n={inner_n}): {records} != {expected}"
            )


class TestBatchedShapeVmapOfScanPreservesOrder:
    """The corrected positive certification: bake the outer axis into execute_scan_axis's
    carry/xs shape directly (no jax.vmap anywhere) -- the recommended recipe."""

    def test_preserves_order_under_stress(self) -> None:
        for trial in range(N_TRIALS):
            records: list[list[int]] = []
            boundary = AxisBoundary(sink=BatchedHostRecordSink(records, ordered=True))
            batch = 2 + (trial % 3)
            steps = 2 + (trial % 4)

            def batched_transition(
                carry: jax.Array, x: jax.Array, boundary: AxisBoundary = boundary
            ) -> tuple[jax.Array, jax.Array]:
                if boundary.sink is not None:
                    boundary.sink(carry)
                return carry + x, carry

            init = jnp.arange(batch) * 100
            xs = jnp.arange(steps)
            run = jax.jit(
                lambda init, xs, batched_transition=batched_transition: jax.lax.scan(
                    batched_transition, init, xs
                ),
                donate_argnums=(0,),
            )
            final, _ys = run(init, xs)
            jax.block_until_ready(final)

            lane_bases = [i * 100 for i in range(batch)]
            running = list(lane_bases)
            expected: list[list[int]] = []
            for step in range(steps):
                expected.append(list(running))
                running = [r + step for r in running]

            assert records == expected, (
                f"trial {trial} (batch={batch}, steps={steps}): {records} != {expected}"
            )


class TestLiteralVmapOfScanOrdering:
    """The 'don't compose it this way' counter-example, and the narrow case where it works."""

    def test_lane_dependent_ordering_fails_loud(self) -> None:
        records: list[int] = []
        inner_boundary = AxisBoundary(sink=HostRecordSink(records, ordered=True))

        def outer_fn(outer_x: jax.Array) -> jax.Array:
            def transition(carry: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
                return carry + x, carry

            final_carry, _ys = execute_scan_axis(transition, outer_x, jnp.arange(4), inner_boundary)
            return final_carry

        outer_xs = jnp.arange(3) * 100
        with pytest.raises(ExecutorError, match="Vmap axis's `fn`"):
            execute_map_axis(outer_fn, outer_xs, Vmap(), None)

    def test_lane_independent_ordering_succeeds(self) -> None:
        """Narrow case: the inner ordered scan's sunk value doesn't depend on the vmapped
        axis at all, so vmap never actually batches the io_callback's argument."""
        records: list[int] = []
        inner_boundary = AxisBoundary(sink=HostRecordSink(records, ordered=True))

        def outer_fn(outer_x: jax.Array) -> jax.Array:
            def transition(carry: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
                return carry, x

            final_carry, _ys = execute_scan_axis(transition, outer_x, jnp.arange(4), inner_boundary)
            return final_carry

        outer_xs = jnp.arange(3) * 100
        out = execute_map_axis(outer_fn, outer_xs, Vmap(), None)
        assert list(out) == list(outer_xs)
        assert records == [0, 1, 2, 3]
