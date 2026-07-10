"""Tests for the T1-04 two-tier boundary executor (#3051)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from xtrax.stages._callback import io_callback
from xtrax.stages.boundaries import AxisBoundary
from xtrax.stages.executor import ExecutorError, execute_map_axis, execute_scan_axis
from xtrax.tiling.strategy import SafeMap, Scan, Vmap


class HostRecordSink:
    """Reference Sink fixture (AC4): writes host records from inside jit via the T1-03 shim.

    Appends each observed index to `records` (a plain Python list the test keeps a
    reference to) so the test can inspect the actual host-observed call order after
    the jitted computation returns.
    """

    def __init__(self, records: list[int], *, ordered: bool) -> None:
        self.records = records
        self.ordered = ordered

    def __call__(self, x: jax.Array) -> None:
        def _write(idx: jax.Array) -> jax.Array:
            self.records.append(int(idx))
            return jnp.int32(0)

        io_callback(_write, jax.ShapeDtypeStruct((), jnp.int32), x, ordered=self.ordered)
        return None


class TestCounterTestOrderPreserved:
    """AC4's core claim: ordered Tap/Sink preserve step order under jit."""

    def test_safemap_ordered_sink_preserves_order_under_jit(self) -> None:
        records: list[int] = []
        boundary = AxisBoundary(sink=HostRecordSink(records, ordered=True))
        xs = jnp.arange(8)

        @jax.jit
        def run(xs: jax.Array) -> jax.Array:
            return execute_map_axis(lambda x: x, xs, SafeMap(batch_size=4), boundary)

        out = run(xs)
        jax.block_until_ready(out)

        assert records == list(range(8))

    def test_scan_ordered_sink_preserves_order_under_jit(self) -> None:
        records: list[int] = []
        boundary = AxisBoundary(sink=HostRecordSink(records, ordered=True))
        xs = jnp.arange(8)

        @jax.jit
        def run(xs: jax.Array) -> tuple[jax.Array, jax.Array]:
            return execute_scan_axis(lambda c, x: (c, x), 0, xs, boundary)

        _final, ys = run(xs)
        jax.block_until_ready(ys)

        assert records == list(range(8))


class TestSafeMapOrderedIgnoresBatchSize:
    """Documents the discovered cliff: ordered SafeMap always runs one element at a time,
    regardless of the configured batch_size (jax.lax.map(..., batch_size=B) batches via
    jax.vmap for any B >= 1, which JAX itself rejects for ordered io_callback).

    Note: jax.vmap/jax.lax.map always trace the body function once with the batch axis
    stripped from the shape it sees, regardless of batch_size -- so the batch_size=1 vs
    batch_size=4 distinction is not observable via the shape the traced fn receives. What
    IS observable, and is the practically meaningful claim, is that changing batch_size
    has zero effect on correctness (or on whether ordering holds) once ordered=True.
    """

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_ordering_holds_regardless_of_configured_batch_size(self, batch_size: int) -> None:
        records: list[int] = []
        boundary = AxisBoundary(sink=HostRecordSink(records, ordered=True))
        xs = jnp.arange(8)

        execute_map_axis(lambda x: x, xs, SafeMap(batch_size=batch_size), boundary)

        assert records == list(range(8))

    def test_unordered_safemap_output_is_unaffected_by_this_change(self) -> None:
        """Regression guard: the unordered path's own logic is untouched by this PR."""
        xs = jnp.arange(8)
        out = execute_map_axis(lambda x: x + 1, xs, SafeMap(batch_size=4), None)
        assert list(out) == list(range(1, 9))


class TestVmapOrderedRejected:
    def test_vmap_with_ordered_tap_raises(self) -> None:
        boundary = AxisBoundary(tap=HostRecordSink([], ordered=True))
        xs = jnp.arange(4)

        with pytest.raises(ExecutorError, match="Vmap strategy cannot host an ordered"):
            execute_map_axis(lambda x: x, xs, Vmap(), boundary)

    def test_vmap_with_ordered_sink_raises(self) -> None:
        boundary = AxisBoundary(sink=HostRecordSink([], ordered=True))
        xs = jnp.arange(4)

        with pytest.raises(ExecutorError, match="Vmap strategy cannot host an ordered"):
            execute_map_axis(lambda x: x, xs, Vmap(), boundary)

    def test_vmap_without_ordered_ops_is_unaffected(self) -> None:
        xs = jnp.arange(4)
        out = execute_map_axis(lambda x: x * 2, xs, Vmap(), None)
        assert list(out) == [0, 2, 4, 6]


class TestTapContinuesSinkDiscards:
    def test_tap_return_value_is_stacked(self) -> None:
        tap = lambda x: x * 10  # noqa: E731
        boundary = AxisBoundary(tap=tap)
        xs = jnp.arange(4)

        out = execute_map_axis(lambda x: x, xs, Vmap(), boundary)

        assert list(out) == [0, 10, 20, 30]

    def test_sink_return_value_does_not_affect_stacked_output(self) -> None:
        class ReturningSink:
            ordered = False

            def __call__(self, x: jax.Array) -> jax.Array:
                return x * 999  # a Sink implementation misbehaving; must still be ignored

        boundary = AxisBoundary(sink=ReturningSink())
        xs = jnp.arange(4)

        out = execute_map_axis(lambda x: x, xs, Vmap(), boundary)

        assert list(out) == [0, 1, 2, 3]

    def test_scan_tap_transforms_y_not_carry(self) -> None:
        boundary = AxisBoundary(tap=lambda y: y * 10)
        xs = jnp.arange(4)

        final_carry, ys = execute_scan_axis(lambda c, x: (c + x, x), 0, xs, boundary)

        assert final_carry == int(xs.sum())  # carry untouched by tap
        assert list(ys) == [0, 10, 20, 30]  # y stream transformed by tap


class TestFuseCalledOncePostHoc:
    def test_fuse_called_exactly_once_with_full_stacked_ys(self) -> None:
        calls: list[tuple[int, ...]] = []

        def fuse(ys: jax.Array) -> jax.Array:
            calls.append(ys.shape)
            return ys.sum()

        boundary = AxisBoundary(fuse=fuse)
        xs = jnp.arange(8)

        out = execute_map_axis(lambda x: x, xs, SafeMap(batch_size=4), boundary)

        assert calls == [(8,)]
        assert out == int(xs.sum())

    def test_scan_fuse_never_receives_final_carry(self) -> None:
        seen_args: list[jax.Array] = []

        def fuse(ys: jax.Array) -> jax.Array:
            seen_args.append(ys)
            return ys.sum()

        boundary = AxisBoundary(fuse=fuse)
        xs = jnp.arange(4)

        final_carry, fused_ys = execute_scan_axis(lambda c, x: (c + x, x), 0, xs, boundary)

        assert final_carry == int(xs.sum())
        assert len(seen_args) == 1
        assert list(seen_args[0]) == list(xs)  # fuse saw ys, never final_carry (a scalar 6)
        assert fused_ys == int(xs.sum())

    def test_no_fuse_returns_raw_stacked_ys(self) -> None:
        xs = jnp.arange(4)
        out = execute_map_axis(lambda x: x, xs, Vmap(), None)
        assert list(out) == list(xs)


def test_execute_map_axis_rejects_unsupported_strategy() -> None:
    """execute_map_axis's own TypeError fallback matters for production use (beartype's
    runtime enforcement is a test-time-only convenience, wired in tests/conftest.py's
    pytest_configure -- not active for a real xtrax consumer without pytest), but under
    pytest here the `strategy: Vmap | SafeMap` annotation is intercepted by beartype first,
    so either error is an acceptable, equally loud rejection of an unsupported strategy.
    """
    from jaxtyping import TypeCheckError

    with pytest.raises((TypeError, TypeCheckError)):
        execute_map_axis(lambda x: x, jnp.arange(4), Scan(), None)
