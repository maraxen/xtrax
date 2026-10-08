"""Opt-in mapped-axis index for AxisBoundary sinks (#2588).

The executor passes ``(y, index)`` only when ``sink_receives_index`` is set.
It does not inspect the sink callable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xtrax.stages._callback import io_callback
from xtrax.stages.boundaries import AxisBoundary
from xtrax.stages.executor import execute_map_axis, execute_scan_axis
from xtrax.tiling.strategy import ChunkedMap, Vmap


class _IndexSink:
    """Records host-side ``(value, index)`` pairs. ``ordered`` stays false so Vmap can run."""

    ordered = False

    def __init__(self) -> None:
        self.pairs: list[tuple[int, int]] = []
        self.dtypes: list[np.dtype] = []

    def __call__(self, y: jax.Array, index: jax.Array) -> None:
        def _write(y: jax.Array, index: jax.Array) -> jax.Array:
            value = np.asarray(y)
            position = np.asarray(index)
            self.pairs.append((int(value), int(position)))
            self.dtypes.append(position.dtype)
            return jnp.int32(0)

        io_callback(_write, jax.ShapeDtypeStruct((), jnp.int32), y, index, ordered=False)


def _run_map(strategy: Vmap | ChunkedMap, n: int) -> tuple[list[int], list[int], list[np.dtype]]:
    sink = _IndexSink()
    boundary = AxisBoundary(sink=sink, sink_receives_index=True)
    xs = jnp.arange(n)

    @jax.jit
    def run(values: jax.Array) -> jax.Array:
        return execute_map_axis(lambda x: x, values, strategy, boundary)

    out = run(xs)
    jax.block_until_ready(out)
    assert list(np.asarray(out)) == list(range(n))
    values = [pair[0] for pair in sink.pairs]
    indices = [pair[1] for pair in sink.pairs]
    return values, indices, sink.dtypes


@pytest.mark.parametrize(
    ("strategy", "n"),
    [
        (Vmap(), 6),
        (Vmap(), 1),
        (ChunkedMap(batch_size=4), 8),
        (ChunkedMap(batch_size=4), 10),
        (ChunkedMap(batch_size=4), 9),
    ],
    ids=["vmap", "vmap-size1", "chunked", "remainder", "remainder-one"],
)
def test_opted_in_sink_receives_global_index(strategy: Vmap | ChunkedMap, n: int) -> None:
    values, indices, dtypes = _run_map(strategy, n)
    # Each element is paired with its global position. A ChunkedMap whose length
    # mod batch_size is 1 peels that last element into a second unordered
    # dispatch, so the host may see it before the prefix; the index is still
    # the global position.
    assert sorted(zip(indices, values, strict=True)) == [(i, i) for i in range(n)]
    assert dtypes == [np.dtype(np.int32)] * n
    peeled = (
        isinstance(strategy, ChunkedMap)
        and strategy.batch_size is not None
        and strategy.batch_size > 1
        and n % strategy.batch_size == 1
    )
    if not peeled:
        assert indices == list(range(n))
        assert values == list(range(n))


def test_opted_in_scan_sink_receives_step_index() -> None:
    sink = _IndexSink()
    boundary = AxisBoundary(sink=sink, sink_receives_index=True)
    xs = jnp.arange(7)

    @jax.jit
    def run(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        return execute_scan_axis(lambda carry, x: (carry + x, x), jnp.int32(0), values, boundary)

    final_carry, ys = run(xs)
    jax.block_until_ready(ys)
    assert int(final_carry) == int(xs.sum())
    assert list(np.asarray(ys)) == list(range(7))
    assert [pair[1] for pair in sink.pairs] == list(range(7))
    assert [pair[0] for pair in sink.pairs] == list(range(7))


def test_one_arg_sink_stays_one_arg_when_index_is_off() -> None:
    """Default is off. A one-argument sink keeps working, including under a remainder chunk."""
    assert AxisBoundary().sink_receives_index is False
    seen: list[int] = []

    class _OneArg:
        ordered = False

        def __call__(self, y: jax.Array) -> None:
            def _write(y: jax.Array) -> jax.Array:
                seen.append(int(np.asarray(y)))
                return jnp.int32(0)

            io_callback(_write, jax.ShapeDtypeStruct((), jnp.int32), y, ordered=False)

    boundary = AxisBoundary(sink=_OneArg(), sink_receives_index=False)
    xs = jnp.arange(10)

    @jax.jit
    def run(values: jax.Array) -> jax.Array:
        return execute_map_axis(lambda x: x + 1, values, ChunkedMap(batch_size=4), boundary)

    out = run(xs)
    jax.block_until_ready(out)
    assert list(np.asarray(out)) == list(range(1, 11))
    assert seen == list(range(1, 11))


def test_executor_does_not_infer_index_from_signature() -> None:
    seen: list[object] = []

    def sink(y: jax.Array, index: object = None) -> None:
        del y
        seen.append(index)

    boundary = AxisBoundary(sink=sink, sink_receives_index=False)
    execute_map_axis(lambda x: x, jnp.arange(4), Vmap(), boundary)
    assert seen
    assert all(item is None for item in seen)


def test_index_flag_without_sink_leaves_output_unchanged() -> None:
    boundary = AxisBoundary(sink_receives_index=True)
    out = execute_map_axis(lambda x: x + 1, jnp.arange(4), Vmap(), boundary)
    assert list(np.asarray(out)) == [1, 2, 3, 4]


def test_tap_keeps_one_argument_when_sink_is_indexed() -> None:
    sink = _IndexSink()

    def tap(y: jax.Array) -> jax.Array:
        return y + 1

    boundary = AxisBoundary(tap=tap, sink=sink, sink_receives_index=True)
    xs = jnp.arange(4)

    @jax.jit
    def run(values: jax.Array) -> jax.Array:
        return execute_map_axis(lambda x: x, values, Vmap(), boundary)

    out = run(xs)
    jax.block_until_ready(out)
    assert list(np.asarray(out)) == [1, 2, 3, 4]
    assert [pair[0] for pair in sink.pairs] == [1, 2, 3, 4]
    assert [pair[1] for pair in sink.pairs] == [0, 1, 2, 3]
