"""Tests for xtrax.profiling.loop_scaling (debt #1983).

Every detector here is paired with a control that must fire: the full-recompute
autoregressive body (the aminx O(L^2) sampler shape) is the negative control,
the incremental body that writes its row into a carried buffer is the positive
control that must NOT fire.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from xtrax.profiling.loop_scaling import (
    LoopStructureMismatchError,
    extent_scaling_report,
    loop_bodies,
)

D = 8


def _weights() -> jax.Array:
    return jnp.eye(D)


def full_recompute_scan(x: jax.Array) -> jax.Array:
    """Each step re-runs the decoder over ALL L positions and keeps one row: O(L^2)."""
    w = _weights()
    n = x.shape[0]

    def body(carry, i):
        h = carry @ w  # (L, D) @ (D, D) every iteration
        row = jax.lax.dynamic_slice_in_dim(h, i, 1, axis=0)
        return jax.lax.dynamic_update_slice_in_dim(carry, row, i, axis=0), None

    out, _ = jax.lax.scan(body, x, jnp.arange(n))
    return out


def incremental_scan(x: jax.Array) -> jax.Array:
    """Each step touches only row i and writes it back into the (L, D) buffer: O(L)."""
    w = _weights()
    n = x.shape[0]

    def body(carry, i):
        row = jax.lax.dynamic_slice_in_dim(carry, i, 1, axis=0) @ w  # (1, D) @ (D, D)
        return jax.lax.dynamic_update_slice_in_dim(carry, row, i, axis=0), None

    out, _ = jax.lax.scan(body, x, jnp.arange(n))
    return out


def full_recompute_while(x: jax.Array) -> jax.Array:
    w = _weights()
    n = x.shape[0]

    def cond(state):
        return state[0] < n

    def body(state):
        i, carry = state
        row = jax.lax.dynamic_slice_in_dim(carry @ w, i, 1, axis=0)
        return i + 1, jax.lax.dynamic_update_slice_in_dim(carry, row, i, axis=0)

    return jax.lax.while_loop(cond, body, (0, x))[1]


def _seq(n: int) -> tuple[jax.Array]:
    return (jnp.ones((n, D)),)


def test_full_recompute_scan_is_flagged() -> None:
    """Negative control: the aminx O(L^2) shape must fire."""
    report = extent_scaling_report(jax.jit(full_recompute_scan), _seq, 16)
    assert len(report.findings) == 1
    (finding,) = report.flagged
    assert finding.primitive == "scan"
    assert finding.ratio == pytest.approx(2.0, rel=0.1)


def test_full_recompute_while_is_flagged() -> None:
    report = extent_scaling_report(jax.jit(full_recompute_while), _seq, 16)
    (finding,) = report.flagged
    assert finding.primitive == "while"


def test_incremental_scan_is_not_flagged() -> None:
    """Positive control: an in-place row update into the (L, D) carry is O(row), not O(L)."""
    report = extent_scaling_report(jax.jit(incremental_scan), _seq, 16)
    assert len(report.findings) == 1
    assert report.flagged == ()
    assert report.findings[0].ratio == pytest.approx(1.0, abs=0.05)


def test_iteration_work_times_trip_count() -> None:
    """(1): scan's trip count multiplies ONE iteration's work, which cost_analysis hides."""

    def f(x):
        def body(c, _):
            return c @ c, None  # one (4,4)@(4,4) dot: 2*16*4 = 128 per iteration

        return jax.lax.scan(body, x, None, length=10)[0]

    (body,) = loop_bodies(f, jnp.ones((4, 4)))
    assert body.trip_count == 10
    assert body.iteration_work == 128
    assert body.total_work == 1280
    assert body.max_dot_output_elements == 16


def test_while_has_unknown_trip_count() -> None:
    (body,) = loop_bodies(jax.jit(full_recompute_while), jnp.ones((4, D)))
    assert body.trip_count is None
    assert body.total_work is None


def test_nested_loops_are_all_reported_outermost_first() -> None:
    def f(x):
        def outer(c, _):
            inner, _ = jax.lax.scan(lambda cc, __: (cc @ cc, None), c, None, length=3)
            return inner, None

        return jax.lax.scan(outer, x, None, length=2)[0]

    outer, inner = loop_bodies(f, jnp.ones((4, 4)))
    assert outer.path.count(":scan") == 1
    assert inner.path.startswith(outer.path)
    assert outer.iteration_work == 3 * inner.iteration_work


def test_structure_change_between_extents_raises() -> None:
    def f(x):
        if x.shape[0] > 20:  # a second loop appears only at the larger extent
            x = jax.lax.scan(lambda c, _: (c * 2.0, None), x, None, length=2)[0]
        return jax.lax.scan(lambda c, _: (c + 1.0, None), x, None, length=2)[0]

    with pytest.raises(LoopStructureMismatchError, match="loop structure differs"):
        extent_scaling_report(f, _seq, 16)


def test_extent_must_be_positive() -> None:
    with pytest.raises(ValueError, match="extent must be >= 1"):
        extent_scaling_report(incremental_scan, _seq, 0)
