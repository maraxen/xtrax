"""Recompile guard: backend-compile counter and static-arg negative control.

Positive gate: a traced, re-sampled value does not compile again after warmup.
Negative control: the same step with a Python int (static under
``eqx.filter_jit``) compiles at least once per new value, and
``assert_no_recompile_after`` must fail on that input.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from xtrax.profiling import assert_no_recompile_after, count_backend_compiles


def _make_step():
    @eqx.filter_jit
    def step(x, depth):
        return x * depth

    return step


def test_traced_resampled_value_does_not_recompile():
    """0 backend compiles across steps once the traced depth has been warmed."""
    step = _make_step()
    x = jnp.ones((8,))
    key = jax.random.key(0)
    args = []
    for _ in range(4):
        key, sub = jax.random.split(key)
        depth = jax.random.randint(sub, (), 0, 16, dtype=jnp.int32)
        args.append((x, depth))
    assert_no_recompile_after(step, args, warmup=1)


def test_python_int_static_arg_recompiles_per_value():
    """Negative control: a new Python int is static, so the guard must fail.

    The counter itself has to register at least one compile per new value --
    otherwise a passing ``count == 0`` gate would be vacuous.
    """
    x = jnp.ones((4,))
    new_values = (9, 10, 11)
    step = _make_step()
    jax.block_until_ready(step(x, 8))
    with count_backend_compiles() as recorded:
        for depth in new_values:
            jax.block_until_ready(step(x, depth))
    assert recorded.count >= len(new_values)
    assert recorded.seconds >= 0.0

    fresh = _make_step()
    with pytest.raises(AssertionError, match="backend compile"):
        assert_no_recompile_after(fresh, [(x, depth) for depth in (8, *new_values)], warmup=1)


def test_listener_is_detached_on_exit():
    from jax._src.monitoring import get_event_duration_listeners

    before = len(get_event_duration_listeners())
    with count_backend_compiles():
        during = len(get_event_duration_listeners())
    after = len(get_event_duration_listeners())
    assert during == before + 1
    assert after == before


def test_unregister_falls_back_to_public_api(monkeypatch):
    from jax._src.monitoring import get_event_duration_listeners

    import xtrax.profiling.compile_count as mod

    def _moved():
        raise AttributeError("unregister_event_duration_listener moved")

    monkeypatch.setattr(mod, "_load_private_unregister", _moved)
    before = len(get_event_duration_listeners())
    with mod.count_backend_compiles():
        assert len(get_event_duration_listeners()) == before + 1
    assert len(get_event_duration_listeners()) == before


def test_unregister_raises_when_jax_moves_both_hooks(monkeypatch):
    import jax.monitoring

    import xtrax.profiling.compile_count as mod

    def _moved():
        raise AttributeError("unregister_event_duration_listener")

    monkeypatch.setattr(mod, "_load_private_unregister", _moved)
    monkeypatch.setattr(mod, "_load_public_unregister", _moved)
    ctx = mod.count_backend_compiles()
    counter = ctx.__enter__()
    try:
        with pytest.raises(RuntimeError, match="JAX moved unregister_event_duration_listener"):
            ctx.__exit__(None, None, None)
    finally:
        jax.monitoring.unregister_event_duration_listener(counter._listener)
