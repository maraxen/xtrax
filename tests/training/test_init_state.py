"""AC4: init_state public API contract tests."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from xtrax.training import init_state
from xtrax.training.types import ResumableState


def _make_model():
    """Minimal eqx.Module for testing."""
    return eqx.nn.Linear(2, 2, key=jax.random.PRNGKey(0))


def _make_optimizer():
    return optax.sgd(0.01)


def test_init_state_step_dtype():
    """AC4 pin: step must be jnp.int32, not Python int or int64."""
    model = _make_model()
    optimizer = _make_optimizer()
    state = init_state(model, optimizer, seed=42)
    assert isinstance(state, ResumableState)
    assert state.step.dtype == jnp.int32, f"Expected int32, got {state.step.dtype}"
    assert int(state.step) == 0


def test_init_state_key_reproducible():
    """AC4 pin: key reproducible — same seed → same key."""
    model = _make_model()
    optimizer = _make_optimizer()
    s1 = init_state(model, optimizer, seed=7)
    s2 = init_state(model, optimizer, seed=7)
    assert jnp.array_equal(s1.key, s2.key), "Same seed must produce same PRNGKey"


def test_init_state_key_varies_with_seed():
    """Different seeds → different keys."""
    model = _make_model()
    optimizer = _make_optimizer()
    s1 = init_state(model, optimizer, seed=1)
    s2 = init_state(model, optimizer, seed=2)
    assert not jnp.array_equal(s1.key, s2.key)


def test_init_state_opt_state_matches_optimizer_init():
    """AC4 pin: opt_state = optimizer.init(eqx.filter(model, eqx.is_array))."""
    model = _make_model()
    optimizer = _make_optimizer()
    state = init_state(model, optimizer, seed=0)
    expected_opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    # Compare leaves structurally
    leaves_actual = jax.tree_util.tree_leaves(state.opt_state)
    leaves_expected = jax.tree_util.tree_leaves(expected_opt_state)
    assert len(leaves_actual) == len(leaves_expected)
    for a, e in zip(leaves_actual, leaves_expected):
        assert jnp.array_equal(a, e)


def test_init_state_extras_empty():
    """AC4 pin: extras must be an empty dict."""
    model = _make_model()
    optimizer = _make_optimizer()
    state = init_state(model, optimizer, seed=0)
    assert state.extras == {}


def test_init_state_public_import():
    """AC4: init_state importable from xtrax.training (public API pin)."""
    import xtrax.training as xt

    assert hasattr(xt, "init_state"), "init_state must be in xtrax.training public API"
    assert "init_state" in xt.__all__, "init_state must be in __all__"
