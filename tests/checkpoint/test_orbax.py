"""Tests for xtrax checkpoint orbax wrappers."""

import tempfile
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import pytest

from xtrax.checkpoint.orbax import (
    get_checkpoint_manager,
    load_checkpoint,
    save_checkpoint,
)
from xtrax.training.types import ResumableState


class DummyModel(eqx.Module):
    """Simple model for testing."""

    weights: jnp.ndarray


@pytest.fixture
def tmp_checkpoint_dir():
    """Temporary directory for checkpoint tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def dummy_state():
    """Create a simple resumable state for testing."""
    model = DummyModel(weights=jnp.array([1.0, 2.0, 3.0]))
    key = jnp.ones((2,), dtype=jnp.uint32)
    opt_state = {"momentum": jnp.array([0.1, 0.2, 0.3])}
    return ResumableState(
        step=jnp.int32(42),
        key=key,
        model=model,
        opt_state=opt_state,
        extras={"info": "test"},
    )


def test_get_checkpoint_manager_creates_manager(tmp_checkpoint_dir):
    """Test that get_checkpoint_manager returns a CheckpointManager."""
    manager = get_checkpoint_manager(tmp_checkpoint_dir)
    assert manager is not None
    assert hasattr(manager, "save")
    assert hasattr(manager, "restore")
    assert hasattr(manager, "wait_until_finished")


def test_get_checkpoint_manager_with_options(tmp_checkpoint_dir):
    """Test get_checkpoint_manager with custom max_to_keep and keep_period."""
    manager = get_checkpoint_manager(tmp_checkpoint_dir, max_to_keep=3, keep_period=10)
    assert manager is not None


def test_save_checkpoint_saves_state(tmp_checkpoint_dir, dummy_state):
    """Test that save_checkpoint successfully saves a ResumableState."""
    manager = get_checkpoint_manager(tmp_checkpoint_dir)
    save_checkpoint(manager, dummy_state)

    # Verify the step was saved
    assert manager.latest_step() == 42


def test_save_checkpoint_with_step_override(tmp_checkpoint_dir, dummy_state):
    """Test save_checkpoint with explicit step override."""
    manager = get_checkpoint_manager(tmp_checkpoint_dir)
    save_checkpoint(manager, dummy_state, step=100)

    assert manager.latest_step() == 100


def test_save_checkpoint_calls_wait_until_finished(tmp_checkpoint_dir, dummy_state):
    """Test that save_checkpoint calls wait_until_finished."""
    manager = get_checkpoint_manager(tmp_checkpoint_dir)
    save_checkpoint(manager, dummy_state, step=0)

    # If wait_until_finished was called, the checkpoint should be immediately available
    latest = manager.latest_step()
    assert latest is not None


def test_load_checkpoint_round_trip(tmp_checkpoint_dir, dummy_state):
    """Test round-trip: save and load produces identical state."""
    manager = get_checkpoint_manager(tmp_checkpoint_dir)
    save_checkpoint(manager, dummy_state, step=0)

    # Load with template
    loaded = load_checkpoint(manager, dummy_state, step=0)

    assert isinstance(loaded, ResumableState)
    assert int(loaded.step) == 42

    # Check model weights
    assert jnp.allclose(loaded.model.weights, dummy_state.model.weights)

    # Check opt_state
    assert jnp.allclose(loaded.opt_state["momentum"], dummy_state.opt_state["momentum"])

    # Check extras
    assert loaded.extras == {"info": "test"}


def test_load_checkpoint_with_latest_step(tmp_checkpoint_dir, dummy_state):
    """Test load_checkpoint using latest_step (step=None)."""
    manager = get_checkpoint_manager(tmp_checkpoint_dir)

    # Save multiple checkpoints with same structure
    save_checkpoint(manager, dummy_state, step=10)
    state2 = ResumableState(
        step=jnp.int32(20),
        key=jnp.ones((2,), dtype=jnp.uint32),
        model=DummyModel(weights=jnp.array([4.0, 5.0, 6.0])),
        opt_state={"momentum": jnp.array([0.4, 0.5, 0.6])},
        extras={"info": "test"},  # Match the structure of dummy_state
    )
    save_checkpoint(manager, state2, step=20)

    # Load without specifying step (should get latest)
    loaded = load_checkpoint(manager, dummy_state)
    assert int(loaded.step) == 20


def test_load_checkpoint_empty_directory_raises(tmp_checkpoint_dir, dummy_state):
    """Test that load_checkpoint on empty directory raises FileNotFoundError."""
    manager = get_checkpoint_manager(tmp_checkpoint_dir)

    with pytest.raises(FileNotFoundError):
        load_checkpoint(manager, dummy_state, step=0)


def test_load_checkpoint_unknown_step_raises(tmp_checkpoint_dir, dummy_state):
    """Test that load_checkpoint with unknown step raises FileNotFoundError."""
    manager = get_checkpoint_manager(tmp_checkpoint_dir)
    save_checkpoint(manager, dummy_state, step=0)

    with pytest.raises(FileNotFoundError):
        load_checkpoint(manager, dummy_state, step=999)


def test_load_checkpoint_requires_template(tmp_checkpoint_dir, dummy_state):
    """Test that load_checkpoint requires state_template for pytree reconstruction."""
    manager = get_checkpoint_manager(tmp_checkpoint_dir)
    save_checkpoint(manager, dummy_state, step=0)

    # The template is required for orbax to know the pytree structure
    loaded = load_checkpoint(manager, dummy_state, step=0)
    assert isinstance(loaded, ResumableState)
    assert isinstance(loaded.model, DummyModel)


def test_save_multiple_checkpoints(tmp_checkpoint_dir, dummy_state):
    """Test saving multiple checkpoints with different steps."""
    manager = get_checkpoint_manager(tmp_checkpoint_dir, max_to_keep=5)

    for i in range(3):
        state = ResumableState(
            step=jnp.int32(i * 10),
            key=jnp.ones((2,), dtype=jnp.uint32),
            model=DummyModel(weights=jnp.ones(3) * i),
            opt_state={},
            extras={},  # Match the structure for loading
        )
        save_checkpoint(manager, state, step=i * 10)

    # Check that all steps are available
    assert manager.latest_step() == 20

    # Load intermediate checkpoint - create template with matching structure
    template = ResumableState(
        step=jnp.int32(0),
        key=jnp.ones((2,), dtype=jnp.uint32),
        model=DummyModel(weights=jnp.ones(3)),
        opt_state={},
        extras={},
    )
    loaded = load_checkpoint(manager, template, step=10)
    assert int(loaded.step) == 10
    assert jnp.allclose(loaded.model.weights, jnp.ones(3) * 1)


def test_checkpoint_preserves_key_precision(tmp_checkpoint_dir):
    """Test that random keys are preserved with correct precision."""
    manager = get_checkpoint_manager(tmp_checkpoint_dir)

    # Create state with a proper JAX random key
    key = jnp.array([0, 1], dtype=jnp.uint32)
    state = ResumableState(
        step=jnp.int32(5),
        key=key,
        model=DummyModel(weights=jnp.array([1.0])),
        opt_state={},
    )

    save_checkpoint(manager, state, step=0)
    loaded = load_checkpoint(manager, state, step=0)

    assert loaded.key.dtype == key.dtype
    assert jnp.array_equal(loaded.key, key)


# --- debt #739: off orbax's deprecated CheckpointManager kwargs -----------------


def _save_steps(manager, state, steps):
    for s in steps:
        save_checkpoint(manager, state, step=s)


def test_retention_is_latest_n_union_every_period(tmp_checkpoint_dir, dummy_state):
    """max_to_keep=3 + keep_period=10 keeps the last 3 steps plus every multiple of 10.

    Behavioral invariant: must hold identically on the legacy options and on
    the preservation_policy form they migrate to.
    """
    manager = get_checkpoint_manager(tmp_checkpoint_dir, max_to_keep=3, keep_period=10)
    _save_steps(manager, dummy_state, range(1, 26))
    assert sorted(manager.all_steps()) == [10, 20, 23, 24, 25]


def test_max_to_keep_none_keeps_everything(tmp_checkpoint_dir, dummy_state):
    manager = get_checkpoint_manager(tmp_checkpoint_dir, max_to_keep=None)
    _save_steps(manager, dummy_state, range(1, 8))
    assert sorted(manager.all_steps()) == list(range(1, 8))


def test_manager_uses_preservation_policy_not_deprecated_fields(tmp_checkpoint_dir):
    """orbax documents max_to_keep/keep_period/item_handlers as 'deprecated, do not use'."""
    import orbax.checkpoint as ocp

    manager = get_checkpoint_manager(tmp_checkpoint_dir, max_to_keep=3, keep_period=10)
    options = manager._options  # noqa: SLF001 -- orbax exposes the resolved options only here
    assert options.max_to_keep is None
    assert options.keep_period is None
    assert isinstance(options.preservation_policy, ocp.checkpoint_managers.AnyPreservationPolicy)


def test_checkpoint_written_by_legacy_item_handlers_api_still_loads(
    tmp_checkpoint_dir, dummy_state
):
    """Runs checkpointed by earlier xtrax (item_handlers + items=) must stay resumable."""
    import orbax.checkpoint as ocp

    legacy = ocp.CheckpointManager(
        tmp_checkpoint_dir.resolve(),
        options=ocp.CheckpointManagerOptions(max_to_keep=5),
        item_handlers={"state": ocp.PyTreeCheckpointHandler()},
    )
    legacy.save(step=7, items={"state": dummy_state})
    legacy.wait_until_finished()
    legacy.close()

    loaded = load_checkpoint(get_checkpoint_manager(tmp_checkpoint_dir), dummy_state)
    assert int(loaded.step) == int(dummy_state.step)
    assert jnp.array_equal(loaded.model.weights, dummy_state.model.weights)
    assert jnp.array_equal(loaded.key, dummy_state.key)
