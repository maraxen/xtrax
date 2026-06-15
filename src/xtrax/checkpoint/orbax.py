"""Checkpoint utilities using Orbax."""

from pathlib import Path
from typing import TYPE_CHECKING

import orbax.checkpoint as ocp

if TYPE_CHECKING:
    from xtrax.training.types import ResumableState


def get_checkpoint_manager(
    directory: str | Path,
    max_to_keep: int | None = 5,
    keep_period: int | None = None,
) -> ocp.CheckpointManager:
    """Create and return a CheckpointManager for the given directory.

    Args:
        directory: Directory to store checkpoints.
        max_to_keep: Maximum number of checkpoints to keep. None keeps all. Defaults to 5.
        keep_period: Period (in steps) for keeping checkpoints. Defaults to None.

    Returns:
        A CheckpointManager instance with PyTreeCheckpointHandler configured.
    """
    directory = Path(directory)

    # Create options for the checkpoint manager
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        keep_period=keep_period,
    )

    # Create the manager with PyTreeCheckpointHandler
    manager = ocp.CheckpointManager(
        directory,
        options=options,
        item_handlers={"state": ocp.PyTreeCheckpointHandler()},
    )

    return manager


def save_checkpoint(
    manager: ocp.CheckpointManager,
    state: "ResumableState",
    step: int | None = None,
) -> None:
    """Save a checkpoint to the manager.

    Args:
        manager: CheckpointManager instance.
        state: ResumableState to save.
        step: Step number to save at. If None, uses int(state.step).

    Note:
        This function must be called outside JAX-traced contexts.
        Calls manager.wait_until_finished() after saving.
    """
    # Extract step: use provided step or convert state.step to int
    step_int = step if step is not None else int(state.step)

    # Save the checkpoint
    manager.save(step=step_int, items={"state": state})

    # Wait for save to complete
    manager.wait_until_finished()


def load_checkpoint(
    manager: ocp.CheckpointManager,
    state_template: "ResumableState",
    step: int | None = None,
) -> "ResumableState":
    """Load a checkpoint from the manager.

    Args:
        manager: CheckpointManager instance.
        state_template: Template ResumableState for pytree structure.
            Required for orbax to reconstruct the pytree.
        step: Step number to load. If None, uses latest_step().

    Returns:
        The loaded ResumableState.

    Raises:
        FileNotFoundError: If checkpoint doesn't exist or directory is empty.
    """
    # Determine which step to load
    if step is None:
        step = manager.latest_step()
        if step is None:
            raise FileNotFoundError("No checkpoints found in directory")

    # Restore the checkpoint
    loaded = manager.restore(step=step, items={"state": state_template})

    # Extract the state from the composite result
    return loaded["state"]
