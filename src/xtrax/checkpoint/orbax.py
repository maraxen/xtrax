"""Checkpoint utilities using Orbax.

Uses orbax's stable ``CheckpointManager`` with its current configuration API --
``preservation_policy`` and ``handler_registry`` plus ``args=`` save/restore --
not the kwargs orbax documents as "deprecated, do not use" (``max_to_keep``,
``keep_period``, ``item_handlers``, ``items=``). orbax's v1 API is still
``orbax.checkpoint.experimental.v1`` as of 0.12.1, so it is deliberately not
adopted here (debt #739). The on-disk layout is unchanged: checkpoints written
through the legacy kwargs by earlier xtrax releases restore as before.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import orbax.checkpoint as ocp

if TYPE_CHECKING:
    from xtrax.training.types import ResumableState

#: Name of the single PyTree item each checkpoint step holds (its subdirectory name).
_ITEM = "state"


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
        A CheckpointManager holding one PyTree item named ``"state"``. Retention
        keeps the latest ``max_to_keep`` steps plus every multiple of
        ``keep_period`` (their union), matching orbax's legacy semantics.
    """
    directory = Path(directory).resolve()
    # Resolve to absolute: orbax CheckpointManager rejects relative paths at
    # save/load time ("Checkpoint path should be absolute"), and both CLI verbs
    # construct checkpoint dirs relative to cwd (.xtrax/runs/<id>/checkpoints).
    # Resolving here fixes save AND resume without changing the manifest's
    # durable relative-path contract.

    policies: list[ocp.checkpoint_managers.PreservationPolicy] = [
        ocp.checkpoint_managers.LatestN(n=max_to_keep)
    ]
    if keep_period is not None:
        policies.append(ocp.checkpoint_managers.EveryNSteps(interval_steps=keep_period))
    options = ocp.CheckpointManagerOptions(
        preservation_policy=ocp.checkpoint_managers.AnyPreservationPolicy(policies=policies),
    )

    registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
    registry.add(_ITEM, ocp.args.PyTreeSave, ocp.PyTreeCheckpointHandler())
    registry.add(_ITEM, ocp.args.PyTreeRestore, ocp.PyTreeCheckpointHandler())
    return ocp.CheckpointManager(directory, options=options, handler_registry=registry)


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
    manager.save(step_int, args=ocp.args.Composite(**{_ITEM: ocp.args.PyTreeSave(state)}))

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
    loaded = manager.restore(
        step, args=ocp.args.Composite(**{_ITEM: ocp.args.PyTreeRestore(state_template)})
    )
    return loaded[_ITEM]
