"""Core training types for xtrax."""

from typing import Any, Protocol, runtime_checkable

import jax

from xtrax._tree import Struct, field

PyTree = Any
Array = jax.Array


@runtime_checkable
class LossFunction(Protocol):
    """Protocol for loss functions.

    A loss function takes predictions and targets (both PyTrees)
    and returns a scalar Array.
    """

    def __call__(self, predictions: PyTree, targets: PyTree) -> Array: ...


@runtime_checkable
class Callback(Protocol):
    """Protocol for training callbacks with 7 hooks.

    All hooks run Python-side (outside JAX traces).
    Mutating state in a callback has no effect on training
    since state is immutable.
    """

    def on_train_start(self, state: "ResumableState") -> None: ...
    def on_train_end(self, state: "ResumableState") -> None: ...
    def on_resume(self, state: "ResumableState") -> None: ...
    def on_epoch_start(self, state: "ResumableState", epoch: int) -> None: ...
    def on_epoch_end(self, state: "ResumableState", epoch: int) -> None: ...
    def on_step_start(self, state: "ResumableState") -> None: ...
    def on_step_end(self, state: "ResumableState", metrics: dict[str, Array]) -> None: ...


class ResumableState(Struct):
    """Immutable training state for resumable training.

    step: Current training step as jnp.int32 scalar (dynamic leaf).
    key: JAX random key for reproducibility.
    model: The trainable model (any pytree, including an ``equinox.Module``).
    opt_state: Optimizer state (arbitrary PyTree).
    extras: Optional extra state dict (default: empty dict).
    """

    step: Array
    key: Array
    model: PyTree
    opt_state: PyTree
    extras: dict[str, PyTree] = field(default_factory=dict)
