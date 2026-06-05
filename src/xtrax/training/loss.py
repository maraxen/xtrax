"""Loss combinators for multi-task and weighted training."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from xtrax.training.types import LossFunction

PyTree = Any
Array = jax.Array


class WeightedLoss(eqx.Module):
    """Combines a loss function with a static weight multiplier.

    The weight field is marked as static to ensure it does not appear
    in JAX array filtering — it is a Python float, not a JAX Array.
    This allows the loss function to be traced and JIT-compiled while
    the weight remains a compile-time constant.

    Attributes:
        loss_fn: A callable implementing the LossFunction protocol.
        weight: A Python float (compile-time constant) multiplying the loss.
    """
    loss_fn: LossFunction
    weight: float = eqx.field(static=True)

    def __call__(self, predictions: PyTree, targets: PyTree) -> Array:
        """Apply the loss function and scale by weight.

        Args:
            predictions: Model predictions (PyTree).
            targets: Target values (PyTree).

        Returns:
            Scalar Array equal to weight * loss_fn(predictions, targets).
        """
        return self.weight * self.loss_fn(predictions, targets)


class MultiTaskLoss(eqx.Module):
    """Combines multiple weighted losses for multi-task learning.

    Each task has its own WeightedLoss. The __call__ method sums
    the contributions from all tasks. Length validation ensures
    predictions, targets, and losses are all the same length.

    Attributes:
        losses: Tuple of WeightedLoss instances, one per task.
    """
    losses: tuple[WeightedLoss, ...]

    def __call__(
        self,
        predictions: tuple[PyTree, ...],
        targets: tuple[PyTree, ...],
    ) -> Array:
        """Compute sum of all task losses.

        Args:
            predictions: Tuple of predictions, one per task.
            targets: Tuple of targets, one per task.

        Returns:
            Scalar Array = sum(loss_i(pred_i, target_i) for all i).

        Raises:
            AssertionError: If len(predictions) != len(targets) != len(self.losses).
        """
        assert len(predictions) == len(targets) == len(self.losses), (
            f"MultiTaskLoss: length mismatch predictions={len(predictions)}, "
            f"targets={len(targets)}, losses={len(self.losses)}"
        )

        # Static-length tuple comprehension — unrolls at trace time.
        # This is NOT a data-axis hot loop (forbidden by spec §1);
        # it is structural unrolling over a fixed-length tuple.
        return jnp.sum(
            jnp.stack(
                [
                    loss(pred, target)
                    for loss, pred, target in zip(
                        self.losses, predictions, targets
                    )
                ]
            )
        )
