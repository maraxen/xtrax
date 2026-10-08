"""Loss combinators for multi-task and weighted training."""

from collections.abc import Callable, Sequence
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
            A traced weight that must survive a value change without
            recompiling belongs on :class:`ComposedLoss`, not here.
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

    Optionally applies a dynamic weight schedule that scales the total loss
    based on the training step.

    Attributes:
        losses: Tuple of WeightedLoss instances, one per task.
        weight_schedule: Optional callable that takes step and returns a scalar
            multiplier for the total loss. If None, no schedule is applied.
            Must be eqx.field(static=True) as it holds a Python callable.
    """

    losses: tuple[WeightedLoss, ...]
    weight_schedule: Callable[[int], Array] | None = eqx.field(default=None, static=True)

    def __call__(
        self,
        predictions: tuple[PyTree, ...],
        targets: tuple[PyTree, ...],
        step: int = 0,
    ) -> Array:
        """Compute sum of all task losses with optional dynamic weight schedule.

        Args:
            predictions: Tuple of predictions, one per task.
            targets: Tuple of targets, one per task.
            step: Training step number for weight schedule. Default is 0.

        Returns:
            Scalar Array = sum(loss_i(pred_i, target_i) for all i), optionally
            multiplied by weight_schedule(step) if schedule is not None.

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
        total = jnp.sum(
            jnp.stack(
                [
                    loss(pred, target)
                    for loss, pred, target in zip(self.losses, predictions, targets)
                ]
            )
        )

        # weight_schedule returns a scalar multiplier; spec Task 7.2 is silent
        # on semantics, so we apply schedule as total = total * schedule(step).
        if self.weight_schedule is not None:
            total = total * self.weight_schedule(step)

        return total


class ComposedLoss(eqx.Module):
    """Sum of loss terms with traced weights and unweighted per-term aux.

    ``WeightedLoss`` holds a static Python float, so changing that float
    recompiles a jitted caller. ``ComposedLoss.weights`` is a rank-1 JAX
    array (a traced pytree leaf). Changing a weight value, a placement-weight
    value, or a per-batch flag array does not change the pytree structure or
    any static field, so a jitted caller does not recompile.

    Each term is ``fn(predictions, targets, **per_batch_flags) -> scalar``.
    The set of keyword names is part of the compile key; the array values
    passed under those names are traced.

    ``placement_weights`` is an optional per-term tuple. A non-None entry is
    forwarded to that term as the keyword ``placement_weights`` (per-output
    supervision: a mask or a weight per output position). A batch that passes
    the same keyword in ``per_batch_flags`` overrides the stored entry.
    ``None`` for a term means that term is called without the keyword.

    Returns ``(weighted_loss, aux)``. ``aux`` maps each term name to its
    scalar **before** multiplication by ``weights`` (placement, when applied
    inside the term, is already included). That dict is what
    ``Trainer(has_aux=True)`` merges into metrics.

    Attributes:
        terms: Static tuple of term callables.
        names: Static tuple of unique term names, aligned with ``terms``.
        weights: Traced array of shape ``(n_terms,)``.
        placement_weights: ``None``, or a tuple of length ``n_terms`` whose
            entries are arrays or ``None``.
    """

    terms: tuple[Callable[..., Array], ...] = eqx.field(static=True)
    names: tuple[str, ...] = eqx.field(static=True)
    weights: Array
    placement_weights: tuple[Array | None, ...] | None = None

    def __init__(
        self,
        terms: Sequence[tuple[str, Callable[..., Array]]],
        weights: Array | Sequence[float],
        placement_weights: Sequence[Array | None] | None = None,
    ):
        names = tuple(name for name, _fn in terms)
        fns = tuple(fn for _name, fn in terms)
        self.terms = fns
        self.names = names
        self.weights = jnp.asarray(weights)
        if placement_weights is None:
            self.placement_weights = None
        else:
            self.placement_weights = tuple(placement_weights)

    def __check_init__(self):
        n = len(self.terms)
        if n == 0:
            raise ValueError("ComposedLoss requires at least one term")
        if len(self.names) != n:
            raise ValueError(f"ComposedLoss: {len(self.names)} names for {n} terms")
        if len(set(self.names)) != n:
            raise ValueError(f"ComposedLoss term names must be unique, got {self.names}")
        if self.weights.shape != (n,):
            raise ValueError(
                f"ComposedLoss.weights must have shape {(n,)}, got {self.weights.shape}"
            )
        if self.placement_weights is not None and len(self.placement_weights) != n:
            raise ValueError(
                "ComposedLoss.placement_weights must have one entry per term, "
                f"got {len(self.placement_weights)} for {n} terms"
            )

    def __call__(
        self,
        predictions: PyTree,
        targets: PyTree,
        **per_batch_flags: Any,
    ) -> tuple[Array, dict[str, Array]]:
        """Return ``(weighted sum, unweighted per-term aux)``.

        Args:
            predictions: Model predictions passed through to every term.
            targets: Targets passed through to every term.
            **per_batch_flags: Extra traced arrays (masks, ablations) forwarded
                to every term. A ``placement_weights`` entry here overrides the
                stored per-term placement weight.

        Returns:
            ``(loss, aux)`` where ``loss = sum(weights * unweighted)`` and
            ``aux[name]`` is that term's unweighted scalar.
        """
        # Static-length tuple comprehension — unrolls at trace time.
        # This is structural unrolling over a fixed-length tuple, same as
        # MultiTaskLoss, not a data-axis hot loop.
        unweighted: list[Array] = []
        for i, fn in enumerate(self.terms):
            flags = dict(per_batch_flags)
            if (
                self.placement_weights is not None
                and self.placement_weights[i] is not None
                and "placement_weights" not in flags
            ):
                flags["placement_weights"] = self.placement_weights[i]
            if flags:
                unweighted.append(fn(predictions, targets, **flags))
            else:
                unweighted.append(fn(predictions, targets))

        values = jnp.stack(unweighted)
        loss = jnp.sum(self.weights * values)
        aux = {name: values[i] for i, name in enumerate(self.names)}
        return loss, aux
