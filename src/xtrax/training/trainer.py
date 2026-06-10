"""Trainer implementation for xtrax (spec §3.13)."""

from typing import Any

import equinox as eqx
import jax
import optax

from xtrax.training.types import LossFunction, ResumableState


class Trainer(eqx.Module):
    """Single-model trainer for supervised learning.

    Attributes:
        loss_fn: Callable[[predictions, targets], scalar loss].
        optimizer: optax.GradientTransformation for parameter updates.

    Invariants:
        - step() is eqx.filter_jit-decorated.
        - Gradients computed via eqx.filter_value_and_grad.
        - Optimizer.update() receives eqx.filter(state.model, eqx.is_array)
          as third argument (required for weight-decay compatibility).
        - Returns new ResumableState with incremented step and updated model/opt_state.
        - Metrics dict always includes at minimum {"loss": scalar}.
    """

    loss_fn: LossFunction
    optimizer: optax.GradientTransformation

    @eqx.filter_jit
    def step(
        self,
        state: ResumableState,
        batch: Any,
    ) -> tuple[ResumableState, dict[str, jax.Array]]:
        """Execute one training step: loss, grad, update, return new state + metrics.

        Args:
            state: ResumableState with model, opt_state, and step counter.
            batch: PyTree with at minimum {"inputs": ..., "targets": ...}.

        Returns:
            (new_state, metrics) where:
              - new_state: ResumableState with step += 1, updated model and opt_state.
              - metrics: dict[str, Array] with at minimum {"loss": scalar}.
        """

        def loss_fn_inner(model):
            predictions = model(batch["inputs"])
            return self.loss_fn(predictions, batch["targets"])

        # Compute loss and gradients (grads are filtered to arrays only)
        loss, grads = eqx.filter_value_and_grad(loss_fn_inner)(state.model)

        # Update: CRITICAL — pass eqx.filter(state.model, eqx.is_array) as 3rd arg
        # This ensures weight decay and other weight-aware optimizers work correctly
        filtered_params = eqx.filter(state.model, eqx.is_array)
        updates, new_opt_state = self.optimizer.update(
            grads, state.opt_state, filtered_params
        )

        # Apply updates to model
        new_model = eqx.apply_updates(state.model, updates)

        # Create new state with incremented step counter
        new_state = eqx.tree_at(
            lambda s: (s.model, s.opt_state, s.step),
            state,
            (new_model, new_opt_state, state.step + 1),
        )

        # Return new state and metrics
        return new_state, {"loss": loss}
