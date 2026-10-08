"""Trainer implementation for xtrax (spec §3.13)."""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import optax

from xtrax.training.types import ResumableState


class Trainer(eqx.Module):
    """Single-model trainer for supervised learning.

    Attributes:
        loss_fn: Loss callable. The signature depends on ``takes_key`` and
            ``has_aux`` (see below).
        optimizer: optax.GradientTransformation for parameter updates.
        has_aux: When True, ``loss_fn`` returns ``(loss, aux)`` and ``aux``
            is merged into the metrics dict. Default False keeps the scalar
            loss form.
        takes_key: When True, ``step`` splits ``state.key`` once and calls
            ``loss_fn(model, batch, step_key)``. Default False keeps the
            classic ``loss_fn(predictions, targets)`` form, and ``state.key``
            is left unchanged.

    Calling conventions (both flags default off, so existing callers are
    unchanged):

    - ``takes_key=False``, ``has_aux=False``: ``loss_fn(predictions, targets) -> scalar``.
      The trainer calls ``model(batch["inputs"])``.
    - ``takes_key=False``, ``has_aux=True``: ``loss_fn(predictions, targets) -> (loss, aux)``.
    - ``takes_key=True``, ``has_aux=False``: ``loss_fn(model, batch, key) -> scalar``.
    - ``takes_key=True``, ``has_aux=True``: ``loss_fn(model, batch, key) -> (loss, aux)``.

    ``aux`` must be a ``dict[str, Array]`` and must not contain the key
    ``"loss"`` (that slot is the scalar loss). Aux keys are part of the
    compiled structure; their array values are traced.

    Invariants:
        - step() is eqx.filter_jit-decorated.
        - Gradients computed via eqx.filter_value_and_grad (has_aux when requested).
        - Optimizer.update() receives eqx.filter(state.model, eqx.is_array)
          as third argument (required for weight-decay compatibility).
        - Returns new ResumableState with incremented step and updated model/opt_state.
        - When takes_key is True, state.key is replaced with the unused half of
          one split (the other half is the step key).
        - Metrics dict always includes at minimum {"loss": scalar}.
    """

    loss_fn: Callable[..., Any]
    optimizer: optax.GradientTransformation
    has_aux: bool = eqx.field(static=True, default=False)
    takes_key: bool = eqx.field(static=True, default=False)

    @eqx.filter_jit
    def step(
        self,
        state: ResumableState,
        batch: Any,
    ) -> tuple[ResumableState, dict[str, jax.Array]]:
        """Execute one training step: loss, grad, update, return new state + metrics.

        Args:
            state: ResumableState with model, opt_state, step counter, and key.
            batch: PyTree with at minimum {"inputs": ..., "targets": ...}
                when ``takes_key`` is False. When ``takes_key`` is True the
                loss callable receives the batch unchanged.

        Returns:
            (new_state, metrics) where:
              - new_state: ResumableState with step += 1, updated model and opt_state.
              - metrics: dict[str, Array] with at minimum {"loss": scalar}.
                When ``has_aux`` is True, the aux dict's entries are included.
        """
        # takes_key / has_aux are static fields, so these branches are
        # resolved at trace time and do not depend on traced values.
        if self.takes_key:
            step_key, new_key = jax.random.split(state.key)
        else:
            new_key = state.key

        def loss_fn_inner(model):
            if self.takes_key:
                return self.loss_fn(model, batch, step_key)
            predictions = model(batch["inputs"])
            return self.loss_fn(predictions, batch["targets"])

        if self.has_aux:
            (loss, aux), grads = eqx.filter_value_and_grad(loss_fn_inner, has_aux=True)(state.model)
        else:
            loss, grads = eqx.filter_value_and_grad(loss_fn_inner)(state.model)
            aux = None

        # Update: CRITICAL — pass eqx.filter(state.model, eqx.is_array) as 3rd arg
        # This ensures weight decay and other weight-aware optimizers work correctly
        filtered_params = eqx.filter(state.model, eqx.is_array)
        updates, new_opt_state = self.optimizer.update(grads, state.opt_state, filtered_params)

        # Apply updates to model
        new_model = eqx.apply_updates(state.model, updates)

        # Create new state with incremented step counter (and the carried key).
        new_state = eqx.tree_at(
            lambda s: (s.model, s.opt_state, s.step, s.key),
            state,
            (new_model, new_opt_state, state.step + 1, new_key),
        )

        if not self.has_aux:
            return new_state, {"loss": loss}

        if not isinstance(aux, dict):
            raise TypeError(
                "has_aux=True requires loss_fn to return (loss, dict[str, Array]), "
                f"got aux of type {type(aux).__name__}"
            )
        if "loss" in aux:
            raise ValueError(
                "aux dict must not include 'loss'; Trainer stores the scalar loss there"
            )
        return new_state, {"loss": loss, **aux}
