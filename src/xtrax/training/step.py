"""SafetyTrainStep and create_train_step implementation (spec §3.14).

DEVIATION NOTE: Spec §3.14 shows @eqx.filter_jit on step() directly, but
err.throw() must run on the host (outside jit). The _step_jit pattern
supersedes that literal reading:
  - _step_jit is filter_jit'd and returns (err, (state, metrics))
  - step() is a plain Python wrapper calling _step_jit then err.throw()

This is the correct C1 resolution for checkify + JIT integration.
"""

from typing import Any

import equinox as eqx
import jax
import jax.experimental.checkify as checkify
import optax

from xtrax.safety.manager import SafetyManager
from xtrax.training.trainer import Trainer
from xtrax.training.types import LossFunction, ResumableState


class SafetyTrainStep(eqx.Module):
    """Safe training step wrapper with NaN/Inf detection via checkify.

    PUBLIC FIELDS (spec §3.14 requirement):
        trainer: Trainer instance for base training logic.
        safety_manager: SafetyManager for configuring checks.

    Architecture:
        - _step_jit: filter_jit'd method returning (err, (new_state, metrics))
        - step: Plain Python wrapper calling _step_jit then err.throw() on host

    This design ensures err.throw() (which raises Python exceptions) runs outside
    the JIT boundary, allowing proper exception handling on the host.
    """

    trainer: Trainer  # PUBLIC field (spec §3.14)
    safety_manager: SafetyManager  # PUBLIC field (spec §3.14)

    @eqx.filter_jit
    def _step_jit(self, state: ResumableState, batch: Any):
        """JIT-compiled checkified step.

        Returns:
            (err, (new_state, metrics)) where:
            - err: checkify.Error object (to be raised on host via err.throw())
            - new_state: ResumableState with incremented step
            - metrics: dict[str, Array] with at minimum {"loss": scalar}
        """
        # Wrap trainer.step with checkify to detect NaN/Inf
        # In a jit-in-jit context, the inner trainer.step's filter_jit is
        # stripped by JAX, and checkify instruments the computation correctly.
        checked_step = checkify.checkify(self.trainer.step, errors=checkify.float_checks)
        return checked_step(state, batch)  # returns (err, (new_state, metrics))

    def step(
        self,
        state: ResumableState,
        batch: Any,
    ) -> tuple[ResumableState, dict[str, jax.Array]]:
        """Execute one training step with optional safety checks.

        If safety_manager.enabled is False, delegates directly to trainer.step
        without checkify overhead.

        Args:
            state: ResumableState with model, opt_state, and step counter.
            batch: PyTree with at minimum {"inputs": ..., "targets": ...}.

        Returns:
            (new_state, metrics) where:
            - new_state: ResumableState with step += 1, updated model and opt_state.
            - metrics: dict[str, Array] with at minimum {"loss": scalar}.

        Raises:
            checkify.Error: If NaN/Inf/overflow detected (when enabled=True).
        """
        if not self.safety_manager.enabled:
            # Safety disabled: direct delegation to trainer.step (zero overhead)
            return self.trainer.step(state, batch)

        # Safety enabled: use _step_jit with checkify
        err, (new_state, metrics) = self._step_jit(state, batch)
        # CRITICAL: err.throw() runs on HOST, outside JIT boundary
        # This allows proper Python exception handling
        err.throw()
        return new_state, metrics


def create_train_step(
    loss_fn: LossFunction,
    optimizer: optax.GradientTransformation,
    safety: bool = False,
    safety_manager: SafetyManager | None = None,
) -> Trainer | SafetyTrainStep:
    """Factory for training step: returns Trainer or SafetyTrainStep.

    Args:
        loss_fn: Loss function callable(predictions, targets) -> scalar Array.
        optimizer: optax.GradientTransformation for parameter updates.
        safety: If False, return Trainer. If True, return SafetyTrainStep.
        safety_manager: SafetyManager instance (only used if safety=True).
                       If None and safety=True, create a default SafetyManager
                       with enabled=True, check_nans=True, check_infs=True.

    Returns:
        Trainer if safety=False.
        SafetyTrainStep if safety=True.
    """
    trainer = Trainer(loss_fn=loss_fn, optimizer=optimizer)

    if not safety:
        return trainer

    # safety=True: wrap in SafetyTrainStep
    if safety_manager is None:
        safety_manager = SafetyManager(enabled=True, check_nans=True, check_infs=True)

    return SafetyTrainStep(trainer=trainer, safety_manager=safety_manager)
