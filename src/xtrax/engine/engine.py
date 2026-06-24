"""High-level training engine for xtrax (spec §3.18).

Engine orchestrates the training loop: iterates over data, calls trainer.step,
manages callbacks, and handles checkpointing.

Key invariants:
  - fit() is async; iterate fresh via data.train_iter() each epoch
  - eval() wraps model in eqx.nn.inference_mode before evaluation
  - Callback hooks fire in: train_start, epoch_start, step_start, step_end (async),
    epoch_end, train_end (skipping on_resume per DEVIATION NOTE)
  - fit_sync() delegates to asyncio.run(fit(...))
"""

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp

from xtrax.data.module import DataModule
from xtrax.engine.io import BoundedCallbackHandler
from xtrax.training.step import SafetyTrainStep
from xtrax.training.trainer import Trainer
from xtrax.training.types import Callback, LossFunction, ResumableState


@runtime_checkable
class TrainStepLike(Protocol):
    """Trainer or duck-typed step implementation used by Engine."""

    def step(self, state: ResumableState, batch: Any) -> tuple[ResumableState, Any]: ...


@runtime_checkable
class DataIterLike(Protocol):
    """DataModule or duck-typed data provider used by Engine."""

    def train_iter(self) -> Iterator[Any]: ...

    def eval_iter(self) -> Iterator[Any]: ...


class Engine(eqx.Module):
    """High-performance training engine.

    Manages training loops with callback hooks, checkpoint saving, and async
    callback execution. Accepts both Trainer and SafetyTrainStep for flexible
    safety configurations.

    Fields (all static — hold non-array Python objects):
        trainer: Trainer | SafetyTrainStep instance for step execution
        callbacks: Tuple of training callbacks (fired during fit)
        validation_callbacks: Tuple of validation callbacks (fired during eval)
    """

    trainer: Trainer | SafetyTrainStep | TrainStepLike = eqx.field(static=True)
    callbacks: tuple[Callback, ...] = eqx.field(static=True)
    validation_callbacks: tuple[Callback, ...] = eqx.field(default=(), static=True)

    async def fit(
        self,
        state: ResumableState,
        data: DataModule | DataIterLike,
        num_epochs: int,
        checkpoint_dir: str | Path | None = None,
        resume: bool = False,
    ) -> ResumableState:
        """Execute multi-epoch training with callback hooks.

        Iterates through data.train_iter() exactly num_epochs times, calling
        trainer.step once per batch. State is incremented by 1 per batch.

        If checkpoint_dir is set, saves state after each epoch via orbax.

        Fires callbacks in order:
          1. on_train_start (once)
          2. For each epoch:
             - on_epoch_start(state, epoch)
             - For each batch:
               - on_step_start(state)
               - trainer.step(state, batch)  [returns new_state, metrics]
               - on_step_end(state, metrics)  [async, via BoundedCallbackHandler]
             - on_epoch_end(state, epoch)
          3. on_train_end (once)

        Args:
            state: Initial ResumableState with model, opt_state, step counter
            data: DataModule with train_iter() generator
            num_epochs: Number of training epochs
            checkpoint_dir: Optional directory for saving checkpoints after each epoch

        Returns:
            Final ResumableState after training completes
        """
        # Initialize checkpoint manager if needed
        if checkpoint_dir is not None:
            from xtrax.checkpoint.orbax import get_checkpoint_manager, save_checkpoint

            manager = get_checkpoint_manager(checkpoint_dir)
        else:
            manager = None

        # Initialize callback handler for async callback dispatch
        callback_handler = BoundedCallbackHandler(max_concurrent=4)

        try:
            # Fire on_train_start hook
            for cb in self.callbacks:
                cb.on_train_start(state)

            if resume:
                for cb in self.callbacks:
                    cb.on_resume(state)

            # Main training loop: num_epochs iterations
            for epoch in range(num_epochs):
                # Fire on_epoch_start hook
                for cb in self.callbacks:
                    cb.on_epoch_start(state, epoch)

                # Iterate through this epoch's data
                # Note: data.train_iter() is a fresh generator each call
                for batch in data.train_iter():
                    # Fire on_step_start hook
                    for cb in self.callbacks:
                        cb.on_step_start(state)

                    # Execute training step
                    state, metrics = self.trainer.step(state, batch)

                    # Fire on_step_end hook asynchronously
                    for cb in self.callbacks:
                        # Convert callback call to coroutine (wrap in async function)
                        async def fire_step_end(callback, s, m):
                            callback.on_step_end(s, m)

                        await callback_handler.submit(fire_step_end(cb, state, metrics))

                # Wait for all pending step callbacks to complete before next epoch
                await callback_handler.wait_all()

                # Fire on_epoch_end hook
                for cb in self.callbacks:
                    cb.on_epoch_end(state, epoch)

                # Save checkpoint after epoch if requested
                if manager is not None:
                    save_checkpoint(manager, state)

        finally:
            # Fire on_train_end hook (always, even on exception)
            for cb in self.callbacks:
                cb.on_train_end(state)

        return state

    async def eval(
        self,
        state: ResumableState,
        data: DataModule | DataIterLike,
        loss_fn: LossFunction | None = None,
    ) -> dict[str, Any]:
        """Evaluate model on a dataset (no training step).

        Wraps state.model in eqx.nn.inference_mode before iteration.
        Collects metrics per batch, then aggregates via jax.tree.map(jnp.mean).

        Fires validation_callbacks only (not self.callbacks).

        Fires validation_callback hooks:
          - on_train_start at start
          - on_train_end at end

        Args:
            state: ResumableState with model to evaluate
            data: DataModule with eval_iter() generator
            loss_fn: Optional loss function; if provided, added to metrics dict

        Returns:
            Aggregated metrics dict[str, Array] with all keys averaged across batches
        """
        # Fire on_train_start hook on validation_callbacks
        for cb in self.validation_callbacks:
            cb.on_train_start(state)

        try:
            # Wrap model in inference mode (disables dropout, stochastic layers, etc.)
            inference_model = eqx.nn.inference_mode(state.model)
            eval_state = eqx.tree_at(lambda s: s.model, state, inference_model)

            # Collect metrics from each batch
            all_metrics = []

            for batch in data.eval_iter():
                # Call trainer.step to get metrics
                # (just for metric computation, state doesn't get updated in eval)
                batch_state, batch_metrics = self.trainer.step(eval_state, batch)

                # If loss_fn provided, compute and add loss to metrics
                if loss_fn is not None:
                    predictions = inference_model(batch["inputs"])
                    loss = loss_fn(predictions, batch["targets"])
                    batch_metrics = {**batch_metrics, "loss": loss}

                all_metrics.append(batch_metrics)

            # Aggregate metrics across batches
            if not all_metrics:
                aggregated = {}
            else:
                # Stack metrics and average across batch dimension
                aggregated = jax.tree.map(
                    lambda *xs: jnp.mean(jnp.stack(xs)),
                    *all_metrics,
                )

        finally:
            # Fire on_train_end hook on validation_callbacks
            for cb in self.validation_callbacks:
                cb.on_train_end(state)

        return aggregated

    def fit_sync(
        self,
        state: ResumableState,
        data: DataModule | DataIterLike,
        num_epochs: int,
        checkpoint_dir: str | Path | None = None,
        resume: bool = False,
    ) -> ResumableState:
        """Synchronous wrapper around fit() using asyncio.run().

        Convenience method for single-threaded use when asyncio event loop
        is not already running.

        Args:
            state: Initial ResumableState
            data: DataModule
            num_epochs: Number of epochs
            checkpoint_dir: Optional checkpoint directory
            resume: Whether to resume training from checkpoint

        Returns:
            Final ResumableState
        """
        return asyncio.run(
            self.fit(
                state,
                data,
                num_epochs,
                checkpoint_dir,
                resume=resume,
            )
        )
