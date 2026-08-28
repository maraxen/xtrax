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
from xtrax.telemetry.callback import TelemetryCallback
from xtrax.telemetry.ledger import RunLedger
from xtrax.telemetry.record import KIND_EVAL, KIND_TRAIN, STATUS_FAILED
from xtrax.training.step import SafetyTrainStep
from xtrax.training.trainer import Trainer
from xtrax.training.types import Callback, LossFunction, ResumableState


def _resolve_ledger(ledger: Any, run_id: str | None, kind: str) -> tuple[RunLedger, bool]:
    """Return ``(ledger, owns_it)``, opening one fail-closed if none was given.

    Ownership matters: a caller-supplied ledger spans more than this call (a
    sweep, a resume chain) and must not be closed here, while one opened here is
    this call's responsibility to close exactly once.

    ``new_run_id`` is imported lazily to keep ``xtrax.engine`` free of an eager
    dependency on ``xtrax.run``, which imports back into the engine's neighbours.
    """
    if ledger is not None:
        return ledger, False
    from xtrax.run.ident import new_run_id

    return RunLedger.open(run_id or new_run_id(), kind=kind), True


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
        *,
        ledger: Any = None,
        run_id: str | None = None,
    ) -> ResumableState:
        """Execute multi-epoch training with callback hooks.

        Telemetry is enforced here rather than in the CLI, because this is the
        only chokepoint that also covers direct library use. If ``ledger`` is
        None a :class:`~xtrax.telemetry.RunLedger` is opened for the duration of
        the call and closed with exactly one row; if it cannot be opened,
        ``LedgerUnavailableError`` propagates and the run does not start.
        Provenance cannot be captured retroactively, so refusing up front is the
        only honest option. Set XTRAX_TELEMETRY_OPTOUT=1 to proceed anyway --
        that still writes a row, marked non-citable.

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

        # Open the run ledger (fail-closed) unless the caller supplied one.
        ledger, owns_ledger = _resolve_ledger(ledger, run_id, KIND_TRAIN)
        telemetry = TelemetryCallback(ledger)
        # Local, not the static field: the telemetry callback is appended per
        # call so an Engine constructed with callbacks=() is still instrumented.
        callbacks = (*self.callbacks, telemetry)

        try:
            # Fire on_train_start hook
            for cb in callbacks:
                cb.on_train_start(state)

            if resume:
                for cb in callbacks:
                    cb.on_resume(state)

            # Main training loop: num_epochs iterations
            for epoch in range(num_epochs):
                # Fire on_epoch_start hook
                for cb in callbacks:
                    cb.on_epoch_start(state, epoch)

                # Iterate through this epoch's data
                # Note: data.train_iter() is a fresh generator each call
                for batch in data.train_iter():
                    # Capture the executed IR once, on the first batch. That
                    # first step IS the compile, so this lands at the compile
                    # boundary with the workload's true shape signature; the
                    # once-only guard lives in TelemetryCallback.
                    telemetry.capture_ir_for(self.trainer.step, state, batch)

                    # Fire on_step_start hook
                    for cb in callbacks:
                        cb.on_step_start(state)

                    # Execute training step
                    state, metrics = self.trainer.step(state, batch)

                    # Fire on_step_end hook asynchronously
                    for cb in callbacks:
                        # Convert callback call to coroutine (wrap in async function)
                        async def fire_step_end(callback, s, m):
                            callback.on_step_end(s, m)

                        await callback_handler.submit(fire_step_end(cb, state, metrics))

                # Wait for all pending step callbacks to complete before next epoch
                await callback_handler.wait_all()

                # Fire on_epoch_end hook
                for cb in callbacks:
                    cb.on_epoch_end(state, epoch)

                # Save checkpoint after epoch if requested
                if manager is not None:
                    save_checkpoint(manager, state)

        except BaseException as exc:
            # A crashed run is when the record matters most; mark it before the
            # finally block writes the row, then let the exception propagate.
            if owns_ledger:
                ledger.set_status(STATUS_FAILED, f"run raised {type(exc).__name__}: {exc}")
            raise
        finally:
            # Fire on_train_end hook (always, even on exception)
            for cb in callbacks:
                cb.on_train_end(state)
            if owns_ledger and not ledger.closed:
                ledger.close()

        return state

    async def eval(
        self,
        state: ResumableState,
        data: DataModule | DataIterLike,
        loss_fn: LossFunction | None = None,
        *,
        ledger: Any = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate model on a dataset (no training step).

        Telemetry is enforced here exactly as it is in ``fit``: inference runs
        are runs, and an evaluation whose numbers get cited needs the same
        reconstructable provenance as the training that produced the weights.
        Before this, eval persisted nothing at all -- metrics were returned in
        memory and the run left no trace.

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
        ledger, owns_ledger = _resolve_ledger(ledger, run_id, KIND_EVAL)
        telemetry = TelemetryCallback(ledger)
        validation_callbacks = (*self.validation_callbacks, telemetry)

        # Fire on_train_start hook on validation_callbacks
        for cb in validation_callbacks:
            cb.on_train_start(state)

        try:
            # Wrap model in inference mode (disables dropout, stochastic layers, etc.)
            inference_model = eqx.nn.inference_mode(state.model)
            eval_state = eqx.tree_at(lambda s: s.model, state, inference_model)

            # Collect metrics from each batch
            all_metrics = []

            for batch in data.eval_iter():
                # Capture the IR of the inference-mode step once, on the first
                # batch. inference_mode changes the graph (dropout and other
                # stochastic layers switch off), so this is genuinely different
                # IR from the training step -- not a duplicate of fit's capture.
                telemetry.capture_ir_for(self.trainer.step, eval_state, batch)

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

        except BaseException as exc:
            if owns_ledger:
                ledger.set_status(STATUS_FAILED, f"eval raised {type(exc).__name__}: {exc}")
            raise
        finally:
            # Fire on_train_end hook on validation_callbacks
            for cb in validation_callbacks:
                cb.on_train_end(state)
            if owns_ledger and not ledger.closed:
                ledger.close()

        return aggregated

    def fit_sync(
        self,
        state: ResumableState,
        data: DataModule | DataIterLike,
        num_epochs: int,
        checkpoint_dir: str | Path | None = None,
        resume: bool = False,
        *,
        ledger: Any = None,
        run_id: str | None = None,
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
                ledger=ledger,
                run_id=run_id,
            )
        )
