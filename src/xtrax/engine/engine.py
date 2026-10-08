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
from collections.abc import Callable, Iterator
from dataclasses import dataclass
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


def _resolve_ledger(
    ledger: Any,
    run_id: str | None,
    kind: str,
    context: dict[str, str] | None = None,
) -> tuple[RunLedger, bool]:
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

    return RunLedger.open(run_id or new_run_id(), kind=kind, context=context), True


@dataclass(frozen=True)
class EarlyStopping:
    """Host-side early stopping config for :meth:`Engine.fit`.

    ``patience`` is the number of consecutive non-improving checks tolerated
    before ``fit`` returns. The first observation sets the best value and does
    not consume patience. An observation improves when ``mode == "min"`` and
    ``value < best - min_delta``, or when ``mode == "max"`` and
    ``value > best + min_delta``.

    Checks come from ``validate_fn`` at each validation point. When no
    ``validate_fn`` is set, ``fit`` checks the last training-step metrics at
    the end of each epoch instead.

    ``fit`` returns the state from the step that exhausted patience. It does
    not roll the model back to the best checkpoint.

    Attributes:
        metric: Key in the metrics dict. The value must be a scalar array.
        mode: ``"min"`` or ``"max"``.
        patience: Positive number of non-improving checks to tolerate.
        min_delta: Non-negative minimum improvement. ``0`` accepts any strict
            improvement.
    """

    metric: str
    mode: str
    patience: int
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in ("min", "max"):
            raise ValueError(f"EarlyStopping.mode must be 'min' or 'max', got {self.mode!r}")
        if self.patience < 1:
            raise ValueError(f"EarlyStopping.patience must be >= 1, got {self.patience}")
        if self.min_delta < 0:
            raise ValueError(f"EarlyStopping.min_delta must be >= 0, got {self.min_delta}")


class _EarlyStopTracker:
    """Host-side patience counter. Metric values are converted with float()."""

    def __init__(self, config: EarlyStopping):
        self.config = config
        self.best: float | None = None
        self.wait = 0

    def update(self, metrics: dict[str, Any]) -> bool:
        if self.config.metric not in metrics:
            raise KeyError(
                f"early stopping metric {self.config.metric!r} is not in metrics {sorted(metrics)}"
            )
        value = float(metrics[self.config.metric])
        if self.best is None or _improved(self.config, self.best, value):
            self.best = value
            self.wait = 0
            return False
        self.wait += 1
        return self.wait >= self.config.patience


def _improved(config: EarlyStopping, best: float, value: float) -> bool:
    if config.mode == "min":
        return value < best - config.min_delta
    return value > best + config.min_delta


def _require_positive(name: str, value: int | None) -> None:
    if value is not None and value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")


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
        context: dict[str, str] | None = None,
        validate_fn: Callable[[ResumableState], dict[str, Any]] | None = None,
        validate_every_steps: int | None = None,
        validate_every_epochs: int | None = None,
        early_stop: EarlyStopping | None = None,
        checkpoint_every_steps: int | None = None,
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

        Iterates through data.train_iter() up to num_epochs times, calling
        trainer.step once per batch. State is incremented by 1 per batch.
        Early stopping returns before num_epochs is exhausted; that return is
        a completed run, not a failed one, and still closes the ledger.

        If checkpoint_dir is set, saves state after each epoch via orbax.
        ``checkpoint_every_steps`` additionally saves whenever the post-step
        counter is a positive multiple of that cadence. The epoch-end save
        still runs, including on the epoch where early stopping fires.

        ``validate_fn(state)`` returns a metrics dict. It runs on the host,
        outside the step jit. With neither cadence set, it runs once per epoch.
        ``validate_every_steps`` fires after a step whose counter is a multiple
        of N. ``validate_every_epochs`` fires after a completed epoch count that
        is a multiple of N. Setting both cadences runs both checks.

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
            resume: When True, fire ``on_resume`` after ``on_train_start``.
                This does not load a checkpoint; use :meth:`restore` for that.
            validate_fn: Optional host callable ``(state) -> metrics dict``.
            validate_every_steps: Step cadence for ``validate_fn``.
            validate_every_epochs: Epoch cadence for ``validate_fn``.
            early_stop: Optional :class:`EarlyStopping` config.
            checkpoint_every_steps: Extra checkpoint cadence, in steps.

        Returns:
            Final ResumableState after training completes (or early-stops).
        """
        _require_positive("validate_every_steps", validate_every_steps)
        _require_positive("validate_every_epochs", validate_every_epochs)
        _require_positive("checkpoint_every_steps", checkpoint_every_steps)
        if checkpoint_every_steps is not None and checkpoint_dir is None:
            raise ValueError("checkpoint_every_steps requires checkpoint_dir")
        if validate_fn is None and (
            validate_every_steps is not None or validate_every_epochs is not None
        ):
            raise ValueError("validate_every_steps/validate_every_epochs requires validate_fn")
        if (
            validate_fn is not None
            and validate_every_steps is None
            and validate_every_epochs is None
        ):
            validate_every_epochs = 1

        # Initialize checkpoint manager if needed
        if checkpoint_dir is not None:
            from xtrax.checkpoint.orbax import get_checkpoint_manager, save_checkpoint

            manager = get_checkpoint_manager(checkpoint_dir)
        else:
            manager = None

        # Initialize callback handler for async callback dispatch
        callback_handler = BoundedCallbackHandler(max_concurrent=4)

        # Open the run ledger (fail-closed) unless the caller supplied one.
        ledger, owns_ledger = _resolve_ledger(ledger, run_id, KIND_TRAIN, context)
        in_flight: BaseException | None = None
        telemetry = TelemetryCallback(ledger)
        # Local, not the static field: the telemetry callback is appended per
        # call so an Engine constructed with callbacks=() is still instrumented.
        callbacks = (*self.callbacks, telemetry)
        tracker = _EarlyStopTracker(early_stop) if early_stop is not None else None
        stopped = False

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
                epoch_metrics: dict[str, Any] | None = None
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

                    epoch_metrics = metrics
                    step_index = int(state.step)
                    if (
                        manager is not None
                        and checkpoint_every_steps is not None
                        and step_index % checkpoint_every_steps == 0
                    ):
                        save_checkpoint(manager, state)

                    if (
                        validate_fn is not None
                        and validate_every_steps is not None
                        and step_index % validate_every_steps == 0
                    ):
                        val_metrics = validate_fn(state)
                        if tracker is not None and tracker.update(val_metrics):
                            stopped = True
                            break

                # Wait for all pending step callbacks to complete before next epoch
                await callback_handler.wait_all()

                # Fire on_epoch_end hook
                for cb in callbacks:
                    cb.on_epoch_end(state, epoch)

                if (
                    not stopped
                    and validate_fn is not None
                    and validate_every_epochs is not None
                    and (epoch + 1) % validate_every_epochs == 0
                ):
                    val_metrics = validate_fn(state)
                    if tracker is not None and tracker.update(val_metrics):
                        stopped = True

                # No validation hook: early stopping watches the last training
                # metrics of the epoch (host-side, after the step jit returns).
                if (
                    not stopped
                    and tracker is not None
                    and validate_fn is None
                    and epoch_metrics is not None
                    and tracker.update(epoch_metrics)
                ):
                    stopped = True

                # Save checkpoint after epoch if requested, including the epoch
                # on which early stopping fired.
                if manager is not None:
                    save_checkpoint(manager, state)

                if stopped:
                    break

        except BaseException as exc:
            # A crashed run is when the record matters most; mark it before the
            # finally block writes the row, then let the exception propagate.
            in_flight = exc
            if owns_ledger:
                ledger.set_status(STATUS_FAILED, f"run raised {type(exc).__name__}: {exc}")
            raise
        finally:
            # Fire on_train_end hook (always, even on exception)
            for cb in callbacks:
                cb.on_train_end(state)
            # close_if_open, not close: a full disk while writing the row must not
            # replace the training exception that is already propagating.
            if owns_ledger:
                ledger.close_if_open(in_flight)

        return state

    async def eval(
        self,
        state: ResumableState,
        data: DataModule | DataIterLike,
        loss_fn: LossFunction | None = None,
        *,
        ledger: Any = None,
        run_id: str | None = None,
        context: dict[str, str] | None = None,
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
        ledger, owns_ledger = _resolve_ledger(ledger, run_id, KIND_EVAL, context)
        in_flight: BaseException | None = None
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
            in_flight = exc
            if owns_ledger:
                ledger.set_status(STATUS_FAILED, f"eval raised {type(exc).__name__}: {exc}")
            raise
        finally:
            # Fire on_train_end hook on validation_callbacks
            for cb in validation_callbacks:
                cb.on_train_end(state)
            if owns_ledger:
                ledger.close_if_open(in_flight)

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
        context: dict[str, str] | None = None,
        validate_fn: Callable[[ResumableState], dict[str, Any]] | None = None,
        validate_every_steps: int | None = None,
        validate_every_epochs: int | None = None,
        early_stop: EarlyStopping | None = None,
        checkpoint_every_steps: int | None = None,
    ) -> ResumableState:
        """Synchronous wrapper around fit() using asyncio.run().

        Convenience method for single-threaded use when asyncio event loop
        is not already running. Keyword arguments match :meth:`fit`.

        Args:
            state: Initial ResumableState
            data: DataModule
            num_epochs: Number of epochs
            checkpoint_dir: Optional checkpoint directory
            resume: Whether to fire on_resume. Does not load a checkpoint.
            validate_fn: Optional host validation callable. See :meth:`fit`.
            validate_every_steps: Step cadence for ``validate_fn``.
            validate_every_epochs: Epoch cadence for ``validate_fn``.
            early_stop: Optional :class:`EarlyStopping` config.
            checkpoint_every_steps: Extra checkpoint cadence, in steps.

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
                context=context,
                validate_fn=validate_fn,
                validate_every_steps=validate_every_steps,
                validate_every_epochs=validate_every_epochs,
                early_stop=early_stop,
                checkpoint_every_steps=checkpoint_every_steps,
            )
        )

    def restore(
        self,
        checkpoint_dir: str | Path,
        state_template: ResumableState,
        step: int | None = None,
    ) -> ResumableState:
        """Load a checkpoint, including ``state.key`` and ``state.extras``.

        The template supplies pytree structure (model, optimizer state, and
        the keys of ``extras``). Orbax matches that structure, then replaces
        the stored values. The returned ``key`` and ``extras`` are the
        checkpoint's values, not the template's. Extras keys absent from the
        template are not invented: the template must carry the same extras
        structure the checkpoint was saved with.

        Args:
            checkpoint_dir: Directory passed to ``fit(..., checkpoint_dir=)``.
            state_template: ResumableState whose structure matches the checkpoint.
            step: Checkpoint step. ``None`` loads the latest step.

        Returns:
            The loaded ResumableState.
        """
        from xtrax.checkpoint.orbax import get_checkpoint_manager, load_checkpoint

        manager = get_checkpoint_manager(checkpoint_dir)
        loaded = load_checkpoint(manager, state_template, step=step)
        if not isinstance(loaded, ResumableState):
            raise TypeError(
                f"checkpoint restore returned {type(loaded).__name__}, expected ResumableState"
            )
        return loaded
