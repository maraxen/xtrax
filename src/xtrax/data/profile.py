"""Throughput and stall measurement for an input pipeline.

The function times a simulated train step (sleep ``step_s``, then the next
batch) and checks itself with two Grain pipelines it builds: a slow source
that must stall, and an instant source that must not.
"""

import time
from dataclasses import dataclass
from typing import Any

from xtrax.data.pipeline import build_input_pipeline
from xtrax.profiling.record import ProbeRecord

# Control loops are short and independent of the caller's step_s. The
# negative source sleeps longer than its step, with one read thread and a
# prefetch of 1, so the step cannot hide the read. The positive step is
# long enough that a buffer pop stays under 5% of the window.
_NEG_DELAY_S = 0.04
_NEG_STEP_S = 0.01
_NEG_STEPS = 3
_NEG_WARMUP = 1
_POS_STEP_S = 0.1
_POS_STEPS = 4
_POS_WARMUP = 2
_CONTROL_SOURCE_LEN = 8

__all__ = [
    "InputPipelineControlError",
    "InputPipelineProfile",
    "PipelineControls",
    "profile_input_pipeline",
]


class InputPipelineControlError(RuntimeError):
    """A built-in input-pipeline control missed its wait-fraction bound."""


@dataclass(frozen=True, slots=True)
class PipelineControls:
    """Wait fractions for the slow-source and instant-source controls."""

    negative_wait_fraction: float
    positive_wait_fraction: float
    negative_passed: bool
    positive_passed: bool

    @property
    def passed(self) -> bool:
        """True when both control bounds hold."""
        return self.negative_passed and self.positive_passed


@dataclass(frozen=True, slots=True)
class InputPipelineProfile:
    """One simulated-loop measurement plus the control block."""

    examples_per_s: float
    wait_fraction: float
    controls: PipelineControls
    controls_passed: bool
    records: tuple[ProbeRecord, ...]


class _TimedSource:
    """Random-access source that sleeps ``delay_s`` on every read."""

    def __init__(self, n: int, delay_s: float) -> None:
        self._n = n
        self._delay_s = delay_s

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: Any) -> Any:
        import numpy as np

        if self._delay_s:
            time.sleep(self._delay_s)
        return np.int32(int(index))


def _example_count(batch: Any) -> int:
    shape = getattr(batch, "shape", None)
    if shape:
        return int(shape[0])
    if isinstance(batch, dict):
        for value in batch.values():
            nested = getattr(value, "shape", None)
            if nested:
                return int(nested[0])
    if isinstance(batch, (list, tuple)) and batch:
        nested = getattr(batch[0], "shape", None)
        if nested:
            return int(nested[0])
    return 1


def _block_until_ready(batch: Any) -> None:
    import jax

    for leaf in jax.tree_util.tree_leaves(batch):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()


def _close(iterator: Any) -> None:
    close = getattr(iterator, "close", None)
    if close is not None:
        close()


def _measure_loop(
    open_iter: Any,
    *,
    n_steps: int,
    step_s: float,
    warmup: int,
) -> tuple[float, float, int]:
    """Return ``(examples_per_s, wait_fraction, n_examples)``.

    ``examples_per_s`` counts examples over the whole window, including the
    ``step_s`` sleeps. ``wait_fraction`` is time blocked in ``next``
    (and ``block_until_ready``) divided by that same window.
    """
    iterator = iter(open_iter())
    try:
        for _ in range(warmup):
            _block_until_ready(next(iterator))
        waited = 0.0
        n_examples = 0
        started = time.perf_counter()
        for _ in range(n_steps):
            wait_started = time.perf_counter()
            batch = next(iterator)
            _block_until_ready(batch)
            waited += time.perf_counter() - wait_started
            n_examples += _example_count(batch)
            time.sleep(step_s)
        total = time.perf_counter() - started
    finally:
        _close(iterator)
    if total <= 0.0:
        total = 1e-12
    return n_examples / total, waited / total, n_examples


def _open_grain(
    source: Any,
    *,
    num_threads: int,
    prefetch_buffer_size: int,
) -> Any:
    def open_iter() -> Any:
        return build_input_pipeline(
            source,
            batch_size=1,
            seed=0,
            num_epochs=None,
            shuffle=False,
            num_threads=num_threads,
            prefetch_buffer_size=prefetch_buffer_size,
            mp_workers=0,
            drop_remainder=True,
        )

    return open_iter


def _run_controls() -> tuple[tuple[float, float, int], tuple[float, float, int]]:
    """Return negative and positive ``(examples_per_s, wait_fraction, n_examples)``."""
    slow = _TimedSource(_CONTROL_SOURCE_LEN, _NEG_DELAY_S)
    instant = _TimedSource(_CONTROL_SOURCE_LEN, 0.0)
    negative = _measure_loop(
        _open_grain(slow, num_threads=1, prefetch_buffer_size=1),
        n_steps=_NEG_STEPS,
        step_s=_NEG_STEP_S,
        warmup=_NEG_WARMUP,
    )
    positive = _measure_loop(
        _open_grain(instant, num_threads=1, prefetch_buffer_size=16),
        n_steps=_POS_STEPS,
        step_s=_POS_STEP_S,
        warmup=_POS_WARMUP,
    )
    return negative, positive


def _platform() -> str:
    import jax

    devices = jax.devices()
    if not devices:
        return "cpu"
    platform = getattr(devices[0], "platform", "cpu")
    if platform == "gpu":
        return "gpu"
    return "cpu"


def _record(
    probe_id: str,
    *,
    n_examples: int,
    examples_per_s: float,
    wait_fraction: float,
    kind: str,
) -> ProbeRecord:
    return ProbeRecord(
        probe_id=probe_id,
        stage=0,
        n_atoms=max(n_examples, 1),
        platform=_platform(),
        metrics={
            "examples_per_s": examples_per_s,
            "wait_fraction": wait_fraction,
        },
        config={"kind": kind},
    )


def profile_input_pipeline(
    iterable_or_factory: Any,
    *,
    step_s: float,
    n_steps: int,
    warmup: int = 0,
    raise_on_control_failure: bool = False,
    negative_min_wait: float = 0.5,
    positive_max_wait: float = 0.05,
) -> InputPipelineProfile:
    """Measure input-pipeline throughput and how long a train step waits on it.

    ``iterable_or_factory`` is either an iterable of batches or a zero-arg
    callable that returns one (called once per measurement so a generator
    can be restarted). Each step sleeps ``step_s`` after taking the next
    batch, standing in for train-step compute.

    The result's ``examples_per_s`` is examples consumed divided by the
    simulated-loop wall time (data waits plus ``step_s``). ``wait_fraction``
    is the fraction of that wall time spent blocked on the next batch.

    Controls run inside this call, on Grain pipelines this function builds,
    and do not use ``step_s`` / ``n_steps``:

    - Negative: each example sleeps a fixed delay, 1 read thread, prefetch
      1. ``wait_fraction`` must be greater than ``negative_min_wait`` (0.5).
    - Positive: an instant source. ``wait_fraction`` must be less than
      ``positive_max_wait`` (0.05).

    ``controls`` / ``controls_passed`` record the outcome. When
    ``raise_on_control_failure`` is true and a bound misses, raises
    ``InputPipelineControlError``.

    Emits one :class:`~xtrax.profiling.record.ProbeRecord` for the measured
    loop and one for each control (stage 0, host-side). ``n_atoms`` is the
    example count of that loop; ``ProbeRecord`` requires it to be positive.

    Raises:
        ValueError: ``n_steps < 1`` or ``step_s <= 0``.
        InputPipelineControlError: A control missed its bound and
            ``raise_on_control_failure`` is true.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}.")
    if step_s <= 0:
        raise ValueError(f"step_s must be positive, got {step_s}.")

    if callable(iterable_or_factory):

        def open_subject() -> Any:
            return iterable_or_factory()

    else:

        def open_subject() -> Any:
            return iterable_or_factory

    examples_per_s, wait_fraction, n_examples = _measure_loop(
        open_subject, n_steps=n_steps, step_s=step_s, warmup=warmup
    )
    (
        (negative_rate, negative_wait, negative_examples),
        (
            positive_rate,
            positive_wait,
            positive_examples,
        ),
    ) = _run_controls()
    negative_passed = negative_wait > negative_min_wait
    positive_passed = positive_wait < positive_max_wait
    controls = PipelineControls(
        negative_wait_fraction=negative_wait,
        positive_wait_fraction=positive_wait,
        negative_passed=negative_passed,
        positive_passed=positive_passed,
    )
    if raise_on_control_failure and not controls.passed:
        raise InputPipelineControlError(
            "input-pipeline controls failed: "
            f"negative wait_fraction={negative_wait:.4f} "
            f"(required > {negative_min_wait}), "
            f"positive wait_fraction={positive_wait:.4f} "
            f"(required < {positive_max_wait})."
        )
    records = (
        _record(
            "input_pipeline",
            n_examples=n_examples,
            examples_per_s=examples_per_s,
            wait_fraction=wait_fraction,
            kind="measured",
        ),
        _record(
            "input_pipeline_negative_control",
            n_examples=negative_examples,
            examples_per_s=negative_rate,
            wait_fraction=negative_wait,
            kind="negative_control",
        ),
        _record(
            "input_pipeline_positive_control",
            n_examples=positive_examples,
            examples_per_s=positive_rate,
            wait_fraction=positive_wait,
            kind="positive_control",
        ),
    )
    return InputPipelineProfile(
        examples_per_s=examples_per_s,
        wait_fraction=wait_fraction,
        controls=controls,
        controls_passed=controls.passed,
        records=records,
    )
