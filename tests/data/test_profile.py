"""profile_input_pipeline throughput, wait fraction, and built-in controls."""

import time

import numpy as np
import pytest

from xtrax.data.profile import InputPipelineControlError, profile_input_pipeline
from xtrax.profiling.record import ProbeRecord


def _batches():
    return (np.ones(4, dtype=np.float32) for _ in range(8))


class TestProfileInputPipeline:
    def test_reports_throughput_wait_and_passing_controls(self):
        """Controls: slow source stalls, instant source does not."""
        result = profile_input_pipeline(_batches, step_s=0.005, n_steps=3)
        assert result.examples_per_s > 0
        assert 0.0 <= result.wait_fraction <= 1.0
        assert result.controls.negative_wait_fraction > 0.5
        assert result.controls.positive_wait_fraction < 0.05
        assert result.controls.negative_passed
        assert result.controls.positive_passed
        assert result.controls.passed
        assert result.controls_passed
        assert len(result.records) == 3
        assert all(isinstance(record, ProbeRecord) for record in result.records)
        measured = result.records[0]
        assert measured.probe_id == "input_pipeline"
        assert measured.metrics["wait_fraction"] == pytest.approx(result.wait_fraction)
        assert measured.metrics["examples_per_s"] == pytest.approx(result.examples_per_s)
        assert result.records[1].metrics["wait_fraction"] > 0.5
        assert result.records[2].metrics["wait_fraction"] < 0.05
        assert [record.config["kind"] for record in result.records] == [
            "measured",
            "negative_control",
            "positive_control",
        ]

    def test_slow_source_wait_fraction_and_throughput(self):
        """A 30 ms batch read stalls the step and is counted in examples/s."""
        batch_len = 4
        n_steps = 4
        delay_s = 0.030
        step_s = 0.005
        sleeps: list[float] = []

        def factory():
            def generate():
                for _ in range(n_steps):
                    started = time.perf_counter()
                    time.sleep(delay_s)
                    sleeps.append(time.perf_counter() - started)
                    yield np.ones(batch_len, dtype=np.float32)

            return generate()

        result = profile_input_pipeline(factory, step_s=step_s, n_steps=n_steps, warmup=0)
        assert result.wait_fraction > 0.5
        total_time = sum(sleeps) + n_steps * step_s
        expected = batch_len * n_steps / total_time
        # 0.4 rejects a per-batch count of 1 (ratio ~0.25) and a /1000 scale,
        # and still allows step-sleep overshoot on a loaded machine.
        ratio = result.examples_per_s / expected
        assert 0.4 < ratio < 2.5

    def test_instant_source_barely_waits(self):
        """An instant source spends almost none of the step blocked on data."""

        def factory():
            return (np.ones(4, dtype=np.float32) for _ in range(8))

        result = profile_input_pipeline(factory, step_s=0.05, n_steps=4, warmup=0)
        assert result.wait_fraction < 0.1

    def test_warmup_steps_are_excluded(self):
        """The first warmup batch is outside the timed window."""
        n_steps = 4
        step_s = 0.04
        batch_len = 4

        def factory():
            def generate():
                time.sleep(0.30)
                yield np.ones(batch_len, dtype=np.float32)
                for _ in range(n_steps):
                    yield np.ones(batch_len, dtype=np.float32)

            return generate()

        result = profile_input_pipeline(factory, step_s=step_s, n_steps=n_steps, warmup=1)
        assert result.wait_fraction < 0.4
        # Instant measured batches run near batch_len/step_s (~100). Including
        # the 0.30 s warmup read drops that below ~40.
        assert result.examples_per_s > 50

    def test_pending_device_compute_counts_as_wait(self):
        """A device array that is still running is part of the wait, including on CPU."""
        import jax
        import jax.numpy as jnp

        @jax.jit
        def burn(values):
            updated = values
            for _ in range(24):
                updated = jnp.sin(updated) * 0.999 + 0.001
            return jnp.stack(
                [jnp.sum(updated), jnp.sum(updated * updated), jnp.min(updated), jnp.max(updated)]
            )

        data = jnp.ones((4_000_000,), dtype=jnp.float32)
        burn(data).block_until_ready()

        def factory():
            return (burn(data) for _ in range(4))

        result = profile_input_pipeline(factory, step_s=0.005, n_steps=3, warmup=0)
        assert result.wait_fraction > 0.5

    def test_raise_on_control_failure(self, monkeypatch):
        """A missed control bound raises when asked, and is reported otherwise."""
        monkeypatch.setattr(
            "xtrax.data.profile._run_controls",
            lambda: ((1.0, 0.1, 1), (1.0, 0.9, 1)),
        )
        with pytest.raises(InputPipelineControlError, match="controls failed"):
            profile_input_pipeline(
                _batches,
                step_s=0.001,
                n_steps=1,
                raise_on_control_failure=True,
            )
        result = profile_input_pipeline(_batches, step_s=0.001, n_steps=1)
        assert result.controls_passed is False
        assert result.controls.negative_passed is False
        assert result.controls.positive_passed is False

    def test_raise_when_only_the_negative_control_fails(self, monkeypatch):
        """A negative miss raises on its own; the positive control can still pass."""
        monkeypatch.setattr(
            "xtrax.data.profile._run_controls",
            lambda: ((1.0, 0.1, 1), (1.0, 0.01, 1)),
        )
        with pytest.raises(InputPipelineControlError, match="controls failed"):
            profile_input_pipeline(
                _batches,
                step_s=0.001,
                n_steps=1,
                raise_on_control_failure=True,
            )
        result = profile_input_pipeline(_batches, step_s=0.001, n_steps=1)
        assert result.controls.negative_passed is False
        assert result.controls.positive_passed is True
        assert result.controls_passed is False

    def test_raise_when_only_the_positive_control_fails(self, monkeypatch):
        """A positive miss raises on its own; the negative control can still pass."""
        monkeypatch.setattr(
            "xtrax.data.profile._run_controls",
            lambda: ((1.0, 0.9, 1), (1.0, 0.2, 1)),
        )
        with pytest.raises(InputPipelineControlError, match="controls failed"):
            profile_input_pipeline(
                _batches,
                step_s=0.001,
                n_steps=1,
                raise_on_control_failure=True,
            )
        result = profile_input_pipeline(_batches, step_s=0.001, n_steps=1)
        assert result.controls.negative_passed is True
        assert result.controls.positive_passed is False
        assert result.controls_passed is False

    def test_iterable_instance_is_measured(self):
        """A concrete iterable, not only a factory, is accepted."""
        result = profile_input_pipeline(
            [np.ones(2, dtype=np.float32) for _ in range(4)],
            step_s=0.002,
            n_steps=2,
        )
        assert result.examples_per_s > 0
