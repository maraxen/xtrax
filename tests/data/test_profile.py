"""profile_input_pipeline throughput, wait fraction, and built-in controls."""

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

    def test_iterable_instance_is_measured(self):
        """A concrete iterable, not only a factory, is accepted."""
        result = profile_input_pipeline(
            [np.ones(2, dtype=np.float32) for _ in range(4)],
            step_s=0.002,
            n_steps=2,
        )
        assert result.examples_per_s > 0
