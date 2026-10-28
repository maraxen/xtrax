"""Tests for the shipped negative controls (spec 261016 X7).

The controls must hold on CPU at B=8 (the sensitivity check) and B=64. A
deliberately broken threshold must fail with a reason naming host_loop.
"""

import pytest

from xtrax.profiling.controls import (
    HOST_LOOP,
    VMAPPED_TWIN,
    run_negative_controls,
)


@pytest.fixture(scope="module")
def controls_report():
    return run_negative_controls(batch_sizes=(8, 64), threshold=0.5, n_steady=2, n_wall=2)


def test_controls_hold_at_b8_and_b64(controls_report):
    assert controls_report.held, controls_report.reasons
    assert controls_report.reasons == ()


def test_host_loop_is_well_above_vmapped_twin(controls_report):
    for batch_size in (8, 64):
        host = controls_report.measurement(HOST_LOOP, batch_size)
        twin = controls_report.measurement(VMAPPED_TWIN, batch_size)
        assert host.exec_per_element >= controls_report.threshold
        assert twin.exec_per_element < controls_report.threshold
        assert host.exec_per_element > 10 * twin.exec_per_element
        assert twin.steady_compile_count == 0


def test_broken_threshold_fails_and_names_host_loop():
    report = run_negative_controls(batch_sizes=(8,), threshold=1000.0, n_steady=1, n_wall=1)
    assert not report.held
    assert any(HOST_LOOP in reason for reason in report.reasons)


def test_rejects_empty_or_nonpositive_batch_sizes():
    with pytest.raises(ValueError):
        run_negative_controls(batch_sizes=())
    with pytest.raises(ValueError):
        run_negative_controls(batch_sizes=(0,))


def test_default_threshold_comes_from_shared_rule():
    import inspect

    from xtrax.profiling.rules import EXEC_PER_ELEMENT_FLAG

    default = inspect.signature(run_negative_controls).parameters["threshold"].default
    assert default == EXEC_PER_ELEMENT_FLAG
