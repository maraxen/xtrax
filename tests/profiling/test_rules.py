"""Tests for the shared steady-state rule (spec 261016 X5, rule half)."""

import pytest

from xtrax.profiling import rules
from xtrax.profiling.rules import EXEC_PER_ELEMENT_FLAG, THRESHOLD_STATUS, steady_state_flags


def test_threshold_default_is_the_calibrated_constant():
    assert EXEC_PER_ELEMENT_FLAG == 0.5
    assert rules.EXEC_PER_ELEMENT_FLAG is EXEC_PER_ELEMENT_FLAG


def test_clean_unit_has_no_flags():
    assert steady_state_flags(0.004, 0.0) == []


def test_dispatch_ratio_at_threshold_is_flagged():
    flags = steady_state_flags(EXEC_PER_ELEMENT_FLAG, 0.0)
    assert len(flags) == 1
    assert "exec/element" in flags[0]


def test_dispatch_ratio_just_below_threshold_passes():
    assert steady_state_flags(EXEC_PER_ELEMENT_FLAG - 1e-9, 0.0) == []


def test_host_loop_magnitude_is_flagged_with_status():
    flags = steady_state_flags(3.0, 0.0)
    assert len(flags) == 1
    assert "exec/element 3.000" in flags[0]
    assert THRESHOLD_STATUS in flags[0]
    assert "host-loop signature" in flags[0]


def test_steady_recompiles_are_flagged_and_named():
    flags = steady_state_flags(0.0, 2.0)
    assert len(flags) == 1
    assert "steady" in flags[0]
    assert "n_compilations" in flags[0]


def test_both_rules_fire_in_fixed_order():
    flags = steady_state_flags(3.0, 1.0)
    assert len(flags) == 2
    assert "n_compilations" in flags[0]
    assert "exec/element" in flags[1]


def test_custom_threshold_overrides_default():
    assert steady_state_flags(0.3, 0.0, threshold=0.2) != []
    assert steady_state_flags(0.3, 0.0) == []


@pytest.mark.parametrize("compiles", [0.0, -0.0])
def test_zero_compiles_is_not_flagged(compiles):
    assert steady_state_flags(0.0, compiles) == []


def test_profile_verb_uses_the_shared_rule():
    """Both verb paths must consult rules.steady_state_flags, not an inline copy."""
    import inspect

    from xtrax.cli import profile_verb

    source = inspect.getsource(profile_verb)
    assert source.count("steady_state_flags(") == 2
    assert profile_verb.EXEC_PER_ELEMENT_FLAG == EXEC_PER_ELEMENT_FLAG
    assert "EXEC_PER_ELEMENT_FLAG" in profile_verb.__all__
