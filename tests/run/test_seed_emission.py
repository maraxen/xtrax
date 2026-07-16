"""Tests for the T2-23 run-level seed emission helper (#2181, AC-16, emission half)."""

import pytest

from xtrax.run.seed_emission import (
    SeedAssignment,
    SeedEmissionInputError,
    seed_assignment,
)


class TestSeedAssignment:
    def test_valid_seed_wraps_cleanly(self):
        assignment = seed_assignment(42)
        assert assignment.seed == 42

    def test_zero_seed_is_valid(self):
        assignment = seed_assignment(0)
        assert assignment.seed == 0

    def test_negative_seed_raises(self):
        with pytest.raises(SeedEmissionInputError, match="non-negative"):
            seed_assignment(-1)

    def test_is_frozen(self):
        assignment = seed_assignment(7)
        with pytest.raises(AttributeError):
            assignment.seed = 8


class TestAsBthRunFlags:
    def test_produces_correct_flag_pair(self):
        assignment = SeedAssignment(seed=42)
        assert assignment.as_bth_run_flags() == ["--seed", "42"]

    def test_flags_are_appendable_to_an_argv_list(self):
        assignment = SeedAssignment(seed=7)
        argv = ["bth", "run", "--campaign", "camp-1", *assignment.as_bth_run_flags(), "script.py"]
        assert argv == [
            "bth",
            "run",
            "--campaign",
            "camp-1",
            "--seed",
            "7",
            "script.py",
        ]

    def test_zero_seed_still_produces_flag(self):
        assignment = SeedAssignment(seed=0)
        assert assignment.as_bth_run_flags() == ["--seed", "0"]
