"""Tests for the diversity-quota-semantic gate (T2-18, #2181, AC-12, F5, PM-6)."""

import pytest

from xtrax.loop.diversity_quota import (
    DiversityQuotaInputError,
    assert_diversity_quota,
    is_structural_mutation,
)


class TestNonStructuralMutations:
    def test_identical_source_is_not_structural(self) -> None:
        source = "def f(a, b):\n    return a + b\n"
        assert is_structural_mutation(source, source) is False

    def test_pure_identifier_rename_is_not_structural(self) -> None:
        previous = "def f(a, b):\n    return a + b\n"
        candidate = "def f(x, y):\n    return x + y\n"
        assert is_structural_mutation(previous, candidate) is False

    def test_commutative_reorder_of_addition_is_not_structural(self) -> None:
        previous = "def f(a, b):\n    return a + b\n"
        candidate = "def f(a, b):\n    return b + a\n"
        assert is_structural_mutation(previous, candidate) is False

    def test_commutative_reorder_of_multiplication_is_not_structural(self) -> None:
        previous = "def f(a, b):\n    return a * b\n"
        candidate = "def f(a, b):\n    return b * a\n"
        assert is_structural_mutation(previous, candidate) is False

    def test_rename_combined_with_commutative_reorder_is_not_structural(self) -> None:
        """The scenario the module docstring calls out explicitly: renaming AND reordering
        operands of a commutative op at once, both a and b positions changed."""
        previous = "def f(a, b):\n    return a + b\n"
        candidate = "def f(x, y):\n    return y + x\n"
        assert is_structural_mutation(previous, candidate) is False

    def test_whitespace_only_change_is_not_structural(self) -> None:
        previous = "def f(a, b):\n    return a + b\n"
        candidate = "def f(a, b):\n\n    return a + b\n"
        assert is_structural_mutation(previous, candidate) is False

    def test_nested_function_shadowing_rename_is_not_structural(self) -> None:
        """Regression test for a self-audit finding: identifier canonicalization must be
        scope-local. A module-global name map would collapse `inner`'s shadowed `a` (same
        string as `outer`'s own `a`) onto the same canonical id as `outer`'s `a` in the first
        source, while the second source's distinctly-named `inner(y)` would get its own id --
        making these two alpha-equivalent programs normalize differently (a false positive)."""
        previous = (
            "def outer(a):\n    def inner(a):\n        return a + 1\n    return inner(a) + 2\n"
        )
        candidate = (
            "def outer(x):\n    def inner(y):\n        return y + 1\n    return inner(x) + 2\n"
        )
        assert is_structural_mutation(previous, candidate) is False


class TestStructuralMutations:
    def test_different_operator_is_structural(self) -> None:
        previous = "def f(a, b):\n    return a + b\n"
        candidate = "def f(a, b):\n    return a - b\n"
        assert is_structural_mutation(previous, candidate) is True

    def test_non_commutative_operand_swap_is_structural(self) -> None:
        previous = "def f(a, b):\n    return a - b\n"
        candidate = "def f(a, b):\n    return b - a\n"
        assert is_structural_mutation(previous, candidate) is True

    def test_added_statement_is_structural(self) -> None:
        previous = "def f(a, b):\n    return a + b\n"
        candidate = "def f(a, b):\n    c = a + b\n    return c\n"
        assert is_structural_mutation(previous, candidate) is True

    def test_different_control_flow_is_structural(self) -> None:
        previous = "def f(a, b):\n    return a + b\n"
        candidate = "def f(a, b):\n    if a > b:\n        return a\n    return b\n"
        assert is_structural_mutation(previous, candidate) is True

    def test_variable_used_twice_vs_two_distinct_variables_is_structural(self) -> None:
        previous = "def f(a, b):\n    return a + a\n"
        candidate = "def f(a, b):\n    return a + b\n"
        assert is_structural_mutation(previous, candidate) is True


class TestInputValidation:
    def test_syntax_error_in_previous_source_raises(self) -> None:
        with pytest.raises(DiversityQuotaInputError, match="not valid Python"):
            is_structural_mutation("def f(:\n    pass\n", "def f():\n    pass\n")

    def test_syntax_error_in_candidate_source_raises(self) -> None:
        with pytest.raises(DiversityQuotaInputError, match="not valid Python"):
            is_structural_mutation("def f():\n    pass\n", "def f(:\n    pass\n")


class TestQuotaDecision:
    def test_quota_met_with_one_structural_mutation_in_window(self) -> None:
        decision = assert_diversity_quota([False, False, False, False, True], window_size=5)
        assert decision.quota_met is True
        assert decision.leap_path_required is False
        assert decision.window == (False, False, False, False, True)

    def test_quota_unmet_with_all_cosmetic_mutations_in_window(self) -> None:
        decision = assert_diversity_quota([False, False, False, False, False], window_size=5)
        assert decision.quota_met is False
        assert decision.leap_path_required is True

    def test_only_last_window_size_flags_are_considered(self) -> None:
        """A structural mutation outside the trailing window doesn't count."""
        decision = assert_diversity_quota([True, False, False, False, False, False], window_size=5)
        assert decision.quota_met is False
        assert decision.window == (False, False, False, False, False)

    def test_fewer_than_window_size_iterations_treats_quota_as_met(self) -> None:
        decision = assert_diversity_quota([False, False], window_size=5)
        assert decision.quota_met is True
        assert decision.leap_path_required is False
        assert decision.window == (False, False)

    def test_empty_history_treats_quota_as_met(self) -> None:
        decision = assert_diversity_quota([], window_size=5)
        assert decision.quota_met is True
        assert decision.window == ()

    def test_custom_window_size(self) -> None:
        decision = assert_diversity_quota([False, False, True], window_size=3)
        assert decision.quota_met is True
        assert decision.window == (False, False, True)
