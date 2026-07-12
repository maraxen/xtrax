"""Tests for the sealed EvaluateFn seam (T1-07, #3060, AC3)."""

import pytest

from xtrax.stages.evaluate import (
    EvaluatorAlreadySealedError,
    SealedEvaluatorRegistry,
    get_sealed_evaluator,
    seal_evaluator,
)


def _trivial_evaluator(frozen_context: object, candidate: object) -> dict[str, float]:
    del frozen_context, candidate
    return {"score": 0.0}


class TestSealedEvaluatorRegistry:
    def test_seal_then_get_returns_same_evaluator(self) -> None:
        registry = SealedEvaluatorRegistry()
        registry.seal(_trivial_evaluator)
        assert registry.get() is _trivial_evaluator

    def test_second_seal_raises(self) -> None:
        registry = SealedEvaluatorRegistry()
        registry.seal(_trivial_evaluator)
        with pytest.raises(EvaluatorAlreadySealedError, match="already sealed"):
            registry.seal(_trivial_evaluator)

    def test_get_before_seal_raises(self) -> None:
        registry = SealedEvaluatorRegistry()
        with pytest.raises(RuntimeError, match="no evaluator has been sealed"):
            registry.get()

    def test_no_mutation_api_on_registry(self) -> None:
        """Regression guard: the registry's public surface is exactly {seal, get} --
        no unseal/replace/reset method should ever be added.
        """
        public_methods = {
            name
            for name in dir(SealedEvaluatorRegistry)
            if not name.startswith("_") and callable(getattr(SealedEvaluatorRegistry, name))
        }
        assert public_methods == {"seal", "get"}

    def test_direct_attribute_assignment_is_blocked(self) -> None:
        """The named-method check above isn't enough on its own -- prove the underlying
        attribute itself can't be reassigned by a caller that reaches around seal().
        """
        registry = SealedEvaluatorRegistry()
        registry.seal(_trivial_evaluator)
        with pytest.raises(AttributeError, match="no attribute mutation"):
            registry._evaluator = lambda frozen_context, candidate: {"score": 999.0}  # noqa: ARG005
        assert registry.get() is _trivial_evaluator

    def test_new_attribute_assignment_is_also_blocked(self) -> None:
        registry = SealedEvaluatorRegistry()
        with pytest.raises(AttributeError):
            registry.some_new_field = "anything"


class TestOneHotSanityCheck:
    """AC3's own worked example (matches BATHOS.md's measurement-verification rule): feed a
    known-best candidate through a synthetic evaluator and assert it scores top.
    """

    def test_known_best_candidate_scores_top(self) -> None:
        known_best = "b"
        fitness_by_candidate = {"a": 0.0, "b": 1.0, "c": 0.0}

        def one_hot_evaluator(frozen_context: object, candidate: str) -> dict[str, float]:
            del frozen_context
            return {"score": fitness_by_candidate[candidate]}

        registry = SealedEvaluatorRegistry()
        registry.seal(one_hot_evaluator)
        evaluator = registry.get()

        scores = {c: evaluator(None, c)["score"] for c in fitness_by_candidate}
        assert scores[known_best] == 1.0
        assert scores[known_best] > scores["a"]
        assert scores[known_best] > scores["c"]
        assert max(scores, key=lambda c: scores[c]) == known_best


def test_module_level_functions_delegate_to_a_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Proves seal_evaluator()/get_sealed_evaluator() actually delegate to a working
    registry, without permanently sealing the real process-wide `_default_registry` for
    the rest of the pytest session -- monkeypatch substitutes a fresh registry for the
    duration of this test and restores the original afterward.
    """
    import xtrax.stages.evaluate as evaluate_module

    monkeypatch.setattr(evaluate_module, "_default_registry", SealedEvaluatorRegistry())

    seal_evaluator(_trivial_evaluator)
    assert get_sealed_evaluator() is _trivial_evaluator
    with pytest.raises(EvaluatorAlreadySealedError):
        seal_evaluator(_trivial_evaluator)
