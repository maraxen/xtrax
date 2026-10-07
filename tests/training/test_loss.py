"""Tests for loss combinators."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from xtrax.training.loss import ComposedLoss, MultiTaskLoss, WeightedLoss
from xtrax.training.types import LossFunction


# Simple test loss functions
def simple_loss(predictions, targets) -> jnp.ndarray:
    """Basic MSE loss."""
    return jnp.mean((predictions - targets) ** 2)


def constant_loss(predictions, targets) -> jnp.ndarray:
    """Loss that always returns 2.0."""
    return jnp.array(2.0)


class TestWeightedLoss:
    """Test suite for WeightedLoss combinator."""

    def test_weighted_loss_basic(self):
        """WeightedLoss should multiply loss by weight."""
        loss_fn = WeightedLoss(loss_fn=simple_loss, weight=3.0)
        preds = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.5, 2.5, 3.5])

        result = loss_fn(preds, targets)
        expected = 3.0 * simple_loss(preds, targets)

        assert jnp.allclose(result, expected)

    def test_weighted_loss_is_loss_function(self):
        """WeightedLoss should be instance of LossFunction protocol."""
        loss_fn = WeightedLoss(loss_fn=simple_loss, weight=2.0)
        assert isinstance(loss_fn, LossFunction)

    def test_weighted_loss_weight_is_static(self):
        """weight field must be static (not in JAX array filter)."""
        loss_fn = WeightedLoss(loss_fn=simple_loss, weight=2.5)
        # Filter out all JAX arrays — weight should not appear
        array_leaves = eqx.filter(loss_fn, eqx.is_array)
        # Directly assert that no JAX array leaves exist
        assert jax.tree_util.tree_leaves(array_leaves) == []

        # More direct: weight is a Python float, not an Array
        assert isinstance(loss_fn.weight, float)

    def test_weighted_loss_zero_weight(self):
        """WeightedLoss with weight=0 should return 0."""
        loss_fn = WeightedLoss(loss_fn=simple_loss, weight=0.0)
        preds = jnp.array([1.0, 2.0])
        targets = jnp.array([3.0, 4.0])

        result = loss_fn(preds, targets)
        assert jnp.allclose(result, jnp.array(0.0))

    def test_weighted_loss_negative_weight(self):
        """WeightedLoss with negative weight should be supported."""
        loss_fn = WeightedLoss(loss_fn=simple_loss, weight=-1.5)
        preds = jnp.array([1.0, 2.0])
        targets = jnp.array([2.0, 3.0])

        result = loss_fn(preds, targets)
        expected = -1.5 * simple_loss(preds, targets)
        assert jnp.allclose(result, expected)


class TestMultiTaskLoss:
    """Test suite for MultiTaskLoss combinator."""

    def test_multitask_loss_basic(self):
        """MultiTaskLoss should sum weighted losses."""
        loss1 = WeightedLoss(loss_fn=simple_loss, weight=1.0)
        loss2 = WeightedLoss(loss_fn=constant_loss, weight=2.0)
        multi_loss = MultiTaskLoss(losses=(loss1, loss2))

        preds1 = jnp.array([1.0, 2.0])
        preds2 = jnp.array([3.0, 4.0])
        targets1 = jnp.array([1.5, 2.5])
        targets2 = jnp.array([3.5, 4.5])

        result = multi_loss((preds1, preds2), (targets1, targets2))

        # Expected: sum of weighted losses
        expected = loss1(preds1, targets1) + loss2(preds2, targets2)

        assert jnp.allclose(result, expected)

    def test_multitask_loss_is_loss_function(self):
        """MultiTaskLoss should be instance of LossFunction protocol."""
        loss1 = WeightedLoss(loss_fn=simple_loss, weight=1.0)
        multi_loss = MultiTaskLoss(losses=(loss1,))
        assert isinstance(multi_loss, LossFunction)

    def test_multitask_loss_length_mismatch_predictions(self):
        """MultiTaskLoss raises AssertionError if predictions length mismatches."""
        loss1 = WeightedLoss(loss_fn=simple_loss, weight=1.0)
        loss2 = WeightedLoss(loss_fn=simple_loss, weight=1.0)
        multi_loss = MultiTaskLoss(losses=(loss1, loss2))

        preds1 = jnp.array([1.0, 2.0])
        targets1 = jnp.array([1.5, 2.5])
        targets2 = jnp.array([2.5, 3.5])

        # Only 1 prediction but 2 losses
        with pytest.raises(AssertionError):
            multi_loss((preds1,), (targets1, targets2))

    def test_multitask_loss_length_mismatch_targets(self):
        """MultiTaskLoss should raise AssertionError if targets length mismatches."""
        loss1 = WeightedLoss(loss_fn=simple_loss, weight=1.0)
        loss2 = WeightedLoss(loss_fn=simple_loss, weight=1.0)
        multi_loss = MultiTaskLoss(losses=(loss1, loss2))

        preds1 = jnp.array([1.0, 2.0])
        preds2 = jnp.array([2.0, 3.0])
        targets1 = jnp.array([1.5, 2.5])

        # 2 predictions but only 1 target
        with pytest.raises(AssertionError):
            multi_loss((preds1, preds2), (targets1,))

    def test_multitask_loss_length_mismatch_losses(self):
        """MultiTaskLoss should raise AssertionError if losses length mismatches."""
        loss1 = WeightedLoss(loss_fn=simple_loss, weight=1.0)
        multi_loss = MultiTaskLoss(losses=(loss1,))

        preds1 = jnp.array([1.0, 2.0])
        preds2 = jnp.array([2.0, 3.0])
        targets1 = jnp.array([1.5, 2.5])
        targets2 = jnp.array([2.5, 3.5])

        # 2 predictions/targets but only 1 loss
        with pytest.raises(AssertionError):
            multi_loss((preds1, preds2), (targets1, targets2))

    def test_multitask_loss_three_tasks(self):
        """MultiTaskLoss should handle 3+ tasks."""
        loss1 = WeightedLoss(loss_fn=simple_loss, weight=1.0)
        loss2 = WeightedLoss(loss_fn=simple_loss, weight=2.0)
        loss3 = WeightedLoss(loss_fn=simple_loss, weight=3.0)
        multi_loss = MultiTaskLoss(losses=(loss1, loss2, loss3))

        preds = (
            jnp.array([1.0, 2.0]),
            jnp.array([2.0, 3.0]),
            jnp.array([3.0, 4.0]),
        )
        targets = (
            jnp.array([1.5, 2.5]),
            jnp.array([2.5, 3.5]),
            jnp.array([3.5, 4.5]),
        )

        result = multi_loss(preds, targets)

        expected = (
            loss1(preds[0], targets[0]) + loss2(preds[1], targets[1]) + loss3(preds[2], targets[2])
        )

        assert jnp.allclose(result, expected)

    def test_weight_schedule_applied(self):
        """MultiTaskLoss with weight_schedule should apply schedule multiplier."""
        loss1 = WeightedLoss(loss_fn=simple_loss, weight=1.0)
        loss2 = WeightedLoss(loss_fn=constant_loss, weight=2.0)

        # weight_schedule returns a scalar multiplier
        multi_loss = MultiTaskLoss(
            losses=(loss1, loss2), weight_schedule=lambda step: jnp.array(2.0)
        )

        preds1 = jnp.array([1.0, 2.0])
        preds2 = jnp.array([3.0, 4.0])
        targets1 = jnp.array([1.5, 2.5])
        targets2 = jnp.array([3.5, 4.5])

        result = multi_loss((preds1, preds2), (targets1, targets2), step=0)

        # Expected: 2.0 * (sum of unscheduled losses)
        unscheduled = loss1(preds1, targets1) + loss2(preds2, targets2)
        expected = 2.0 * unscheduled

        assert jnp.allclose(result, expected)

    def test_weight_schedule_step_dependent(self):
        """MultiTaskLoss weight_schedule should receive and use step parameter."""
        loss1 = WeightedLoss(loss_fn=constant_loss, weight=1.0)

        # weight_schedule depends on step: returns (step + 1)
        multi_loss = MultiTaskLoss(
            losses=(loss1,), weight_schedule=lambda step: jnp.array(float(step + 1))
        )

        preds = jnp.array([1.0, 2.0])
        targets = jnp.array([1.5, 2.5])

        # At step=3, schedule should return 4.0
        result = multi_loss((preds,), (targets,), step=3)

        # Expected: 4.0 * (sum of unscheduled losses)
        unscheduled = loss1(preds, targets)
        expected = 4.0 * unscheduled

        assert jnp.allclose(result, expected)

    def test_weight_schedule_none_unchanged(self):
        """MultiTaskLoss with weight_schedule=None ignores step parameter."""
        loss1 = WeightedLoss(loss_fn=simple_loss, weight=1.0)
        loss2 = WeightedLoss(loss_fn=constant_loss, weight=2.0)

        multi_loss = MultiTaskLoss(losses=(loss1, loss2), weight_schedule=None)

        preds1 = jnp.array([1.0, 2.0])
        preds2 = jnp.array([3.0, 4.0])
        targets1 = jnp.array([1.5, 2.5])
        targets2 = jnp.array([3.5, 4.5])

        # Call with step=5, should be same as without step
        result_with_step = multi_loss((preds1, preds2), (targets1, targets2), step=5)
        result_without_step = multi_loss((preds1, preds2), (targets1, targets2))

        assert jnp.allclose(result_with_step, result_without_step)

    def test_weight_schedule_none_is_default(self):
        """MultiTaskLoss weight_schedule defaults to None."""
        loss1 = WeightedLoss(loss_fn=simple_loss, weight=1.0)
        multi_loss = MultiTaskLoss(losses=(loss1,))

        assert multi_loss.weight_schedule is None


_BACKEND_COMPILE = "/jax/core/compile/backend_compile_duration"


def _mse(predictions, targets, **_flags):
    return jnp.mean((predictions - targets) ** 2)


def _count_backend_compiles(fn) -> int:
    """Count JAX backend compiles during ``fn``. The listener is removed after."""
    seen: list[str] = []

    def listener(event, duration_secs, **kwargs):
        if event == _BACKEND_COMPILE:
            seen.append(event)

    jax.monitoring.register_event_duration_secs_listener(listener)
    try:
        fn()
    finally:
        jax.monitoring.unregister_event_duration_listener(listener)
    return len(seen)


class TestComposedLoss:
    """Traced-weight composer. Aux values are unweighted term scalars."""

    def test_weighted_sum_and_unweighted_aux(self):
        def mae(predictions, targets, **_flags):
            return jnp.mean(jnp.abs(predictions - targets))

        composed = ComposedLoss(
            terms=(("mse", _mse), ("mae", mae)),
            weights=jnp.array([2.0, 0.5]),
        )
        preds = jnp.array([1.0, 2.0, 4.0])
        targets = jnp.array([1.0, 0.0, 1.0])

        loss, aux = composed(preds, targets)

        assert jnp.allclose(aux["mse"], _mse(preds, targets))
        assert jnp.allclose(aux["mae"], mae(preds, targets))
        assert jnp.allclose(loss, 2.0 * aux["mse"] + 0.5 * aux["mae"])

    def test_weight_value_does_not_change_unweighted_aux(self):
        composed = ComposedLoss(terms=(("mse", _mse),), weights=jnp.array([1.0]))
        heavier = eqx.tree_at(lambda m: m.weights, composed, jnp.array([4.0]))
        preds = jnp.array([1.0, 3.0])
        targets = jnp.array([0.0, 1.0])

        loss_a, aux_a = composed(preds, targets)
        loss_b, aux_b = heavier(preds, targets)

        assert jnp.allclose(aux_a["mse"], aux_b["mse"])
        assert jnp.allclose(loss_b, 4.0 * loss_a)

    def test_per_batch_flag_and_placement_weights(self):
        def head_loss(index):
            def fn(predictions, targets, placement_weights, **_flags):
                err = (predictions[index] - targets[index]) ** 2
                return jnp.sum(err * placement_weights) / jnp.sum(placement_weights)

            return fn

        preds = (jnp.array([1.0, 3.0]), jnp.array([0.0, 4.0]))
        targets = (jnp.array([1.0, 1.0]), jnp.array([2.0, 2.0]))
        placement = (jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))
        composed = ComposedLoss(
            terms=(("h0", head_loss(0)), ("h1", head_loss(1))),
            weights=jnp.array([0.25, 0.75]),
            placement_weights=placement,
        )

        loss, aux = composed(preds, targets)
        manual_0 = head_loss(0)(preds, targets, placement_weights=placement[0])
        manual_1 = head_loss(1)(preds, targets, placement_weights=placement[1])
        assert jnp.allclose(aux["h0"], manual_0)
        assert jnp.allclose(aux["h1"], manual_1)
        assert jnp.allclose(loss, 0.25 * manual_0 + 0.75 * manual_1)

        def masked(predictions, targets, mask=None, placement_weights=None):
            weights = mask if placement_weights is None else placement_weights
            err = (predictions - targets) ** 2
            return jnp.sum(err * weights) / jnp.sum(weights)

        row_preds = jnp.array([1.0, 3.0])
        row_targets = jnp.array([1.0, 1.0])
        stored = jnp.array([1.0, 0.0])
        flagged = ComposedLoss(
            terms=(("mse", masked),),
            weights=jnp.array([1.0]),
            placement_weights=(stored,),
        )
        _stored_loss, stored_aux = flagged(row_preds, row_targets)
        assert jnp.allclose(stored_aux["mse"], masked(row_preds, row_targets, stored))

        override = jnp.array([0.0, 1.0])
        _over_loss, over_aux = flagged(row_preds, row_targets, placement_weights=override)
        assert jnp.allclose(over_aux["mse"], masked(row_preds, row_targets, override))

        batch_mask = jnp.array([1.0, 0.0, 1.0])
        batch_preds = jnp.array([1.0, 9.0, 3.0])
        batch_targets = jnp.array([0.0, 0.0, 0.0])
        with_flag = ComposedLoss(terms=(("mse", masked),), weights=jnp.array([2.0]))
        flagged_loss, flagged_aux = with_flag(batch_preds, batch_targets, mask=batch_mask)
        assert jnp.allclose(flagged_aux["mse"], masked(batch_preds, batch_targets, batch_mask))
        assert jnp.allclose(flagged_loss, 2.0 * flagged_aux["mse"])

    def test_traced_weight_does_not_recompile(self):
        """Changing a traced weight value must not backend-compile again."""
        composed = ComposedLoss(terms=(("mse", _mse),), weights=jnp.array([1.0]))
        preds = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([0.0, 0.0, 0.0])
        apply = eqx.filter_jit(ComposedLoss.__call__)
        apply(composed, preds, targets)
        cache_before = apply._cached._cache_size()

        heavier = eqx.tree_at(lambda m: m.weights, composed, jnp.array([4.0]))

        def second_call():
            out, _aux = apply(heavier, preds, targets)
            out.block_until_ready()

        compiles = _count_backend_compiles(second_call)
        assert compiles == 0
        assert apply._cached._cache_size() == cache_before

    def test_static_python_float_weight_does_recompile(self):
        """Negative control: a static Python-float weight must recompile.

        This fails if WeightedLoss.weight is accidentally traced. A second
        float has to miss the jit cache and backend-compile again.
        """

        def mse(predictions, targets):
            return jnp.mean((predictions - targets) ** 2)

        preds = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([0.0, 0.0, 0.0])
        apply = eqx.filter_jit(WeightedLoss.__call__)
        apply(WeightedLoss(loss_fn=mse, weight=1.0), preds, targets)
        cache_before = apply._cached._cache_size()

        def second_call():
            out = apply(WeightedLoss(loss_fn=mse, weight=4.0), preds, targets)
            out.block_until_ready()

        compiles = _count_backend_compiles(second_call)
        assert compiles >= 1
        assert apply._cached._cache_size() > cache_before
