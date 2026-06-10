"""Tests for loss combinators."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from xtrax.training.loss import MultiTaskLoss, WeightedLoss
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
            loss1(preds[0], targets[0])
            + loss2(preds[1], targets[1])
            + loss3(preds[2], targets[2])
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
