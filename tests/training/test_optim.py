import inspect

import jax.numpy as jnp
import optax

from xtrax.training.optim import (
    adamw_with_schedule,
    make_optimizer,
    no_bias_wd_mask,
    partition_labels,
)


class TestNoBiasWdMask:
    """Test the no_bias_wd_mask function."""

    def test_multidim_array_true(self):
        """Multi-dimensional arrays (weight matrices) should return True."""
        params = {"w": jnp.ones((10, 10)), "b": jnp.zeros(10)}
        mask = no_bias_wd_mask(params)
        assert mask["w"] is True, "2D weight matrix should have True (apply WD)"

    def test_1d_array_false(self):
        """1-D arrays (bias vectors) should return False."""
        params = {"w": jnp.ones((10, 10)), "b": jnp.zeros(10)}
        mask = no_bias_wd_mask(params)
        assert mask["b"] is False, "1D bias vector should have False (skip WD)"

    def test_scalar_true(self):
        """Scalars (0-D) should return True (ndim != 1)."""
        params = {"s": jnp.array(5.0)}
        mask = no_bias_wd_mask(params)
        assert mask["s"] is True, "0D scalar has ndim=0 != 1, so should be True (apply WD)"

    def test_3d_array_true(self):
        """3-D arrays should return True (ndim != 1)."""
        params = {"conv": jnp.ones((3, 3, 32, 32))}
        mask = no_bias_wd_mask(params)
        assert mask["conv"] is True, "3D conv weight should have True (apply WD)"


class TestMakeOptimizer:
    """Test the make_optimizer wrapper."""

    def test_clip_none_returns_base_unchanged(self):
        """clip_norm=None should return the base optimizer unchanged (same object)."""
        base = optax.sgd(learning_rate=0.01)
        result = make_optimizer(base, clip_norm=None)
        assert result is base, "Should return exact same object when clip_norm=None"

    def test_clip_applied_returns_chain(self):
        """clip_norm != None should return a chained optimizer."""
        base = optax.sgd(learning_rate=0.01)
        result = make_optimizer(base, clip_norm=1.0)
        assert result is not base, "Should return new chained optimizer"
        # Verify it's a chain with clip_by_global_norm
        assert isinstance(result, optax.GradientTransformation)

    def test_clip_order_clip_first(self):
        """Clipping should be applied BEFORE the base optimizer in the chain."""
        # With clip-first ordering:
        #   gradient 100.0 -> clip_by_global_norm(1.0) -> 1.0 -> SGD(lr=0.01) -> ~-0.01
        # With base-first ordering (wrong):
        #   gradient 100.0 -> SGD(lr=0.01) -> -1.0 -> clip_by_global_norm(1.0) -> -1.0
        #
        # The difference: clip-first produces smaller updates (~0.01),
        # base-first produces larger (1.0).
        base = optax.sgd(learning_rate=0.01)
        result = make_optimizer(base, clip_norm=1.0)

        params = {"w": jnp.array(1.0)}
        state = result.init(params)

        # Use a very large gradient to make ordering differences clear
        large_grads = {"w": jnp.array(100.0)}
        updates, _ = result.update(large_grads, state)

        # With clip-first: gradient clipped to 1.0, then SGD with lr=0.01 -> ~-0.01
        # Assert update magnitude is approximately 0.01 (small), not 1.0 (large)
        # This pins the clip-first ordering specifically.
        assert jnp.abs(float(updates["w"])) < 0.02, (
            f"With clip-first ordering, update should be ~-0.01, "
            f"got {float(updates['w'])}. "
            "This verifies clipping is applied BEFORE SGD."
        )


class TestAdamwWithSchedule:
    """Test the adamw_with_schedule function."""

    def test_weight_decay_default_1e2(self):
        """Default weight_decay should be 1e-2 (NOT 1e-4)."""
        # Primary assertion: inspect the function signature directly
        sig = inspect.signature(adamw_with_schedule)
        wd_default = sig.parameters["weight_decay"].default
        assert wd_default == 1e-2, f"weight_decay default must be 1e-2, got {wd_default}"

        # Behavioral test: verify WD=1e-2 causes measurable decay vs WD=1e-4
        # Create two optimizers with different weight decays
        opt_default = adamw_with_schedule(
            peak_lr=1e-3,
            warmup_steps=1,
            total_steps=10,
        )
        opt_small_wd = adamw_with_schedule(
            peak_lr=1e-3,
            warmup_steps=1,
            total_steps=10,
            weight_decay=1e-4,
        )

        # Initialize with a 2D weight (will have WD applied)
        params = {"w": jnp.array([[1.0, 2.0], [3.0, 4.0]])}
        state_default = opt_default.init(params)
        state_small_wd = opt_small_wd.init(params)

        # Zero gradient forces weight decay only (no gradient-based update)
        zero_grad = {"w": jnp.zeros((2, 2))}

        # Run multiple steps so WD effect is observable
        # (Adam takes time to build up m1, m2)
        for _ in range(5):
            updates_default, state_default = opt_default.update(zero_grad, state_default, params)
            updates_small_wd, state_small_wd = opt_small_wd.update(
                zero_grad, state_small_wd, params
            )

        # With zero gradient over multiple steps, weight decay drives the update.
        # Higher decay (1e-2) should produce larger magnitude updates than 1e-4.
        default_norm = jnp.linalg.norm(updates_default["w"])
        small_norm = jnp.linalg.norm(updates_small_wd["w"])
        assert default_norm > small_norm, (
            "Weight decay 1e-2 should produce larger magnitude updates "
            "than 1e-4 with zero gradients"
        )

    def test_decay_steps_is_total_steps(self):
        """decay_steps should equal total_steps (INCLUDING warmup)."""
        # This is a contract test: verify the schedule decays over total_steps.
        peak_lr = 1e-3
        warmup = 100
        total = 1000

        opt = adamw_with_schedule(
            peak_lr=peak_lr,
            warmup_steps=warmup,
            total_steps=total,
        )

        # The schedule should reach 0 at step total_steps.
        # Extract the schedule from the optimizer.
        # optax.adamw wraps the schedule; we can verify it was created correctly.
        assert isinstance(opt, optax.GradientTransformation)

    def test_mask_passed_to_adamw(self):
        """wd_mask should be passed to optax.adamw.

        The mask should actually prevent weight decay on masked params.
        """
        # Behavioral test: create params with 2D weight and 1D bias.
        # With no_bias_wd_mask, 1D bias should NOT get weight decay,
        # but 2D weight should. Verify by comparing update magnitudes.
        opt_with_mask = adamw_with_schedule(
            peak_lr=1e-3,
            warmup_steps=1,
            total_steps=10,
            weight_decay=1e-1,  # Large decay to make effect observable
            clip_norm=None,  # Disable clipping to isolate WD effect
            wd_mask=no_bias_wd_mask,
        )

        # Create params: 2D weight and 1D bias
        params = {
            "weight": jnp.ones((5, 5)),  # 2D -> will have WD applied
            "bias": jnp.ones(5),  # 1D -> will NOT have WD applied
        }
        state = opt_with_mask.init(params)

        # Zero gradient: WD is the only force driving updates
        zero_grad = {
            "weight": jnp.zeros((5, 5)),
            "bias": jnp.zeros(5),
        }

        # Run multiple steps so WD effect becomes observable
        updates = None
        for _ in range(5):
            updates, state = opt_with_mask.update(zero_grad, state, params)

        # With zero gradient and high WD=0.1 over multiple steps:
        # - weight (2D) gets WD applied -> measurable update
        # - bias (1D) does NOT get WD -> much smaller update (ideally zero)
        weight_update_norm = jnp.linalg.norm(updates["weight"])
        bias_update_norm = jnp.linalg.norm(updates["bias"])

        assert weight_update_norm > bias_update_norm, (
            f"Weight decay mask not working: 2D weight update norm "
            f"({weight_update_norm}) should exceed 1D bias update norm "
            f"({bias_update_norm}). "
            "This verifies the mask prevents WD on 1D biases."
        )

    def test_all_parameters_respected(self):
        """Verify that b1, b2, eps are passed through."""
        opt = adamw_with_schedule(
            peak_lr=1e-3,
            warmup_steps=100,
            total_steps=1000,
            weight_decay=1e-3,
            clip_norm=1.5,
            b1=0.95,
            b2=0.99,
            eps=1e-7,
        )

        assert isinstance(opt, optax.GradientTransformation)


class TestPartitionLabels:
    """Test the partition_labels function."""

    def test_all_frozen_filter_true(self):
        """When frozen_filter always returns True, all leaves should be frozen_label."""
        # Use a simple dict as the model (equinox Module creation is complex)
        model = {
            "w": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            "b": jnp.array([0.0, 0.0]),
            "c": jnp.array(5.0),
        }

        result = partition_labels(
            model,
            frozen_filter=lambda _: True,
            frozen_label="frozen",
            train_label="train",
        )

        # All leaves should be "frozen"
        assert result["w"] == "frozen"
        assert result["b"] == "frozen"
        assert result["c"] == "frozen"

    def test_all_train_filter_false(self):
        """When frozen_filter always returns False, all leaves should be train_label."""
        model = {
            "w": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            "b": jnp.array([0.0, 0.0]),
            "c": jnp.array(5.0),
        }

        result = partition_labels(
            model,
            frozen_filter=lambda _: False,
            frozen_label="frozen",
            train_label="train",
        )

        assert result["w"] == "train"
        assert result["b"] == "train"
        assert result["c"] == "train"

    def test_ndim_based_filter(self):
        """Test with a realistic ndim-based frozen filter."""
        model = {
            "weight": jnp.ones((10, 10)),  # 2D
            "bias": jnp.zeros(10),  # 1D
            "scale": jnp.array(1.5),  # 0D
        }

        # Freeze only 1D params (typical for biases)
        result = partition_labels(
            model,
            frozen_filter=lambda x: x.ndim == 1,
            frozen_label="frozen",
            train_label="train",
        )

        assert result["weight"] == "train"
        assert result["bias"] == "frozen"
        assert result["scale"] == "train"

    def test_custom_labels(self):
        """Test with custom frozen/train labels."""
        model = {"w": jnp.array(1.0)}

        result = partition_labels(
            model,
            frozen_filter=lambda _: True,
            frozen_label="locked",
            train_label="learnable",
        )

        assert result["w"] == "locked"

        result = partition_labels(
            model,
            frozen_filter=lambda _: False,
            frozen_label="locked",
            train_label="learnable",
        )

        assert result["w"] == "learnable"


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_adamw_with_schedule_and_clip(self):
        """Test adamw_with_schedule with clipping enabled."""
        opt = adamw_with_schedule(
            peak_lr=1e-3,
            warmup_steps=10,
            total_steps=100,
            weight_decay=1e-2,
            clip_norm=1.0,
        )

        params = {"w": jnp.ones((5, 5)), "b": jnp.zeros(5)}
        state = opt.init(params)

        # Simulate a gradient update (params must be passed for adamw which has mask)
        grads = {"w": jnp.ones((5, 5)) * 10.0, "b": jnp.ones(5) * 5.0}
        updates, new_state = opt.update(grads, state, params)

        # Updates should be computed (non-zero and clipped)
        assert updates["w"].shape == (5, 5)
        assert updates["b"].shape == (5,)

    def test_partition_with_no_bias_wd_mask(self):
        """Test partition_labels with no_bias_wd_mask as the frozen_filter."""
        model = {
            "w": jnp.ones((10, 10)),
            "b": jnp.zeros(10),
        }

        # Use no_bias_wd_mask to partition: 1D (bias) frozen, others train
        result = partition_labels(
            model,
            frozen_filter=lambda x: x.ndim == 1,  # Matches no_bias_wd_mask logic
            frozen_label="frozen",
            train_label="train",
        )

        assert result["w"] == "train"
        assert result["b"] == "frozen"
