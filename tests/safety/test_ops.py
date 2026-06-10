import jax
import jax.numpy as jnp

from xtrax.safety.ops import safe_norm, safe_reciprocal


class TestSafeNorm:
    """Tests for safe_norm function."""

    def test_safe_norm_uses_eps_squared(self):
        """Verify safe_norm uses eps**2 (=1e-16), not finfo.eps."""
        # safe_norm(zeros(3)) should equal sqrt(eps**2) where eps=1e-8
        result = safe_norm(jnp.zeros(3))
        expected = jnp.sqrt(1e-8**2)
        assert jnp.allclose(result, expected), f"Expected {expected}, got {result}"

    def test_safe_norm_with_custom_eps(self):
        """Verify safe_norm respects custom eps kwarg."""
        x = jnp.zeros(3)
        eps = 2e-8
        result = safe_norm(x, eps=eps)
        expected = jnp.sqrt(eps**2)
        assert jnp.allclose(result, expected)

    def test_safe_norm_grad_at_zero_is_finite(self):
        """Verify gradient of safe_norm at x=zeros is finite (not NaN)."""

        def fn(x):
            return jnp.sum(safe_norm(x))

        x = jnp.zeros(3)
        grad_fn = jax.grad(fn)
        grad = grad_fn(x)

        # Gradient should be finite and not NaN
        assert jnp.all(jnp.isfinite(grad)), f"Gradient contains NaN or Inf: {grad}"

    def test_safe_norm_basic_computation(self):
        """Verify safe_norm computes correct norm for non-zero values."""
        x = jnp.array([3.0, 4.0])
        result = safe_norm(x)
        # norm = sqrt(9 + 16 + eps**2) = sqrt(25 + 1e-16) ≈ 5.0
        expected = jnp.sqrt(25.0 + 1e-8**2)
        assert jnp.allclose(result, expected, rtol=1e-6)

    def test_safe_norm_with_axis(self):
        """Verify safe_norm respects axis parameter."""
        x = jnp.array([[3.0, 4.0], [5.0, 12.0]])
        result = safe_norm(x, axis=1)
        # Result shape should be (2,)
        assert result.shape == (2,)
        # First norm: sqrt(9 + 16 + eps**2)
        assert jnp.allclose(result[0], jnp.sqrt(25.0 + 1e-8**2), rtol=1e-6)

    def test_safe_norm_with_keepdims(self):
        """Verify safe_norm respects keepdims parameter."""
        x = jnp.array([[3.0, 4.0], [5.0, 12.0]])
        result = safe_norm(x, axis=1, keepdims=True)
        # Result shape should be (2, 1) with keepdims=True
        assert result.shape == (2, 1)


class TestSafeReciprocal:
    """Tests for safe_reciprocal function."""

    def test_safe_reciprocal_at_zero(self):
        """Verify reciprocal(0.0) == 1/(0+eps) = 1e8 with default eps."""
        result = safe_reciprocal(0.0)
        expected = 1.0 / 1e-8
        assert jnp.allclose(result, expected), f"Expected {expected}, got {result}"

    def test_safe_reciprocal_at_two(self):
        """Verify reciprocal(2.0) ≈ 1/(2+eps) ≈ 0.5."""
        result = safe_reciprocal(2.0)
        expected = 1.0 / (2.0 + 1e-8)
        msg = f"Expected {expected}, got {result}"
        assert jnp.allclose(result, expected, rtol=1e-6), msg

    def test_safe_reciprocal_with_custom_eps(self):
        """Verify safe_reciprocal respects custom eps kwarg."""
        x = 2.0
        eps = 2e-8
        result = safe_reciprocal(x, eps=eps)
        expected = 1.0 / (x + eps)
        assert jnp.allclose(result, expected, rtol=1e-6)

    def test_safe_reciprocal_smooth_formula(self):
        """Verify safe_reciprocal uses smooth unconditional formula 1.0 / (x + eps)."""
        # Test that result is smooth across zero
        x = jnp.array([-0.5, -0.1, 0.0, 0.1, 0.5])
        result = safe_reciprocal(x)
        expected = 1.0 / (x + 1e-8)
        assert jnp.allclose(result, expected, rtol=1e-6)

    def test_safe_reciprocal_array_input(self):
        """Verify safe_reciprocal works with array inputs."""
        x = jnp.array([1.0, 2.0, 4.0])
        result = safe_reciprocal(x)
        expected = 1.0 / (x + 1e-8)
        assert jnp.allclose(result, expected, rtol=1e-6)

    def test_safe_reciprocal_grad_is_finite(self):
        """Verify gradient of safe_reciprocal is finite."""

        def fn(x):
            return jnp.sum(safe_reciprocal(x))

        x = jnp.array([0.0, 1.0, 2.0])
        grad_fn = jax.grad(fn)
        grad = grad_fn(x)

        # Gradient should be finite and not NaN
        assert jnp.all(jnp.isfinite(grad)), f"Gradient contains NaN or Inf: {grad}"
