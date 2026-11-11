"""Tests for custom_vjp with symbolic_zeros=True and primal+grad fusion (#3192)."""

import jax
import jax.numpy as jnp
import numpy as np

from xtrax.intent.context import Context
from xtrax.kernel_seam import VjpKernel, op


class TestSymbolicZerosVJP:
    """Verification controls for symbolic_zeros=True in VJP seams."""

    def test_custom_vjp_receives_symbolic_zero_cotangent(self):
        """When an output is not differentiated, custom_vjp receives SymbolicZero cotangent."""
        fwd_call_count = 0
        bwd_call_count = 0

        def ref_fn(x, y):
            return x * 2.0, y * 3.0

        def kernel_fwd(x, y):
            nonlocal fwd_call_count
            # eval_shape in _check_signature passes DynamicJaxprTracer
            if not any("DynamicJaxpr" in type(a).__name__ for a in (x, y)):
                fwd_call_count += 1
            # Returns (out, residuals)
            return (x * 2.0, y * 3.0), (x, y)

        def kernel_bwd(residuals, ct):
            nonlocal bwd_call_count
            bwd_call_count += 1
            ct_out1, ct_out2 = ct
            # Compute input cotangents
            dx = ct_out1 * 2.0
            dy = ct_out2 * 3.0
            return dx, dy

        vk = VjpKernel(fwd=kernel_fwd, bwd=kernel_bwd)
        custom_op = op(
            ref_fn,
            vjp_kernels={"cpu": vk},
            context=Context(kernels=True, backend="cpu"),
        )

        # Differentiate ONLY with respect to the first output (out1), ignoring out2
        def loss_fn(x, y):
            out1, out2 = custom_op(x, y)
            return jnp.sum(out1)  # out2 has zero cotangent (SymbolicZero)

        x_val = jnp.array([1.0, 2.0], dtype=jnp.float32)
        y_val = jnp.array([3.0, 4.0], dtype=jnp.float32)

        grad_x, grad_y = jax.grad(loss_fn, argnums=(0, 1))(x_val, y_val)

        # Forward ran once, backward ran once
        assert fwd_call_count == 1
        assert bwd_call_count == 1

        # grad_x is 2.0, grad_y is 0.0
        np.testing.assert_allclose(grad_x, jnp.array([2.0, 2.0]))
        np.testing.assert_allclose(grad_y, jnp.array([0.0, 0.0]))

    def test_all_symbolic_zeros_skips_kernel_backward(self):
        """When all output cotangents are SymbolicZero, backward kernel dispatch is bypassed."""
        bwd_called = False

        def ref_fn(x):
            return x * 5.0

        def kernel_fwd(x):
            return x * 5.0, x

        def kernel_bwd(res, ct):
            nonlocal bwd_called
            bwd_called = True
            return (ct * 5.0,)

        vk = VjpKernel(fwd=kernel_fwd, bwd=kernel_bwd)
        custom_op = op(
            ref_fn,
            vjp_kernels={"cpu": vk},
            context=Context(kernels=True, backend="cpu"),
        )

        # Function that computes op(x) but returns a constant (cotangent is SymbolicZero)
        def loss_fn(x):
            _ = custom_op(x)
            return 42.0

        grad_val = jax.grad(loss_fn)(jnp.array([1.0, 2.0]))
        np.testing.assert_allclose(grad_val, jnp.zeros(2))
        assert not bwd_called

    def test_fused_primal_and_grad_evaluation(self):
        """Op.fused_primal_and_grad computes both primal output and gradient."""

        def ref_fn(x):
            return jnp.sum(x**2)

        def custom_fused(x):
            out = jnp.sum(x**2)
            grad = 2.0 * x
            return out, (grad,)

        vk = VjpKernel(
            fwd=lambda x: (ref_fn(x), x),
            bwd=lambda res, ct: (ct * 2.0 * res,),
            fused_primal_and_grad=custom_fused,
        )

        custom_op = op(
            ref_fn,
            vjp_kernels={"cpu": vk},
            context=Context(kernels=True, backend="cpu"),
        )

        x = jnp.array([2.0, 3.0], dtype=jnp.float32)
        out, grads = custom_op.fused_primal_and_grad(x)

        np.testing.assert_allclose(out, 13.0)
        np.testing.assert_allclose(grads[0], jnp.array([4.0, 6.0]))

    def test_fused_primal_and_grad_with_caller_cotangent(self):
        """Op.fused_primal_and_grad forwards caller-supplied cotangent."""

        def ref_fn(x):
            return x * 3.0

        def custom_fused(x, cotangent=None):
            out = x * 3.0
            ct = 1.0 if cotangent is None else cotangent
            grad = ct * 3.0
            return out, (grad,)

        vk = VjpKernel(
            fwd=lambda x: (ref_fn(x), x),
            bwd=lambda res, ct: (ct * 3.0,),
            fused_primal_and_grad=custom_fused,
        )
        custom_op = op(
            ref_fn,
            vjp_kernels={"cpu": vk},
            context=Context(kernels=True, backend="cpu"),
        )
        x = jnp.array([1.0, 2.0], dtype=jnp.float32)
        ct = jnp.array([0.5, 2.0], dtype=jnp.float32)
        out, grads = custom_op.fused_primal_and_grad(x, cotangent=ct)
        np.testing.assert_allclose(out, jnp.array([3.0, 6.0]))
        np.testing.assert_allclose(grads[0], jnp.array([1.5, 6.0]))

    def test_pytree_symbolic_zeros_backward(self):
        """All-SymbolicZero backward correctly returns PyTree structure."""

        def ref_fn(tree):
            return tree["a"] * 2.0 + tree["b"] * 3.0

        def kernel_fwd(tree):
            return ref_fn(tree), tree

        def kernel_bwd(res, ct):
            return ({"a": ct * 2.0, "b": ct * 3.0},)

        vk = VjpKernel(fwd=kernel_fwd, bwd=kernel_bwd)
        custom_op = op(
            ref_fn,
            vjp_kernels={"cpu": vk},
            context=Context(kernels=True, backend="cpu"),
        )

        def loss_fn(tree):
            _ = custom_op(tree)
            return 42.0

        tree_val = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}
        grad_val = jax.grad(loss_fn)(tree_val)
        np.testing.assert_allclose(grad_val["a"], jnp.zeros(2))
        np.testing.assert_allclose(grad_val["b"], jnp.zeros(2))
