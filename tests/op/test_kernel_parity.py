"""Tests for the kernel parity helpers (xtrax.testing.kernel_parity).

Every gate here has a control that turns it red.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.op._kernels import (
    make_hist_kernel,
    make_hist_reference,
    matmul_kernel,
    matmul_reference,
)
from xtrax.intent import Context
from xtrax.kernel_seam import VjpKernel, op, op_context
from xtrax.testing.kernel_parity import (
    KernelParityError,
    assert_kernel_parity,
    assert_kernel_ran,
    assert_vjp_parity,
    compare_outputs,
    dropped_gradient_control,
    permuted_weight_control,
    require_corruption_red,
    require_vjp_controls_red,
    scale_error_control,
    scale_relative_atol,
    vjp_parity,
)
from xtrax.testing.lanes import NegativeControlError

KEY = jax.random.PRNGKey(7)


def _mm_vjp():
    def fwd(x, w):
        return matmul_kernel(x, w), (x, w)

    def bwd(res, ct):
        x, w = res
        return matmul_kernel(ct, w.T), matmul_kernel(x.T, ct)

    return VjpKernel(fwd, bwd)


def _args():
    kx, kw = jax.random.split(jax.random.PRNGKey(0))
    return jax.random.normal(kx, (16, 24)), jax.random.normal(kw, (24, 16))


# --- forward parity ------------------------------------------------------------------------


def test_scale_relative_atol_floor_and_scaling():
    assert scale_relative_atol(np.array([0.5])) == pytest.approx(1e-5)
    assert scale_relative_atol(np.array([-150.0, 3.0])) == pytest.approx(1.5e-3)


def test_float_tolerance_is_scale_relative_not_elementwise():
    ref = np.array([150.0, 1.0], np.float32)
    got = ref + np.array([5e-4, 0.0], np.float32)  # fails 1e-5 absolute, passes 1e-5 * 150
    assert compare_outputs(got, ref).ok
    assert not compare_outputs(ref + np.array([5e-3, 0.0], np.float32), ref).ok


def test_integer_leaves_need_exact_equality():
    ref = np.array([3, 4, 5], np.int32)
    assert compare_outputs(ref.copy(), ref).ok
    off_by_one = compare_outputs(ref + np.array([0, 0, 1], np.int32), ref)
    assert not off_by_one.ok and off_by_one.int_mismatches == 1


def test_dtype_shape_and_structure_mismatches_fail():
    ref = {"a": np.zeros(3, np.float32)}
    assert not compare_outputs({"a": np.zeros(3, np.float64)}, ref).ok
    assert not compare_outputs({"a": np.zeros(4, np.float32)}, ref).ok
    assert not compare_outputs({"b": np.zeros(3, np.float32)}, ref).ok


def test_assert_kernel_parity_passes_and_raises():
    x, w = _args()
    assert_kernel_parity(matmul_kernel, matmul_reference, x, w)
    with pytest.raises(KernelParityError):
        assert_kernel_parity(lambda a, b: matmul_kernel(a, b) * 1.01, matmul_reference, x, w)


def test_integer_histogram_parity_is_exact():
    idx = jnp.array([0, 1, 1, 4, 4, 4, 2, 0], jnp.int32)
    assert_kernel_parity(make_hist_kernel(5), make_hist_reference(5), idx)
    with pytest.raises(KernelParityError):
        assert_kernel_parity(lambda i: make_hist_kernel(5)(i) + 1, make_hist_reference(5), idx)


def test_corruption_control_turns_red_and_a_noop_corruption_is_rejected():
    x, w = _args()

    def corrupt_one_block(kernel):
        def run(a, b):
            return kernel(a, b).at[:8, :8].add(0.5)

        return run

    result = require_corruption_red(matmul_kernel, matmul_reference, (x, w), corrupt_one_block)
    assert not result.ok
    with pytest.raises(NegativeControlError):
        require_corruption_red(matmul_kernel, matmul_reference, (x, w), lambda k: k)


def test_assert_kernel_ran_detects_pallas_and_flags_a_silent_fallback():
    x, w = _args()
    mm = op(matmul_reference, kernels={"cpu": matmul_kernel})
    with op_context(Context(kernels=True, backend="cpu")):
        assert_kernel_ran(mm, x, w)
    with pytest.raises(KernelParityError):  # kernels=False: the reference ran
        assert_kernel_ran(mm, x, w)


# --- vjp parity ----------------------------------------------------------------------------


def test_vjp_parity_passes_for_a_correct_backward_kernel():
    x, w = _args()
    result = assert_vjp_parity(_mm_vjp(), matmul_reference, (x, w), key=KEY)
    assert result.ok and all(e < 1e-6 for e in result.rel_l2)


def test_vjp_parity_skips_integer_arguments():
    ref = lambda x, i: x[i]  # noqa: E731
    vk = VjpKernel(
        lambda x, i: (x[i], (x.shape[0], i)),
        lambda res, ct: (jnp.zeros(res[0]).at[res[1]].add(ct), None),
    )
    x, i = jnp.arange(6.0), jnp.array([0, 2, 2, 5])
    result = assert_vjp_parity(vk, ref, (x, i), key=KEY)
    assert result.rel_l2[1] is None and result.rel_l2[0] < 1e-6


@pytest.mark.parametrize("name", ["scale", "permute", "drop"])
def test_each_negative_control_turns_vjp_parity_red(name):
    x, w = _args()
    vk = _mm_vjp()
    bad = {
        "scale": scale_error_control(vk),
        "permute": permuted_weight_control(vk, 1, (x, w)),
        "drop": dropped_gradient_control(vk, 1),
    }[name]
    assert vjp_parity(vk, matmul_reference, (x, w), key=KEY).ok
    result = vjp_parity(bad, matmul_reference, (x, w), key=KEY)
    assert not result.ok
    with pytest.raises(KernelParityError):
        assert_vjp_parity(bad, matmul_reference, (x, w), key=KEY)


def test_scale_control_error_is_about_one_tenth_of_a_percent():
    x, w = _args()
    result = vjp_parity(scale_error_control(_mm_vjp()), matmul_reference, (x, w), key=KEY)
    assert all(1e-3 * 0.9 < e < 1e-3 * 1.1 for e in result.rel_l2)


def test_require_vjp_controls_red_returns_the_failing_results():
    x, w = _args()
    results = require_vjp_controls_red(
        _mm_vjp(), matmul_reference, (x, w), weight_argnum=1, key=KEY
    )
    assert set(results) == {"scale", "permute", "drop"}
    assert not any(r.ok for r in results.values())


def test_require_vjp_controls_red_rejects_a_gate_too_loose_to_see_the_errors():
    x, w = _args()
    with pytest.raises(NegativeControlError, match="scale"):
        require_vjp_controls_red(
            _mm_vjp(), matmul_reference, (x, w), weight_argnum=1, key=KEY, rtol=1e-2
        )


def test_an_all_ones_cotangent_would_miss_a_row_permutation():
    """Why the cotangent is random: an all-ones ct hides a permutation of its rows."""
    x, w = _args()
    vk = _mm_vjp()

    def bwd_permuted_ct(res, ct):
        return vk.bwd(res, ct[::-1])

    permuted = VjpKernel(vk.fwd, bwd_permuted_ct)
    out = vk.fwd(x, w)[0]
    ones = jnp.ones_like(out)
    # dw = x.T @ ct. A row permutation of an all-ones ct leaves dw unchanged ...
    np.testing.assert_allclose(
        permuted.bwd(vk.fwd(x, w)[1], ones)[1], vk.bwd(vk.fwd(x, w)[1], ones)[1], rtol=1e-5
    )
    # ... while the random-cotangent gate sees it.
    assert not vjp_parity(permuted, matmul_reference, (x, w), key=KEY).ok
