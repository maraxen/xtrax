"""Graded parity stub for safe_map — MVP leaf kernel (AC-3, AC-4)."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xtrax.transforms.map import safe_map

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference" / "safe_map"
BASELINE_PATH = REFERENCE_DIR / "baseline_io.json"


def _load_baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _fn_from_registry(name: str) -> object:
    if name == "identity":
        return lambda x: x
    if name == "double":
        return lambda x: x * 2.0
    raise ValueError(f"unknown fn registry key: {name!r}")


def _case_arrays(case: dict) -> tuple[np.ndarray, object, str]:
    dtype = case["dtype"]
    xs = np.asarray(case["xs"], dtype=np.dtype(dtype))
    fn = _fn_from_registry(case["fn"])
    return xs, fn, dtype


@pytest.mark.tier_1
def test_tier_1_dtype_shape(oracle, port_target) -> None:
    baseline = _load_baseline()
    for case in baseline["cases"]:
        xs, fn, dtype = _case_arrays(case)
        ref_out = oracle.safe_map_reference(fn, xs)
        jax_dtype = jnp.dtype(dtype)
        jax_out = np.asarray(safe_map(fn, jnp.asarray(xs, dtype=jax_dtype)))
        assert ref_out.shape == jax_out.shape, case["name"]
        if dtype == "float32":
            assert ref_out.dtype == jax_out.dtype, case["name"]


@pytest.mark.tier_2
def test_tier_2_float64(oracle, port_target) -> None:
    jax.config.update("jax_enable_x64", True)
    baseline = _load_baseline()
    case = next(c for c in baseline["cases"] if c["dtype"] == "float64")
    xs, fn, _ = _case_arrays(case)
    ref_out = oracle.safe_map_reference(fn, xs)
    jax_out = np.asarray(safe_map(fn, jnp.asarray(xs, dtype=jnp.float64)))
    np.testing.assert_allclose(jax_out, ref_out, rtol=1e-10, atol=1e-10)


@pytest.mark.tier_3
def test_tier_3_float32(oracle, port_target) -> None:
    baseline = _load_baseline()
    case = next(c for c in baseline["cases"] if c["dtype"] == "float32")
    xs, fn, _ = _case_arrays(case)
    ref_out = oracle.safe_map_reference(fn, xs.astype(np.float32))
    with jax.default_matmul_precision("highest"):
        jax_out = np.asarray(safe_map(fn, jnp.asarray(xs, dtype=jnp.float32)))
    rtol, atol = 1e-4, 1e-4
    np.testing.assert_allclose(jax_out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.tier_4
def test_tier_4_gradient_ad(oracle, port_target) -> None:
    """Gradient parity — runs only when ad_critical=true (conftest skips otherwise)."""
    xs = jnp.arange(4.0, dtype=jnp.float32)

    def _loss(params: jax.Array) -> jax.Array:
        out = safe_map(lambda x: x * params, xs)
        return jnp.sum(out)

    grad = jax.grad(_loss)(jnp.array(2.0))
    assert jnp.isfinite(grad)


@pytest.mark.tier_5
def test_tier_5_jit_invariance(oracle, port_target) -> None:
    baseline = _load_baseline()
    case = baseline["cases"][0]
    xs, fn, _ = _case_arrays(case)
    xs_jax = jnp.asarray(xs)
    eager = np.asarray(safe_map(fn, xs_jax))
    jit_safe_map = jax.jit(safe_map, static_argnums=(0, 2))
    jitted = np.asarray(jit_safe_map(fn, xs_jax))
    np.testing.assert_allclose(jitted, eager, rtol=1e-6, atol=1e-6)
