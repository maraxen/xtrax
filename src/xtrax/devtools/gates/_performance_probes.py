"""Trace probes for audit performance gate (D4 / #1584)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from xtrax.sparse.config import SparseConfig
from xtrax.sparse.inference import sparse_filter_jit, sparsify_model
from xtrax.sparse.policy import SparsePolicy


def _make_policy(budget: int) -> SparsePolicy:
    return SparsePolicy(
        config=SparseConfig(
            nse_budget=budget,
            update_schedule=lambda _step: True,
            fallback_mode="dense_mask",
        )
    )


@sparse_filter_jit
@chex.assert_max_traces(n=1)
def sparse_filter_jit_kernel(m: eqx.nn.Linear, x: jnp.ndarray) -> tuple[int, ...]:
    """Representative sparse_filter_jit kernel for trace-count gate."""
    return m.weight.shape


def probe_sparse_filter_jit_kernel(guarded: Callable[..., Any]) -> None:
    """Exercise sparse_filter_jit kernel with stable inputs."""
    key = jax.random.PRNGKey(0)
    model = eqx.nn.Linear(4, 4, key=key)
    sparse_model = sparsify_model(model, _make_policy(budget=8))
    x = jnp.ones((4,))
    guarded(sparse_model, x)
    guarded(sparse_model, x)


def probe_stable_jnp_kernel(guarded: Callable[..., Any]) -> None:
    """Minimal stable-input probe for integration tests."""
    x = jnp.zeros(3)
    guarded(x)
    guarded(x)
