from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

from xtrax.sparse.config import SparseConfig

Array = jax.Array
PyTree = Any


class SparsePolicy(eqx.Module):
    config: SparseConfig = eqx.field(static=True)

    def should_update(self, step: int) -> bool:
        return self.config.update_schedule(step)

    def make_mask(self, weights: Array, step: int) -> Array:
        # argsort-based: returns EXACTLY nse_budget True values (no threshold-tie).
        flat = jnp.abs(weights.ravel())
        n = min(self.config.nse_budget, flat.size)
        sorted_idx = jnp.argsort(flat)[-n:]
        mask_flat = jnp.zeros(flat.size, dtype=jnp.bool_).at[sorted_idx].set(True)
        return mask_flat.reshape(weights.shape)

    def apply_mask(self, weights: Array, mask: Array) -> Array | BCOO:
        # Eager Python-side only — NOT safe under filter_jit/jax.jit.
        # n_true branches on a traced Array under jit → TracerBoolConversionError.
        # Callers must invoke apply_mask outside of jit.
        n_true = jnp.sum(mask)
        if self.config.fallback_mode == "error" and n_true > self.config.nse_budget:
            msg = f"True nonzeros {int(n_true)} exceeds nse_budget {self.config.nse_budget}"
            raise ValueError(msg)
        if n_true > self.config.nse_budget:
            return weights * mask  # dense_mask fallback
        # Fixed-nse BCOO: argwhere on 2D mask → (nse_budget, 2) indices.
        # fill_value=0 means padded slots point to (0,0); zeroed via validity mask.
        indices = jnp.argwhere(mask, size=self.config.nse_budget, fill_value=0)
        valid = jnp.arange(self.config.nse_budget) < n_true
        data = weights[indices[:, 0], indices[:, 1]]
        data = jnp.where(valid, data, jnp.zeros_like(data[0]))
        return BCOO((data, indices), shape=weights.shape)
