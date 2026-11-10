"""Declarative sparsity specification for tensors and axes (#3044)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import jax.numpy as jnp


@dataclass(frozen=True)
class SparseSpec:
    """Declarative specification of tensor sparsity.

    Attributes
    ----------
    density : float
        Expected proportion of non-zero elements in (0.0, 1.0].
    format : {"bcoo", "coo", "csr"}
        Sparse layout format (default: "bcoo").
    index_dtype : Any
        Dtype of sparse indices (default: jnp.int32).
    """

    density: float
    format: Literal["bcoo", "coo", "csr"] = "bcoo"
    index_dtype: Any = jnp.int32

    def __post_init__(self) -> None:
        if not (0.0 < float(self.density) <= 1.0):
            raise ValueError(f"SparseSpec: density must be in (0.0, 1.0], got {self.density}")
