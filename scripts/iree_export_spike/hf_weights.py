"""Pull weights from HuggingFace and map them into an Equinox module.

xtrax has no HuggingFace integration at all today: persistence is Orbax-only
(``src/xtrax/checkpoint/orbax.py``), and ``load_checkpoint`` requires a
``state_template`` -- you cannot restore without an already-constructed,
structurally identical module. ``InputResolver`` resolves *feature batches*, not
weights, and understands no URI schemes. So this is written from scratch.

Deliberately narrow: a hand-built MLP whose weights load from a real safetensors
checkpoint. That exercises download -> safetensors -> pytree -> baked-constants
without a full architecture port, which is separate work and not what the spike
is testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp


class HFWeightsError(Exception):
    """Raised when a checkpoint cannot be fetched or mapped."""


class TinyMLP(eqx.Module):
    """Two-layer MLP. Weights are plain array leaves, so they bake in cleanly.

    Kept intentionally simple: the point of the spike is the export path, not the
    architecture. ``__call__`` is pure JAX with static shapes and no Python control
    flow on traced values, which is the whole StableHLO requirement list.
    """

    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        h = jnp.tanh(x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2


@dataclass(frozen=True)
class WeightReport:
    """What actually happened during the load -- printed by the driver."""

    source: str
    tensors_seen: int
    tensors_used: int
    dtypes_cast: tuple[str, ...]


def random_mlp(in_dim: int, hidden: int, out_dim: int, seed: int = 0) -> TinyMLP:
    """Deterministic fallback model, used when no HF id is supplied."""
    k1, k2, k3, k4 = jax.random.split(jax.random.key(seed), 4)
    scale = 0.5
    return TinyMLP(
        w1=jax.random.normal(k1, (in_dim, hidden)) * scale,
        b1=jax.random.normal(k2, (hidden,)) * scale,
        w2=jax.random.normal(k3, (hidden, out_dim)) * scale,
        b2=jax.random.normal(k4, (out_dim,)) * scale,
    )


def _load_safetensors(repo_id: str, filename: str) -> dict[str, Any]:
    """Download one safetensors file and return its tensors as numpy arrays."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - exercised via fake module in tests
        raise HFWeightsError("huggingface_hub is required: uv sync --group export-spike") from exc
    try:
        from safetensors.numpy import load_file
    except ImportError as exc:  # pragma: no cover - exercised via fake module in tests
        raise HFWeightsError("safetensors is required: uv sync --group export-spike") from exc

    path = hf_hub_download(repo_id=repo_id, filename=filename)
    return load_file(path)


def mlp_from_hf(
    repo_id: str,
    *,
    filename: str = "model.safetensors",
    in_dim: int,
    hidden: int,
    out_dim: int,
    dtype: Any = jnp.float32,
) -> tuple[TinyMLP, WeightReport]:
    """Build a ``TinyMLP`` from the first usable 2-D tensors in an HF checkpoint.

    This is intentionally shape-driven rather than name-driven: tiny test
    checkpoints differ wildly in naming, and the spike only needs *real weights of
    the right shape*, not a faithful architecture port. A real port would map by
    parameter name.

    HF ships f16/bf16 routinely, so every tensor is cast explicitly -- an implicit
    cast would silently change the exported StableHLO's dtypes.
    """
    tensors = _load_safetensors(repo_id, filename)
    if not tensors:
        raise HFWeightsError(f"{repo_id}/{filename} contained no tensors")

    cast: list[str] = []
    matrices: list[jax.Array] = []
    for name in sorted(tensors):
        arr = tensors[name]
        if getattr(arr, "ndim", 0) != 2:
            continue
        if str(arr.dtype) != jnp.dtype(dtype).name:
            cast.append(f"{name}:{arr.dtype}->{jnp.dtype(dtype).name}")
        matrices.append(jnp.asarray(arr, dtype=dtype))

    if len(matrices) < 2:
        raise HFWeightsError(
            f"{repo_id}/{filename} has {len(matrices)} 2-D tensors; need at least 2. "
            "Pick a checkpoint with dense layers, or use --random."
        )

    def _fit(src: jax.Array, rows: int, cols: int) -> jax.Array:
        """Slice or zero-pad a real tensor to the target shape."""
        out = jnp.zeros((rows, cols), dtype=dtype)
        r = min(rows, src.shape[0])
        c = min(cols, src.shape[1])
        return out.at[:r, :c].set(src[:r, :c])

    model = TinyMLP(
        w1=_fit(matrices[0], in_dim, hidden),
        b1=jnp.zeros((hidden,), dtype=dtype),
        w2=_fit(matrices[1], hidden, out_dim),
        b2=jnp.zeros((out_dim,), dtype=dtype),
    )
    report = WeightReport(
        source=f"{repo_id}/{filename}",
        tensors_seen=len(tensors),
        tensors_used=2,
        dtypes_cast=tuple(cast[:2]),
    )
    return model, report


__all__ = [
    "HFWeightsError",
    "TinyMLP",
    "WeightReport",
    "mlp_from_hf",
    "random_mlp",
]
