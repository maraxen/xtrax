"""Host randomness shared by a candidate and a reference.

JAX and torch draw from different generators. An equal integer seed does not
produce the same uniforms, the same normals, or the same permutation. Build the
arrays once and pass them into both callables.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["InjectedSource", "order_from_randn"]


def order_from_randn(mask: ArrayLike, randn: ArrayLike, eps: float) -> np.ndarray:
    """Decoding order for ``argsort((mask + eps) * abs(randn))``.

    The sort is ascending and stable: equal scores keep the lower index. That
    matches ``numpy.argsort(..., kind="stable")``, ``jax.numpy.argsort`` (stable
    by default), and ``torch.argsort(..., stable=True)``. Default
    ``torch.argsort`` leaves ties unspecified, so pass this array into both
    implementations instead of sorting on each side.

    ``mask`` and ``randn`` share a shape. The order is along the last axis.
    Unmasked sites (``mask == 0``) score ``eps * abs(randn)`` and, for
    ``eps >= 0`` and a 0/1 mask, sort before masked sites.

    Args:
        mask: Broadcast factor in the score. Finite, same shape as ``randn``.
        randn: Standard-normal draws. Only the absolute value enters the score.
        eps: Added to ``mask`` before multiplying. Finite and ``>= 0``.

    Returns:
        Integer indices, the stable argsort of the score along the last axis.
    """
    mask_array = np.asarray(mask, dtype=np.float64)
    randn_array = np.asarray(randn, dtype=np.float64)
    if mask_array.shape != randn_array.shape:
        raise ValueError(f"mask shape {mask_array.shape} != randn shape {randn_array.shape}")
    eps_value = float(eps)
    if not np.isfinite(eps_value) or eps_value < 0.0:
        raise ValueError(f"eps must be finite and >= 0, got {eps!r}")
    if not np.isfinite(mask_array).all() or not np.isfinite(randn_array).all():
        raise ValueError("mask and randn must be finite")
    scores = (mask_array + eps_value) * np.abs(randn_array)
    return np.argsort(scores, axis=-1, kind="stable")


@dataclass(frozen=True)
class InjectedSource:
    """Order, noise, and uniform arrays to feed to both implementations.

    Attributes:
        order: Decoding order, often the return value of :func:`order_from_randn`.
        noise: Noise draws (Gaussian or otherwise) shared by both callables.
        uniform: Uniform draws in ``[0, 1)`` shared by both callables.
    """

    order: np.ndarray | None = None
    noise: np.ndarray | None = None
    uniform: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "order", _as_optional_array(self.order))
        object.__setattr__(self, "noise", _as_optional_array(self.noise))
        object.__setattr__(self, "uniform", _as_optional_array(self.uniform))

    def parameters(self) -> dict[str, np.ndarray]:
        """Keyword arguments carrying whichever arrays were provided."""
        provided: dict[str, np.ndarray] = {}
        if self.order is not None:
            provided["order"] = self.order
        if self.noise is not None:
            provided["noise"] = self.noise
        if self.uniform is not None:
            provided["uniform"] = self.uniform
        return provided

    def bind(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Return ``fn`` with this source passed as keywords.

        Keywords supplied at the call site override the injected arrays. Bind
        the same source to the candidate and to the reference.
        """
        injected = self.parameters()

        @functools.wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **{**injected, **kwargs})

        return wrapped


def _as_optional_array(value: ArrayLike | None) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value)
