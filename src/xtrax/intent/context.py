"""Caller context for intent resolution.

A frozen record of constraints. Ambient configuration is not read here;
``resolved_backend`` is the one place that falls back to the JAX default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax

from xtrax.tiling.budget import MemoryBudget


@dataclass(frozen=True)
class Context:
    """Constraints a resolver must satisfy.

    Attributes:
        memory_budget: A ``MemoryBudget``, a byte count the resolver later
            turns into a budget, or ``None``. Stored as given.
        target: ``"jit"`` or ``"export"``.
        export_format: Export format name. Required when ``target`` is
            ``"export"``.
        backend: Device backend name, or ``None`` for the JAX default.
        chunk_size: Positive chunk size used by chunked strategies.
        kernels: Whether kernel implementations are preferred.
    """

    memory_budget: MemoryBudget | int | None = None
    target: Literal["jit", "export"] = "jit"
    export_format: str | None = None
    backend: str | None = None
    chunk_size: int = 32
    kernels: bool = False

    def __post_init__(self) -> None:
        """Reject export without a format, and a non-positive chunk size.

        Parameters
        ----------
        self : Context
            Instance under construction.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            ``target`` is ``"export"`` and ``export_format`` is unset, or
            ``chunk_size`` is less than 1.
        """
        if self.target == "export" and self.export_format is None:
            raise ValueError("target='export' requires export_format")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")

    def resolved_backend(self) -> str:
        """Return the backend this context should run on.

        Parameters
        ----------
        self : Context
            Context whose ``backend`` field is read.

        Returns
        -------
        str
            ``backend`` when set, otherwise ``jax.default_backend()``.
        """
        return self.backend or jax.default_backend()
