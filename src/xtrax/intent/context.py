"""Caller context for intent resolution.

A frozen record of constraints. Ambient configuration is not read here;
``resolved_backend`` is the one place that falls back to the JAX default.
"""

from dataclasses import dataclass
from typing import Any, Literal

import jax

from xtrax.tiling.budget import MemoryBudget
from xtrax.tiling.sparse_spec import SparseSpec


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
        sparsity: Optional tensor sparsity specification or density float in (0.0, 1.0].
    """

    memory_budget: MemoryBudget | int | None = None
    target: Literal["jit", "export"] = "jit"
    export_format: str | None = None
    backend: str | None = None
    chunk_size: int = 32
    kernels: bool = False
    sparsity: float | SparseSpec | None = None
    safety: bool = False
    safety_checks: tuple[str, ...] = ("nan", "div")
    mesh: Any = None
    sharding_policy: Any = None

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
            ``chunk_size`` is less than 1, or ``sparsity`` is out of (0.0, 1.0].
        """
        if self.target == "export" and self.export_format is None:
            raise ValueError("target='export' requires export_format")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if self.sparsity is not None:
            if isinstance(self.sparsity, (int, float)):
                if not (0.0 < float(self.sparsity) <= 1.0):
                    raise ValueError(
                        f"Context: sparsity must be in (0.0, 1.0], got {self.sparsity}"
                    )
            elif not isinstance(self.sparsity, SparseSpec):
                name = type(self.sparsity).__name__
                raise TypeError(f"Context: sparsity must be float, SparseSpec, or None, got {name}")

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
