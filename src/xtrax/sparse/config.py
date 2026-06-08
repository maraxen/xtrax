from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SparseConfig:
    nse_budget: int
    update_schedule: Callable[[int], bool]
    fallback_mode: Literal["dense_mask", "error"] = "dense_mask"

    def __post_init__(self) -> None:
        if self.nse_budget < 1:
            raise ValueError(f"nse_budget must be >= 1, got {self.nse_budget}")
