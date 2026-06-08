from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SparseConfig:
    nse_budget: int
    update_schedule: Callable[[int], bool]
    fallback_mode: Literal["dense_mask", "error"] = "dense_mask"
