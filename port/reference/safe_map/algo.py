# REFERENCE: DO NOT MODIFY
"""Sealed reference oracle for safe_map parity (MVP leaf kernel)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import numpy as np

T = TypeVar("T")


def safe_map_reference(fn: Callable[[T], T], xs: np.ndarray) -> np.ndarray:
    """Reference batch map: stack per-element applications along axis 0."""
    return np.stack([fn(x) for x in xs], axis=0)
