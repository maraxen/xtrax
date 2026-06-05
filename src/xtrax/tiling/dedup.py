"""Deduplication specification and bucket sizing for tiling strategies."""

from __future__ import annotations

from dataclasses import dataclass


def get_k_bucket(n: int) -> int:
    """Return the smallest power of 2 >= n.

    Mandated integer implementation: 1 << (n - 1).bit_length()

    Args:
        n: Positive integer.

    Returns:
        Smallest power of 2 >= n.

    Raises:
        ValueError: If n <= 0.

    Examples:
        get_k_bucket(1) == 1
        get_k_bucket(7) == 8
        get_k_bucket(8) == 8
        get_k_bucket(9) == 16
    """
    if n <= 0:
        raise ValueError(f"get_k_bucket: n must be > 0, got {n}")
    return 1 << (n - 1).bit_length()


@dataclass(frozen=True)
class DedupSpec:
    """Specification for deduplication allocation sizing.

    Attributes:
        k_bucket: Power-of-2 padded allocation size for unique elements.
            Must be a positive power of 2 and >= max_unique.
        max_unique: Maximum number of unique elements.
    """

    k_bucket: int
    max_unique: int

    def __post_init__(self) -> None:
        """Validate k_bucket is a power of 2 and k_bucket >= max_unique."""
        if self.k_bucket <= 0:
            raise ValueError(
                f"DedupSpec: k_bucket={self.k_bucket} must be a positive power of 2."
            )
        if (self.k_bucket & (self.k_bucket - 1)) != 0:
            raise ValueError(
                f"DedupSpec: k_bucket={self.k_bucket} must be a power of 2."
            )
        if self.k_bucket < self.max_unique:
            raise ValueError(
                f"DedupSpec: k_bucket={self.k_bucket} must be >= "
                f"max_unique={self.max_unique}."
            )
