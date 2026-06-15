"""Protocol definitions for pluggable xtrax execution components."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AggregationFn(Protocol):
    """Aggregation function protocol."""

    def __call__(self, *args: object) -> object: ...


@runtime_checkable
class DecodeFn(Protocol):
    """Decode function protocol."""

    def __call__(self, *args: object) -> object: ...


@runtime_checkable
class NoiseFn(Protocol):
    """Noise function protocol."""

    def __call__(self, *args: object) -> object: ...
