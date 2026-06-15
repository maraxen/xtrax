"""Input resolution and runtime bundling for xtrax execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NewType, Protocol, runtime_checkable

import equinox as eqx

from xtrax.run.spec import RunSpec

FeatureBatch = NewType("FeatureBatch", dict[str, Any])


@dataclass
class RuntimeBundle:
    """Materialized execution context (produced before InputResolver fires)."""

    iterator: Any
    model: eqx.Module


@runtime_checkable
class InputResolver(Protocol):
    """Map (spec, materialized bundle) to a feature batch.

    Do NOT make this a generic Protocol[S, T] over two TypeVars.
    Use @functools.singledispatch for subclass-specific implementations.
    """

    def __call__(self, spec: RunSpec, bundle: RuntimeBundle) -> FeatureBatch: ...
