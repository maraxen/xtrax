"""Sealed EvaluateFn seam (T1-07, #3060, AC3): the immutable fitness-oracle contract.

The third, previously-greenfield leg of the mutation triad (roadmap research synthesis
§1.2): xtrax already has mutation boundaries (Fuse/AxisBoundary/StageBundle purity
contracts) and async JIT-safe tracking (Tap/Sink ordered flags + the topology validator),
but no primitive making fitness scalars come from exactly one immutable source. This module
is that primitive: a `(frozen_context, candidate) -> dict[str, float]` contract, registered
into a `SealedEvaluatorRegistry` exactly once -- a second registration raises, and the
registry exposes no unseal/replace method at all (not merely an unused one).

`frozen_context` and `candidate` are intentionally opaque, caller-supplied types -- the
concrete fitness registry and any #2181 population representation are a documented
extension seam (fork 11 of the design spec), not defined here. Callers needing their own
independent write-once slot (e.g. a future #2181 concrete registry) construct their own
`SealedEvaluatorRegistry()` rather than sharing this module's default instance.
"""

from typing import Any, Protocol, TypeVar, runtime_checkable

FrozenContext = TypeVar("FrozenContext")
Candidate = TypeVar("Candidate")


class EvaluatorAlreadySealedError(Exception):
    """Raised by SealedEvaluatorRegistry.seal() when a registry already holds an evaluator.

    The seam holds a monopoly on fitness scalars (AC3) -- allowing re-registration would let
    a later call silently swap the immutable fitness oracle out from under an in-flight
    #2181 loop.
    """


@runtime_checkable
class EvaluateFn(Protocol[FrozenContext, Candidate]):
    """The sealed evaluator contract: (frozen_context, candidate) -> dict[str, float]."""

    def __call__(self, frozen_context: FrozenContext, candidate: Candidate) -> dict[str, float]: ...


class SealedEvaluatorRegistry:
    """A write-once registration slot for one EvaluateFn.

    No unseal/replace method exists on this class -- that IS the "no mutation API"
    guarantee AC3 calls for, not just an omission at current call sites.
    """

    def __init__(self) -> None:
        self._evaluator: EvaluateFn[Any, Any] | None = None

    def seal(self, fn: EvaluateFn[Any, Any]) -> None:
        """Register `fn` as this registry's evaluator. Callable exactly once."""
        if self._evaluator is not None:
            msg = "an evaluator is already sealed on this registry"
            raise EvaluatorAlreadySealedError(msg)
        self._evaluator = fn

    def get(self) -> EvaluateFn[Any, Any]:
        """Return the sealed evaluator."""
        if self._evaluator is None:
            msg = "no evaluator has been sealed yet; call seal() first"
            raise RuntimeError(msg)
        return self._evaluator


_default_registry = SealedEvaluatorRegistry()


def seal_evaluator(fn: EvaluateFn[Any, Any]) -> None:
    """Register `fn` as THE evaluator for this process. Callable exactly once."""
    _default_registry.seal(fn)


def get_sealed_evaluator() -> EvaluateFn[Any, Any]:
    """Return the process-wide sealed evaluator."""
    return _default_registry.get()


__all__ = [
    "Candidate",
    "EvaluateFn",
    "EvaluatorAlreadySealedError",
    "FrozenContext",
    "SealedEvaluatorRegistry",
    "get_sealed_evaluator",
    "seal_evaluator",
]
