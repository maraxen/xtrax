"""Error types and sentinel values for signature inference.

AmbiguousAxisError and AxisRole are defined in xtrax.tiling.roles (a
pure-stdlib leaf) and re-exported here for backward compatibility so that
``from xtrax.inference.errors import AmbiguousAxisError, AxisRole`` and
``from xtrax.inference import AmbiguousAxisError, AxisRole`` continue to work.
"""

from __future__ import annotations

# Re-exports from the leaf module — tiling.roles has zero xtrax imports.
from xtrax.tiling.roles import AmbiguousAxisError, AxisRole

__all__ = [
    "AmbiguousAxisError",
    "AxisRole",
    "CseTraceError",
    "MemoDonationError",
    "MemoImpurityError",
    "MemoKeyUnsupportedLeafError",
    "MemoMultiDeviceError",
    "MemoStalenessError",
    "StructureMismatchError",
    "XtraxInferenceError",
]


class XtraxInferenceError(Exception):
    """Base exception for xtrax.inference module."""

    pass


class StructureMismatchError(XtraxInferenceError):
    """Raised when eval_shape structure diverges from concrete batch.

    This error indicates that a traced computation structure does not match
    the structure inferred from a concrete batch. This typically occurs when
    control flow or branching introduces different tree structures depending
    on runtime values.

    Example:
        If eval_shape produces a tree with different branches than the actual
        concrete execution, this error is raised to signal the structural mismatch.
    """

    pass


class MemoImpurityError(XtraxInferenceError):
    """Static screen detected a likely-impure function at admission."""


class MemoDonationError(MemoImpurityError):
    """Static screen detected donation markers in the traced jaxpr (spec §4.2 item 6).

    A subclass of ``MemoImpurityError`` so the wrap-time/first-call latch and
    ``memo_rewrap()`` machinery, and any existing ``except MemoImpurityError``
    caller, keep working unchanged, while remaining distinguishable by type.

    ``sites`` is a tuple of
    ``(path, carrier, eqn_operand_indices, wrapped_input_leaf_indices)``
    entries, one per donation-carrying equation/carrier pair found during the
    recursive jaxpr walk:

    - ``path``: human-readable location of the offending equation (dotted/
      bracketed trail through nested jaxprs, e.g. ``branches[1]``).
    - ``carrier``: ``"donated_invars"`` or ``"copy_semantics"``.
    - ``eqn_operand_indices``: flattened operand positions *within that
      equation* that carry the donation marker (not argnums of the wrapped
      function).
    - ``wrapped_input_leaf_indices``: for top-level equations only, the index
      ``j`` such that the donated operand *is* (identity) the wrapped
      function's ``j``-th flattened positional-arg pytree leaf. Empty for
      nested equations (identity lookup, not provenance tracing).
    """

    def __init__(
        self,
        message: str,
        *,
        sites: tuple[tuple[str, str, tuple[int, ...], tuple[int, ...]], ...],
    ) -> None:
        super().__init__(message)
        self.sites = sites


class MemoMultiDeviceError(XtraxInferenceError):
    """Wrapper requires exactly one local device (spec N5)."""


class MemoKeyUnsupportedLeafError(XtraxInferenceError):
    """A pytree leaf type cannot be safely digested for the cache key."""


class MemoStalenessError(XtraxInferenceError):
    """Spot-check found a cached entry diverging from fresh computation."""


class CseTraceError(XtraxInferenceError):
    """Raised when analyze_cse cannot trace the target function.

    Wraps tracing failures (unsupported control flow, wrong argument count,
    non-traceable operations) so callers can distinguish analysis failures
    from computation failures.
    """

    pass


# Component C (dedup-spec synthesis) error classes live in
# xtrax.tiling.dedup_synthesis, not here: xtrax.tiling may not import
# xtrax.inference (import-linter contract "tiling must not import
# inference"), and dedup_synthesis.py is itself in xtrax.tiling.
