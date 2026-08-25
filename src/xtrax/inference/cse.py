"""CseReport — jaxpr-level duplicate-subexpression detection (spec 260825 §4.1).

Detection only: reports structurally duplicate equation classes in a traced
function's jaxpr. Never rewrites anything — XLA performs intra-program CSE
after lowering; this tool exposes the redundancy XLA silently removes.

Equivalence follows XLA's hlo_cse.cc rule (opcode + params + operand identity,
commutativity-insensitive for commutative ops is out of scope for v1) computed
by FIXPOINT iteration with literal-value canonicalization:

- Var operands are unified through union-find representatives, rewritten each
  round, so merging duplicated ``sin`` eqns makes downstream duplicated ``mul``
  eqns operand-identical in the NEXT round (transitive duplicates reported).
- Literal operands join equivalence classes keyed by ``(dtype, value.tobytes())``
  so two distinct Literal objects holding equal values share one representative.

Reports are keyed on the PRE-optimization representation — the same identity
choice JAX's persistent compilation cache makes — so they stay stable across
XLA versions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from weakref import WeakKeyDictionary

import jax

from xtrax.inference.errors import CseTraceError

__all__ = [
    "CseDuplicateClass",
    "CseReport",
    "analyze_cse",
]


@dataclass(frozen=True)
class CseDuplicateClass:
    """One equivalence class of structurally identical eqns.

    Attributes:
        primitive: Primitive name, e.g. ``"sin"``.
        eqn_count: Number of duplicate eqns in this class (always >= 2).
        params_fingerprint: Stable repr-based fingerprint of the shared params.
        invar_shapes: Shapes of the (canonical) input variables.
        est_wasted_bytes: Summed output-leaf bytes over all but one member.
    """

    primitive: str
    eqn_count: int
    params_fingerprint: str
    invar_shapes: tuple[tuple[int, ...], ...]
    est_wasted_bytes: int


@dataclass(frozen=True)
class CseReport:
    """Duplicate-op report for one traced function.

    Attributes:
        duplicates: Duplicate classes sorted descending by est_wasted_bytes.
        total_eqns: Total eqn count of the traced jaxpr.
        duplicate_eqns: Sum of eqn_count over all duplicate classes.
        note: Human-readable scoping note (XLA correspondence caveats).
        trace_cache_hit: Whether make_jaxpr returned a memoized ClosedJaxpr
            for this exact function object + shapes. When True, closure
            mutations made after a previous analysis are NOT reflected here;
            pass a fresh callable to force a fresh trace.
    """

    duplicates: tuple[CseDuplicateClass, ...]
    total_eqns: int
    duplicate_eqns: int
    note: str = (
        "XLA performs intra-program CSE after lowering; these duplicates cost "
        "compile time and report clarity, not steady-state FLOPs. Reports may "
        "include classes XLA folds differently (constant folding, "
        "commutativity)."
    )
    trace_cache_hit: bool = False


def _literal_key(lit) -> tuple[str, bytes]:
    """Canonical key for a Literal operand: (dtype string, raw value bytes)."""
    val = lit.val
    return (str(val.dtype), np_ascontiguous_tobytes(val))


def np_ascontiguous_tobytes(val) -> bytes:
    """Contiguous C-order bytes for any array-like literal value."""
    import numpy as np

    return np.ascontiguousarray(np.asarray(val)).tobytes(order="C")


def _params_repr(params: dict) -> str:
    items = sorted((k, repr(v)) for k, v in params.items())
    return repr(items)


# Per-function trace memo: fn object -> last ClosedJaxpr seen. make_jaxpr itself
# memoizes per (fn identity, shapes) and returns the SAME ClosedJaxpr object on a
# cache hit; we detect that by identity comparison here so the report can flag
# stale-closure risk to callers.
_TRACE_MEMO: WeakKeyDictionary = WeakKeyDictionary()


def analyze_cse(
    fn: Callable[..., jax.Array],
    abstract_inputs: Sequence,
) -> CseReport:
    """Analyze ``fn`` for structurally duplicate subexpressions.

    Args:
        fn: A pure, traceable JAX function. It is traced ONCE via
            ``jax.make_jaxpr``; no concrete execution occurs.
        abstract_inputs: One ShapeDtypeStruct (or (shape, dtype) pair) per
            positional argument of ``fn``.

    Returns:
        CseReport with duplicate classes sorted by estimated wasted bytes.

    Raises:
        CseTraceError: If tracing fails (unsupported control flow, wrong
            argument count, non-traceable operations).

    Note:
        ``jax.make_jaxpr`` memoizes per (function identity, shapes): analyzing
        the same fn object twice returns the identical ClosedJaxpr, flagged via
        ``trace_cache_hit=True``. Mutated closures require a fresh callable.
    """
    try:
        closed = jax.make_jaxpr(fn)(*abstract_inputs)
    except Exception as exc:
        raise CseTraceError(
            f"analyze_cse could not trace {getattr(fn, '__qualname__', fn)!r}: {exc}"
        ) from exc

    cache_hit = _TRACE_MEMO.get(fn) is closed
    try:
        _TRACE_MEMO[fn] = closed
    except TypeError:
        pass  # unhashable/unweakrefable fn: skip memo tracking

    inner = closed.jaxpr
    eqns = inner.eqns
    n = len(eqns)

    # Union-find over eqn indices.
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    def fingerprint_round() -> dict[tuple, list[int]]:
        """One fixpoint round: hash eqns under current representative map."""
        var_token: dict[int, tuple] = {}
        for i, eq in enumerate(eqns):
            tok = ("eq", find(i))
            for ov in eq.outvars:
                var_token[id(ov)] = tok

        def op_token(v) -> tuple:
            if type(v).__name__ == "Literal":
                return ("lit", _literal_key(v))
            return var_token.get(id(v), ("free", id(v)))

        fps: dict[tuple, list[int]] = {}
        for i, eq in enumerate(eqns):
            fp = (
                eq.primitive.name,
                _params_repr(eq.params),
                tuple(op_token(v) for v in eq.invars),
            )
            fps.setdefault(fp, []).append(i)
        return fps

    # Fixpoint loop: merge, rewrite representatives, repeat until stable.
    prev_canonical = None
    while True:
        fps = fingerprint_round()
        changed = False
        for members in fps.values():
            if len(members) > 1:
                anchor = members[0]
                for other in members[1:]:
                    if find(anchor) != find(other):
                        union(anchor, other)
                        changed = True
        canonical = tuple(find(i) for i in range(n))
        if not changed or canonical == prev_canonical:
            break
        prev_canonical = canonical

    # Collect final equivalence classes.
    classes: dict[int, list[int]] = {}
    for i in range(n):
        classes.setdefault(find(i), []).append(i)

    duplicates: list[CseDuplicateClass] = []
    for members in classes.values():
        if len(members) < 2:
            continue
        eq0 = eqns[members[0]]
        out_shapes: list[tuple[int, ...]] = []
        for ov in eq0.outvars:
            shape = getattr(ov.aval, "shape", ())
            out_shapes.append(tuple(int(d) for d in shape))
        # Wasted bytes: all-but-one member's output size (smallest member kept).
        nbytes = 0
        for ov in eq0.outvars:
            aval = ov.aval
            size = 1
            for dim in getattr(aval, "shape", ()):
                size *= int(dim)
            nbytes += size * getattr(aval, "dtype", __import__("numpy").float32).itemsize
        duplicates.append(
            CseDuplicateClass(
                primitive=eq0.primitive.name,
                eqn_count=len(members),
                params_fingerprint=_params_repr(eq0.params)[:64],
                invar_shapes=tuple(out_shapes),
                est_wasted_bytes=nbytes * (len(members) - 1),
            )
        )

    duplicates.sort(key=lambda c: c.est_wasted_bytes, reverse=True)
    total_dup = sum(c.eqn_count for c in duplicates)
    return CseReport(
        duplicates=tuple(duplicates),
        total_eqns=n,
        duplicate_eqns=total_dup,
        trace_cache_hit=cache_hit,
    )
