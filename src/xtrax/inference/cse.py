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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from weakref import WeakKeyDictionary

import jax
import numpy as np

from xtrax.inference.errors import CseTraceError
from xtrax.profiling.jaxpr import iter_jaxpr_eqns, sub_jaxprs

__all__ = [
    "CseDuplicateClass",
    "CseReport",
    "align_call_invars",
    "analyze_cse",
    "trace_input_dependence",
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


def _is_literal(value: Any) -> bool:
    """True for a jaxpr Literal operand.

    Jaxpr literals are not a public class the walker can import without jax,
    so the check is the runtime type name, the same test ``analyze_cse`` uses
    when it canonicalizes constant operands.
    """
    return type(value).__name__ == "Literal"


def _literal_key(lit) -> tuple[str, bytes]:
    """Canonical key for a Literal operand: (dtype string, raw value bytes)."""
    val = lit.val
    return (str(val.dtype), np_ascontiguous_tobytes(val))


def np_ascontiguous_tobytes(val: Any) -> bytes:
    """Contiguous C-order bytes for any array-like literal value."""
    return np.ascontiguousarray(np.asarray(val)).tobytes(order="C")


def _params_repr(params: dict) -> str:
    items = sorted((k, repr(v)) for k, v in params.items())
    return repr(items)


# Per-function trace memo: fn object -> last ClosedJaxpr seen. make_jaxpr itself
# memoizes per (fn identity, shapes) and returns the SAME ClosedJaxpr object on a
# cache hit; we detect that by identity comparison here so the report can flag
# stale-closure risk to callers.
_TRACE_MEMO: WeakKeyDictionary = WeakKeyDictionary()


class _UnionFind:
    """Union-find over equation indices, with path halving."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _trace_once(
    fn: Callable[..., jax.Array],
    abstract_inputs: Sequence,
) -> tuple[Any, bool]:
    """Trace ``fn`` through the memo; return ``(closed_jaxpr, cache_hit)``."""
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
    return closed, cache_hit


def _op_token(value: Any, var_token: dict[int, tuple]) -> tuple:
    if _is_literal(value):
        return ("lit", _literal_key(value))
    return var_token.get(id(value), ("free", id(value)))


def _fingerprint_round(eqns: Sequence[Any], uf: _UnionFind) -> dict[tuple, list[int]]:
    """One fixpoint round: hash eqns under current representative map."""
    var_token: dict[int, tuple] = {}
    for i, eq in enumerate(eqns):
        tok = ("eq", uf.find(i))
        for ov in eq.outvars:
            var_token[id(ov)] = tok

    fps: dict[tuple, list[int]] = {}
    for i, eq in enumerate(eqns):
        fp = (
            eq.primitive.name,
            _params_repr(eq.params),
            tuple(_op_token(v, var_token) for v in eq.invars),
        )
        fps.setdefault(fp, []).append(i)
    return fps


def _merge_round(fps: dict[tuple, list[int]], uf: _UnionFind) -> bool:
    """Union every fingerprint group onto its first member; True if anything merged."""
    changed = False
    for members in fps.values():
        if len(members) > 1:
            anchor = members[0]
            for other in members[1:]:
                if uf.find(anchor) != uf.find(other):
                    uf.union(anchor, other)
                    changed = True
    return changed


def _merge_to_fixpoint(eqns: Sequence[Any], uf: _UnionFind) -> None:
    """Merge, rewrite representatives, repeat until stable."""
    n = len(eqns)
    prev_canonical = None
    while True:
        changed = _merge_round(_fingerprint_round(eqns, uf), uf)
        canonical = tuple(uf.find(i) for i in range(n))
        if not changed or canonical == prev_canonical:
            break
        prev_canonical = canonical


def _aval_nbytes(aval: Any) -> int:
    size = 1
    for dim in getattr(aval, "shape", ()):
        size *= int(dim)
    dtype = getattr(aval, "dtype", np.dtype(np.float32))
    return size * np.dtype(dtype).itemsize


def _duplicate_class(members: list[int], eqns: Sequence[Any]) -> CseDuplicateClass:
    eq0 = eqns[members[0]]
    out_shapes = tuple(tuple(int(d) for d in getattr(ov.aval, "shape", ())) for ov in eq0.outvars)
    # Wasted bytes: all-but-one member's output size (smallest member kept).
    nbytes = sum(_aval_nbytes(ov.aval) for ov in eq0.outvars)
    return CseDuplicateClass(
        primitive=eq0.primitive.name,
        eqn_count=len(members),
        params_fingerprint=_params_repr(eq0.params)[:64],
        invar_shapes=out_shapes,
        est_wasted_bytes=nbytes * (len(members) - 1),
    )


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
    closed, cache_hit = _trace_once(fn, abstract_inputs)
    eqns = closed.jaxpr.eqns
    n = len(eqns)

    uf = _UnionFind(n)
    _merge_to_fixpoint(eqns, uf)

    # Collect final equivalence classes.
    classes: dict[int, list[int]] = {}
    for i in range(n):
        classes.setdefault(uf.find(i), []).append(i)

    duplicates = [
        _duplicate_class(members, eqns) for members in classes.values() if len(members) >= 2
    ]
    duplicates.sort(key=lambda c: c.est_wasted_bytes, reverse=True)
    total_dup = sum(c.eqn_count for c in duplicates)
    return CseReport(
        duplicates=tuple(duplicates),
        total_eqns=n,
        duplicate_eqns=total_dup,
        trace_cache_hit=cache_hit,
    )


def align_call_invars(
    outer: Sequence[Any],
    inner: Sequence[Any],
) -> tuple[tuple[Any, Any | None], ...]:
    """Pair callee invars with the caller invars that bind them.

    Pairs are aligned from the right. A longer caller list drops its prefix
    (``cond``'s predicate is that prefix). A longer callee list pairs its
    leading invars with ``None`` (closed-over consts with no caller input).

    Args:
        outer: Invars of the calling equation, in order.
        inner: Invars of the callee jaxpr, in order.

    Returns:
        ``(callee invar, caller invar or None)`` pairs, one per callee invar.
    """
    if len(inner) <= len(outer):
        start = len(outer) - len(inner)
        return tuple((inner[i], outer[start + i]) for i in range(len(inner)))
    pad = len(inner) - len(outer)
    leading = tuple((inner[i], None) for i in range(pad))
    trailing = tuple((inner[pad + i], outer[i]) for i in range(len(outer)))
    return leading + trailing


@dataclass(frozen=True)
class InputDependence:
    """Data dependence of a traced function on its top-level invars.

    Attributes:
        intermediates: ``(primitive, input indices)`` for every produced value,
            in walker order. A value's index set is the top-level inputs it reads.
        outputs: One input-index set per top-level output, in output order.
    """

    intermediates: tuple[tuple[str, frozenset[int]], ...]
    outputs: tuple[frozenset[int], ...]


def _dep_of(value: Any, deps: dict[int, frozenset[int]]) -> frozenset[int]:
    if _is_literal(value):
        return frozenset()
    return deps.get(id(value), frozenset())


def _scan_counts(params: Mapping[str, Any]) -> tuple[int, int]:
    """Return ``(num_consts, num_carry)`` for a ``scan`` equation's params.

    Older JAX stores the counts as ``num_consts`` / ``num_carry``. Newer JAX
    drops them and keeps only the ``ft_in`` / ``ft_out`` flattree packs, whose
    ``elts`` are one group per (consts, carry, xs) and (carry, ys). The counts
    are leaf counts because scan invars are flattened.
    """
    if "num_consts" in params:
        return params["num_consts"], params["num_carry"]
    return len(params["ft_in"].elts[0].elts), len(params["ft_out"].elts[0].elts)


class _DependenceWalk:
    """Mutable state for one :func:`trace_input_dependence` call.

    Holds the value-to-dependence map, the produced-intermediates log, and the
    equation order. Each method handles one equation family; the walk order and
    the record bookkeeping are the same as the former nested closures.
    """

    def __init__(self, closed: Any) -> None:
        self.deps: dict[int, frozenset[int]] = {}
        for const in getattr(closed, "constvars", ()):
            self.deps[id(const)] = frozenset()
        for index, invar in enumerate(closed.invars):
            self.deps[id(invar)] = frozenset((index,))
        self.order = {id(eqn): index for index, eqn in enumerate(iter_jaxpr_eqns(closed))}
        self.produced: list[tuple[int, str, frozenset[int]]] = []

    def run(self, closed: Any) -> InputDependence:
        self._visit(closed)
        self.produced.sort(key=lambda item: item[0])
        outputs = tuple(_dep_of(out, self.deps) for out in closed.outvars)
        intermediates = tuple((name, dep) for _, name, dep in self.produced)
        return InputDependence(intermediates, outputs)

    def _record(self, eqn: Any, per_out: Sequence[frozenset[int]]) -> None:
        slot = self.order[id(eqn)]
        for out, dep in zip(eqn.outvars, per_out, strict=True):
            self.deps[id(out)] = dep
            self.produced.append((slot, eqn.primitive.name, dep))

    def _bind(self, eqn: Any, sub: Any, paired_outer: set[int]) -> list[frozenset[int]]:
        for inner, outer in align_call_invars(eqn.invars, sub.invars):
            if outer is None:
                self.deps[id(inner)] = frozenset()
                continue
            paired_outer.add(id(outer))
            self.deps[id(inner)] = _dep_of(outer, self.deps)
        self._visit(sub)
        return [_dep_of(out, self.deps) for out in sub.outvars]

    def _seed(self, jaxpr: Any, in_deps: Sequence[frozenset[int]]) -> list[frozenset[int]]:
        for inner, dep in zip(jaxpr.invars, in_deps, strict=True):
            self.deps[id(inner)] = dep
        self._visit(jaxpr)
        return [_dep_of(out, self.deps) for out in jaxpr.outvars]

    def _carry_fixpoint(
        self,
        body: Any,
        lead: Sequence[frozenset[int]],
        carry: Sequence[frozenset[int]],
        tail: Sequence[frozenset[int]],
        mark: int,
    ) -> tuple[list[frozenset[int]], list[frozenset[int]]]:
        """Iterate a loop body until its carry dependence stops growing.

        A carry slot can pick up an input only on a later iteration (the body
        forwards it from another slot), so one pass is not enough. Each round
        unions carry-in with carry-out; the sets only grow, and a chain through
        ``n`` carry slots settles within ``n + 1`` rounds. Records from
        superseded rounds are dropped, so the converged round is the only one
        kept.

        Returns:
            The converged carry sets and the body's per-output sets from the
            last round, which ran on those converged sets.
        """
        current = list(carry)
        outs: list[frozenset[int]] = []
        for _ in range(len(current) + 1):
            del self.produced[mark:]
            outs = self._seed(body, [*lead, *current, *tail])
            grown = [c | o for c, o in zip(current, outs[: len(current)], strict=True)]
            if grown == current:
                break
            current = grown
        return current, outs

    def _visit_scan(self, eqn: Any, op_deps: tuple[frozenset[int], ...]) -> None:
        n_consts, n_carry = _scan_counts(eqn.params)
        body = next(sub_jaxprs(eqn.params["jaxpr"]))
        carry, outs = self._carry_fixpoint(
            body,
            op_deps[:n_consts],
            op_deps[n_consts : n_consts + n_carry],
            op_deps[n_consts + n_carry :],
            len(self.produced),
        )
        self._record(eqn, [*carry, *outs[n_carry:]])

    def _visit_while(self, eqn: Any, op_deps: tuple[frozenset[int], ...]) -> None:
        n_cond = eqn.params["cond_nconsts"]
        n_body = eqn.params["body_nconsts"]
        cond_consts = op_deps[:n_cond]
        body = next(sub_jaxprs(eqn.params["body_jaxpr"]))
        cond = next(sub_jaxprs(eqn.params["cond_jaxpr"]))
        carry, _ = self._carry_fixpoint(
            body,
            op_deps[n_cond : n_cond + n_body],
            op_deps[n_cond + n_body :],
            (),
            len(self.produced),
        )
        # The trip count is decided by the predicate, so a carry the predicate
        # reads makes every carry output depend on it. This is conservative:
        # it does not track which iterations actually run.
        self._seed(cond, [*cond_consts, *carry])
        trips = frozenset().union(*cond_consts, *carry)
        self._record(eqn, [dep | trips for dep in carry])

    def _visit_call(self, eqn: Any, subs: list[Any]) -> None:
        """Equation with sub-jaxprs: per-output union over every callee, plus extras.

        Operands that are not paired with a callee invar are unioned into every
        output, so a non-branch input such as a ``cond`` predicate still counts.
        """
        paired_outer: set[int] = set()
        first, *rest = subs
        merged = self._bind(eqn, first, paired_outer)
        for sub in rest:
            nxt = self._bind(eqn, sub, paired_outer)
            merged = [left | right for left, right in zip(merged, nxt, strict=True)]
        extra: frozenset[int] = frozenset()
        for value in eqn.invars:
            if id(value) not in paired_outer:
                extra = extra | _dep_of(value, self.deps)
        self._record(eqn, [dep | extra for dep in merged])

    def _visit(self, jaxpr: Any) -> None:
        for eqn in jaxpr.eqns:
            op_deps = tuple(_dep_of(v, self.deps) for v in eqn.invars)
            name = eqn.primitive.name
            if name == "scan":
                self._visit_scan(eqn, op_deps)
                continue
            if name == "while":
                self._visit_while(eqn, op_deps)
                continue
            subs = list(sub_jaxprs(eqn.params))
            if not subs:
                union = frozenset().union(*op_deps)
                self._record(eqn, (union,) * len(eqn.outvars))
                continue
            self._visit_call(eqn, subs)


def trace_input_dependence(closed: Any) -> InputDependence:
    """Map every produced value to the top-level inputs it reads.

    ``analyze_cse`` stays detection-only. This walk reuses its literal test
    and the public jaxpr walker (``iter_jaxpr_eqns``, ``sub_jaxprs``) so a
    ``sin`` inside ``remat`` / ``custom_jvp`` / ``jit`` (pjit) is visible.

    Call-equation outputs take the callee's per-output dependence, not the
    union of every operand. ``cond`` also unions the predicate, which is not
    a branch invar.

    ``scan`` and ``while`` bodies are iterated until the carry dependence
    stops growing. A ``while`` output also reads every input its predicate
    reads, since the predicate decides how many iterations run.

    Args:
        closed: A closed jaxpr from ``jax.make_jaxpr``, or a bare jaxpr.

    Returns:
        InputDependence for intermediates and outputs.
    """
    return _DependenceWalk(closed).run(closed)
