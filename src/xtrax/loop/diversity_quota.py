"""Diversity-quota-semantic (T2-18, #2181, AC-12, F5, PM-6).

AC-12's claim: over N consecutive iterations (default N=5), at least one candidate mutation must
be semantically-structural (not merely cosmetic) -- an evolutionary loop that only ever tweaks
constants or renames variables is "syntactic counting theater" (PM-6), not genuine search. When
the quota is unmet, the epic's Leap-Path mechanism should fire.

Grounding note (verified before writing any code, not guessed): the DAG text's "AST/composition-IR
structural diff" framing doesn't hold up under research -- `xtrax.composition`'s `HostPrepGraph`/
`HostPrepGraphNode` have no diff/compare logic anywhere, and no loop candidate has ever been
converted into a `HostPrepGraph`; every T2-1x module treats "candidate" as a source-file `Path` +
`callable_name`, never a composition-IR object. A naive `HostPrepGraph.__eq__` would also BE the
"syntactic theater" PM-6 warns against (`nodes` is a positionally-compared tuple, so a cosmetic
reorder would register as structural). Confirmed as a design decision: this module diffs the
**candidate's own source code via `ast`**, matching every prior module's candidate concept.

"Leap-Path" (per `.praxia/docs/research/260702_roadmap-research-synthesis.md:44` and T2-03's own
research note) is a term borrowed directly from the AutoSOTA paper's "Leap Path bifurcation"
(forcing the next iteration onto a structurally-novel idea after a run of parameter-only
iterations) -- not an xtrax-specified mechanism, and no detail exists anywhere on *how* a forced
structural mutation gets proposed/generated. Extension seam (mirrors every other T2-0X module's
stance): this module returns a `DiversityQuotaDecision` signal. It does NOT implement Leap-Path
itself -- no #2181 loop controller exists yet to drive it.

Cosmetic-invariance scope (confirmed as a design decision, not assumed): PM-6's "renames,
commutative reorders don't count" is bounded to what's safely automatable --

- Identifier renaming: every `Name`/`arg`/`FunctionDef` identifier is canonicalized to `_id{N}`
  based on first-occurrence order, so two structurally-identical trees using different variable
  names normalize identically.
- Commutative binary operators (`+`, `*`, `|`, `&`, `^` specifically): `a + b` and `b + a` are
  canonicalized to the same operand order.

General statement-level reordering is explicitly **not** attempted -- it can change semantics via
side effects or data dependencies (a real dataflow-analysis problem, out of scope here) --
documented as a known, bounded limitation rather than silently attempted and gotten wrong.

The commutative-operand swap is applied AFTER renaming (not before): this is what makes
rename-then-compare and compare-then-rename produce the same answer for the common case of a pure
variable rename inside a commutative expression -- e.g. `def f(a, b): return a + b` and
`def f(x, y): return y + x` both canonicalize identifiers by first-occurrence order to the same
`_id0`, `_id1` sequence, and only then does the commutative sort (applied to the already-renamed
operand dumps) put them in the same order.

Identifier canonicalization is scope-local (a fresh name map per `FunctionDef`, restored on exit),
not module-global -- self-audit finding, fixed before merge: a module-global map would collapse two
unrelated nested functions that happen to reuse the same parameter name (e.g. two inner functions
each shadowing an outer `a`) onto the same canonical id by string coincidence, misclassifying an
unrelated pair of candidates as cosmetically identical, or the reverse. This still does not track
true lexical scoping -- a closure that *reads* (rather than shadows) an outer-scope name is treated
as a fresh local in its own scope, not resolved to the enclosing binding -- out of scope, same
bounded-limitation treatment as the statement-reordering gap above.

Extension seam (mirrors every other T2-0X module's stance): this module is a pure decision
function, like T2-17's `compute_ratchet_decision` -- an unmet quota is a normal, expected outcome
in an evolutionary loop (most iterations are legitimately incremental), not an anomaly. Only
malformed inputs (a `SyntaxError` in candidate source) raise.
"""

import ast
from collections.abc import Sequence
from dataclasses import dataclass

_COMMUTATIVE_OPS: tuple[type[ast.operator], ...] = (
    ast.Add,
    ast.Mult,
    ast.BitOr,
    ast.BitAnd,
    ast.BitXor,
)


class DiversityQuotaInputError(Exception):
    """A supplied candidate source string failed to parse as Python."""


@dataclass(frozen=True, slots=True)
class DiversityQuotaDecision:
    """The composed AC-12 decision: whether the structural-mutation quota was met.

    `window` echoes the (possibly truncated) tail of `recent_structural_flags` actually used for
    the decision, for caller introspection/logging.
    """

    quota_met: bool
    leap_path_required: bool
    window: tuple[bool, ...]


class _NormalizingTransformer(ast.NodeTransformer):
    """Canonicalizes identifier names (by first-occurrence order) and commutative-binop operand
    order, so two ASTs that differ only cosmetically dump to the same string.
    """

    def __init__(self) -> None:
        self._canonical_names: dict[str, str] = {}

    def _canonicalize(self, original: str) -> str:
        if original not in self._canonical_names:
            self._canonical_names[original] = f"_id{len(self._canonical_names)}"
        return self._canonical_names[original]

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self._canonicalize(node.id)
        return self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self._canonicalize(node.arg)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # Function bodies get a fresh, scope-local name map before descending, and the
        # enclosing scope's map is restored afterward. Without this, two DIFFERENT nested
        # functions that happen to reuse the same parameter name (e.g. shadowing an outer `a`)
        # would collapse onto the same canonical id purely by string coincidence, misclassifying
        # an unrelated pair of candidates as identical -- or the reverse: two alpha-equivalent
        # candidates using differently-named locals in a nested scope would fail to normalize to
        # the same form. This does not track true lexical scoping (a closure that reads, rather
        # than shadows, an outer-scope name is treated as a fresh local) -- out of scope, same as
        # the module docstring's statement-reordering limitation.
        node.name = self._canonicalize(node.name)
        outer_scope = self._canonical_names
        self._canonical_names = {}
        result = self.generic_visit(node)
        self._canonical_names = outer_scope
        return result

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, _COMMUTATIVE_OPS):
            left_dump = ast.dump(node.left)
            right_dump = ast.dump(node.right)
            if left_dump > right_dump:
                node.left, node.right = node.right, node.left
        return node


def _normalize(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        msg = f"candidate source is not valid Python: {exc}"
        raise DiversityQuotaInputError(msg) from exc
    normalized = _NormalizingTransformer().visit(tree)
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized)


def is_structural_mutation(previous_source: str, candidate_source: str) -> bool:
    """Pairwise check: `True` iff `candidate_source` differs from `previous_source` after
    identifier-renaming + commutative-binary-op normalization.

    Raises:
        DiversityQuotaInputError: either source string fails to parse as Python.
    """
    return _normalize(previous_source) != _normalize(candidate_source)


def assert_diversity_quota(
    recent_structural_flags: Sequence[bool],
    *,
    window_size: int = 5,
) -> DiversityQuotaDecision:
    """The composed AC-12 gate: at least one structural mutation over the last `window_size`
    iterations.

    Args:
        recent_structural_flags: the caller-maintained rolling window of `is_structural_mutation()`
            results, oldest to newest -- this module never computes or stores this history itself.
        window_size: default 5, per AC-12's own text ("N=5").

    Returns:
        A `DiversityQuotaDecision`. `quota_met=False` (`leap_path_required=True`) is a normal
        outcome, not an error -- it signals a future loop controller to force a structural
        mutation, it does not perform one. If fewer than `window_size` iterations of history exist
        yet, the quota is treated as met (there aren't yet `window_size` consecutive iterations to
        judge, per AC-12's own "over N consecutive iterations" framing) -- `window` reflects
        exactly what was supplied.
    """
    window = tuple(recent_structural_flags[-window_size:])
    if len(window) < window_size:
        return DiversityQuotaDecision(quota_met=True, leap_path_required=False, window=window)

    quota_met = any(window)
    return DiversityQuotaDecision(
        quota_met=quota_met, leap_path_required=not quota_met, window=window
    )


__all__ = [
    "DiversityQuotaDecision",
    "DiversityQuotaInputError",
    "assert_diversity_quota",
    "is_structural_mutation",
]
