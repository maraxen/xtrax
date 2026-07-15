"""Structure-tripwire (T2-13, #2181, AC-3).

AC-3's claim: a schema-passing candidate (T2-12) must be checked, on one tiny concrete batch,
that `jax.eval_shape`'s abstract output structure actually matches what the candidate produces
when really run -- guarding against data-dependent control flow or side-effecting tree operations
silently diverging from the traced shape. Unlike T2-11/T2-12, this gate is **not zero-cost**: it
executes the candidate exactly once.

The verification machinery already exists, fully built: `xtrax.inference.verify.verify_structure`
does exactly this abstract-vs-concrete pytree/shape/dtype comparison, and
`xtrax.inference.errors.StructureMismatchError` is already the correctly-named, correctly-shaped
exception AC-3's own text asks for. This module is the same thin glue layer T2-12 is: resolve the
candidate (reusing T2-12's `resolve_candidate_callable` directly -- a real composition dependency,
mirroring the DAG's own `T2-13 depends_on: [T2-12]` edge, the same way T2-06 composes T2-05 and
T2-29 composes T2-05), then call `verify_structure` and let its own exception propagate unchanged.

`StructureMismatchError` is deliberately **not** re-wrapped in a new loop-local exception type --
it already is exactly the exception AC-3 names, already carries the right pytree/shape/dtype
diagnostic detail, and re-wrapping it here would only strip information for no benefit. This
differs from cases like T2-06, which introduces its own exception because it covers a genuinely
distinct failure mode its composed dependency doesn't already name -- that reasoning doesn't apply
here, since AC-3 and `verify_structure`'s own contract are the same claim. A new local exception,
`CandidateResolutionError` (re-exported from `schema_gate`, not redefined), still covers the one
failure mode `verify_structure` itself has no opinion on: the candidate couldn't be imported at
all.

Extension seam (mirrors every other T2-0X module's stance): this module raises and stops. It does
not decide what a future loop controller does after a `StructureMismatchError` (retry with a
different candidate? escalate? log to a campaign ledger?) -- no #2181 loop controller exists yet
to validate that design against.
"""

from pathlib import Path
from typing import Any

from xtrax.inference.errors import StructureMismatchError
from xtrax.inference.verify import verify_structure
from xtrax.loop.schema_gate import CandidateResolutionError, resolve_candidate_callable

__all__ = [
    "CandidateResolutionError",
    "StructureMismatchError",
    "assert_structure_tripwire",
]


def assert_structure_tripwire(
    candidate_path: Path,
    callable_name: str,
    *,
    abstract_inputs: list[Any],
    concrete_inputs: list[Any],
) -> None:
    """The composed AC-3 gate: resolve the candidate, then verify abstract == concrete structure.

    Executes the candidate exactly once (inside `verify_structure`'s own concrete call) -- this is
    AC-3's explicit "not zero-cost, first-concrete-run tripwire" claim, not a zero-FLOPs check like
    T2-11/T2-12.

    Raises:
        CandidateResolutionError: `candidate_path` can't be imported, or `callable_name` isn't a
            callable attribute (see `xtrax.loop.schema_gate.resolve_candidate_callable`).
        StructureMismatchError: the concrete output's pytree structure, a leaf's shape, or a
            leaf's dtype diverges from the abstract (`jax.eval_shape`) trace. Raised by, and
            propagated unchanged from, `xtrax.inference.verify.verify_structure`.
    """
    fn = resolve_candidate_callable(candidate_path, callable_name)
    verify_structure(fn, abstract_inputs, concrete_inputs)
