"""#2181 loop admission gate onto T1-10's deterministic validation gate (T2-01, #3069, AC-E1).

A thin, real consumption edge onto `xtrax.composition.validate.validate_graph` -- not a
reimplementation. `xtrax.cli.graph_verb.run_graph_validate` (T1-10) already provides the
CLI-facing exit-1 + structured-JSON-envelope behavior AC-E1's gate text describes; this
module exists to give #2181's future in-process loop code (not yet built) a named, stable,
loop-scoped entry point to import against, rather than every future loop item reaching into
`xtrax.composition` directly and re-deriving the "PASS admits, FAIL/NEEDS_WORK refuses"
framing for itself each time.

No `CandidateRefusedError` or other loop-specific exception type is introduced here. No loop
exists yet to define what "refusal" should actually DO (retry with mutation? log to a
campaign ledger? drop the candidate silently?) -- inventing a stronger contract now, with
zero consumers to validate it against, would be speculative and likely wrong-shaped once
#2181's real loop lands. Callers get the same `GraphValidationResult` T1-10 already
produces and check `.passed` themselves, exactly as `run_graph_validate` does today --
mirrors T1-07's `evaluate.py` leaving `frozen_context`/`candidate` deliberately opaque for
the same reason.
"""

from pathlib import Path

from xtrax.composition.graph import HostPrepGraph
from xtrax.composition.validate import GraphValidationResult, validate_graph

__all__ = ["admit_candidate"]


def admit_candidate(graph: HostPrepGraph, *, root: Path | None = None) -> GraphValidationResult:
    """Run the T1 deterministic validation gate on a candidate composition graph.

    PASS (`result.passed is True`, no failures) admits the candidate into the loop.
    FAIL/NEEDS_WORK (`result.passed is False`) refuses it -- `result.failures` names each
    failing check exactly as T1-10's CLI envelope does. Performs no normalization or
    reinterpretation of `validate_graph`'s verdict.
    """
    return validate_graph(graph, root=root)
