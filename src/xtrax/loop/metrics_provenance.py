"""Metrics-provenance gate (T2-06, #2181, AC-8, F1).

AC-8: every fitness scalar must trace 100% to the immutable evaluator + an attestation --
agent-reported numbers are never accepted (MLR-Bench found an 80% self-reported-metric
fabrication rate; this is the fast/loud backstop against that failure mode).

Terminology note: the DAG text says "traces to the immutable-evaluator stdout envelope", but the
accepted #2181 architecture (Fork 15/16; T1-07's `EvaluateFn`) is an in-process
``(frozen_context, candidate) -> dict[str, float]`` callable that never writes to stdout -- the
literal subprocess-stdout evaluator model (ORTH-3) was proposed but never accepted in the spec's
own decision log. This module reads "envelope" as a structured provenance record wrapping the
fitness dict, not a literal stdout mechanism: :class:`MetricsProvenanceRecord` ties a fitness dict
to the :class:`~xtrax.loop.closure_lock.ClosureManifest` hash that T2-05's `guarded_evaluate`
already verifies before returning it -- that hash *is* the concrete "traces to the immutable
evaluator" proof AC-8 asks for.

Tool-citation correction: the DAG's backlog text says this "rides existing bathos
claim_register/claim_attest_parity -- no BUILD item needed." Both claims are stale. First,
``claim_attest_parity`` is hard-bound to literature-parity semantics (validates
``metadata.parity_run_type == 'literature_parity'``) and was the subject of a NO-GO decision on
extensibility (bathos gate #3488, 260714, decided after this DAG was authored) -- it does not fit
a general metrics-provenance record. Second, and more fundamentally: this module *is* the BUILD
item this repo needs, because nothing before it produces a provenance artifact at all. What this
module does NOT do -- matching :mod:`xtrax.run.repro_floor`'s established "cross-tool integration
seam" pattern exactly -- is import bathos or decide which bathos tool/kind anchors the artifact
into a catalog. It writes an attestation-shaped TOML (``kind = "metrics_provenance"``, mirroring
bathos's `[attestation]` schema *by convention*) and leaves registration to a caller/orchestrating
agent, the same way :func:`xtrax.run.repro_floor.write_repro_floor_attestation` does.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from xtrax.loop.closure_lock import ClosureManifest, guarded_evaluate
from xtrax.run.repro_floor import toml_quote
from xtrax.stages.evaluate import Candidate, EvaluateFn, FrozenContext


class UnprovenancedMetricsError(Exception):
    """A fitness dict's provenance hash doesn't match the currently-locked evaluator closure.

    AC-8's "agent-reported numbers never accepted" gate: fires when a `MetricsProvenanceRecord`
    is checked against a `ClosureManifest` it wasn't actually produced against -- either it never
    went through `evaluate_with_provenance`, or the locked closure has since changed.
    """


@dataclass(frozen=True, slots=True)
class MetricsProvenanceRecord:
    """A fitness dict plus the proof it traces to one specific, immutable evaluator closure."""

    fitness: Mapping[str, float]
    evaluator_closure_hash: str
    run_id: str
    created_by: str = "xtrax.loop.metrics_provenance"
    created_at: str = ""


def evaluate_with_provenance(
    locked: ClosureManifest,
    evaluator: EvaluateFn[FrozenContext, Candidate],
    frozen_context: FrozenContext,
    candidate: Candidate,
    *,
    current_config: Mapping[str, object],
    run_id: str,
    candidate_touched_paths: frozenset[Path] = frozenset(),
    now: datetime | None = None,
) -> MetricsProvenanceRecord:
    """Run the evaluator through T2-05's `guarded_evaluate` and wrap the result with provenance.

    Reuses `guarded_evaluate` rather than calling `evaluator` directly -- the closure-hash,
    protected-path, and unlisted-read checks it performs are exactly what makes the resulting
    `evaluator_closure_hash` a meaningful provenance proof, not a bypassable formality.
    """
    fitness = guarded_evaluate(
        locked,
        evaluator,
        frozen_context,
        candidate,
        current_config=current_config,
        candidate_touched_paths=candidate_touched_paths,
    )
    return MetricsProvenanceRecord(
        fitness=fitness,
        evaluator_closure_hash=locked.closure_hash,
        run_id=run_id,
        created_at=(now or datetime.now(UTC)).isoformat(),
    )


def verify_metrics_provenance(record: MetricsProvenanceRecord, *, locked: ClosureManifest) -> None:
    """Refuse a fitness record that doesn't trace to the currently-locked evaluator closure.

    Raises:
        UnprovenancedMetricsError: `record.evaluator_closure_hash` doesn't match
            `locked.closure_hash` -- the record was never produced by `evaluate_with_provenance`
            against this closure, or the closure has since changed.
    """
    if record.evaluator_closure_hash != locked.closure_hash:
        msg = (
            f"metrics provenance mismatch: record traces to closure_hash="
            f"{record.evaluator_closure_hash!r}, currently-locked closure_hash="
            f"{locked.closure_hash!r} -- discard, void iteration"
        )
        raise UnprovenancedMetricsError(msg)


def build_metrics_provenance_attestation_toml(record: MetricsProvenanceRecord) -> str:
    """Serialize `record` into bathos's `[attestation]` (kind="metrics_provenance") TOML body.

    Mirrors `xtrax.run.repro_floor.build_repro_floor_attestation_toml`'s shape and escaping
    (`toml_quote`) exactly -- see module docstring for why this doesn't call any bathos tool.
    """
    fitness_toml = ", ".join(f"{toml_quote(k)} = {v}" for k, v in sorted(record.fitness.items()))
    return (
        "# Attestation (metrics_provenance)\n"
        "# Generated via xtrax.loop.metrics_provenance -- register with bathos's\n"
        "# attestation_register MCP tool (or `bth attestation register <path>`)\n"
        "# to anchor into the bathos catalog (see module docstring).\n\n"
        "[attestation]\n"
        'kind = "metrics_provenance"\n'
        f"attested = {{ run_id = {toml_quote(record.run_id)}, "
        f"evaluator_closure_hash = {toml_quote(record.evaluator_closure_hash)} }}\n"
        f"fitness = {{ {fitness_toml} }}\n"
        f"created_by = {toml_quote(record.created_by)}\n"
        f"created_at = {toml_quote(record.created_at)}\n"
    )


def write_metrics_provenance_attestation(record: MetricsProvenanceRecord, path: Path) -> Path:
    """Write `record`'s attestation TOML to `path`. Creates parent directories as needed."""
    toml_text = build_metrics_provenance_attestation_toml(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(toml_text)
    return path


__all__ = [
    "MetricsProvenanceRecord",
    "UnprovenancedMetricsError",
    "build_metrics_provenance_attestation_toml",
    "evaluate_with_provenance",
    "verify_metrics_provenance",
    "write_metrics_provenance_attestation",
]
