"""Prereg-match (T2-16, #2181, AC-6, F8).

AC-6's claim: a candidate's run config must equal the pre-registered hypothesis+metric sidecar
(the DAG text names bathos "gate_check" as the mechanism) -- fast/loud rejection with a structured
`GateErrorPayload` on any mismatch.

Bathos integration stance, resolved before writing this module: no literal `gate_check` tool
exists in bathos (checked directly against the connected bathos MCP server's tool list and the
installed `bth` CLI's own `--help` -- the closest real analogs are `bth claim validate` /
`mcp__bathos__claim_validate`; the DAG's naming is imprecise, the same class of staleness T2-06
found for `claim_attest_parity`). More fundamentally, this module never imports bathos at all --
`xtrax.run.repro_floor`'s own documented architectural rule ("xtrax has no dependency on bathos,
by design -- a compute library must not depend on the orchestration/experiment-tracking tool built
on top of it") already governs every bathos-adjacent module in this repo, and T2-06's
`metrics_provenance.py` follows it exactly (writes an artifact "by convention," never decides
which bathos tool anchors it). This module is a **pure xtrax-side comparison**: the caller/
orchestrating agent is responsible for having already fetched the pre-registered sidecar from
bathos (via `claim_validate` or equivalent) before calling `assert_prereg_match` -- this module
never reaches out to bathos itself.

`RunConfig`/`PreregSidecar`/`GateErrorPayload` have no existing xtrax type to reuse (`RunSpec` is
an unrelated JAX batching/tiling config; `Candidate`/`FrozenContext` are deliberately opaque
TypeVars) -- these are new, small, frozen dataclasses mirroring bathos's own conceptual shape "by
convention" (same "mirrors bathos's schema by convention, not by import" stance T2-06's TOML
attestation already established), not an untyped dict-diff utility.

Unlike every other T2-1x module, this one does not compose `xtrax.loop.schema_gate`'s
`resolve_candidate_callable` or any candidate-file concept at all -- it operates on run-
configuration *intent* (what hypothesis/metrics a run declares), not the candidate source file or
callable itself. `T2-16 depends_on: [T2-15]` is a DAG pipeline-ordering edge (this gate runs after
checkified-execution passes), not a code composition dependency -- the same relationship T2-11 has
to T2-08.

Extension seam (mirrors every other T2-0X module's stance): raises `PreregMatchError` and stops.
Does not decide what a future loop controller does with a denied candidate -- no #2181 loop
controller exists yet.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RunConfig:
    """A candidate's own run configuration -- the xtrax-side half of the prereg-match comparison.

    Mirrors bathos's `Run` schema's prereg-relevant fields "by convention" (not by import).
    """

    script_sha256: str
    hypothesis: str
    metrics: tuple[str, ...]
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PreregSidecar:
    """A pre-registered hypothesis+metric sidecar snapshot.

    Caller-supplied -- already fetched from bathos (e.g. via `claim_validate`) by whatever
    orchestrates this gate; this module never fetches it itself. Same fields as `RunConfig` by
    design (they're compared for equality); kept as a distinct type so a caller/type-checker can't
    accidentally pass the same object for both sides of the comparison.
    """

    script_sha256: str
    hypothesis: str
    metrics: tuple[str, ...]
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GateErrorPayload:
    """Structured denial payload -- mirrors bathos's own gate-denial shape "by convention".

    `mismatched_fields` is an exhaustive diff (every mismatching dimension, not just the first) so
    a caller debugging a prereg mismatch sees everything wrong in one shot, not one gate-cycle at
    a time. `run_config`/`sidecar` are the original compared objects, for caller introspection.
    """

    reason: str
    mismatched_fields: tuple[str, ...]
    run_config: RunConfig
    sidecar: PreregSidecar


class PreregMatchError(Exception):
    """AC-6's rejection error. Carries the structured `GateErrorPayload` as `.payload`."""

    def __init__(self, payload: GateErrorPayload) -> None:
        super().__init__(payload.reason)
        self.payload = payload


def _diff(run_config: RunConfig, sidecar: PreregSidecar) -> tuple[str, ...]:
    """Return every mismatching dimension between `run_config` and `sidecar` (empty iff equal)."""
    diffs: list[str] = []

    if run_config.script_sha256 != sidecar.script_sha256:
        diffs.append(
            f"script_sha256 mismatch: run_config={run_config.script_sha256!r} "
            f"sidecar={sidecar.script_sha256!r}"
        )

    if run_config.hypothesis != sidecar.hypothesis:
        diffs.append(
            f"hypothesis mismatch: run_config={run_config.hypothesis!r} "
            f"sidecar={sidecar.hypothesis!r}"
        )

    run_metrics = frozenset(run_config.metrics)
    sidecar_metrics = frozenset(sidecar.metrics)
    if run_metrics != sidecar_metrics:
        missing = sidecar_metrics - run_metrics
        extra = run_metrics - sidecar_metrics
        if missing:
            diffs.append(f"metrics missing from run_config: {sorted(missing)}")
        if extra:
            diffs.append(f"metrics present in run_config but not pre-registered: {sorted(extra)}")

    if dict(run_config.config) != dict(sidecar.config):
        diffs.append(
            f"config mismatch: run_config={dict(run_config.config)!r} "
            f"sidecar={dict(sidecar.config)!r}"
        )

    return tuple(diffs)


def assert_prereg_match(run_config: RunConfig, sidecar: PreregSidecar) -> None:
    """The composed AC-6 gate: `run_config == sidecar`, field by field, exhaustive diff on failure.

    Raises:
        PreregMatchError: `run_config` diverges from `sidecar` in any of `script_sha256`,
            `hypothesis`, `metrics`, or `config` -- `.payload.mismatched_fields` names every
            divergence found, not just the first.
    """
    mismatched = _diff(run_config, sidecar)
    if mismatched:
        payload = GateErrorPayload(
            reason=(
                "candidate run config does not match the pre-registered hypothesis+metric sidecar"
            ),
            mismatched_fields=mismatched,
            run_config=run_config,
            sidecar=sidecar,
        )
        raise PreregMatchError(payload)


__all__ = [
    "GateErrorPayload",
    "PreregMatchError",
    "PreregSidecar",
    "RunConfig",
    "assert_prereg_match",
]
