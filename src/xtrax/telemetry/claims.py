"""Claim-time refusal: a result may not be cited unless its run was recorded.

This is the enforcement tier that survives an opt-out. XTRAX_TELEMETRY_OPTOUT
lets a run proceed without capture, but the row it writes is marked
``opted_out``, and :func:`assert_run_citable` refuses it here. Opting out of
capture is therefore not a way to obtain a citable result -- only a way to get a
result that is honestly labelled as uncitable.

Why a standalone predicate rather than an extension of
``xtrax.profiling.claims.assert_claim_supported``. That function guards
``ProbeRecord``, which is keyed by ``probe_id`` and carries no ``run_id`` at all.
Threading run identity through it would mean adding a field to ProbeRecord and,
under that module's own documented bump rule, a MAJOR ``CONTRACT_VERSION``
bump -- invalidating every existing profiling record for a reason unrelated to
profiling. The two contracts guard different things and are better kept
separate: this one asks "was this run recorded?", that one asks "does this
measurement support this class of claim?". A caller wanting both calls both.

Modelled on ``xtrax.loop.metrics_provenance``'s ``UnprovenancedMetricsError``
and its AC-8 rule, *agent-reported numbers are never accepted*.
"""

from pathlib import Path

from xtrax.telemetry.ledger import find_run
from xtrax.telemetry.record import RunLedgerRecord


class UnrecordedRunError(RuntimeError):
    """Raised when a result is cited but its run has no citable ledger row.

    Carries the run_id and the reason so the caller can tell the three cases
    apart: no row at all (the run never went through xtrax, or the ledger was
    lost), a row marked non-citable (degraded, opted out, or failed), and a run
    that is genuinely fine.
    """


def run_is_citable(run_id: str, root: "Path | str | None" = None) -> bool:
    """Whether ``run_id`` has a ``complete`` ledger row. Never raises."""
    record = find_run(run_id, root)
    return record is not None and record.is_citable


def assert_run_citable(run_id: str, root: "Path | str | None" = None) -> RunLedgerRecord:
    """Return the run's row, or raise :class:`UnrecordedRunError`.

    Returning the record rather than None on success is deliberate: a caller
    that has just proved a run is citable almost always wants its provenance
    next, and handing it back removes the temptation to re-read the ledger with
    an unchecked lookup.
    """
    record = find_run(run_id, root)
    if record is None:
        raise UnrecordedRunError(
            f"run_id={run_id!r} has no ledger row, so nothing about it can be "
            "cited: there is no record of what code, environment, or computation "
            "produced this result. If the run predates the ledger, re-run it; if "
            "the ledger was moved, point XTRAX_LEDGER_ROOT at it."
        )
    if not record.is_citable:
        raise UnrecordedRunError(
            f"run_id={run_id!r} is recorded but not citable "
            f"(telemetry_status={record.telemetry_status!r}: "
            f"{record.status_reason or 'no reason recorded'}). A result from a run "
            "whose provenance is incomplete cannot be attributed to the code that "
            "produced it."
        )
    return record


def filter_citable(
    run_ids: "list[str] | tuple[str, ...]",
    root: "Path | str | None" = None,
) -> "tuple[list[str], dict[str, str]]":
    """Split ``run_ids`` into ``(citable, {rejected_id: reason})``.

    The batch form, for an aggregation that should proceed on the runs it can
    legitimately use while reporting exactly which it dropped and why. Silently
    skipping the rest would reintroduce the failure this whole subsystem exists
    to prevent.
    """
    citable: list[str] = []
    rejected: dict[str, str] = {}
    for run_id in run_ids:
        try:
            assert_run_citable(run_id, root)
        except UnrecordedRunError as exc:
            rejected[run_id] = str(exc)
        else:
            citable.append(run_id)
    return citable, rejected


__all__ = [
    "UnrecordedRunError",
    "assert_run_citable",
    "filter_citable",
    "run_is_citable",
]
