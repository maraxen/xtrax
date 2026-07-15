"""Info-barrier-lint (T2-07, #2181, AC-9, F3, PM-5, ORTH-2).

PM-5's failure mode (`.praxia/docs/specs/260702_design-the-2181-agentic-algorithm-evolut.md`):
ORTH-2 redacted raw stdout and a lint gate confirmed no raw-log read path was reachable from
agent context -- but the schema-validated JSON failure envelope itself carried enough
structured signal (exact failing shapes, error taxonomy, per-iteration fitness deltas at full
precision) that the agent reverse-engineered held-out characteristics and overfit. "The barrier
guarded the channel we designed and ignored the channel we sanctioned." This module is the fix:
`lint_agent_envelope` bounds the sanctioned envelope's *content*, not merely its channel --
(a) every field must be in a fixed whitelist (`SANCTIONED_FIELDS`, no raw-log-named field, no
value that itself looks like a raw-log path), (b) no per-split metric breakdown, and (c) any
revealed numeric fitness delta must respect a `DeltaExposurePolicy` (rounded precision + how
often a raw delta is revealed at all -- other iterations get only the coarse, non-numeric
`fitness_trend` label).

Composes T2-06's `MetricsProvenanceRecord` directly (T2-07 depends_on T2-06): the only fitness
value `build_sanctioned_envelope` ever reads comes from an already-provenanced record, so an
envelope built any other way has no upstream source to route through this module's own gate.

Extension seam (mirrors `xtrax.loop.admission`/`closure_lock`/`evaluator_completeness`'s
stance): this module raises on a lint violation and stops there. It does not decide what a
human-in-the-loop response looks like, and it does not define what "convergence" or "iteration
voided" mean operationally -- no #2181 loop controller exists yet to own that.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from xtrax.loop.metrics_provenance import MetricsProvenanceRecord

SANCTIONED_FIELDS = frozenset(
    {"status", "run_id", "iteration", "error_taxonomy", "fitness_trend", "fitness_delta"}
)

RAW_LOG_FIELD_NAMES = frozenset(
    {
        "raw_log",
        "stdout",
        "stderr",
        "log_path",
        "raw_output",
        "traceback",
        "full_log",
        "log",
        "logs",
        "log_file",
    }
)

# Values (not just field names) are also scanned for raw-log markers: a whitelisted field
# (e.g. error_taxonomy) could still leak a raw-log read path through its *value* rather than
# its key -- AC-9(a) is about reachability, not merely field naming. This is a small, fixed
# heuristic list, not an exhaustive path-detection grammar (it won't catch e.g. ".out",
# "core.dump", or Windows-style paths) -- acceptable for a lint gate over an in-repo envelope
# shape, but a known coverage gap, not a completeness guarantee.
_RAW_LOG_VALUE_MARKERS = (".log", "/logs/", "stdout", "stderr", "traceback")

PER_SPLIT_FIELD_NAMES = frozenset({"per_split", "splits", "split_scores"})
PER_SPLIT_PREFIXES = ("train_", "val_", "test_", "eval_", "holdout_", "split_")


class InfoBarrierViolationError(Exception):
    """Base for AC-9's info-barrier-lint failures.

    Iteration blocked; LOUD-FAIL; never skip-on-drift.
    """


class RawLogPathExposedError(InfoBarrierViolationError):
    """A field name or value in the envelope is reachable back to a raw-log read path."""


class NonWhitelistedFieldError(InfoBarrierViolationError):
    """The envelope carries a field outside `SANCTIONED_FIELDS`."""


class PerSplitGranularityError(InfoBarrierViolationError):
    """The envelope exposes a per-split metric breakdown -- AC-9(b) forbids this."""


class FitnessDeltaExposureError(InfoBarrierViolationError):
    """A revealed `fitness_delta` exceeds the `DeltaExposurePolicy`'s precision bound."""


def _looks_like_raw_log_field(key: str) -> bool:
    return key.lower() in RAW_LOG_FIELD_NAMES


def _looks_like_per_split_field(key: str) -> bool:
    lowered = key.lower()
    return lowered in PER_SPLIT_FIELD_NAMES or lowered.startswith(PER_SPLIT_PREFIXES)


def _value_exposes_raw_log_path(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    lowered = value.lower()
    return any(marker in lowered for marker in _RAW_LOG_VALUE_MARKERS)


def lint_agent_envelope(
    envelope: Mapping[str, Any],
    *,
    delta_policy: "DeltaExposurePolicy | None" = None,
) -> None:
    """The composed AC-9 gate: raw-log, per-split, whitelist, then delta-precision checks.

    Checked in this order (worst violation first) so the raised exception names the most
    severe problem when an envelope trips more than one check at once.

    Args:
        envelope: the candidate agent-facing JSON envelope (already a plain dict/mapping --
            this function does not itself perform JSON schema validation of *shape*, only of
            *content*; a caller wanting shape validation constructs a `SanctionedEnvelope` via
            `build_sanctioned_envelope`, which can only produce well-shaped envelopes).
        delta_policy: if given, also enforces that `envelope["fitness_delta"]` (when present)
            carries no more precision than `delta_policy.precision` decimal places.

    Raises:
        RawLogPathExposedError: a field name or string value looks like a raw-log read path.
        PerSplitGranularityError: a field name looks like a per-split metric breakdown.
        NonWhitelistedFieldError: a field name is outside `SANCTIONED_FIELDS`.
        FitnessDeltaExposureError: `fitness_delta` exceeds `delta_policy`'s precision bound.
    """
    raw_log_field_hits = {k for k in envelope if _looks_like_raw_log_field(k)}
    raw_log_value_hits = {k for k, v in envelope.items() if _value_exposes_raw_log_path(v)}
    raw_log_hits = raw_log_field_hits | raw_log_value_hits
    if raw_log_hits:
        msg = f"envelope exposes a raw-log read path via field(s): {sorted(raw_log_hits)}"
        raise RawLogPathExposedError(msg)

    per_split_hits = {k for k in envelope if _looks_like_per_split_field(k)}
    if per_split_hits:
        msg = f"envelope exposes per-split granularity via field(s): {sorted(per_split_hits)}"
        raise PerSplitGranularityError(msg)

    unknown = set(envelope) - SANCTIONED_FIELDS
    if unknown:
        msg = f"envelope carries non-whitelisted field(s): {sorted(unknown)}"
        raise NonWhitelistedFieldError(msg)

    delta = envelope.get("fitness_delta")
    if delta is not None and delta_policy is not None:
        try:
            numeric_delta = float(delta)
        except (TypeError, ValueError) as exc:
            msg = f"fitness_delta={delta!r} is not numeric: {exc}"
            raise FitnessDeltaExposureError(msg) from exc
        rounded = round(numeric_delta, delta_policy.precision)
        if rounded != numeric_delta:
            msg = (
                f"fitness_delta={delta!r} exceeds the allowed precision "
                f"({delta_policy.precision} decimal places); rounded would be {rounded!r}"
            )
            raise FitnessDeltaExposureError(msg)


@dataclass(frozen=True, slots=True)
class DeltaExposurePolicy:
    """Rate/precision limits on cross-iteration fitness-delta exposure (AC-9(b), PM-5).

    `precision` bounds how many decimal places a revealed delta may carry. `reveal_every`
    bounds how often a numeric delta is revealed at all -- iterations that don't land on the
    boundary get only the coarse, non-numeric `fitness_trend` label, which carries no precision
    to reverse-engineer against.
    """

    precision: int = 2
    reveal_every: int = 3

    def __post_init__(self) -> None:
        if self.reveal_every < 1:
            msg = f"reveal_every must be >= 1 (got {self.reveal_every}) -- 0 would divide by zero"
            raise ValueError(msg)
        if self.precision < 0:
            msg = f"precision must be >= 0 (got {self.precision})"
            raise ValueError(msg)


DEFAULT_DELTA_POLICY = DeltaExposurePolicy()


@dataclass(frozen=True, slots=True)
class SanctionedEnvelope:
    """The agent-facing status/failure envelope -- the only shape an agent may ever see.

    Every field here is in `SANCTIONED_FIELDS`. `to_json_dict` drops `None` fields rather than
    emitting explicit nulls, keeping the envelope minimal. Constructing this dataclass directly
    (bypassing `build_sanctioned_envelope`) is possible but unlints itself -- callers should
    route through `build_sanctioned_envelope`, which always lints its own output before
    returning.
    """

    status: str
    run_id: str
    iteration: int
    error_taxonomy: str | None = None
    fitness_trend: str | None = None
    fitness_delta: float | None = None

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize to the sanctioned JSON dict shape (drops unset/None fields)."""
        return {k: v for k, v in asdict(self).items() if v is not None}


def build_sanctioned_envelope(
    record: MetricsProvenanceRecord,
    *,
    iteration: int,
    status: str = "ok",
    error_taxonomy: str | None = None,
    previous_score: float | None = None,
    score_key: str = "score",
    delta_policy: DeltaExposurePolicy = DEFAULT_DELTA_POLICY,
) -> SanctionedEnvelope:
    """Build and self-lint the sanctioned envelope for one iteration's agent-facing output.

    `previous_score` is the caller's own responsibility to track across iterations (this
    module holds no iteration state); passing `None` (e.g. iteration 0, or no prior record)
    omits both `fitness_trend` and `fitness_delta` from the result.

    `fitness_trend` (`"improved"/"regressed"/"unchanged"`) is always safe to reveal every
    iteration -- it carries no numeric precision. The raw numeric delta is rate/precision
    limited per `delta_policy`: rounded to `delta_policy.precision` places, and included only
    on iterations where `iteration % delta_policy.reveal_every == 0`.

    Raises:
        InfoBarrierViolationError: (via the final self-lint) if the constructed envelope
            somehow still violates AC-9 -- defense in depth; should be unreachable given this
            function's own construction logic, but never silently trusted.
    """
    trend: str | None = None
    delta: float | None = None
    if previous_score is not None:
        score = record.fitness[score_key]
        raw_delta = score - previous_score
        trend = "improved" if raw_delta > 0 else "regressed" if raw_delta < 0 else "unchanged"
        if iteration % delta_policy.reveal_every == 0:
            delta = round(raw_delta, delta_policy.precision)

    envelope = SanctionedEnvelope(
        status=status,
        run_id=record.run_id,
        iteration=iteration,
        error_taxonomy=error_taxonomy,
        fitness_trend=trend,
        fitness_delta=delta,
    )
    lint_agent_envelope(envelope.to_json_dict(), delta_policy=delta_policy)
    return envelope


__all__ = [
    "DEFAULT_DELTA_POLICY",
    "PER_SPLIT_FIELD_NAMES",
    "PER_SPLIT_PREFIXES",
    "RAW_LOG_FIELD_NAMES",
    "SANCTIONED_FIELDS",
    "DeltaExposurePolicy",
    "FitnessDeltaExposureError",
    "InfoBarrierViolationError",
    "NonWhitelistedFieldError",
    "PerSplitGranularityError",
    "RawLogPathExposedError",
    "SanctionedEnvelope",
    "build_sanctioned_envelope",
    "lint_agent_envelope",
]
