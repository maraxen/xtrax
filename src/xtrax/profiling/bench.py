"""Bridge pytest-benchmark wall-clock stats into ProbeRecords.

Optional provenance for ``benchmarks/`` runs: by default a bench session
leaves only terminal output; when ``XTRAX_BENCH_RECORD_DIR`` is set (by the
caller -- typically CI), benchmarks/conftest.py emits one ProbeRecord per
benchmark into that directory. This module holds the pure conversion logic;
the conftest holds all pytest-session wiring, so nothing here imports
pytest or pytest_benchmark.

Design rules inherited from the upstream contract (see record.py and
.praxia/docs/specs/260824_upstream-profiling-probe-tooling-from-prolix.md):

  - **Declared, not inferred.** A wall-clock bench has no intrinsic stage
    or molecular scale, so each bench test must DECLARE its own via
    ``benchmark.extra_info["xtrax_stage"]`` and
    ``benchmark.extra_info["xtrax_n_atoms"]``. An undeclared bench is never
    silently recorded under guessed semantics -- parse_bench_extra_info
    raises with the reason, and the conftest reports it as a skipped-with-
    -reason bench in the terminal summary.

  - **Strict stats schema.** The accepted stat names are pinned to
    pytest-benchmark's Stats.fields as installed (pytest-benchmark >=4,
    fields verified against 5.x). An unknown key raises rather than being
    dropped: a plugin upgrade that changes the stats schema must be
    examined before its numbers flow into citable records. Duration-like
    fields (seconds internally) are converted to milliseconds with an
    explicit ``_ms`` suffix; count fields pass through unsuffixed.

  - **Vocabulary-free core.** Only the ``xtrax_`` extra_info namespace is
    interpreted here; every other extra_info key is ignored (it belongs to
    pytest-benchmark or the bench author). Per D8 of the scope doc, domain
    vocabulary lives at call sites: benches declare, this module ingests.
"""

import dataclasses

from xtrax.profiling.claims import ClaimValidityError
from xtrax.profiling.record import ProbeRecord

DECLARATION_STAGE_KEY = "xtrax_stage"
DECLARATION_N_ATOMS_KEY = "xtrax_n_atoms"
_DECLARATION_PREFIX = "xtrax_"

# pytest_benchmark.stats.Stats.fields as installed. Duration-valued entries
# are seconds internally; everything else is a count. Keep both sets in sync
# with an actual plugin upgrade, not speculatively.
_DURATION_STAT_FIELDS = frozenset(
    {
        "min",
        "max",
        "mean",
        "stddev",
        "median",
        "iqr",
        "q1",
        "q3",
        "ld15iqr",
        "hd15iqr",
        "total",
    }
)
_COUNT_STAT_FIELDS = frozenset({"rounds", "iqr_outliers", "stddev_outliers", "ops"})
# Display-only composite: pytest-benchmark's Stats.outliers is the STRING
# "iqr;stddev" rendered in its terminal table, not a number. Both components
# are already recorded above; the composite is neither coercible nor
# additive information, so it is deliberately dropped rather than smuggled
# into float-only metrics.
_NON_NUMERIC_STAT_FIELDS = frozenset({"outliers"})


def _capture_platform() -> str:
    """Platform string for the record ("cpu" | "gpu"), lazily importing jax."""
    import jax

    devices = jax.devices()
    platform = getattr(devices[0], "platform", None) if devices else None
    return str(platform) if platform else "cpu"


def sanitize_bench_fullname(fullname: str) -> str:
    """Node id -> filesystem-safe, deterministic probe_id stem.

    "benchmarks/bench_tiling.py::test_tiling_dispatch_overhead[vmap]" ->
    "bench_tiling.py_test_tiling_dispatch_overhead[vmap]". Brackets are kept:
    they distinguish parametrized benches from their base name, which is
    exactly the identity two records of one bench function must not lose.
    """
    stem = fullname
    if stem.startswith("benchmarks/"):
        stem = stem[len("benchmarks/") :]
    sanitized = []
    for ch in stem:
        if ch.isalnum() or ch in "._-[]":
            sanitized.append(ch)
        else:
            sanitized.append("_")
    out = "".join(sanitized)
    # Collapse the "::" double underscore produced above.
    while "__" in out:
        out = out.replace("__", "_")
    return out


def parse_bench_extra_info(
    extra_info: dict[str, object],
) -> tuple[int, int, dict[str, str]]:
    """Extract (stage, n_atoms, config) from a bench's extra_info dict.

    Required declarations: xtrax_stage (int in {0, 1, 2, 3}; whether the
    declared value is *claimable* on this machine is ProbeRecord's job --
    e.g. stage >= 2 on CPU fails construction) and the n_atoms key
    xtrax_n_atoms (int > 0). Every other ``xtrax_*`` key becomes a config
    entry (stringified); non-xtrax keys are ignored. Raises
    ClaimValidityError carrying the precise reason on any missing or
    malformed declaration.
    """
    if DECLARATION_STAGE_KEY not in extra_info:
        raise ClaimValidityError(
            f"bench declares no {DECLARATION_STAGE_KEY!r} in extra_info -- "
            "a wall-clock bench has no intrinsic stage, so it must declare "
            "one to be recorded"
        )
    if DECLARATION_N_ATOMS_KEY not in extra_info:
        raise ClaimValidityError(
            f"bench declares no {DECLARATION_N_ATOMS_KEY!r} in extra_info -- "
            "a wall-clock bench has no intrinsic molecular scale, so it "
            "must declare one to be recorded"
        )
    raw_stage = extra_info[DECLARATION_STAGE_KEY]
    if isinstance(raw_stage, bool) or not isinstance(raw_stage, (str, int)):
        raise ClaimValidityError(
            f"{DECLARATION_STAGE_KEY}={raw_stage!r} must be an int or an "
            "int-valued string (bools rejected: true/false would silently "
            "become stage 1/0)"
        )
    try:
        stage = int(raw_stage)
    except (TypeError, ValueError) as exc:
        raise ClaimValidityError(
            f"{DECLARATION_STAGE_KEY}={raw_stage!r} is not coercible to int"
        ) from exc
    raw_n = extra_info[DECLARATION_N_ATOMS_KEY]
    if isinstance(raw_n, bool) or not isinstance(raw_n, (str, int)):
        raise ClaimValidityError(
            f"{DECLARATION_N_ATOMS_KEY}={raw_n!r} must be an int or an "
            "int-valued string (bools rejected: true/false would silently "
            "become n_atoms 1/0)"
        )
    try:
        n_atoms = int(raw_n)
    except (TypeError, ValueError) as exc:
        raise ClaimValidityError(
            f"{DECLARATION_N_ATOMS_KEY}={raw_n!r} is not coercible to int"
        ) from exc

    config: dict[str, str] = {}
    for key, value in extra_info.items():
        if key.startswith(_DECLARATION_PREFIX) and key not in (
            DECLARATION_STAGE_KEY,
            DECLARATION_N_ATOMS_KEY,
        ):
            config[key[len(_DECLARATION_PREFIX) :]] = str(value)
    return stage, n_atoms, config


def bench_metrics_from_stats(
    stats_dict: dict[str, float | int | str],
) -> dict[str, float | int | str]:
    """Stats.as_dict() -> contract metrics: durations s->ms, counts through.

    Unknown stat keys raise ClaimValidityError (strict schema, see module
    docstring). The known non-numeric display field (``outliers``) is
    dropped -- see _NON_NUMERIC_STAT_FIELDS. Values arrive already numeric
    from Stats.as_dict(); any NaN or inf survives to
    ProbeRecord.__post_init__, which rejects it there -- a diverged
    benchmark run is not a citable metric.
    """
    unknown = sorted(
        set(stats_dict) - _DURATION_STAT_FIELDS - _COUNT_STAT_FIELDS - _NON_NUMERIC_STAT_FIELDS
    )
    if unknown:
        raise ClaimValidityError(
            f"benchmark stats contain field(s) {unknown} not in the pinned "
            "pytest-benchmark schema -- examine the plugin upgrade before "
            "recording its numbers, then extend bench.py deliberately"
        )
    metrics: dict[str, float | int | str] = {}
    for name, value in stats_dict.items():
        if name in _NON_NUMERIC_STAT_FIELDS:
            continue
        if isinstance(value, bool):
            raise ClaimValidityError(
                f"benchmark stat {name!r}={value!r} is boolean -- a flag is "
                "not a citable timing metric"
            )
        try:
            float(value)
        except (TypeError, ValueError) as exc:
            raise ClaimValidityError(
                f"benchmark stat {name!r}={value!r} is neither numeric nor a "
                "known non-numeric display field -- refusing to record"
            ) from exc
        if name in _DURATION_STAT_FIELDS:
            metrics[f"{name}_ms"] = float(value) * 1000.0
        else:
            metrics[name] = float(value)
    return metrics


@dataclasses.dataclass(frozen=True, slots=True)
class BenchRecordPlan:
    """Everything needed to write one bench record, pre-validation.

    Splitting plan from write keeps the conftest dumb: it extracts plain
    attributes off the fixture, calls build_bench_record_plan, writes
    plan.probe_id + ".json", and reports plan-or-reason either way. No
    pytest types cross this boundary.
    """

    probe_id: str
    stage: int
    n_atoms: int
    platform: str
    metrics: dict[str, float | int | str]
    config: dict[str, str]


def build_bench_record_plan(
    *,
    fullname: str,
    params: dict[str, object] | None,
    extra_info: dict[str, object],
    stats_dict: dict[str, float | int | str],
) -> BenchRecordPlan:
    """Assemble one bench's ProbeRecord inputs; raises on any violation.

    Config precedence: params override extra_info-declared config on key
    collision -- the parametrize id is the bench's primary identity axis
    (e.g. which tiling strategy ran), and a record whose config disagreed
    with its own probe_id suffix would be self-laundering provenance.
    """
    stage, n_atoms, extra_config = parse_bench_extra_info(extra_info)
    config = dict(extra_config)
    for key, value in (params or {}).items():
        config[str(key)] = str(value)
    return BenchRecordPlan(
        probe_id=sanitize_bench_fullname(fullname),
        stage=stage,
        n_atoms=n_atoms,
        platform=_capture_platform(),
        metrics=bench_metrics_from_stats(stats_dict),
        config=config,
    )


def check_probe_id_collision(probe_id: str, seen: dict[str, str]) -> str | None:
    """Return the colliding fullname if probe_id was already claimed.

    sanitize_bench_fullname collapses underscore runs, so distinct node ids
    differing only in '_' runs (e.g. params "a_b" vs "a__b") normalize to
    the SAME filename -- writing both would silently overwrite one record
    with the other. The emission hook claims names through this helper and
    treats a collision as skip-with-reason instead of an overwrite.
    """
    prior = seen.get(probe_id)
    return prior


def record_from_plan(plan: BenchRecordPlan) -> ProbeRecord:
    """Construct the validated ProbeRecord for a plan.

    Separate from build_bench_record_plan so tests can pin validation
    failures (stage>=2 on CPU, non-finite stats) without a filesystem.
    """
    return ProbeRecord(
        probe_id=plan.probe_id,
        stage=plan.stage,
        n_atoms=plan.n_atoms,
        platform=plan.platform,
        metrics=plan.metrics,
        scopes=None,
        attribution_method=None,
        config=plan.config,
    )
