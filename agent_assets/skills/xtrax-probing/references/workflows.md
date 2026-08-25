# Probing Workflows Reference

Step-by-step procedures for each integration surface. Verify-paths cite the
owning module; when code and document disagree, the code wins.

## 1. Stage-0 / Stage-1 Probe Drivers

Drivers live in `scripts/` and are the ONLY sanctioned writers of
`outputs/profiling/stage<N>/` records. Label vocabulary (e.g.
`tiling_vmap`, `tiling_safemap`, `tiling_dedup_gather`) lives in the drivers,
never in the library package (D8).

- `scripts/prof_stage0_tiling_cost.py` -- cost analysis only, never executes.
- `scripts/prof_stage1_tiling_micro.py` -- one jitted program exercising
  Vmap/SafeMap/DedupGather under named scopes; two-input trace+HLO
  attribution; warm-up outside the measurement window; live self-check that
  TERM_RANKING over its own output fails closed.

Always export `XTRAX_GIT_SHA=$(git rev-parse HEAD)` so records stamp a clean,
resolvable sha instead of `-dirty`.

## 2. Benchmark Bridge (pytest-benchmark -> ProbeRecords)

Opt-in via `XTRAX_BENCH_RECORD_DIR`; without it the hook does nothing and
local runs never dirty the tree (`benchmarks/conftest.py::pytest_sessionfinish`).

Declaration protocol (declared-not-inferred, fail-closed) -- inside each bench
test, before `benchmark(...)`:

```python
benchmark.extra_info.update(
    {
        "xtrax_stage": 1,                # int 0..3; claimability checked at record time
        "xtrax_n_atoms": 32,             # int > 0
        "xtrax_scale_basis": "batch_rows",   # free-form xtrax_* config
    }
)
```

- Undeclared benches are NEVER recorded; they surface as skipped-with-reason
  lines in a terminal summary.
- Stats schema is pinned to the installed pytest-benchmark `Stats.fields`;
  durations convert s->ms with an `_ms` suffix, counts pass through, the
  display-string composite `outliers` ("iqr;stddev") is dropped, and any
  unknown field aborts loudly -- examine plugin upgrades before their numbers
  become citable.
- Node ids differing only in underscore runs normalize to one filename;
  duplicates are detected and refused instead of silently overwriting.

## 3. Performance Gate Tripwires

In `performance_targets.toml`, per probe: `max_compilations`,
`max_jit_traces` (dispatch tripwires from a real traced run) and
`emit_probe_record = true`. Ceilings must be positive ints -- TOML booleans
are rejected rather than becoming ceiling 1.

Behavior:

- The `performance.dispatch_violation_count` baseline ratchet activates only
  when some probe configures ceilings; legacy TOMLs behave byte-identically.
- A crashing dispatch probe becomes a counted major finding
  (`performance.dispatch_probe_error`), never an uncaught gate crash.
- An opt-in record whose probe was skipped warns on stderr instead of
  vanishing.
- One guarded callable is built once and reused for warm-up + traced call --
  re-wrapping per call recompiles inside the window and poisons
  n_compilations. Do not reintroduce `run_trace_gate()` here.

## 4. Controller Pass Provenance

`run_one_candidate_pass(..., probe_record_dir=...)` writes one timestamped
Stage-0 record per COMPLETED pass (including hard-blocked ones; verdicts ride
in config). Semantics pinned by review findings:

- `accepted` = run success AND both gate decisions honored/held -- gates
  passing while the run failed records accepted=false.
- Emission failures are contained: the pass result stands, the missing record
  is reported on stderr. A completed pass is never discarded for
  bookkeeping reasons.

## 5. Report Generation

`xtrax.profiling.report.discover_records` loads
`outputs/profiling/stage*/*.json` (or explicit paths). Unreadable files are
SKIPPED LOUDLY: each gets a stderr warning plus an INCOMPLETE-evidence-set
summary -- a silently dropped record could otherwise shrink a unanimity set
invisibly. Rendering is claim-gated through
`assert_claim_supported`; TERM_RANKING tables render scope rows with
attribution method and pct-of-total.

## ClaimValidityError Catalogue

| message fragment | cause | fix |
|---|---|---|
| `stage must be in {0,1,2,3}` / boolean | bad stage value incl. JSON true | pass a real int stage |
| `requires platform='gpu'` / `requires device_kind` | stage>=2 on CPU or missing kind | measure on GPU or lower the stage |
| `exclusive_seconds=... finite, non-negative` | NaN/inf/negative smuggled via scopes | fix the tracer capture; never widen |
| `n_occurrences ... integer >= 1` | zero/negative/float count | fix occurrence counting |
| `attribution_method/scopes disagree` | missing or ghost attribution entries | cover every measured label; drop unknowns |
| `DISPATCH_COUNT requires stage>=1` | cost-analysis record backing dispatch counts | run a stage-1 traced probe |
| `TERM_RANKING requires stage>=2` | CPU sources under a ranking claim | obtain GPU stage-2 records |
| `unverifiable git_sha` | unknown/dirty/unverified/empty sha | commit, export XTRAX_GIT_SHA, re-measure |
| `not unanimous on 'xla_flags'` (etc.) | environment drift across sources | unify flags/device/platform, re-measure |
| `extrapolation ratio ... exceeds SCALE_EXTRAPOLATION_LIMIT` | target_n_atoms too far above min source scale | narrow the claim or add a larger-scale source |
| `metrics[...] is not coercible / boolean / not finite` | non-float, bool, NaN/inf/overflow metric | fix the producer; metrics are citable floats only |
| `benchmark stats contain field(s) [...] not in the pinned schema` | pytest-benchmark upgrade changed fields | diff Stats.fields, extend bench.py deliberately |
| `collides with already-written` | two node ids normalize to one filename | rename params (avoid '_' vs '__' distinctions) |

## GPU trace attribution (open tuning item, first L40S dogfood 2026-08-25)

On real GPU runs of aminx's ConditionalDecode, executed trace events name
executor-level thunks (post-fusion), NOT HLO instructions -- so instruction-
keyed scope maps match nothing and records degrade to DISPATCH_COUNT+
STRUCTURAL grade even with healthy traces and correct Compiled.as_text()
HLO dumps. CPU traces join fine; the gap is GPU-executor event naming.
Candidate directions: sub-scope granularity below fusion boundaries,
non-fused wrapper ops around measured regions, or an executor-thunk ->
HLO-instruction mapping from the XLA runtime. Until then, treat GPU records
as dispatch/structural evidence only.
