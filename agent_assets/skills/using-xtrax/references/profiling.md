> Part of the `using-xtrax` skill (`agent_assets/skills/using-xtrax/SKILL.md`) — TIER-2 deep reference.

# Profiling: ProbeRecords and the claim-validity contract (`xtrax.profiling`)

Upstreamed from prolix `scripts/profiling` on 2026-08-24 (branch
`wt-20260807-132628`); scope + phase status in
`.praxia/docs/specs/260824_upstream-profiling-probe-tooling-from-prolix.md`.

## What it is (and what it is not)

`xtrax.profiling` produces **ProbeRecords**: durable, provenance-stamped
profiling measurement artifacts evaluated by a machine-checked claim-validity
contract. This is NOT the "probe" you may already know from the CI gates —
gate probes (`xtrax.devtools.gates._trace_probe`, chex `assert_max_traces`)
are ephemeral pass/fail checks; a ProbeRecord is evidence that outlives the
run and can be cited in reports only as far as its stage and provenance
allow. Verify: `src/xtrax/profiling/__init__.py`

```python
from xtrax.profiling import (  # verify: src/xtrax/profiling/__init__.py __all__
    CONTRACT_VERSION,          # "3.0" — MAJOR bump rule lives in claims.py
    ClaimClass,                # STRUCTURAL / DISPATCH_COUNT / TERM_RANKING / END_TO_END
    ClaimValidityError,
    ProbeRecord,
    assert_claim_supported,
    paired_configs,
    permitted_claims,
    select_sources,
)
```

## The one rule to internalize

**A record's fields decide what it may back; the contract enforces it;
you never weaken a claim to make data fit.**

| Claim class | Requires | Typical source |
|---|---|---|
| `STRUCTURAL` | any record | Stage 0 (`cost_analysis`, never executes) |
| `DISPATCH_COUNT` | metrics `{n_executions, n_compilations, n_jit_traces}` | Stage 1 CPU micro under `jax.profiler.trace` |
| `TERM_RANKING` | ≥2 attributed scopes AND **stage≥2 GPU** unanimity | Stage 2 GPU sweep |
| `END_TO_END` | `total_step_seconds` + declared `target_n_atoms` within 10× min source scale | Stage 2 |

Fail-closed everywhere: `assert_claim_supported` raises rather than returns a
verdict; `select_sources` raises naming WHICH filter emptied the candidate
set ("add this metric" is the wrong fix when scope attribution is missing);
unanimity on `x64_enabled/xla_flags/device_kind/platform/git_sha` is required
across all sources of a ranking/end-to-end claim; `"unknown"`/`"-dirty"`/`"-unverified"`
git shas are rejected outright. Verify: `src/xtrax/profiling/claims.py`

## Emitting a record

```python
from xtrax.profiling.emitters import emit_probe_record  # verify: src/xtrax/profiling/emitters.py

emit_probe_record(
    path="outputs/profiling/stage1/my_probe.json",
    probe_id="stage1_my_probe",
    stage=1, n_atoms=256, platform="cpu",
    metrics={"n_executions": 20, "n_compilations": 0, "n_jit_traces": 40},
    scopes={"my_label": None},                    # None value = expected but absent, NEVER 0.0
    attribution_method={"my_label": "named_scope"},
    config={"kernel": "..."},                     # non-numeric identity axes
)
```

Provenance (git_sha/timestamp/jax versions/XLA_FLAGS/device_kind) is
auto-captured at construction; override per-kwarg only for synthetic test
fixtures. Cluster scratch without `.git`: set `XTRAX_GIT_SHA` before the run
(or write a repo-root `.git_sha` file), then `restamp_git_sha` for chain-of-
custody. **Empty-attribution trap:** when a trace was captured but every
label came back absent, pass an EMPTY dict for `attribution_method` (use
`attribution_from_scopes`) — `or None` there broke a real prolix cluster run.

## Existing xtrax probes (reuse these drivers)

```bash
# Stage 0: XLA cost analysis over Vmap/SafeMap/DedupGather (never executes)
XTRAX_GIT_SHA=$(git rev-parse HEAD) uv run python scripts/prof_stage0_tiling_cost.py
# Stage 1: CPU micro, trace+HLO two-input attribution over the same strategies
XTRAX_GIT_SHA=$(git rev-parse HEAD) uv run python scripts/prof_stage1_tiling_micro.py
```

Records land in `outputs/profiling/stage<N>/` (raw traces are gitignored).
The stage-1 driver self-checks that `TERM_RANKING` over its own output fails
closed. Scope-label vocabulary (`tiling_vmap`, `tiling_safemap`,
`tiling_dedup_gather`) lives in the driver, never in the library package.

## Gate + controller integration (opt-in, zero behavior change by default)

Performance gate: per-probe `max_compilations` / `max_jit_traces` ceilings
(dispatch tripwires from a real traced run) and `emit_probe_record = true`
(durable gate artifact). The `performance.dispatch_violation_count` baseline
ratchet activates only when some probe configures ceilings. Verify:
`src/xtrax/devtools/gates/performance.py`,
`src/xtrax/devtools/gates/_dispatch_probe.py`

Controller: `run_one_candidate_pass(..., probe_record_dir=...)` writes one
Stage-0 provenance record per completed pass (campaign id, lineage sha, wall
seconds, gate verdicts) — the only bathos-adjacent site allowed to do so.
Verify: `controller/main_loop.py::_emit_candidate_pass_probe_record`

## JAX-version fragility (read before trusting new traces)

trace.py's event-name expectations were spike-verified on jax 0.10.2 (see D9
result in the scope doc): executed events carry post-fusion thunk names in
`args["hlo_op"]`; named_scope paths survive ONLY in compiled HLO text
(`op_name="jit(f)/<scope>/<primitive>"`); `backend_compile_and_load` appears
only if compilation happens inside the window; `PjitFunction(<fn>)` counts 2×
per Python call. Re-spike presence-not-spelling after ANY jax upgrade.
Verify: `src/xtrax/profiling/trace.py` module docstring
