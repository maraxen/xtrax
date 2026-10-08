---
name: xtrax-probing
description: "Use when a JAX timing, dispatch count, or compile count has to stay citable, when a benchmark result is being quoted past the scale that was measured, or when a performance gate needs a dispatch-count ceiling. Covers stored probe records, claim-validity checks, and bottleneck reports."
xtrax_version: 0.4.0a12
triggers:
  - a timing or dispatch count that a later claim will cite
  - benchmark result quoted beyond the scale that was measured
  - performance gate needs a dispatch-count or compile-count ceiling
  - bottleneck report from stored measurements
  - persist a benchmark so the number stays attachable to a claim
---

# xtrax-probing

## Purpose

Package every JAX measurement made in xtrax into a `ProbeRecord`: a frozen,
stage- and scale-stamped artifact whose fields state what the measurement may
be cited for. The claim-validity contract (`xtrax.profiling.claims`) turns
sets of records into verdicts -- fail-closed: an unsupported claim raises,
never silently passes.

Verify-paths (house convention): every rule below cites the module that owns
it. When this skill and the code disagree, the code wins -- then update this
skill.

## Non-Negotiables

1. Never hand-write record JSON. Construct via `ProbeRecord(...)` or
   `emit_probe_record(...)`; validation runs at construction and invalid
   artifacts never reach disk.
2. Never widen a guard to make a claim pass. Narrow the claim instead.
3. No first-party imports inside `xtrax.profiling` (leaf package, AST-
   enforced); jax is imported lazily only inside provenance factories.
4. New modules under `src/xtrax/profiling/` must not use future-annotations.

## Quick Start

Stage-0 cost probe (no execution) and stage-1 micro-execution probe over the
tiling strategies; records land in `outputs/profiling/stage<N>/`, raw traces
stay gitignored:

```bash
XTRAX_GIT_SHA=$(git rev-parse HEAD) \
    uv run python scripts/prof_stage0_tiling_cost.py
XTRAX_GIT_SHA=$(git rev-parse HEAD) \
    uv run python scripts/prof_stage1_tiling_micro.py
```

Benchmark wall-clock persistence (opt-in; benches DECLARE their own
`xtrax_stage`/`xtrax_n_atoms` via `benchmark.extra_info`, undeclared benches
are skipped-with-reason):

```bash
XTRAX_GIT_SHA=$(git rev-parse HEAD) \
XTRAX_BENCH_RECORD_DIR=outputs/profiling/stage1 \
    uv run pytest benchmarks --benchmark-only
```

Contract + fixture audit:

```bash
just audit-profiling-contract
uv run python scripts/audit_profiling_contract.py
```

Claim-gated bottleneck report over `outputs/profiling/stage*/*.json`:
`scripts/prof_report.py` style entry via `xtrax.profiling.report`.

## Stage / Claim Decision Table

| stage | platform | STRUCTURAL | DISPATCH_COUNT | TERM_RANKING | END_TO_END |
|---|---|---|---|---|---|
| 0 cost-analysis | any | yes | no | no | no |
| 1 micro-exec | cpu ok | yes | yes | no | no |
| 2+ device-measured | gpu required (+device_kind) | yes | yes | set-based | set-based |

TERM_RANKING additionally requires >=2 attributed scopes per source carrying
`total_step_seconds`, unanimity on x64_enabled/xla_flags/device_kind/
platform/git_sha, verifiable (clean) shas. END_TO_END requires a positive
target_n_atoms within SCALE_EXTRAPOLATION_LIMIT of min(source n_atoms).
Single-record checks go through `permitted_claims`; set-backed claims through
`assert_claim_supported` -- both raise `ClaimValidityError` on violation.

## Integration Surfaces (all opt-in, zero behavior change by default)

- **Performance gate**: per-probe `max_compilations` / `max_jit_traces`
  ceilings and `emit_probe_record = true`; the
  `performance.dispatch_violation_count` baseline ratchet activates only when
  some probe configures ceilings. A crashing probe becomes a counted major
  finding, never a gate crash.
- **Controller**: `run_one_candidate_pass(..., probe_record_dir=...)` writes
  one timestamped Stage-0 provenance record per completed pass; emission
  failures are contained and reported without discarding the pass.
- **pytest-benchmark**: see the bench protocol above; strict stats schema,
  collision refusal, loud skips.

## Additional Resources

Load as needed -- do not read both up front:

- **`references/contract.md`** -- full ProbeRecord field/guard reference,
  claim rules with verify-paths, contract-versioning bump rule.
- **`references/workflows.md`** -- step-by-step workflows (drivers, bench
  declarations, gate/controller wiring), plus the ClaimValidityError
  catalogue: message -> cause -> fix.

Acting on a measurement -- deciding what to optimize and what evidence each
class of change owes -- is the `xtrax-optimizing` skill's job (three-tier
taxonomy + preregister-and-probe protocol; its drivers emit the records this
skill governs).
