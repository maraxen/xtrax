# outputs/profiling/

ProbeRecord artifacts from xtrax-native probe drivers (Phase B of
`.praxia/docs/specs/260824_upstream-profiling-probe-tooling-from-prolix.md`).
One JSON file per record under `stage<N>/`, generated ONLY via
`xtrax.profiling.emitters.emit_probe_record`; `*_traces/` directories hold
raw Perfetto exports and are gitignored (prolix convention: traces stay
off-repo; records + HLO text are the durable artifacts).

Regeneration (records stamp git_sha via XTRAX_GIT_SHA so provenance points
at the producing commit):

```bash
XTRAX_GIT_SHA=$(git rev-parse HEAD) \
    uv run python scripts/prof_stage0_tiling_cost.py
XTRAX_GIT_SHA=$(git rev-parse HEAD) \
    uv run python scripts/prof_stage1_tiling_micro.py
# One-hot encoding exemplar (P1 of 260825_jax-optimizing-skill-scope.md):
XTRAX_GIT_SHA=$(git rev-parse HEAD) \
    uv run python scripts/prof_stage0_onehot_cost.py
XTRAX_GIT_SHA=$(git rev-parse HEAD) \
    uv run python scripts/prof_stage1_onehot_micro.py
```

Current contents are Stage 0/1 (CPU-only jaxlib on this machine): they
support STRUCTURAL and DISPATCH_COUNT claims. A TERM_RANKING over them fails
closed by design (`xtrax.profiling.claims.assert_claim_supported`) until
GPU-measured Stage-2 records exist -- verified live by the stage-1 driver's
self-check.

## Benchmark runs (opt-in)

`benchmarks/` sessions can persist one ProbeRecord per benchmark via the
declaration protocol in `src/xtrax/profiling/bench.py`: each bench declares
`xtrax_stage` / `xtrax_n_atoms` (+ free-form `xtrax_*` config) through
`benchmark.extra_info`, and nothing is recorded without a declaration. Off
by default; emission requires:

```bash
XTRAX_GIT_SHA=$(git rev-parse HEAD) \
XTRAX_BENCH_RECORD_DIR=outputs/profiling/stage1 \
    uv run pytest benchmarks --benchmark-only
```

Undeclared or empty-stats benches are reported skipped-with-reason in a
terminal summary; records land as `<probe_id>.json` named from the node id.

