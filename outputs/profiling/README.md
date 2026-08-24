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
```

Current contents are Stage 0/1 (CPU-only jaxlib on this machine): they
support STRUCTURAL and DISPATCH_COUNT claims. A TERM_RANKING over them fails
closed by design (`xtrax.profiling.claims.assert_claim_supported`) until
GPU-measured Stage-2 records exist -- verified live by the stage-1 driver's
self-check.
