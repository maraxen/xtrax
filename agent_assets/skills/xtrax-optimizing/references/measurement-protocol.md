# Measurement Protocol

Preregistration -> baseline -> single-tier candidate -> paired probe ->
claim-gated keep/revert. Every step cites its owning module (verify-paths;
code wins over this doc).

## 1. Preregister the hypothesis

Before any edit, write down: mechanism ("why would this be faster"), tier
(1/2/3), target metric, minimum claim class, and the probe driver that will
decide. Store it beside the record as config keys -- `emit_probe_record`'s
`config=` dict is string-valued identity (`src/xtrax/profiling/emitters.py`),
mirroring bathos sidecar pre-registration. A hypothesis that cannot name its
tier and probe is not ready to be tested.

## 2. Baseline first

Run the chosen driver on the unmodified code, same sha, same machine, and keep
the emitted record. The stage-1 drivers stamp git provenance automatically
(`src/xtrax/profiling/record.py::_capture_git_sha`); a dirty tree yields a
"-dirty" suffix which `xtrax.profiling.claims` rejects for TERM_RANKING /
END_TO_END sources -- commit your work before measuring anything you intend to
rank.

## 3. One tier per candidate

Never mix a Tier-1 tweak with a Tier-3 rewrite in one measured change; if both
move the number you cannot attribute either. Sequence them as separate
candidates with separate records.

## 4. Paired probes per tier

| Tier | Probe recipe | What invalidates the pair |
|---|---|---|
| 1 | Stage-1 micro over callback variants under named scopes (pattern: `scripts/prof_stage1_tiling_micro.py`) | warmup/compile inside the timed window; ordered callbacks left in the traced loop |
| 2 | Two steady-state walls: feed-starved vs overlapped iteration (`xtrax.engine.io.async_indexed_stream`, vary `buffer_size`) | comparing different buffer sizes without isolating queue depth; host jitter from tracing itself -- time untraced |
| 3 | Stage-0 cost probe FIRST (`scripts/prof_stage0_onehot_cost.py` pattern), then Stage-1 micro with parity gate | skipping stage-0 on a losing idea; parity metric missing when numerics could shift |

Paired records must agree on platform / device_kind / x64_enabled / xla_flags /
git_sha before comparison -- `xtrax.profiling.claims.paired_configs` enforces
exactly this unanimity set.

## 5. Claim-gated keep/revert

- Keep requires a record whose permitted claims cover what you intend to say.
  Check with `permitted_claims(record)`; assert set-backed verdicts through
  `assert_claim_supported` (`src/xtrax/profiling/claims.py`). Both raise
  `ClaimValidityError` fail-closed.
- Stage discipline: Stage-0 backs STRUCTURAL only; Stage-1 adds
  DISPATCH_COUNT; rankings need Stage-2+ GPU records (construction refuses
  stage>=2 without platform="gpu" + device_kind --
  `src/xtrax/profiling/record.py::__post_init__`).
- Both winner AND loser records persist under `outputs/profiling/`. Losers go
  to `outputs/profiling/rejected/` so reports stay readable while regressions
  stay auditable (`discover_records` accepts explicit paths,
  `src/xtrax/profiling/report.py`).

## Worked micro-example

The one-hot pair in `scripts/prof_stage1_onehot_micro.py` demonstrates the
full protocol at Tier 3: parity gated BEFORE measurement, combined-program
trace with disjoint named scopes, advisory untraced walls recorded alongside
attribution-backed scope numbers, dispatch counts in-metric, and the printed
claim ceiling showing exactly how far that record may be cited. Its stated
limitation (H2D transfer excluded) is part of the evidence, not a footnote:
Tier-2 conclusions require the Tier-2 driver, not extrapolation.
