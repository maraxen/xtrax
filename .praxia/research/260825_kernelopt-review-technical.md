# Adversarial Technical Review: 260825_kernelopt-spec-proposals.md

Date: 2026-08-25 | Branch: feat/profiling-stage2-evidence
Reviewer role: adversarial (technical soundness). Target: the 20-item proposal list in
`.praxia/research/260825_kernelopt-spec-proposals.md` against its evidence base
(`260825_kernelopt-grounding-report.md`) and delivered scope
(`.praxia/docs/specs/260825_jax-optimizing-skill-scope.md`).
Every claim below was verified against actual code on this branch; file:line cited throughout.
Severity scale: FATAL / MAJOR / MINOR. Nothing dies wholesale; one sub-component dies as proposed.

---

## HEADLINE FINDINGS

### FATAL-TO-SUBCOMPONENT #1 — B1(b): self-declared clock provenance contradicts the contract's own anti-laundering design; the proposals never address it

record.py's class docstring (`src/xtrax/profiling/record.py:164-172`) states verbatim:
"A provenance field a caller can forget is a provenance field that will be forgotten --
device_kind auto-capture in particular closes a laundering hole: a freeform caller-set string
could otherwise falsely agree (or disagree) with another source's value under the unanimity guard."

B1 proposes `clock_locked` / `gpu_clock_mhz` as DRIVER-DECLARED config strings -- exactly the
"freeform caller-set string" the design calls a laundering hole. B1 names only the
pynvml-dependency benefit and never engages the documented philosophy (attack vector c confirmed).
Two failure modes depending on where enforcement lands:

1. If these keys ever join claims.py `_UNANIMITY_FIELDS` (`src/xtrax/profiling/claims.py:43`)
   or `paired_configs` hold_fixed keys, two drivers that both lie `clock_locked="true"` trivially
   agree -- the precise false-agreement hole device_kind auto-capture closed.
2. If they stay out of all gates, nothing enforces them for TERM_RANKING; D2's "minimum credible
   protocol" rests on honor-system strings while citing MARLIN's finding that unlocked clocks
   distort rankings.

B1/D1 never state WHERE the n_runs/dispersion/clock gate lives (REQUIRED_METRICS? unanimity set?
driver-side?). This is the load-bearing unspecified decision of the whole list. B1(b) survives only
as explicitly advisory metadata with a written trust model, or with harness-side (non-leaf)
auto-capture cross-checking nvidia-smi.

### MAJOR #2 — A1/B3 premise failure: xtrax's flagship scan has autodiff INSIDE the body (where scan-body checkpointing does not apply), and the function has zero production callers

A1 cites f7 correctly as literature but justifies "single highest-value documentation addition"
via "xtrax pipelines are scan-heavy (`accumulate_grads` microbatches via lax.scan)". Verified
`src/xtrax/training/grad.py:80-94`: `scan_fn` computes `eqx.filter_value_and_grad(loss_fn)` INSIDE
the body, carry=None, outputs=(grads, loss). The textbook scan-remat problem targets
grad-OUTSIDE-scan (per-step activations must persist across the scan for backward); here each step
closes its own backward, intermediates die within the step, and what crosses steps is stacked
gradient OUTPUTS -- an output-materialization cost remat cannot fix.

Worse: grep shows `accumulate_grads` has no callers anywhere in src/ or scripts/ -- it is an
unwired utility. The other scan path, `execute_scan_axis` (`src/xtrax/stages/executor.py:220-248`),
serves boundary/tap-sink streaming and its only driver caller passes an identity lambda
(`scripts/prof_stage1_host_boundary.py:75`). "Scan-heavy pipelines" is currently aspirational, not
descriptive. B3 inherits this: its driver pair would measure a regime (grad-outside-scan) no
current xtrax code path inhabits unless deliberately constructed to.

### MAJOR #3 — A3's evidence instrument is mis-cited against trace.py's own findings

A3 requires a "fused-thunk-count delta" claiming "stage-0 cost_analysis() + named-scope
attribution already produce the ingredients". But `src/xtrax/profiling/trace.py:11-25` documents
empirically that post-fusion traces name events after their POST-FUSION thunk/HLO op name and
"name no named_scope at all" -- named_scope labels survive only via `scope_map_from_hlo_text`
reconstruction. So named_scope attribution cannot directly count fused thunks; what CAN work is
counting executed-trace thunk events themselves (they are post-fusion entities by construction),
which trace.py supports. And `scripts/prof_stage0_onehot_cost.py:61,94` emits normalized numeric
cost fields from cost_analysis() -- no demonstrated thunk/fusion counter exists there. Direction
sound, cited mechanism wrong; the rule as written mandates evidence whose instrument doesn't exist yet.

### MAJOR #4 — B1(c)/D1 ignore the contract's own bump mechanics and re-spike rule

Adding wall_median_seconds / wall_dispersion_seconds / n_runs to
`REQUIRED_METRICS[ClaimClass.TERM_RANKING]` (`src/xtrax/profiling/claims.py:89-94`) newly REJECTS
previously-passing claims. `claims.py:84-88` explicitly warns: "re-spike before adding a required
metric, never after records exist" -- written after CONTRACT_VERSION went 2.0 -> 3.0 for exactly
this class of change. D1 asks the retroactivity question but never mentions CONTRACT_VERSION or
the re-spike house rule its answer would trigger. The house-rule audit section (proposals:114-116)
therefore misses its own most relevant guard.

### MAJOR #5 — B2 backend-cost-model mismatch glossed

B2 claims intensity-vs-L40S-ceiling comparison at "ZERO runtime cost" by extending the stage-0
pattern. Stage-0 lowers+compiles for the LOCAL backend (`prof_stage0_onehot_cost.py:94`); this
machine is CPU-only jaxlib (scope doc section 7b). XLA CPU cost-model numbers vs L40S published
ceilings mix compiler cost models. "Zero runtime cost" holds only if intensity is derived
analytically from HLO op counts (backend-independent arithmetic) or collected ON the L40S during
the re-run -- neither stated. Also "the dogfood run" supplying the first ceilings row is referenced
nowhere in the grounding report or scope doc; unverifiable from the listed artifacts.

---

## ATTACK VECTOR (a): CITATION INTEGRITY (9 proposals spot-checked)

| Proposal | Cites | Verdict |
|---|---|---|
| A2(i)(ii)(iii) | d5/d6/d7 | FAITHFUL. MARLIN 4-specialized-kernels+roofline pairing, vLLM 20-26% slower micro / 2-4x macro, FA-3 Table-2 ablations match report sections d5-d7 in substance. |
| A4 | f6 | FAITHFUL (vLLM crossover-curves-not-verdicts framing). |
| A5 | f8 | FAITHFUL (invalidation, keyword-args-never-donated, PyTree sweep, silent-drop warning). |
| A6 | f13 | FAITHFUL (Micikevicius FP16 recipe; E4M3/E5M2 split). |
| A7 | f11/d8 | FAITHFUL (ADEPT regime shifts; measured-edge bucketing). |
| A1 | f7 | Literature faithful; APPLICATION overreaches (see MAJOR #2). |
| B1 | f3/d1/d3 | Citations faithful BUT the proposed n_runs>=3 floor is weaker than every protocol it cites (Ansor n=5, FA n=10, FA-3 n=100) with zero justification; dispersion with n=3 is statistically fragile. |
| A8 | f15 | GDS is flagged single-query coverage (q08 only) in the coverage ledger (report line 122) and A8 carries that caveat nowhere; also the motivating ~1040-1450x constant is OUR OWN measurement (`agent_assets/skills/xtrax-optimizing/references/tier1-host-boundary.md:10-11`), NOT corpus material -- fine to cite, but the text implies literature grounding. |
| C2 | d1/d3 + SKILL.md | Overreach: "uncontrolled-clock CPU micro benchmarks cannot back directional claims AT ALL" exceeds both citations and the contract, which permits CPU STRUCTURAL/DISPATCH_COUNT directional-mechanism claims (`claims.py:97-108`). |

## ATTACK VECTOR (b): VERIFY-PATH HONESTY (10 paths read)

VERIFIED TRUE:
- metrics annotation `dict[str, float | int | str]` (record.py:196) admits flat keys;
  `__post_init__` coercion (record.py:226-247) means B1(a)'s "NO schema surgery" is genuinely
  correct -- and non-finite dispersion is already rejected at construction (record.py:241-245),
  a synergy B1 doesn't even claim credit for.
- config `dict[str, str]` (record.py:202; emitters.py:53) matches "string-valued identity".
- `sparse_filter_jit` EXISTS at `src/xtrax/sparse/inference.py:146-178`, donate kwargs ->
  eqx.filter_jit. A5's surface claim holds.
- `make_axis_dispatch` rejects Scan on heterogeneous axes (`src/xtrax/tiling/dispatch.py:86-93`).
- `--feed-sleep-ms` / `--buffer-size` knobs exist (`scripts/prof_stage1_feed_overlap.py:103-104`);
  A4/B4's ladder needs no new plumbing.
- `PARITY_TOLERANCE` per-driver confirmed (`scripts/prof_stage1_onehot_micro.py:68`); D5's premise true.
- ZarrStagingSink (`src/xtrax/run/zarr_sink.py:124`), async_indexed_stream
  (`src/xtrax/engine/io.py:17`), `src/xtrax/tiling/bucket.py`,
  stage>=2 platform/device_kind enforcement (record.py:218-225),
  tests/scripts/test_prof_optimizing_drivers.py (11 tests) -- all real.
- The house-audit verify-path list (proposals:116) contains no fabricated paths.

OVERSTATED:
- make_axis_dispatch as B3's risk verify-path is decorative -- a synthetic probe driver constructs
  its own scan and bypasses strategy dispatch entirely; the real static-carry constraint is
  JAX-inherent, not dispatch-enforced.
- A5 codifying sparse_filter_jit as "the single audited surface... policy, not accident" omits
  that inference.py:159-160 itself recommends preferring the make_sparse_forward_fn closure
  pattern "for maximum safety".

## ATTACK VECTOR (c): TECHNICAL CORRECTNESS OF PROPOSED MECHANISMS

Beyond FATAL #1:
1. gpu_clock_mhz "per run" (D2) collides with one-config-snapshot-per-record: N runs need N
   records or a declared median-of-runs convention; unstated.
2. device_kind auto-captures devices[0] only (record.py:145-148); multi-GPU rigs make per-record
   clock identity ambiguous. Unaddressed anywhere in the list.
3. wall_dispersion_seconds statistic undefined (std? MAD? range?) -- records become incomparable
   across drivers/later rounds.

## ATTACK VECTOR (d): FEASIBILITY -- B3 SCAN-REMAT DRIVER PAIR

Mechanically FEASIBLE but narrower than pitched. `execute_scan_axis` wraps the user fn inside
`_wrapped_transition` (executor.py:238-247) then calls safe_scan -> jax.lax.scan
(`src/xtrax/transforms/scan.py:47`). A driver CAN pass `jax.checkpoint(fn)` as fn -- it composes
through the closure, and boundary tap/sink fire OUTSIDE the checkpointed region, cleanly isolating
callbacks from remat. However:
1. safe_scan and JaxScanIterator (`src/xtrax/tiling/iterator.py:154-173`) expose NO remat-policy
   passthrough, so pipeline-level adoption later requires API surgery the proposals don't scope.
2. Name-based policies (`save_only_these_names`, offload variants) act on differentiated-argument
   names; plumbing them through eqx-pytree closures is real work -- effort L, not M/L.
3. The exemplar must be explicitly constructed grad-outside-scan or it measures nothing relevant
   to xtrax (MAJOR #2).
4. Parity gate across remat/offload variants is trivially satisfied (all are semantics-preserving),
   so parity there signals nothing -- fine, but worth saying so preregistration isn't padded.

## ATTACK VECTOR (e): MISSING ATTACKS THE LIST SHOULD HAVE ANTICIPATED

1. Enforcement-location decision for every new gate B1/D1 propose (the FATAL #1 question).
2. Contract-version bump + re-spike mechanics (MAJOR #4).
3. Multi-GPU device/clock ambiguity.
4. Justification gap for n>=3 vs cited n=5/10/100 protocols.
5. B3-as-measured-regime vs pipelines-as-shipped gap (MAJOR #2).
6. C2's demotion silently applies to the skill's own headline 0.70x negative result -- itself a
   directional CPU claim; prospective phrasing protects old citations rhetorically, but the new
   policy reclassifies the flagship artifact without acknowledging it.

MINOR issues (consolidated): A5 lacks any concrete method for "confirm donation actually took
effect"; A8 citation hygiene (single-query caveat + own-artifact provenance); B1 dispersion
statistic unspecified; C1/A6 consistency wrinkle (both are GPU-future speculations; one documented,
one cut -- defensible via parity-machinery-readiness, but the standard should be stated).

---

## WHAT SURVIVES THE ATTACK

The list is largely honest: every cited code path exists, most literature citations land where
claimed, both cuts (C1/C3) are well-supported (pinned-memory truly has zero corpus support;
Foldcomp never cited per the coverage ledger), and the gating-first sequencing (variance fields
BEFORE any GPU re-run) is correct and defensible. No item dies wholesale.

## VERDICT TABLE

| Item | Verdict |
|---|---|
| A1 scan-remat recipe | SURVIVES-WITH-CHANGES (add explicit branch: AD-inside-body scans like accumulate_grads are NOT helped by body-checkpointing) |
| A2 measurement-protocol bundle | SURVIVES (cleanest item; citations exact) |
| A3 opaque-kernel cliff rule | SURVIVES-WITH-CHANGES (recite instrument honestly: post-fusion thunk-event counts from executed traces per trace.py:11-25; cost_analysis alone provides no thunk count) |
| A4 async-overlap crossover | SURVIVES |
| A5 donation box | SURVIVES-WITH-CHANGES (acknowledge closure-pattern preference at inference.py:159-160; specify donation-effect verification method) |
| A6 mixed-precision recipe | SURVIVES-WITH-CHANGES (its own misread-guard is mandatory, not optional) |
| A7 regime-shift buckets | SURVIVES |
| A8 dual-path costing | SURVIVES-WITH-CHANGES (citation hygiene: GDS single-query caveat + own-artifact provenance of 1040-1450x) |
| B1 variance+clock fields | SURVIVES-WITH-CHANGES overall; sub-item (b) DIES AS PROPOSED if clock keys are intended for any gate/unanimity role -- survives only as advisory metadata with a written trust model or harness-side auto-capture; (c) needs bump mechanics, re-spike treatment, n-floor justification |
| B2 intensity emitter + ceilings | SURVIVES-WITH-CHANGES (fix backend mismatch: analytic HLO-derived intensity or collect cost_analysis on-GPU; drop "zero runtime cost" otherwise) |
| B3 scan-remat driver pair | SURVIVES-WITH-CHANGES (reframe exemplar as grad-outside-scan construction, acknowledge no current pipeline inhabits that regime, effort L; drop decorative make_axis_dispatch verify-path) |
| B4 crossover ladder | SURVIVES |
| C1 cut pinned-memory | SURVIVES |
| C2 demote CPU walls | SURVIVES-WITH-CHANGES (replace "AT ALL" overreach with claim-class language; acknowledge effect on the 0.70x flagship result) |
| C3 defer Foldcomp | SURVIVES |
| D1-D5 open questions | SURVIVE as asked; D1 must be expanded with contract-bump mechanics before it can gate B1's final shape; D5's factual premise verified (PARITY_TOLERANCE at prof_stage1_onehot_micro.py:68) |

## BOTTOM LINE

Adopt the list after two corrections: (1) make B1's enforcement-location decision explicit and do
not let self-declared clock strings anywhere near a gate until the trust model is written down;
(2) fix A1/B3's scan premise (grad-inside-body vs grad-outside-scan regimes). A2+A4+A7+C1+C3 can
proceed today unmodified.
