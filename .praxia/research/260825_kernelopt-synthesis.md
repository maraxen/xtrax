# Synthesis: kernel-opt grounding -> spec-optimization recommendations

Date: 2026-08-25 | Branch: feat/profiling-stage2-evidence
Inputs: 260825_kernelopt-grounding-report.md (evidence), 260825_kernelopt-spec-proposals.md
(20 items), 260825_kernelopt-review-technical.md (duck), 260825_kernelopt-review-strategy.md
(frog). This doc reconciles the four into ONE decision-ready set for Marielle.
Reviewer disagreements are resolved explicitly where they conflict.

## 1. The three findings that reorder everything

1. **[frog BLOCKER-1] GPU scope attribution is the true critical path.** All landed L40S
   stage-2 records carry null scopes; `claims.select_sources` requires >=2 non-null scopes
   before TERM_RANKING. Every GPU record emitted today is ranking-dead regardless of variance
   fields. None of the 20 proposals addressed this. It becomes new workstream W0.
   (56a9f55 fixed HLO-text attribution but not executed-trace fused-computation naming.)
2. **[frog BLOCKER-2] Stage-2 provenance gap.** Landed records carry git_sha `c01e4d93...`
   which does not resolve in this repo, were emitted by a driver absent from scripts/, and test
   an unplanned fourth probe family (`aminx.ConditionalDecode`). `_reject_unverifiable_git_sha`
   checks format, not resolvability -- a laundering hole by the contract's own philosophy.
3. **[duck FATAL #1] B1(b) self-declared clock provenance contradicts record.py's documented
   anti-laundering design** ("a freeform caller-set string could otherwise falsely agree...
   under the unanimity guard"). Clock keys may be advisory metadata or harness-side
   auto-captured -- never gate-material as driver-declared strings.

## 2. Reconciled workstreams

### W0 -- UNBLOCK RANKING (new; supersedes "B1 gates everything")
- W0a: one attribution spike from workflows.md's three candidate directions
  (sub-scope granularity / non-fused wrapper ops / executor-thunk->HLO mapping). Owner + pick
  needed from Marielle (D8).
- W0b: canonicalize the stage-2 pipeline: in-repo (or sha-pinned) driver for
  ebm_cond_decode, wheel sha-stamping policy (D6/D7). Consider adding sha-resolvability to
  `_reject_unverifiable_git_sha` (strictly tighter; house-rule compatible).

### W1 -- SHIP TODAY (doc commit; all survive both reviews unmodified or near-so)
- A2 measurement-protocol bundle [both endorse; strongest item]: strongest-shipped-baseline
  rule, micro/macro dissonance clause, ablation preregistration keys.
- A4 async-overlap crossover framing [both endorse].
- A7 regime-shift bucket boundaries [frog amendment: word conditionally until sweeps exist].
- A8 dual-path boundary costing [duck amendment: GDS single-query caveat + own-artifact
  provenance for the ~1040x constant].
- A5 donation sharp-edges box [duck amendment: acknowledge closure-pattern preference at
  inference.py:159-160; specify donation-effect verification method].
- C1 pinned-memory cut [frog amendment: one cross-ref line to Tier-3 offload text].
- C3 Foldcomp query: **run today, minutes-cheap** [frog rejects deferral; duck's "survives"
  conceded on cost asymmetry].

### W2 -- REDESIGNED B1 (before any NEW GPU campaign claims rankings)
Duck's corrections absorbed:
- Enforcement location explicit: n_runs>=3 + finite dispersion become REQUIRED_METRICS for
  TERM_RANKING only if Marielle accepts CONTRACT_VERSION MAJOR bump + permanent mixability
  split of pre-bump records + fixture churn (D1 reframed per frog CONCERN-a3; note zero
  rankable records exist today so retroactivity loss is empty-set -- say so explicitly).
- Clock keys: advisory config metadata with written trust model, OR harness-side
  auto-capture via nvidia-smi outside the leaf package. Never unanimity/gate material.
- Dispersion statistic defined (std over N runs; declared per record).
- n>=3 floor justified or raised toward the cited n=5/10 norms.
- gpu_clock_mhz semantics under multi-GPU stated (devices[0] convention documented).

### W3 -- P4 BUILD (resequenced per both reviews)
- B3 scan-remat driver pair, REFRAMED per duck MAJOR #2: exemplar must be constructed
  grad-OUTSIDE-scan (the regime remat actually helps); document that accumulate_grads
  (grad-inside-body) is NOT helped and has no production callers. Effort L. A1 recipe ships
  WITH B3 (house rule; frog CONCERN-b1), not before.
- B2 intensity emitter + device-ceilings.md, FIXED per duck MAJOR #5: intensity derived
  analytically from HLO op counts (backend-independent) or collected on-GPU; ceilings from
  published specs, achieved columns pending attribution fix; named maintainer (D9).
- A6 mixed-precision recipe: demoted below W2/W3, labeled literature-only-no-xtrax-measurement.
- B4 crossover ladder: deferred unless GPU imminent (its own crossover logic).
- A1's AD-inside-body branch added per duck verdict.
- A3 opaque-kernel rule: land as prospective insurance tied to first CustomCall/Pallas PR;
  recite honest instrument (post-fusion thunk-event counts from executed traces;
  cost_analysis provides no thunk count).

### W4 -- CUTS/POLICY
- C2 CPU-wall demotion: adopt with claim-class language (not "AT ALL"), prospective phrasing,
  AND one cheap CPU-variance spike (median-of-n, taskset) before permanent demotion [both
  reviewers amended]. Note honestly: this reclassifies our own flagship 0.70x result.

## 3. Consolidated decisions for Marielle (was D1-D5, now D1-D10)

| # | Decision | Gates |
|---|---|---|
| D1 | Tighten TERM_RANKING floor (n_runs>=3+dispersion) accepting CONTRACT_VERSION MAJOR bump + record mixability split? Zero rankable records exist today, so cost is future-only. | W2 |
| D2 | Can L40S lock app clocks (nvidia-smi -lgc needs root)? If not: unlocked-clock protocol = recorded mhz + widened dispersion? | W2 |
| D3 | Stage-2 matrix priority order: proposed attribution-fix > driver-canonicalization > B3 scan-remat > one-hot > feed-ladder > host-boundary. L40S-hours available? | W0-W3 |
| D4 | Performance-gate tripwires stay unwired (repo pins no-dispatch-config)? | policy |
| D5 | Parity-tolerance ownership for dtype era: per-experiment declared vs central table? | W3+ |
| D6 | Wheel provenance policy: how are dogfood wheels sha-stamped so record git_shas resolve? Add resolvability check to _reject_unverifiable_git_sha? | W0b |
| D7 | Where does the ebm_cond_decode stage-2 driver live? In-repo, or out-of-repo tooling formally acknowledged? Does the matrix expand to 4 families? | W0b |
| D8 | Which attribution direction gets the next spike, and who owns it? | W0a |
| D9 | Is L40S the only target device? Who owns device-ceilings.md against hardware churn? | W3/B2 |
| D10 | Who pays CI cost of stricter floors (fixture churn)? | W2 |

## 4. Opportunity-cost verdict (adopted from frog)

The next engineering day goes to W0 (canonicalize stage-2 pipeline + attribution spike),
NOT to more instrument polishing. The founding diagnosis was "measurement yes, optimization
no"; five days of instruments without a measured GPU win would repeat the asymmetry at a
higher level. W1 ships today because it protects everything downstream; W2 lands when D1/D2
are answered; B3 remains the highest-value build AFTER W0 unblocks rankable output.
