# Adversarial Strategy Review — 260825_kernelopt-spec-proposals.md

Date: 2026-08-25 | Branch: feat/profiling-stage2-evidence (verified at 28409ac)
Reviewer role: adversarial (strategic + process attack: priorities, sequencing, scope discipline, omissions)
Objects under review:
- `.praxia/research/260825_kernelopt-spec-proposals.md` (20 items: A1-A8, B1-B4, C1-C3, D1-D5)
- `.praxia/docs/specs/260825_jax-optimizing-skill-scope.md` (P1-P3 delivered)
- `.praxia/research/260825_kernelopt-grounding-report.md` (evidence base)
- Fresh commits: 56a9f55 (fused-computation attribution), 8385dfb (L40S GPU stage-2 records)

Every criticism below was verified against repo state; verification method listed at the end.

---

## VERDICT UP FRONT

The list is well-grounded in its corpus and its house-rule audit is honest, but it is written as if GPU re-runs were still in the future. They are not: two L40S dogfood campaigns already landed (56a9f55 at 14:14, 8385dfb at 15:08 on 2026-08-25), their records are TERM_RANKING-dead for a reason no proposal addresses, and they were produced by tooling that does not exist in this repo from a wheel whose sha is not in this history. The plan's single "GATING" item (B1) is necessary but NOT sufficient, and the actually-binding constraint is missing from all 20 items.

---

## BLOCKERS

### BLOCKER-1 — The real gate (GPU scope attribution) is absent from the proposal list

All three landed L40S stage-2 records (`outputs/profiling/stage2/states{1,8,32}/stage2_ebm_cond_decode_*.json`, commit 8385dfb) carry `scopes = {"ebm_conditional_decode": null, "ebm_state_axis": null}` — zero non-null attributed scopes. `src/xtrax/profiling/claims.py::select_sources` (~lines 126-139) requires >=2 non-None scopes before any record set can back TERM_RANKING. Therefore NO GPU record can reach TERM_RANKING today, variance fields or not.

`agent_assets/skills/xtrax-probing/references/workflows.md` (section added in 8385dfb) documents this as an open tuning item with three named candidate directions: sub-scope granularity below fusion boundaries; non-fused wrapper ops around measured regions; executor-thunk -> HLO-instruction mapping from the XLA runtime.

Note also: 56a9f55 ("attribute fused-computation trace events to named scopes") landed BEFORE 8385dfb and did NOT fix the GPU case — the fix worked for HLO text taken from the compiled executable, but executed trace events name FUSED computations (`copy_bitcast_fusion.N`, `ynn_fusion.N`), never inner instructions.

B1 is labeled "highest priority overall … GATING"; it is at best co-gate #2. A 20-item plan that omits the one problem demonstrably killing every existing GPU record has misidentified its critical path.

### BLOCKER-2 — Stage-2 provenance gap the plan never mentions

The landed records carry `git_sha c01e4d93f8a87e87ae735da59439725903514345`, which does not exist in this repo (`git cat-file` fails); commit 8385dfb's message says "current wheel". No script in `scripts/` emits `stage2_ebm_cond_decode` (`ls scripts/ | grep -i stage2` is empty); record config says `kernel: aminx.ConditionalDecode`, which is NOT one of the three probe families scope doc section 7b defines as the Stage-2 matrix (one-hot, host-boundary, feed-overlap).

Consequences:
(a) `claims.py::_reject_unverifiable_git_sha` only string-checks shas ("unknown", "-dirty", "-unverified") — a plausible-looking sha from a wheel built off-repo passes format checks while being unverifiable in practice; unanimity on git_sha then permanently chains any future claim to a commit outside this history.
(b) Stage-2 evidence collection has already drifted off-plan (a fourth, unplanned family) without the plan noticing.

Neither D1-D5 asks where the stage-2 driver lives or how wheels are sha-stamped.

---

## (a) SEQUENCING VALIDITY

### CONCERN-a1 — B1's premise "BEFORE any GPU re-run" is factually stale

GPU runs happened twice before this proposal round (first external dogfood per 56a9f55; second properly-resourced run 21235335 per 8385dfb). The plan never acknowledges records emitted under the weaker protocol. The honest retroactivity story exists and is actually benign — the 8385dfb records are already citable only at DISPATCH_COUNT+STRUCTURAL grade (commit message), and are independently TERM_RANKING-dead via null scopes, so B1/D1 invalidates nothing that was ever rankable — but the plan must SAY this explicitly. As written, D1's debate over "retroactive rankability loss" is theater over an empty set: there are zero rankable GPU records today to lose (verified across all of `outputs/profiling/stage2/**`).

### CONCERN-a2 — DO NEXT ordering is executable only vacuously

A1+A2+A3 as one doc commit: yes, doable today. But B1 (effort M, medium risk, shared infra pinned by `tests/scripts/test_prof_optimizing_drivers.py` + `tests/profiling/test_claim_contract.py`) is gated on D1/D2 answers that do not exist yet, while GPU-hours continue burning on runs whose output cannot rank regardless of B1. The sequencing summary's claim "B1 gates everything GPU" inverts the dependency: attribution-fix + stage-2-driver-canonicalization gate everything GPU; B1 gates only the trustworthiness of rankings that cannot currently exist.

### CONCERN-a3 — B1's "no schema surgery" materially understates blast radius

If option (c) lands, n_runs/dispersion become required for TERM_RANKING. Per claims.py's own documented bump rule (~lines 73-88): changing REQUIRED_METRICS triggers a CONTRACT_VERSION MAJOR bump (the removal precedent bumped 2.0->3.0), and `select_sources` enforces MAJOR-component unanimity — meaning ALL pre-bump records become permanently unmixable with new ones in any claim, which is strictly stronger than "losing rankability." Pinned fixtures (`test_claim_contract.py` hardcodes `total_step_seconds` at lines 67/76/234) plus the 11 driver tests need rework.

Worse, the same claims.py comment states the house precedent verbatim: "re-spike before adding a required metric, never after records exist." Records now exist. D1 should be reframed around this precedent and the mixability consequence, not just "stricter floor OK?"

---

## (b) SCOPE CREEP AUDIT

House style per scope doc section 5 / skill non-negotiables: every reference ships WITH its measured driver; code wins over skill text; no aspirational citations.

### CONCERN-b1 — A1 [DO NEXT] breaks the house rule by construction

The proposal itself admits B3 "gives A1 its measured exemplar," yet A1 lands first. tier3-composition.md would gain a scan-remat decision tree whose branches ("auto-remat suffices iff stage-0 bytes curve is flat") reference curves nobody has produced for xtrax scan bodies. If B3 slips, the skill grows aspirational content — exactly what P1 promised never to ship ("real verify-paths only -- no aspirational citations"). Amendment: land A1 as a stub explicitly marked unmeasured-until-B3, or swap the order.

### CONCERN-b2 — A6 is the clearest creep item

Pure literature transcription (f13 Micikevicius / FP8 formats) with zero motivating probe, zero dtype candidates, CPU-only machine. Its own risk note ("could be misread as license to coerce dtypes") argues against shipping now. Counter-case for keeping: pre-registering the numerics gate before GPU dtype candidates arrive prevents post-hoc tolerance shopping; cost is S. Verdict: keep but demote below the B-items and mark clearly as literature-grounded-with-no-xtrax-measurement.

### NIT-b3 — A3 self-describes as "pure insurance"

For a custom-op path that does not exist yet. Insurance rules whose verify-paths point at scripts that never exercise the rule are dead letters. Acceptable at effort S; tie activation explicitly to the first CustomCall/Pallas PR.

### NIT-b4 — B4 builds an instrument for a regime its own words say is unreachable

"on GPU, real H2D makes the beneficial regime reachable for the first time" — so on this CPU-only box, the crossover-sweep mode maps a regime the skill says is empty. Fine if GPU access is imminent; otherwise defer by B4's own crossover logic.

### CONCERN-b5 — B2's data source is misstated

"L40S row filled first from the dogfood run": the dogfood run contains total_step_seconds + dispatch counts only — no bandwidth or ceiling measurements — and achieved-vs-ceiling needs attributed scopes (which do not exist; see BLOCKER-1). Ceilings must come from published specs; achieved columns stay pending. Also creates a new maintenance surface (see D9 below).

---

## (c) CUT FAIRNESS

### C1 (cut pinned-memory from Tier-2) — endorse WITH amendment

Strongest counter-case: A1's own content includes name-based remat policies with residual OFFLOAD TO PINNED HOST MEMORY (f7). Tier-2 pinned placement and Tier-3 remat-offload are the same physical mechanism surfacing in two tiers; cutting Tier-2 entirely while A1 adds offload content to Tier-3 creates an internal inconsistency readers will trip on. The proposal's own fallback ("keep at most a one-line mention") is correct — make it a cross-ref line to the Tier-3 offload text, not silence. Also noted: the cut text exists only in scope doc section 2, never shipped in tier2-data-movement.md — this edits a spec and removes nothing user-facing. Value near zero either way.

### C2 (demote CPU micro-wall precision work) — endorse direction, CONCERN on reasoning

Strongest counter-case: the demotion declares CPU walls unrescuable without ever trying the cheap rescue — median-of-n, taskset/core pinning, warm-up discipline — i.e., exactly the variance machinery B1 spends M-effort to build for GPU. The list considers spread-reporting worth building for the L40S but forecloses it for the machine every CI run and smoke test executes on. Result: every sub-TERM_RANKING wall claim loses quantitative evidence forever on this hardware, and pre-GPU candidate triage goes blind. Amendments: phrase prospectively (the proposal already does), AND add "attempt one cheap CPU-variance spike before accepting permanent demotion."

### C3 (defer Foldcomp follow-up query) — REJECT the deferral as stated

The grounding report itself flags Foldcomp as a coverage hole (never cited by any of 12 answers; 25 sources vs 21 observed IDs). Cost of one follow-up nlm query: minutes, zero code risk. C3 budgets more effort writing the deferral rationale than doing the thing. This is misordered economy inside a list that ships speculative doc items at S/M effort. Run the query today.

---

## (d) MISSING MARIELLE QUESTIONS

D1-D5 are genuinely decision-shaped (D2 on `nvidia-smi -lgc` root access is excellent). Missing:

- **D6 (BLOCKER-adjacent) — Wheel provenance policy**: how are dogfood wheels built and sha-stamped such that record git_shas resolve in this repo? Should `_reject_unverifiable_git_sha` gain resolvability checking? The c01e4d93 records are format-valid but unverifiable in practice.
- **D7 — Stage-2 driver ownership/versioning**: where does the ebm_cond_decode stage-2 driver live? Does the Stage-2 matrix formally expand to include it, or is out-of-repo tooling emitting records into `outputs/profiling/` acceptable?
- **D8 — GPU attribution fix ownership**: which of workflows.md's three candidate directions gets the next spike, who owns it, and does it precede all Stage-2 ranking ambitions?
- **D9 — Device fleet & ceilings maintenance**: device_kind is a unanimity field, so every claim is per-device forever. Is L40S the only target? Who owns `references/device-ceilings.md` as hardware churns (stale ceiling rows would silently distort predicted-vs-achieved attribution)?
- **D10 — CI cost accounting for stricter floors**: contract MAJOR bump or guard-tightening both churn pinned fixtures; who pays and when?

---

## (e) OPPORTUNITY COST

Yes, there is a higher-leverage day, and it is not among the 20 items: **canonicalize the Stage-2 evidence pipeline** — in-repo (or sha-pinned) stage-2 driver, wheel provenance policy, one attribution-direction spike from workflows.md's candidate list. Rationale: two GPU campaigns already produced three STRUCTURAL-grade-only records; every additional un-fixed run accumulates dead inventory and burns L40S-hours on numbers that can never back a ranking claim.

Second alternative worth naming: **ship ONE end-to-end measured GPU win** (B3's scan-remat on `accumulate_grads` at states32 scale) to prove the skill's loop closes on real hardware.

The 20-item list is ~80% protocol and documentation refinement; the scope doc's founding diagnosis was "measurement yes, optimization no." A fifth consecutive day of instrument-polishing before the first measured GPU win risks perfecting the instrument instead of using it. B1/B2 remain right answers — but as amendments to that pipeline day, not as its replacement.

---

## WHAT SURVIVES

The core survives. A2's strawman-baseline clause (i) and micro/macro-dissonance clause (ii) are the strongest items in the list — they directly protect future records and are grounded in d5/d6. A4/A5/A7/A8 are honest small doc work with verified anchors (`--feed-sleep-ms`/`--buffer-size` confirmed in prof_stage1_feed_overlap.py; the 0.5x-1.9x concession confirmed at SKILL.md:82). B1 and B2 survive as ideas but must be re-sequenced behind/among the blocker fixes, and B1 needs the contract-bump honesty amendment. The cuts survive mostly: C1 amended to cross-ref, C2 amended with a variance spike, C3 reversed.

---

## VERDICT TABLE

| Item | Verdict |
|---|---|
| A1 scan-remat recipe | endorses-with-amendments (land after/with B3 driver, or stub-mark unmeasured) |
| A2 measurement bundle (i/ii/iii) | endorses |
| A3 opaque-kernel cliff rule | endorses-with-amendments (mark prospective; activate on first CustomCall PR) |
| A4 async-overlap crossover framing | endorses |
| A5 donation sharp-edges box | endorses |
| A6 mixed-precision recipe | endorses-with-amendments (demote below B-items; label literature-only) |
| A7 regime-shift bucket boundaries | endorses-with-amendments (word conditionally; shift points unmeasured until sweeps exist) |
| A8 dual-path boundary costing | endorses |
| B1 variance + clock fields | endorses-with-amendments (acknowledge landed records; CONTRACT_VERSION MAJOR honesty; re-spike precedent; co-gate status with attribution fix) |
| B2 intensity emitter + device-ceilings | endorses-with-amendments (ceilings from published specs; achieved columns pending; assign maintainer) |
| B3 scan-remat driver pair | endorses (sequence after attribution spike; highest-value build) |
| B4 crossover-sweep mode | endorses-with-amendments (defer unless GPU imminent) |
| C1 pinned-memory cut | endorses-with-amendments (one cross-ref line to Tier-3 offload, not silence) |
| C2 CPU-wall demotion | endorses-with-amendments (add one cheap CPU-variance spike before permanent demotion) |
| C3 Foldcomp deferral | rejects (run the minutes-cheap query now) |
| D1 floor tightening question | endorses-with-amendments (reframe around contract-bump/mixability/re-spike precedent) |
| D2 clock-lock feasibility | endorses |
| D3 GPU budget/priority order | endorses-with-amendments (insert attribution fix and stage-2 driver canonicalization as priorities 0 and 1) |
| D4 tripwire wiring | endorses |
| D5 parity-tolerance ownership | endorses |
| D6-D10 (missing) | propose adding: wheel provenance; stage-2 driver ownership; attribution-fix ownership; fleet/ceilings maintenance; CI cost of floors |

Severity tally: 2 BLOCKERS, 9 CONCERNS, 4 NITs.

---

## VALIDATION PERFORMED

- Read all three source documents in full.
- Inspected commits 8385dfb and 56a9f55 diffs (including workflows.md addition).
- Parsed all three landed stage-2 JSON records: scopes all null, metrics = {n_compilations, n_executions, n_jit_traces, total_step_seconds}, git_sha c01e4d93…, jax/jaxlib 0.10.2.
- Read `src/xtrax/profiling/claims.py` in full: REQUIRED_METRICS, CONTRACT_VERSION bump-rule comment, select_sources scope filter, _UNANIMITY_FIELDS, _reject_unverifiable_git_sha.
- Verified git_sha c01e4d93 unresolvable via `git cat-file`.
- Confirmed absence of any stage2 script under `scripts/`.
- Spot-checked verify-path claims: `--feed-sleep-ms`/`--buffer-size` in prof_stage1_feed_overlap.py; 0.5x-1.9x fluctuation concession at SKILL.md:82 and tier3-composition.md:46-47; pinned-memory absent from shipped tier2-data-movement.md; `cost_analysis()` present in prof_stage0_onehot_cost.py; emitters.py config typed `dict[str, str]`; record.py metrics coerced float-only post-init.
