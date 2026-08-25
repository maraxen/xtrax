# Spec-challenger memo r3 — FINAL VERIFICATION — 260825_xtrax-cse-runtime-opt (DRAFT v4)

Reviewer: SPEC-CHALLENGER round 3 (final). Method: each OBJ-R2-0x checked against the v4 fix's
MECHANISM, not its letter; spec citations are line numbers of
`.praxia/docs/specs/260825_xtrax-cse-runtime-opt-spec.md` (DRAFT v4). JAX behavior claims re-tested
this session in `.venv/bin/python` (JAX 0.10.2): [V1] implements the SPECIFIED §4.1 fingerprint
algorithm verbatim on a real jaxpr; [V2] replicates E2 faithfully (captured-array consts) and tests
the const-folded digest. No source files modified; this memo is the sole artifact.

---

## Part 1 — Disposition audit (OBJ-R2-01..08)

| OBJ | Verdict | One-line reason |
|-----|---------|-----------------|
| R2-01 | RESOLVED | §4.1 now canonicalizes Literal operands into classes keyed by `(dtype, value.tobytes())` (spec:88-91) alongside union-find var reps. [V1]: premise replicated (two distinct Literal objects for `2.0`); the specified algorithm converges in 1 round to exactly AC1's TWO duplicate classes ({sin,sin}, {mul,mul}). Mechanism fixed, not papered over. |
| R2-02 | RESOLVED | §4.2.2 folds every `ClosedJaxpr.consts` entry through the leaf-digest primitive in ascending const-var order (spec:143-149); A1′ restated WITHOUT injectivity (A1′′, spec:313-317); AC18 pins discrimination (spec:286). [V2]: E2 replicates (byte-identical `str` for different 64-elem captured arrays) and folded digests differ. Digest-once-per-wrap is sound since program+consts are fixed at wrap. |
| R2-03 | RESOLVED | §4.2.4 defines brackets: op time runs first-call-start → readiness-confirmed (check INSIDE bracket, always); key-build sync waits attributable to previously-enqueued work re-attributed to OP time (spec:171-176); advisory DISABLED in pipelining mode (spec:177-179); AC16 pinned to blocking mode (spec:284). All three mis-attribution channels closed. Minor note: detecting "had to drain prior work" is conservative-implementable (attribute all key-build sync waits to op time), acceptable. |
| R2-04 | RESOLVED | §4.2.2 gives every leaf an explicit digest: arrays per core+extension; scalars/bools `(type-tag, repr-of-value)`; strings NFC; other leaf types raise `MemoKeyUnsupportedLeafError` (admission restricted, not silently under-keyed) (spec:153-158). AC17 pins zero cross-float-value hits (spec:285). np.float64 subclasses float → repr path carries value; other np scalars hit the typed error. Silent-wrong-hit class closed either way. |
| R2-05 | RESOLVED | §4.2.3 qualifications: spot-check awaits readiness scoped to spot-check calls only (i); pending entries excluded from `bytes_cached`, eviction drops reference without freeing in-flight compute (ii); concurrency documented (iii); memory-bound claim qualified to blocking mode ONLY — pipelining bounds entry count, not bytes (iv) (spec:162-170). The "either way" contradiction is deleted, not defended. AC9 stays pinned to blocking mode (spec:277). |
| R2-06 | RESOLVED | `_stamp_override` ctor guard: raises unless `XTRAX_MEMO_STAMP_OVERRIDE=1` (spec:123); AC19 tests the raise (spec:287). Frozen-dataclass `__post_init__` raise is implementable; a config-driven builder can no longer ship an injected stamp without the production process carrying the env var. Containment at the right chokepoint. |
| R2-07 | RESOLVED | `DedupSynthesisResult` payload carries `stage` (4 distinguishable values), `sampled_ratio`, `transfer_bytes_spent`, `k_bucket_bytes` (spec:209-216) — the three former-None outcomes are observable and AC8/AC10 are testable; budget bullet now cites real metadata (spec:259-262). Sampling rule PINNED: uniform-stride linspace over `[0,N)` dedup+sorted, residual between-stride false-negative direction mandated to docstring (spec:230-235). Nit (non-blocking): rule for splitting `"no_duplication"` vs `"below_threshold"` labels at ratio==0-vs-small unstated; trivially implementable, no AC depends on the split. |
| R2-08 | RESOLVED | Deferred-screen latch: first `MemoImpurityError` poisons the wrapper until explicit `.memo_rewrap()`; subsequent calls raise immediately without re-screen/re-trace (spec:137-142); the one traced execution on the deferred path is a DOCUMENTED cost, not an unacknowledged effect (spec:140-142). AC20 pins both call behaviors (spec:288). Both sub-points addressed. |

Tally: 8 RESOLVED, 0 PARTIAL, 0 FAILED. Both empirically-demonstrated blockers were re-verified by
mechanism this session ([V1], [V2]); neither fix is cosmetic.

## Experiment log (this session, `.venv/bin/python`, JAX 0.10.2)

[V1] Implemented §4.1's exact fingerprint scheme (union-find var reps + `(dtype, tobytes())`
literal classes, iterate-to-fixpoint) on `make_jaxpr` of duplicated `sin(x)*2.0`: 2 distinct
Literal objects confirmed; round-0 fingerprints collapse sin pair, round-1 collapses mul pair;
final duplicate classes = {sin: 2, mul: 2}. AC1 achievable.
[V2] E2-faithful replication (spike-at-0 vs spike-at-63 captured array consts):
`str(ClosedJaxpr)` byte-IDENTICAL; sha256(text + consts sorted by var, dtype-name + C-order
tobytes) digests DIFFER. AC18 mechanism confirmed.
(Note: a first V2 attempt using `.at[idx].set(1.0)` inside the trace was discarded — the index is
an inlined scalar literal, so texts differ legitimately; does not exercise E2.)

## Part 2 — Internal coherence findings

C1 (MINOR, pre-plan repair REQUIRED): §6 phased delivery does NOT cover every AC. P0 lists
AC1/AC2; P1 lists AC3–AC6, AC9, AC12–AC16; P2 lists AC7/AC8/AC10/AC11. **AC17–AC20 — the four v4
additions guarding the two demonstrated hazards (E2/E4), the stamp guard, and the screen latch —
are assigned to no phase.** Placement is unambiguous (all four are wrapper-scope → P1), so
planning is not blocked, but the §6 lists must be amended to include them when the implementation
plan is written, or the round's central regression tests silently drop out of delivery.
C2 (MINOR): Decision-log row D6 (spec:55) still summarizes program identity as "canonical text
digest (v3)" without the const-folding amendment; §4.2.2 and A1′′ supersede explicitly, so this is
staleness, not contradiction. Refresh the row when convenient.
C3 (observation): Two reset surfaces coexist — `.memo_reset()` clears the poisoned spot-check
counter (R1-14 lifecycle) and `.memo_rewrap()` clears the deferred-screen latch (R2-08). Distinct
concerns, both defined; fine, but document the distinction in one place to avoid caller confusion.
C4 (observation): Two collision error types — `DedupSynthesisCollisionError` (synthesize path,
spec:251) vs `DedupSpecCollisionError` (merge helper, spec:255). Intentionally component-scoped,
both named; consistent with G4's typed-error convention.
C5: All named errors have defining mentions (CseTraceError with module `inference/errors.py`
named; MemoImpurityError, MemoMultiDeviceError, MemoStalenessError,
MemoKeyUnsupportedLeafError, DedupSynthesisUnsupportedError, DedupSynthesisCollisionError,
DedupSpecCollisionError). Only CseTraceError states its module; P1/P2 bundle errors with their
modules, acceptable.
C6: Cross-section consistency holds where it mattered: pipelining-mode claims in §4.2.3(iv),
advisory-off in §4.2.4, and AC9/AC16's blocking-mode pins agree; AC13's injected stamps presuppose
AC19's env gate (test sets the env var). All 20 ACs have implementable GWT (AC9's
"`.is_ready()`-equivalent" and AC15's simulated multi-device are loosely worded but operationally
clear).

## Part 3 — FINAL VERDICT

**VERDICT: ACCEPT** (confidence 0.85)

Justification:
1. All 8 R2 objections RESOLVED against their mechanisms. The two BLOCKERs were re-demonstrated
   fixed empirically on the repo's own venv this session: the specified fixpoint algorithm passes
   AC1's flagship example ([V1]), and the const-folded program digest discriminates the exact E2
   collision case [V2]. These were the only defects capable of producing wrong results or
   unachievable ACs, and both are closed with pinned ACs (AC1 unchanged-achievable, AC17/AC18 new).
2. The three R2 MAJORs received definitional fixes (timing brackets + mode-disable; leaf-admission
   restriction; pipelining semantics with the false memory claim deleted). No measurement-validity
   or silent-under-keying channel remains open.
3. Remaining findings are MINOR-only. C1 (AC17–20 missing from §6 phase lists) is the largest, but
   it is bookkeeping: every affected AC has a concrete GWT row, an obvious unambiguous component
   home (P1), and no design ambiguity. It does not block implementation planning of any phase,
   PROVIDED the planner amends §6 as its first step — recorded here as a REQUIRED pre-plan repair,
   not a spec defect requiring another revision cycle.
4. No architectural doubt survives three rounds: the decorator-first architecture, two-stage
   synthesis, measured advisory, and text+const program identity are internally consistent across
   §§2-8, and the risk register honestly carries what remains open (R2 format drift, R3
   heterogeneous axes, R4 budget bypass, R6 trace-cache masking).

Conditions carried forward (binding on the implementation plan, not the spec): (a) fold AC17–AC20
into P1's test list; (b) pin the `"no_duplication"` vs `"below_threshold"` label rule in the P2
docstring; (c) refresh the D6 row opportunistically.
