# Spec: xtrax runtime compute-reuse layer (CSE detection, content-keyed memoization, dedup synthesis)

- **task_id**: `260825_xtrax-cse-runtime-opt`
- **status**: FINAL v4.1 — adversarially converged, verdict ACCEPT (confidence 0.85)
- **origin**: contemplex brainstorm session `3d29614e` (winner recorded, INVEST passed)
- **references**: `.praxia/docs/research/260825_cse-recon.md`, `.praxia/docs/research/260825_jax-cse-ecosystem.md`,
  `.praxia/docs/research/260825_spec-challenger-r1.md`

## 0. Revision history

| Ver | Change |
|-----|--------|
| v2 | Recon + research merged |
| v3 | Challenger r1 dispositions: OBJ-R1-01..16 addressed; program identity redesigned on empirical evidence (see A1′) |
| v4 | Challenger r2 dispositions: literal-value canonicalization in CSE equivalence (R2-01); const-folding program digest (R2-02); async-honest timing brackets (R2-03); non-array leaf keys (R2-04); pipelining-mode semantics (R2-05); stamp-seam guard (R2-06); synthesis result payload + pinned sampling rule (R2-07); screen latch (R2-08); named collision error (R1-03 completion) |
| v4.1 | Challenger-r3 ACCEPT conditions applied: AC17–AC20 folded into P1 (condition a); stage-label rule pinned (condition b); D6 row refreshed (condition c). Final verdict: **ACCEPT @ 0.85** (`260825_spec-challenger-r3.md`) |

## 1. Problem statement

JAX users pay twice for compute they already paid for: (1) duplicated subexpressions inside one
program are removed by XLA only *after* lowering, so no tool tells the author their code has
redundant structure; (2) identical calls across invocations are never value-cached by `jax.jit`
(its cache is keyed on structure/shape/dtype, never values); (3) duplicated batch rows are
recomputed per-row unless the caller hand-builds a `DedupSpec`. xtrax already owns the dedup
execution machinery (`DedupSpec`/`DedupGather`, k-bucketed) and the structural introspection
pattern (`infer_bundle` via `jax.eval_shape`) but provides no detection or automation for any of
the three reuse opportunities.

## 2. Goals / Non-goals

**Goals**
- G1: Detect and *report* duplicate-op structure in a traced function's jaxpr (detection only).
- G2: Provide an opt-in content-keyed value cache around pure jitted inference callables.
- G3: Auto-synthesize exact `DedupSpec`s when sampled evidence indicates row duplication.
- G4: All three preserve xtrax fail-loud conventions: opt-in activation, typed errors, observable telemetry.

**Non-goals**
- N1: No jaxpr rewriting or source transformation. XLA owns intra-program CSE (verified).
- N2: No cross-process/disk persistence of cached device arrays in v1.
- N3: No auto-wiring into `BatchPlanner.plan()` signature in v1 (plan() stays stable).
- N4: No proof of purity. We reject what we can detect; we attest what we cannot prove.
- N5: Single-device scope, MECHANICALLY ENFORCED: `memoize_jaxpr` asserts
  `len(jax.local_devices()) == 1` at wrap time, else raises `MemoMultiDeviceError` (resolves
  OBJ-R1-06: the environment stamp alone permitted same-kind multi-device hits; with the
  assertion, sharding-provenance mismatch is out of scope by construction, not by declaration).

## 3. Decision log (from brainstorm + adversarial rounds)

| # | Decision | Rejected alternatives | Rationale |
|---|----------|----------------------|-----------|
| D1 | Decorator/standalone-first architecture (A1) | A2 planner-integrated, A4 graph-collapse | plan() signature stability + single responsibility; HostPrepGraph nodes are host-prep semantics not device compute |
| D2 | Purity admission = caller attestation + static jaxpr screening tripwire (B3′) | B1 static-only, B2 runtime double-run | Static screen catches stateful primitives/host callbacks/unkeyed random; documented limits corrected per OBJ-R1-09 (derivative-rule bodies irrelevant to forward memo; true blind spots are out-of-trace state, time/I/O, unstable representations) |
| D3 | Cost gating = OBSERVED runtime advisory (v3 amendment of C1/C2) | wrap-time static estimate | OBJ-R1-07: estimator is shape-driven and unavailable on the decorator path; advisory now fires from measured per-entry op-vs-hash wall times after warmup |
| D4 | Dedup synthesis = sample-gated, exact-on-fire construction (v3 amendment of D2-sampling) | always-O(N), always-sampled | OBJ-R1-01: sampled specs cannot be exact; sampling decides WHETHER to pay the exact O(N) pass, never substitutes for it |
| D5 | Salt + spot-check defense against invisible state (pre-mortem) | none raised | Confirmed constructible by experiment A1′: single-capture wrappers hold literalized closure constants permanently |
| D6 | Program identity = text digest **+ folded const values** (v4 amendment) | `hash(ClosedJaxpr)` | `__hash__` non-deterministic across traces (A1′); bare text non-injective over array consts (A1′′/OBJ-R2-02); const-folding closes both |

## 4. Component specs

> Layout: package root `src/xtrax/`; controller loops live in top-level `controller/`.
> No jaxpr introspection exists in src/ today (rg-verified) — component (a) is greenfield.
> Zero production `DedupSpec(` constructors exist today — component (c) is the first.

### 4.1 Component A — `analyze_cse` reporter

```python
# src/xtrax/inference/cse.py (new module)
@dataclass(frozen=True)
class CseDuplicateClass:
    eqn_count: int
    primitive: str
    params_fingerprint: str
    invar_shapes: tuple[tuple[int, ...], ...]
    est_wasted_bytes: int

@dataclass(frozen=True)
class CseReport:
    duplicates: tuple[CseDuplicateClass, ...]   # sorted desc by est_wasted_bytes
    total_eqns: int
    duplicate_eqns: int
    note: str  # states that XLA performs intra-program CSE post-lowering

def analyze_cse(fn, abstract_inputs: Sequence[ShapeDtypeStruct]) -> CseReport: ...
```

- Implementation: `jax.make_jaxpr(fn)(*abstract_inputs)`; equivalence classes computed by
  **fixpoint iteration** mirroring hlo_cse.cc's iterate-to-fixpoint semantics. Fingerprint of an
  eqn = `(primitive.name, params, operand-class representatives)` where operand classes unify
  BOTH (a) var ids through union-find representatives rewritten each round AND (b)
  **Literal operands canonicalized by value**: every Literal joins a class keyed by
  `(dtype, value.tobytes())` so two distinct Literal objects holding equal values share one
  representative (OBJ-R2-01: distinct `mul ... 2.0` Literals otherwise never merge, making AC1
  unachievable). Iterate until no new merges — transitive duplicates (duplicated `sin` AND
  downstream duplicated `mul`) both reported.
- Correspondence claim (scoped): reports match what XLA *can* merge structurally; XLA additionally
  exploits commutativity and constant folding, so reports may UNDERCOUNT nothing but may include
  classes XLA folds differently. The `note` field carries this scoping.
- Detection only. Never returns or applies a rewritten jaxpr.
- Errors: `CseTraceError` (in `inference/errors.py`) if tracing fails.
- Emission: `xtrax explain --report cse` via existing `cli/emit.py` json/text/html paths. JSON
  envelope: **new top-level key** `cse_report` with its own `schema_version` (no bump of the plan
  stats `_meta`). png unsupported (no render semantics).
- Trace-cache interaction (empirical): `make_jaxpr` memoizes per (function identity, shapes);
  analyzing the same fn object twice returns the identical ClosedJaxpr. Documented: callers who
  mutated a closure and want a fresh view must pass a fresh callable (or clear caches); the tool
  surfaces `trace_cache_hit: bool` in the report meta.
- Report shape mirrors the eda TypedDict pattern (`eda/types.py` style) for composition with
  `extract_plan_stats`.

### 4.2 Component B — `memoize_jaxpr` content-keyed wrapper

```python
# src/xtrax/inference/memo.py (new module)
@dataclass(frozen=True)
class MemoPolicy:
    max_entries: int = 128              # LRU bound; sized against device_memory_budget()
    salt: str = ""                      # bump on ANY state change invisible to tracing
    spot_check_every: int = 0           # K=recompute every Kth call via UNWRAPPED fn, allclose-compare
    spot_check_rtol: float = 1e-5
    spot_check_atol: float = 1e-8
    copy_on_return: bool = False        # defensive copies vs downstream donation
    block_on_miss: bool = True          # block_until_ready before store; False = pipelining mode
    slow_ratio_warn: float = 1.0        # measured op/hash seconds below this -> RuntimeWarning once
    _stamp_override: str | None = None  # TEST SEAM: requires XTRAX_MEMO_STAMP_OVERRIDE=1 env, else ctor raises

def memoize_jaxpr(fn=None, *, policy: MemoPolicy = MemoPolicy()) -> Any: ...
```

Semantics:
1. **Admission (B3′)**: wrapping is the purity attestation. Wrap-time static screen walks
   `make_jaxpr(fn)(abstract_probe)` raising `MemoImpurityError` on DETECTABLE violations:
   stateful primitives, callback primitives, unkeyed randomness. Documented blind spots
   (corrected per OBJ-R1-09): (i) Python-side state consumed outside the traced region,
   (ii) time/I/O dependence, (iii) objects whose traced representation is unstable across
   calls. Derivative-rule bodies are NOT a blind spot for forward value memoization (only the
   primal executes; primal bodies inline into the jaxpr and are screened).
   The probe trace requires shape hints: decorator form without shapes defers screening and
   cost telemetry to the FIRST REAL CALL (screen runs there; a violation aborts the call with
   `MemoImpurityError` before any caching begins). **Deferred-screen latch (OBJ-R2-08)**: the
   first `MemoImpurityError` poisons the wrapper — all subsequent calls raise immediately
   without re-screening or re-tracing, until an explicit `.memo_rewrap()`; documented cost of
   the deferred path: one traced execution (host-side effects run once under tracers) before
   the verdict.
2. **Key**: `(program_digest, pytree structure, per-leaf digests, policy.salt, environment stamp)`
   - `program_digest` = `sha256(normalized str(ClosedJaxpr)) **+ folded const values**:
     `str()` is deterministic but NOT INJECTIVE — it prints array constants as type-only free
     vars (OBJ-R2-02 experiment E2), so the digest additionally folds every entry of
     `ClosedJaxpr.consts` through the leaf-digest primitive below, in ascending const-var order.
     NEVER `hash(ClosedJaxpr)`: empirically non-deterministic across traces even for identical
     text (A1′). Digest computed once per wrap (the program is fixed at wrap time).
   - Leaf digest = `update_array_digest` CORE (`zarr_integrity.py:65`: canonicalize +
     `tobytes(order="C")`) PLUS an explicit xtrax extension folding `weak_type` flag and dtype
     name into the digest stream (OBJ-R1-10). Bit-exact by default; -0.0 / NaN payloads hash
     distinctly; canonicalization policy out of scope v1.
   - **Non-array pytree leaves** (OBJ-R2-04): Python scalars become traced INPUTS (not
     literals), so program text does not discriminate their values; every leaf therefore gets an
     explicit digest: arrays per above; Python scalars/bools via `(type-tag, repr-of-value)`;
     strings NFC-normalized; any other leaf type → `MemoKeyUnsupportedLeafError` at wrap/first-
     call screening (admission restricted rather than silently under-keyed).
   - Environment stamp `(jax version, jaxlib version, backend kind, device kind, device index)`
     + the N5 single-device assertion. PRNG keys are values: same-key-same-result holds within an
     RNG implementation, which the version stamp bounds (JEP 9263/263 history).
3. **Async safety (revised per OBJ-R1-15)**: retention of not-yet-ready futures is safe per the
   async-dispatch contract; `block_on_miss=True` (default) blocks before store for
   simplest-correct v1 semantics; `False` stores futures for eager-pipelining callers.
   **Pipelining-mode qualifications (OBJ-R2-05)**: (i) spot-check awaits the cached value's
   readiness before comparing (the await is scoped to spot-check calls only); (ii) evicting a
   pending future drops the reference without freeing in-flight compute — pending entries are
   excluded from `bytes_cached` accounting; (iii) original execution and spot-check recompute may
   run concurrently (inherent to the mode, documented); (iv) the memory-bound guarantee holds
   ONLY in blocking mode — pipelining bounds entry COUNT, not device bytes.
4. **Cost advisory (measured, async-honest brackets per OBJ-R2-03)**: per entry accumulate
   `cum_op_seconds` and `cum_hash_seconds`. Timer brackets are DEFINED: op time = first-call
   start until returned buffers' readiness is confirmed (readiness check INSIDE the bracket,
   always — this is what makes the number backend-meaningful); hash time = key-build section
   only, and if a device→host transfer inside key-build had to drain previously-enqueued work,
   the sync wait is attributed to OP time (measured as readiness delta) not hash time.
   In `block_on_miss=False` mode the advisory is DISABLED (honest attribution is impossible
   without readiness waits; the warning explains this rather than emitting garbage numbers).
   After warmup (first 8 calls), ratio below `slow_ratio_warn` emits one RuntimeWarning per
   wrap site naming measured seconds. Honest about caching making code slower.
5. **Telemetry**: `.memo_stats`: hits, misses, evictions, bytes_cached, last_hit_age,
   cum_op_seconds, cum_hash_seconds, spot_check_mismatches.
   **Spot-check lifecycle (OBJ-R1-14)**: mismatch detected → evict corrupted entry → increment
   counter → raise `MemoStalenessError` IMMEDIATELY. Counter persists (poisoned);
   `.memo_reset()` clears it deliberately. Spot-check recomputes via the UNWRAPPED original
   callable (fresh Python execution, so out-of-trace closure state is re-read) and compares
   NUMERICALLY (allclose rtol/atol) — bitwise comparison would false-positive on legitimate
   GPU nondeterminism (xla determinism.md).
6. **Donation, both directions**: `donate_argnums` users rejected at wrap (input invalidation);
   consumers warned not to pass cached outputs into donation sites; `copy_on_return` offers
   defensive copies (default off).
6. **Reset surfaces** (documented together to avoid caller confusion, per challenger-r3 C3):
   `.memo_reset()` clears the poisoned spot-check counter (staleness lifecycle, §4.2.5);
   `.memo_rewrap()` clears the deferred-screen latch (admission lifecycle, §4.2.1). Distinct
   concerns; neither implies the other.
7. **Seam containment**: wrapper MUST NOT be applied across `guarded_evaluate`
   (`closure_lock.py:233`) — drift detection treats out-of-seam evaluation as HALT. Enforced by
   an alias-aware AST lint test (OBJ-R1-13: resolves `import ... as` aliases; fails if
   `memoize_jaxpr` is applied to `guarded_evaluate` or a function that wraps it; vacuous-green
   until wiring exists, which the test documents).
8. **Honest hit envelope (corrected per OBJ-R1-04)**: the previously claimed within-pass regime
   does NOT exist — candidate_smoke runs in a cpu-only subprocess (`candidate_smoke.py:107-166`),
   checkified_execution path-reloads the callable (`checkified_execution.py:92-97`), and neither
   shares process state with an in-process wrapper. The SOLE v1 hit regimes: (a) unchanged-
   candidate retries in-process (gate failures, smoke reruns routed back through the same
   evaluator), (b) user-owned loops invoking the same wrapped callable repeatedly with identical
   inputs. Cross-candidate hits are impossible by design (source mutation changes the digest).
   The advisory message states this envelope.

### 4.3 Component C — `synthesize_dedup_spec` (exact-on-fire redesign)

```python
# src/xtrax/tiling/dedup_synthesis.py (new module)
@dataclass(frozen=True)
class DedupSynthesisResult:
    spec: DedupSpec | None
    stage: str          # "below_threshold" | "k_over_limit" | "synthesized" | "no_duplication"
    sampled_ratio: float        # estimated duplication ratio from the sample stage
    transfer_bytes_spent: int   # total device->host bytes moved across stages
    k_bucket_bytes: int         # padded working-set bytes when synthesized (budget warning input)

def synthesize_dedup_spec(
    batch_leaves: Sequence[jax.Array],
    *,
    axis: int = 0,
    threshold: float = 0.5,
    max_sample_rows: int = 4096,
    max_unique_k: int = 256,
    existing_specs: Mapping[str, DedupSpec] | None = None,   # OBJ-R1-02
) -> DedupSynthesisResult: ...
```

Two-stage construction (OBJ-R1-01):
1. **Sample stage (cheap gate)**: rows sampled by UNIFORM STRIDE over `[0, N)`
   (`idx = linspace(0, N-1, min(N, max_sample_rows)).astype(int)`, deduplicated, sorted) — NOT a
   prefix sample, so tail-concentrated duplication within the strided grid is visible; residual
   false-negative direction (duplication confined between stride points at low density) stated
   in the docstring. Estimate duplication ratio; below `threshold` → result with
   `stage="no_duplication"|"below_threshold"`, zero O(N) spend. Sampling NEVER produces a spec.
2. **Exact stage (only on fire)**: transfer ALL N rows once; compute exact `unique_indices` /
   `index_map` covering every one of the N positions (satisfying `len(index_map) == N` —
   `DedupSpec.__post_init__` does not check this, so the synthesizer self-asserts it before
   construction). If exact `k > max_unique_k` → result with `stage="k_over_limit"`, carrying
   the O(N) bytes actually spent (OBJ-R2-07: the three None-ish outcomes are now
   distinguishable and AC10's metadata channel exists).
   
Rationale restated (repairs D4's contradiction): the O(N) transfer is unavoidable whenever an
exact spec is built; sampling's value is deciding whether to PAY it when duplication is absent,
and the common all-unique case never pays. The profitable envelope (OBJ-R1-16, stated honestly):
contiguous-row axes, high duplication ratio, k ≤ ~256, N large enough that k ≪ N. Heterogeneous
axes raise `DedupSynthesisUnsupportedError` in v1 — acknowledged gap: `dedup.py`'s own docstring
motivates heterogeneous dedup; padding normalization is the recorded future-work unlock.

Collision & merge policy:
- If `existing_specs` contains the axis name → raise `DedupSynthesisCollisionError`;
  caller-declared intent always wins (OBJ-R1-02 implementable; AC11 has a concrete When-step).
- New helper `merge_dedup_specs(*spec_mappings)` performs collision detection for ANY path
  (caller-vs-synthesized, caller-vs-caller), raising **`DedupSpecCollisionError`** (named here
  completing OBJ-R1-03) on ANY duplicate axis_name regardless of entry route — ONE loud failure
  semantic for the whole subsystem. `plan()`'s silent last-wins dict (`plan.py:215`) is untouched
  per N3; the backlog item records migrating plan() onto the helper post-v1 (inconsistency is
  thereby SCOPED and scheduled, not silent).
- Budget-mode accounting: synthesized decisions flow through Phase 0b which bypasses
  MemoryBudget estimation; the return metadata reports k_bucket-padded working-set bytes and
  the docstring warns budget mode treats dedup axes as free (real budget integration deferred
  with explicit backlog note).
- `verify_dedup_spec(spec, leaves)` opt-in O(N) exactness re-check documented.

## 5. Acceptance criteria (GWT)

| ID | Given | When | Then |
|----|-------|------|------|
| AC1 | fn with two identical `sin(x)*2.0` subexpressions feeding separate consumers | `analyze_cse` runs | TWO duplicate classes reported (both `sin` pair and downstream `mul` pair) — fixpoint semantics |
| AC2 | fn with zero duplicates | analyze | empty duplicates tuple; clean emission in json/text/html |
| AC3 | impure fn using unkeyed randomness | wrap (or first call, deferred-screen path) | `MemoImpurityError`; fn NEVER CONCRETELY EXECUTED (spy on concrete dispatch; tracing itself necessarily runs the body under tracers) |
| AC4 | pure fn called twice, equal-valued inputs | second call | `.memo_stats.hits` increments; op not re-executed (execution-count spy) |
| AC5 | same fn, different `policy.salt` | call twice each | four executions, zero cross-salt hits |
| AC6 | closure dict mutated between iterations, salt unchanged | spot_check_every=K enabled; Kth call | spot-check recomputes via UNWRAPPED fn (sees mutated state), allclose fails → entry evicted, `MemoStalenessError` raised immediately, counter poisoned until `.memo_reset()` |
| AC7 | batch with >50% duplicated rows (N=10000, ~30 uniques) | synthesize | result.stage=="synthesized"; spec passes self-assertion `len(index_map)==N`; `to_dedup_gather()` round-trips |
| AC8 | all-unique batch | synthesize | result.stage=="no_duplication", spec is None; zero O(N) transfer occurred (transfer_bytes_spent covers sample only) |
| AC9 | miss followed by immediate second call (block_on_miss=True) | inspect stored entry | stored buffer is ready (`.is_ready()`-equivalent) and synchronous second call returns allclose-equal values (positive invariant replacing unfalsifiable "no flake") |
| AC10 | exact k > max_unique_k after exact stage | synthesize | result.stage=="k_over_limit", spec None, transfer_bytes_spent reflects full O(N) pass |
| AC11 | existing_specs already declares same axis_name | synthesize | `DedupSynthesisCollisionError`; caller spec untouched |
| AC12 | controller wiring applies memoize_jaxpr to guarded_evaluate chain | AST lint test (alias-resolving) | test FAILS naming the seam rule |
| AC13 | two injected stamps (via `_stamp_override`) | same inputs under each | zero cross-stamp hits |
| AC14 | stored LRU entry manually swapped to corrupt value | spot-check call | `MemoStalenessError` (operational definition: test mutates private LRU entry — stated) |
| AC15 | multi-device session (simulated ≥2 local devices) | wrap | `MemoMultiDeviceError` at wrap time |
| AC16 | persistent-hash-cost-dominated workload (tiny ops, big leaves), 8+ calls, blocking mode | measured telemetry | RuntimeWarning citing measured op/hash seconds with defined brackets |
| AC17 | fn taking a Python float argument, two different values | call twice | ZERO cross-value hits (non-array leaf digests discriminate; E4 hazard closed) |
| AC18 | two wrappers whose closures hold DIFFERENT array constants, equal everything else | cross-call | zero cross-wrapper hits (const-folded program digest discriminates; E2 hazard closed) |
| AC19 | `_stamp_override` set without XTRAX_MEMO_STAMP_OVERRIDE=1 env | construct MemoPolicy | constructor raises |
| AC20 | impure fn on decorator-without-shapes path | first real call, then next call | first call raises MemoImpurityError after one traced execution; second call raises IMMEDIATELY (latched, no re-trace) until memo_rewrap() |

## 6. Phased delivery

- **P0 (reporter)**: `cse.py` + CLI wiring + tests AC1/AC2 (+ trace-cache-hit meta test).
- **P1 (wrapper)**: `memo.py` + errors + tests AC3–AC6, AC9, AC12–AC20 (AC13/AC14 via
  `_stamp_override` seam; AC17–AC20 added per challenger-r3 condition (a)).
- **P2 (synthesis)**: `dedup_synthesis.py` + `merge_dedup_specs` + tests AC7/AC8/AC10/AC11;
  docstring pins the `"no_duplication"` vs `"below_threshold"` label rule at ratio==0 boundary
  (challenger-r3 condition (b): `stage="no_duplication"` iff sampled_ratio == 0.0).

## 7. Risks

- R1: RESOLVED into measured-runtime advisory (§4.2.4) + AC16.
- R2: RESOLVED by D6: text-digest identity; `str(jaxpr)` stability across JAX versions still
  unpinned upstream — CI pins current behavior; format drift = wholesale cache misses (safe
  direction), documented.
- R3: Heterogeneous axes — typed refusal in v1; padding normalization recorded as the unlock.
- R4: Synthesized specs bypass MemoryBudget (Phase 0b); mitigated by metadata + docstring;
  budget integration backlog-noted.
- R5: Re-tracing loaded fns can trip chex guards elsewhere; analyze_cse documents the effect
  and avoids guarded kernels.
- R6 (new): `make_jaxpr`'s own per-function memoization can mask closure mutations in
  ANALYSIS contexts (observed empirically); documented with `trace_cache_hit` surfacing.

## 8. Assumptions

- A1′′ (amends A1′ per OBJ-R2-02): `str(ClosedJaxpr)` is deterministic across traces but NOT
  injective — array constants print type-only (E2), scalar constants inline (E3), Python-scalar
  args trace as valueless inputs (E4). Program identity = text digest PLUS const-value folding;
  non-array leaf digests cover argument values. Verified experiments recorded in
  `260825_spec-challenger-r2.md` [E1-E4].
- A2 (corrected): unchanged-candidate retries and user-owned loops are the sole v1 memo hit
  regimes; within-controller-pass sharing does not exist (subprocess/path-reload boundaries).
- A3: Users accept explicit opt-in ceremony in exchange for correctness guarantees.

## 9. Research grounding (cited)

Primary sources fetched and cited in `.praxia/docs/research/260825_jax-cse-ecosystem.md`:
no public JAX CSE API (optimization_barrier blocks, not performs); pre-opt IR is upstream's
cache identity; RNG-implementation-bounded key semantics (JEP 9263/263); GPU autotune/atomics
nondeterminism → numeric comparison; bidirectional donation hazard; static-sizing convention for
data-dependent dedup (`jnp.unique(size=)`, BCOO nse); disk stores inherit code-execution trust
(v1 declines disk persistence, N2 stands). Code-grounded corrections from challenger r1 cited
inline throughout §4.

## 10. Empirical verification appendix (independent post-verdict execution, 260825)

After verdict ACCEPT, the two load-bearing mechanisms were REIMPLEMENTED INDEPENDENTLY from the
spec text (not from challenger code) and executed on the project venv (JAX 0.10.2), plus the
component-C output contract was exercised through xtrax's real public interfaces:

- **§4.1 fixpoint algorithm**: verbatim reimplementation converges (3 rounds) on AC1's flagship
  function and reports exactly TWO duplicate classes ({sin: 2}, {mul: 2}); clean control fn
  yields zero classes. AC1 achievable by the specified algorithm alone.
- **§4.2.2 const-folded program digest**: bare text digest PROVABLY collides for two programs
  differing only in captured 64-elem array constant (str() byte-identical); folded digest
  discriminates them; equal programs produce equal folded digests (determinism direction).
- **A1′ hash non-determinism**: re-pinned — three fresh traces of an equal-text program yield
  three distinct `hash(ClosedJaxpr)` values while text remains stable.
- **Component C real-path exercise (acceptance path)**: an exact-stage-built
  `DedupSpec(axis_name="batch", unique_indices, index_map, k=30)` over N=10000 rows passes
  `__post_init__`, round-trips through `to_dedup_gather()` (k_bucket 30→32 =
  `get_k_bucket(30)`), and drives the REAL planner end-to-end via the documented injection
  route — `BatchPlanner(dedup_specs=[spec])` (constructor param, `plan.py:150/184`) — producing
  a genuine `DedupGather` AxisDecision with the expected reasoning string. NOTE: `plan()` takes
  NO dedup_specs kwarg; the constructor is the injection seam.
- **F1 (validates self-assertion requirement)**: `DedupSpec.__post_init__` demonstrably does NOT
  reject a half-length index_map — the synthesizer-side `len(index_map)==N` self-assertion in
  §4.3 is mandatory, now proven rather than assumed.
- **F2 (validates collision design)**: the silent last-wins hazard is LIVE — two caller-declared
  specs on axis 'batch' (k=30 then k=5) silently plan with k=5. `merge_dedup_specs` /
  `DedupSpecCollisionError` address a demonstrated defect, and the backlog migration of plan()
  onto the helper is justified by observed behavior.

### 10.1 Integration-boundary round 2 (component B foundation + CLI + full pipeline)

- **§4.2.2 leaf-digest over the real primitive**: `update_array_digest` composes cleanly with
  the weak_type/dtype extension slot; digest deterministic across equal arrays; digest length
  contract holds. **Empirical caveat for implementers**: on JAX 0.10.2, `weak_type` reads
  `False` on every surface probed (`jnp.float32(1.)`, python-scalar broadcast,
  `asarray(np.float32)`); weak types evidently do not survive to public array attributes in
  common construction paths on this version. The key-slot design remains correct (slot present,
  folds `False`), but the AC asserting weak-type discrimination must be written against a
  traced-aval path or downgraded — flagged as an implementation-planning note, not a spec defect.
- **CLI emit() boundary**: json machine contract holds when a new top-level `cse_report` key is
  added beside `_meta`; the text router tolerates non-stats payloads without crashing; the
  png-without-out CLIError footgun guard is live. The v4.1 OQ1 decision (`--report cse`) is
  compatible with all three observed behaviors.
- **Full user-facing pipeline (the acceptance path)**: `@axis_config` →
  `infer_bundle` (roles KNOWN) → exact-stage-style DedupSpec →
  `BatchPlanner(dedup_specs=[spec]).plan()` → decisions `[DedupGather(k=30,k_bucket=32),
  SafeMap]` → `emit(..., fmt="json")` produces valid machine-contract JSON containing the real
  DedupGather count. **PASS** end-to-end on public interfaces only.
- **Bonus observation**: planner emitted a RuntimeWarning that cardinality 10000 is not
  divisible by batch_size 128 ("will raise ValueError at make_axis_dispatch time") for the
  non-dedup axis — the fail-loud convention is observable in practice, corroborating G4.

### 10.2 Runtime numerical validation + two REQUIRED spec amendments (F3, F4)

DedupGather was executed end-to-end through the real runtime (`axis_dispatch`) on semantically
coherent inputs (duplicated rows carry identical payloads):

- **k=1 and k=N degenerate cases**: exact. N=1000/k=12: exact.
- **N=10000/k=30**: numerically equal to `allclose(rtol=1e-6)` but NOT bitwise (max |diff|
  1.9e-6) — vmap-per-row and gather-per-canonical exercise different XLA fusion layouts.
  Consequence: the spec's own `verify_dedup_spec` hook MUST compare numerically
  (`allclose` with policy rtol/atol), never bitwise — consistent with §4.2.5's spot-check
  rationale; now demonstrated for component C too.

**F3 (REQUIRED amendment, construction contract)**: `unique_indices` MUST be the ASCENDING
FIRST-OCCURRENCE POSITIONS of distinct rows, with slot j of `index_map` defined as "the row at
first-occurrence position positions[j]". The docstring's "indices of unique elements" is
ambiguous between value-identities and positions; both satisfy every existing invariant check,
and the wrong reading produces silently WRONG results (demonstrated max error ≈2e5, caught by
nothing). The synthesizer's exact stage builds this convention by construction; a unit AC pinning
it is added to P2: given rows [v0,v1,v0,v1], unique_indices == [0,1] (positions), not [0,1]
(values) — distinguished when first occurrence order differs from sorted-value order.

**F4 (REQUIRED amendment, padding aliasing)**: `to_dedup_gather()` edge-pads `unique_indices`
by REPEATING the last real index. A padded slot k..k_bucket-1 therefore computes a DUPLICATE of
the last canonical row's result. This is safe only because no `index_map` entry selects padded
slots — an invariant the synthesizer must state and preserve (self-assertion already covers
index_map range), but any future custom `gather_fn` or index arithmetic that touches slots ≥ k
silently reads aliased data. Recorded as a documented invariant with a P2 test asserting
padded-slot results are never selected.

Both amendments are additive to §4.3; no prior decision changes.

### 10.3 CLI subprocess acceptance + runtime numerical round (round 3)

- **DedupGather runtime numerics via `axis_dispatch`** (real execution path): semantically
  coherent inputs (duplicate rows → identical payloads) give EXACT equality at k=1, k=N, and
  N=1000/k=12; N=10000/k=30 is numerically equal but not bitwise (max |Δ| 1.9e-6 — different
  XLA fusion layouts between vmap-per-row and gather-per-canonical). `verify_dedup_spec` must
  therefore compare NUMERICALLY (mirrors §4.2.5 spot-check rationale).
- **F3 (construction contract, REQUIRED)**: demonstrated that the docstring's "indices of unique
  elements" is dangerously ambiguous — a value-as-index reading passes every existing
  `__post_init__` invariant yet yields silently wrong results (max error ≈2e5). Pinned contract:
  `unique_indices` = ascending FIRST-OCCURRENCE POSITIONS; slot j ↔ positions[j]. New P2 AC:
  rows [v1,v0,v0,v1] (first-occurrence order ≠ sorted-value order) must produce
  unique_indices [0,1], not [0,1]-by-value.
- **F4 (padding aliasing invariant, REQUIRED)**: edge-padding repeats the LAST real index into
  slots k..k_bucket-1, so padded slots compute a duplicate of the final canonical row. Safe only
  while no index_map entry selects ≥k; documented as an invariant with a P2 test; custom
  gather_fns must preserve it.
- **CLI subprocess acceptance**: real `.venv/bin/xtrax plan --fn module:symbol --shapes ...`
  runs end-user workflows end-to-end (exit 0, human summary with per-axis strategy/reasoning);
  real `xtrax explain --fmt json` emits the machine contract (`_meta.schema_version == 1`,
  axes/strategy_counts/dedup_stats keys) that parses cleanly. Two usability observations for the
  backlog: shapes parser rejects `<float32>` (docs say `<dtype>`; only f32/f64/i32/bool aliases
  accepted) and its error message quotes the angle-bracket form it does not accept.

### 10.4 Round 4 — F3b refinement + remaining seam checks

- **F3b (sharpening of F3)**: JAX device-side gathers SILENTLY CLAMP out-of-range indices
  (demonstrated: `xs[jnp.asarray([9])]` on N=4 returns row 3, no exception). Consequence: an
  invalid `unique_indices` entry never errors at runtime — it silently reads a wrong-but-real
  row. Since `DedupSpec` has no N field and cannot bounds-check positions itself, the
  synthesizer's exact-stage self-assertion MUST include `unique_indices.max() < N` (and
  `min >= 0`) alongside `len(index_map) == N`. Added to P2 AC list.
- **F3 discriminator verified**: rows [9,3,9,3] → corrected convention yields
  unique_indices=[0,1] (first-occurrence POSITIONS); naive value-reading would yield [3,9],
  executing without error but gathering wrong rows entirely ([54,54,54,54] vs
  [6,22,6,22] canonical results).

### 10.5 Round 5 — gate loader, html format, repo test baseline

- **Data-driven probe registration CONFIRMED against the real loader**: appending a
  `[[probes]]` TOML entry (simulating the future memo trace-stability probe) and calling
  `load_performance_targets` yields both probes — no code change, exactly as the spec's P1
  performance-gate plan assumes (`audit/performance_targets.toml` + `performance.py:76`).
- **html CLI format exercised**: `xtrax explain --fmt html --out …` exit 0, writes a real
  55 KB self-contained HTML document (DOCTYPE + embedded SVG). All four documented emit
  formats now observed live except png (guard verified separately).
- **Repo acceptance baseline**: `pytest tests/tiling/test_dedup.py tests/tiling/test_plan.py
  tests/inference -q` → **181 passed** in 10.7s on this working tree. The validation work did
  not disturb existing behavior; the spec's integration claims rest on the same tree that
  passes its own suites.
- Minor API note: `load_performance_targets` requires a `pathlib.Path` (a bare str raises
  AttributeError) — trivially fixed at implementation time; recorded for the P1 planner.

### 10.6 Round 6 — budget bypass observed live, dispatch division verified, png format, full suite

- **R4's budget bypass OBSERVED LIVE**: identical tight MemoryBudget (1 byte) + impossible
  estimate → without a DedupSpec the planner correctly raises `BudgetInfeasibleError`; WITH a
  DedupSpec on the same axis, Phase 0b plans DedupGather and the budget error fires anyway
  naming "final strategies: batch=DedupGather" — i.e., even in budget mode the dedup decision
  escapes estimation and then *fails the whole plan*. This sharpens R4: auto-synthesis under
  budget mode would not merely produce optimistic estimates, it could make previously-feasible
  plans infeasible. Strengthens the case for the deferred budget-integration backlog item.
- **Dispatch division of labor verified**: `make_axis_dispatch(DedupGather)` raises
  `DispatchRejected` ("handled elsewhere") while `axis_dispatch` executes it — the runtime path
  component C relies on is exactly the documented one.
- **png CLI format**: exit 0, writes a real 45 KB PNG. All four emit formats now exercised.
- **Full repo suite**: 205 passed / 1 failed / 1 skipped. The single failure
  (`test_bathos_mcp_reachable_from_a_no_claude_node`) is environmental and unrelated to this
  sprint: it probes the external bathos MCP server, whose entry point fails on a missing
  `cisternal` dependency in ~/projects/bathos. This sprint modified no source files.

### 10.7 Round 7 — remaining inference APIs, SafeMap runtime, fail-loud chain

- **`synthesize_axes(abstract_inputs, overrides=)`**: with overrides produces KNOWN-role,
  correctly-named AxisSpecs (batch/feat) — the exact inputs component C's callers will hold.
  Zero-config `infer_bundle` confirmed UNKNOWN/fail-loud as documented.
- **`emit_ir_schema()`**: zero-arg; emits a valid draft-2020-12 JSON-Schema document
  (`$id: xtrax://composition-ir`). API note: takes no arguments (a BundleSchema arg raises
  TypeError) — P1 planner note for any docs implying otherwise.
- **`verify_against=` purity-guard analog ACCEPTED + DETECTED**: matching concrete inputs pass;
  host-state flip between abstract and concrete passes yields typed `StructureMismatchError`.
  This is the same detect-divergence-fail-loud pattern §4.2's spot-check generalizes — the
  library precedent is real and behaves.
- **SafeMap runtime via `axis_dispatch`**: exact vs vmap at n=12/b=4. Indivisible case confirms
  the full deferred-failure chain: plan-time RuntimeWarning → dispatch-time typed ValueError
  ("safe_map: n=10 is not divisible by batch_size=4") — G4's fail-loud convention verified
  end-to-end across two layers.
