# E1-MVP Backlog DAG (post adversarial plan-audit)

- **task_id:** `260622_e1-signature-inference`
- **date:** 2026-06-23
- **spec:** `.praxia/docs/specs/260623_e1-signature-inference-core-how-should-x.md` (AC1–AC8)
- **provenance:** staff decomposition → adversarial plan-audit (Opus, verified against source) → verdict **NEEDS_WORK** with 8 required fixes (3 critical). This doc is the **revised** DAG with all 8 folded in.

## Audit verdict (summary)
Per-dimension (×/5): decomposability 3, dependency-ordering 4, verification-design 3, risk-mitigation 3, completeness 3. Critical findings, all source-verified:
1. **`BatchPlanner.plan:159` is the `def` line, not an enforcement point** — `plan()`/`_decide_strategy` read only `name, cardinality, default_batch_size, bucket_boundaries, dedup_eligible, heterogeneous`; there is no `role` concept anywhere in `src` (grep: 0 matches). The fail-loud site must be **designed**, inside the per-spec loop (`plan.py` ~180–228), **after** Phase-0/0b pre-demotion, **before** `_decide_strategy`.
2. **`AxisSpec` (plan.py:25-87) is frozen + auto-hashable with a `__post_init__` and a `__getattr__` deprecation shim**; positional sites like `AxisSpec('batch',64,32)` exist. A new field is only safe **appended last** (after `bucket_boundaries`).
3. **Two escalations change E1.3's public surface and must resolve BEFORE coding** (w1.5 gate below). Sentinel-cardinality encoding rejected: it would corrupt `eda/stats.py` (reads `spec.cardinality`) and planner cardinality math.

## Resolved decisions (w1.5 gate — settled here)
- **D1 — UNKNOWN carrier = an explicit `role` field on `AxisSpec`** (an `AxisRole` enum incl. `UNKNOWN`), **appended last** with `default=AxisRole.UNKNOWN`'s safe counterpart. NOTE: default must be a **concrete** role for backward-compat of existing positional/kw sites (they are known-role), with `UNKNOWN` only ever **set by `synthesize_axes`**, never by default construction. Reject sentinel-cardinality (corrupts `eda/stats.py`) and reject a parallel UNKNOWN set (changes `plan()` signature).
- **D2 — AC6 uses an in-test, Protocol-conformant `InputResolver` adapter.** `run/resolver.py:41` ships only a `@runtime_checkable Protocol`; no concrete resolver exists, so building one is out of MVP scope.
- **D3 — `E1.1` (`abstract.py`) raises only stdlib errors** (`ValueError`/`TypeError`) for malformed declarations → genuinely independent of `E1.0` (deps stay `[]`).

## Revised task DAG

| id | size | deps | ACs | test-first | notes (post-audit) |
|----|------|------|-----|-----------|--------------------|
| **E1.0**-pkg-errors | S | — | enables AC3,AC5 | no (smoke import) | new `xtrax.inference` pkg; `AmbiguousAxisError`, `StructureMismatchError`, `AxisRole` enum (incl. `UNKNOWN`). No jaxtyping import. |
| **E1.1**-abstract-inputs | S | — | enables AC1 | yes | `build_abstract_inputs`; **stdlib errors only (D3)** → independent of E1.0. |
| **E1.2**-bundleschema-extractor | M | E1.0,E1.1 | AC1, AC2(struct) | yes | `eval_shape`→`BundleSchema` via `tree_flatten_with_path`/`GetAttrKey` + positional fallback. **+ carry_specs passthrough field & its passthrough test (fix #8).** |
| **E1.3a**-axisspec-role-field | M | E1.2 | (enables AC3) | yes | **SPLIT from E1.3 (fix #4).** Append `role` field LAST on `AxisSpec` + `UNKNOWN` sentinel + `synthesize_axes`/`resolve_axis_role` (Tier-1 hook > Tier-3 UNKNOWN). **NO planner behavior change.** Regression tests: existing positional sites construct unchanged, `AxisSpec` still hashable (set test), `batch_size`/`granularity` deprecation shim still warns (fix #2). |
| **E1.3b**-batchplanner-failloud | M | E1.3a | AC3, AC2(axis-cov) | yes | **KEYSTONE enforcement.** Insert UNKNOWN check in `plan()` per-spec loop, **after Phase-0/0b, before `_decide_strategy`** → `AmbiguousAxisError(axis, how-to-resolve)`. Drop the bogus `:159` cite (fix #1). Red test: UNKNOWN→raises; residues-first axis NEVER auto-batched; existing `tiling/test_plan.py` stays green. |
| **E1.4**-axis-config-override | M | **E1.3a** | AC4 | yes | **dep re-pointed to E1.3a (fix #6)** — needs only the sentinel + `resolve_axis_role` hook, not the planner guard. `@axis_config` sidecar; field-by-field merge. **+ A3 red test: missing `default_batch_size` raises (fix #8).** |
| **E1.5**-purity-guard | M | E1.2 | AC5 | yes | `eval_shape` structure vs concrete batch → `StructureMismatchError`; no-op without sample. Synthetic-divergence fixture (BATHOS discipline). |
| **E1.6**-seam-conformance | M | E1.3b | AC6 | yes | **AC6 via in-test Protocol adapter (D2).** RunSpec.axes / BatchPlanner / FeatureBatch / StageBundle conformance; **no seam signature changes.** |
| **E1.7**-jaxtyping-optional | S | E1.2 | AC7 | yes | jaxtyping-absent leg + static no-import assertion under `src/xtrax/inference`. |
| **E1.8**-parity-oracle | M | E1.4,E1.6 | AC8 | yes | **Named sites + ≥1 non-default flag (fix #9):** prolix `N_ATOMS` (`tile_granularity=128`), `N_CONFORMERS` (`heterogeneous=True`), + a `bucket_boundaries` site. Assert ALL seven `AxisSpec` fields match. |
| **E1.9**-public-api | S | E1.4,E1.5 | integrates AC1-AC5; **OWNS AC2 e2e** | yes | `infer_bundle(fn, abstract_inputs, *, verify_against=None)` + exports. **Owns the zero-config end-to-end AC2 test (fix #5):** no-decorator fn → valid `BundleSchema` AND every input axis present with `role=UNKNOWN`. |
| **E1.10**-docs-changelog | S | E1.9 | — | no | **NEW (fix #8):** document the public `infer_bundle`/`@axis_config` surface + CHANGELOG entry. |
| TBD-T1/T2/T3 | — | — | — | — | Tier-2 jaxtyping adapter / libcst codegen / CarrySpec auto-derivation — gated placeholders, NOT planned (see spec TBDs). |

## Waves (revised)
```
w0:  E1.0 ∥ E1.1
w1:  E1.2
w1.5 DECISION GATE: D1 (role field) + D2 (AC6 adapter) recorded above — unblocks E1.3a
w2:  E1.3a ∥ E1.5 ∥ E1.7
w2b: E1.3b            (immediately after E1.3a; same wave window — the unenforced-UNKNOWN window never reaches a consumer because 3a adds no resolver call sites)
w3:  E1.4 ∥ E1.6
w4:  E1.8 ∥ E1.9
w5:  E1.10
```
**Critical path:** E1.0 → E1.2 → E1.3a → E1.3b → E1.6 → E1.8.
**Coverage:** AC1 (E1.2), AC2 (E1.9 e2e, owned), AC3 (E1.3b), AC4 (E1.4), AC5 (E1.5), AC6 (E1.6), AC7 (E1.7), AC8 (E1.8) — every AC has exactly one owning task.

## Status
Revised DAG addresses all 8 required fixes + the AC8 suggestion. Recommended next: optional re-audit of this revision, then load into the praxia backlog (`depends_on` per the table) and route E1.3a/E1.3b through `spec_driven_dev` with the keystone red tests written first.
