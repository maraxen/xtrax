# Recon memo: runtime compute reuse integration points (260825)

Task: 260825_xtrax-cse-runtime-opt. Read-only recon for components:
(a) jaxpr-level CSE detect/report tool, (b) content-keyed memoization of jitted
inference callables, (c) auto-synthesis of DedupGather/DedupSpec in planner pre-pass.
Repo layout note: package lives in `src/xtrax/`; the campaign loop modules live in a
top-level `controller/` dir (NOT src/xtrax/controller/).

## Component (a): jaxpr-level CSE detection/report tool

- `src/xtrax/inference/__init__.py:9-26` | definition | `__all__` exports
  Public API surface of xtrax.inference is exactly: `AmbiguousAxisError`, `AxisOverride`,
  `AxisRole`, `BundleSchema`, `StructureMismatchError`, `axis_config`, `emit_ir_schema`,
  `infer_bundle`, `synthesize_axes`. A public CSE tool should be added here to match.
- `src/xtrax/inference/schema.py:79` | reference | `jax.eval_shape` trace in BundleSchema extraction
  The ONLY tracing in the package is eval_shape (no jaxpr capture). A CSE tool needs
  `jax.make_jaxpr`/`ClosedJaxpr`, which nothing in src/ currently imports (see Negatives).
- `src/xtrax/eda/explain.py:13` | reference | `explain_plan(plan) -> PlanStatsDict`
  Existing "analyze plan, produce report dict" precedent; CSE report generation should
  mirror this thin-wrapper-over-stats pattern.
- `src/xtrax/eda/stats.py:57-92` | reference | `extract_plan_stats` builds `dedup_stats`/`bucket_stats`
- `src/xtrax/eda/types.py:33` | definition | `DedupStatsEntry(TypedDict)`
  Where new CSE-report fields (dup-eqn classes, est. savings) could surface as TypedDicts.
- `src/xtrax/cli/explain.py:52-110` | reference | `run_explain` pipeline
  Verb pipeline: `load_fn -> parse_shapes -> infer_bundle -> BatchPlanner().plan ->
  explain_plan -> emit(stats, plan, fmt, out)`. A new CSE verb clones this shape;
  AmbiguousAxisError -> CLIError conversion at lines 97-104 is the error-handling idiom.
- `src/xtrax/cli/emit.py:17` | definition | `_VALID_FMTS = {"json","text","html","png"}`
- `src/xtrax/cli/emit.py:20-48` | definition | `emit()` router
  Adding a new report type requires: extend `_VALID_FMTS`, add a branch + `_emit_*`
  helper, and honor the out-path contract (`_emit_render`, lines 113-154: png requires
  --out; html stdout-or-file; confirmations to stderr).
- `src/xtrax/cli/emit.py:50-60` | definition | `_emit_json`
  JSON machine contract: single object, `{"_meta": {"schema_version": 1}, **stats}`.
  Consumers MUST check schema_version first; a new report payload needs its own
  envelope/version decision (bump vs new key) before shipping.
- Tests to extend: `tests/cli/` explain/emit suites exercise the router; CSE report
  should get analogous coverage (not run during this recon per constraints).

## Component (b): content-keyed memoization wrapper

- `src/xtrax/inference/api.py:56` + `src/xtrax/inference/schema.py:34-79` | reference | `infer_bundle`
  Main jitted-adjacent entry point; memo wrapper sits around user inference callables
  that feed this (key = jaxpr hash + pytree structure + leaf content hashes).
- `src/xtrax/sparse/inference.py:146` | definition | `sparse_filter_jit(fn, **kwargs) -> Callable`
  Only existing decorator-style jit-wrapper in the repo; best template for the memo
  wrapper's API surface (decorator returning wrapped Callable, kwargs passthrough).
- `controller/multi_iteration_loop.py:431-476` | other | per-candidate for-loop body
  `for candidate_index in range(max_candidates): ... result = run_one_candidate_pass(...)`
  at line 439 forwards UNCHANGED `concrete_inputs`, `callable_name`, `current_config`,
  `frozen_context` every iteration — exactly the repeated-identical-input regime a
  content-keyed memo hits. Note: candidates normally mutate source between iterations,
  so cache keys including the callable's source/jaxpr hash will mostly MISS by design;
  unchanged-candidate retries (gate failures, smoke reruns) are the true hit case.
- `controller/loop_run.py:404` | definition | `run_campaign_loop`
  Outer wrapper: campaign_create -> run_multi_iteration_loop -> campaign_conclude,
  forwarding the same `dispatch_backend`/inputs through every iteration.
- `controller/main_loop.py:440` | definition | `run_one_candidate_pass`
  Sequences structure_tripwire -> candidate_smoke -> checkified_execution -> dispatch ->
  guarded_evaluate; smoke/checkified steps execute the SAME callable on the SAME
  `concrete_inputs` within one pass (multiple concrete invocations per candidate).
- `src/xtrax/loop/closure_lock.py:233` | definition | `guarded_evaluate(locked, score_raw_artifacts, ...)`
  The guarded evaluation seam. main_loop docs state score_raw_artifacts must NEVER be
  called outside this seam; a memo wrapper placed inside/outside this boundary changes
  drift-detection semantics (see Risks).
- `src/xtrax/run/zarr_integrity.py:65-72` | reference | `update_array_digest(digest, array)`
  Reusable content-hash primitive: canonicalizes then `digest.update(canonical.tobytes(order="C"))`.
  This is the house style for leaf-array hashing; reuse rather than inventing one.
- `src/xtrax/findings.py:37`, `src/xtrax/cli/hash.py:20` | reference | sha256 helpers
  String/canonical-form hashing precedents (findings ids, cli hash verb).
- Cache-invalidation hooks already present: `script_sha256` threading through
  `src/xtrax/loop/prereg_match.py:100-103` shows the repo's existing
  hash-comparison-and-reject pattern a memo layer can imitate for drift rejection.

## Component (c): auto-synthesis of DedupGather/DedupSpec (planner pre-pass)

- `src/xtrax/tiling/plan.py:214-215` | reference | `dedup_by_name = {ds.axis_name: ds ...}`
  Injection point: a synthesis pre-pass populates `dedup_specs` before plan(); dict is
  built inside plan(), so either BatchPlanner gains a hook or callers pass synthesized
  specs. Duplicate axis_name silently last-wins (dict comprehension).
- `src/xtrax/tiling/plan.py:263-278` | definition | Phase 0b block in `BatchPlanner.plan()`
  Exact mechanics: `ds.to_dedup_gather()` -> AxisDecision(spec, batch_size=ds.k,
  reasoning=f"dedup-gather (DedupSpec for '{name}', k=..., k_bucket=...)", strategy=dg);
  `continue` skips UNKNOWN-role guard and standard rules. Synthesized specs flow through
  this unchanged — no plan.py edit strictly required for synthesis itself.
- `src/xtrax/tiling/plan.py:31-52` | definition | `AxisSpec` (class at 31), `dedup_eligible: bool = False` (line 52)
- `src/xtrax/tiling/plan.py:433-438` | other | Rule 2 comment in `_decide_strategy`
  Explicit gap: `dedup_eligible=True` WITHOUT a DedupSpec falls through to cardinality
  rules ("Rule-based dedup_eligible without explicit DedupSpec falls through").
  Component (c) is precisely the feature that closes this gap: detect duplicated rows
  among dedup_eligible axes and synthesize the missing DedupSpec.
- `src/xtrax/tiling/plan.py:120-529` read fully. Remaining structure relevant to (c):
  budget mode defers non-bucket axes to `_plan_joint_budget` (305-413) which starts all
  pending at Vmap and greedily demotes to SafeMap; a synthesized DedupSpec decision
  bypasses budget estimation entirely (Phase 0b `continue`s before pending-append at
  291-294) — i.e. dedup axes are treated as free w.r.t. MemoryBudget. Synthesis must
  account for k_bucket memory or budget estimates go optimistic.
- `src/xtrax/tiling/dedup.py:46-90` | definition | `DedupSpec` frozen dataclass + `__post_init__`
  Invariants: k == len(unique_indices), len(np.unique(index_map)) == k, values in [0,k).
  Any synthesizer MUST satisfy these or construction raises ValueError.
- `src/xtrax/tiling/dedup.py:23-43` | definition | `get_k_bucket(k)` power-of-2 pad
  TODO(high-k) note at lines 29-35 flags waste >256 uniques; synthesis should prefer
  skipping synthesis when k is large relative to N.
- `src/xtrax/tiling/dedup.py:92-118` | definition | `to_dedup_gather()` edge-pads unique_indices
- `src/xtrax/tiling/dedup.py:121-140` | definition | `validate_dedup_carry_names`
- `src/xtrax/tiling/strategy.py:82` | definition | `DedupGather` strategy class
  (`_default_dedup_fn`:48, `_default_gather_fn`:53; protocols DedupFn:19/GatherFn:26).
- `src/xtrax/tiling/dispatch.py:173-178` | reference | dedup Phase 1 / gather Phase 3 in `axis_dispatch`
  Runtime consumer of any synthesized spec; also `make_axis_dispatch`:31, `axis_dispatch`:120.
- DedupSpec CONSTRUCTION SITES TODAY (rg `DedupSpec(`): production code has ZERO.
  Only `tests/tiling/test_plan.py:238,491`, `tests/tiling/test_dedup.py:46,62,74,85,102,117,129,142,157,172`,
  `tests/tiling/test_budget_plan.py:153`, and docs `agent_assets/skills/using-xtrax/references/tiling.md:318,394`.
  Component (c) would be the first production constructor — no existing caller behavior
  to preserve, but also no production hardening exists yet.

## Performance gate wiring (new probe registration)

- `audit/performance_targets.toml:1-9` | config | `[gate]` + `[[probes]]` entries
  Schema `performance-gate-v0`; each probe: `qualname` (module.attr of traced kernel),
  `max_traces` (default `[gate].max_traces_default`=1), optional `trace_probe`
  ("module:function" string exercising it). Registering a CSE/memo probe = ADD A TOML
  ENTRY ONLY — loader is data-driven, no code registry to touch.
- `src/xtrax/devtools/gates/performance.py:42-45` | definition | `ProbeSpec(qualname, max_traces, trace_probe)`
- `src/xtrax/devtools/gates/performance.py:76-128` | definition | `load_performance_targets(path)`
  Validates positive ints, emit_probe_record bool; unknown keys tolerated. Entry point:
  `scripts/audit_performance_gate.py:23 main(argv)` takes `--path` (default targets file).
  Also wired from `src/xtrax/devtools/bootstrap.py`.
- `src/xtrax/devtools/gates/_performance_probes.py:28-44` | definition | probe kernel + probe fn pattern
  Kernel: `@sparse_filter_jit @chex.assert_max_traces(n=1)`; probe fn signature is
  `probe_x(guarded: Callable[..., Any]) -> None` calling `guarded(...)` TWICE on stable
  inputs. A memo-wrapper probe follows identically (assert traces stay at 1 across two
  identical calls proves memo hit); a CSE-tool probe has no compile count to guard and
  likely belongs in a different gate (correctness/runtime, not trace-count).
- `src/xtrax/devtools/gates/_performance_probes.py:45-49` | definition | `probe_stable_jnp_kernel`
  Minimal stable-input probe used by integration tests — simplest template.

## HostPrepGraph node-collapse feasibility

- `src/xtrax/composition/graph.py:29-55` | definition | `HostPrepGraphNode(id, callable_ref, metadata, frozen)`
  Nodes wrap LIVE Python callables (docstring lines 4-6: serialization is T1-08's job,
  explicitly not done). There is no structural equality: two nodes with different ids
  wrapping the same function object are NOT detected as duplicates anywhere.
- `src/xtrax/composition/graph.py:77-88` | definition | `HostPrepGraph.__post_init__`
  Validates unique node IDS and edge endpoint existence only. Graph-level collapse of
  identical callables is therefore POSSIBLE but requires introducing an equivalence key
  (e.g. callable qualname + metadata digest); nothing today computes one. `frozen` flag
  guards mutation, not dedup; metadata is validated against node_metadata_schema.toml
  (`xtrax.composition.node_metadata`) so a metadata-derived hash is well-defined.
- Implication for (b)/(c): memoization at graph level would need a pre-topology pass
  rewriting edges after collapse; FrozenNodeError (`composition/errors.py`) fires on any
  post-construction attribute mutation, so collapse must happen at graph construction.

## Loop call patterns hitting a memo wrapper

- `controller/multi_iteration_loop.py:439` — N identical-arg `run_one_candidate_pass`
  calls (unchanged frozen_context/current_config/concrete_inputs).
- `controller/main_loop.py` — within ONE pass: candidate_smoke + checkified_execution +
  guarded_evaluate each execute the candidate callable on `concrete_inputs`.
- `controller/loop_run.py:404` `run_campaign_loop` — wraps the above per campaign;
  `LoopEvent`/`CampaignLoopResult` at 286-338 are the telemetry shapes a memo-hit
  counter would report through.

## Verified negatives (rg evidence)

- "No jaxpr introspection anywhere in src/: verified by `rg -ni 'jaxpr' src/ --type py`" -> NO MATCH.
  Component (a) is greenfield; no ClosedJaxpr/make_jaxpr imports exist.
- "No eqn-level hashing/CSE: verified by `rg -ni 'eqn' src/ --type py | rg -i 'hash|fingerprint|equiv|cse'`" -> NO MATCH.
- "No lru/content caching of jitted fns: verified by `rg -n 'lru_cache|functools\\.cache|cache\\(' src/ --type py`" -> NO MATCH in src/.
- "No row-dedup detection/producer: verified by `rg -n 'unique_indices|index_map' src/` excluding tiling/dedup.py + tiling/strategy.py" -> only consumers: `src/xtrax/tiling/dispatch.py:173-178` (runtime gather/scatter) and a comment `src/xtrax/tiling/plan.py:434`. Nothing computes uniqueness from batch rows.
- "No content-keyed input memoization: all hashlib uses are sidecars/manifests/integrity:
  `src/xtrax/findings.py:37`, `src/xtrax/cli/hash.py:20`, `src/xtrax/run/zarr_integrity.py:65-121`,
  `src/xtrax/run/component_binding.py:93`, `src/xtrax/loop/*.py` gates" -> none keyed on
  jitted-callable inputs/pytree leaves.
- "No production DedupSpec constructor: verified by `rg -n 'DedupSpec\\('`" -> tests + skill docs only.

## Integration risks

1. Component (a) adds the repo's first jaxpr introspection; tracing arbitrary loaded fns
   (cli/loader path) can trigger re-tracing side effects and chex trace-guards elsewhere.
2. Memo wrapper must not cross the guarded_evaluate seam (closure_lock drift detection
   treats out-of-seam evaluation as HALT signal); cache placement is a correctness issue.
3. Leaf content hashing requires device->host copies (tobytes precedent); for large N the
   hash cost can exceed recomputation — needs size threshold policy.
4. Synthesized DedupSpec decisions bypass MemoryBudget estimation (Phase 0b continues
   before joint-budget demotion), so auto-synthesis can silently invalidate budget mode.
5. dedup_by_name dict silently last-wins on duplicate axis_name (plan.py:215); synthesis
   colliding with caller-declared specs would override explicit user intent undetected.
