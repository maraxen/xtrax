# E1 — Signature-Inference Core (Epic Decomposition)

- **task_id:** `260622_e1-signature-inference`
- **date:** 2026-06-22
- **status:** grounded; pre-brainstorm (decomposition for review — no brainstorm/impl yet)
- **mission anchor:** `.praxia/mission.md` composition surface item 1 ("Auto-inference bundle"); backlog epic **#2174**
- **grounding:** internal recon (E1 seams) + external prior-art research (Tyro / jaxtyping / jax.export / precedent) — both logged under this task_id

---

## 0. Thesis

The pure typed JAX function is the **single source of truth**. From one annotated function we *derive* — and keep *inspectable* — everything downstream:

```
typed pure fn  ──▶  Bundle pytree schema          (what flows)
   (jaxtyping  ──▶  AxisSpec roles → BatchPlan     (how it tiles)   ← the moat
    + return  ──▶  Tyro-backed CLI verbs           (how it's run)
    type)     ──▶  jax.export artifact             (how it ships)
```

E1 is the **keystone**: E2 (CLI), E3 (host-prep), and E4 (chain-map) all consume its derived schema. It directly removes xtrax's biggest ergonomic tax — hand-declaring `AxisSpec` (found hand-written 16+ times across tests/src/docs, e.g. `tests/tiling/test_plan.py:22`).

---

## 1. Resolved design forks (evidence-bound)

### Fork A — Inference is a *runtime* problem; AST/CST is a *codegen* problem. **Do not conflate.**
- **Inference of structure/shape → `jax.eval_shape`** (zero-FLOP abstract trace → output pytree of `ShapeDtypeStruct`). This is JAX's canonical shape oracle. Static AST cannot reliably read runtime types (forward refs, generics, dynamically-built unions).
- **Emission of code → libcst** (already a dev dep): generate the Bundle class, CLI module, and `.pyi` stubs from the inferred schema *without* importing/running user code.
- **Rule pinned:** runtime introspection IN, CST codegen OUT. Greenfield confirmed — `jax.eval_shape`, `jax.export`, `inspect.signature`, `get_type_hints`, `libcst` are **not used anywhere in `src/`** today.

### Fork B — jaxtyping carries axis *roles*, NOT pytree *structure*. (Correction to first framing.)
Research confirmed precise capability boundaries:
- **jaxtyping CAN:** per-leaf shape/dtype (`Float[Array, "batch dim"]`), structural name-binding across a call (`PyTree[Float, "T"]`), wildcard `...`, per-leaf `?foo`.
- **jaxtyping CANNOT:** name dataclass fields in an output type; annotate a whole pytree class as batched (`Shaped[Bundle, "batch"]` — confirmed broken, [jaxtyping#242](https://github.com/patrick-kidger/jaxtyping/issues/242)); expose dim-name bindings via any public API after a call. It also **does not check return types** at all.
- **Therefore the division of labor is:**
  - **Bundle structure + field names** ← `jax.eval_shape` output pytree + the return type's own fields (dataclass / `eqx.Module` / **`register_pytree_with_keys`** keyed leaves). NOT from jaxtyping.
  - **Axis roles** (which axis is batch / dedup-eligible / sequence) ← parse the jaxtyping **dim-name strings** ourselves (recover from the annotation object / `Annotated.__metadata__` via `get_type_hints`), then map names → roles via a lookup table.

### Fork C — Auto-CLI: delegate parsing to **Tyro**; xtrax owns the **verbs**.
- **Tyro** (brentyi) is the de-facto JAX-community standard: v1.0 stable (Apr 2026), native `typing.Annotated` / dataclass / `Union`→subcommand / `Literal`→choices. jsonargparse is PyTorch/Lightning-world; draccus is smaller/less proven.
- We do **not** build a parser (that would violate xtrax's "delegation, not reimplementation" discipline). Tyro binds args; xtrax supplies the experiment-lifecycle verb set: **`plan` / `explain` / `sweep` / `run` / `resume` / `export`**, wired to the existing BatchPlan / EDA / `ResumableState` + `PreemptionHandler` surface. That CLI is also the machine interface an autoresearch orchestrator drives.

---

## 2. E1 inference pipeline (target design)

```
@xtrax.signature  (thin decorator; optional @axis_config for non-annotatable knobs)
        │
        ▼
1. get_type_hints(fn)          → recover param/return annotations (incl. jaxtyping dim strings)
2. build abstract inputs        → jax.ShapeDtypeStruct per param from declared shapes/dims
3. jax.eval_shape(fn, *abstract)→ output pytree structure + per-leaf ShapeDtypeStruct  (Bundle schema)
4. dim-name → AxisSpec role     → parse dim strings, map via role table → AxisSpec(name, cardinality, …)
5. assemble                     → RunSpec.axes (list[AxisSpec]) + Bundle schema → BatchPlanner
        │
        ▼
   (optional) jax.export.export → portable artifact w/ symbolic dims for variable axis sizes
```

**Integration seams (recon, file:line):**
| Seam | Location | E1's role |
|------|----------|-----------|
| `AxisSpec` (hand-built 16×) | `src/xtrax/tiling/plan.py:26` | auto-derive instead of hand-write |
| `TransformFn` / `RollingFn` | `src/xtrax/stages/protocols.py:19,29` | the functions E1 introspects |
| `FeatureBatch` / `InputResolver` | `src/xtrax/run/resolver.py:21,41` | Bundle schema must conform |
| `RunSpec.axes` | `src/xtrax/run/spec.py:17` | E1 populates this list |
| `StageBundle` (`Optional[Callable]` fields) | `src/xtrax/stages/bundle.py:22` | derived bundle subclasses it |
| `CarrySpec` (axis_name ↔ AxisSpec.name) | `src/xtrax/tiling/carry.py:22` | auto-derive from `RollingFn` carry? (open) |
| jaxtyping annotations (data source) | `src/xtrax/safety/ops.py:2`, `devtools/_beartype_probe.py:5` | dim-string extraction |

---

## 3. Open questions → these seed the BRAINSTORM phase (next, not now)

From recon gaps — the forks the contemplex brainstorm should resolve before speccing:
1. **dim-name → AxisSpec.name mapping.** `Float[Array, "batch dim"]` → `name="batch"`? First-token heuristic vs. full string vs. user-supplied role table. **Recommend: explicit role table, dim-name as key.**
2. **cardinality binding.** `eval_shape` gives shape, not the *total to tile*. Source: runtime input length? declared via `@axis_config`? symbolic dim from `jax.export`? (Likely: symbolic/declared, resolved at run time.)
3. **non-annotatable knobs.** `default_batch_size`, `tile_granularity`, `heterogeneous`, `dedup_eligible` have no annotation carrier → companion `@axis_config(...)` decorator or config map. Design the decorator surface.
4. **Bundle structure recovery + field names.** Confirm `eval_shape` + `register_pytree_with_keys` (or dataclass/`eqx.Module` fields) recovers named fields; verify exact jaxtyping dim-string introspection API by test.
5. **CarrySpec auto-derivation** from `RollingFn` carry pytree — in scope for E1 or deferred?
6. **Bundle emission** — does E1 return a schema object, or codegen a `StageBundle` subclass (libcst)? MVP likely returns schema; codegen is a follow-on.

---

## 4. Prior art & novelty

| Tool | Relevance | Verdict |
|------|-----------|---------|
| **Tyro** | typed CLI from dataclasses/signatures | **delegate to it** (Fork C) |
| jsonargparse / draccus | typed CLI | not chosen (Lightning-world / smaller) |
| **jax.eval_shape** | abstract output pytree + ShapeDtypeStruct | **core primitive** |
| **jax.export.symbolic_shape** | dynamic/symbolic dims for portable artifact | export leg; eq-comparison unsound (gotcha) |
| Penzai named-axes | runtime axis-name inference from `named_shape` | closest analog — but **runtime arrays, not annotations** |
| Fiddle / gin-config | config wiring / DI | no axis/shape awareness |
| `register_pytree_with_keys` (JAX core) | field-level string keys through flatten | **candidate** to close the named-field gap jaxtyping can't |

**Novelty:** deriving **batching-axis roles from type-annotation dim-names has no prior art** in any surveyed JAX/Python library. This is the defensible core. (Confidence: high.)

**Key gotchas to bake into the spec:** `eval_shape` needs pure `tree_unflatten` (side effects break abstract trace); symbolic-dim equality is unsound; jaxtyping must stay an *optional* refinement, never a hard dependency of inference.

---

## 5. Epic DAG

```
#2174 Pure-JAX composition layer
  └─ E1  Signature-inference core      ◀── KEYSTONE (this doc)
       ├─ E2  Auto-CLI (Tyro verbs)        depends: E1 schema for plan/run
       ├─ E3  Host-prep composition        depends: E1 boundary contract
       └─ E4  Chain-map UI (research)       depends: E2 graph serialization; later
#2175 Agent-composition tooling (research)  ── parallel track
```

**E1 sub-tasks (provisional, to be confirmed post-brainstorm):**
- E1.1 — abstract-input builder (shape/dim spec → `ShapeDtypeStruct` pytree)
- E1.2 — `eval_shape` Bundle-schema extractor (output pytree + field names via keyed pytrees)
- E1.3 — dim-name parser + role table → `AxisSpec` synthesis
- E1.4 — `@signature` / `@axis_config` decorator surface
- E1.5 — `RunSpec.axes` auto-population + BatchPlanner handoff
- E1.6 — (optional) `jax.export` artifact with symbolic dims
- E1.7 — parity tests vs. the 16 existing hand-written `AxisSpec` sites (the regression oracle)

---

## 6. Risks
- **Creep into "framework."** Each derived surface stays a thin, independently-testable primitive. Mitigation: E1 returns inspectable data, no hidden magic.
- **`eval_shape` purity requirement.** Document + test against custom pytree nodes; fail loud on impure `tree_unflatten`.
- **jaxtyping introspection fragility.** Pin the exact dim-string extraction mechanism with a contract test before building on it; degrade gracefully when annotations absent.
- **Cardinality is not statically knowable.** The hardest semantic gap — resolved via declared/symbolic dims, not invented.

---

## 7. Proposed `mission.md` sharpenings (for review — NOT yet applied)
1. Item 1 mechanism → "**runtime introspection (`eval_shape`/`export`) for inference; libcst for codegen**" (lock the boundary).
2. Add objective: "**jaxtyping dim-names → `AxisSpec` roles**" as the type→planner bridge (the novel keystone).
3. Auto-CLI objective → "**delegate parsing to Tyro; xtrax owns `plan/explain/sweep/run/resume/export` verbs**."

---

## 8. Next steps (sequenced)
1. **Brainstorm** (contemplex) the §3 open forks — resolve dim-name→role, cardinality binding, decorator surface, Bundle emission. *(awaiting go-ahead)*
2. **Spec** E1 (specification-specialist) with **adversarial spec review** (challenger/defender) — it's load-bearing.
3. **Backlog**: add E1.1–E1.7 under #2174 with `depends_on` DAG; emit a sprint once E1 is spec-locked.
4. Apply §7 mission sharpenings on approval.
