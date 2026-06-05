// Sprint 1 runner — emitted by `praxia dw emit-sprint`
// Source: .praxia/sprint_plans/sprint_plan.toml
// Regenerate: praxia dw emit-sprint sprint_plan.toml
// task_id: 260605_xtrax-s1-foundation   sprint_id: 1
//
// RACE SAFETY (memory: parallel fixers race on git-status scope checks in praxia):
//   the writing chain (A,B,C,D,G,E,F,H) runs STRICTLY SEQUENTIAL —
//   exactly one fixer touches the working tree at a time.

export const meta = {
  name: "260605_xtrax-s1-foundation",
  description: "Scaffold, safe_map/safe_scan transforms, and full tiling layer (AxisStrategy, BatchPlanner, Iterators, Dedup, Dispatch)",
  phases: [
    { title: "Track A — Phase 0.1: Scaffold (#1128)" },
    { title: "Track B — Phase 1.1: safe_map (#1129)" },
    { title: "Track C — Phase 1.2: safe_scan (#1130)" },
    { title: "Track D — Phase 2.1: AxisStrategy types (#1131)" },
    { title: "Track G — Phase 2.4: Dedup (#1134)" },
    { title: "Track E — Phase 2.2: BatchPlan / BatchPlanner (#1132)" },
    { title: "Track F — Phase 2.3: Iterators (#1133)" },
    { title: "Track H — Phase 2.5: make_axis_dispatch (#1135)" },
  ],
};

const SPRINT_ID = "260605_xtrax-s1-foundation";
const TASK_ID = "260604_xtrax-shape";
const MAX_FIX_RETRIES = 2;


const VERDICT_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["item_id", "verdict", "summary"],
  properties: {
    item_id: { type: "string" },
    verdict: { type: "string", enum: ["PASS", "NEEDS_WORK", "FAIL"] },
    summary: { type: "string" },
    issues: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["where", "problem", "fix"],
        properties: {
          where: { type: "string" },
          problem: { type: "string" },
          fix: { type: "string" },
        },
      },
    },
  },
};

// Shared context for the writing tracks (from recon, task 260605_xtrax-s1-foundation).
const EMITTER_CTX = `xtrax is a domain-agnostic, high-performance composable JAX library (Python 3.13+) that sits above JAX and Equinox. Spec at .praxia/docs/specs/260604_xtrax-spec.md (oracle-approved R3).\n\nKey invariants for all implementations:\n- jax>=0.4.36 required (lax.map batch_size= arg)\n- equinox>=0.11.0, optax>=0.2.3, orbax-checkpoint>=0.6.0, grain>=0.2.0\n- No Python-loop hot paths — use jax.tree.map, lax.scan, lax.map, optax.chain\n- uv run python / uv run pytest — never bare python/pytest\n- ruff format + ruff check before every commit\n- All files under src/xtrax/ using src layout; tests mirror under tests/\n`;

// ---- per-track stage helpers ---------------------------------------------
const fixer = (prompt, label, phaseName) =>
  agent(prompt, {
    agentType: "fixer",
    label,
    phase: phaseName,
  });

const reviewer = (itemId, prompt, label, phaseName) =>
  agent(`Backlog item: ${itemId}\n${prompt}`, { agentType: "reviewer", label, phase: phaseName, schema: VERDICT_SCHEMA });

// Sequential implement->review with bounded NEEDS_WORK repair cycles.
async function track(itemId, phaseName, fixerPrompt, reviewerPrompt) {
  log(`[${itemId}] implement`);
  await fixer(fixerPrompt, `fix:${itemId}`, phaseName);
  let verdict = await reviewer(itemId, reviewerPrompt, `review:${itemId}`, phaseName);
  for (let retry = 0; retry < MAX_FIX_RETRIES && verdict && verdict.verdict === "NEEDS_WORK"; retry++) {
    log(`[${itemId}] NEEDS_WORK — repair cycle ${retry + 1}/${MAX_FIX_RETRIES}`);
    const issues = (verdict.issues || [])
      .map((i) => `- ${i.where}: ${i.problem} -> ${i.fix}`)
      .join("\n");
    await fixer(
      `${fixerPrompt}\n\nA reviewer found issues — fix exactly these, nothing else:\n${issues}`,
      `fix:${itemId}:repair:${retry}`,
      phaseName
    );
    verdict = await reviewer(itemId, reviewerPrompt, `review:${itemId}:re:${retry}`, phaseName);
  }
  return verdict;
}

// ===== TRACK A — Track A — Phase 0.1: Scaffold (#1128) =========================
const trackA = () =>
  track(
    "1128",
    "Track A — Phase 0.1: Scaffold (#1128)",
    `task_id: ${TASK_ID}\nProject root: /home/marielle/projects/xtrax\n\nReplace the blank scaffold with the full xtrax package structure.\n\n1. Replace pyproject.toml entirely:\n[project]\nname = "xtrax"\nversion = "0.1.0"\ndescription = "High-performance composable JAX library"\nreadme = "README.md"\nrequires-python = ">=3.13"\ndependencies = [\n    "jax>=0.4.36", "jaxlib>=0.4.36", "equinox>=0.11.0",\n    "optax>=0.2.3", "orbax-checkpoint>=0.6.0", "grain>=0.2.0", "numpy>=1.26",\n]\n[project.optional-dependencies]\ndev = ["pytest>=8.0","pytest-asyncio>=0.23","ruff>=0.4.0","pyright>=1.1.360","jaxtyping>=0.2.28"]\n[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n[tool.hatch.build.targets.wheel]\npackages = ["src/xtrax"]\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\nasyncio_mode = "auto"\n[tool.ruff]\ntarget-version = "py313"\n[tool.ruff.lint]\nselect = ["E","F","I","UP"]\n\n2. Create src/xtrax/__init__.py with: __version__ = "0.1.0"\n\n3. Create empty __init__.py stubs in: src/xtrax/transforms/, src/xtrax/tiling/, src/xtrax/safety/, src/xtrax/stages/, src/xtrax/training/, src/xtrax/distributed/, src/xtrax/checkpoint/, src/xtrax/engine/, src/xtrax/data/\n\n4. Create stub .py files (content: "# stub") in each module under each subpackage:\n   transforms/: map.py, scan.py\n   tiling/: strategy.py, plan.py, iterator.py, dedup.py, dispatch.py\n   safety/: ops.py, manager.py, preemption.py\n   stages/: protocols.py, bundle.py\n   training/: types.py, trainer.py, step.py, loss.py, optim.py, grad.py\n   distributed/: sharding.py, init.py\n   checkpoint/: orbax.py\n   engine/: io.py, engine.py\n   data/: module.py, pipeline.py\n\n5. Create tests/__init__.py and matching test stub directories/files mirroring src/xtrax/ layout (tests/transforms/test_map.py etc, all containing "# tests").\n\n6. Create ruff.toml:\ntarget-version = "py313"\n[lint]\nselect = ["E","F","I","UP"]\n\nGate: \`uv run ruff check .\` exits 0; \`uv run pytest --collect-only\` exits 0 (0 tests).\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify scaffold:\n1. pyproject.toml has jax>=0.4.36, hatchling build backend, packages=["src/xtrax"]\n2. src/xtrax/ has 9 subpackage __init__.py files\n3. All stub module files exist\n4. tests/ mirror exists with __init__.py files\n5. ruff.toml exists\n6. Run: \`uv run ruff check .\` — must exit 0\n7. Run: \`uv run pytest --collect-only\` — must exit 0\n`,
  );

// ===== TRACK B — Track B — Phase 1.1: safe_map (#1129) =========================
const trackB = () =>
  track(
    "1129",
    "Track B — Phase 1.1: safe_map (#1129)",
    `task_id: ${TASK_ID}\nFile: src/xtrax/transforms/map.py | Test: tests/transforms/test_map.py\n\nImplement safe_map in src/xtrax/transforms/map.py:\n\n\`\`\`python\nfrom collections.abc import Callable\nfrom typing import TypeVar\nimport jax\n\nPyTree = TypeVar("PyTree")\n\ndef safe_map(fn: Callable, xs: PyTree, batch_size: int | None = None) -> PyTree:\n    n = jax.tree.leaves(xs)[0].shape[0]\n    if batch_size is None or n <= batch_size:\n        return jax.vmap(fn)(xs)\n    if n % batch_size != 0:\n        raise ValueError(\n            f"safe_map: n={n} is not divisible by batch_size={batch_size}. "\n            "Ensure cardinality is a multiple of batch_size."\n        )\n    return jax.lax.map(fn, xs, batch_size=batch_size)\n\`\`\`\n\nUpdate src/xtrax/transforms/__init__.py:\n\`\`\`python\nfrom xtrax.transforms.map import safe_map\n__all__ = ["safe_map"]\n\`\`\`\n\nWrite tests/transforms/test_map.py with tests for:\n- batch_size=None → vmap path (identity fn, compare with jax.vmap)\n- n<=batch_size → vmap path\n- n>batch_size, divisible → lax.map path, result matches vmap\n- n>batch_size, non-divisible → raises ValueError matching "not divisible"\n- pytree xs (dict with "a","b" keys)\n- shape invariant (output shape == input shape)\n\nGate: \`uv run pytest tests/transforms/test_map.py -v\` all pass; \`uv run ruff check src/xtrax/transforms/map.py\` clean.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify safe_map:\n1. batch_size=None or n<=batch_size → jax.vmap (NOT lax.map)\n2. n>batch_size AND divisible → jax.lax.map(fn, xs, batch_size=batch_size)\n3. n>batch_size AND non-divisible → ValueError with "not divisible"\n4. No Python loop in implementation\n5. \`uv run pytest tests/transforms/test_map.py -v\` — all tests pass\n6. \`uv run ruff check src/xtrax/transforms/map.py\` — clean\n`,
  );

// ===== TRACK C — Track C — Phase 1.2: safe_scan (#1130) =========================
const trackC = () =>
  track(
    "1130",
    "Track C — Phase 1.2: safe_scan (#1130)",
    `task_id: ${TASK_ID}\nFile: src/xtrax/transforms/scan.py | Test: tests/transforms/test_scan.py\n\nImplement safe_scan in src/xtrax/transforms/scan.py:\n\n\`\`\`python\nfrom collections.abc import Callable\nfrom typing import TypeVar\nimport jax\n\nCarry = TypeVar("Carry")\nX = TypeVar("X")\nY = TypeVar("Y")\n\ndef safe_scan(\n    fn: Callable[[Carry, X], tuple[Carry, Y]],\n    init: Carry,\n    xs: X,\n    length: int | None = None,\n    reverse: bool = False,\n    unroll: int | bool = 1,\n) -> tuple[Carry, Y]:\n    effective_length = length\n    if effective_length is None:\n        leaves = jax.tree.leaves(xs)\n        effective_length = leaves[0].shape[0] if leaves else 0\n    if effective_length == 0:\n        raise ValueError("safe_scan: xs has leading axis of length 0.")\n    return jax.lax.scan(fn, init, xs, length=length, reverse=reverse, unroll=unroll)\n\`\`\`\n\nCRITICAL: Do NOT add \`if init is None: raise\`. None is a legal lax.scan carry.\n\nUpdate src/xtrax/transforms/__init__.py to also export safe_scan.\n\nWrite tests/transforms/test_scan.py with tests for:\n- normal scan (compare with jax.lax.scan)\n- empty xs (length=0 inferred) → ValueError\n- length=0 explicit → ValueError\n- reverse=True\n- unroll>1 (same result as unroll=1)\n- init=None is legal (must not raise)\n\nGate: \`uv run pytest tests/transforms/test_scan.py -v\` all pass; \`uv run ruff check src/xtrax/transforms/scan.py\` clean.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify safe_scan:\n1. length==0 or inferred xs leading-axis==0 → ValueError before tracing\n2. init=None is NOT rejected (legal lax.scan carry) — test_init_none_is_legal must pass\n3. Delegates to jax.lax.scan with all kwargs passed through\n4. No \`init is None\` check anywhere in the code\n5. \`uv run pytest tests/transforms/test_scan.py -v\` — all tests pass\n6. \`uv run ruff check src/xtrax/transforms/scan.py\` — clean\n`,
  );

// ===== TRACK D — Track D — Phase 2.1: AxisStrategy types (#1131) =========================
const trackD = () =>
  track(
    "1131",
    "Track D — Phase 2.1: AxisStrategy types (#1131)",
    `task_id: ${TASK_ID}\nFile: src/xtrax/tiling/strategy.py | Test: tests/tiling/test_strategy.py\n\nImplement AxisStrategy sealed union in src/xtrax/tiling/strategy.py:\n\n\`\`\`python\nfrom __future__ import annotations\nfrom collections.abc import Callable\nfrom dataclasses import dataclass\nfrom typing import Any, Protocol, runtime_checkable\n\n@runtime_checkable\nclass ScanTransition(Protocol):\n    def __call__(self, carry: Any, x: Any) -> tuple[Any, Any]: ...\n\n@runtime_checkable\nclass DedupFn(Protocol):\n    def __call__(self, xs: Any) -> tuple[Any, Any]: ...\n\n@runtime_checkable\nclass GatherFn(Protocol):\n    def __call__(self, ys: Any, gather_indices: Any) -> Any: ...\n\n@dataclass(frozen=True)\nclass Vmap:\n    pass\n\n@dataclass(frozen=True)\nclass SafeMap:\n    batch_size: int\n\n@dataclass(frozen=True)\nclass Scan:\n    # NO batch_size field — intentional\n    transition: ScanTransition\n\n@dataclass(frozen=True)\nclass DedupGather:\n    dedup_fn: DedupFn\n    gather_fn: GatherFn\n    k_bucket: int  # metadata only\n\nAxisStrategy = Vmap | SafeMap | Scan | DedupGather\n\`\`\`\n\nWrite tests/tiling/test_strategy.py verifying: all 4 variants instantiate; Scan has no batch_size field; all are frozen (mutation raises); isinstance checks pass.\n\nGate: \`uv run pytest tests/tiling/test_strategy.py -v\` all pass; \`uv run ruff check src/xtrax/tiling/strategy.py\` clean.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify AxisStrategy:\n1. Vmap: frozen, no fields\n2. SafeMap: frozen, batch_size: int\n3. Scan: frozen, transition field ONLY — NO batch_size field (critical)\n4. DedupGather: frozen, dedup_fn + gather_fn + k_bucket\n5. All are frozen dataclasses (mutation raises)\n6. \`uv run pytest tests/tiling/test_strategy.py -v\` — all tests pass\n7. \`uv run ruff check src/xtrax/tiling/strategy.py\` — clean\nConfirm: \`hasattr(Scan(transition=lambda c,x:(c,x)), "batch_size")\` is False.\n`,
  );

// ===== TRACK G — Track G — Phase 2.4: Dedup (#1134) =========================
const trackG = () =>
  track(
    "1134",
    "Track G — Phase 2.4: Dedup (#1134)",
    `task_id: ${TASK_ID}\nFile: src/xtrax/tiling/dedup.py | Test: tests/tiling/test_dedup.py\n\nImplement in src/xtrax/tiling/dedup.py:\n\n\`\`\`python\nfrom dataclasses import dataclass\n\ndef get_k_bucket(n: int) -> int:\n    # Mandated: integer bit_length — NOT 2**ceil(log2(n))\n    if n <= 0:\n        raise ValueError(f"get_k_bucket: n must be > 0, got {n}")\n    return 1 << (n - 1).bit_length()\n\n@dataclass(frozen=True)\nclass DedupSpec:\n    k_bucket: int\n    max_unique: int\n    def __post_init__(self) -> None:\n        if self.k_bucket <= 0 or (self.k_bucket & (self.k_bucket - 1)) != 0:\n            raise ValueError(f"DedupSpec: k_bucket={self.k_bucket} must be a positive power of 2.")\n        if self.k_bucket < self.max_unique:\n            raise ValueError(f"DedupSpec: k_bucket={self.k_bucket} must be >= max_unique={self.max_unique}.")\n\`\`\`\n\nWrite tests verifying: get_k_bucket(7)==8, get_k_bucket(8)==8, get_k_bucket(9)==16, get_k_bucket(1)==1, get_k_bucket(0) raises, get_k_bucket(-1) raises; DedupSpec valid; k_bucket not power-of-2 raises; k_bucket < max_unique raises; k_bucket == max_unique valid.\n\nGate: \`uv run pytest tests/tiling/test_dedup.py -v\` all pass; \`uv run ruff check src/xtrax/tiling/dedup.py\` clean.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify Dedup:\n1. get_k_bucket uses \`1 << (n-1).bit_length()\` — verify no log2 or ** in implementation\n2. get_k_bucket(7)==8, (8)==8, (9)==16, (1)==1\n3. get_k_bucket(0) and (-1) raise ValueError\n4. DedupSpec: k_bucket not power-of-2 raises; k_bucket < max_unique raises; k_bucket == max_unique is valid\n5. \`uv run pytest tests/tiling/test_dedup.py -v\` — all tests pass\n6. \`uv run ruff check src/xtrax/tiling/dedup.py\` — clean\n`,
  );

// ===== TRACK E — Track E — Phase 2.2: BatchPlan / BatchPlanner (#1132) =========================
const trackE = () =>
  track(
    "1132",
    "Track E — Phase 2.2: BatchPlan / BatchPlanner (#1132)",
    `task_id: ${TASK_ID}\nFile: src/xtrax/tiling/plan.py | Test: tests/tiling/test_plan.py\n\nImplement BatchPlan and BatchPlanner in src/xtrax/tiling/plan.py.\n\nSelection rules (in priority order):\n1. spec.dedup_eligible → DedupGather (with placeholder dedup/gather fns)\n2. spec.cardinality <= self.batch_size → Vmap()\n3. cardinality % batch_size == 0 → SafeMap(batch_size)\n4. non-divisible → SafeMap(batch_size) + warnings.warn with "not divisible" in message\n   DEFERRED-FAILURE CONTRACT: this plan will raise ValueError at make_axis_dispatch time.\n\nScan is NEVER returned by plan(). Pure Python — no jax tracing except optional memory_estimator device query.\n\nmemory_estimator: when provided, returns estimated bytes for Vmap. If > device limit (default 4 GiB from jax.devices()[0].memory_stats()), prefer SafeMap. If query fails, fall back silently.\n\nAxisSpec: cardinality: int, batch_size: int, dedup_eligible: bool = False\nAxisDecision: spec: AxisSpec, strategy: AxisStrategy\nBatchPlan: decisions: list[AxisDecision]\n\nWrite tests verifying all 4 rules, Scan-never-returned for all cardinalities, length matches specs, memory_estimator override.\n\nGate: \`uv run pytest tests/tiling/test_plan.py -v\` all pass; \`uv run ruff check src/xtrax/tiling/plan.py\` clean.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify BatchPlanner:\n1. Rule 1: dedup_eligible=True → DedupGather\n2. Rule 2: cardinality<=batch_size → Vmap\n3. Rule 3: divisible → SafeMap(batch_size)\n4. Rule 4: non-divisible → SafeMap + warnings.warn (deferred-failure contract in code comment)\n5. Scan NEVER returned — test for multiple cardinalities\n6. memory_estimator override tested\n7. plan() creates no JAX-traced values\n8. \`uv run pytest tests/tiling/test_plan.py -v\` — all tests pass\n9. \`uv run ruff check src/xtrax/tiling/plan.py\` — clean\n`,
  );

// ===== TRACK F — Track F — Phase 2.3: Iterators (#1133) =========================
const trackF = () =>
  track(
    "1133",
    "Track F — Phase 2.3: Iterators (#1133)",
    `task_id: ${TASK_ID}\nFile: src/xtrax/tiling/iterator.py | Test: tests/tiling/test_iterator.py\n\nImplement in src/xtrax/tiling/iterator.py:\n\nVmapIterator(fn, xs): __iter__ → jax.vmap(fn)(xs) then yield each item (shape loses leading dim).\nSafeMapIterator(fn, xs, batch_size): __iter__ → safe_map(fn, xs, batch_size) then yield each; ValueError for non-divisible propagates.\nBucketIterator(boundaries, batch_sizes, fn, xs): raises ValueError at __init__ if len(batch_sizes) != len(boundaries)+1.\n\nUse jax.tree.map(lambda x: x[i], results) to yield individual items.\n\nWrite tests for: VmapIterator shape; SafeMapIterator==VmapIterator when batch_size>=n; SafeMapIterator ValueError for non-divisible; BucketIterator construction ValueError; BucketIterator valid case.\n\nGate: \`uv run pytest tests/tiling/test_iterator.py -v\` all pass; \`uv run ruff check src/xtrax/tiling/iterator.py\` clean.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify Iterators:\n1. VmapIterator yields n items, each without leading batch dim\n2. SafeMapIterator == VmapIterator results when batch_size >= n\n3. SafeMapIterator propagates ValueError for non-divisible n\n4. BucketIterator raises ValueError at construction for len mismatch (boundaries=[10,20], batch_sizes=[8,16] → raises)\n5. \`uv run pytest tests/tiling/test_iterator.py -v\` — all tests pass\n6. \`uv run ruff check src/xtrax/tiling/iterator.py\` — clean\n`,
  );

// ===== TRACK H — Track H — Phase 2.5: make_axis_dispatch (#1135) =========================
const trackH = () =>
  track(
    "1135",
    "Track H — Phase 2.5: make_axis_dispatch (#1135)",
    `task_id: ${TASK_ID}\nFile: src/xtrax/tiling/dispatch.py | Test: tests/tiling/test_dispatch.py\n\nImplement make_axis_dispatch in src/xtrax/tiling/dispatch.py:\n\n\`\`\`python\nfrom collections.abc import Callable\nfrom typing import Any\nimport jax\nfrom xtrax.tiling.strategy import AxisStrategy, DedupGather, SafeMap, Scan, Vmap\nfrom xtrax.transforms.map import safe_map\nfrom xtrax.transforms.scan import safe_scan\n\ndef make_axis_dispatch(strategy: AxisStrategy, fn: Callable, xs: Any, init: Any = None) -> Any:\n    if isinstance(strategy, Vmap):\n        return jax.vmap(fn)(xs)\n    elif isinstance(strategy, SafeMap):\n        return safe_map(fn, xs, batch_size=strategy.batch_size)\n    elif isinstance(strategy, Scan):\n        if init is None:\n            raise ValueError("make_axis_dispatch: Scan strategy requires a non-None init carry.")\n        # fn is IGNORED for Scan — use strategy.transition\n        return safe_scan(strategy.transition, init, xs)\n    elif isinstance(strategy, DedupGather):\n        deduped_xs, gather_indices = strategy.dedup_fn(xs)  # explicit unpacking\n        ys = safe_map(fn, deduped_xs)\n        return strategy.gather_fn(ys, gather_indices)\n    else:\n        raise TypeError(f"Unknown strategy type: {type(strategy)}")\n\`\`\`\n\nWrite tests for all 4 strategies: Vmap dispatches; SafeMap dispatches + ValueError for non-divisible; Scan with init=None raises; Scan with init uses strategy.transition (fn ignored); DedupGather with explicit dedup/gather fns.\n\nGate: \`uv run pytest tests/tiling/test_dispatch.py -v\` all pass; \`uv run ruff check src/xtrax/tiling/dispatch.py\` clean.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify make_axis_dispatch:\n1. Vmap → jax.vmap(fn)(xs)\n2. SafeMap → safe_map(fn, xs, batch_size=strategy.batch_size)\n3. Scan + init=None → ValueError\n4. Scan + init → safe_scan(strategy.transition, init, xs) — fn IS IGNORED\n5. DedupGather → deduped_xs, gather_indices = strategy.dedup_fn(xs) then safe_map then gather_fn\n6. \`uv run pytest tests/tiling/test_dispatch.py -v\` — all tests pass including Scan-ignores-fn test\n7. \`uv run ruff check src/xtrax/tiling/dispatch.py\` — clean\n`,
  );

// ---- orchestrate: writing chain (A -> B -> C -> D -> G -> E -> F -> H, sequential) ----
const SKIPPED = (id, reason) => ({ item_id: id, verdict: "SKIPPED", summary: reason });
const passed = (v) => v?.verdict === "PASS";

log("xtrax Sprint 1: Foundation — writing chain A→B→C→D→G→E→F→H (sequential, fail-fast)");

const a = await trackA();
if (!passed(a)) {
  return { task_id: TASK_ID, sprint_id: SPRINT_ID, aborted: "scaffold (1128) did not PASS", verdicts: { "1128": a } };
}

const b = await trackB();
const c = await trackC();
const d = await trackD();
const g = await trackG();

const e = passed(d)
  ? await trackE()
  : SKIPPED("1132", "dep 1131 (AxisStrategy) not PASS");

const f = (passed(b) && passed(c) && passed(d))
  ? await trackF()
  : SKIPPED("1133", "dep 1129/1130/1131 not all PASS");

const h = (passed(b) && passed(c) && passed(d) && passed(g) && passed(e) && passed(f))
  ? await trackH()
  : SKIPPED("1135", "one or more upstream tracks did not PASS");

return {
  task_id: TASK_ID,
  sprint_id: SPRINT_ID,
  verdicts: {
    "1128": a,
    "1129": b,
    "1130": c,
    "1131": d,
    "1134": g,
    "1132": e,
    "1133": f,
    "1135": h,
  },
};
