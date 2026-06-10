// Sprint 35 runner — emitted by `praxia dw emit-sprint`
// Source: .praxia/sprint_plans/sprint_plan.toml
// Regenerate: praxia dw emit-sprint sprint_plan.toml
// task_id: 260610_xtrax-packaging   sprint_id: 35
//
// RACE SAFETY (memory: parallel fixers race on git-status scope checks in praxia):
//   the writing chain (A,B,C,D,E,F,G,H,I,J,K,L) runs STRICTLY SEQUENTIAL —
//   exactly one fixer touches the working tree at a time.

export const meta = {
  name: "260610_xtrax-packaging",
  description: "pip-installable + PyPI + CI + docs",
  phases: [
    { title: "Track A — [N0] Remove stale committed .coverage + gitignore coverage artifacts (#1451)" },
    { title: "Track B — [N1] Version single-sourcing (__init__-as-truth, 0.2.0) + wheel verification (#1452)" },
    { title: "Track C — [N3] Hybrid + lazy public API surface (populate empty __init__ + PEP562 __getattr__) (#1453)" },
    { title: "Track D — [N9] Configure PyPI + TestPyPI OIDC Trusted Publisher (human gate) (#1454)" },
    { title: "Track E — [N2] Apache-2.0 LICENSE + pyproject metadata + py.typed-in-wheel (#1455)" },
    { title: "Track F — [N6] Fresh coverage gate + minimal 3.13 ci.yml (#1456)" },
    { title: "Track G — [N4a] Autodoc plumbing (Sphinx conf.py + autosummary + RTD + docs CI -W) (#1457)" },
    { title: "Track H — [N4b] Narrative docs prose (quickstart + architecture, seeded from internal specs) (#1458)" },
    { title: "Track I — [N5] Output-sink docs (io callbacks + orbax checkpoint) + re-export doctests in CI (#1459)" },
    { title: "Track J — [N8] README + CHANGELOG + CONTRIBUTING + CITATION.cff + delete main.py (#1460)" },
    { title: "Track K — [N7] publish.yml OIDC Trusted Publishing (TestPyPI -> PyPI, tag-triggered) (#1461)" },
    { title: "Track L — [N10] Epic-done release-readiness gate (convergence) (#1462)" },
  ],
};

const TASK_ID = "260610_xtrax-packaging";
const MAX_FIX_RETRIES = 1;

function extractVerdict(text) {
  const m = String(text ?? "").match(/verdict:\s*([a-z_]+)/i);
  return m ? m[1].toLowerCase() : "advance";
}

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

// Shared context for the writing tracks (from recon, task 260610_xtrax-packaging).
const EMITTER_CTX = `xtrax is a JAX/Equinox library (src-layout, hatchling, requires-python>=3.13). It is functionally\ngreen — 414 tests pass, real coverage 96.5% (the committed .coverage at 27.5% is STALE), ruff clean —\nbut NOT distribution-ready. This sprint makes it pip-installable + PyPI-publishable with CI, hosted\ndocs, and a real README. Spec: .praxia/docs/specs/260610_make-the-xtrax-jax-library-distribution.md.\nDAG plan: .praxia/docs/plans/260610_xtrax-packaging-dag.md.\n\nTwo FIXER gates are root nodes that MUST land + be verified before the cursor fan-out:\n  - N1 (Track b): version single-sourcing + wheel verify\n  - N3 (Track c): hybrid + lazy public API surface\nStack rules: \`uv run pytest\`; \`uv run ruff check .\` clean before commit; never bare python.\nAll version strings reconcile to 0.2.0 (pyproject currently 0.2.0, __init__ currently 0.1.0).\n`;

// ---- per-track stage helpers ---------------------------------------------
const fixer = (prompt, label, phaseName, isolation = null) => {
  const opts = { agentType: "fixer", label, phase: phaseName };
  if (isolation) opts.isolation = isolation;
  return agent(`${prompt}\n\nWhen done, end your message with 'verdict: done' on its own line.`, opts);
};

const reviewer = (itemId, prompt, label, phaseName, isolation = null) => {
  const opts = { agentType: "reviewer", label, phase: phaseName, schema: VERDICT_SCHEMA };
  if (isolation) opts.isolation = isolation;
  return agent(prompt, opts);
};

// Sequential implement->review with bounded NEEDS_WORK repair cycles.
async function track(itemId, phaseName, fixerPrompt, reviewerPrompt, isolation = null) {
  log(`[${itemId}] implement`);
  await fixer(fixerPrompt, `fix:${itemId}`, phaseName, isolation);
  let verdict = await reviewer(itemId, reviewerPrompt, `review:${itemId}`, phaseName, isolation);
  for (let retry = 0; retry < MAX_FIX_RETRIES && verdict && verdict.verdict === "NEEDS_WORK"; retry++) {
    log(`[${itemId}] NEEDS_WORK — repair cycle ${retry + 1}/${MAX_FIX_RETRIES}`);
    const issues = (verdict.issues || [])
      .map((i) => `- ${i.where}: ${i.problem} -> ${i.fix}`)
      .join("\n");
    await fixer(
      `${fixerPrompt}\n\nA reviewer found issues — fix exactly these, nothing else:\n${issues}`,
      `fix:${itemId}:repair:${retry}`,
      phaseName,
      isolation
    );
    verdict = await reviewer(itemId, reviewerPrompt, `review:${itemId}:re:${retry}`, phaseName, isolation);
  }
  return verdict;
}

// ===== TRACK A — Track A — [N0] Remove stale committed .coverage + gitignore coverage artifacts (#1451) =========================
const trackA = () =>
  track(
    "1451",
    "Track A — [N0] Remove stale committed .coverage + gitignore coverage artifacts (#1451)",
    `You are a delegating agent. Your task:\n1. Call mcp__praxia__dispatch with payload: {\n\x20\x20\'action\': \'create\',\n\x20\x20\'target\': "cursor",\n\x20\x20\'task_id\': args?.task_id ?? \'\',\n\x20\x20\'prompt\': <the prompt below>,\n\x20\x20\'execute\': true\n}\n2. Capture the returned dispatch id.\n3. Poll mcp__praxia__dispatch with action: \'status\' and that id roughly every 15 seconds.\n4. Terminal statuses are: completed | failed | killed | cancelled — stop on any of them.\n5. IMPORTANT: Poll for at most ~15 minutes (~60 attempts). The dispatch rescuer recycles\na stalled dispatch back to pending, so you MUST stop at your own deadline rather than\nwaiting indefinitely.\n6. On timeout: return \'verdict: needs_work\' with a note \'surface dispatch on cursor timed out\'.\n7. On terminal: return the dispatch result text verbatim.\n\nThe original prompt to dispatch:\n${`task_id: ${TASK_ID}. You are removing the stale committed coverage artifact from xtrax (DAG node N0). task_id: 260610_xtrax-packaging.\n\nWHY: A \`.coverage\` file is committed showing a stale 27.5% (real coverage is 96.5%). It MUST be removed and\ngitignored BEFORE the coverage gate (N6) is wired, or \`pytest --cov-fail-under=90\` will read the stale 27.5%\nartifact and red-bar CI on day one (pre-mortem-b).\n\nTASK (repo root /home/marielle/projects/xtrax):\n1. \`git rm --cached .coverage\` (and \`git rm --cached coverage.xml\` if it is tracked — check \`git ls-files | grep coverage\`).\n2. Ensure \`.gitignore\` contains both lines (append if missing):\n     .coverage\n     coverage.xml\n3. Do NOT delete the working-copy file if a dev needs it locally — only untrack it. Do not touch .coverage_html/ rules if already ignored.\n\nACCEPTANCE:\n- \`git ls-files | grep -cE '(^|/)\.coverage'\` prints \`0\`.\n- \`.gitignore\` contains \`.coverage\` and \`coverage.xml\`.\n- \`git status\` shows .coverage as no longer tracked.\n\n\n${EMITTER_CTX}`}`,
    `task_id: ${TASK_ID}. Verify Track A (N0 — stale coverage cleanup). task_id: 260610_xtrax-packaging.\n\nRun:\n- \`git ls-files | grep -cE '(^|/)\.coverage'\`  → MUST be 0  (PASS/FAIL)\n- \`grep -E '^\.coverage$|^coverage\.xml$' .gitignore\`  → both present  (PASS/FAIL)\n- \`git status --short | grep -E '\.coverage'\`  → not shown as tracked-modified  (PASS/FAIL)\n\nPASS only if .coverage and coverage.xml are untracked AND gitignored. This node gates N6 (coverage gate).\nEmit verdict: PASS or NEEDS_WORK with the failing check.\n`,
  );

// ===== TRACK B — Track B — [N1] Version single-sourcing (__init__-as-truth, 0.2.0) + wheel verification (#1452) =========================
const trackB = () =>
  track(
    "1452",
    "Track B — [N1] Version single-sourcing (__init__-as-truth, 0.2.0) + wheel verification (#1452)",
    `task_id: ${TASK_ID}. You are single-sourcing the xtrax package version and proving the wheel builds (DAG node N1 — ROOT GATE 1).\ntask_id: 260610_xtrax-packaging. Edit only the named files.\n\nCONTEXT — the bug: pyproject.toml declares version = "0.2.0"; src/xtrax/__init__.py declares __version__ = "0.1.0".\nThey must reconcile to a single source of truth = 0.2.0. Hatchling reads the version by REGEX-SCANNING the source\nfile (no import), so this is build-safe for a JAX-heavy package.\n\nTASK:\n1. In pyproject.toml [project]: remove the static \`version = "0.2.0"\` line and add \`dynamic = ["version"]\`.\n   Add a \`[tool.hatch.version]\` table: \`path = "src/xtrax/__init__.py"\`.\n2. In src/xtrax/__init__.py: set \`__version__ = "0.2.0"\` and keep it as the LITERAL FIRST executable line of the\n   module (above any imports or PEP 562 __getattr__ re-export block that node N3 will add) so the regex finds it.\n3. Confirm no other source file hardcodes the version: \`grep -rn '0\.1\.0' src/\` must be empty after the change.\n\nACCEPTANCE (run all):\n- \`uv build\` succeeds (builds sdist + wheel into dist/).\n- Clean-venv import prints the right version:\n    \`uv run --isolated --with dist/*.whl python -c "import xtrax; print(xtrax.__version__)"\`  → prints \`0.2.0\`\n- \`unzip -p dist/*.whl '**/METADATA' | grep '^Version:'\`  → \`Version: 0.2.0\`\n- \`uv run twine check dist/*\`  → PASSED\n- \`grep -rn '0\.1\.0' src/\`  → no output\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify Track B (N1 — version single-sourcing). task_id: 260610_xtrax-packaging. This is a ROOT GATE — be strict.\n\nRun:\n- \`uv build\`  → succeeds, dist/*.whl present  (PASS/FAIL)\n- \`unzip -p dist/*.whl '**/METADATA' | grep '^Version:'\`  → \`0.2.0\`  (PASS/FAIL)\n- \`uv run --isolated --with dist/*.whl python -c "import xtrax; print(xtrax.__version__)"\`  → \`0.2.0\`  (PASS/FAIL)\n- \`uv run twine check dist/*\`  → PASSED  (PASS/FAIL)\n- \`grep -rn '0\.1\.0' src/\`  → empty  (PASS/FAIL)\n- pyproject uses \`dynamic = ["version"]\` + \`[tool.hatch.version] path = ...\` (no static version)  (PASS/FAIL)\n\nPASS only if the built wheel ships Version 0.2.0 AND the installed import reports 0.2.0. Downstream nodes\n(N2, N6, N7, N8, N10) depend on this. Emit PASS or NEEDS_WORK with the failing command output.\n`,
  );

// ===== TRACK C — Track C — [N3] Hybrid + lazy public API surface (populate empty __init__ + PEP562 __getattr__) (#1453) =========================
const trackC = () =>
  track(
    "1453",
    "Track C — [N3] Hybrid + lazy public API surface (populate empty __init__ + PEP562 __getattr__) (#1453)",
    `task_id: ${TASK_ID}. You are building the public API surface for xtrax (DAG node N3 — ROOT GATE 2). task_id: 260610_xtrax-packaging.\n\nCONTEXT: 8 of 11 subpackages already export __all__ (engine, sparse, checkpoint, io, distributed, transforms,\nsafety, stages). THREE are EMPTY: src/xtrax/training/__init__.py, src/xtrax/data/__init__.py,\nsrc/xtrax/tiling/__init__.py. There is also NO curated top-level surface — users must import subpackages explicitly.\nA naive eager flat re-export would force \`import xtrax\` to import all JAX-heavy submodules (multi-second cold import,\npre-mortem-a). The fix is a HYBRID surface with a LAZY top-level (PEP 562 module-level __getattr__).\n\nTASK:\n1. Populate the three empty __init__.py with curated __all__ re-exporting their real public types — read the actual\n   modules to get the names:\n     - training/: Trainer (training/trainer.py) + the public train-step/types API actually defined there.\n     - data/: the dataset/pipeline/module API (BatchPlan lives in tiling — check; export data's real public types).\n     - tiling/: AxisSpec, BatchPlan, BatchPlanner + the tiling primitives (strategy/plan/iterator) that are public.\n   Match the style of an existing populated subpackage __init__ (e.g. src/xtrax/sparse/__init__.py).\n2. In src/xtrax/__init__.py add a curated flat top-level via PEP 562 lazy __getattr__ (NOT eager imports):\n     __all__ = ["Trainer", "Engine", "AxisSpec", "BatchPlan", "BoundedCallbackHandler",\n                "save_checkpoint", "load_checkpoint", ...]   # the curated 80% surface incl. output-sink names\n     _LAZY = {"Trainer": "xtrax.training", "Engine": "xtrax.engine", ...}\n     def __getattr__(name):\n         if name in _LAZY:\n             import importlib\n             return getattr(importlib.import_module(_LAZY[name]), name)\n         raise AttributeError(f"module 'xtrax' has no attribute {name!r}")\n   Keep \`__version__ = "0.2.0"\` as the FIRST line (node N1) — the lazy block goes BELOW it.\n\nACCEPTANCE (run all):\n- \`uv run python -c "from xtrax import Trainer, Engine; print(Trainer, Engine)"\`  → works.\n- All 11 subpackages export a non-empty __all__:\n    \`uv run python -c "import importlib; [print(p, bool(getattr(importlib.import_module('xtrax.'+p),'__all__',[]))) for p in ['engine','training','data','tiling','sparse','checkpoint','io','distributed','transforms','safety','stages']]"\`  → all True\n- Cold-import time bound (clean process, JAX not pre-imported):\n    \`uv run python -c "import time; t=time.perf_counter(); import xtrax; e=time.perf_counter()-t; assert e<0.5, f'cold import {e:.2f}s exceeds 500ms'; print(f'{e*1000:.0f}ms')"\`\n- \`uv run pytest -q\` still green (no import regressions); \`uv run ruff check .\` clean.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify Track C (N3 — hybrid + lazy API). task_id: 260610_xtrax-packaging. ROOT GATE — be strict.\n\nRun:\n- \`uv run python -c "from xtrax import Trainer, Engine"\`  → no error  (PASS/FAIL)\n- the 11-subpackage __all__ check (all True)  (PASS/FAIL)\n- cold-import assertion \`import xtrax\` < 500ms with JAX not pre-imported  (PASS/FAIL)\n- CONFIRM the top-level uses PEP 562 \`__getattr__\` lazy re-exports, NOT eager \`from .training import Trainer\`\n  (inspect src/xtrax/__init__.py)  (PASS/FAIL — fail if eager, it regresses import time)\n- \`uv run pytest -q\`  → green  (PASS/FAIL)\n- \`uv run ruff check .\`  → clean  (PASS/FAIL)\n\nPASS only if the flat surface works AND cold import stays under the bound AND it is lazy. Nodes N4a, N8, N10\ndepend on this. Emit PASS or NEEDS_WORK.\n`,
  );

// ===== TRACK D — Track D — [N9] Configure PyPI + TestPyPI OIDC Trusted Publisher (human gate) (#1454) =========================
const trackD = () =>
  track(
    "1454",
    "Track D — [N9] Configure PyPI + TestPyPI OIDC Trusted Publisher (human gate) (#1454)",
    `task_id: ${TASK_ID}. HUMAN GATE (DAG node N9) — this is an out-of-band configuration task that NO automated agent can complete.\ntask_id: 260610_xtrax-packaging. Your job is to surface the exact checklist and HALT for the project owner.\n\nWHY: N7 (publish.yml) uses PyPI OIDC Trusted Publishing (no stored token). The trust relationship must be\nconfigured ON THE PYPI SIDE first, or the first \`git tag\` publish fails and someone pastes a long-lived token\nunder deadline pressure (pre-mortem-c). This gate blocks N7.\n\nEMIT THIS CHECKLIST for the owner (do not attempt to perform it):\n1. On https://pypi.org → project \`xtrax\` → Publishing → add a GitHub Actions Trusted Publisher:\n     owner = <github-org-or-user>, repository = xtrax, workflow = publish.yml, environment = (match publish.yml).\n2. Repeat on https://test.pypi.org for the TestPyPI staging stage.\n3. If the \`xtrax\` project does not yet exist on PyPI, register the name via a first manual/TestPyPI upload OR\n   pre-create the pending publisher.\n\nACCEPTANCE: the project owner confirms both PyPI and TestPyPI Trusted Publishers are configured. This item is\nDONE only on explicit human confirmation — do not mark it complete autonomously.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify Track D (N9 — OIDC human gate). task_id: 260610_xtrax-packaging.\n\nThis gate cannot be verified by command — it is an external PyPI configuration. PASS only if the project owner\nhas explicitly confirmed that BOTH the PyPI and TestPyPI Trusted Publishers for owner/xtrax + publish.yml are\nconfigured. Otherwise NEEDS_WORK (blocking). Do NOT auto-PASS. N7 (publish) must not run until this is PASS.\n`,
  );

// ===== TRACK E — Track E — [N2] Apache-2.0 LICENSE + pyproject metadata + py.typed-in-wheel (#1455) =========================
const trackE = () =>
  track(
    "1455",
    "Track E — [N2] Apache-2.0 LICENSE + pyproject metadata + py.typed-in-wheel (#1455)",
    `You are a delegating agent. Your task:\n1. Call mcp__praxia__dispatch with payload: {\n\x20\x20\'action\': \'create\',\n\x20\x20\'target\': "cursor",\n\x20\x20\'task_id\': args?.task_id ?? \'\',\n\x20\x20\'prompt\': <the prompt below>,\n\x20\x20\'execute\': true\n}\n2. Capture the returned dispatch id.\n3. Poll mcp__praxia__dispatch with action: \'status\' and that id roughly every 15 seconds.\n4. Terminal statuses are: completed | failed | killed | cancelled — stop on any of them.\n5. IMPORTANT: Poll for at most ~15 minutes (~60 attempts). The dispatch rescuer recycles\na stalled dispatch back to pending, so you MUST stop at your own deadline rather than\nwaiting indefinitely.\n6. On timeout: return \'verdict: needs_work\' with a note \'surface dispatch on cursor timed out\'.\n7. On terminal: return the dispatch result text verbatim.\n\nThe original prompt to dispatch:\n${`task_id: ${TASK_ID}. You are completing xtrax packaging metadata (DAG node N2). task_id: 260610_xtrax-packaging. Depends on N1 (version).\n\nTASK (repo root /home/marielle/projects/xtrax):\n1. Add an \`Apache-2.0\` LICENSE file at repo root (full Apache License 2.0 text; matches the orbax/JAX upstream stack).\n2. In pyproject.toml [project] add:\n     license = "Apache-2.0"   # SPDX expression\n     keywords = ["jax", "equinox", "training", "sparse", "distributed"]\n     authors = [{ name = "<author>", email = "<email>" }]\n     classifiers = [\n       "Development Status :: 4 - Beta",\n       "Programming Language :: Python :: 3.13",\n       "Intended Audience :: Science/Research",\n       "License :: OSI Approved :: Apache Software License",\n       "Typing :: Typed",\n     ]\n   Add [project.urls]: Homepage, Documentation, Repository, Issues, Changelog.\n3. Create the marker file src/xtrax/py.typed (empty) and ENSURE it ships in the wheel. With hatchling src-layout add:\n     [tool.hatch.build.targets.wheel.force-include]\n     "src/xtrax/py.typed" = "xtrax/py.typed"\n   (or the artifacts/include mechanism that lands it inside the installed package).\n\nACCEPTANCE (run all):\n- \`uv build\` then \`uv run twine check dist/*\`  → PASSED with complete metadata.\n- \`unzip -l dist/*.whl | grep -c 'xtrax/py.typed'\`  → 1.\n- py.typed present in the INSTALLED wheel:\n    \`uv run --isolated --with dist/*.whl python -c "import importlib.resources as r, xtrax; assert (r.files('xtrax')/'py.typed').is_file()"\`\n- LICENSE file exists; classifiers include \`License :: OSI Approved :: Apache Software License\` and \`Typing :: Typed\`.\n\n\n${EMITTER_CTX}`}`,
    `task_id: ${TASK_ID}. Verify Track E (N2 — license + metadata + py.typed). task_id: 260610_xtrax-packaging.\n\nRun:\n- \`test -f LICENSE && head -1 LICENSE\`  → Apache 2.0  (PASS/FAIL)\n- \`uv build && uv run twine check dist/*\`  → PASSED  (PASS/FAIL)\n- \`unzip -l dist/*.whl | grep -c 'xtrax/py.typed'\`  → 1  (PASS/FAIL — this is the pre-mortem-d silent defect; must be in the WHEEL not just src/)\n- installed-wheel py.typed assertion (the importlib.resources one-liner)  (PASS/FAIL)\n- classifiers + [project.urls] present in pyproject  (PASS/FAIL)\n\nPASS only if py.typed is INSIDE the built wheel. Emit PASS or NEEDS_WORK.\n`,
  );

// ===== TRACK F — Track F — [N6] Fresh coverage gate + minimal 3.13 ci.yml (#1456) =========================
const trackF = () =>
  track(
    "1456",
    "Track F — [N6] Fresh coverage gate + minimal 3.13 ci.yml (#1456)",
    `You are a delegating agent. Your task:\n1. Call mcp__praxia__dispatch with payload: {\n\x20\x20\'action\': \'create\',\n\x20\x20\'target\': "cursor",\n\x20\x20\'task_id\': args?.task_id ?? \'\',\n\x20\x20\'prompt\': <the prompt below>,\n\x20\x20\'execute\': true\n}\n2. Capture the returned dispatch id.\n3. Poll mcp__praxia__dispatch with action: \'status\' and that id roughly every 15 seconds.\n4. Terminal statuses are: completed | failed | killed | cancelled — stop on any of them.\n5. IMPORTANT: Poll for at most ~15 minutes (~60 attempts). The dispatch rescuer recycles\na stalled dispatch back to pending, so you MUST stop at your own deadline rather than\nwaiting indefinitely.\n6. On timeout: return \'verdict: needs_work\' with a note \'surface dispatch on cursor timed out\'.\n7. On terminal: return the dispatch result text verbatim.\n\nThe original prompt to dispatch:\n${`task_id: ${TASK_ID}. You are adding CI for xtrax (DAG node N6). task_id: 260610_xtrax-packaging. Depends on N0 (stale-coverage removed) and N1.\n\nTASK: create .github/workflows/ci.yml — a lean Python-3.13-only workflow (requires-python>=3.13, so no matrix breadth).\nUse uv (astral-sh/setup-uv). Jobs/steps:\n  - checkout\n  - \`uv sync --all-extras\` (or \`--dev\`)\n  - \`uv run ruff check .\`\n  - \`uv run ruff format --check .\`\n  - type check: \`uv run pyright\` (dev deps include pyright)\n  - \`uv run pytest --cov=xtrax --cov-branch --cov-fail-under=90\`\nTrigger on push + pull_request to main. ubuntu-latest (JAX CPU wheels).\n\nIMPORTANT (pre-mortem-b): N0 already untracked + gitignored the stale .coverage, so the gate reads FRESH coverage\n(real ~96.5%). Do not re-introduce a committed .coverage. The 90% floor leaves refactor headroom under 96.5%.\n\nACCEPTANCE:\n- .github/workflows/ci.yml exists and is valid YAML (\`uv run python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml'))"\`).\n- Locally, the gate command passes fresh: \`uv run pytest --cov=xtrax --cov-branch --cov-fail-under=90 -q\` → green, coverage ≥ 90%.\n- \`git ls-files | grep -cE '(^|/)\.coverage'\` still 0 (N0 invariant preserved).\n\n\n${EMITTER_CTX}`}`,
    `task_id: ${TASK_ID}. Verify Track F (N6 — coverage gate + ci.yml). task_id: 260610_xtrax-packaging.\n\nRun:\n- ci.yml exists + parses as YAML  (PASS/FAIL)\n- ci.yml runs ruff lint + ruff format --check + type check + pytest with \`--cov-fail-under=90\`  (PASS/FAIL)\n- local fresh run \`uv run pytest --cov=xtrax --cov-branch --cov-fail-under=90 -q\`  → green ≥90%  (PASS/FAIL)\n- \`git ls-files | grep -cE '(^|/)\.coverage'\`  → 0 (the gate reads fresh, not the stale artifact)  (PASS/FAIL)\n\nPASS only if the gate passes against FRESH coverage. Emit PASS or NEEDS_WORK.\n`,
  );

// ===== TRACK G — Track G — [N4a] Autodoc plumbing (Sphinx conf.py + autosummary + RTD + docs CI -W) (#1457) =========================
const trackG = () =>
  track(
    "1457",
    "Track G — [N4a] Autodoc plumbing (Sphinx conf.py + autosummary + RTD + docs CI -W) (#1457)",
    `task_id: ${TASK_ID}. You are standing up the docs build for xtrax (DAG node N4a — autodoc plumbing). task_id: 260610_xtrax-packaging.\nDepends on N3 (public API surface). The existing docstrings ARE the source of truth — autodoc consumes them.\n\nTASK:\n1. Create docs/ with a Sphinx project: conf.py using extensions sphinx.ext.autodoc + autosummary + napoleon\n   (Google/NumPy docstrings), html_theme = "furo". Set project/version (read xtrax.__version__).\n2. Build an API reference organized along the 11 subpackages with an autosummary index; surface the OUTPUT-SINK\n   group (io + checkpoint) as a TOP-LEVEL nav entry, not buried. Leave content stubs for narrative pages (N4b fills them).\n3. Add a docs optional-dependency group to pyproject:\n     [dependency-groups] docs = ["sphinx", "furo", "sphinx-autodoc-typehints"]   (or [project.optional-dependencies] docs)\n4. Add .readthedocs.yaml pinning python 3.13 + the docs group (build.os ubuntu, build.tools.python "3.13",\n   python.install the docs extra, sphinx.configuration docs/conf.py).\n5. Add a docs job to .github/workflows/ (ci.yml or docs.yml) that builds with warnings-as-errors in a FRESH env\n   matching RTD isolation (pre-mortem-e): \`uv sync --only-group docs\` then \`uv run sphinx-build -W -n -b html docs docs/_build\`.\n\nACCEPTANCE:\n- \`uv sync --only-group docs && uv run sphinx-build -W -n -b html docs docs/_build\`  → exits 0 (no warnings).\n- .readthedocs.yaml exists, pins python 3.13 + docs group.\n- API reference covers all 11 subpackages; output-sink (io+checkpoint) is a top-level nav group.\n- The CI docs job uses the SAME \`uv sync --only-group docs\` clean-env mode RTD uses.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify Track G (N4a — autodoc plumbing). task_id: 260610_xtrax-packaging.\n\nRun:\n- \`uv sync --only-group docs && uv run sphinx-build -W -n -b html docs docs/_build\`  → exit 0, no warnings  (PASS/FAIL)\n- .readthedocs.yaml pins python 3.13 + docs group  (PASS/FAIL)\n- docs CI job runs sphinx-build -W in a fresh \`--only-group docs\` venv (RTD parity, pre-mortem-e)  (PASS/FAIL)\n- API reference includes all 11 subpackages; output-sink is a top-level nav group  (PASS/FAIL)\n\nPASS only if -W -n build is clean in the docs-only env. N4b/N5/N8 depend on this. Emit PASS or NEEDS_WORK.\n`,
  );

// ===== TRACK H — Track H — [N4b] Narrative docs prose (quickstart + architecture, seeded from internal specs) (#1458) =========================
const trackH = () =>
  track(
    "1458",
    "Track H — [N4b] Narrative docs prose (quickstart + architecture, seeded from internal specs) (#1458)",
    `You are a delegating agent. Your task:\n1. Call mcp__praxia__dispatch with payload: {\n\x20\x20\'action\': \'create\',\n\x20\x20\'target\': "cursor",\n\x20\x20\'task_id\': args?.task_id ?? \'\',\n\x20\x20\'prompt\': <the prompt below>,\n\x20\x20\'execute\': true\n}\n2. Capture the returned dispatch id.\n3. Poll mcp__praxia__dispatch with action: \'status\' and that id roughly every 15 seconds.\n4. Terminal statuses are: completed | failed | killed | cancelled — stop on any of them.\n5. IMPORTANT: Poll for at most ~15 minutes (~60 attempts). The dispatch rescuer recycles\na stalled dispatch back to pending, so you MUST stop at your own deadline rather than\nwaiting indefinitely.\n6. On timeout: return \'verdict: needs_work\' with a note \'surface dispatch on cursor timed out\'.\n7. On terminal: return the dispatch result text verbatim.\n\nThe original prompt to dispatch:\n${`task_id: ${TASK_ID}. You are writing the narrative docs prose for xtrax (DAG node N4b). task_id: 260610_xtrax-packaging.\nDepends on N4a (Sphinx scaffold exists). Author into the docs/ tree N4a created.\n\nTASK: write public-facing narrative pages, seeding from the internal specs and STRIPPING internal-only references\n(no .praxia/ paths, no task_ids, no sprint jargon):\n  - docs/quickstart: a 30-second example using the flat import \`from xtrax import Trainer, Engine\` (the N3 surface).\n    The quickstart code block must be doctest-executable.\n  - docs/architecture + docs/concepts: adapt from .praxia/docs/specs/260604_xtrax-spec.md (overall design) and\n    260608_xtrax-s5-sparse.md (sparse subsystem). Public framing only.\nCross-link to the API reference (N4a) and the output-sink chapter (N5).\n\nACCEPTANCE:\n- Pages build under \`uv run sphinx-build -W -n -b html docs docs/_build\` (no warnings, no dead xrefs).\n- The quickstart code block is doctest-executable (sphinx doctest or pytest --doctest-glob).\n- No internal references (\`.praxia\`, \`task_id\`, sprint/track jargon) leak into rendered pages.\n\n\n${EMITTER_CTX}`}`,
    `task_id: ${TASK_ID}. Verify Track H (N4b — narrative prose). task_id: 260610_xtrax-packaging.\n\nRun:\n- \`uv run sphinx-build -W -n -b html docs docs/_build\`  → clean (PASS/FAIL)\n- quickstart uses the flat \`from xtrax import ...\` surface and is doctest-executable  (PASS/FAIL)\n- \`grep -rIE '\.praxia|task_id|Track [A-Z]' docs/ --include='*.md' --include='*.rst'\`  → empty (no internal leakage)  (PASS/FAIL)\n\nPASS only if narrative builds clean AND has no internal-doc leakage. Emit PASS or NEEDS_WORK.\n`,
  );

// ===== TRACK I — Track I — [N5] Output-sink docs (io callbacks + orbax checkpoint) + re-export doctests in CI (#1459) =========================
const trackI = () =>
  track(
    "1459",
    "Track I — [N5] Output-sink docs (io callbacks + orbax checkpoint) + re-export doctests in CI (#1459)",
    `You are a delegating agent. Your task:\n1. Call mcp__praxia__dispatch with payload: {\n\x20\x20\'action\': \'create\',\n\x20\x20\'target\': "cursor",\n\x20\x20\'task_id\': args?.task_id ?? \'\',\n\x20\x20\'prompt\': <the prompt below>,\n\x20\x20\'execute\': true\n}\n2. Capture the returned dispatch id.\n3. Poll mcp__praxia__dispatch with action: \'status\' and that id roughly every 15 seconds.\n4. Terminal statuses are: completed | failed | killed | cancelled — stop on any of them.\n5. IMPORTANT: Poll for at most ~15 minutes (~60 attempts). The dispatch rescuer recycles\na stalled dispatch back to pending, so you MUST stop at your own deadline rather than\nwaiting indefinitely.\n6. On timeout: return \'verdict: needs_work\' with a note \'surface dispatch on cursor timed out\'.\n7. On terminal: return the dispatch result text verbatim.\n\nThe original prompt to dispatch:\n${`task_id: ${TASK_ID}. You are documenting the OUTPUT-SINK surface of xtrax (DAG node N5 — user-flagged in scope). task_id: 260610_xtrax-packaging.\nDepends on N4a. The output-sink surface = (a) io callbacks: BoundedCallbackHandler + async_indexed_stream, which are\nRE-EXPORTED from xtrax.engine.io via xtrax.io.callbacks; (b) orbax checkpoint persistence: save_checkpoint /\nload_checkpoint / get_checkpoint_manager.\n\nTASK:\n1. Write a unified "Output sinks" narrative chapter framing the two halves of "getting results out":\n   streaming/observability (io callbacks) vs durable state (orbax checkpoints). Per-surface autodoc tables underneath.\n2. STATE THE CANONICAL IMPORT PATH explicitly: \`from xtrax.io import BoundedCallbackHandler, async_indexed_stream\`,\n   and document the io -> engine.io re-export boundary so users learn the canonical path (io/callbacks.py is a thin shim).\n3. Add a save->restore round-trip example for the checkpoint surface; cross-link to orbax upstream rather than duplicating it.\n4. Make the documented re-export usage examples DOCTESTS run in CI (sphinx doctest or \`pytest --doctest-modules\` over the\n   io docstrings) so a moved re-export FAILS the build instead of silently rotting.\n\nACCEPTANCE:\n- The output-sink chapter builds under \`sphinx-build -W -n\` and is reachable from the top-level nav (N4a group).\n- Doctests pass in CI: \`uv run pytest --doctest-modules src/xtrax/io src/xtrax/engine/io.py -q\` (or the sphinx-doctest target) → green.\n- The canonical \`from xtrax.io import ...\` path and the io->engine.io boundary are documented.\n- README has "Streaming outputs" + "Checkpointing" sections (or stubs N8 will expand).\n\n\n${EMITTER_CTX}`}`,
    `task_id: ${TASK_ID}. Verify Track I (N5 — output-sink docs + doctests). task_id: 260610_xtrax-packaging.\n\nRun:\n- output-sink chapter builds under \`sphinx-build -W -n\` and is in top-level nav  (PASS/FAIL)\n- doctests of the io re-export examples run in CI and pass  (PASS/FAIL — this is what stops silent rot)\n- canonical \`from xtrax.io import ...\` path + io->engine.io boundary documented  (PASS/FAIL)\n\nPASS only if the re-export doctests actually execute in CI. Emit PASS or NEEDS_WORK.\n`,
  );

// ===== TRACK J — Track J — [N8] README + CHANGELOG + CONTRIBUTING + CITATION.cff + delete main.py (#1460) =========================
const trackJ = () =>
  track(
    "1460",
    "Track J — [N8] README + CHANGELOG + CONTRIBUTING + CITATION.cff + delete main.py (#1460)",
    `You are a delegating agent. Your task:\n1. Call mcp__praxia__dispatch with payload: {\n\x20\x20\'action\': \'create\',\n\x20\x20\'target\': "cursor",\n\x20\x20\'task_id\': args?.task_id ?? \'\',\n\x20\x20\'prompt\': <the prompt below>,\n\x20\x20\'execute\': true\n}\n2. Capture the returned dispatch id.\n3. Poll mcp__praxia__dispatch with action: \'status\' and that id roughly every 15 seconds.\n4. Terminal statuses are: completed | failed | killed | cancelled — stop on any of them.\n5. IMPORTANT: Poll for at most ~15 minutes (~60 attempts). The dispatch rescuer recycles\na stalled dispatch back to pending, so you MUST stop at your own deadline rather than\nwaiting indefinitely.\n6. On timeout: return \'verdict: needs_work\' with a note \'surface dispatch on cursor timed out\'.\n7. On terminal: return the dispatch result text verbatim.\n\nThe original prompt to dispatch:\n${`task_id: ${TASK_ID}. You are writing the repo hygiene files for xtrax (DAG node N8). task_id: 260610_xtrax-packaging.\nDepends on N1 (version), N3 (API surface), N4a (docs links). README.md is currently 0 bytes.\n\nTASK:\n1. README.md — standard JAX-lib structure:\n   one-line tagline + badges (PyPI version, CI status, docs, coverage, license)\n   → "Why xtrax" → install (\`pip install xtrax\`, note Python 3.13)\n   → 30-second quickstart using the FLAT import (\`from xtrax import Trainer, Engine\`)\n   → feature highlights INCLUDING the output-sink surface (streaming callbacks + orbax checkpoints)\n   → links to hosted docs → license line. The README IS the PyPI long-description (readme = "README.md").\n2. CHANGELOG.md — Keep-a-Changelog format; a \`[0.2.0]\` entry recording the version reconciliation (0.1.0 -> 0.2.0)\n   and the distribution-readiness work.\n3. CONTRIBUTING.md — dev workflow: uv setup, \`uv run pytest\`, \`uv run ruff check/format\`, the 90% coverage gate,\n   building docs locally, and the tag-to-publish release flow.\n4. CITATION.cff — citable metadata (research audience); version field == 0.2.0 (keep in sync with the single source).\n5. DELETE the root main.py (83-byte unused scaffolding; not referenced anywhere — confirm with \`grep -rn 'main.py\|from main\|import main' .\`).\n\nACCEPTANCE:\n- README.md is non-empty, renders as the PyPI long-description (validate via \`uv run twine check dist/*\` after rebuild,\n  or TestPyPI preview); badges resolve; quickstart uses the flat import.\n- CHANGELOG.md, CONTRIBUTING.md, CITATION.cff exist; CITATION version == 0.2.0.\n- \`test ! -f main.py\`  → main.py is gone.\n\n\n${EMITTER_CTX}`}`,
    `task_id: ${TASK_ID}. Verify Track J (N8 — README + hygiene). task_id: 260610_xtrax-packaging.\n\nRun:\n- \`test -s README.md\`  → non-empty  (PASS/FAIL); contains install + flat-import quickstart + output-sink highlights\n- \`uv build && uv run twine check dist/*\`  → PASSED (README renders as long-description)  (PASS/FAIL)\n- CHANGELOG.md / CONTRIBUTING.md / CITATION.cff exist; \`grep -E '0\.2\.0' CITATION.cff\`  (PASS/FAIL)\n- \`test ! -f main.py\`  → removed  (PASS/FAIL)\n\nPASS only if README is real + renders AND main.py is gone. Emit PASS or NEEDS_WORK.\n`,
  );

// ===== TRACK K — Track K — [N7] publish.yml OIDC Trusted Publishing (TestPyPI -> PyPI, tag-triggered) (#1461) =========================
const trackK = () =>
  track(
    "1461",
    "Track K — [N7] publish.yml OIDC Trusted Publishing (TestPyPI -> PyPI, tag-triggered) (#1461)",
    `You are a delegating agent. Your task:\n1. Call mcp__praxia__dispatch with payload: {\n\x20\x20\'action\': \'create\',\n\x20\x20\'target\': "cursor",\n\x20\x20\'task_id\': args?.task_id ?? \'\',\n\x20\x20\'prompt\': <the prompt below>,\n\x20\x20\'execute\': true\n}\n2. Capture the returned dispatch id.\n3. Poll mcp__praxia__dispatch with action: \'status\' and that id roughly every 15 seconds.\n4. Terminal statuses are: completed | failed | killed | cancelled — stop on any of them.\n5. IMPORTANT: Poll for at most ~15 minutes (~60 attempts). The dispatch rescuer recycles\na stalled dispatch back to pending, so you MUST stop at your own deadline rather than\nwaiting indefinitely.\n6. On timeout: return \'verdict: needs_work\' with a note \'surface dispatch on cursor timed out\'.\n7. On terminal: return the dispatch result text verbatim.\n\nThe original prompt to dispatch:\n${`task_id: ${TASK_ID}. You are writing the release/publish workflow for xtrax (DAG node N7 — terminal publish). task_id: 260610_xtrax-packaging.\nDepends on N1 (version), N2 (metadata), N6 (CI green), and N9 (PyPI Trusted Publisher configured — HUMAN GATE: do not\npublish to real PyPI until N9 is confirmed).\n\nTASK: create .github/workflows/publish.yml triggered on \`v*\` tag push. Use OIDC Trusted Publishing (NO stored token):\n  - job builds sdist+wheel: \`uv build\`; then \`uv run twine check dist/*\`.\n  - STAGE 1 (runs FIRST, de-risk): publish to TestPyPI via pypa/gh-action-pypi-publish with\n    repository-url https://test.pypi.org/legacy/ and \`permissions: id-token: write\` (OIDC). Then verify a clean-venv\n    install from TestPyPI imports and prints 0.2.0.\n  - STAGE 2: publish to real PyPI via pypa/gh-action-pypi-publish (default repo), OIDC, gated on Stage 1 success.\n  - NO \`password:\`/token anywhere; rely on the Trusted Publisher (configured by N9).\n\nACCEPTANCE:\n- publish.yml exists, valid YAML, triggers on \`v*\` tags, has \`permissions: id-token: write\`.\n- Uses pypa/gh-action-pypi-publish with NO stored token; TestPyPI stage precedes PyPI stage.\n- \`grep -riE 'PYPI_TOKEN|password:|secrets\.PYPI' .github/workflows/\`  → empty (no long-lived token).\n- (Live publish is gated on N9 + an actual tag — do not push a real tag here; validate the workflow statically + via TestPyPI when the owner is ready.)\n\n\n${EMITTER_CTX}`}`,
    `task_id: ${TASK_ID}. Verify Track K (N7 — publish.yml OIDC). task_id: 260610_xtrax-packaging.\n\nRun:\n- publish.yml exists, parses, triggers on \`v*\`, has \`permissions: id-token: write\`  (PASS/FAIL)\n- uses pypa/gh-action-pypi-publish; TestPyPI stage runs BEFORE PyPI stage  (PASS/FAIL)\n- \`grep -riE 'PYPI_TOKEN|password:|secrets\.PYPI' .github/workflows/\`  → empty (OIDC, no token — pre-mortem-c)  (PASS/FAIL)\n- N9 (human OIDC config) is confirmed PASS before any real-PyPI publish is attempted  (PASS/FAIL)\n\nPASS only if the workflow is tokenless OIDC with TestPyPI-first. Emit PASS or NEEDS_WORK.\n`,
  );

// ===== TRACK L — Track L — [N10] Epic-done release-readiness gate (convergence) (#1462) =========================
const trackL = () =>
  track(
    "1462",
    "Track L — [N10] Epic-done release-readiness gate (convergence) (#1462)",
    `task_id: ${TASK_ID}. CONVERGENCE / EPIC-DONE GATE (DAG node N10 — terminal sink). task_id: 260610_xtrax-packaging.\nThis is a verification rollup, not an implementation task. Depends on N7 (publish), N4b (prose), N5 (output-sink docs),\nN8 (README/hygiene). Do not write feature code — assemble and run the final acceptance checks and report.\n\nRUN the full release-readiness checklist and report each result:\n1. \`pip install xtrax\` from PyPI installs v0.2.0 and imports:\n     \`uv run --isolated --with xtrax==0.2.0 python -c "import xtrax; assert xtrax.__version__=='0.2.0'"\`  (after N7 publishes)\n2. Hosted docs render full API + output-sink chapter + quickstart (RTD build green).\n3. README renders on the PyPI project page (long-description ok).\n4. \`test ! -f main.py\`  → scaffolding gone.\n5. CI green on main; coverage gate ≥ 90% reading FRESH coverage.\n\nACCEPTANCE: all five hold. If any fails, report which node (N7/N4a/N4b/N5/N8/N6) owns the gap — do NOT mark the\nepic done until all are green.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify Track L (N10 — epic-done convergence). task_id: 260610_xtrax-packaging.\n\nConfirm ALL hold (each maps to an upstream node):\n- pip-install from PyPI → import → __version__ == 0.2.0 (N1/N7)  (PASS/FAIL)\n- hosted docs render full API + output-sink + quickstart (N4a/N4b/N5)  (PASS/FAIL)\n- README renders on PyPI page (N8)  (PASS/FAIL)\n- main.py removed (N8)  (PASS/FAIL)\n- CI green on main, coverage ≥ 90% fresh (N0/N6)  (PASS/FAIL)\n\nPASS only if EVERY check holds — this is the gate that prevents a published-but-undocumented release.\nEmit PASS (epic complete) or NEEDS_WORK naming the owning node(s).\n`,
  );

// ---- orchestrate: writing chain (A -> B -> C -> D -> E -> F -> G -> H -> I -> J -> K -> L, sequential) ----
log("xtrax distribution-readiness: writing chain (A -> B -> C -> D -> E -> F -> G -> H -> I -> J -> K -> L, sequential)");
const a = await trackA();
const b = await trackB();
const c = await trackC();
const d = await trackD();
const e = await trackE();
const f = await trackF();
const g = await trackG();
const h = await trackH();
const i = await trackI();
const j = await trackJ();
const k = await trackK();
const l = await trackL();

return {
  task_id: TASK_ID,
  sprint_id: 35,
  verdicts: {
    "1451": a,
    "1452": b,
    "1453": c,
    "1454": d,
    "1455": e,
    "1456": f,
    "1457": g,
    "1458": h,
    "1459": i,
    "1460": j,
    "1461": k,
    "1462": l
  },
};
