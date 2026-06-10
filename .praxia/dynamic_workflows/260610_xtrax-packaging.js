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
const EMITTER_CTX = `[MANUAL: paste recon findings here before running emit-sprint]`;

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
    `You are a delegating agent. Your task:\n1. Call mcp__praxia__dispatch with payload: {\n\x20\x20\'action\': \'create\',\n\x20\x20\'target\': "cursor",\n\x20\x20\'task_id\': args?.task_id ?? \'\',\n\x20\x20\'prompt\': <the prompt below>,\n\x20\x20\'execute\': true\n}\n2. Capture the returned dispatch id.\n3. Poll mcp__praxia__dispatch with action: \'status\' and that id roughly every 15 seconds.\n4. Terminal statuses are: completed | failed | killed | cancelled — stop on any of them.\n5. IMPORTANT: Poll for at most ~15 minutes (~60 attempts). The dispatch rescuer recycles\na stalled dispatch back to pending, so you MUST stop at your own deadline rather than\nwaiting indefinitely.\n6. On timeout: return \'verdict: needs_work\' with a note \'surface dispatch on cursor timed out\'.\n7. On terminal: return the dispatch result text verbatim.\n\nThe original prompt to dispatch:\n${`task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nDistribution-readiness DAG node N0 (task 260610_xtrax-packaging). Enforces pre-mortem-b ordering at the DAG level: the stale .coverage (27.5%) must be removed BEFORE the coverage gate (N6) is wired, else it red-bars CI day one. ACCEPTANCE: \`git rm --cached .coverage\` (and coverage.xml if tracked); .gitignore contains \`.coverage\` and \`coverage.xml\`; verify \`git ls-files | grep -cE '(^|/)\.coverage'\` == 0. Dispatch: cursor. Source spec: .praxia/docs/specs/260610_make-the-xtrax-jax-library-distribution.md\n\n[AUTO: no recon anchors — paste file:line references here, then write the full fixer prompt]\n\n${EMITTER_CTX}`}`,
    `task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nDistribution-readiness DAG node N0 (task 260610_xtrax-packaging). Enforces pre-mortem-b ordering at the DAG level: the stale .coverage (27.5%) must be removed BEFORE the coverage gate (N6) is wired, else it red-bars CI day one. ACCEPTANCE: \`git rm --cached .coverage\` (and coverage.xml if tracked); .gitignore contains \`.coverage\` and \`coverage.xml\`; verify \`git ls-files | grep -cE '(^|/)\.coverage'\` == 0. Dispatch: cursor. Source spec: .praxia/docs/specs/260610_make-the-xtrax-jax-library-distribution.md\n\nREVIEWER CHECKLIST (fill in before running dw emit-sprint):\nVERIFY: cargo command or test that must pass — e.g. \`cargo nextest run -p <crate>\`\nVERIFY: observable assertion — e.g. "new test exists and calls the handler directly"\nVERIFY: no regression — e.g. "existing tests still pass"\nPASS if all VERIFY items are satisfied.\nFAIL if any VERIFY item is not met or untestable as written.`,
  );

// ===== TRACK B — Track B — [N1] Version single-sourcing (__init__-as-truth, 0.2.0) + wheel verification (#1452) =========================
const trackB = () =>
  track(
    "1452",
    "Track B — [N1] Version single-sourcing (__init__-as-truth, 0.2.0) + wheel verification (#1452)",
    `task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nDistribution-readiness DAG node N1 — ROOT GATE 1 (task 260610_xtrax-packaging). Resolves the 0.1.0/0.2.0 mismatch bug as part of packaging infra. Single-source via [tool.hatch.version] path="src/xtrax/__init__.py" + dynamic=["version"] in [project]; reconcile to 0.2.0; keep __version__="0.2.0" as the literal first executable line of __init__.py (above any re-export block). ACCEPTANCE: \`uv build\` green; clean-venv \`python -c "import xtrax; print(xtrax.__version__)"\` prints 0.2.0; \`unzip -p dist/*.whl '**/METADATA' | grep '^Version:'\`==0.2.0; \`twine check dist/*\` passes; \`grep -rn '0\.1\.0' src/\` empty. Dispatch: fixer (judgment/cross-cutting). Spec: .praxia/docs/specs/260610_make-the-xtrax-jax-library-distribution.md\n\n[AUTO: no recon anchors — paste file:line references here, then write the full fixer prompt]\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nDistribution-readiness DAG node N1 — ROOT GATE 1 (task 260610_xtrax-packaging). Resolves the 0.1.0/0.2.0 mismatch bug as part of packaging infra. Single-source via [tool.hatch.version] path="src/xtrax/__init__.py" + dynamic=["version"] in [project]; reconcile to 0.2.0; keep __version__="0.2.0" as the literal first executable line of __init__.py (above any re-export block). ACCEPTANCE: \`uv build\` green; clean-venv \`python -c "import xtrax; print(xtrax.__version__)"\` prints 0.2.0; \`unzip -p dist/*.whl '**/METADATA' | grep '^Version:'\`==0.2.0; \`twine check dist/*\` passes; \`grep -rn '0\.1\.0' src/\` empty. Dispatch: fixer (judgment/cross-cutting). Spec: .praxia/docs/specs/260610_make-the-xtrax-jax-library-distribution.md\n\nREVIEWER CHECKLIST (fill in before running dw emit-sprint):\nVERIFY: cargo command or test that must pass — e.g. \`cargo nextest run -p <crate>\`\nVERIFY: observable assertion — e.g. "new test exists and calls the handler directly"\nVERIFY: no regression — e.g. "existing tests still pass"\nPASS if all VERIFY items are satisfied.\nFAIL if any VERIFY item is not met or untestable as written.`,
  );

// ===== TRACK C — Track C — [N3] Hybrid + lazy public API surface (populate empty __init__ + PEP562 __getattr__) (#1453) =========================
const trackC = () =>
  track(
    "1453",
    "Track C — [N3] Hybrid + lazy public API surface (populate empty __init__ + PEP562 __getattr__) (#1453)",
    `task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nDistribution-readiness DAG node N3 — ROOT GATE 2 (task 260610_xtrax-packaging). Populate empty training/data/tiling __init__.py with curated __all__; expose curated flat top-level (Trainer, Engine, AxisSpec, BatchPlan + output-sink names) via PEP 562 module-level __getattr__ lazy re-exports. ACCEPTANCE: \`from xtrax import Trainer, Engine\` works; all 11 subpackages export non-empty __all__; runnable cold-import bound (clean process, JAX not pre-imported): \`python -c 'import time; t=time.perf_counter(); import xtrax; e=time.perf_counter()-t; assert e<0.5, f"cold import {e:.2f}s exceeds 500ms"'\`; no JAX device init on bare import. Dispatch: fixer (taste call + reads real APIs). Spec: .praxia/docs/specs/260610_make-the-xtrax-jax-library-distribution.md\n\n[AUTO: no recon anchors — paste file:line references here, then write the full fixer prompt]\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nDistribution-readiness DAG node N3 — ROOT GATE 2 (task 260610_xtrax-packaging). Populate empty training/data/tiling __init__.py with curated __all__; expose curated flat top-level (Trainer, Engine, AxisSpec, BatchPlan + output-sink names) via PEP 562 module-level __getattr__ lazy re-exports. ACCEPTANCE: \`from xtrax import Trainer, Engine\` works; all 11 subpackages export non-empty __all__; runnable cold-import bound (clean process, JAX not pre-imported): \`python -c 'import time; t=time.perf_counter(); import xtrax; e=time.perf_counter()-t; assert e<0.5, f"cold import {e:.2f}s exceeds 500ms"'\`; no JAX device init on bare import. Dispatch: fixer (taste call + reads real APIs). Spec: .praxia/docs/specs/260610_make-the-xtrax-jax-library-distribution.md\n\nREVIEWER CHECKLIST (fill in before running dw emit-sprint):\nVERIFY: cargo command or test that must pass — e.g. \`cargo nextest run -p <crate>\`\nVERIFY: observable assertion — e.g. "new test exists and calls the handler directly"\nVERIFY: no regression — e.g. "existing tests still pass"\nPASS if all VERIFY items are satisfied.\nFAIL if any VERIFY item is not met or untestable as written.`,
  );

// ===== TRACK D — Track D — [N9] Configure PyPI + TestPyPI OIDC Trusted Publisher (human gate) (#1454) =========================
const trackD = () =>
  track(
    "1454",
    "Track D — [N9] Configure PyPI + TestPyPI OIDC Trusted Publisher (human gate) (#1454)",
    `task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nDistribution-readiness DAG node N9 — HUMAN GATE (task 260610_xtrax-packaging). Out-of-band prerequisite that blocks N7 publish; prevents pre-mortem-c (forgotten config -> hasty token). ACCEPTANCE (owner-verified): PyPI Trusted Publisher configured for owner/xtrax repo + publish.yml workflow name + environment; same configured on test.pypi.org. Marked done by the project owner only. Dispatch: user_handoff PCW template (manual-approval gate, no automated agent). Spec: .praxia/docs/specs/260610_make-the-xtrax-jax-library-distribution.md\n\n[AUTO: no recon anchors — paste file:line references here, then write the full fixer prompt]\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nDistribution-readiness DAG node N9 — HUMAN GATE (task 260610_xtrax-packaging). Out-of-band prerequisite that blocks N7 publish; prevents pre-mortem-c (forgotten config -> hasty token). ACCEPTANCE (owner-verified): PyPI Trusted Publisher configured for owner/xtrax repo + publish.yml workflow name + environment; same configured on test.pypi.org. Marked done by the project owner only. Dispatch: user_handoff PCW template (manual-approval gate, no automated agent). Spec: .praxia/docs/specs/260610_make-the-xtrax-jax-library-distribution.md\n\nREVIEWER CHECKLIST (fill in before running dw emit-sprint):\nVERIFY: cargo command or test that must pass — e.g. \`cargo nextest run -p <crate>\`\nVERIFY: observable assertion — e.g. "new test exists and calls the handler directly"\nVERIFY: no regression — e.g. "existing tests still pass"\nPASS if all VERIFY items are satisfied.\nFAIL if any VERIFY item is not met or untestable as written.`,
  );

// ===== TRACK E — Track E — [N2] Apache-2.0 LICENSE + pyproject metadata + py.typed-in-wheel (#1455) =========================
const trackE = () =>
  track(
    "1455",
    "Track E — [N2] Apache-2.0 LICENSE + pyproject metadata + py.typed-in-wheel (#1455)",
    `You are a delegating agent. Your task:\n1. Call mcp__praxia__dispatch with payload: {\n\x20\x20\'action\': \'create\',\n\x20\x20\'target\': "cursor",\n\x20\x20\'task_id\': args?.task_id ?? \'\',\n\x20\x20\'prompt\': <the prompt below>,\n\x20\x20\'execute\': true\n}\n2. Capture the returned dispatch id.\n3. Poll mcp__praxia__dispatch with action: \'status\' and that id roughly every 15 seconds.\n4. Terminal statuses are: completed | failed | killed | cancelled — stop on any of them.\n5. IMPORTANT: Poll for at most ~15 minutes (~60 attempts). The dispatch rescuer recycles\na stalled dispatch back to pending, so you MUST stop at your own deadline rather than\nwaiting indefinitely.\n6. On timeout: return \'verdict: needs_work\' with a note \'surface dispatch on cursor timed out\'.\n7. On terminal: return the dispatch result text verbatim.\n\nThe original prompt to dispatch:\n${`task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nDistribution-readiness DAG node N2 (task 260610_xtrax-packaging). Apache-2.0 LICENSE at root; license="Apache-2.0" SPDX + classifiers (Dev Status, Programming Language :: Python :: 3.13, Intended Audience :: Science/Research, License :: OSI Approved :: Apache Software License, Typing :: Typed) + [project.urls] + authors; src/xtrax/py.typed retained in wheel via hatch force-include. ACCEPTANCE: \`twine check dist/*\` complete metadata; \`unzip -l dist/*.whl | grep py.typed\` returns exactly one result; \`python -c "import importlib.resources as r, xtrax; assert (r.files('xtrax')/'py.typed').is_file()"\`. Dispatch: cursor. Pins pre-mortem-d.\n\n[AUTO: no recon anchors — paste file:line references here, then write the full fixer prompt]\n\n${EMITTER_CTX}`}`,
    `task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nDistribution-readiness DAG node N2 (task 260610_xtrax-packaging). Apache-2.0 LICENSE at root; license="Apache-2.0" SPDX + classifiers (Dev Status, Programming Language :: Python :: 3.13, Intended Audience :: Science/Research, License :: OSI Approved :: Apache Software License, Typing :: Typed) + [project.urls] + authors; src/xtrax/py.typed retained in wheel via hatch force-include. ACCEPTANCE: \`twine check dist/*\` complete metadata; \`unzip -l dist/*.whl | grep py.typed\` returns exactly one result; \`python -c "import importlib.resources as r, xtrax; assert (r.files('xtrax')/'py.typed').is_file()"\`. Dispatch: cursor. Pins pre-mortem-d.\n\nREVIEWER CHECKLIST (fill in before running dw emit-sprint):\nVERIFY: cargo command or test that must pass — e.g. \`cargo nextest run -p <crate>\`\nVERIFY: observable assertion — e.g. "new test exists and calls the handler directly"\nVERIFY: no regression — e.g. "existing tests still pass"\nPASS if all VERIFY items are satisfied.\nFAIL if any VERIFY item is not met or untestable as written.`,
  );

// ===== TRACK F — Track F — [N6] Fresh coverage gate + minimal 3.13 ci.yml (#1456) =========================
const trackF = () =>
  track(
    "1456",
    "Track F — [N6] Fresh coverage gate + minimal 3.13 ci.yml (#1456)",
    `You are a delegating agent. Your task:\n1. Call mcp__praxia__dispatch with payload: {\n\x20\x20\'action\': \'create\',\n\x20\x20\'target\': "cursor",\n\x20\x20\'task_id\': args?.task_id ?? \'\',\n\x20\x20\'prompt\': <the prompt below>,\n\x20\x20\'execute\': true\n}\n2. Capture the returned dispatch id.\n3. Poll mcp__praxia__dispatch with action: \'status\' and that id roughly every 15 seconds.\n4. Terminal statuses are: completed | failed | killed | cancelled — stop on any of them.\n5. IMPORTANT: Poll for at most ~15 minutes (~60 attempts). The dispatch rescuer recycles\na stalled dispatch back to pending, so you MUST stop at your own deadline rather than\nwaiting indefinitely.\n6. On timeout: return \'verdict: needs_work\' with a note \'surface dispatch on cursor timed out\'.\n7. On terminal: return the dispatch result text verbatim.\n\nThe original prompt to dispatch:\n${`task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nDistribution-readiness DAG node N6 (task 260610_xtrax-packaging). Minimal 3.13 ci.yml: ruff lint + ruff format-check + type-check + \`pytest --cov --cov-fail-under=90\`. Stale-artifact removal guaranteed upstream by N0. ACCEPTANCE: CI computes fresh coverage >=90% (real ~96.5%) and passes; ruff/type/test jobs green. Dispatch: cursor.\n\n[AUTO: no recon anchors — paste file:line references here, then write the full fixer prompt]\n\n${EMITTER_CTX}`}`,
    `task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nDistribution-readiness DAG node N6 (task 260610_xtrax-packaging). Minimal 3.13 ci.yml: ruff lint + ruff format-check + type-check + \`pytest --cov --cov-fail-under=90\`. Stale-artifact removal guaranteed upstream by N0. ACCEPTANCE: CI computes fresh coverage >=90% (real ~96.5%) and passes; ruff/type/test jobs green. Dispatch: cursor.\n\nREVIEWER CHECKLIST (fill in before running dw emit-sprint):\nVERIFY: cargo command or test that must pass — e.g. \`cargo nextest run -p <crate>\`\nVERIFY: observable assertion — e.g. "new test exists and calls the handler directly"\nVERIFY: no regression — e.g. "existing tests still pass"\nPASS if all VERIFY items are satisfied.\nFAIL if any VERIFY item is not met or untestable as written.`,
  );

// ===== TRACK G — Track G — [N4a] Autodoc plumbing (Sphinx conf.py + autosummary + RTD + docs CI -W) (#1457) =========================
const trackG = () =>
  track(
    "1457",
    "Track G — [N4a] Autodoc plumbing (Sphinx conf.py + autosummary + RTD + docs CI -W) (#1457)",
    `task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nDistribution-readiness DAG node N4a (task 260610_xtrax-packaging). Sphinx (autodoc+autosummary+napoleon, furo); conf.py; autosummary stubs; docs optional-dependency group in pyproject; API org by 11 subpackages with output-sink as top-level nav group; .readthedocs.yaml pins 3.13 + docs extra; docs CI job. ACCEPTANCE (pre-mortem-e parity): docs CI job runs \`sphinx-build -W -n\` in a FRESH venv built with \`uv sync --only-group docs\` (no dev deps) — same isolation .readthedocs.yaml uses; must be green before close. Dispatch: fixer (autodoc plumbing reads real API).\n\n[AUTO: no recon anchors — paste file:line references here, then write the full fixer prompt]\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nDistribution-readiness DAG node N4a (task 260610_xtrax-packaging). Sphinx (autodoc+autosummary+napoleon, furo); conf.py; autosummary stubs; docs optional-dependency group in pyproject; API org by 11 subpackages with output-sink as top-level nav group; .readthedocs.yaml pins 3.13 + docs extra; docs CI job. ACCEPTANCE (pre-mortem-e parity): docs CI job runs \`sphinx-build -W -n\` in a FRESH venv built with \`uv sync --only-group docs\` (no dev deps) — same isolation .readthedocs.yaml uses; must be green before close. Dispatch: fixer (autodoc plumbing reads real API).\n\nREVIEWER CHECKLIST (fill in before running dw emit-sprint):\nVERIFY: cargo command or test that must pass — e.g. \`cargo nextest run -p <crate>\`\nVERIFY: observable assertion — e.g. "new test exists and calls the handler directly"\nVERIFY: no regression — e.g. "existing tests still pass"\nPASS if all VERIFY items are satisfied.\nFAIL if any VERIFY item is not met or untestable as written.`,
  );

// ===== TRACK H — Track H — [N4b] Narrative docs prose (quickstart + architecture, seeded from internal specs) (#1458) =========================
const trackH = () =>
  track(
    "1458",
    "Track H — [N4b] Narrative docs prose (quickstart + architecture, seeded from internal specs) (#1458)",
    `You are a delegating agent. Your task:\n1. Call mcp__praxia__dispatch with payload: {\n\x20\x20\'action\': \'create\',\n\x20\x20\'target\': "cursor",\n\x20\x20\'task_id\': args?.task_id ?? \'\',\n\x20\x20\'prompt\': <the prompt below>,\n\x20\x20\'execute\': true\n}\n2. Capture the returned dispatch id.\n3. Poll mcp__praxia__dispatch with action: \'status\' and that id roughly every 15 seconds.\n4. Terminal statuses are: completed | failed | killed | cancelled — stop on any of them.\n5. IMPORTANT: Poll for at most ~15 minutes (~60 attempts). The dispatch rescuer recycles\na stalled dispatch back to pending, so you MUST stop at your own deadline rather than\nwaiting indefinitely.\n6. On timeout: return \'verdict: needs_work\' with a note \'surface dispatch on cursor timed out\'.\n7. On terminal: return the dispatch result text verbatim.\n\nThe original prompt to dispatch:\n${`task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nDistribution-readiness DAG node N4b (task 260610_xtrax-packaging). Quickstart (flat-import snippet), architecture + concepts chapters, seeded from internal specs .praxia/docs/specs/260604_xtrax-spec.md and 260608_xtrax-s5-sparse.md with internal refs stripped. ACCEPTANCE: pages build under \`sphinx-build -W -n\`; quickstart code block is doctest-executable. Dispatch: cursor (bounded prose).\n\n[AUTO: no recon anchors — paste file:line references here, then write the full fixer prompt]\n\n${EMITTER_CTX}`}`,
    `task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nDistribution-readiness DAG node N4b (task 260610_xtrax-packaging). Quickstart (flat-import snippet), architecture + concepts chapters, seeded from internal specs .praxia/docs/specs/260604_xtrax-spec.md and 260608_xtrax-s5-sparse.md with internal refs stripped. ACCEPTANCE: pages build under \`sphinx-build -W -n\`; quickstart code block is doctest-executable. Dispatch: cursor (bounded prose).\n\nREVIEWER CHECKLIST (fill in before running dw emit-sprint):\nVERIFY: cargo command or test that must pass — e.g. \`cargo nextest run -p <crate>\`\nVERIFY: observable assertion — e.g. "new test exists and calls the handler directly"\nVERIFY: no regression — e.g. "existing tests still pass"\nPASS if all VERIFY items are satisfied.\nFAIL if any VERIFY item is not met or untestable as written.`,
  );

// ===== TRACK I — Track I — [N5] Output-sink docs (io callbacks + orbax checkpoint) + re-export doctests in CI (#1459) =========================
const trackI = () =>
  track(
    "1459",
    "Track I — [N5] Output-sink docs (io callbacks + orbax checkpoint) + re-export doctests in CI (#1459)",
    `You are a delegating agent. Your task:\n1. Call mcp__praxia__dispatch with payload: {\n\x20\x20\'action\': \'create\',\n\x20\x20\'target\': "cursor",\n\x20\x20\'task_id\': args?.task_id ?? \'\',\n\x20\x20\'prompt\': <the prompt below>,\n\x20\x20\'execute\': true\n}\n2. Capture the returned dispatch id.\n3. Poll mcp__praxia__dispatch with action: \'status\' and that id roughly every 15 seconds.\n4. Terminal statuses are: completed | failed | killed | cancelled — stop on any of them.\n5. IMPORTANT: Poll for at most ~15 minutes (~60 attempts). The dispatch rescuer recycles\na stalled dispatch back to pending, so you MUST stop at your own deadline rather than\nwaiting indefinitely.\n6. On timeout: return \'verdict: needs_work\' with a note \'surface dispatch on cursor timed out\'.\n7. On terminal: return the dispatch result text verbatim.\n\nThe original prompt to dispatch:\n${`task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nDistribution-readiness DAG node N5 (task 260610_xtrax-packaging). Unified 'Output sinks' chapter (io callbacks vs orbax checkpoints) + per-surface autodoc; canonical \`from xtrax.io import ...\` path + io->engine.io re-export boundary stated explicitly. ACCEPTANCE: re-export usage examples are doctests run in CI (pytest --doctest-modules / sphinx doctest) and pass; README streaming + checkpointing sections present. Dispatch: cursor. (User-flagged output-sink surface in scope.)\n\n[AUTO: no recon anchors — paste file:line references here, then write the full fixer prompt]\n\n${EMITTER_CTX}`}`,
    `task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nDistribution-readiness DAG node N5 (task 260610_xtrax-packaging). Unified 'Output sinks' chapter (io callbacks vs orbax checkpoints) + per-surface autodoc; canonical \`from xtrax.io import ...\` path + io->engine.io re-export boundary stated explicitly. ACCEPTANCE: re-export usage examples are doctests run in CI (pytest --doctest-modules / sphinx doctest) and pass; README streaming + checkpointing sections present. Dispatch: cursor. (User-flagged output-sink surface in scope.)\n\nREVIEWER CHECKLIST (fill in before running dw emit-sprint):\nVERIFY: cargo command or test that must pass — e.g. \`cargo nextest run -p <crate>\`\nVERIFY: observable assertion — e.g. "new test exists and calls the handler directly"\nVERIFY: no regression — e.g. "existing tests still pass"\nPASS if all VERIFY items are satisfied.\nFAIL if any VERIFY item is not met or untestable as written.`,
  );

// ===== TRACK J — Track J — [N8] README + CHANGELOG + CONTRIBUTING + CITATION.cff + delete main.py (#1460) =========================
const trackJ = () =>
  track(
    "1460",
    "Track J — [N8] README + CHANGELOG + CONTRIBUTING + CITATION.cff + delete main.py (#1460)",
    `You are a delegating agent. Your task:\n1. Call mcp__praxia__dispatch with payload: {\n\x20\x20\'action\': \'create\',\n\x20\x20\'target\': "cursor",\n\x20\x20\'task_id\': args?.task_id ?? \'\',\n\x20\x20\'prompt\': <the prompt below>,\n\x20\x20\'execute\': true\n}\n2. Capture the returned dispatch id.\n3. Poll mcp__praxia__dispatch with action: \'status\' and that id roughly every 15 seconds.\n4. Terminal statuses are: completed | failed | killed | cancelled — stop on any of them.\n5. IMPORTANT: Poll for at most ~15 minutes (~60 attempts). The dispatch rescuer recycles\na stalled dispatch back to pending, so you MUST stop at your own deadline rather than\nwaiting indefinitely.\n6. On timeout: return \'verdict: needs_work\' with a note \'surface dispatch on cursor timed out\'.\n7. On terminal: return the dispatch result text verbatim.\n\nThe original prompt to dispatch:\n${`task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nDistribution-readiness DAG node N8 (task 260610_xtrax-packaging). README (tagline+badges -> why -> \`pip install xtrax\` (3.13) -> flat-import quickstart -> output-sink highlights -> docs links -> license); CHANGELOG (Keep-a-Changelog, [0.2.0] records 0.1.0->0.2.0); CONTRIBUTING (uv/ruff/coverage/tag-publish); CITATION.cff; delete main.py. ACCEPTANCE: README long-description renders on TestPyPI; badges resolve; main.py gone; CITATION version==0.2.0. Dispatch: cursor.\n\n[AUTO: no recon anchors — paste file:line references here, then write the full fixer prompt]\n\n${EMITTER_CTX}`}`,
    `task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nDistribution-readiness DAG node N8 (task 260610_xtrax-packaging). README (tagline+badges -> why -> \`pip install xtrax\` (3.13) -> flat-import quickstart -> output-sink highlights -> docs links -> license); CHANGELOG (Keep-a-Changelog, [0.2.0] records 0.1.0->0.2.0); CONTRIBUTING (uv/ruff/coverage/tag-publish); CITATION.cff; delete main.py. ACCEPTANCE: README long-description renders on TestPyPI; badges resolve; main.py gone; CITATION version==0.2.0. Dispatch: cursor.\n\nREVIEWER CHECKLIST (fill in before running dw emit-sprint):\nVERIFY: cargo command or test that must pass — e.g. \`cargo nextest run -p <crate>\`\nVERIFY: observable assertion — e.g. "new test exists and calls the handler directly"\nVERIFY: no regression — e.g. "existing tests still pass"\nPASS if all VERIFY items are satisfied.\nFAIL if any VERIFY item is not met or untestable as written.`,
  );

// ===== TRACK K — Track K — [N7] publish.yml OIDC Trusted Publishing (TestPyPI -> PyPI, tag-triggered) (#1461) =========================
const trackK = () =>
  track(
    "1461",
    "Track K — [N7] publish.yml OIDC Trusted Publishing (TestPyPI -> PyPI, tag-triggered) (#1461)",
    `You are a delegating agent. Your task:\n1. Call mcp__praxia__dispatch with payload: {\n\x20\x20\'action\': \'create\',\n\x20\x20\'target\': "cursor",\n\x20\x20\'task_id\': args?.task_id ?? \'\',\n\x20\x20\'prompt\': <the prompt below>,\n\x20\x20\'execute\': true\n}\n2. Capture the returned dispatch id.\n3. Poll mcp__praxia__dispatch with action: \'status\' and that id roughly every 15 seconds.\n4. Terminal statuses are: completed | failed | killed | cancelled — stop on any of them.\n5. IMPORTANT: Poll for at most ~15 minutes (~60 attempts). The dispatch rescuer recycles\na stalled dispatch back to pending, so you MUST stop at your own deadline rather than\nwaiting indefinitely.\n6. On timeout: return \'verdict: needs_work\' with a note \'surface dispatch on cursor timed out\'.\n7. On terminal: return the dispatch result text verbatim.\n\nThe original prompt to dispatch:\n${`task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nDistribution-readiness DAG node N7 — terminal publish (task 260610_xtrax-packaging). publish.yml on v* tag-push builds sdist+wheel, uploads via OIDC Trusted Publishing (pypa/gh-action-pypi-publish, no token); TestPyPI staging stage runs FIRST. ACCEPTANCE: pre-release tag green to TestPyPI + clean-venv install succeeds; v0.2.0 green to PyPI; repo secrets show no long-lived PyPI token. Depends on N9 human OIDC config gate. Dispatch: cursor.\n\n[AUTO: no recon anchors — paste file:line references here, then write the full fixer prompt]\n\n${EMITTER_CTX}`}`,
    `task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nDistribution-readiness DAG node N7 — terminal publish (task 260610_xtrax-packaging). publish.yml on v* tag-push builds sdist+wheel, uploads via OIDC Trusted Publishing (pypa/gh-action-pypi-publish, no token); TestPyPI staging stage runs FIRST. ACCEPTANCE: pre-release tag green to TestPyPI + clean-venv install succeeds; v0.2.0 green to PyPI; repo secrets show no long-lived PyPI token. Depends on N9 human OIDC config gate. Dispatch: cursor.\n\nREVIEWER CHECKLIST (fill in before running dw emit-sprint):\nVERIFY: cargo command or test that must pass — e.g. \`cargo nextest run -p <crate>\`\nVERIFY: observable assertion — e.g. "new test exists and calls the handler directly"\nVERIFY: no regression — e.g. "existing tests still pass"\nPASS if all VERIFY items are satisfied.\nFAIL if any VERIFY item is not met or untestable as written.`,
  );

// ===== TRACK L — Track L — [N10] Epic-done release-readiness gate (convergence) (#1462) =========================
const trackL = () =>
  track(
    "1462",
    "Track L — [N10] Epic-done release-readiness gate (convergence) (#1462)",
    `task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nDistribution-readiness DAG node N10 — TERMINAL SINK (task 260610_xtrax-packaging). Closes the dangling docs/hygiene subgraph: the epic is NOT done until publish AND docs AND hygiene all land. ACCEPTANCE (all must hold): \`pip install xtrax\` from PyPI installs v0.2.0 and imports with __version__==0.2.0; hosted docs render full API + output-sink + quickstart; README renders on PyPI project page; main.py gone; CI green on main; coverage gate >=90% fresh. Dispatch: complete_workflow PCW template (convergence/rollup gate).\n\n[AUTO: no recon anchors — paste file:line references here, then write the full fixer prompt]\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nDistribution-readiness DAG node N10 — TERMINAL SINK (task 260610_xtrax-packaging). Closes the dangling docs/hygiene subgraph: the epic is NOT done until publish AND docs AND hygiene all land. ACCEPTANCE (all must hold): \`pip install xtrax\` from PyPI installs v0.2.0 and imports with __version__==0.2.0; hosted docs render full API + output-sink + quickstart; README renders on PyPI project page; main.py gone; CI green on main; coverage gate >=90% fresh. Dispatch: complete_workflow PCW template (convergence/rollup gate).\n\nREVIEWER CHECKLIST (fill in before running dw emit-sprint):\nVERIFY: cargo command or test that must pass — e.g. \`cargo nextest run -p <crate>\`\nVERIFY: observable assertion — e.g. "new test exists and calls the handler directly"\nVERIFY: no regression — e.g. "existing tests still pass"\nPASS if all VERIFY items are satisfied.\nFAIL if any VERIFY item is not met or untestable as written.`,
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
