# xtrax Distribution-Readiness — Backlog DAG (rev 3, post adversarial audit round 2)

## rev3 changelog (audit round 2, NEEDS_WORK → 2 required fixes applied)
- NEW N10 convergence/epic-done gate, depends_on [N7, N4b, N5, N8] — closes the dangling docs/hygiene subgraph so the epic cannot be "done" with a published-but-undocumented package. Single terminal sink.
- N9 re-modeled: praxia `backlog add` has NO `agent` field (routing is sprint-composition metadata, not a backlog field), so N9 stays a real node but is annotated to route to the `user_handoff` PCW template (manual-approval gate) — not an automated agent. The N7→N9 dependency edge is the actual safeguard and is preserved.

---
# (rev 2 base below, with rev3 edits inline)

task_id: 260610_xtrax-packaging
source spec: .praxia/docs/specs/260610_make-the-xtrax-jax-library-distribution.md
contemplex session: 3413ff7e
workspace: /home/marielle/projects/xtrax

## Changelog
- rev2 (audit round 1, NEEDS_WORK → applied all 6 required + 2 suggested fixes):
  1. N6 depends_on now [N0, N1] — dropped phantom N3 edge.
  2. NEW N0 — extract stale-.coverage cleanup as standalone node (enforces pre-mortem-b ordering at DAG level).
  3. N4 split → N4a (autodoc plumbing, fixer) + N4b (narrative prose, cursor); N5/N8 re-wired to N4a.
  4. NEW N9 — human-gate node for OIDC Trusted Publisher config; N7 depends on it.
  5. N3 cold-import acceptance is now a runnable command.
  6. N4a carries a clean-venv docs-extra-only parity assertion (pre-mortem-e).
  7. N1 category bug → infrastructure (resolves version-mismatch bug as part of infra work).
  8. N2 carries explicit `unzip` py.typed-in-wheel assertion (pre-mortem-d).

## DAG shape (12 nodes)

```
ROOTS (depends_on: []):
  N0  stale-.coverage cleanup            (cursor, debt, quick)
  N1  version single-sourcing + verify   (fixer, infrastructure, standard)   [GATE]
  N3  hybrid+lazy public API surface     (fixer, feature, standard)          [GATE]
  N9  OIDC Trusted Publisher config       (user_handoff template, infrastructure, quick)  [HUMAN GATE]

FAN-OUT:
  N2   license + metadata + py.typed-in-wheel   depends: [N1]
  N6   fresh coverage gate + ci.yml             depends: [N0, N1]
  N4a  autodoc plumbing (conf.py, RTD, CI -W)   depends: [N3]
  N4b  narrative prose (quickstart/arch)        depends: [N4a]
  N5   output-sink docs + re-export doctests    depends: [N4a]
  N8   README + CHANGELOG + CONTRIBUTING + CITATION + rm main.py  depends: [N1, N3, N4a]
  N7   publish.yml OIDC TestPyPI->PyPI          depends: [N1, N2, N6, N9]

CONVERGENCE (terminal sink):
  N10  epic-done release-readiness gate         depends: [N7, N4b, N5, N8]
```

Critical path: N1 → N6 → N7 → N10  (and N1 → N2 → N7 → N10).  Docs chain: N3 → N4a → {N4b, N5, N8} → N10.
Acyclic; 4 roots (N0,N1,N3,N9); N10 is the single terminal sink.

## Nodes (praxia backlog payloads)

### N0 — stale-.coverage cleanup  [enforces pre-mortem-b at DAG level]
- priority: P1 · category: debt · difficulty: quick · depends_on: [] · agent: cursor
- THEN: `git rm --cached .coverage` (and coverage.xml if tracked); `.gitignore` contains `.coverage` and `coverage.xml`; verify `git ls-files | grep -cE '(^|/)\.coverage'` == 0.
- Rationale: a separate atomic node guarantees removal lands BEFORE N6 wires `--cov-fail-under`, so the 27.5% stale artifact can never red-bar the gate. (Could not be left as prose inside N6 — a fixer could reorder.)

### N1 — version single-sourcing + wheel verification  [ROOT GATE 1]
- priority: P1 · category: infrastructure · difficulty: standard · depends_on: [] · agent: fixer
- note: resolves the 0.1.0/0.2.0 version-mismatch bug as part of this infrastructure task.
- Single-source via `[tool.hatch.version] path = "src/xtrax/__init__.py"` + `dynamic = ["version"]`; reconcile to `0.2.0`; keep `__version__ = "0.2.0"` as the literal first executable line of `__init__.py` (above any re-export block).
- THEN: `uv build` green; clean-venv `python -c "import xtrax; print(xtrax.__version__)"` prints `0.2.0`; `unzip -p dist/*.whl '**/METADATA' | grep '^Version:'` == `0.2.0`; `twine check dist/*` passes; `grep -rn '0\.1\.0' src/` empty.

### N3 — hybrid + lazy public API surface  [ROOT GATE 2]
- priority: P1 · category: feature · difficulty: standard · depends_on: [] · agent: fixer
- Populate empty `training/`, `data/`, `tiling/` `__init__.py` with curated `__all__`; expose curated flat top-level (Trainer, Engine, AxisSpec, BatchPlan, + output-sink names) via PEP 562 module-level `__getattr__` lazy re-exports.
- THEN: `from xtrax import Trainer, Engine` works; all 11 subpackages export non-empty `__all__`; cold-import bound is a RUNNABLE assertion (clean process, JAX not pre-imported):
  `python -c 'import time; t=time.perf_counter(); import xtrax; e=time.perf_counter()-t; assert e<0.5, f"cold import {e:.2f}s exceeds 500ms"'`
  and assert no JAX device init triggered by the bare import.

### N9 — OIDC Trusted Publisher configuration  [HUMAN GATE]
- priority: P1 · category: infrastructure · difficulty: quick · depends_on: []
- dispatch: `user_handoff` PCW template (manual-approval gate — no automated agent). praxia `backlog add` has no agent field; this routing is applied at sprint composition.
- THEN (owner-verified): PyPI Trusted Publisher configured for `owner/xtrax` repo + `publish.yml` workflow name + environment; same configured on test.pypi.org. Marked done by the project owner only.
- Rationale: first-class node so the out-of-band prereq is visible/schedulable and blocks N7; prevents pre-mortem-c (forgotten config → hasty token).

### N2 — license + metadata + py.typed-in-wheel
- priority: P1 · category: infrastructure · difficulty: quick · depends_on: [N1] · agent: cursor
- Apache-2.0 LICENSE at root; `license="Apache-2.0"` SPDX + classifiers (Dev Status, `Programming Language :: Python :: 3.13`, `Intended Audience :: Science/Research`, `License :: OSI Approved :: Apache Software License`, `Typing :: Typed`) + `[project.urls]` + `authors`; `src/xtrax/py.typed` retained in wheel via hatch `force-include`.
- THEN: `twine check dist/*` complete metadata; `unzip -l dist/*.whl | grep py.typed` returns exactly one result; py.typed asserted present in installed wheel via `python -c "import importlib.resources as r, xtrax; assert (r.files('xtrax')/'py.typed').is_file()"`.

### N6 — fresh coverage gate + ci.yml
- priority: P1 · category: infrastructure · difficulty: standard · depends_on: [N0, N1] · agent: cursor
- Minimal 3.13 `ci.yml`: ruff lint + ruff format-check + type-check + `pytest --cov --cov-fail-under=90`.
- THEN: CI computes fresh coverage ≥ 90% (real ≈ 96.5%) and passes; ruff/type/test jobs green. (Stale-artifact removal guaranteed upstream by N0.)

### N4a — autodoc plumbing
- priority: P2 · category: infrastructure · difficulty: standard · depends_on: [N3] · agent: fixer
- Sphinx project (autodoc+autosummary+napoleon, furo); `conf.py`; autosummary stubs; docs optional-dependency group in pyproject; API org by 11 subpackages with output-sink as a top-level nav group; `.readthedocs.yaml` pins 3.13 + docs extra; docs CI job.
- THEN (pre-mortem-e parity): docs CI job runs `sphinx-build -W -n` in a FRESH venv built with `uv sync --only-group docs` (no dev deps) — same isolation model `.readthedocs.yaml` uses; this CI job must be green before N4a closes. (Converts the external-RTD dependency into a locally runnable assertion.)

### N4b — narrative prose
- priority: P2 · category: infrastructure · difficulty: standard · depends_on: [N4a] · agent: cursor
- Quickstart (flat-import snippet), architecture + concepts chapters, seeded from internal specs `.praxia/docs/specs/260604_xtrax-spec.md` and `260608_xtrax-s5-sparse.md` with internal refs stripped.
- THEN: pages build under `sphinx-build -W -n`; quickstart code block is doctest-executable.

### N5 — output-sink docs + re-export doctests
- priority: P2 · category: infrastructure · difficulty: standard · depends_on: [N4a] · agent: cursor
- Unified "Output sinks" chapter (io callbacks vs orbax checkpoints) + per-surface autodoc; canonical `from xtrax.io import ...` path + io→engine.io re-export boundary stated explicitly.
- THEN: re-export usage examples are doctests run in CI (`pytest --doctest-modules` / sphinx doctest) and pass; README streaming + checkpointing sections present.

### N8 — README + CHANGELOG + CONTRIBUTING + CITATION + delete main.py
- priority: P2 · category: infrastructure · difficulty: standard · depends_on: [N1, N3, N4a] · agent: cursor
- README (tagline+badges → why → `pip install xtrax` (3.13) → flat-import quickstart → output-sink highlights → docs links → license); CHANGELOG (Keep-a-Changelog, `[0.2.0]` records 0.1.0→0.2.0); CONTRIBUTING (uv/ruff/coverage/tag-publish); CITATION.cff; delete `main.py`.
- THEN: README long-description renders on TestPyPI; badges resolve; `main.py` gone; CITATION version == `0.2.0`.

### N7 — publish.yml OIDC TestPyPI → PyPI  [terminal sink]
- priority: P1 · category: infrastructure · difficulty: standard · depends_on: [N1, N2, N6, N9] · agent: cursor
- `publish.yml` on `v*` tag-push builds sdist+wheel, uploads via OIDC Trusted Publishing (`pypa/gh-action-pypi-publish`, no token); TestPyPI staging stage runs FIRST.
- THEN: pre-release tag green to TestPyPI + clean-venv install succeeds; `v0.2.0` green to PyPI; `grep -r` of repo secrets shows no long-lived PyPI token.

### N10 — epic-done release-readiness gate  [TERMINAL SINK]
- priority: P1 · category: audit · difficulty: quick · depends_on: [N7, N4b, N5, N8]
- dispatch: `complete_workflow` PCW template (convergence/rollup gate).
- Closes the dangling docs/hygiene subgraph: the epic is NOT done until publish AND docs AND hygiene all land.
- THEN (all must hold): `v0.2.0` is installable via `pip install xtrax` from PyPI and imports with `__version__==0.2.0`; hosted docs render the full API + output-sink + quickstart; README renders on the PyPI project page; `main.py` is gone; CI green on main; coverage gate ≥ 90% fresh.

## Dispatch routing — feeds dw emit + cursor dispatch rewiring
- fixer (judgment/cross-cutting/reads real APIs): N1, N3, N4a.
- cursor (mechanical/template-fill/isolated): N0, N2, N6, N4b, N5, N8, N7.
- user_handoff template (human gate): N9.
- complete_workflow template (convergence): N10.
