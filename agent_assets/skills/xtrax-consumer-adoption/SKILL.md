---
name: xtrax-consumer-adoption
description: "Use when an agent swarm migrates a dependent project onto xtrax.pack, xtrax.eda or xtrax campaign tooling (debts #3257, #3143, #3142), from spec through adversarial review, PR and review cycles. For the APIs themselves, use using-xtrax."
---

# Migrating a dependent project onto xtrax

The procedure a swarm follows to move one project onto xtrax, from spec to a reviewed PR. The swarm stops at the PR; the owner merges.

Start a run with:

```text
/goal /teamwork /boost
Use skill xtrax-consumer-adoption. PROJECT=aminx TASK_ID=261105_xtrax-adopt-aminx WORKSTREAMS=pack,eda,campaign DEBTS=#3147
```

`PROJECT` is a directory under `~/projects`. `DEBTS` lists the project's own debts to fold in, or `none`. Put `TASK_ID` in every log call, dispatch, commit and PR. A "move xtrax pin to 0.4.1b6" debt is superseded by the `>=0.4.1b7` target.

## Inputs

- xtrax debts: #3257 pack, #3143 eda, #3142 campaign. Read them with `debt(action="list", payload={"workspace":"/home/marielle/projects/xtrax","query":"Dependents:","detail":"full"})`.
- Consumer scan, dated 0.4.1b6 and pattern-based, so verify it: `git -C ~/projects/xtrax show origin/docs/xtrax-dependent-scope:.praxia/docs/research/261030_xtrax-0-4-1b6-dependent-scope.md`.

## Role

The lead orchestrates, delegates, and keeps only conclusions. No role grades its own output. State lives in `.praxia/loop/<TASK_ID>/state.md`: per item, status, branch, PR, bth run id, verdict and blocker. Read it at the start of each iteration and update it at the end.

## Skills to load

Load each by name before the phase that needs it:

| When | Skills |
|---|---|
| Always | `orchestration`, `praxia_subagents`, `using-xtrax`, `xtrax-io-on-shared-filesystems`, `using-bathos`, `using-jax`, `maraxiom-review-pages` |
| On a cluster | `myxcel-slurm-submit`, `myxcel-slurm-monitor` |
| Bio projects | `prose-biosafe`, `affigit-components` |
| MD-derived numbers | `md-analysis` |

Freshness check: `using-xtrax/references/adoption.md` contains "Per-unit outputs". If it does not, the plugin is stale: stop and report. If another skill will not load, name it in the report and read its source (`~/projects/affigit/agent_assets/skills/affigit-components/SKILL.md` for the one Antigravity lacks today).

## P0: premises and scope

1. Base everything on a fresh `origin/main`; local checkouts can be stale dogfood branches. Check `git status` first and leave the owner's working tree alone.
2. Give each workstream a verdict from the project's code, with file:line evidence: ADOPT, NOT-APPLICABLE or BLOCKED.
   - **pack** applies where finished per-unit output directories have a completion stamp or a done-check, or where file-count caps bind.
   - **eda** applies only where the project builds xtrax plans: `xtrax.eda` inspects `BatchPlan` schedules. Figure-data checks belong to maraxiom `scientific-eda`.
   - **campaign** is BLOCKED for a project that needs a sweep runner, unless a re-check of `origin/main` finds one in `src/xtrax`. See [Campaign surface](#campaign-surface).
3. Create one backlog epic with child items and dependency edges, in this project's workspace, unless one exists. Record the ids in the spec.

### Campaign surface

As of 0.4.1b7:

| Code | Ships in the wheel | What it is |
|---|---|---|
| `xtrax.loop.campaign_approval_gate` (`assert_campaign_approved`) | Yes | Approval gate; aminx-ensemble uses it |
| `xtrax.run.unit_index_adapters` | Yes, internal for now | Reads consumer per-unit records into the unit index |
| `controller/loop_run.py` (`run_campaign_loop`), `controller/bathos_campaign_adapter.py` | **No**: the wheel packages `src/xtrax` only | The #2181 algorithm-evolution loop: candidates, ratchets, invariant manifests. Not a sweep runner |

A project may adopt the gate where it launches approval-bound campaigns. For a sweep runner, note the BLOCKED verdict on the project's epic with this evidence. Build no runner, import nothing from `controller/`, and leave local loops such as aminx's `src/aminx/host/campaign.py` in place. Design unit identity so a later runner can treat units as work items.

## P1: spec

File it with `docs(action="add", payload={category:"specs", workspace:<abs cwd>, ...})`. It contains:

- An assumption table tagging each claim about code, tools, data layouts or APIs as verified (read at file:line, or probed) or unverified. Read the code, not its docstring, README or a subagent's summary.
- For pack, an old-vs-new table of every edge case in the old writer: failure stamps, regenerated loose files, empty units, `OSError` wrapping, what resume reuses, and existing on-disk units (reader compatibility or migration). alynxr #49 lost six of these on first pass.
- A pin plan: `xtrax[pack]>=0.4.1b7` and/or `xtrax[eda]`. Check the jax range it forces, any wheel override, and vendored copies a pin bump does not reach. If the project does not depend on xtrax, weigh the jax dependency against embedding `xtrax.pack.node_source()`. If xtrax becomes a core dependency and the project uses `xtrax.run` zarr modules, require zarr directly: those modules import without it and raise a bare `ImportError` when called.
- For consumer stamps, a thin shim binding `record_name=` and `check=`, as `alynxr.unit_pack` does.
- A resume design and verification plan, with a control proving each new check can fail. Existing bathos `[outcomes]` stay unchanged.

## P2: adversarial review and spikes, until coherent

Each round, as separate agents:

1. **Challenger** lists incoherences with section, quoted text, severity (FATAL, MAJOR or MINOR) and file:line evidence.
2. **Defender** rebuts from spec and source, conceding only real gaps.
3. **Adjudicator** rules accept or reject, with a reason.
4. **Editor** applies accepted findings only, as the smallest edit.

A finding resting on an unverified assumption gets a throwaway spike before the ruling, and the tag updates. A spike whose number gets cited is re-run as a tracked script with a sidecar.

Converged means one round with no accepted FATAL or MAJOR, and the lead has checked every citation at source; a swarm's own CONVERGED has been wrong before. The cap is 5 rounds. If it has not converged by then, stop, list what is open, and implement nothing. Log rounds in `.praxia/docs/audits/`.

## P3: implement

One worktree and branch per workstream off fresh `origin/main`, with the pin bump first if the others need it. Tests come first and are shown red for the right reason. Stage files by name; `.agents/` and `.praxia` telemetry stay uncommitted.

- **Pre-registration:** every run producing a finding gets a tracked script and a committed `.bth.toml` with `[outcomes]` in a commit before it runs. Use `bth`, never `uv run bth`. Verify from the run record, not the exit code.
- **pack acceptance:** units are committed with `write_unit_record` or a stamp via `record_name=` and `check=`; resume reuses a unit only when its `content_digest` verifies, and a corrupted unit is refused (the control); packed members read back byte-identical.
- **eda acceptance:** plan panels render from `xtrax.eda`, and the manifest sha256 digests verify against the export.
- **Compute:** run one test file or `-k` locally, with threads capped; a whole JAX or pytest suite kills the local box, so heavy suites go to titanix or Engaging. Long runs use per-unit completion stamps, one process and timeout per chunk, sha256-matched resume, and a profile of one unit at production shapes with a wall estimate in the sidecar. A Python loop over replicates, resamples, pairs, cells or frames is a defect until shown otherwise.
- **Derived data:** locations resolve argument > env > `[tool.<name>]` > XDG config > off, with no default path in library code.
- **Findings a person will judge** go on a private `mrx review build` page, audited first with `mrx review audit`.

## Bio projects: py2Dmol

Assess in P1 and implement in P3. py2Dmol is the viewer, not a typo for py3Dmol, and structure views do not fall back to matplotlib. Wherever the project shows structures, per-residue tracks, contacts or sites, the spec decides between built-in views, a py2Dmol renderer plugin, or not applicable, with one line of reasoning.

Read `~/repos/py2Dmol/CLAUDE.md` and `docs/PLUGINS.md` first. The agent skills are on an unmerged branch; if they are not installed, read them with `git -C ~/repos/py2Dmol show fork/agent-surface:agent_assets/skills/<skill>/SKILL.md`, where `<skill>` is `using-py2Dmol`, `developing-py2Dmol-plugins`, `augmenting-views` or `embedding-py2Dmol`.

- A plugin fits geometry that is not a residue: isosurfaces, site markers, marker-to-residue contacts, MI or coupling edges. If built-ins do the job, write no plugin.
- PyPI 1.6.5 and maraxiom's vendored copy have no `add_plugin`. Pin a git SHA at or after `3f9e851`, and before designing, assert `py2Dmol.__file__` and `hasattr(py2Dmol.viewer.view, "add_plugin")` in the project's env.
- Plugin contract: `apiVersion` 1, register through `window.py2dmolPlugins` (never `py2dmolMolParts`), over-budget geometry throws, no legend or `remove_plugin` yet. Take the test pattern from `tests/plugin_browser.py`. The plugin lives in the project; `~/repos/py2Dmol` is read-only. Pin the structure, bundle and plugin sha256s so figures are reproducible.

Bio prose follows `prose-biosafe`.

## P4: push and PR

Push the branch and open one PR per workstream against `main`, naming the debts, epic, spec, bth run ids, and what CI does and does not verify. Never push to `main`, force-push, merge or close a PR.

## P5: review and remediate, until satisfied

Each round, fresh reviewers who did not write the code review the full diff against the spec: correctness, the edge-case table, false greens, and whether tests can fail. Findings carry file:line, a failure scenario and a severity. Verify each before acting; a fixer applies confirmed ones on the same branch. Then read CI's actual failures and logs, not just its status.

Satisfied means one full round with no confirmed BLOCKER or MAJOR, and CI green. The cap is 5 rounds; then stop and list what is open. After two failed attempts at the same check, stop that item, record what was tried, and move on.

## Boundaries

- `~/projects/xtrax`, other projects and `~/repos/*` are read-only. An xtrax gap gets a debt filed from an xtrax-cwd session or a note in the report.
- Questions for the owner go in `.praxia/loop/<TASK_ID>/owner_questions.md` with a recommended default. The swarm does not answer them itself and keeps working on whatever they do not block.

## Done

Report the workstream verdicts with evidence; epic id; spec and audit log paths; rounds to converge at each gate; PR URLs; bth run ids; the py2Dmol decision; debts to resolve or re-scope (#3257, #3142, #3143 and project debts); and open owner questions. A workstream is complete only with its verification record.
