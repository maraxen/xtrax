---
name: xtrax-consumer-adoption
description: "Use when migrating a dependent project onto xtrax.pack, xtrax.eda or xtrax campaign tooling (debts #3257, #3143, #3142), from spec through adversarial review, PR and review cycles. For the APIs themselves, use using-xtrax."
---

# Migrating a dependent project onto xtrax

When you load this skill, you are the lead orchestrator moving one dependent project onto xtrax, from spec to a reviewed PR (e.g. `/goal /teamwork /boost migrate aminx onto xtrax pack and eda`). Drive P0–P5 yourself and stop at the PR; the owner merges.

## Inputs

Derive your inputs from the user's request and the target repo, asking the user only for what you cannot determine from either:

- **Project (`<project>`):** Identify the target directory under `~/projects/<project>` from the user's prompt or current working directory. Ask the user only if neither names it.
- **Task ID (`<task_id>`):** Mint `YYMMDD_xtrax-adopt-<project>` from today's date (unless the user supplied one) and include it in every log call, subagent dispatch, commit message, and PR.
- **xtrax debts and consumer scan:** Read xtrax debts #3257 (pack), #3143 (eda), and #3142 (campaign) with `debt(action="list", payload={"workspace":"/home/marielle/projects/xtrax","query":"Dependents:","detail":"full"})`. Read the pattern-based consumer scan (dated 0.4.1b6, so verify every claim in code): `git -C ~/projects/xtrax show origin/docs/xtrax-dependent-scope:.praxia/docs/research/261030_xtrax-0-4-1b6-dependent-scope.md`.
- **Workstreams (`pack`, `eda`, `campaign`):** Treat any workstreams the user named as scope hints, and evaluate applicability of `pack`, `eda`, and `campaign` against the project's code in P0.
- **Project debts:** List open debts in `~/projects/<project>` (`debt(action="list", payload={"workspace":"/home/marielle/projects/<project>"})`) and fold in those touching xtrax, packing, batch plans, or campaigns, plus any the user named. A "move xtrax pin to 0.4.1b6" debt is superseded by the `>=0.4.1b7` target. Ask the user only if folding in a borderline debt is ambiguous.

## Role

Act as lead: orchestrate, delegate, and keep only conclusions. No role grades its own output—dispatch challenger, defender, adjudicator, editor, implementers, and reviewers as separate agents. Create and maintain `.praxia/loop/<task_id>/state.md` recording per-item status, branch, PR, `bth` run id, verdict, and blocker. Read it at the start of each iteration and update it at the end.

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

File the spec with `docs(action="add", payload={category:"specs", workspace:<abs cwd>, ...})`. Include:

- An assumption table tagging each claim about code, tools, data layouts or APIs as verified (read at file:line, or probed) or unverified. Read the code, not its docstring, README or a subagent's summary.
- For pack, an old-vs-new table of every edge case in the old writer: failure stamps, regenerated loose files, empty units, `OSError` wrapping, what resume reuses, and existing on-disk units (reader compatibility or migration). alynxr #49 lost six of these on first pass.
- A pin plan: `xtrax[pack]>=0.4.1b7` and/or `xtrax[eda]`. Check the jax range it forces, any wheel override, and vendored copies a pin bump does not reach. If the project does not depend on xtrax, weigh the jax dependency against embedding `xtrax.pack.node_source()`. If xtrax becomes a core dependency and the project uses `xtrax.run` zarr modules, require zarr directly: those modules import without it and raise a bare `ImportError` when called.
- For consumer stamps, a thin shim binding `record_name=` and `check=`, as `alynxr.unit_pack` does.
- A resume design and verification plan, with a control proving each new check can fail. Existing bathos `[outcomes]` stay unchanged.

## P2: adversarial review and spikes, until coherent

Each round, dispatch as separate agents:

1. **Challenger** lists incoherences with section, quoted text, severity (FATAL, MAJOR or MINOR) and file:line evidence.
2. **Defender** rebuts from spec and source, conceding only real gaps.
3. **Adjudicator** rules accept or reject, with a reason.
4. **Editor** applies accepted findings only, as the smallest edit.

Run a throwaway spike before the ruling on any finding resting on an unverified assumption, and update the tag. Re-run any spike whose number gets cited as a tracked script with a sidecar.

Converged means one round with no accepted FATAL or MAJOR, after you have checked every citation at source; a swarm's own CONVERGED has been wrong before. The cap is 5 rounds. If it has not converged by then, stop, list what is open, and implement nothing. Log rounds in `.praxia/docs/audits/`.

## P3: implement

Create one worktree and branch per workstream off fresh `origin/main`, with the pin bump first if the others need it, and dispatch separate implementer agents. Write tests first and show them red for the right reason. Stage files by name; keep `.agents/` and `.praxia` telemetry uncommitted.

- **Pre-registration:** every run producing a finding gets a tracked script and a committed `.bth.toml` with `[outcomes]` in a commit before it runs. Use `bth`, never `uv run bth`. Verify from the run record, not the exit code.
- **pack acceptance:** units are committed with `write_unit_record` or a stamp via `record_name=` and `check=`; resume reuses a unit only when its `content_digest` verifies, and a corrupted unit is refused (the control); packed members read back byte-identical.
- **eda acceptance:** plan panels render from `xtrax.eda`, and the manifest sha256 digests verify against the export.
- **Compute:** run one test file or `-k` locally, with threads capped; a whole JAX or pytest suite kills the local box, so heavy suites go to titanix or Engaging. Long runs use per-unit completion stamps, one process and timeout per chunk, sha256-matched resume, and a profile of one unit at production shapes with a wall estimate in the sidecar. A Python loop over replicates, resamples, pairs, cells or frames is a defect until shown otherwise.
- **Derived data:** locations resolve argument > env > `[tool.<name>]` > XDG config > off, with no default path in library code.
- **Findings a person will judge** go on a private `mrx review build` page, audited first with `mrx review audit`.

## Bio projects: py2Dmol

Assess in P1 and implement in P3. py2Dmol is the viewer, not a typo for py3Dmol, and structure views do not fall back to matplotlib. Wherever the project shows structures, per-residue tracks, contacts or sites, decide in the spec between built-in views, a py2Dmol renderer plugin, or not applicable, with one line of reasoning.

Read `~/repos/py2Dmol/CLAUDE.md` and `docs/PLUGINS.md` first. The agent skills are on unmerged draft `sokrypton/py2Dmol#31` (rebased onto upstream main 2026-11-05), so check their claims against current `docs/PLUGINS.md`. If they are not installed, read them from the remote branch, since local refs go stale: `gh api -H 'Accept: application/vnd.github.raw' 'repos/maraxen/py2Dmol/contents/agent_assets/skills/<skill>/SKILL.md?ref=agent-surface'`, where `<skill>` is `using-py2Dmol`, `developing-py2Dmol-plugins`, `augmenting-views` or `embedding-py2Dmol`.

- A plugin fits geometry that is not a residue: isosurfaces, site markers, marker-to-residue contacts, MI or coupling edges. If built-ins do the job, write no plugin.
- PyPI 1.6.5 and maraxiom's vendored copy have no `add_plugin`. Pin a git SHA from `sokrypton/py2Dmol` main at or after `19317016`: the first `add_plugin` commits lack later fixes to live payloads and framing, and the fork and a local `~/repos/py2Dmol` can lag upstream. Before designing, assert `py2Dmol.__file__` and `hasattr(py2Dmol.viewer.view, "add_plugin")` in the project's env.
- Plugin contract: `apiVersion` 1, register through `window.py2dmolPlugins` (never `py2dmolMolParts`), over-budget geometry throws, no legend or `remove_plugin` yet. Take the test pattern from `tests/plugin_browser.py`. The plugin lives in the project; `~/repos/py2Dmol` is read-only. Pin the structure, bundle and plugin sha256s so figures are reproducible.

Bio prose follows `prose-biosafe`.

## P4: push and PR

Push the branch and open one PR per workstream against `main`, naming the debts, epic, spec, `bth` run ids, and what CI does and does not verify. Never push to `main`, force-push, merge or close a PR.

## P5: review and remediate, until satisfied

Each round, dispatch fresh reviewers who did not write the code to review the full diff against the spec: correctness, the edge-case table, false greens, and whether tests can fail. Findings carry file:line, a failure scenario and a severity. Verify each before acting; dispatch a fixer to apply confirmed ones on the same branch. Then read CI's actual failures and logs, not just its status.

Satisfied means one full round with no confirmed BLOCKER or MAJOR, and CI green. The cap is 5 rounds; then stop and list what is open. After two failed attempts at the same check, stop that item, record what was tried, and move on.

## Boundaries

- `~/projects/xtrax`, other projects and `~/repos/*` are read-only. File any xtrax gap as a debt from an xtrax-cwd session or note it in the report.
- Put questions for the owner in `.praxia/loop/<task_id>/owner_questions.md` with a recommended default. Do not answer them yourself; keep working on whatever they do not block.

## Done

Report the workstream verdicts with evidence; epic id; spec and audit log paths; rounds to converge at each gate; PR URLs; `bth` run ids; the py2Dmol decision; debts to resolve or re-scope (#3257, #3142, #3143 and project debts); and open owner questions. A workstream is complete only with its verification record.
