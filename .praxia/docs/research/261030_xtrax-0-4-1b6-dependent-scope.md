---
title: xtrax 0.4.1b6 dependent scope
description: Verified list of xtrax consumers across ~/projects, with pin ranges, removed-API hits and vendored copies, for the dependent-update debt items 3142 and 3143.
status: draft
task_id: 261030_xtrax-dependent-scope
date: '261030'
confidence: medium
sources: 'scan of ~/projects/*/pyproject.toml and uv.lock, nested affigit/packages/* and mpnn_ext/external/*; CHANGELOG.md Removed sections for 0.4.0a12 and 0.4.1b5'
---
# xtrax 0.4.1b6 dependent scope

Scope of the dependent-update work for xtrax 0.4.1b6, filed under debt 3142 (campaign execution) and 3143 (EDA visualization).

## Method and limits

- Scanned `~/projects/*/pyproject.toml` and `uv.lock`, plus nested `affigit/packages/*` and `mpnn_ext/external/*`.
- Only `~/projects` was scanned. Other workspaces are not covered.
- Not checked: CHANGELOG `Changed` sections against each pin range, and dependents' test suites.

## Direct consumers

| Project | Declared pin | Locked | Notes |
|---|---|---|---|
| alynxr | `xtrax[io]>=0.4.1b5` | 0.4.1b5 | |
| aminx | `xtrax[io,export]>=0.4.0a11,<0.5` | 0.4.1b3 | Upper bound `<0.5` admits 0.4.1b6 |
| aminx-esm | `>=0.3.1` | 0.3.1 | |
| asr | `>=0.4.0a5` | 0.4.0a12 | Vendors aminx (see below) |
| demistify | `>=0.4.1b3` | 0.4.1b5 | |
| denxity | `>=0.4.0a12` | 0.4.0a12 | |
| plegadx | `xtrax[cli]` | path-editable | |
| prolix | `>=0.4.1b2,<0.5` | 0.4.1b5 | |
| tev_design | `>=0.3.0` | wheel override 0.4.0a9 | Override is load-bearing; see memory `project_xtrax-consumer-pin-topology` |

## Path-editable consumers

blindsight, colliculix, lathex, protamer, affigit-wire.

## Transitive consumers

biosensors, denxity-experimental, hautespout, mpnn_ext, naurmalade.

## Removed-API hits

Source: CHANGELOG `Removed` for 0.4.0a12 and 0.4.1b5 (`SafeMap`). No `xtrax.sparse` imports found in dependents.

- aminx `scripts/ebm/benchmarks/langevin_benchmark.py:148`: unguarded.
- plegadx `tests/probe_safemap_export.py:16`, `scripts/run_intellifold_e2e_030_ensemble.py:240-241`.
- prolix `src/prolix/tiling/axis_dispatch.py`, `src/prolix/api/ensemble_dispatch.py`, `src/prolix/tiling/xtrax_adapter.py`: try/except fallbacks, so guarded.

## Vendored copies

These do not move when a consumer bumps its pin:

- `asr/aminx`
- `mpnn_ext/external/aminx`
- `hautespout/vendor/aminx`
- `biosensors/vendor/demistify`
- `ksweep-bench/denxity`, `denxity/denxity` (nested)

## Open verification

1. Read the CHANGELOG `Changed` sections for each pin range.
2. Run dependents' test suites against 0.4.1b6, one project at a time, on the remote for heavy suites.
3. Check aminx-esm and the path-editable projects in depth.
4. Decide whether to scan other workspaces.
