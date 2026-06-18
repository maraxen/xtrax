CC5 routing matrix: see [`routing.toml`](routing.toml) for severity×track disposition rules.

## Judgment rubrics (N4.2)

Eight dimension rubrics live under [`rubrics/`](rubrics/) as `*.toml` anchor tables (scores 1–5) for judgment-track `RubricScorer` grounding. Loader: `xtrax.devtools.rubrics.load_rubric` / `load_all_rubrics`.

- `domain=dimension` — default matrix for audit dimension findings (`deterministic`/`judgment` × `info`/`minor`/`major`/`critical` → `block_ci`, `found_issues`, or `backlog_node`).
- `domain=port` — port-validation wave rows (signal/condition-qualified); interim defaults from epic #2180.

**Ownership:** `domain=port` rows are proposed interim defaults pending audit-fw maintainer review (**TD-2180-03**). `domain=dimension` rows are owned by the audit framework (#1579).

Resolve API: `xtrax.devtools.routing.resolve_destination`.
