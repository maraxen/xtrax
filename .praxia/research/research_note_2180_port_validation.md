---
research_note_id: 260618_research_note_2180
epic_backlog_id: 2180
task_id: 260617_xtrax-composition-mission
sources:
  - .praxia/research/synthesis_2180_port_validation.md (librarian v2, NLM DR-1..6)
  - recon: 260618_recon_2180_port_validation
  - contemplex session: 073d0395 (phases 1-3)
---

# Research Note: #2180 Implementation Validation Pipeline

## Executive summary

Extend xtrax audit-fw (#1573) with a **port_validation** vertical slice: vendored oracle, graded parity (T1–T3+T5 MVP), two-track agentic/mechanical CI, `domain=port` emit records. **Winner architecture (brainstorm 073d0395):** `port/` dev-only extra owns reference + parity tests + emit stub; code translates **in-place to `src/xtrax/`** (no `port/jax_port/` staging); optional `port/bridge/composition_map.toml` deferred to Phase 2 (#2174).

## Evidence stack

| Layer | Artifact |
|-------|----------|
| NLM | Notebook `2e509f42`, DR-1..6, librarian Q1–Q9 |
| Codebase | N0 gates land (`tests/audit/`, `audit_jaxlint_json.py`); no `port/` yet |
| Brainstorm | Session `073d0395` — Faction C hybrid wins over port-centric A |

## Locked decisions

- Two-track: agents write pytest/scripts; CI runs scripts only
- New PCW template `port_validation` (6 phases incl. P1.5-TOPO)
- Emit stub-first `port/emit/port_emit.py` parallel to N1.1 #1577
- T4 deferred unless `port_target.toml` sets `ad_critical: true`
- Hook-extended `subagent-stop` with `tier_verdict` payload
- `port/` excluded from wheel (dev extra)
- #2181 blocked until #2180 MVP

## First implementation wave (recon)

1. Scaffold `port/reference/`, `port/tests/`, `port/emit/`
2. Shared emit serializer + contract test
3. T1–T3+T5 parity harness for first topo-sorted kernel
4. `just audit-port` + CI wiring
5. Registry semver 0.2.0 (+ reference-vendor, specification-specialist, test-designer)
