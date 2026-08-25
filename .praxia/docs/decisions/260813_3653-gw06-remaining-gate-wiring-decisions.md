---
title: "[GW-06] Remaining human-approval gate wiring decisions (T2-29, T2-30, T2-31)"
task_id: 260813_epic2181-gw-sprint-compose
backlog_id: "3653"
status: "active"
created: 260813
---

## Overview

Backlog #3653 ([GW-06], epic #2181) addresses three independent sub-gates from the loop composition's human-approval tier: T2-29 (evaluator_change_gate), T2-30 (promotion_gate), and T2-31 (scope_expansion_gate). This document records the real, evidence-backed decision for each, including deferrals with documented follow-up items and concrete next actions.

## T2-29 (evaluator_change_gate) — Defer to backlog #4141

**Decision**: Do not wire T2-29 in this cycle. Defer with a documented depends_on edge to backlog #4141.

**Justification**: The recon discovered that T2-29's only production caller (`build_closure_manifest` in the loop controller) is an open design question explicitly deferred to backlog item #4141 (decision documented in `.praxia/docs/decisions/260714_2181-autoresearch-loop-constitution.md`, authored 2 days before this recon). Building any T2-29 call site into the loop controller now would guarantee downstream rework or an incorrectly-shaped seam. The gate module itself (`xtrax.loop.evaluator_change_gate`) is complete and unit-tested; the wiring seam and call-site design remain open.

**Next Action**: Filed as backlog row with `depends_on:[4141]`. Once #4141 settles the `build_closure_manifest` design, #4141's owner can file a new item for T2-29's actual wiring seam.

## T2-30 (promotion_gate) — CI-required-check, label-gated

**Decision**: Build a PR-merge-required CI workflow that reads `.praxia/loop_human_gates.toml` for fresh approval attestations, not an in-process controller wiring and not a human-run merge script.

**Justification**: The loop constitution's binding constraint is "No autonomous push or merge to main" — merge authority must never be granted to the loop itself, only to human review gates. A script that performs the merge would violate this constraint. A CI-required-check is structurally enforced by GitHub and never grants git-write authority to the loop; it only reports pass/fail. This design respects the constitution's intent.

**Precedent claim (AUDIT CORRECTION)**: The original plan claimed "reuses an established, already-well-tested composition pattern... confirmed present for ~15 other gates, e.g. audit_correctness_gate.py / audit_type_hardening_gate.py". This claim is **FALSE**. Verified against the actual repo:

- `.github/workflows/` contains only 4 files: `ci.yml`, `audit-judgment.yml` (cron-only), `docs.yml`, `publish.yml`
- Every `audit_*.py` script is chained into ONE shared `audit-deterministic` Justfile recipe
- That recipe is invoked by ONE shared `ci.yml` job
- There is ZERO existing precedent for a standalone per-gate workflow file or for label-gated conditional `pull_request` triggering

**Correction**: This plan proceeds with the label-gated standalone workflow file anyway (it is still the right design given the constitution's promotion policy), but treats it explicitly as **NOVEL CI plumbing with no in-repo precedent**, requiring correspondingly higher scrutiny than a routine port. Manual verification of both labeled and unlabeled test-PR behavior is mandatory before relying on this as a required check (see step 4.2/4.3 in the decomposition).

**Implementation summary**:
1. Create `scripts/audit_promotion_gate.py` (CLI script wrapping `xtrax.loop.promotion_gate.assert_promotion_approved`)
2. Create `tests/audit/test_promotion_gate_ci.py` (CLI test cases covering approved/unapproved/expired/malformed-TOML scenarios)
3. Add a Justfile recipe for local testing
4. Create `.github/workflows/promotion-gate.yml` (label-gated, no-op on ordinary PRs)
5. Document the manual approval-authoring workflow (below)

### Manual approval-authoring workflow (AUDIT-REQUIRED ADDITION)

**Problem**: The CI check reads `.praxia/loop_human_gates.toml` for a `[[gates]]` entry where `id="T2-30"` AND `event_ref == head_sha`. If that entry doesn't exist, the check fails. But how does a human add that entry before labeling a PR for promotion?

**Solution**: Before applying the `loop:promotion-candidate` label to a PR, Marielle (the human approval holder) must take one of these actions:

**Option A (Preferred): Amend the promotion PR itself**
1. Checkout the PR's branch locally
2. Fetch its HEAD SHA: `git rev-parse HEAD`
3. Edit `.praxia/loop_human_gates.toml` on the promotion PR branch, adding a new `[[gates]]` block:
   ```toml
   [[gates]]
   id = "T2-30"
   event_ref = "<HEAD_SHA>"
   attested_at = "<ISO8601 timestamp>"
   ttl_days = 30
   attested_by = "Marielle Russo"
   note = "Promotion approved after engineering review (promotion PR #<N>)"
   ```
4. Commit this change to the promotion PR branch
5. Push to GitHub
6. Label the PR with `loop:promotion-candidate` — the workflow will re-run automatically and pass

**Option B (If the PR is already merged and you're retroactively approving)**: 
1. Create a new commit on `main` amending `.praxia/loop_human_gates.toml` with the above block, using the already-merged commit's SHA
2. Push the amendment commit to `main`
3. (No label needed; the gate is now already approved for that SHA in perpetuity)

**Option C (Pre-approval before PR submission)**:
1. If you know a promotion is coming, pre-add its expected SHA to `.praxia/loop_human_gates.toml` on `main`
2. When the PR lands with that SHA, it will immediately pass the check

**Timing constraint**: The attestation's `attested_at` + `ttl_days` must extend beyond the time the PR needs to merge. Default `ttl_days=30` means approval is fresh for 30 days from attestation.

## T2-31 (scope_expansion_gate) — Defer entirely, explicit documented non-scope

**Decision**: Do not wire T2-31 in this cycle or any near-term cycle. Defer without a specific depends_on edge (document as open design question).

**Justification**: Recon found zero existing capability-expansion action anywhere in the codebase to attach a gate to. The only named future consumer for T2-31 is task T2-33 (Scope Expansion Orchestration), which is itself marked P4/Deferred in the sprint rubric. Wiring a gate call site now would require fabricated inputs, creating a false-positive gate that reports "OK to expand" when there is no actual expansion action to regulate.

**Next Action**: Filed as a backlog row with a clear "open design question / no existing action to gate" note. This item stays in the backlog as a future reference; when/if scope-expansion automation lands, revisit this gate's wiring seam.

---

## Branch-protection registration (manual, repo-admin only)

Once this PR lands with the workflow file and passes validation in this worktree, Marielle must perform these repo-admin actions on the main GitHub repo (not scoped to this plan, but documented for completeness):

1. **Create the `loop:promotion-candidate` label** (one-time, or verify it already exists):
   ```bash
   gh label create loop:promotion-candidate \
     --description "Mark a PR as a candidate for promotion via T2-30 CI gate" \
     --color "FFD700"
   ```

2. **Register `promotion-gate` as a required status check** on the `main` branch:
   ```bash
   gh api repos/marielle/xtrax/branches/main/protection/required_status_checks/contexts \
     -X POST -f "context=promotion-gate"
   ```
   Or via GitHub UI: Settings → Branches → Branch protection rules (main) → Add status check → search "promotion-gate" → Save

3. **Note on skipped jobs counting as PASSING**: GitHub's required-status-check logic treats a job that is skipped (via `if:` conditional) as PASSING for branch-protection purposes. The workflow's conditional `if: contains(github.event.pull_request.labels.*.name, 'loop:promotion-candidate')` will skip the job on ordinary PRs, reporting conclusion `skipped`. This is the mechanism that makes label-gating safe to register as a required check — only labeled PRs are required to pass; unlabeled PRs are unaffected.

**Out of scope**: Agent cannot execute GitHub API calls or branch-protection changes directly. This is a documented manual handoff to be completed before merging this track's PR.

---

## Summary table

| Gate | Decision | Next Action | Depends On |
|------|----------|-------------|------------|
| T2-29 (evaluator_change_gate) | Defer | File backlog row | #4141 |
| T2-30 (promotion_gate) | Ship this cycle (CI wiring only) | Merge this PR + manual approval-authoring per Option A/B/C above | (none) |
| T2-31 (scope_expansion_gate) | Defer entirely | File backlog row (no specific depends_on) | (open design) |
