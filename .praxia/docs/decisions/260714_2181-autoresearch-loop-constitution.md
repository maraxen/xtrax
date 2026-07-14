# Constitution: #2181 agentic-autoresearch loop

**Status:** SIGNED (approved as-drafted, 2026-07-14) — satisfies AC-21, T2-28, #3046
**Governs:** epic #2181 ("agentic algorithm evolution & autoresearch," xtrax + bathos)
**Authority:** Marielle Russo (sole approving authority for all gates below, until amended)

## 1. Purpose

This document is the human-authored governance record required by AC-21 (T2-28, gate a) before
the #2181 evolve-loop may start or continue running. It exists to make explicit, in one place,
who has approval authority over the loop's five irreversible or safety-relevant decision points,
and what each approval actually commits to. It is not a design spec (see
`.praxia/docs/specs/260702_design-the-2181-agentic-algorithm-evolut.md` for that) — it is the
record of *who gets to say yes*, not *what the loop does*.

## 2. Non-negotiable invariants (already locked by the design spec, restated here for one-place reference)

These are not re-decided by this document — they are binding constraints on the loop regardless
of any gate approval below:

- **No autonomous push or merge to `main`.** Every code change the loop produces stays in a
  sandbox lineage until gate (c) explicitly promotes it.
- **The evaluator holds a sealed monopoly on fitness scalars** (AC3, T1-07's
  `SealedEvaluatorRegistry`). The loop never scores its own candidates by any other path.
- **An external, out-of-process watchdog holds an always-available, loop-unrevokable kill switch**
  (AC-13, T2-09) — the loop can never disable its own kill switch, and a wedged/hung candidate
  cannot wedge the watchdog.
- **Fitness scalars must be 100% traceable to the immutable evaluator's own output** (AC-8) —
  self-reported numbers from inside the loop are never trusted.
- **Agent-facing outputs are schema-validated, field-whitelisted JSON with no raw-log read path**
  (AC-9, the information barrier).

## 3. The five gates

Each gate is a human-approval checkpoint. Approval is recorded as a machine-checkable
`xtrax.devtools.freshness.Attestation` (T3-05/AC-X6) — a hermetic, offline, TTL-bounded claim
that expires and must be renewed, not a one-time checkbox that stays green forever.

### (a) Constitution authorship (AC-21) — this document

Any future amendment to this document (adding, removing, or materially changing a gate's policy
below) requires the same sign-off process as the original: a new dated revision, explicit
approval, and a fresh attestation. The loop cannot start or continue if this document's own
attestation has expired.

### (b) Evaluator change (AC-22) — standing runtime gate

**Policy:** any change to evaluator code, test splits, or metric definitions requires Marielle's
explicit sign-off before the changed evaluator is trusted, *and* forces a closure-hash re-lock
(T2-05) of the new evaluator's complete closure (code + splits + metric defs + pinned deps +
config). The agent never approves its own judge, under any circumstance. This gate fires on
every evaluator-change *event*, not once — there is no standing blanket approval.

### (c) Promotion to `main` (AC-23)

**Policy:** evolved code proposed for promotion out of the sandbox lineage into `xtrax` `main`
requires Marielle's explicit engineering review and approval per promotion. Refused-by-default:
absent explicit approval, code stays in the sandbox lineage indefinitely. This is a per-promotion
gate, not a standing grant — passing review once does not pre-approve a later promotion.

### (d) Scope / allowlist expansion (AC-24)

**Policy:** any expansion of the loop's network access, tool allowlist, or sandbox capabilities
(including adding new evolve-block surface or new effectful tools) requires Marielle's explicit
approval before the expansion takes effect. Denied-by-default absent that approval.

### (e) Kill-switch / campaign approval (AC-25)

**Policy:** every campaign start requires Marielle's explicit approval before the campaign may
run. The external watchdog's kill authority (see §2) is always available regardless of campaign
approval status, and cannot be revoked by the loop under any circumstance, including mid-campaign.

## 4. Attestation record

| Field | Value |
|---|---|
| `attested_at` | 2026-07-14T00:00:00Z |
| `ttl_days` | 365 |
| `attested_by` | Marielle Russo |
| `note` | Initial constitution sign-off; approves the policy in §2-3 as the governing record for AC-21 (T2-28). Renew before TTL expiry via a dated revision to this document + a fresh attestation. |

## 5. Explicit scope note

This document authorizes the *policy* the five gates enforce. It does not itself build the
runtime mechanisms that enforce them (the standing-gate hooks for (b)/(d)/(e), the promotion
review workflow for (c), or the machine-checkable probe/attestation wiring binding this
document's own freshness to T3-05). Those remain real implementation work, tracked under their
own backlog items (T2-04 depends on this document's attestation existing; T2-29/31/32 are
themselves the standing-gate-mechanism backlog items).
