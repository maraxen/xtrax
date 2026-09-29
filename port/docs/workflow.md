# port_validation: shape and the xtrax in-place variant

## Shape: 7 executable steps, 6 gates

`agent_assets/workflows/port_validation.yaml` (and its hand-maintained dispatch form,
`.claude/workflows/port-validation.js`) runs seven steps in order. Six are gates; one
is an artifact.

| # | step | kind | produces / checks |
|---|------|------|-------------------|
| 1 | P0-ORACLE | gate | sealed `port/reference/<subtree>/` + `oracle_id` (`just audit-port-oracle-seal`) |
| 2 | P1-SPEC | gate | jaxtyping contracts for the target symbol |
| 3 | P1.5-TOPO | **artifact, not a gate** | `port/manifests/<wave_id>.toml`: topo-sorted qualnames + `manifest_hash` |
| 4 | P2-STATIC | gate | jaxlint + trace count (`just audit-port-static`) |
| 5 | P3-PARITY | gate | graded tiers in `port/tests/` (`just audit-port-parity`) |
| 6 | P4-EMIT | gate | `domain=port` tier verdicts in `.praxia/audits.jsonl` (`port/emit/port_emit.py`) |
| 7 | P5-ROUTE | gate | `audit/routing.toml` `domain=port` rows applied to the emitted verdicts |

P1.5 cannot fail a wave: it orders the work that P2–P5 then gate. Older design notes
call this a "6-phase" pipeline, counting only the gates.

P3's tiers run in blocking order — T1 dtype/shape, T2 float64, T3 float32, T5 JIT
invariance — and a failure skips every later tier ("blocked after ... failure",
`port/tests/conftest.py`). T4 (gradient parity) runs only when `port_target.toml`
sets `[parity] ad_critical = true`.

**What blocks CI.** `routing.toml` routes a deterministic tier FAIL to `block_ci`; the
enforcement is the `audit-port` CI job (`just audit-port`, triggered by changes under
`src/xtrax/**` or `port/**`), where a failing tier fails pytest.
`tests/audit/test_port_block_ci_wiring.py` pins every link of that chain and proves
it with a planted wrong `safe_map` that must fail the harness.

## The xtrax in-place variant of the jax-port workflow

The generic `jax-port` skill stages a port beside its reference and deletes the
reference once parity is green. xtrax deviates on purpose (design
`.praxia/docs/designs/260618_2180_port_validation_design.md` §"In-place translation",
spec AC-2):

| jax-port skill default | xtrax |
|---|---|
| translate into a `jax_port/` staging subtree | translate **in place** into `src/xtrax/<module>/` from the first commit |
| delete `reference/` after parity passes | `port/reference/` is **permanently sealed** (`# REFERENCE: DO NOT MODIFY`); it is the oracle for every future re-run |
| parity tests import the staged port | `port/tests/test_parity_*.py` import the **production** symbol from `src/xtrax/` |
| promotion = moving code out of staging | promotion = the tier gates going green; directory location never changes |
| phases 0–7 as a full lifecycle | the workflow maps them onto P0–P5 and scopes v0.1 to deterministic, static-shape kernels. `port_target.toml` `[capabilities]` declares `stochastic = false`, `dynamic_shape = false`, but nothing reads those flags yet (TD-2180-04) -- a stochastic or ragged kernel is out of scope by convention, not by a gate |

Checklist for a new port wave:

1. Vendor the reference under `port/reference/<subtree>/`, seal it, record `oracle_id`
   in `port/port_target.toml` (P0). Fixers are read-only on that subtree
   (`[access] fixer_read_only_on_reference = true`).
2. Write the jaxtyping contract for the target symbol (P1).
3. Generate `port/manifests/<wave_id>.toml` and point `port_target.toml` `wave_id` at it (P1.5).
4. Translate directly into `src/xtrax/`; no staging directory.
5. Add `port/tests/test_parity_<kernel>.py` importing the `src/xtrax/` symbol, with
   tier markers `tier_1`…`tier_5`; set `ad_critical` if gradients matter.
6. `just audit-port` locally; CI re-runs it on the PR.
7. Keep `port/reference/` forever. When parity fails later, the first-divergence
   trace (`agent_assets/skills/xtrax-activation-parity/`) tells you where.
