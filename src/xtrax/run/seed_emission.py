"""Run-level seed emission for the xtrax<->bathos bridge (T2-23, #2181, AC-16, emission half).

AC-16's claim: "loop emits `Run.seed` + campaign mode; conclude enforces >=3 seeds per
`script_sha256` AND N >= 29 trials per `script_sha256` per hypothesis clause...". This module is
the EMISSION half only -- the loop must surface the seed it used for a run so bathos can persist
it on `Run.seed` (B2-02, merged) and later count it
(`bathos.campaigns.count_seeds_for_script`, also B2-02). The conclude-time counting/hard-block
half lives in `xtrax.loop.seed_gate` (see that module for why this is split in two).

The "+ campaign mode" half of AC-16's emission clause is deliberately NOT this module's concern:
campaign mode is already fully wired at campaign creation, not per-run -- `bth campaign create
--mode <mode>` (bathos CLI, confirmed on origin/main) already carries it, and every prior
T2-1x/T2-2x gate in this epic (`capability_probe_gate.CampaignMode`, `stats_battery_gate.
CampaignMode`) already threads that same mode through as a plain caller-supplied value. Re-emitting
it again per-run here would duplicate an already-closed concern, not fill a genuine gap.

Grounding correction (verified before writing any code, not guessed): unlike B2-08's
`--component-id` / `--component-sidecar-sha256`, which B2-08 wired all the way through `bth run`'s
own CLI (`src/bathos/cli.py`'s `run` command, confirmed on bathos origin/main), B2-02 (PR #15,
"Run.seed field (+ baseline_hpo_trials/compute_budget)", commit 99af1c7) added ONLY the schema
columns (`seed`, `baseline_hpo_trials`, `baseline_hpo_compute_budget`, all optional, defaulting to
`None` rather than 0 so "no seed recorded" stays distinguishable from "seed 0 was used") plus two
conclude-time counting helpers in `bathos/campaigns.py` (`count_seeds_for_script`,
`count_runs_for_script`) -- it did NOT add a `--seed` flag to `bth run`'s CLI (grepped
`src/bathos/cli.py` on origin/main: the `run` command has no `seed` option, unlike its already-
wired `component_id`/`component_sidecar_sha256` pair). This module computes the flag xtrax would
pass once that CLI flag exists (or once a loop controller calls bathos's Python `run_script`
directly with a `seed=` kwarg) -- by the same naming convention
`ComponentSidecarBinding.as_bth_run_flags()` established (`--<field-name> <value>`), matching the
schema field name `seed` verbatim. Wiring the actual bathos-side CLI flag is out of scope here and
is not this module's job to add (xtrax never edits bathos).

Cross-repo integration seam (matches `xtrax.run.component_binding` (B2-08) and
`xtrax.run.repro_floor` exactly -- both already-merged #2181 items crossing into bathos): xtrax
has no dependency on bathos, by design. This module only computes the flag/value pair a caller
appends to its own `bth run` argv (or passes to bathos's Python API); it does not invoke `bth`,
does not import bathos, and does not decide what a future loop controller does with the emitted
value beyond returning it.
"""

from dataclasses import dataclass


class SeedEmissionInputError(Exception):
    """`seed` is not a valid non-negative integer for a bathos `Run.seed` value."""


@dataclass(frozen=True, slots=True)
class SeedAssignment:
    """The seed a loop iteration used for one run, to surface for bathos's `Run.seed` column
    (B2-02, merged) so a later conclude-time count
    (`bathos.campaigns.count_seeds_for_script`) can see it.
    """

    seed: int

    def as_bth_run_flags(self) -> list[str]:
        """The flag/value pair a caller appends to its own `bth run` argv to record this seed
        (see module docstring: bathos's CLI does not wire `--seed` yet as of this writing; this
        is the flag xtrax would pass once it does, or the kwarg name to pass to bathos's Python
        `run_script` directly). Pure string formatting -- this method does not invoke `bth` or
        any subprocess itself.
        """
        return ["--seed", str(self.seed)]


def seed_assignment(seed: int) -> SeedAssignment:
    """Validate and wrap a run's seed for emission.

    Args:
        seed: the integer seed this loop iteration used for its run.

    Raises:
        SeedEmissionInputError: `seed` is negative. Bathos's `Run.seed` column is `int | None`
            with no documented lower bound, and `xtrax.run.spec.RunSpec.seed` is likewise a bare
            `int` with no bound of its own -- but a negative seed being persisted as a real
            `Run.seed` value that a later conclude-time distinct-seed count treats as legitimate
            (e.g. an uninitialized sentinel of -1 leaking through) is a plausible caller bug, not
            a meaningful PRNG seed in any of xtrax's own seeding paths
            (`xtrax.training.state` feeds `seed` straight into `jax.random.PRNGKey`, which itself
            expects a non-negative value) -- fail loud here rather than silently emitting and
            later counting a bogus seed.
    """
    if seed < 0:
        msg = f"seed must be a non-negative integer, got {seed}"
        raise SeedEmissionInputError(msg)
    return SeedAssignment(seed=seed)


__all__ = [
    "SeedAssignment",
    "SeedEmissionInputError",
    "seed_assignment",
]
