#!/usr/bin/env python3
"""Smoke-test the #2181 P1 walking-skeleton + core P2 primitives, end-to-end.

Every T2-04 through T2-32 item built for epic #2181 is a standalone, composable gate or
pure-decision function -- each one explicitly defers actual sequencing to "a future loop
controller" in its own docstring, and no #2181 DAG item was ever scoped as "build that
controller." This script is a ONE-OFF smoke test, not that controller: it manually sequences the
already-built P1 walking-skeleton gates (AC-E2, AC-1..9, AC-13, AC-14) plus two core P2 primitives
(AC-10 ratchet, AC-12 diversity-quota) against a tiny synthetic evolve-target, to prove the pieces
genuinely compose end-to-end -- exactly the DAG's own stated P1 goal: "proves the immutable-
evaluator monopoly end-to-end on one function before any rigor machinery."

Explicitly NOT covered here (each already unit-tested independently, this proves composition on
the happy path only, not each gate's own failure modes): unlisted-read detection, closure-hash
drift, watchdog SIGKILL behavior, crash-recovery after an injected mid-ratchet crash. No bathos
calls of any kind (xtrax never imports bathos) -- the prereg sidecar is a locally-fabricated
stand-in for "already fetched from bathos." No GPU compute -- toy scalar arithmetic only.

Not a permanent xtrax.loop module, not registered as a CLI verb, not wired into CI.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from jax import ShapeDtypeStruct

from xtrax.inference.schema import extract_schema
from xtrax.loop.candidate_smoke import assert_candidate_smoke
from xtrax.loop.candidate_static import assert_candidate_static
from xtrax.loop.checkified_execution import assert_checkified_execution
from xtrax.loop.closure_lock import build_closure_manifest
from xtrax.loop.diversity_quota import assert_diversity_quota, is_structural_mutation
from xtrax.loop.external_stop_watchdog import WatchdogCriteria, start_watchdog
from xtrax.loop.info_barrier_lint import build_sanctioned_envelope
from xtrax.loop.metrics_provenance import evaluate_with_provenance
from xtrax.loop.multi_metric_ratchet import compute_ratchet_decision
from xtrax.loop.prereg_match import PreregSidecar, RunConfig, assert_prereg_match
from xtrax.loop.ratchet_crash_atomicity import advance_best_so_far, create_pending_commit
from xtrax.loop.schema_gate import assert_schema_gate, resolve_candidate_callable
from xtrax.loop.structure_tripwire import assert_structure_tripwire
from xtrax.stages.evaluate import EvaluatorAlreadySealedError, seal_evaluator

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("smoke_2181")

_TEST_INPUT = 2.0
_TARGET = 10.0

_BASELINE_SOURCE = "import jax.numpy as jnp\n\n\ndef f(x):\n    return jnp.asarray(x) * 4.0\n"
_IMPROVED_SOURCE = "import jax.numpy as jnp\n\n\ndef f(x):\n    return jnp.asarray(x) * 5.0\n"

_EVALUATOR_SOURCE = (
    "def evaluate(frozen_context, candidate_fn):\n"
    "    result = candidate_fn(frozen_context['test_input'])\n"
    "    error = float(result) - frozen_context['target']\n"
    "    return {'score': -abs(error), 'squared_error': -(error ** 2)}\n"
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        msg = f"git {' '.join(args)} failed: {result.stderr.strip()}"
        raise RuntimeError(msg)
    return result


def _step(description: str) -> None:
    log.info("PASS: %s", description)


def run_smoke_test(tmp_dir: Path, repo_root: Path) -> None:
    # --- Step 0 (T2-04, AC-E2): seal the evaluator seam; a second seal must raise. ---
    evaluator_path = tmp_dir / "evaluator.py"
    evaluator_path.write_text(_EVALUATOR_SOURCE, encoding="utf-8")
    evaluator_fn = resolve_candidate_callable(evaluator_path, "evaluate")
    seal_evaluator(evaluator_fn)
    try:
        seal_evaluator(evaluator_fn)
    except EvaluatorAlreadySealedError:
        pass
    else:
        msg = "second seal_evaluator() call did not raise -- registration lock is not real"
        raise AssertionError(msg)
    _step("AC-E2 no-stub sealed-seam lock (T2-04): registration-locked, re-seal raises")

    # --- Candidate fixtures on disk. ---
    baseline_path = tmp_dir / "baseline.py"
    baseline_path.write_text(_BASELINE_SOURCE, encoding="utf-8")
    improved_path = tmp_dir / "improved.py"
    improved_path.write_text(_IMPROVED_SOURCE, encoding="utf-8")

    # --- AC-1 (T2-11): static gate on both candidates. ---
    assert_candidate_static(baseline_path, root=repo_root)
    assert_candidate_static(improved_path, root=repo_root)
    _step("AC-1 candidate-static (T2-11): clean import + zero jaxlint errors, both candidates")

    # --- AC-2 (T2-12): schema gate -- derive the declared schema from the baseline once. ---
    abstract_inputs = [ShapeDtypeStruct(shape=(), dtype="float32")]
    baseline_fn = resolve_candidate_callable(baseline_path, "f")
    declared_schema = extract_schema(baseline_fn, abstract_inputs)
    assert_schema_gate(
        baseline_path, "f", abstract_inputs=abstract_inputs, declared_schema=declared_schema
    )
    assert_schema_gate(
        improved_path, "f", abstract_inputs=abstract_inputs, declared_schema=declared_schema
    )
    _step("AC-2 schema-gate (T2-12): both candidates match the declared output schema")

    # --- AC-3 (T2-13): structure tripwire -- one concrete execution per candidate. ---
    concrete_inputs: list[Any] = [_TEST_INPUT]
    assert_structure_tripwire(
        baseline_path, "f", abstract_inputs=abstract_inputs, concrete_inputs=concrete_inputs
    )
    assert_structure_tripwire(
        improved_path, "f", abstract_inputs=abstract_inputs, concrete_inputs=concrete_inputs
    )
    _step("AC-3 structure-tripwire (T2-13): concrete output matches the abstract trace, both")

    # --- AC-4 (T2-14): candidate-smoke -- real subprocess dry-run + CPU smoke. ---
    assert_candidate_smoke(
        baseline_path, "f", concrete_inputs=concrete_inputs, wall_clock_budget_seconds=30.0
    )
    assert_candidate_smoke(
        improved_path, "f", concrete_inputs=concrete_inputs, wall_clock_budget_seconds=30.0
    )
    _step("AC-4 candidate-smoke (T2-14): L1 dry-run + L2 CPU smoke, both candidates")

    # --- AC-5 (T2-15): checkified execution -- NaN/Inf guarded. ---
    assert_checkified_execution(baseline_path, "f", concrete_inputs=concrete_inputs)
    assert_checkified_execution(improved_path, "f", concrete_inputs=concrete_inputs)
    _step("AC-5 checkified-execution (T2-15): NaN/Inf-guarded execution clean, both candidates")

    # --- AC-6 (T2-16): prereg-match -- a locally-fabricated sidecar, simulating bathos. ---
    script_sha256 = hashlib.sha256(improved_path.read_bytes()).hexdigest()
    run_config = RunConfig(
        script_sha256=script_sha256,
        hypothesis="an improved candidate beats the baseline on `score`",
        metrics=("score",),
    )
    sidecar = PreregSidecar(
        script_sha256=script_sha256,
        hypothesis="an improved candidate beats the baseline on `score`",
        metrics=("score",),
    )
    assert_prereg_match(run_config, sidecar)
    _step("AC-6 prereg-match (T2-16): run config matches the pre-registered sidecar")

    # --- AC-7 + AC-8 (T2-05 + T2-06): guarded, provenance-wrapped evaluator calls. ---
    locked = build_closure_manifest(
        evaluator_paths=(evaluator_path,),
        split_paths=(),
        metric_def_paths=(),
        config={"smoke_test": True},
    )
    frozen_context = {"test_input": _TEST_INPUT, "target": _TARGET}
    improved_fn = resolve_candidate_callable(improved_path, "f")

    baseline_record = evaluate_with_provenance(
        locked,
        evaluator_fn,
        frozen_context,
        baseline_fn,
        current_config={"smoke_test": True},
        run_id="smoke-baseline",
    )
    improved_record = evaluate_with_provenance(
        locked,
        evaluator_fn,
        frozen_context,
        improved_fn,
        current_config={"smoke_test": True},
        run_id="smoke-improved",
    )
    _step("AC-7+AC-8 guarded-evaluate + metrics-provenance (T2-05/T2-06): both candidates scored")

    # --- AC-9 (T2-07): sanctioned envelope, agent-facing. ---
    envelope = build_sanctioned_envelope(
        improved_record,
        iteration=1,
        previous_score=baseline_record.fitness["score"],
    )
    _step(f"AC-9 info-barrier-lint (T2-07): sanctioned envelope self-lints clean ({envelope!r})")

    # --- AC-10 (T2-17): multi-metric ratchet. ---
    decision = compute_ratchet_decision(
        dict(improved_record.fitness),
        dict(baseline_record.fitness),
        higher_is_better={"score": True, "squared_error": True},
    )
    if not decision.improved:
        msg = f"expected improved=True, got {decision!r}"
        raise AssertionError(msg)
    _step(f"AC-10 multi-metric-ratchet (T2-17): improved candidate ratchets in ({decision!r})")

    # --- AC-12 (T2-18): diversity quota, semantic. ---
    structural = is_structural_mutation(_BASELINE_SOURCE, _IMPROVED_SOURCE)
    window = [False, False, False, False, structural]
    quota_decision = assert_diversity_quota(window, window_size=5)
    _step(
        f"AC-12 diversity-quota (T2-18): structural={structural}, "
        f"quota_met={quota_decision.quota_met}"
    )

    # --- AC-14 (T2-10): ratchet crash-atomicity, real git objects in a scratch repo. ---
    scratch_repo = tmp_dir / "scratch_repo"
    scratch_repo.mkdir()
    _git(scratch_repo, "init", "--quiet")
    _git(scratch_repo, "config", "user.email", "smoke@example.com")
    _git(scratch_repo, "config", "user.name", "Smoke Test")
    (scratch_repo / "f.py").write_text(_BASELINE_SOURCE, encoding="utf-8")
    _git(scratch_repo, "add", "f.py")
    _git(scratch_repo, "commit", "--quiet", "-m", "initial: baseline candidate")
    parent_sha = _git(scratch_repo, "rev-parse", "HEAD").stdout.strip()

    (scratch_repo / "f.py").write_text(_IMPROVED_SOURCE, encoding="utf-8")
    _git(scratch_repo, "add", "f.py")
    tree_sha = _git(scratch_repo, "write-tree").stdout.strip()

    pending_sha = create_pending_commit(
        scratch_repo, tree_sha, parent_sha, "ratchet: improved candidate"
    )
    advance_best_so_far(scratch_repo, "refs/xtrax/best-so-far", pending_sha, expected_old_sha=None)
    _step(f"AC-14 ratchet-crash-atomicity (T2-10): best-so-far advanced to {pending_sha[:12]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        # --- AC-13 (T2-09): external-stop watchdog supervises the whole run. ---
        handle = start_watchdog(os.getpid(), WatchdogCriteria(wall_clock_budget_seconds=120.0))
        try:
            run_smoke_test(tmp_dir, repo_root)
            _step("AC-13 external-stop watchdog (T2-09): supervised the run without interfering")
        finally:
            handle.stop()

    log.info("")
    log.info("ALL 13 WALKING-SKELETON CHECKPOINTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
