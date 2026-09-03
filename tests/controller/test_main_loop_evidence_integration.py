"""Integration tests for run_one_candidate_pass's post-run integrity/provenance gates
([GW-01], backlog #3648) against a REAL bathos catalog.

Deliberately kept separate from test_main_loop.py: that file's own docstring states "no live
bathos/praxia infrastructure required for any test in this module" -- these tests genuinely
need the real `bathos` package (controller extra) and a real on-disk catalog (built via
bathos.catalog.init_catalog/write_run), so they must not be collected in tiers that only
install `--extra dev`. Mirrors test_bathos_library_wrappers_integration.py's separation
rationale exactly.

These tests omit `evidence_candidate_fn`/`sidecar_drift_signal_fn` entirely, so
run_one_candidate_pass's REAL defaults (`get_evidence_candidate_for_run` /
`get_sidecar_drift_signal`) run against a real catalog. That is the whole point: main covers
the wrapper functions in isolation and covers run_one_candidate_pass with stubs, but nothing
otherwise proves the two are wired to each other -- the failure mode where a mis-wired default
still passes every stubbed test.

Ported from branch `gw03-campaign-approval-gate` (backlog #4801), where it could not run: it
predates main's signature. Two adaptations were forced, both worth knowing about.

`catalog_dir` is gone from the signature, and it is not merely renamed -- run_one_candidate_pass
hardcodes `catalog_dir=""` at its own evidence call site, so the real defaults always resolve
`Path.home() / ".bth"`. There is therefore no argument that can point them at a temp catalog,
and the `_home_catalog` fixture redirects `Path.home()` instead. Pointing BTH_CATALOG_DIR at
tmp_path would NOT work here: the wrappers compute the home path themselves and never read that
variable, so the tests would silently read the developer's real catalog while looking isolated.

`sidecar_drift_agent_mode` is now spelled `agent_mode`.
"""

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import controller.main_loop as main_loop_module
from controller.bathos_campaign_adapter import BathosCampaignAdapter
from controller.dispatch import MockDispatchBackend
from controller.evaluate_adapter import BathosFrozenContext
from controller.main_loop import run_one_candidate_pass
from xtrax.loop.closure_lock import ClosureManifest
from xtrax.loop.compile_time_clock import TwoPhaseTiming
from xtrax.loop.seed_gate import SeedTrialCounts
from xtrax.loop.sidecar_drift_gate import SidecarHashMismatchError
from xtrax.loop.stats_battery_gate import BathosStatsBatteryVerdict

bathos_catalog = pytest.importorskip("bathos.catalog")
bathos_schema = pytest.importorskip("bathos.schema")
pytest.importorskip("bathos.prereg")
pytest.importorskip("bathos.query")

_CANDIDATE_CONTENT = "candidate-source"
_RUN_ID = "run-evidence-integration"

_FROZEN_CONTEXT = BathosFrozenContext(
    locked=ClosureManifest(
        evaluator_paths=(),
        split_paths=(),
        metric_def_paths=(),
        pinned_deps_source=Path("uv.lock"),
        config={},
        closure_hash="unused-in-tests-guarded_evaluate_fn-is-stubbed",
    ),
    campaign_adapter=None,  # type: ignore[arg-type]  # unused: guarded_evaluate_fn is stubbed
    campaign_id="camp-1",
    score_fn=lambda *_args: {},
)


def _passing_guarded_evaluate_fn(*_args: Any, **_kwargs: Any) -> dict[str, float]:
    return {"accuracy": 0.9, "loss": 0.1}


def _passing_timing_fn(*_args: Any, **_kwargs: Any) -> TwoPhaseTiming:
    return TwoPhaseTiming(compile_time_seconds=0.0, runtime_seconds=0.0, result=None)


def _passing_stats_verdict(**_kwargs: Any) -> BathosStatsBatteryVerdict:
    return BathosStatsBatteryVerdict(
        verdict="pass",
        scipy_available=True,
        reasons=(),
        cohens_d=0.5,
        win_rate=0.8,
        breakdown_point=0.3,
        p_superiority=0.8,
        wilcoxon_p_value=0.01,
        icc=0.995,
        baseline_budget_equivalent=True,
    )


def _passing_seed_counts(
    _db: Any, script_sha256: str, hypothesis_clause_id: str = ""
) -> SeedTrialCounts:
    return SeedTrialCounts(script_sha256=script_sha256, distinct_seed_count=5, trial_count=40)


def _ok_run_envelope(**extra: Any) -> dict[str, Any]:
    return {
        "ok": True,
        "error_code": None,
        "error": None,
        "resolution_hint": None,
        "script_path": "candidate.py",
        "run_id": _RUN_ID,
        "exit_code": 0,
        "success": True,
        **extra,
    }


class _RecordingTransport:
    def __init__(self, envelope: dict[str, Any]) -> None:
        self.envelope = envelope
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((tool_name, dict(arguments)))
        return self.envelope


@pytest.fixture(autouse=True)
def _stub_crash_atomicity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub T2-10 crash-atomicity git calls -- this module exercises the evidence gates."""
    monkeypatch.setattr(main_loop_module, "read_best_so_far", lambda repo, ref_name: None)
    monkeypatch.setattr(
        main_loop_module,
        "create_pending_commit",
        lambda repo, tree_sha, parent_sha, message: "pending-sha",
    )
    monkeypatch.setattr(
        main_loop_module,
        "advance_best_so_far",
        lambda repo, ref_name, new_sha, expected_old_sha: None,
    )
    monkeypatch.setattr(
        main_loop_module, "reset_worktree_to_best_so_far", lambda repo, ref_name: "best-sha"
    )


@pytest.fixture
def home_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An initialised bathos catalog that the production defaults will actually find.

    run_one_candidate_pass passes `catalog_dir=""` to its evidence wrapper, so the wrapper
    resolves `Path.home() / ".bth"`. Redirecting home is therefore the only way to aim the
    REAL default at a temp catalog without reintroducing an injection seam -- and using the
    seam is exactly what this module exists not to do.
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    catalog_dir = tmp_path / ".bth"
    bathos_catalog.init_catalog(catalog_dir)
    return catalog_dir


@pytest.fixture(autouse=True)
def _stub_metrics_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate metrics-provenance writes and supply a non-empty run_id (#3075 HALTs on empty)."""
    monkeypatch.setattr(
        "controller.bathos_campaign_adapter.BathosCampaignAdapter._query_run_id_by_script_sha256",
        lambda self, script_path, catalog_dir: _RUN_ID,
    )


def _write_run(
    catalog_dir: Path,
    *,
    manifest_sha256: str = "",
    manifest_path: str = "",
    sidecar_sha256: str = "",
) -> None:
    run = bathos_schema.Run(
        project_slug="test-project",
        command="candidate.py",
        argv=["python", "candidate.py"],
        git_hash="deadbeef",
        git_branch="main",
        git_dirty=False,
        id=_RUN_ID,
        timestamp=datetime.now(UTC),
        manifest_sha256=manifest_sha256,
        manifest_path=manifest_path,
        sidecar_sha256=sidecar_sha256,
    )
    bathos_catalog.write_run(run, catalog_dir)


def _run_pass(
    tmp_path: Path,
    *,
    campaign_mode: str = "confirmation",
    agent_mode: str = "collaborative",
    stdout_verified: bool | None = None,
) -> Any:
    """Call the real run_one_candidate_pass, injecting nothing on the evidence path.

    Everything stubbed here is upstream or downstream of the gates under test (dispatch,
    scoring, timing, git). `evidence_candidate_fn` and `sidecar_drift_signal_fn` are
    conspicuously absent -- their production defaults are the subject.
    """
    dispatch_backend = MockDispatchBackend(
        candidate_path=tmp_path / "candidate.py", candidate_content=_CANDIDATE_CONTENT
    )
    transport = _RecordingTransport(_ok_run_envelope())
    adapter = BathosCampaignAdapter(transport=transport, token="test-token")

    provenance_dir = tmp_path / "metrics_provenance"
    provenance_dir.mkdir(exist_ok=True)

    return run_one_candidate_pass(
        dispatch_backend,
        adapter,
        campaign_id="camp-1",
        campaign_mode=campaign_mode,
        candidate_static_fn=lambda path, root=None: None,
        agent_mode=agent_mode,
        # Non-empty by requirement, not decoration: main gained a pre-scoring gate that
        # rejects an empty output_paths outright, which the gw03-era version of this file
        # predates. guarded_evaluate_fn is stubbed, so the path is never read.
        output_paths=["artifact.json"],
        stats_battery_kwargs={},
        stats_battery_fn=_passing_stats_verdict,
        seed_trial_counts_fn=_passing_seed_counts,
        # Admission needs manifest_verified AND stdout_verified; the default None is itself
        # an exclusion reason ("stdout hash never recorded"), so a test that wants an
        # admitted run has to supply this even with a perfectly good manifest.
        stdout_verified=stdout_verified,
        frozen_context=_FROZEN_CONTEXT,
        current_config={},
        guarded_evaluate_fn=_passing_guarded_evaluate_fn,
        repo=tmp_path / "unused-repo",
        ratchet_ref_name="refs/xtrax/best-so-far",
        commit_tree_sha="unused-tree-sha",
        commit_parent_sha="unused-parent-sha",
        callable_name="unused_callable",
        concrete_inputs=[],
        abstract_inputs=[],
        measure_two_phase_timing_fn=_passing_timing_fn,
        structure_tripwire_fn=lambda *a, **k: None,
        candidate_smoke_fn=lambda *a, **k: None,
        checkified_execution_fn=lambda *a, **k: None,
        metrics_provenance_dir=provenance_dir,
        verify_metrics_provenance_fn=lambda *a, **k: None,
        iteration=1,
    )


def _drifted_run_pair(tmp_path: Path, catalog_dir: Path, manifest_file: Path) -> None:
    """Write a baseline run and a later run whose sidecar hash differs from it."""
    manifest_sha256 = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
    script_path = tmp_path / "candidate.py"
    common = {
        "project_slug": "test-project",
        "command": str(script_path),
        "argv": ["python", str(script_path)],
        "git_hash": "deadbeef",
        "git_branch": "main",
        "git_dirty": False,
    }
    baseline_run = bathos_schema.Run(
        **common,
        id="run-baseline",
        timestamp=datetime.now(UTC) - timedelta(hours=1),
        sidecar_sha256="baseline-hash",
    )
    current_run = bathos_schema.Run(
        **common,
        id=_RUN_ID,
        timestamp=datetime.now(UTC),
        manifest_sha256=manifest_sha256,
        manifest_path=str(manifest_file),
        sidecar_sha256="drifted-hash",
    )
    bathos_catalog.write_run(baseline_run, catalog_dir)
    bathos_catalog.write_run(current_run, catalog_dir)


def _write_manifest(tmp_path: Path) -> Path:
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text('{"real": "manifest"}')
    return manifest_file


class TestRealEvidenceGateEndToEnd:
    """The real get_evidence_candidate_for_run default, against a real catalog.

    These assertions target main's evidence model, which differs from the one the gw03
    version of this file was written against. There, evidence integrity was a hard gate
    with a confirmation/exploration split that could force `accepted=False`. In main,
    `admit_evidence` returns an admitted/excluded partition recorded on the gate outcome
    (main_loop.py:1078) and consulted by nothing -- the call site's own comment says
    "advisory only, not hard-blocking", and there is no campaign-mode branch on it.

    So the hard-block assertions are gone: asserting them would have meant asserting
    behaviour that does not exist. What survives is the claim #4801 actually wanted --
    that the production default is wired to the gate and partitions a real catalog's runs
    correctly, which no stubbed test can show.
    """

    def test_valid_manifest_admits_the_run_as_evidence(
        self, tmp_path: Path, home_catalog: Path
    ) -> None:
        manifest_file = _write_manifest(tmp_path)
        manifest_sha256 = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
        _write_run(home_catalog, manifest_sha256=manifest_sha256, manifest_path=str(manifest_file))

        result = _run_pass(tmp_path, campaign_mode="confirmation", stdout_verified=True)

        admission = result.gate_outcome.evidence_admission
        assert admission is not None, "the real default did not run: nothing populated admission"
        assert [c.run_id for c in admission.admitted] == [_RUN_ID]
        assert admission.excluded == ()
        assert result.accepted is True

    def test_unrecorded_stdout_hash_excludes_an_otherwise_valid_run(
        self, tmp_path: Path, home_catalog: Path
    ) -> None:
        """A good manifest is not sufficient on its own -- admission needs both checks.

        This is the older-run case ExcludedRun's docstring calls out: a run predating
        stdout-hash capture is excluded even though nothing about it is wrong.
        """
        manifest_file = _write_manifest(tmp_path)
        manifest_sha256 = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
        _write_run(home_catalog, manifest_sha256=manifest_sha256, manifest_path=str(manifest_file))

        result = _run_pass(tmp_path, campaign_mode="confirmation", stdout_verified=None)

        admission = result.gate_outcome.evidence_admission
        assert admission is not None
        assert admission.admitted == ()
        reasons = admission.excluded[0].reasons
        assert any("stdout" in reason for reason in reasons), reasons

    def test_tampered_manifest_excludes_the_run_but_does_not_block_acceptance(
        self, tmp_path: Path, home_catalog: Path
    ) -> None:
        """A manifest hash that no longer matches its file excludes the run, loudly.

        `accepted is True` alongside it is not an oversight in the test -- it is main's
        documented advisory-only posture, asserted here so that a future change making
        evidence hard-blocking has to come past this test rather than slip by it.
        """
        manifest_file = _write_manifest(tmp_path)
        stale_sha256 = hashlib.sha256(b"different content").hexdigest()
        _write_run(home_catalog, manifest_sha256=stale_sha256, manifest_path=str(manifest_file))

        result = _run_pass(tmp_path, campaign_mode="confirmation", stdout_verified=True)

        admission = result.gate_outcome.evidence_admission
        assert admission is not None
        assert admission.admitted == ()
        assert [e.candidate.run_id for e in admission.excluded] == [_RUN_ID]
        # stdout_verified=True above so the manifest is the ONLY fault; a bare default would
        # exclude for two reasons at once and prove nothing about manifest verification.
        reasons = admission.excluded[0].reasons
        assert any("manifest" in reason for reason in reasons), reasons
        assert result.accepted is True

    def test_campaign_mode_does_not_change_evidence_admission(
        self, tmp_path: Path, home_catalog: Path
    ) -> None:
        """Exploration and confirmation partition evidence identically in main.

        The gw03 version asserted the opposite -- advisory under exploration, hard-blocking
        under confirmation. Pinning the actual behaviour keeps the difference visible instead
        of leaving it to be rediscovered from a deleted test.
        """
        manifest_file = _write_manifest(tmp_path)
        stale_sha256 = hashlib.sha256(b"different content").hexdigest()
        _write_run(home_catalog, manifest_sha256=stale_sha256, manifest_path=str(manifest_file))

        result = _run_pass(tmp_path, campaign_mode="exploration", stdout_verified=True)

        admission = result.gate_outcome.evidence_admission
        assert admission is not None
        assert admission.admitted == ()
        assert [e.candidate.run_id for e in admission.excluded] == [_RUN_ID]
        assert result.accepted is True

    def test_run_with_no_manifest_at_all_is_excluded_as_unverifiable(
        self, tmp_path: Path, home_catalog: Path
    ) -> None:
        """AC-19's own text is "unverifiable run excluded from evidence".

        This is the shape of a `no_sidecar=True` exploratory run later reused as evidence.
        """
        _write_run(home_catalog)

        result = _run_pass(tmp_path, campaign_mode="confirmation")

        admission = result.gate_outcome.evidence_admission
        assert admission is not None
        assert admission.admitted == ()
        assert [e.candidate.run_id for e in admission.excluded] == [_RUN_ID]

    def test_run_missing_from_the_catalog_leaves_evidence_unrecorded(
        self, tmp_path: Path, home_catalog: Path
    ) -> None:
        """Nothing written to the catalog: the wrapper raises, main_loop swallows it, no result.

        This doubles as the negative control for every other test in this class. They assert
        `admission is not None`, which is only meaningful if some reachable state makes it
        None -- otherwise a wrapper that silently never ran would satisfy them all. The
        `except (ValueError, ImportError): pass` around the evidence call is that state, and
        an empty catalog is what reaches it.
        """
        result = _run_pass(tmp_path, campaign_mode="confirmation", stdout_verified=True)

        assert result.gate_outcome.evidence_admission is None

    def test_sidecar_drift_under_autonomous_mode_raises_and_skips_gate_checks(
        self, tmp_path: Path, home_catalog: Path
    ) -> None:
        _drifted_run_pair(tmp_path, home_catalog, _write_manifest(tmp_path))

        with pytest.raises(SidecarHashMismatchError):
            _run_pass(tmp_path, campaign_mode="confirmation", agent_mode="autonomous")

    def test_sidecar_drift_under_collaborative_mode_only_warns(
        self, tmp_path: Path, home_catalog: Path
    ) -> None:
        _drifted_run_pair(tmp_path, home_catalog, _write_manifest(tmp_path))

        result = _run_pass(tmp_path, campaign_mode="confirmation", agent_mode="collaborative")

        drift = result.gate_outcome.sidecar_drift
        assert drift is not None, "the real get_sidecar_drift_signal default did not run"
        assert drift.drifted is True
        assert drift.should_warn is True
        assert drift.reason, "a drifted decision must name the script and both hashes"
        # Warn, not block: the autonomous-mode counterpart above raises instead.
        assert result.accepted is True


__all__ = ["TestRealEvidenceGateEndToEnd"]
