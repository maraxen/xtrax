"""Bathos-free unit tests for `scripts/smoke_lc12_real_e2e.py`'s backlog #4216 machinery (AC-1
through AC-6, AC-11, plus the wiring-spy class).

Follows the `tests/scripts/` convention already established by
`tests/scripts/test_smoke_outlines_constrained_decode.py`: `from scripts.smoke_lc12_real_e2e
import ...`, no conftest needed.

**CRITICAL SCOPE BOUNDARY (AC-10): every test in this file is real-bathos-free.** No test
invokes `--i-understand-this-creates-real-bathos-state`, calls `main(`, instantiates a real
`PraxiaMcpTransport`/`BathosMcpTransport`, or touches `~/.bth/catalog/bathos.db`. The two tests
that exercise `_run_pass_campaign`/`_run_fail_campaign` directly (`TestCampaignWiring`) always
monkeypatch `scripts.smoke_lc12_real_e2e.run_campaign_loop` to a local recording stub first --
the real `controller.loop_run.run_campaign_loop` (which would create real bathos campaigns) is
never called. Constructing a `PraxiaDispatchBackend`/`BathosCampaignAdapter` instance is itself
inert (config storage only, per direct reading of both classes' `__init__`) -- real I/O only
happens inside their `dispatch_candidate()`/MCP-call methods, which nothing in this file reaches.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import types
from pathlib import Path
from typing import Any

import jax
import pytest

from controller.dispatch import CandidateHandoff
from controller.main_loop import compute_candidate_tree_sha
from scripts.smoke_lc12_real_e2e import (
    _HIGHER_IS_BETTER,
    _RUN_SUFFIX,
    _TIMING_CONCRETE_INPUTS,
    _TIMING_PROBE_NAME,
    _build_smoke_closure_manifest,
    _candidate_source,
    _head_sha,
    _make_scratch_git_repo,
    _make_smoke_frozen_context,
    _make_smoke_score_fn,
    _run_fail_campaign,
    _run_git,
    _run_pass_campaign,
    _smoke_campaign_approval_fn,
)
from xtrax.devtools.freshness import Attestation
from xtrax.loop.compile_time_clock import TwoPhaseTiming, measure_two_phase_timing
from xtrax.loop.schema_gate import resolve_candidate_callable


class TestBuildSmokeClosureManifest:
    """AC-1: `_build_smoke_closure_manifest` produces a real `ClosureManifest` with empty
    path tuples, and its `closure_hash` genuinely varies with `config`."""

    def test_empty_path_tuples(self) -> None:
        manifest = _build_smoke_closure_manifest({"a": 1})
        assert manifest.evaluator_paths == ()
        assert manifest.split_paths == ()
        assert manifest.metric_def_paths == ()

    def test_same_config_same_closure_hash(self) -> None:
        config = {"lc12_smoke_campaign": "lc12-smoke-test-pass-abcd1234"}
        first = _build_smoke_closure_manifest(config)
        second = _build_smoke_closure_manifest(dict(config))
        assert first.closure_hash == second.closure_hash

    def test_different_config_different_closure_hash(self) -> None:
        first = _build_smoke_closure_manifest({"lc12_smoke_campaign": "lc12-smoke-test-pass-aaaa"})
        second = _build_smoke_closure_manifest({"lc12_smoke_campaign": "lc12-smoke-test-fail-bbbb"})
        assert first.closure_hash != second.closure_hash


class TestCandidateSourceExtended:
    """AC-2: `_candidate_source` compiles cleanly, defines a callable `smoke_timing_probe` when
    exec'd, and produces zero stdout output."""

    def test_compiles(self) -> None:
        source = _candidate_source("pass", 0)
        compile(source, "<candidate>", "exec")  # raises SyntaxError on failure

    def test_exec_defines_callable_smoke_timing_probe(self) -> None:
        source = _candidate_source("pass", 1)
        namespace: dict[str, Any] = {}
        exec(compile(source, "<candidate>", "exec"), namespace)  # noqa: S102
        assert "smoke_timing_probe" in namespace
        assert callable(namespace["smoke_timing_probe"])
        assert namespace["smoke_timing_probe"](1.0, 2.0) == 3.0

    def test_zero_stdout_on_exec(self) -> None:
        source = _candidate_source("fail", 0)
        namespace: dict[str, Any] = {}
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(compile(source, "<candidate>", "exec"), namespace)  # noqa: S102
        assert buffer.getvalue() == ""


class TestMeasureTwoPhaseTimingDependencyChain:
    """AC-3: the real `resolve_candidate_callable` -> `jax.jit` -> `measure_two_phase_timing`
    dependency chain works end-to-end against a real generated candidate file."""

    def test_resolve_and_jit_call(self, tmp_path: Path) -> None:
        candidate_path = tmp_path / "candidate.py"
        candidate_path.write_text(_candidate_source("pass", 0), encoding="utf-8")

        fn = resolve_candidate_callable(candidate_path, _TIMING_PROBE_NAME)
        result = jax.jit(fn)(*_TIMING_CONCRETE_INPUTS)
        assert float(result) == 3.0

    def test_measure_two_phase_timing_real_call(self, tmp_path: Path) -> None:
        candidate_path = tmp_path / "candidate.py"
        candidate_path.write_text(_candidate_source("fail", 0), encoding="utf-8")

        timing = measure_two_phase_timing(
            candidate_path, _TIMING_PROBE_NAME, concrete_inputs=_TIMING_CONCRETE_INPUTS
        )
        assert isinstance(timing, TwoPhaseTiming)
        assert timing.compile_time_seconds >= 0.0
        assert timing.runtime_seconds >= 0.0
        assert float(timing.result) == 3.0


class TestMakeScratchGitRepo:
    """AC-4: `_make_scratch_git_repo`/`_head_sha` produce a real repo + real HEAD sha, and the
    real (never-mocked) `compute_candidate_tree_sha` succeeds standalone against them."""

    def test_returns_real_repo_and_head_sha(self, tmp_path: Path) -> None:
        repo = _make_scratch_git_repo(tmp_path)
        assert repo.is_dir()
        assert (repo / "existing.txt").exists()

        sha = _head_sha(repo)
        assert len(sha) == 40
        assert all(c in "0123456789abcdef" for c in sha)

    def test_run_git_raises_on_failure(self, tmp_path: Path) -> None:
        not_a_repo = tmp_path / "not_a_repo"
        not_a_repo.mkdir()
        with pytest.raises(ValueError, match="git"):
            _run_git(not_a_repo, "rev-parse", "HEAD")

    def test_compute_candidate_tree_sha_standalone(self, tmp_path: Path) -> None:
        repo = _make_scratch_git_repo(tmp_path)
        base_sha = _head_sha(repo)

        candidate_file = tmp_path / "candidate_source.py"
        candidate_content = "_value = 1\n"
        candidate_file.write_text(candidate_content, encoding="utf-8")
        # CandidateHandoff.content_sha256 must be a full 64-char sha256 hex digest (its own
        # __post_init__ enforces this) -- the real sha256 of the file bytes, not a git blob sha.
        content_sha256 = hashlib.sha256(candidate_content.encode("utf-8")).hexdigest()
        handoff = CandidateHandoff(path=candidate_file, content_sha256=content_sha256)

        tree_sha = compute_candidate_tree_sha(handoff, repo, Path("candidate.py"), base_sha)
        assert isinstance(tree_sha, str)
        assert tree_sha != ""


class TestSmokeCampaignApprovalFn:
    """AC-11: `_smoke_campaign_approval_fn` unconditionally returns a valid `Attestation` and
    never reads `toml_path`."""

    def test_returns_valid_attestation(self) -> None:
        attestation = _smoke_campaign_approval_fn("some-campaign-id")
        assert isinstance(attestation, Attestation)
        assert attestation.attested_by
        assert attestation.ttl_days > 0.0
        assert attestation.attested_at

    def test_never_reads_toml_path(self) -> None:
        nonexistent = Path("/nonexistent/does/not/exist/loop_human_gates.toml")
        assert not nonexistent.exists()
        # No exception/read attempt for a path that structurally cannot be opened.
        attestation = _smoke_campaign_approval_fn("some-campaign-id", toml_path=nonexistent)
        assert isinstance(attestation, Attestation)


class TestMakeSmokeScoreFnFreshness:
    """AC-5: `_make_smoke_score_fn()` is a genuine per-call factory -- two independent calls
    produce independently-counting closures, not a shared/module-level singleton."""

    def test_independent_closures_do_not_share_counter_state(self) -> None:
        first = _make_smoke_score_fn()
        second = _make_smoke_score_fn()

        first(raw_artifact_paths=(), split_paths=())
        first(raw_artifact_paths=(), split_paths=())
        first(raw_artifact_paths=(), split_paths=())

        second_result = second(raw_artifact_paths=(), split_paths=())
        assert second_result == {"metric_a": 1.0, "metric_b": 1.0}


class TestAssertPassCampaignResultRatchetAssertion:
    """AC-6: `_assert_pass_campaign_result`'s per-iteration loop raises `AssertionError` unless
    every iteration's `ratchet_decision.improved` is `True`."""

    @staticmethod
    def _fake_iteration(*, improved: bool) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            run_result=types.SimpleNamespace(success=True),
            gate_outcome=types.SimpleNamespace(
                stats_battery=types.SimpleNamespace(honored=True),
                hard_blocked=False,
            ),
            accepted=True,
            ratchet_decision=types.SimpleNamespace(improved=improved),
        )

    def _fake_result(self, *, improved_flags: list[bool]) -> types.SimpleNamespace:
        iterations = [self._fake_iteration(improved=flag) for flag in improved_flags]
        return types.SimpleNamespace(
            loop_result=types.SimpleNamespace(iterations=iterations),
            campaign_id="fake-campaign-id",
        )

    def test_all_improved_true_passes_silently(self) -> None:
        from scripts.smoke_lc12_real_e2e import _assert_pass_campaign_result

        result = self._fake_result(improved_flags=[True, True])
        _assert_pass_campaign_result(result)  # must not raise

    def test_one_improved_false_raises_assertion_error(self) -> None:
        from scripts.smoke_lc12_real_e2e import _assert_pass_campaign_result

        result = self._fake_result(improved_flags=[True, False])
        with pytest.raises(AssertionError, match="ratchet_decision.improved"):
            _assert_pass_campaign_result(result)


class _RecordingRunCampaignLoop:
    """Records every kwarg passed to a `run_campaign_loop(...)` call, never touching real bathos.

    Positional args (`dispatch_backend`, `campaign_adapter`) are recorded too, for completeness,
    but every new #4216 kwarg this test cares about is keyword-only at the real call sites.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> types.SimpleNamespace:
        self.calls.append(kwargs)
        # Return value is never inspected by _run_pass_campaign/_run_fail_campaign themselves
        # (the campaign functions just return it) -- a bare sentinel is sufficient here since
        # this class's whole job is to intercept the call, not simulate a real result shape.
        return types.SimpleNamespace(campaign_id="stub-campaign-id")


class TestCampaignWiring:
    """The most important test in this file (Step 6 wiring). Monkeypatches
    `scripts.smoke_lc12_real_e2e.run_campaign_loop` to a recording stub -- NEVER the real
    function -- then calls `_run_pass_campaign`/`_run_fail_campaign` directly, and asserts every
    new #4216 kwarg is wired correctly.

    **The critical assertion (plan-audit-caught blocking gap from Track A): `output_paths` must
    be a non-empty list on both calls.** Verifying this directly is this class's whole point --
    Track A's own report claims the fix, this test proves it independently.
    """

    def test_run_pass_campaign_wiring(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _RecordingRunCampaignLoop()
        monkeypatch.setattr("scripts.smoke_lc12_real_e2e.run_campaign_loop", recorder)

        _run_pass_campaign(tmp_path, wall_clock_budget_seconds=180.0)

        assert len(recorder.calls) == 1
        kwargs = recorder.calls[0]

        # output_paths: THE critical, plan-audit-caught assertion.
        assert "output_paths" in kwargs
        assert isinstance(kwargs["output_paths"], list)
        assert len(kwargs["output_paths"]) > 0

        # frozen_context: real BathosFrozenContext with a real, fresh score_fn.
        frozen_context = kwargs["frozen_context"]
        assert frozen_context.score_fn is not None
        first = frozen_context.score_fn(raw_artifact_paths=(), split_paths=())
        assert first == {"metric_a": 1.0, "metric_b": 1.0}

        # current_config matches the pass campaign name.
        assert "lc12_smoke_campaign" in kwargs["current_config"]
        assert "pass" in kwargs["current_config"]["lc12_smoke_campaign"]

        # repo: a real git repo under tmp_path.
        repo = kwargs["repo"]
        assert isinstance(repo, Path)
        assert repo.is_relative_to(tmp_path)
        assert (repo / ".git").exists()

        # ratchet_ref_name: contains _RUN_SUFFIX and the correct "-pass" suffix.
        ratchet_ref_name = kwargs["ratchet_ref_name"]
        assert _RUN_SUFFIX in ratchet_ref_name
        assert ratchet_ref_name.endswith("-pass")

        # callable_name / concrete_inputs / higher_is_better.
        assert kwargs["callable_name"] == _TIMING_PROBE_NAME
        assert kwargs["concrete_inputs"] == _TIMING_CONCRETE_INPUTS
        assert kwargs["higher_is_better"] == _HIGHER_IS_BETTER

        # candidate_target_path.
        assert kwargs["candidate_target_path"] == Path("candidate.py")

        # bootstrap_commit_sha matches the real repo's HEAD sha.
        assert kwargs["bootstrap_commit_sha"] == _head_sha(repo)

        # campaign_approval_fn is the real override, by identity.
        assert kwargs["campaign_approval_fn"] is _smoke_campaign_approval_fn

        # commit_tree_sha / allow_fresh_start_despite_existing_lineage must be ABSENT.
        assert "commit_tree_sha" not in kwargs
        assert "allow_fresh_start_despite_existing_lineage" not in kwargs

    def test_run_fail_campaign_wiring(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _RecordingRunCampaignLoop()
        monkeypatch.setattr("scripts.smoke_lc12_real_e2e.run_campaign_loop", recorder)

        _run_fail_campaign(tmp_path, wall_clock_budget_seconds=180.0)

        assert len(recorder.calls) == 1
        kwargs = recorder.calls[0]

        assert "output_paths" in kwargs
        assert isinstance(kwargs["output_paths"], list)
        assert len(kwargs["output_paths"]) > 0

        frozen_context = kwargs["frozen_context"]
        assert frozen_context.score_fn is not None
        first = frozen_context.score_fn(raw_artifact_paths=(), split_paths=())
        assert first == {"metric_a": 1.0, "metric_b": 1.0}

        assert "lc12_smoke_campaign" in kwargs["current_config"]
        assert "fail" in kwargs["current_config"]["lc12_smoke_campaign"]

        repo = kwargs["repo"]
        assert isinstance(repo, Path)
        assert repo.is_relative_to(tmp_path)
        assert (repo / ".git").exists()

        ratchet_ref_name = kwargs["ratchet_ref_name"]
        assert _RUN_SUFFIX in ratchet_ref_name
        assert ratchet_ref_name.endswith("-fail")

        assert kwargs["callable_name"] == _TIMING_PROBE_NAME
        assert kwargs["concrete_inputs"] == _TIMING_CONCRETE_INPUTS
        assert kwargs["higher_is_better"] == _HIGHER_IS_BETTER

        assert kwargs["candidate_target_path"] == Path("candidate.py")

        assert kwargs["bootstrap_commit_sha"] == _head_sha(repo)

        assert kwargs["campaign_approval_fn"] is _smoke_campaign_approval_fn

        assert "commit_tree_sha" not in kwargs
        assert "allow_fresh_start_despite_existing_lineage" not in kwargs

    def test_frozen_context_factory_called_fresh_per_campaign(self) -> None:
        """AC-5, wiring-level: `_make_smoke_frozen_context` builds an independent score_fn
        closure each call -- two calls for two different campaign names never share counter
        state, so one campaign's fitness values can never depend on the other's invocation
        order/count."""
        first = _make_smoke_frozen_context("campaign-a", {"lc12_smoke_campaign": "campaign-a"})
        second = _make_smoke_frozen_context("campaign-b", {"lc12_smoke_campaign": "campaign-b"})

        first.score_fn(raw_artifact_paths=(), split_paths=())
        first.score_fn(raw_artifact_paths=(), split_paths=())

        second_first_call = second.score_fn(raw_artifact_paths=(), split_paths=())
        assert second_first_call == {"metric_a": 1.0, "metric_b": 1.0}
