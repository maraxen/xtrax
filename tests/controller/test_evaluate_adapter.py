"""Tests for controller.evaluate_adapter (SPLIT_COMPUTE evaluator, #4133, AC1-6).

AC1-6 coverage:
1. **AC1** (documentation-level): docstring + BathosCandidate.output_paths docs state that
   output_paths must contain only raw artifacts, never a pre-computed summary.
2. **AC2**: real temp files, real file reads in score_fn, real fitness dict computation.
3. **AC3/AC4**: injected transport records exactly one `run` call with correct parameters.
4. **AC5**: end-to-end integration with guarded_evaluate and closure lock verification.
5. **AC6**: evaluate_with_provenance integrates correctly and sets evaluator_closure_hash.
6. **Failure path**: dispatch failure raises RawArtifactsUnavailableError, score_fn never called.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from controller.bathos_campaign_adapter import BathosCampaignAdapter
from controller.evaluate_adapter import (
    BathosCandidate,
    BathosFrozenContext,
    BathosSplitComputeEvaluator,
    RawArtifactsUnavailableError,
    score_raw_artifacts,
)
from xtrax.loop.closure_lock import ClosureManifest, build_closure_manifest, guarded_evaluate
from xtrax.loop.metrics_provenance import evaluate_with_provenance


class _RecordingTransport:
    """Injection seam for BathosCampaignAdapter -- records every call and returns a
    pre-programmed envelope."""

    def __init__(self, envelope: dict[str, Any]) -> None:
        self.envelope = envelope
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((tool_name, dict(arguments)))
        return self.envelope


def _ok_envelope(**extra: Any) -> dict[str, Any]:
    """A bathos `traced_tool`-shaped success envelope."""
    return {"ok": True, "error_code": None, "error": None, "resolution_hint": None, **extra}


# ---------------------------------------------------------------------------
# AC1: documentation-level contract that output_paths contains only raw artifacts
# ---------------------------------------------------------------------------


class TestAC1Documentation:
    def test_ac1_module_docstring_documents_split_compute_invariant(self) -> None:
        """AC1 (weak/trivial test -- this AC is a design contract xtrax cannot mechanically
        verify about a candidate script's internal behavior). The module docstring documents
        that the bathos-dispatched subprocess must produce only RAW artifacts, never a
        pre-computed summary fitness value (SPLIT_COMPUTE's core invariant)."""
        import controller.evaluate_adapter

        docstring = controller.evaluate_adapter.__doc__
        assert docstring is not None
        assert "raw artifacts" in docstring.lower()
        assert "SPLIT_COMPUTE" in docstring
        assert "fitness" in docstring.lower(), (
            "docstring must document what raw artifacts are scored against"
        )

    def test_ac1_bathoscandidate_docstring_documents_output_paths_constraint(self) -> None:
        """AC1: BathosCandidate.output_paths docstring must state that output_paths must
        contain only raw artifacts, never a pre-computed summary."""
        docstring = (
            BathosCandidate.__dataclass_fields__["output_paths"].metadata.get("__doc__")
            or BathosCandidate.__doc__
            or ""
        )
        assert "raw artifacts" in docstring.lower() or "output_paths" in docstring.lower(), (
            "BathosCandidate docstring must document output_paths constraint "
            "(checked via class __doc__)"
        )
        # Stronger: verify the class docstring mentions this constraint explicitly
        assert BathosCandidate.__doc__ is not None
        assert "output_paths" in BathosCandidate.__doc__
        assert "raw artifacts" in BathosCandidate.__doc__.lower()


# ---------------------------------------------------------------------------
# #4139: leftover overclaims in evaluate_adapter module + class docstrings
# ---------------------------------------------------------------------------


def _evaluate_adapter_module_and_class_docs() -> tuple[str, str]:
    """Return (module __doc__, BathosSplitComputeEvaluator __doc__), never None."""
    import controller.evaluate_adapter

    module_doc = controller.evaluate_adapter.__doc__
    class_doc = BathosSplitComputeEvaluator.__doc__
    assert module_doc is not None, "controller.evaluate_adapter must have a module docstring"
    assert class_doc is not None, "BathosSplitComputeEvaluator must have a class docstring"
    return module_doc, class_doc


class TestDocstring4139:
    """#4139: module + BathosSplitComputeEvaluator class docstrings must drop leftover
    overclaims and name the production scoring path (score_raw_artifacts via guarded_evaluate).
    """

    def test_module_docstring_omits_stats_battery_kwargs(self) -> None:
        """Circularity framed as this module's fitness dict feeding stats_battery_kwargs is
        false: that mapping stays caller-supplied. Phrase must be absent from the module doc."""
        module_doc, _ = _evaluate_adapter_module_and_class_docs()
        assert "stats_battery_kwargs" not in module_doc, (
            "module docstring must not claim this module's fitness dict is what "
            "stats_battery_kwargs needs (caller-supplied; not produced here)"
        )

    def test_module_docstring_omits_wired_in_a_follow_on_item(self) -> None:
        """score_raw_artifacts is already wired via guarded_evaluate in
        run_one_candidate_pass; the module docstring must not claim a follow-on item."""
        module_doc, _ = _evaluate_adapter_module_and_class_docs()
        assert "wired in a follow-on item" not in module_doc, (
            "module docstring must not say score_raw_artifacts is "
            "'wired in a follow-on item' — it is already wired"
        )

    def test_class_docstring_omits_sole_call_site(self) -> None:
        """Production fitness is score_raw_artifacts via guarded_evaluate. Calling
        BathosSplitComputeEvaluator.__call__ from the pass would double-dispatch (#4137)."""
        _, class_doc = _evaluate_adapter_module_and_class_docs()
        assert "sole call site" not in class_doc.lower(), (
            "BathosSplitComputeEvaluator class docstring must not claim it is the "
            "'sole call site' through which a candidate's fitness is obtained"
        )

    def test_module_or_class_docstring_names_score_raw_artifacts(self) -> None:
        module_doc, class_doc = _evaluate_adapter_module_and_class_docs()
        combined = f"{module_doc}\n{class_doc}"
        assert "score_raw_artifacts" in combined, (
            "module and/or BathosSplitComputeEvaluator docstring must name "
            "score_raw_artifacts as the production scoring path"
        )

    def test_module_or_class_docstring_names_guarded_evaluate(self) -> None:
        module_doc, class_doc = _evaluate_adapter_module_and_class_docs()
        combined = f"{module_doc}\n{class_doc}"
        assert "guarded_evaluate" in combined, (
            "module and/or BathosSplitComputeEvaluator docstring must name "
            "guarded_evaluate (production path is guarded_evaluate(..., "
            "score_raw_artifacts, ...))"
        )

    def test_module_or_class_docstring_names_4137_or_double_dispatch(self) -> None:
        module_doc, class_doc = _evaluate_adapter_module_and_class_docs()
        combined = f"{module_doc}\n{class_doc}"
        has_ticket = "#4137" in combined
        has_double_dispatch = (
            "double-dispatch" in combined.lower() or "double dispatch" in combined.lower()
        )
        assert has_ticket or has_double_dispatch, (
            "module and/or BathosSplitComputeEvaluator docstring must mention "
            "#4137 or double-dispatch (calling __call__ from the pass would "
            "double-dispatch)"
        )

    def test_module_docstring_keeps_ac1_grep_strings(self) -> None:
        """GREEN must not drop TestAC1Documentation's module-docstring greps."""
        module_doc, _ = _evaluate_adapter_module_and_class_docs()
        assert "raw artifacts" in module_doc.lower()
        assert "SPLIT_COMPUTE" in module_doc
        assert "fitness" in module_doc.lower(), (
            "docstring must document what raw artifacts are scored against"
        )


# ---------------------------------------------------------------------------
# AC2: real temp files, real file reads, real fitness computation
# ---------------------------------------------------------------------------


class TestAC2RealFileReadsAndComputation:
    def test_ac2_score_fn_reads_real_files_and_computes_fitness(self, tmp_path: Path) -> None:
        """AC2: score_fn reads REAL files and computes a fitness dict genuinely based on their
        contents. Proves scoring executes in-process against real data, not a stub returning
        a canned dict."""
        # Write fake raw artifact and fake split/ground-truth to disk
        raw_artifact = tmp_path / "artifact.txt"
        raw_artifact.write_text("hello world", encoding="utf-8")

        ground_truth = tmp_path / "ground_truth.txt"
        ground_truth.write_text("expected output", encoding="utf-8")

        # Injectable score_fn that reads REAL files via its OWN parameters (not closed-over
        # test variables) -- proves BathosSplitComputeEvaluator threads candidate.output_paths
        # and frozen_context.locked.split_paths through correctly, not just that some file
        # gets read from somewhere.
        def score_fn(
            raw_artifact_paths: tuple[Path, ...], split_paths: tuple[Path, ...]
        ) -> dict[str, float]:
            """Reads actual files via its parameters and returns a fitness dict computed from
            their contents."""
            artifact_text = raw_artifact_paths[0].read_text(encoding="utf-8")
            ground_text = split_paths[0].read_text(encoding="utf-8")
            # Genuine computation: combined length of both files
            combined_length = len(artifact_text) + len(ground_text)
            return {"combined_length": float(combined_length), "match_score": 1.0}

        transport = _RecordingTransport(_ok_envelope(script_path="c.py", exit_code=0, success=True))
        adapter = BathosCampaignAdapter(transport=transport, token="test-token")

        locked = ClosureManifest(
            evaluator_paths=(),
            split_paths=(ground_truth,),
            metric_def_paths=(),
            pinned_deps_source=Path("uv.lock"),
            config={},
            closure_hash="fake-hash",
        )

        frozen_context = BathosFrozenContext(
            locked=locked,
            campaign_adapter=adapter,
            campaign_id="camp-1",
            score_fn=score_fn,
        )

        candidate = BathosCandidate(
            script_path="c.py",
            output_paths=(str(raw_artifact),),
        )

        evaluator = BathosSplitComputeEvaluator()
        result = evaluator(frozen_context, candidate)

        # Assert the result reflects genuine computation over real file contents
        # combined_length = len("hello world") + len("expected output") = 11 + 15 = 26
        assert result == {"combined_length": 26.0, "match_score": 1.0}


# ---------------------------------------------------------------------------
# score_raw_artifacts: standalone scoring function extracted from __call__'s scoring half
# (#3649, GW-02 Track A)
# ---------------------------------------------------------------------------


class TestScoreRawArtifacts:
    def test_score_raw_artifacts_matches_call_fitness_dict_shape_and_values(
        self, tmp_path: Path
    ) -> None:
        """score_raw_artifacts(frozen_context, output_paths), called directly, returns the
        SAME fitness dict shape/values that BathosSplitComputeEvaluator.__call__ previously
        produced inline for identical inputs -- proves the extraction is behavior-preserving."""
        raw_artifact = tmp_path / "artifact.txt"
        raw_artifact.write_text("hello world", encoding="utf-8")

        ground_truth = tmp_path / "ground_truth.txt"
        ground_truth.write_text("expected output", encoding="utf-8")

        def score_fn(
            raw_artifact_paths: tuple[Path, ...], split_paths: tuple[Path, ...]
        ) -> dict[str, float]:
            artifact_text = raw_artifact_paths[0].read_text(encoding="utf-8")
            ground_text = split_paths[0].read_text(encoding="utf-8")
            combined_length = len(artifact_text) + len(ground_text)
            return {"combined_length": float(combined_length), "match_score": 1.0}

        locked = ClosureManifest(
            evaluator_paths=(),
            split_paths=(ground_truth,),
            metric_def_paths=(),
            pinned_deps_source=Path("uv.lock"),
            config={},
            closure_hash="fake-hash",
        )

        transport = _RecordingTransport(_ok_envelope(script_path="c.py", exit_code=0, success=True))
        adapter = BathosCampaignAdapter(transport=transport, token="test-token")

        frozen_context = BathosFrozenContext(
            locked=locked,
            campaign_adapter=adapter,
            campaign_id="camp-1",
            score_fn=score_fn,
        )

        output_paths = (str(raw_artifact),)

        # Calling score_raw_artifacts directly, with no dispatch involved.
        direct_result = score_raw_artifacts(frozen_context, output_paths)

        # combined_length = len("hello world") + len("expected output") = 11 + 15 = 26
        assert direct_result == {"combined_length": 26.0, "match_score": 1.0}

        # Same inputs threaded through __call__ (which now delegates to score_raw_artifacts
        # after its own dispatch) must produce a byte-identical fitness dict.
        candidate = BathosCandidate(script_path="c.py", output_paths=output_paths)
        evaluator = BathosSplitComputeEvaluator()
        call_result = evaluator(frozen_context, candidate)

        assert call_result == direct_result

    def test_score_raw_artifacts_raises_for_empty_output_paths(self) -> None:
        """score_raw_artifacts raises RawArtifactsUnavailableError when given no output_paths,
        without ever invoking score_fn."""
        score_fn_calls: list[bool] = []

        def score_fn(
            _raw_artifact_paths: tuple[Path, ...], _split_paths: tuple[Path, ...]
        ) -> dict[str, float]:
            score_fn_calls.append(True)
            return {"score": 1.0}

        locked = ClosureManifest(
            evaluator_paths=(),
            split_paths=(),
            metric_def_paths=(),
            pinned_deps_source=Path("uv.lock"),
            config={},
            closure_hash="fake-hash",
        )

        transport = _RecordingTransport(_ok_envelope(script_path="c.py", exit_code=0, success=True))
        adapter = BathosCampaignAdapter(transport=transport, token="test-token")

        frozen_context = BathosFrozenContext(
            locked=locked,
            campaign_adapter=adapter,
            campaign_id="camp-1",
            score_fn=score_fn,
        )

        with pytest.raises(RawArtifactsUnavailableError):
            score_raw_artifacts(frozen_context, ())

        assert score_fn_calls == []


# ---------------------------------------------------------------------------
# AC3/AC4: transport records exactly one `run` call with correct parameters
# ---------------------------------------------------------------------------


class TestAC3AC4TransportRecording:
    def test_ac3_ac4_transport_records_run_call_with_script_path_and_campaign_id(self) -> None:
        """AC3/AC4: transport's `.calls` records exactly ONE `run` tool call after invoking
        BathosSplitComputeEvaluator, with script_path and campaign_id matching the input."""
        transport = _RecordingTransport(_ok_envelope(script_path="c.py", exit_code=0, success=True))
        adapter = BathosCampaignAdapter(transport=transport, token="test-token")

        def dummy_score_fn(
            _raw_artifact_paths: tuple[Path, ...], _split_paths: tuple[Path, ...]
        ) -> dict[str, float]:
            return {"score": 1.0}

        locked = ClosureManifest(
            evaluator_paths=(),
            split_paths=(),
            metric_def_paths=(),
            pinned_deps_source=Path("uv.lock"),
            config={},
            closure_hash="fake-hash",
        )

        frozen_context = BathosFrozenContext(
            locked=locked,
            campaign_adapter=adapter,
            campaign_id="camp-123",
            score_fn=dummy_score_fn,
        )

        candidate = BathosCandidate(
            script_path="script.py",
            output_paths=("output.txt",),
        )

        evaluator = BathosSplitComputeEvaluator()
        evaluator(frozen_context, candidate)

        # Assert exactly one call was made
        assert len(transport.calls) == 1
        tool_name, arguments = transport.calls[0]

        # Assert it's a `run` call
        assert tool_name == "run"

        # Assert parameters match
        assert arguments["script_path"] == "script.py"
        assert arguments["campaign_id"] == "camp-123"

    def test_ac3_ac4_transport_records_all_forwarded_parameters(self) -> None:
        """AC3/AC4: transport records all parameters forwarded from BathosCandidate to
        campaign_adapter.run."""
        transport = _RecordingTransport(_ok_envelope(script_path="c.py", exit_code=0, success=True))
        adapter = BathosCampaignAdapter(transport=transport, token="test-token")

        def dummy_score_fn(
            _raw_artifact_paths: tuple[Path, ...], _split_paths: tuple[Path, ...]
        ) -> dict[str, float]:
            return {"score": 1.0}

        locked = ClosureManifest(
            evaluator_paths=(),
            split_paths=(),
            metric_def_paths=(),
            pinned_deps_source=Path("uv.lock"),
            config={},
            closure_hash="fake-hash",
        )

        frozen_context = BathosFrozenContext(
            locked=locked,
            campaign_adapter=adapter,
            campaign_id="camp-1",
            score_fn=dummy_score_fn,
        )

        candidate = BathosCandidate(
            script_path="script.py",
            output_paths=("output1.txt", "output2.txt"),
            derived_from="parent-run-id",
            run_args=("--flag1", "value1"),
            tags=("tag1", "tag2"),
            agent_mode="test",
            no_sidecar=True,
        )

        evaluator = BathosSplitComputeEvaluator()
        evaluator(frozen_context, candidate)

        assert len(transport.calls) == 1
        _, arguments = transport.calls[0]

        assert arguments["script_path"] == "script.py"
        assert arguments["campaign_id"] == "camp-1"
        assert arguments["derived_from"] == "parent-run-id"
        assert arguments["args"] == ["--flag1", "value1"]
        assert arguments["output_paths"] == ["output1.txt", "output2.txt"]
        assert arguments["tags"] == ["tag1", "tag2"]
        assert arguments["agent_mode"] == "test"
        assert arguments["no_sidecar"] is True


# ---------------------------------------------------------------------------
# AC5: end-to-end integration with guarded_evaluate and closure lock
# ---------------------------------------------------------------------------


class TestAC5EndToEndWithGuardedEvaluate:
    def test_ac5_guarded_evaluate_succeeds_with_declared_metric_def_path(
        self, tmp_path: Path
    ) -> None:
        """AC5: call guarded_evaluate with a real ClosureManifest built via
        build_closure_manifest, where the raw-artifact path is declared in metric_def_paths.
        Assert this succeeds with no UnlistedReadError."""
        # Real files on disk
        evaluator_file = tmp_path / "evaluator.py"
        evaluator_file.write_text("# evaluator", encoding="utf-8")

        raw_artifact = tmp_path / "artifact.txt"
        raw_artifact.write_text("artifact content", encoding="utf-8")

        split_file = tmp_path / "split.txt"
        split_file.write_text("split content", encoding="utf-8")

        config = {"test": "value"}

        # Build a real locked closure manifest
        locked = build_closure_manifest(
            evaluator_paths=(evaluator_file,),
            split_paths=(split_file,),
            metric_def_paths=(raw_artifact,),  # Declare the artifact path
            config=config,
            pinned_deps_source=Path("uv.lock"),
        )

        transport = _RecordingTransport(_ok_envelope(script_path="c.py", exit_code=0, success=True))
        adapter = BathosCampaignAdapter(transport=transport, token="test-token")

        def score_fn(
            raw_artifact_paths: tuple[Path, ...], split_paths: tuple[Path, ...]
        ) -> dict[str, float]:
            # Read via the parameters the module actually passed -- proves
            # candidate.output_paths and locked.split_paths are threaded through correctly.
            artifact_text = raw_artifact_paths[0].read_text(encoding="utf-8")
            _ = split_paths[0].read_text(encoding="utf-8")  # also declared; read is legal
            return {"score": float(len(artifact_text))}

        frozen_context = BathosFrozenContext(
            locked=locked,
            campaign_adapter=adapter,
            campaign_id="camp-1",
            score_fn=score_fn,
        )

        candidate = BathosCandidate(
            script_path="script.py",
            output_paths=(str(raw_artifact),),
        )

        evaluator = BathosSplitComputeEvaluator()

        # This should succeed -- raw_artifact was declared in metric_def_paths
        result = guarded_evaluate(
            locked,
            evaluator,
            frozen_context,
            candidate,
            current_config=config,
        )

        assert result == {"score": 16.0}

    def test_ac5_guarded_evaluate_raises_unlisted_read_error_for_undeclared_path(
        self, tmp_path: Path
    ) -> None:
        """AC5: call guarded_evaluate where the raw-artifact path is NOT declared in the
        locked closure. Assert guarded_evaluate raises UnlistedReadError -- proving
        track_reads genuinely observes the module's file access."""
        evaluator_file = tmp_path / "evaluator.py"
        evaluator_file.write_text("# evaluator", encoding="utf-8")

        raw_artifact = tmp_path / "artifact.txt"
        raw_artifact.write_text("artifact content", encoding="utf-8")

        split_file = tmp_path / "split.txt"
        split_file.write_text("split content", encoding="utf-8")

        undeclared_path = tmp_path / "undeclared.txt"
        undeclared_path.write_text("undeclared content", encoding="utf-8")

        config = {"test": "value"}

        # Build locked closure WITHOUT declaring the undeclared_path
        locked = build_closure_manifest(
            evaluator_paths=(evaluator_file,),
            split_paths=(split_file,),
            metric_def_paths=(raw_artifact,),  # DO NOT declare undeclared_path
            config=config,
            pinned_deps_source=Path("uv.lock"),
        )

        transport = _RecordingTransport(_ok_envelope(script_path="c.py", exit_code=0, success=True))
        adapter = BathosCampaignAdapter(transport=transport, token="test-token")

        def score_fn(
            raw_artifact_paths: tuple[Path, ...], split_paths: tuple[Path, ...]
        ) -> dict[str, float]:
            # Reads its declared paths first (legal, via its own parameters) -- proving the
            # gate isn't merely tripping on the module's own official reads -- then ALSO reads
            # an out-of-band undeclared path. Even a score_fn whose "official" wiring is
            # correct still gets caught if it reaches outside the locked closure anywhere.
            _ = raw_artifact_paths[0].read_text(encoding="utf-8")
            _ = split_paths[0].read_text(encoding="utf-8")
            undeclared_text = undeclared_path.read_text(encoding="utf-8")
            return {"score": float(len(undeclared_text))}

        frozen_context = BathosFrozenContext(
            locked=locked,
            campaign_adapter=adapter,
            campaign_id="camp-1",
            score_fn=score_fn,
        )

        candidate = BathosCandidate(
            script_path="script.py",
            output_paths=(str(raw_artifact),),
        )

        evaluator = BathosSplitComputeEvaluator()

        # This should raise UnlistedReadError
        from xtrax.loop.closure_lock import UnlistedReadError

        with pytest.raises(UnlistedReadError, match="evaluator read undeclared path"):
            guarded_evaluate(
                locked,
                evaluator,
                frozen_context,
                candidate,
                current_config=config,
            )


# ---------------------------------------------------------------------------
# AC6: evaluate_with_provenance integrates and sets evaluator_closure_hash
# ---------------------------------------------------------------------------


class TestAC6ProvenanceIntegration:
    def test_ac6_evaluate_with_provenance_sets_closure_hash(self, tmp_path: Path) -> None:
        """AC6: call evaluate_with_provenance with BathosSplitComputeEvaluator and assert
        the returned MetricsProvenanceRecord.evaluator_closure_hash == locked.closure_hash."""
        evaluator_file = tmp_path / "evaluator.py"
        evaluator_file.write_text("# evaluator", encoding="utf-8")

        raw_artifact = tmp_path / "artifact.txt"
        raw_artifact.write_text("artifact content", encoding="utf-8")

        split_file = tmp_path / "split.txt"
        split_file.write_text("split content", encoding="utf-8")

        config = {"test": "value"}

        locked = build_closure_manifest(
            evaluator_paths=(evaluator_file,),
            split_paths=(split_file,),
            metric_def_paths=(raw_artifact,),
            config=config,
            pinned_deps_source=Path("uv.lock"),
        )

        transport = _RecordingTransport(_ok_envelope(script_path="c.py", exit_code=0, success=True))
        adapter = BathosCampaignAdapter(transport=transport, token="test-token")

        def score_fn(
            _raw_artifact_paths: tuple[Path, ...], _split_paths: tuple[Path, ...]
        ) -> dict[str, float]:
            return {"score": 42.0}

        frozen_context = BathosFrozenContext(
            locked=locked,
            campaign_adapter=adapter,
            campaign_id="camp-1",
            score_fn=score_fn,
        )

        candidate = BathosCandidate(
            script_path="script.py",
            output_paths=(str(raw_artifact),),
        )

        evaluator = BathosSplitComputeEvaluator()

        record = evaluate_with_provenance(
            locked,
            evaluator,
            frozen_context,
            candidate,
            current_config=config,
            run_id="test-run-123",
        )

        assert record.evaluator_closure_hash == locked.closure_hash
        assert record.run_id == "test-run-123"
        assert record.fitness == {"score": 42.0}


# ---------------------------------------------------------------------------
# Failure path: dispatch failure raises RawArtifactsUnavailableError
# ---------------------------------------------------------------------------


class TestFailurePathDispatchFailure:
    def test_failure_dispatch_failure_raises_raw_artifacts_unavailable_error(self) -> None:
        """Failure path: injected transport returns success: False. Assert
        BathosSplitComputeEvaluator raises RawArtifactsUnavailableError (not AttributeError/
        KeyError/etc) and that score_fn is NEVER called."""
        score_fn_calls = []

        def score_fn(
            _raw_artifact_paths: tuple[Path, ...], _split_paths: tuple[Path, ...]
        ) -> dict[str, float]:
            score_fn_calls.append(True)
            return {"score": 1.0}

        # Return a failed envelope
        failing_envelope = _ok_envelope(script_path="c.py", exit_code=1, success=False)
        transport = _RecordingTransport(failing_envelope)
        adapter = BathosCampaignAdapter(transport=transport, token="test-token")

        locked = ClosureManifest(
            evaluator_paths=(),
            split_paths=(),
            metric_def_paths=(),
            pinned_deps_source=Path("uv.lock"),
            config={},
            closure_hash="fake-hash",
        )

        frozen_context = BathosFrozenContext(
            locked=locked,
            campaign_adapter=adapter,
            campaign_id="camp-1",
            score_fn=score_fn,
        )

        candidate = BathosCandidate(
            script_path="script.py",
            output_paths=("output.txt",),
        )

        evaluator = BathosSplitComputeEvaluator()

        # Assert the right exception is raised
        with pytest.raises(RawArtifactsUnavailableError, match="did not succeed"):
            evaluator(frozen_context, candidate)

        # Assert score_fn was never called
        assert score_fn_calls == []

    def test_failure_non_zero_exit_code_treated_as_failure(self) -> None:
        """Failure path: exit_code != 0 is treated as failure even if the bathos tool reported
        ok: True (though bathos should not do this). Assert RawArtifactsUnavailableError."""
        score_fn_calls = []

        def score_fn(
            _raw_artifact_paths: tuple[Path, ...], _split_paths: tuple[Path, ...]
        ) -> dict[str, float]:
            score_fn_calls.append(True)
            return {"score": 1.0}

        # Non-zero exit code + success=True (bathos ok: True via _ok_envelope).
        # __call__ must still HALT; today's gate only checks success, so this is RED until fixer.
        failing_envelope = _ok_envelope(script_path="c.py", exit_code=127, success=True)
        transport = _RecordingTransport(failing_envelope)
        adapter = BathosCampaignAdapter(transport=transport, token="test-token")

        locked = ClosureManifest(
            evaluator_paths=(),
            split_paths=(),
            metric_def_paths=(),
            pinned_deps_source=Path("uv.lock"),
            config={},
            closure_hash="fake-hash",
        )

        frozen_context = BathosFrozenContext(
            locked=locked,
            campaign_adapter=adapter,
            campaign_id="camp-1",
            score_fn=score_fn,
        )

        candidate = BathosCandidate(
            script_path="missing.py",
            output_paths=("output.txt",),
        )

        evaluator = BathosSplitComputeEvaluator()

        with pytest.raises(RawArtifactsUnavailableError):
            evaluator(frozen_context, candidate)

        assert score_fn_calls == []


# ---------------------------------------------------------------------------
# #4136: evaluator __call__ edges — success=True + exit_code!=0 must HALT
# ---------------------------------------------------------------------------


class TestEvaluatorEdges4136:
    """#4136: BathosSplitComputeEvaluator.__call__ must refuse to score when dispatch
    reports success=True but exit_code != 0, and must still raise on empty output_paths
    after a successful (exit_code=0) dispatch. Happy path (success=True, exit_code=0,
    non-empty output_paths) still scores.
    """

    def test_call_success_true_nonzero_exit_code_raises_and_does_not_score(self) -> None:
        """__call__ with success=True, exit_code=127 (ok: True via _ok_envelope) must raise
        RawArtifactsUnavailableError, never call score_fn, and make exactly one transport
        `run` call. RED today: __call__ only checks `if not run_result.success`."""
        score_fn_calls: list[bool] = []

        def score_fn(
            _raw_artifact_paths: tuple[Path, ...], _split_paths: tuple[Path, ...]
        ) -> dict[str, float]:
            score_fn_calls.append(True)
            return {"score": 1.0}

        envelope = _ok_envelope(script_path="c.py", exit_code=127, success=True)
        transport = _RecordingTransport(envelope)
        adapter = BathosCampaignAdapter(transport=transport, token="test-token")

        locked = ClosureManifest(
            evaluator_paths=(),
            split_paths=(),
            metric_def_paths=(),
            pinned_deps_source=Path("uv.lock"),
            config={},
            closure_hash="fake-hash",
        )

        frozen_context = BathosFrozenContext(
            locked=locked,
            campaign_adapter=adapter,
            campaign_id="camp-1",
            score_fn=score_fn,
        )

        candidate = BathosCandidate(
            script_path="missing.py",
            output_paths=("output.txt",),
        )

        evaluator = BathosSplitComputeEvaluator()

        with pytest.raises(RawArtifactsUnavailableError):
            evaluator(frozen_context, candidate)

        assert score_fn_calls == []
        assert len(transport.calls) == 1
        tool_name, _arguments = transport.calls[0]
        assert tool_name == "run"

    def test_call_success_true_empty_output_paths_raises_and_does_not_score(self) -> None:
        """__call__ with success=True, exit_code=0, and candidate.output_paths=() must raise
        RawArtifactsUnavailableError (score_raw_artifacts already does this) and never call
        score_fn."""
        score_fn_calls: list[bool] = []

        def score_fn(
            _raw_artifact_paths: tuple[Path, ...], _split_paths: tuple[Path, ...]
        ) -> dict[str, float]:
            score_fn_calls.append(True)
            return {"score": 1.0}

        envelope = _ok_envelope(script_path="c.py", exit_code=0, success=True)
        transport = _RecordingTransport(envelope)
        adapter = BathosCampaignAdapter(transport=transport, token="test-token")

        locked = ClosureManifest(
            evaluator_paths=(),
            split_paths=(),
            metric_def_paths=(),
            pinned_deps_source=Path("uv.lock"),
            config={},
            closure_hash="fake-hash",
        )

        frozen_context = BathosFrozenContext(
            locked=locked,
            campaign_adapter=adapter,
            campaign_id="camp-1",
            score_fn=score_fn,
        )

        candidate = BathosCandidate(
            script_path="script.py",
            output_paths=(),
        )

        evaluator = BathosSplitComputeEvaluator()

        with pytest.raises(RawArtifactsUnavailableError):
            evaluator(frozen_context, candidate)

        assert score_fn_calls == []

    def test_call_success_true_zero_exit_code_still_scores(self, tmp_path: Path) -> None:
        """Happy __call__: success=True, exit_code=0, non-empty output_paths still scores
        and returns the fitness dict from score_fn."""
        raw_artifact = tmp_path / "artifact.txt"
        raw_artifact.write_text("edge-4136", encoding="utf-8")

        def score_fn(
            _raw_artifact_paths: tuple[Path, ...], _split_paths: tuple[Path, ...]
        ) -> dict[str, float]:
            return {"score": 7.0}

        envelope = _ok_envelope(script_path="c.py", exit_code=0, success=True)
        transport = _RecordingTransport(envelope)
        adapter = BathosCampaignAdapter(transport=transport, token="test-token")

        locked = ClosureManifest(
            evaluator_paths=(),
            split_paths=(),
            metric_def_paths=(),
            pinned_deps_source=Path("uv.lock"),
            config={},
            closure_hash="fake-hash",
        )

        frozen_context = BathosFrozenContext(
            locked=locked,
            campaign_adapter=adapter,
            campaign_id="camp-1",
            score_fn=score_fn,
        )

        candidate = BathosCandidate(
            script_path="script.py",
            output_paths=(str(raw_artifact),),
        )

        evaluator = BathosSplitComputeEvaluator()
        result = evaluator(frozen_context, candidate)

        assert result == {"score": 7.0}


# ---------------------------------------------------------------------------
# #4164: production constructor + declaration-lists helper
# ---------------------------------------------------------------------------


def _dummy_lock_score_fn(
    _raw_artifact_paths: tuple[Path, ...], _split_paths: tuple[Path, ...]
) -> dict[str, float]:
    return {"score": 1.0}


def _lock_factory_kwargs(
    evaluator: Path, split: Path, metric: Path, pinned: Path
) -> dict[str, Any]:
    transport = _RecordingTransport(_ok_envelope(script_path="c.py", exit_code=0, success=True))
    adapter = BathosCampaignAdapter(transport=transport, token="test-token")
    return {
        "evaluator_paths": (evaluator,),
        "split_paths": (split,),
        "metric_def_paths": (metric,),
        "config": {"k": "v"},
        "pinned_deps_source": pinned,
        "campaign_adapter": adapter,
        "campaign_id": "camp-1",
        "score_fn": _dummy_lock_score_fn,
    }


def _write_declared_closure_files(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text("# evaluator v1\n", encoding="utf-8")
    split = tmp_path / "split.txt"
    split.write_text("split v1\n", encoding="utf-8")
    metric = tmp_path / "metric.txt"
    metric.write_text("metric v1\n", encoding="utf-8")
    pinned = tmp_path / "uv.lock"
    pinned.write_text("# pinned v1\n", encoding="utf-8")
    return evaluator, split, metric, pinned


class TestLockBathosFrozenContext:
    def test_lock_bathos_frozen_context_hashes_declared_files(self, tmp_path: Path) -> None:
        """#4164: lock_bathos_frozen_context hashes declared evaluator/split/metric files
        via build_closure_manifest. Hash is deterministic; changing a declared file changes it.
        Empty path tuples remain valid (existing tests/smoke) -- this test uses real files.
        """
        from controller.evaluate_adapter import lock_bathos_frozen_context

        evaluator, split, metric, pinned = _write_declared_closure_files(tmp_path)
        kwargs = _lock_factory_kwargs(evaluator, split, metric, pinned)

        first = lock_bathos_frozen_context(**kwargs)
        second = lock_bathos_frozen_context(**kwargs)
        assert first.locked.closure_hash == second.locked.closure_hash

        expected = build_closure_manifest(
            evaluator_paths=(evaluator,),
            split_paths=(split,),
            metric_def_paths=(metric,),
            config={"k": "v"},
            pinned_deps_source=pinned,
        )
        assert first.locked.closure_hash == expected.closure_hash
        assert first.campaign_id == "camp-1"
        assert first.score_fn is _dummy_lock_score_fn

        evaluator.write_text("# evaluator v2\n", encoding="utf-8")
        after = lock_bathos_frozen_context(**kwargs)
        assert after.locked.closure_hash != first.locked.closure_hash


class TestClosureDeclarationLists:
    def test_closure_declaration_lists_roundtrip_write_manifest_dict(self, tmp_path: Path) -> None:
        """#4164: factory lock → closure_declaration_lists → write_manifest_dict produces
        manifest["closure"] with those string paths. Tests MAY import write_manifest_dict;
        controller production code must not.
        """
        from controller.evaluate_adapter import (
            closure_declaration_lists,
            lock_bathos_frozen_context,
        )
        from xtrax.cli.manifest import write_manifest_dict

        evaluator, split, metric, pinned = _write_declared_closure_files(tmp_path)
        frozen = lock_bathos_frozen_context(
            **_lock_factory_kwargs(evaluator, split, metric, pinned)
        )
        evaluator_paths, split_paths, metric_def_paths = closure_declaration_lists(frozen.locked)

        cfg_dict = dict(
            schema_version=1,
            model={"path": "tests.cli._run_fixtures:make_model", "kwargs": {}},
            optimizer={"path": "xtrax.training.optim:adamw_with_schedule", "kwargs": {}},
            loss={"path": "tests.cli._run_fixtures:make_loss", "kwargs": {}},
            data={
                "factory": "tests.cli._run_fixtures:make_dataset",
                "kwargs": {},
                "batch_size": 4,
            },
            seed=42,
            num_epochs=3,
        )
        manifest = write_manifest_dict(
            run_dir=str(tmp_path / "runs" / "roundtrip"),
            cfg_dict=cfg_dict,
            run_id="roundtrip",
            config_hash_val="roundtrip",
            evaluator_paths=evaluator_paths,
            split_paths=split_paths,
            metric_def_paths=metric_def_paths,
        )
        assert manifest["closure"] == {
            "evaluator_paths": evaluator_paths,
            "split_paths": split_paths,
            "metric_def_paths": metric_def_paths,
        }
        assert evaluator_paths == [str(evaluator.resolve())]
        assert split_paths == [str(split.resolve())]
        assert metric_def_paths == [str(metric.resolve())]
