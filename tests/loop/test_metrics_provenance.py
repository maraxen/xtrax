"""Tests for the metrics-provenance gate (T2-06, #2181, AC-8, F1)."""

import tomllib
from pathlib import Path

import pytest

from xtrax.loop.closure_lock import (
    ProtectedPathMutatedError,
    UnlistedReadError,
    build_closure_manifest,
)
from xtrax.loop.metrics_provenance import (
    MetricsProvenanceRecord,
    UnprovenancedMetricsError,
    build_metrics_provenance_attestation_toml,
    evaluate_with_provenance,
    verify_metrics_provenance,
    write_metrics_provenance_attestation,
)


@pytest.fixture
def closure_dir(tmp_path: Path) -> Path:
    (tmp_path / "evaluator.py").write_text("def evaluate(): return 1\n", encoding="utf-8")
    (tmp_path / "splits.json").write_text('{"train": [1, 2]}', encoding="utf-8")
    (tmp_path / "metric_defs.json").write_text('{"metric": "accuracy"}', encoding="utf-8")
    (tmp_path / "uv.lock").write_text("# pinned deps v1\n", encoding="utf-8")
    return tmp_path


def _locked_manifest(closure_dir: Path):
    return build_closure_manifest(
        evaluator_paths=(closure_dir / "evaluator.py",),
        split_paths=(closure_dir / "splits.json",),
        metric_def_paths=(closure_dir / "metric_defs.json",),
        config={"lr": 0.1},
        pinned_deps_source=closure_dir / "uv.lock",
    )


def _evaluator(frozen_context: object, candidate: object) -> dict[str, float]:
    del frozen_context, candidate
    return {"score": 0.9, "loss": 0.1}


class TestEvaluateWithProvenance:
    def test_returns_record_traced_to_locked_closure_hash(self, closure_dir: Path) -> None:
        locked = _locked_manifest(closure_dir)
        record = evaluate_with_provenance(
            locked, _evaluator, None, None, current_config={"lr": 0.1}, run_id="run-1"
        )
        assert record.fitness == {"score": 0.9, "loss": 0.1}
        assert record.evaluator_closure_hash == locked.closure_hash
        assert record.run_id == "run-1"
        assert record.created_at != ""

    def test_still_enforces_closure_drift_check(self, closure_dir: Path) -> None:
        """evaluate_with_provenance wraps guarded_evaluate -- it must not bypass T2-05's own
        checks just because it adds a provenance layer on top.
        """
        locked = _locked_manifest(closure_dir)
        (closure_dir / "evaluator.py").write_text("def evaluate(): return 999\n", encoding="utf-8")

        from xtrax.loop.closure_lock import ClosureHashMismatchError

        with pytest.raises(ClosureHashMismatchError):
            evaluate_with_provenance(
                locked, _evaluator, None, None, current_config={"lr": 0.1}, run_id="run-1"
            )

    def test_still_enforces_protected_path_check(self, closure_dir: Path) -> None:
        locked = _locked_manifest(closure_dir)
        touched = frozenset({(closure_dir / "splits.json").resolve()})
        with pytest.raises(ProtectedPathMutatedError):
            evaluate_with_provenance(
                locked,
                _evaluator,
                None,
                None,
                current_config={"lr": 0.1},
                run_id="run-1",
                candidate_touched_paths=touched,
            )

    def test_still_enforces_unlisted_read_check(self, closure_dir: Path) -> None:
        locked = _locked_manifest(closure_dir)
        secret = closure_dir / "not_in_manifest.json"
        secret.write_text("{}", encoding="utf-8")

        def reads_undeclared_path(frozen_context: object, candidate: object) -> dict[str, float]:
            del frozen_context, candidate
            secret.read_text()
            return {"score": 0.9}

        with pytest.raises(UnlistedReadError):
            evaluate_with_provenance(
                locked,
                reads_undeclared_path,
                None,
                None,
                current_config={"lr": 0.1},
                run_id="run-1",
            )


class TestVerifyMetricsProvenance:
    def test_passes_silently_when_hash_matches(self, closure_dir: Path) -> None:
        locked = _locked_manifest(closure_dir)
        record = evaluate_with_provenance(
            locked, _evaluator, None, None, current_config={"lr": 0.1}, run_id="run-1"
        )
        verify_metrics_provenance(record, locked=locked)

    def test_raises_on_hash_mismatch(self, closure_dir: Path) -> None:
        locked = _locked_manifest(closure_dir)
        forged = MetricsProvenanceRecord(
            fitness={"score": 1.0}, evaluator_closure_hash="not-the-real-hash", run_id="forged"
        )
        with pytest.raises(UnprovenancedMetricsError, match="discard, void iteration"):
            verify_metrics_provenance(forged, locked=locked)


class TestAttestationToml:
    def test_round_trips_through_tomllib(self, closure_dir: Path) -> None:
        locked = _locked_manifest(closure_dir)
        record = evaluate_with_provenance(
            locked, _evaluator, None, None, current_config={"lr": 0.1}, run_id="run-1"
        )
        toml_text = build_metrics_provenance_attestation_toml(record)
        parsed = tomllib.loads(toml_text)

        assert parsed["attestation"]["kind"] == "metrics_provenance"
        assert parsed["attestation"]["attested"]["run_id"] == "run-1"
        assert parsed["attestation"]["attested"]["evaluator_closure_hash"] == locked.closure_hash
        assert parsed["attestation"]["fitness"] == {"loss": 0.1, "score": 0.9}

    def test_escapes_run_id_containing_quote_and_backslash(self) -> None:
        record = MetricsProvenanceRecord(
            fitness={"score": 1.0},
            evaluator_closure_hash="abc123",
            run_id='weird"run\\id',
        )
        toml_text = build_metrics_provenance_attestation_toml(record)
        parsed = tomllib.loads(toml_text)
        assert parsed["attestation"]["attested"]["run_id"] == 'weird"run\\id'

    def test_write_creates_parent_directories(self, tmp_path: Path) -> None:
        record = MetricsProvenanceRecord(
            fitness={"score": 1.0}, evaluator_closure_hash="abc123", run_id="run-1"
        )
        dest = tmp_path / "nested" / "dir" / "attestation.toml"
        result_path = write_metrics_provenance_attestation(record, dest)
        assert result_path == dest
        assert dest.exists()
        assert tomllib.loads(dest.read_text())["attestation"]["kind"] == "metrics_provenance"
