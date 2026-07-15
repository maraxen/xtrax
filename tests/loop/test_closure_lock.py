"""Tests for the eval-closure-invariant / full-closure SHA lock (T2-05, #2181, AC-7, PM-1)."""

from pathlib import Path

import pytest

from xtrax.loop.closure_lock import (
    ClosureHashMismatchError,
    ProtectedPathMutatedError,
    UnlistedReadError,
    build_closure_manifest,
    guarded_evaluate,
    track_reads,
    verify_closure,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def closure_dir(tmp_path: Path) -> Path:
    (tmp_path / "evaluator.py").write_text("def evaluate(): return 1\n", encoding="utf-8")
    (tmp_path / "splits.json").write_text('{"train": [1, 2]}', encoding="utf-8")
    (tmp_path / "metric_defs.json").write_text('{"metric": "accuracy"}', encoding="utf-8")
    (tmp_path / "uv.lock").write_text("# pinned deps v1\n", encoding="utf-8")
    return tmp_path


def _locked_manifest(closure_dir: Path, config: dict | None = None):
    return build_closure_manifest(
        evaluator_paths=(closure_dir / "evaluator.py",),
        split_paths=(closure_dir / "splits.json",),
        metric_def_paths=(closure_dir / "metric_defs.json",),
        config=config or {"lr": 0.1},
        pinned_deps_source=closure_dir / "uv.lock",
    )


class TestBuildClosureManifest:
    def test_stable_hash_for_identical_inputs(self, closure_dir: Path) -> None:
        a = _locked_manifest(closure_dir)
        b = _locked_manifest(closure_dir)
        assert a.closure_hash == b.closure_hash

    def test_stable_under_declared_path_reordering(self, closure_dir: Path) -> None:
        forward = build_closure_manifest(
            evaluator_paths=(closure_dir / "evaluator.py",),
            split_paths=(closure_dir / "splits.json",),
            metric_def_paths=(closure_dir / "metric_defs.json",),
            config={"lr": 0.1, "batch": 8},
            pinned_deps_source=closure_dir / "uv.lock",
        )
        reordered_config = build_closure_manifest(
            evaluator_paths=(closure_dir / "evaluator.py",),
            split_paths=(closure_dir / "splits.json",),
            metric_def_paths=(closure_dir / "metric_defs.json",),
            config={"batch": 8, "lr": 0.1},
            pinned_deps_source=closure_dir / "uv.lock",
        )
        assert forward.closure_hash == reordered_config.closure_hash

    def test_hash_changes_when_evaluator_code_changes(self, closure_dir: Path) -> None:
        before = _locked_manifest(closure_dir)
        _write(closure_dir / "evaluator.py", "def evaluate(): return 2\n")
        after = _locked_manifest(closure_dir)
        assert before.closure_hash != after.closure_hash

    def test_hash_changes_when_config_changes(self, closure_dir: Path) -> None:
        before = _locked_manifest(closure_dir, config={"lr": 0.1})
        after = _locked_manifest(closure_dir, config={"lr": 0.2})
        assert before.closure_hash != after.closure_hash

    def test_hash_changes_when_pinned_deps_change(self, closure_dir: Path) -> None:
        before = _locked_manifest(closure_dir)
        _write(closure_dir / "uv.lock", "# pinned deps v2\n")
        after = _locked_manifest(closure_dir)
        assert before.closure_hash != after.closure_hash


class TestVerifyClosure:
    def test_passes_silently_when_nothing_changed(self, closure_dir: Path) -> None:
        locked = _locked_manifest(closure_dir)
        verify_closure(locked, current_config={"lr": 0.1})

    def test_raises_on_evaluator_code_drift(self, closure_dir: Path) -> None:
        locked = _locked_manifest(closure_dir)
        _write(closure_dir / "evaluator.py", "def evaluate(): return 999\n")
        with pytest.raises(ClosureHashMismatchError, match="closure hash drift"):
            verify_closure(locked, current_config={"lr": 0.1})

    def test_raises_on_config_drift_even_though_locked_config_is_unchanged(
        self, closure_dir: Path
    ) -> None:
        """Regression guard: recomputing against `locked.config` instead of a freshly-supplied
        `current_config` would make this drift undetectable by construction.
        """
        locked = _locked_manifest(closure_dir, config={"lr": 0.1})
        with pytest.raises(ClosureHashMismatchError, match="closure hash drift"):
            verify_closure(locked, current_config={"lr": 0.999})

    def test_raises_on_missing_closure_member(self, closure_dir: Path) -> None:
        locked = _locked_manifest(closure_dir)
        (closure_dir / "evaluator.py").unlink()
        with pytest.raises(ClosureHashMismatchError, match="vanished"):
            verify_closure(locked, current_config={"lr": 0.1})

    def test_raises_on_protected_path_mutation(self, closure_dir: Path) -> None:
        locked = _locked_manifest(closure_dir)
        touched = frozenset({(closure_dir / "splits.json").resolve()})
        with pytest.raises(ProtectedPathMutatedError, match="protected closure path"):
            verify_closure(locked, current_config={"lr": 0.1}, candidate_touched_paths=touched)

    def test_passes_when_candidate_touches_an_unprotected_path(self, closure_dir: Path) -> None:
        locked = _locked_manifest(closure_dir)
        touched = frozenset({(closure_dir / "unrelated_output.json").resolve()})
        verify_closure(locked, current_config={"lr": 0.1}, candidate_touched_paths=touched)


class TestTrackReads:
    def test_records_a_file_opened_inside_the_context(self, tmp_path: Path) -> None:
        target = _write(tmp_path / "watched.txt", "hi")
        with track_reads() as observed:
            target.read_text()
        assert target.resolve() in observed

    def test_does_not_record_opens_outside_the_context(self, tmp_path: Path) -> None:
        before = _write(tmp_path / "before.txt", "hi")
        before.read_text()
        with track_reads() as observed:
            pass
        assert before.resolve() not in observed

    def test_nested_contexts_do_not_leak_into_each_other(self, tmp_path: Path) -> None:
        inner_file = _write(tmp_path / "inner.txt", "hi")
        outer_file = _write(tmp_path / "outer.txt", "hi")
        with track_reads() as outer_observed:
            outer_file.read_text()
            with track_reads() as inner_observed:
                inner_file.read_text()
            assert inner_file.resolve() in inner_observed
            assert inner_file.resolve() not in outer_observed
        assert outer_file.resolve() in outer_observed


class TestGuardedEvaluate:
    def test_returns_evaluator_result_when_reads_are_all_declared(self, closure_dir: Path) -> None:
        locked = _locked_manifest(closure_dir)

        def evaluator(frozen_context: object, candidate: object) -> dict[str, float]:
            del frozen_context, candidate
            (closure_dir / "splits.json").read_text()
            return {"score": 1.0}

        result = guarded_evaluate(
            locked, evaluator, None, None, current_config={"lr": 0.1}
        )
        assert result == {"score": 1.0}

    def test_raises_unlisted_read_error_for_undeclared_path(self, closure_dir: Path) -> None:
        locked = _locked_manifest(closure_dir)
        secret = _write(closure_dir / "not_in_manifest.json", "{}")

        def evaluator(frozen_context: object, candidate: object) -> dict[str, float]:
            del frozen_context, candidate
            secret.read_text()
            return {"score": 1.0}

        with pytest.raises(UnlistedReadError, match="undeclared path"):
            guarded_evaluate(locked, evaluator, None, None, current_config={"lr": 0.1})

    def test_hash_drift_halts_before_evaluator_is_ever_called(self, closure_dir: Path) -> None:
        locked = _locked_manifest(closure_dir)
        _write(closure_dir / "evaluator.py", "def evaluate(): return 999\n")

        calls = []

        def evaluator(frozen_context: object, candidate: object) -> dict[str, float]:
            calls.append(1)
            return {"score": 1.0}

        with pytest.raises(ClosureHashMismatchError):
            guarded_evaluate(locked, evaluator, None, None, current_config={"lr": 0.1})
        assert calls == []

    def test_protected_path_mutation_halts_before_evaluator_is_ever_called(
        self, closure_dir: Path
    ) -> None:
        locked = _locked_manifest(closure_dir)
        touched = frozenset({(closure_dir / "metric_defs.json").resolve()})

        calls = []

        def evaluator(frozen_context: object, candidate: object) -> dict[str, float]:
            calls.append(1)
            return {"score": 1.0}

        with pytest.raises(ProtectedPathMutatedError):
            guarded_evaluate(
                locked,
                evaluator,
                None,
                None,
                current_config={"lr": 0.1},
                candidate_touched_paths=touched,
            )
        assert calls == []
