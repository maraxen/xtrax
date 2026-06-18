"""Merge-blocking contract tests for xtrax.devtools.baseline (N1.2 / #1578)."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from xtrax.devtools.baseline import (
    BASELINE_SCHEMA_VERSION,
    AuditBaseline,
    MetricEntry,
    compare_metric,
    evaluate_metric,
    load_baseline,
    save_baseline,
    update_metric,
)
from xtrax.devtools.emit import SchemaVersionMismatchError

SEED_BASELINE = AuditBaseline(
    schema_version=BASELINE_SCHEMA_VERSION,
    updated_at="2026-06-19T00:00:00+00:00",
    metrics={
        "jax_purity.jl_violation_count": MetricEntry(
            key="jax_purity.jl_violation_count",
            value=0.0,
            comparator="minimize",
        ),
        "type_hardening.annotation_coverage_pct": MetricEntry(
            key="type_hardening.annotation_coverage_pct",
            value=0.0,
            comparator="maximize",
        ),
        "documentation.interrogate_coverage_pct": MetricEntry(
            key="documentation.interrogate_coverage_pct",
            value=0.0,
            comparator="maximize",
        ),
    },
)


def test_round_trip_load_save_preserves_metrics(tmp_path: Path) -> None:
    path = tmp_path / "audit_baseline.json"
    save_baseline(SEED_BASELINE, path=path)
    restored = load_baseline(path=path)
    assert restored.schema_version == BASELINE_SCHEMA_VERSION
    assert restored.metrics == SEED_BASELINE.metrics


def test_minimize_regression_and_tighten() -> None:
    baseline = SEED_BASELINE
    key = "jax_purity.jl_violation_count"

    assert compare_metric(0.0, 1.0, "minimize") is False
    assert compare_metric(0.0, 0.0, "minimize") is True
    assert compare_metric(0.0, -1.0, "minimize") is True

    passes, should_update = evaluate_metric(baseline, key, 1.0)
    assert passes is False
    assert should_update is False

    passes, should_update = evaluate_metric(baseline, key, 0.0)
    assert passes is True
    assert should_update is False

    passes, should_update = evaluate_metric(baseline, key, -1.0)
    assert passes is True
    assert should_update is True

    tightened = update_metric(baseline, key, -1.0, "minimize")
    assert tightened.metrics[key].value == -1.0


def test_maximize_regression_and_tighten() -> None:
    baseline = SEED_BASELINE
    key = "type_hardening.annotation_coverage_pct"

    assert compare_metric(0.0, -1.0, "maximize") is False
    assert compare_metric(0.0, 0.0, "maximize") is True
    assert compare_metric(0.0, 42.5, "maximize") is True

    passes, should_update = evaluate_metric(baseline, key, -1.0)
    assert passes is False
    assert should_update is False

    passes, should_update = evaluate_metric(baseline, key, 42.5)
    assert passes is True
    assert should_update is True

    tightened = update_metric(baseline, key, 42.5, "maximize")
    assert tightened.metrics[key].value == 42.5


def test_best_ever_recorded_not_blocked() -> None:
    assert compare_metric(10.0, 99.0, "best_ever") is True
    assert compare_metric(10.0, 5.0, "best_ever") is True

    baseline = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            "performance.wall_time_ms": MetricEntry(
                key="performance.wall_time_ms",
                value=10.0,
                comparator="best_ever",
            ),
        },
    )
    passes, should_update = evaluate_metric(baseline, "performance.wall_time_ms", 8.0)
    assert passes is True
    assert should_update is True

    passes, should_update = evaluate_metric(baseline, "performance.wall_time_ms", 12.0)
    assert passes is True
    assert should_update is False


def test_schema_version_mismatch_loud_fails(tmp_path: Path) -> None:
    path = tmp_path / "audit_baseline.json"
    stale = {
        "schema_version": 99,
        "updated_at": "2026-06-19T00:00:00+00:00",
        "metrics": {},
    }
    path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(SchemaVersionMismatchError, match="schema_version mismatch"):
        load_baseline(path=path)


def test_missing_metric_bootstraps_on_evaluate_and_update() -> None:
    baseline = SEED_BASELINE
    key = "structure.cognitive_complexity_max"

    passes, should_update = evaluate_metric(baseline, key, 12.0)
    assert passes is True
    assert should_update is True

    bootstrapped = update_metric(baseline, key, 12.0, "minimize")
    assert bootstrapped.metrics[key] == MetricEntry(
        key=key,
        value=12.0,
        comparator="minimize",
    )


def test_atomic_save_does_not_leave_corrupt_partial_file(tmp_path: Path) -> None:
    path = tmp_path / "audit_baseline.json"
    save_baseline(SEED_BASELINE, path=path)
    original = path.read_text(encoding="utf-8")

    def fail_replace(src: str | Path, dst: str | Path) -> None:
        raise OSError("simulated rename failure")

    with patch("xtrax.devtools.baseline.os.replace", side_effect=fail_replace):
        with pytest.raises(OSError, match="simulated rename failure"):
            save_baseline(SEED_BASELINE, path=path)

    assert path.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob(".audit_baseline.json.*.tmp")) == []


def test_committed_seed_baseline_loads() -> None:
    repo_baseline = Path(".praxia/audit_baseline.json")
    if not repo_baseline.is_file():
        pytest.skip("seed baseline not present in checkout")
    loaded = load_baseline(path=repo_baseline)
    assert loaded.schema_version == BASELINE_SCHEMA_VERSION
    assert len(loaded.metrics) >= 3
