"""Tests for prereg-match (T2-16, #2181, AC-6, F8)."""

import pytest

from xtrax.loop.prereg_match import (
    PreregMatchError,
    PreregSidecar,
    RunConfig,
    assert_prereg_match,
)

_SHA = "a" * 64


def _run_config(**overrides: object) -> RunConfig:
    defaults: dict[str, object] = {
        "script_sha256": _SHA,
        "hypothesis": "candidate beats baseline on accuracy",
        "metrics": ("accuracy", "loss"),
        "config": {"lr": 0.1, "seed": 42},
    }
    defaults.update(overrides)
    return RunConfig(**defaults)  # type: ignore[arg-type]


def _sidecar(**overrides: object) -> PreregSidecar:
    defaults: dict[str, object] = {
        "script_sha256": _SHA,
        "hypothesis": "candidate beats baseline on accuracy",
        "metrics": ("accuracy", "loss"),
        "config": {"lr": 0.1, "seed": 42},
    }
    defaults.update(overrides)
    return PreregSidecar(**defaults)  # type: ignore[arg-type]


class TestAssertPreregMatch:
    def test_matching_config_and_sidecar_does_not_raise(self) -> None:
        assert_prereg_match(_run_config(), _sidecar())

    def test_metrics_order_does_not_matter(self) -> None:
        run_config = _run_config(metrics=("loss", "accuracy"))
        assert_prereg_match(run_config, _sidecar())

    def test_script_sha256_mismatch_is_named(self) -> None:
        run_config = _run_config(script_sha256="b" * 64)
        with pytest.raises(PreregMatchError) as exc_info:
            assert_prereg_match(run_config, _sidecar())
        assert any("script_sha256" in field for field in exc_info.value.payload.mismatched_fields)

    def test_hypothesis_mismatch_is_named(self) -> None:
        run_config = _run_config(hypothesis="a different hypothesis entirely")
        with pytest.raises(PreregMatchError) as exc_info:
            assert_prereg_match(run_config, _sidecar())
        assert any("hypothesis" in field for field in exc_info.value.payload.mismatched_fields)

    def test_missing_metric_is_named(self) -> None:
        run_config = _run_config(metrics=("accuracy",))
        with pytest.raises(PreregMatchError) as exc_info:
            assert_prereg_match(run_config, _sidecar())
        diffs = exc_info.value.payload.mismatched_fields
        assert any("missing from run_config" in field and "loss" in field for field in diffs)

    def test_extra_metric_is_named(self) -> None:
        run_config = _run_config(metrics=("accuracy", "loss", "f1"))
        with pytest.raises(PreregMatchError) as exc_info:
            assert_prereg_match(run_config, _sidecar())
        diffs = exc_info.value.payload.mismatched_fields
        assert any("not pre-registered" in field and "f1" in field for field in diffs)

    def test_config_mismatch_is_named(self) -> None:
        run_config = _run_config(config={"lr": 0.5, "seed": 42})
        with pytest.raises(PreregMatchError) as exc_info:
            assert_prereg_match(run_config, _sidecar())
        assert any("config mismatch" in field for field in exc_info.value.payload.mismatched_fields)

    def test_multiple_mismatches_all_surface_at_once(self) -> None:
        run_config = _run_config(
            hypothesis="wrong hypothesis",
            metrics=("accuracy",),
            config={"lr": 0.9},
        )
        with pytest.raises(PreregMatchError) as exc_info:
            assert_prereg_match(run_config, _sidecar())
        diffs = exc_info.value.payload.mismatched_fields
        assert len(diffs) == 3
        assert any("hypothesis" in d for d in diffs)
        assert any("missing from run_config" in d for d in diffs)
        assert any("config mismatch" in d for d in diffs)

    def test_payload_carries_original_objects_for_introspection(self) -> None:
        run_config = _run_config(hypothesis="mismatched")
        sidecar = _sidecar()
        with pytest.raises(PreregMatchError) as exc_info:
            assert_prereg_match(run_config, sidecar)
        payload = exc_info.value.payload
        assert payload.run_config == run_config
        assert payload.sidecar == sidecar
        assert payload.reason
