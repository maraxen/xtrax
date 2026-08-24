"""Emitter regression pins, ported from prolix's test_emit_probe_record.py.

Prolix cluster array 20762713: timing succeeded, then ClaimValidityError
because `{... if value is not None} or None` collapsed an EMPTY attribution
dict to None when every scope was absent from the trace. scopes-without-
attribution and no-trace-at-all are different facts; both tests below pin
that distinction through the xtrax generic emitter.
"""

from __future__ import annotations

from xtrax.profiling.emitters import (
    attribution_from_scopes,
    emit_probe_record,
)
from xtrax.profiling.record import ProbeRecord


def test_attribution_from_scopes_all_none_is_empty_dict_not_none():
    scopes = {"tiling_vmap": None, "tiling_safemap": None}
    assert attribution_from_scopes(scopes) == {}


def test_attribution_from_scopes_keeps_present_labels():
    scopes = {"tiling_vmap": (0.1, 3), "tiling_safemap": None}
    assert attribution_from_scopes(scopes) == {"tiling_vmap": "named_scope"}


def test_attribution_from_scopes_no_trace_returns_empty_dict():
    assert attribution_from_scopes(None) == {}


def test_emit_keeps_all_none_scopes(tmp_path):
    path = tmp_path / "record.json"
    scopes = {"tiling_vmap": None, "tiling_dedup_gather": None}
    emit_probe_record(
        path=path,
        probe_id="stage1_tiling_cpu",
        stage=1,
        n_atoms=512,
        platform="cpu",
        metrics={"total_step_seconds": 0.8e-3},
        scopes=scopes,
        attribution_method=attribution_from_scopes(scopes),
        config={"strategy": "vmap", "n_padded_rows": "1024"},
        # git_sha/timestamp/jax versions auto-captured from this machine.
    )
    rec = ProbeRecord.read(path)
    assert rec.scopes == scopes
    assert rec.attribution_method == {}


def test_emit_omits_attribution_when_no_trace_captured(tmp_path):
    path = tmp_path / "record.json"
    emit_probe_record(
        path=path,
        probe_id="stage0_tiling_cost",
        stage=0,
        n_atoms=512,
        platform="cpu",
        metrics={"flops": 1.0e6},
        config={"strategy": "safemap"},
    )
    rec = ProbeRecord.read(path)
    assert rec.scopes is None
    assert rec.attribution_method is None


def test_emit_scopes_without_attribution_raises_before_write(tmp_path):
    import pytest

    from xtrax.profiling.claims import ClaimValidityError

    path = tmp_path / "never_written.json"
    with pytest.raises(ClaimValidityError):
        emit_probe_record(
            path=path,
            probe_id="bad",
            stage=1,
            n_atoms=8,
            platform="cpu",
            metrics={"x": 1.0},
            scopes={"tiling_vmap": (0.1, 1)},
            attribution_method=None,
        )
    assert not path.exists(), "invalid record must not land on disk"
