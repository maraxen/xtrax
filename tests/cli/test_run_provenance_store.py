"""#457(1) acceptance tests: run CLI persists provenance through derive_sink_spec.

These drive the REAL public interface (`run_from_config`) end-to-end against a
real zarr store AND real orbax checkpoints -- no mocks for the persistence
layers. The single mocked surface (Engine in the crash-window test) exists
solely to inject a mid-training failure.

Contract under test:
- store lands at `.xtrax/runs/<run_id>/metrics.zarr`
- store root `run_id` == manifest `run_id` (single-sourced join key)
- `("run", "final")` record attrs echo config_hash/seed/num_epochs/
  checkpoint_dir + resolved component class names
- finalize() consolidated the store (open_consolidated succeeds)
- CLI layer never constructs SinkSpec literally (seam boundary)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

zarr = pytest.importorskip("zarr")

from tests.cli.test_run_from_config import _make_cfg  # noqa: E402
from xtrax.cli.run import run_from_config  # noqa: E402


def _e2e_cfg(**overrides):
    """Config whose data factory returns a RAW dataset.

    `_make_cfg` defaults to the already-built-DataModule factory, which is what
    the double-wrap identity test needs -- but that combination cannot survive
    REAL batch iteration (run_from_config re-wraps unconditionally, M4/AC3, and
    train_iter() then yields from a DataModule). Every existing CLI test mocks
    Engine.fit_sync, so this surface was never exercised until these tests.
    """
    data = {
        "factory": "tests.cli._run_fixtures:make_dict_dataset",
        "kwargs": {},
        "batch_size": 2,
    }
    return _make_cfg(data=data, **overrides)


def _run_and_load_manifest(monkeypatch, tmp_path, **run_kwargs):
    """Run one real training pass in tmp cwd; return (manifest dict, run_id)."""
    monkeypatch.chdir(tmp_path)
    cfg = _e2e_cfg()
    final_state = run_from_config(cfg, **run_kwargs)
    assert final_state is not None
    runs_root = Path(".xtrax/runs")
    candidates = sorted(p for p in runs_root.iterdir() if p.is_dir())
    assert len(candidates) >= 1
    # With an explicit run_id there is exactly one dir; auto ids reserve
    # exactly one too (collision suffix would show up as a second only if
    # the first makedirs lost a race, which cannot happen in a fresh tmp).
    run_dirs = [p for p in candidates if (p / "manifest.json").exists()]
    assert len(run_dirs) == 1, f"expected exactly one run dir, got {run_dirs}"
    run_dir = run_dirs[0]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return manifest, run_dir


def test_run_persists_provenance_store_with_join_key(monkeypatch, tmp_path) -> None:
    """AC2: store exists at .xtrax/runs/<run_id>/metrics.zarr; root run_id joins manifest."""
    manifest, run_dir = _run_and_load_manifest(monkeypatch, tmp_path)
    store_path = run_dir / "metrics.zarr"
    assert store_path.exists(), f"expected persisted store at {store_path}"
    root = zarr.open_group(str(store_path), mode="r")
    assert root.attrs["run_id"] == manifest["run_id"], (
        "store root run_id must equal the manifest run_id -- the whole point "
        "of routing CLI output through derive_sink_spec"
    )
    # Core sink provenance captured automatically (PR96 contract, inherited free).
    assert "git_sha" in root.attrs
    assert "created_at" in root.attrs


def test_final_record_attrs_match_manifest(monkeypatch, tmp_path) -> None:
    """AC4: ('run','final') attrs echo manifest fields + resolved component names."""
    manifest, run_dir = _run_and_load_manifest(monkeypatch, tmp_path)
    final_group = zarr.open_group(str(run_dir / "metrics.zarr"), mode="r")["run/final"]
    attrs = dict(final_group.attrs)
    assert attrs["config_hash"] == manifest["config_hash"]
    assert attrs["seed"] == manifest["seed"]
    assert attrs["num_epochs"] == manifest["num_epochs"]
    assert attrs["checkpoint_dir"] == manifest["checkpoint_dir"]
    for name, expected in (("model", "Linear"), ("data", "DataModule")):
        # 'data' records the DataModule WRAPPER (what train_iter actually
        # drives), not the raw factory product -- that is the honest value.
        assert attrs[name] == expected, f"resolved {name} class name must be recorded exactly"
    for name in ("optimizer", "loss"):
        assert isinstance(attrs.get(name), str) and attrs[name], (
            f"resolved {name} must be recorded as a non-empty string"
        )
    # Per-key provenance pointer stamped by ZarrStagingSink.drain().
    assert final_group.attrs["run_id"] == manifest["run_id"]
    assert "git_sha" in final_group.attrs


def test_explicit_run_id_flows_to_store(monkeypatch, tmp_path) -> None:
    """AC1/AC3: caller-supplied run_id is the single id across manifest + store."""
    monkeypatch.chdir(tmp_path)
    cfg = _e2e_cfg()
    # Outside any git repo the sink must degrade HONESTLY: warn + record
    # git_sha='unknown' rather than silently skipping provenance.
    with pytest.warns(UserWarning, match="could not determine git state"):
        run_from_config(cfg, run_id="explicit-test-id")
    manifest = json.loads(Path(".xtrax/runs/explicit-test-id/manifest.json").read_text())
    assert manifest["run_id"] == "explicit-test-id"
    root = zarr.open_group(".xtrax/runs/explicit-test-id/metrics.zarr", mode="r")
    assert root.attrs["run_id"] == "explicit-test-id"


def test_store_is_finalized_after_run(monkeypatch, tmp_path) -> None:
    """AC5 lifecycle proof: the persisted store opens CONSOLIDATED post-run.

    zarr v3 evidence: zarr.open_consolidated succeeds only after
    finalize()'s consolidate_metadata call, giving an artifact-based,
    refactor-tolerant observable (no call-spies).
    """
    monkeypatch.chdir(tmp_path)
    cfg = _e2e_cfg()
    with warnings.catch_warnings():
        # zarr warns that consolidation is not yet in the v3 spec; expected.
        warnings.filterwarnings("ignore", message=".*not part in the Zarr format 3 spec.*")
        run_from_config(cfg)
    runs_root = Path(".xtrax/runs")
    run_dir = next(p for p in runs_root.iterdir() if (p / "manifest.json").exists())
    store = str(run_dir / "metrics.zarr")
    consolidated = zarr.open_consolidated(store, mode="r")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert consolidated.attrs["run_id"] == manifest["run_id"]
    assert consolidated["run/final"].attrs["config_hash"] == manifest["config_hash"]


def test_crash_mid_fit_leaves_provenance_tombstone(monkeypatch, tmp_path) -> None:
    """AC5 crash window: sink created pre-fit; mid-fit failure leaves a
    root-provenance tombstone store AND propagates the training error."""
    monkeypatch.chdir(tmp_path)
    cfg = _e2e_cfg()

    with patch("xtrax.cli.run.Engine") as mock_engine_cls:
        instance = MagicMock()
        mock_engine_cls.return_value = instance
        instance.fit_sync.side_effect = RuntimeError("boom mid-fit")
        with pytest.raises(RuntimeError, match="boom mid-fit"):
            run_from_config(cfg)

    run_dirs = [p for p in Path(".xtrax/runs").iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    root = zarr.open_group(str(run_dirs[0] / "metrics.zarr"), mode="r")
    assert root.attrs["run_id"], "tombstone must carry the run's provenance record"
    assert "git_sha" in root.attrs
    # No final record staged: the crash happened before any stage call.
    # Compare against TOP-LEVEL group names ('run'), not joined paths --
    # 'run/final' is a nested path and would make this assertion vacuous.
    assert "run" not in set(_group_keys(root)), (
        "crashed run must not carry a ('run','final') record"
    )


def _group_keys(group):
    """Best-effort child-group listing compatible with zarr v3 Group API."""
    try:
        return list(group.group_keys())
    except AttributeError:  # pragma: no cover - older zarr
        return [k for k, v in group.items() if hasattr(v, "groups")]


def test_real_fit_saves_checkpoints_with_relative_cli_dir(monkeypatch, tmp_path) -> None:
    """Regression proof for the orbax boundary fix: the CLI's RELATIVE
    checkpoint dir (.xtrax/runs/<id>/checkpoints/) must survive a real
    epoch-end save (orbax rejects relative paths; get_checkpoint_manager now
    resolves). No stubs: full fit including checkpoint persistence."""
    monkeypatch.chdir(tmp_path)
    cfg = _e2e_cfg()
    run_from_config(cfg)
    checkpoints = list(Path(".xtrax/runs").glob("*/checkpoints/*"))
    assert checkpoints, "epoch-end checkpoint must be persisted by the real fit"


def test_cli_layer_never_constructs_sink_spec_literally() -> None:
    """AC1 seam boundary (source-audit gate): all CLI sink construction goes
    through derive_sink_spec/make_sink -- no literal SinkSpec( in cli/.

    Mirrors the repo's audit-gate pattern (cf. test_no_future_annotations).
    """
    cli_dir = Path(__file__).resolve().parents[2] / "src" / "xtrax" / "cli"
    offenders = [
        p.name
        for p in sorted(cli_dir.glob("*.py"))
        if "SinkSpec(" in p.read_text(encoding="utf-8")
    ]
    assert offenders == [], (
        f"literal SinkSpec( construction found in cli layer: {offenders} -- "
        "route through derive_sink_spec so precedence stays single-sourced"
    )
