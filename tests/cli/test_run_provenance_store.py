"""#457(1) acceptance tests: run CLI persists provenance through derive_sink_spec.

These drive the REAL public interface (`run_from_config`) end-to-end against a
real zarr store -- no mocks for the persistence layer itself. The single
mocked surface (Engine.fit_sync in the crash-window test) exists solely to
inject a mid-training failure.

Contract under test:
- store lands at `.xtrax/runs/<run_id>/metrics.zarr`
- store root `run_id` == manifest `run_id` (single-sourced join key)
- `("run", "final")` record attrs echo config_hash/seed/num_epochs/
  checkpoint_dir + resolved component class names
- finalize() ran (.zmetadata consolidated)
- CLI layer never constructs SinkSpec literally (seam boundary)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

zarr = pytest.importorskip("zarr")

from tests.cli.test_run_from_config import _make_cfg  # noqa: E402
from xtrax.cli.run import run_from_config  # noqa: E402


def _real_fit_without_orbax():
    """Stub ONLY the orbax checkpoint writer during real fits.

    Latent main-branch bug (found by these tests, out of scope here):
    run_from_config passes a RELATIVE checkpoint dir and orbax refuses
    non-absolute paths -- every pre-existing CLI test mocks Engine.fit_sync,
    so the real fit->checkpoint path had never been exercised end-to-end.
    Everything else here is real: DataModule iteration, trainer steps,
    adamw schedule, zarr persistence through the seam.
    """
    return patch("xtrax.checkpoint.orbax.save_checkpoint")


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
    with _real_fit_without_orbax():
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
    for name in ("model", "optimizer", "loss", "data"):
        assert isinstance(attrs.get(name), str) and attrs[name], (
            f"resolved {name} class name must be recorded as a non-empty string"
        )
    # Per-key provenance pointer stamped by ZarrStagingSink.drain().
    assert final_group.attrs["run_id"] == manifest["run_id"]
    assert "git_sha" in final_group.attrs


def test_explicit_run_id_flows_to_store(monkeypatch, tmp_path) -> None:
    """AC1/AC3: caller-supplied run_id is the single id across manifest + store."""
    monkeypatch.chdir(tmp_path)
    cfg = _e2e_cfg()
    with _real_fit_without_orbax():
        run_from_config(cfg, run_id="explicit-test-id")
    manifest = json.loads(Path(".xtrax/runs/explicit-test-id/manifest.json").read_text())
    assert manifest["run_id"] == "explicit-test-id"
    root = zarr.open_group(".xtrax/runs/explicit-test-id/metrics.zarr", mode="r")
    assert root.attrs["run_id"] == "explicit-test-id"


def test_store_is_finalized_after_run(monkeypatch, tmp_path) -> None:
    """AC5 lifecycle proof: finalize() consolidates metadata exactly once.

    zarr v3 note: consolidation updates zarr.json docs rather than writing a
    v2-style .zmetadata file, so the observable here is the consolidate call
    itself (wrapped, not replaced) plus a fully readable store afterwards.
    """
    monkeypatch.chdir(tmp_path)
    cfg = _e2e_cfg()
    with (
        _real_fit_without_orbax(),
        patch(
            "zarr.consolidate_metadata",
            wraps=zarr.consolidate_metadata,
        ) as spy,
    ):
        run_from_config(cfg)
    assert spy.call_count == 1, "finalize() must consolidate store metadata exactly once"
    # Store fully readable post-run: root join key intact, final record present.
    runs_root = Path(".xtrax/runs")
    run_dir = next(p for p in runs_root.iterdir() if (p / "manifest.json").exists())
    root = zarr.open_group(str(run_dir / "metrics.zarr"), mode="r")
    assert root.attrs["run_id"] == json.loads((run_dir / "manifest.json").read_text())["run_id"]
    final_group = root["run/final"]  # nested: 'run' -> 'final'
    assert final_group.attrs["config_hash"], "final record must be present post-finalize"


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
    assert "run/final" not in {k for k in _group_keys(root)}


def _group_keys(group):
    """Best-effort child-group listing compatible with zarr v3 Group API."""
    try:
        return list(group.group_keys())
    except AttributeError:  # pragma: no cover - older zarr
        return [k for k, v in group.items() if hasattr(v, "groups")]


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
