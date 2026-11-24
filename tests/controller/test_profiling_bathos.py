"""Tests for optional controller.profiling_bathos bridge (tier4_controller)."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from controller.profiling_bathos import resolve_bathos_provenance


def test_resolve_bathos_provenance_raises_when_bathos_absent(monkeypatch):
    orig_import = importlib.__import__

    def _no_bathos(name, *args, **kwargs):
        if name == "bathos" or name.startswith("bathos."):
            raise ImportError("no bathos")
        return orig_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "bathos", raising=False)
    monkeypatch.delitem(sys.modules, "bathos.config", raising=False)
    monkeypatch.delitem(sys.modules, "bathos.query", raising=False)
    monkeypatch.setattr("builtins.__import__", _no_bathos)

    with pytest.raises(ImportError, match=r"Install xtrax\[controller\]"):
        resolve_bathos_provenance("run-abc")


def test_resolve_bathos_provenance_mocked(monkeypatch, tmp_path):
    queries: list[tuple[str, Path]] = []

    def _fake_get_run(run_id: str, catalog_dir: Path):
        queries.append((run_id, catalog_dir))
        if run_id == "missing":
            return None
        return SimpleNamespace(
            id=run_id,
            campaign_id="camp-1",
            git_hash="a" * 40,
            sidecar_sha256="f" * 64,
            status="finished",
        )

    fake_cfg = SimpleNamespace(default_catalog_dir=lambda: str(tmp_path / "default_bth"))
    fake_query = SimpleNamespace(get_run=_fake_get_run)
    fake_bathos = SimpleNamespace(config=fake_cfg, query=fake_query)

    monkeypatch.setitem(sys.modules, "bathos", fake_bathos)
    monkeypatch.setitem(sys.modules, "bathos.config", fake_cfg)
    monkeypatch.setitem(sys.modules, "bathos.query", fake_query)

    res_default = resolve_bathos_provenance("run-1")
    assert res_default["run_id"] == "run-1"
    assert res_default["campaign_id"] == "camp-1"
    assert res_default["git_sha"] == "a" * 40
    assert res_default["sidecar_sha256"] == "f" * 64
    assert res_default["status"] == "finished"
    assert res_default["catalog_dir"] == str(tmp_path / "default_bth")

    monkeypatch.setenv("BTH_CATALOG_DIR", str(tmp_path / "env_bth"))
    res_env = resolve_bathos_provenance("run-2")
    assert res_env["catalog_dir"] == str(tmp_path / "env_bth")

    assert resolve_bathos_provenance("missing", catalog_dir=str(tmp_path / "explicit")) == {}
    assert queries[-1] == ("missing", tmp_path / "explicit")


def test_resolve_bathos_provenance_real_run_schema(monkeypatch, tmp_path):
    bathos_schema = pytest.importorskip("bathos.schema")
    bathos_query = pytest.importorskip("bathos.query")
    real_run = bathos_schema.Run(
        project_slug="prolix",
        command="xtrax profile",
        argv=["xtrax", "profile"],
        git_hash="b" * 40,
        git_branch="main",
        git_dirty=False,
        id="run-real-1",
        campaign_id="camp-real",
        sidecar_sha256="c" * 64,
        status="completed",
    )
    monkeypatch.setattr(bathos_query, "get_run", lambda _rid, catalog_dir: real_run)
    res = resolve_bathos_provenance("run-real-1", catalog_dir=str(tmp_path))
    assert res == {
        "run_id": "run-real-1",
        "campaign_id": "camp-real",
        "git_sha": "b" * 40,
        "sidecar_sha256": "c" * 64,
        "status": "completed",
        "catalog_dir": str(tmp_path),
    }


def test_resolve_bathos_provenance_live_integration(tmp_path):
    pytest.importorskip("bathos.query")
    result = resolve_bathos_provenance("nonexistent-run-id", catalog_dir=str(tmp_path))
    assert isinstance(result, dict)
