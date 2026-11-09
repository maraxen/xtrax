"""Debt #3308: a manifest row's ``output_locator`` is bound to its ``manifest_row_hash``.

Planner rows keep the default locator ``<campaign_id>/<row_hash>.zarr`` and their hash is
unchanged; a row with any other locator hashes the locator too. ``input_hash`` never
covers the locator, so aminx parity (GATE-11) is unaffected.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.campaign.unit_fns import ARRAY_UNIT_FN
from xtrax.campaign.executors.local import _resolve_output_path
from xtrax.campaign.manifest import (
    CampaignManifest,
    ManifestIntegrityError,
    ManifestRow,
    build_manifest_row,
    plan_campaign,
)
from xtrax.campaign.spec import ArmSpec, CampaignSpec, ParamGrid
from xtrax.cli.campaign_verb import CampaignArgs, run_campaign
from xtrax.cli.errors import CLIError

#: Planner-row hashes computed on main ad643f5, before #3308 (the back-compat guard).
GOLDEN_PLANNER_ROW_HASHES = [
    "2bfd54756d1aa594769dde80f89c27d6afa8568e730f1c63af30379d8807b0e7",
    "d373bf8b1c540e6b5acc5ce112a49dcc10ea5677bda15c4f138e1102af1c1d4a",
]


def _planned_manifest_dict(tmp_path: Path) -> dict:
    spec = CampaignSpec(
        campaign_id="integrity_campaign",
        target_inputs=["target.pdb"],
        arms=[
            ArmSpec(
                arm_id="arm_1",
                backend="b1",
                checkpoint_id="c1",
                parameters={"alpha": 0.1, "neg_zero": -0.0, "tags": ["x", "y"]},
            )
        ],
        grid=ParamGrid(axes={"temp": [0.1, 0.5]}),
        samples_per_target=100,
        chunk_size=100,
        seed=7,
        parameters={"scale": 2.5},
    )
    path = tmp_path / "manifest.json"
    plan_campaign(spec, output_manifest_path=path)
    return json.loads(path.read_text(encoding="utf-8"))


def _custom_row(locator: str, **overrides: object) -> ManifestRow:
    kwargs: dict = {
        "campaign_id": "c",
        "job_id": "j",
        "chunk_id": 0,
        "sample_start": 0,
        "sample_count": 10,
        "target_input": "t.pdb",
        "arm_id": "a",
        "arm_backend": "b",
        "checkpoint_id": "ck",
        "parameters": {"p": 1},
        "output_locator": locator,
    }
    kwargs.update(overrides)
    return build_manifest_row(**kwargs)


# --- manifest-level -----------------------------------------------------------------


def test_planner_row_hashes_unchanged(tmp_path: Path) -> None:
    data = _planned_manifest_dict(tmp_path)
    assert [r["manifest_row_hash"] for r in data["rows"]] == GOLDEN_PLANNER_ROW_HASHES
    for r in data["rows"]:
        assert r["output_locator"] == f"integrity_campaign/{r['manifest_row_hash']}.zarr"
    CampaignManifest.from_dict(data)


def test_redirecting_a_row_to_another_rows_store_is_refused(tmp_path: Path) -> None:
    data = _planned_manifest_dict(tmp_path)
    data["rows"][0]["output_locator"] = data["rows"][1]["output_locator"]
    with pytest.raises(ManifestIntegrityError, match="output_locator"):
        CampaignManifest.from_dict(data)


@pytest.mark.parametrize("locator", ["/elsewhere/store.zarr", "../escape.zarr", "x.zarr"])
def test_editing_a_default_locator_is_refused(tmp_path: Path, locator: str) -> None:
    data = _planned_manifest_dict(tmp_path)
    data["rows"][1]["output_locator"] = locator
    with pytest.raises(ManifestIntegrityError, match="output_locator"):
        CampaignManifest.from_dict(data)


def test_custom_locator_round_trips_and_is_covered() -> None:
    row = _custom_row("u/x.zarr")
    loaded = ManifestRow.from_dict(row.to_dict())
    assert loaded.output_locator == "u/x.zarr"
    assert loaded.manifest_row_hash == row.manifest_row_hash

    edited = row.to_dict()
    edited["output_locator"] = "u/y.zarr"
    with pytest.raises(ManifestIntegrityError, match="output_locator"):
        ManifestRow.from_dict(edited)


def test_custom_locator_changes_row_hash_not_input_hash() -> None:
    x, y = _custom_row("u/x.zarr"), _custom_row("u/y.zarr")
    default = build_manifest_row(**{**_custom_kwargs(), "output_locator": None})
    assert len({x.manifest_row_hash, y.manifest_row_hash, default.manifest_row_hash}) == 3
    assert x.input_hash == y.input_hash == default.input_hash


def test_passing_the_default_locator_explicitly_equals_omitting_it() -> None:
    default = build_manifest_row(**{**_custom_kwargs(), "output_locator": None})
    explicit = _custom_row(default.output_locator)
    assert explicit.manifest_row_hash == default.manifest_row_hash


def test_custom_row_cannot_be_relabelled_as_a_default_row() -> None:
    """Rewriting a custom row's locator to its own default form must not verify."""
    row = _custom_row("u/x.zarr")
    edited = row.to_dict()
    edited["output_locator"] = f"c/{row.manifest_row_hash}.zarr"
    with pytest.raises(ManifestIntegrityError, match="output_locator"):
        ManifestRow.from_dict(edited)


def _custom_kwargs() -> dict:
    return {
        "campaign_id": "c",
        "job_id": "j",
        "chunk_id": 0,
        "sample_start": 0,
        "sample_count": 10,
        "target_input": "t.pdb",
        "arm_id": "a",
        "arm_backend": "b",
        "checkpoint_id": "ck",
        "parameters": {"p": 1},
    }


# --- path containment ---------------------------------------------------------------


def test_relative_locator_escaping_output_dir_is_refused(tmp_path: Path) -> None:
    row = _custom_row("../escape.zarr")
    with pytest.raises(ValueError, match="escapes"):
        _resolve_output_path(tmp_path / "out", row)


def test_absolute_locator_is_allowed_when_hashed(tmp_path: Path) -> None:
    target = tmp_path / "abs" / "store.zarr"
    row = _custom_row(str(target))
    assert _resolve_output_path(tmp_path / "out", row) == target


# --- executor-level: the destructive redirect ---------------------------------------


def _tree_snapshot(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): (
            hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "<dir>"
        )
        for p in sorted(root.rglob("*"))
    }


def test_tampered_locator_cannot_quarantine_a_finished_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    spec_file = tmp_path / "spec.toml"
    CampaignSpec(
        campaign_id="redirect_camp",
        target_inputs=["targets/p1.pdb"],
        arms=[ArmSpec(arm_id="arm", backend="b", checkpoint_id="v1")],
        grid=ParamGrid(axes={"steps": [10, 20]}),
        output_dir=str(out_dir),
    ).to_toml(spec_file)
    manifest = tmp_path / "manifest.json"
    run_campaign(CampaignArgs(action="plan", spec_path=str(spec_file), output=str(manifest)))

    run_args = CampaignArgs(
        action="run",
        fn=ARRAY_UNIT_FN,
        manifest=str(manifest),
        output_dir=str(out_dir),
        executor="local-seq",
    )
    assert run_campaign(run_args).completed_units == 2
    before = _tree_snapshot(out_dir)

    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["rows"][0]["output_locator"] = data["rows"][1]["output_locator"]
    manifest.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(CLIError, match="output_locator") as excinfo:
        run_campaign(run_args)
    assert isinstance(excinfo.value.__cause__, ManifestIntegrityError)
    assert _tree_snapshot(out_dir) == before
