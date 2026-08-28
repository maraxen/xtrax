"""End-to-end: CLI verbs produce ledger rows that join to their other artifacts.

A ledger row is only useful if it can be tied to the run's other outputs. These
tests pin the join key (run_id, shared with manifest.json, the checkpoint dir,
and the zarr provenance record) and the lineage edge that `resume` records.

Fixture factories are module-level in tests.cli._run_fixtures because load_fn
resolves "module:attr" paths and cannot see pytest closures.
"""

import pytest

from xtrax.cli.config import TrainConfig
from xtrax.cli.export import ExportArgs, run_export
from xtrax.cli.run import run_from_config
from xtrax.telemetry.ledger import find_run, iter_rows
from xtrax.telemetry.record import KIND_EXPORT, KIND_TRAIN, STATUS_COMPLETE

zarr = pytest.importorskip("zarr")


def exported_fn(x):
    """Module-level so load_fn's "module:attr" resolution can find it."""
    return x * 2


def _make_cfg(**overrides):
    defaults = dict(
        schema_version=1,
        model={"path": "tests.cli._run_fixtures:make_model", "kwargs": {}},
        optimizer={
            "path": "xtrax.training.optim:adamw_with_schedule",
            "kwargs": {"peak_lr": 1e-3, "warmup_steps": 1, "total_steps": 30},
        },
        loss={"path": "tests.cli._run_fixtures:make_loss", "kwargs": {}},
        # A plain dataset, not a DataModule: run_from_config always wraps.
        data={
            "factory": "tests.cli._run_fixtures:make_dict_dataset",
            "kwargs": {},
            "batch_size": 2,
        },
        seed=0,
        num_epochs=1,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


@pytest.fixture
def _in_tmp(tmp_path, monkeypatch):
    """Run inside a temp cwd so .xtrax/ledger is created and inspected there."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("XTRAX_LEDGER_ROOT", raising=False)
    monkeypatch.delenv("XTRAX_TELEMETRY_OPTOUT", raising=False)
    return tmp_path


def test_run_from_config_writes_a_ledger_row(_in_tmp):
    run_from_config(_make_cfg())
    rows = list(iter_rows())
    assert len(rows) == 1
    assert rows[0].kind == KIND_TRAIN
    assert rows[0].telemetry_status == STATUS_COMPLETE


def test_the_ledger_run_id_joins_to_the_manifest(_in_tmp):
    """The join key. Without it the row cannot be tied to anything else."""
    import json

    run_from_config(_make_cfg())
    row = next(iter(iter_rows()))
    manifest_path = _in_tmp / ".xtrax" / "runs" / row.run_id / "manifest.json"
    assert manifest_path.is_file(), f"no manifest for ledger run_id {row.run_id}"
    assert json.loads(manifest_path.read_text())["run_id"] == row.run_id


def test_the_ledger_run_id_joins_to_the_checkpoint_dir(_in_tmp):
    run_from_config(_make_cfg())
    row = next(iter(iter_rows()))
    assert (_in_tmp / ".xtrax" / "runs" / row.run_id / "checkpoints").is_dir()


def test_context_carries_the_config_identity(_in_tmp):
    run_from_config(_make_cfg(seed=7))
    ctx = next(iter(iter_rows())).context
    assert ctx["seed"] == "7"
    assert ctx["config_hash"]
    assert ctx["num_epochs"] == "1"
    assert ctx["manifest"].endswith("manifest.json")


def test_run_captures_the_executed_ir(_in_tmp):
    run_from_config(_make_cfg())
    row = next(iter(iter_rows()))
    assert {ref.kind for ref in row.ir} >= {"jaxpr", "stablehlo"}
    assert all(ref.bytes > 0 for ref in row.ir), [r.reason for r in row.ir]


def test_find_run_locates_the_row_by_its_cli_run_id(_in_tmp):
    run_from_config(_make_cfg())
    run_id = next(iter(iter_rows())).run_id
    assert find_run(run_id) is not None


# --- resume lineage ---------------------------------------------------------


def test_resume_records_its_parent_as_a_lineage_edge(_in_tmp):
    """resume previously built no sink and recorded no provenance at all."""
    from xtrax.cli.resume_verb import ResumeArgs, run_resume

    run_from_config(_make_cfg())
    parent_id = next(iter(iter_rows())).run_id

    run_resume(ResumeArgs(run_id=parent_id, epochs=1))

    rows = list(iter_rows())
    children = [r for r in rows if r.derived_from is not None]
    assert len(children) == 1, f"expected one resumed run, got {[r.run_id for r in rows]}"
    assert children[0].derived_from == parent_id
    assert children[0].run_id != parent_id


def test_resume_rows_form_a_chain(_in_tmp):
    from xtrax.cli.resume_verb import ResumeArgs, run_resume

    run_from_config(_make_cfg())
    first = next(iter(iter_rows())).run_id
    run_resume(ResumeArgs(run_id=first, epochs=1))
    second = next(r for r in iter_rows() if r.derived_from == first).run_id
    run_resume(ResumeArgs(run_id=second, epochs=1))

    edges = {r.run_id: r.derived_from for r in iter_rows()}
    assert edges[first] is None
    assert edges[second] == first
    third = next(rid for rid, parent in edges.items() if parent == second)
    assert third not in (first, second)


# --- export -----------------------------------------------------------------


def test_export_records_a_row_with_its_stablehlo(_in_tmp):
    """export produced exactly the audit artifact and discarded its provenance."""
    run_export(
        ExportArgs(
            fn="tests.telemetry.test_cli_integration:exported_fn",
            shapes="x=(4,)f32",
            out=str(_in_tmp / "out.mlir"),
        )
    )
    rows = [r for r in iter_rows() if r.kind == KIND_EXPORT]
    assert len(rows) == 1
    assert rows[0].context["fn"] == "tests.telemetry.test_cli_integration:exported_fn"
    assert rows[0].provenance.git_sha
    assert {ref.kind for ref in rows[0].ir} >= {"jaxpr", "stablehlo"}


def test_export_is_fail_open_unlike_fit(_in_tmp, monkeypatch, capsys):
    """An export must still succeed where a ledger cannot be written.

    Deliberately asymmetric with fit: an unrecorded *execution* is
    unrecoverable, whereas an export can simply be run again.
    """
    blocked = _in_tmp / "blocked"
    blocked.mkdir()
    blocked.chmod(0o500)
    monkeypatch.setenv("XTRAX_LEDGER_ROOT", str(blocked / "ledger"))
    try:
        run_export(
            ExportArgs(
                fn="tests.telemetry.test_cli_integration:exported_fn",
                shapes="x=(4,)f32",
                out=str(_in_tmp / "out.mlir"),
            )
        )
    finally:
        blocked.chmod(0o700)
    assert (_in_tmp / "out.mlir").is_file()
    assert "could not record export telemetry" in capsys.readouterr().err
