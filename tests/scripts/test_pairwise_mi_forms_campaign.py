"""CPU tests for ``scripts/benchmarks/pairwise_mi_forms_campaign.py`` (backlog #6099).

The driver runs real cell subprocesses at tiny shapes (Pallas not needed: forms A and B in
CPU mode), so each test that spawns cells costs a few seconds. The aggregate arithmetic
is tested separately on fabricated cell records, where a corrupted cell must change the
fields it is supposed to change.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.benchmarks import pairwise_mi_forms_campaign as camp

TINY = [
    "--forms", "A,B", "--shapes", "custom", "--Ks", "6", "--entropy-dtypes", "float64",
    "--n-used-map", "6:5", "--n-res", "10", "--n-frames", "150", "--n-pairs", "20",
    "--n-sample", "12", "--reps", "2", "--warmup", "1", "--pair-chunk", "8",
    "--pair-tile", "4", "--frame-tile", "64", "--a-batch-size", "8",
]  # fmt: skip


def _run(out_dir: Path, *extra: str) -> dict[str, Any]:
    assert camp.main(["--out-dir", str(out_dir), *TINY, *extra]) == 0
    return json.loads((out_dir / "campaign_result.json").read_text())


def test_runs_reuses_and_emits_all_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bth = tmp_path / "bth.json"
    monkeypatch.setenv("BTH_RESULTS_PATH", str(bth))
    first = _run(tmp_path / "out")
    assert set(camp.FIELDS) <= set(first)
    assert list(first)[: len(camp.FIELDS)] == camp.FIELDS
    assert json.loads(bth.read_text()) == first
    assert (first["n_cells_planned"], first["n_cells_ok"]) == (2, 2)
    assert first["n_cells_error"] == 0 and first["n_cells_timeout"] == 0
    assert first["a_b_correct"] is True and first["max_recompiles"] == 0
    assert first["a_b_full_ok"] is False  # shape custom: no planned full-shape cell
    assert first["jax_version"] and first["device_kind"] != "unknown"

    calls = []
    real = camp.run_cell_subprocess
    monkeypatch.setattr(camp, "run_cell_subprocess", lambda *a: calls.append(a) or real(*a))
    second = _run(tmp_path / "out")
    assert calls == [] and second == first  # stamps verified: nothing re-run
    reuse = (tmp_path / "out" / "reuse_log.jsonl").read_text().splitlines()
    assert len(reuse) == 2


def test_cell_timeout_is_counted_not_raised_and_rerun_completes(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    timed = _run(out_dir, "--cell-timeout-s", "0.2")  # a Python+jax start takes longer
    assert timed["n_cells_timeout"] == 2 and timed["n_cells_ok"] == 0
    assert timed["a_b_correct"] is False  # no ok A/B cell: not vacuously true
    assert timed["max_recompiles"] == -1 and timed["device_kind"] == "unknown"
    assert len(list(out_dir.glob("*.timeout.json"))) == 2

    done = _run(out_dir, "--cell-timeout-s", "600")
    assert done["n_cells_ok"] == 2 and done["n_cells_timeout"] == 0
    assert list(out_dir.glob("*.timeout.json")) == []


def test_negative_control_corrupted_cell_makes_a_b_correct_false(tmp_path: Path) -> None:
    clean = _run(tmp_path / "clean")
    bad = _run(tmp_path / "bad", "--corrupt", "shift_state")
    assert clean["a_b_correct"] is True
    assert bad["n_cells_ok"] == 2 and bad["a_b_correct"] is False


def _result(form: str, edt: str, k: int, shape: str, **kw: Any) -> dict[str, Any]:
    base = {
        "form": form, "entropy_dtype": edt, "K": k, "shape": shape, "error": "",
        "counts_exact": True, "rowsums_ok": True, "mi_all_zero": False, "mi_nonfinite": 0,
        "mi_max_abs_diff": 1e-15, "step_median_s": 1.0, "recompiles": 0,
        "jax_version": "0.11.1", "device_kind": "NVIDIA H200",
    }  # fmt: skip
    return {**base, **kw}


def _rec(form: str, edt: str, k: int, shape: str, status: str = "ok", **kw: Any) -> Any:
    result = _result(form, edt, k, shape, **kw) if status != "timeout" else None
    return camp.CellRec(f"{form}-{edt}-{k}-{shape}", form, edt, shape, k, status, result)


def _grid(steps: dict[str, float], **over: dict[str, Any]) -> list[Any]:
    recs = []
    for form in ("A", "B", "C_triton", "D"):
        for k in camp.KS:
            for shape in ("full", "tree", "boundary"):
                for edt in ("float64", "float32"):
                    kw = {"step_median_s": steps[form] * (1 + 0.1 * (k == 129))}
                    kw.update(over.get(f"{form}/{k}/{shape}/{edt}", {}))
                    recs.append(_rec(form, edt, k, shape, **kw))
    return recs


STEPS = {"A": 4.0, "B": 2.0, "C_triton": 0.5, "D": 1.0}


def test_aggregate_all_good() -> None:
    out = camp.aggregate(_grid(STEPS))
    assert out["n_cells_planned"] == 48 and out["n_cells_ok"] == 48
    assert out["a_b_correct"] and out["a_b_full_ok"] and out["boundary_b_ok"]
    assert out["step_b_k11"] == 2.0 and out["step_b_k129"] == pytest.approx(2.2)
    assert out["fastest_k11"] == "C_triton" and out["kernel_ok_k11"] is True
    assert out["best_kernel_step_k11"] == 0.5 and out["ratio_b_over_kernel_k11"] == 4.0
    assert out["ratio_a_over_kernel_k11"] == 8.0
    assert all(out[f"f32_entropy_ok_{f}"] for f in camp.FORM_KEYS)
    assert out["max_recompiles"] == 0 and list(out)[: len(camp.FIELDS)] == camp.FIELDS


def test_aggregate_incorrect_and_failed_cells_are_excluded() -> None:
    over = {
        "C_triton/11/full/float64": {"counts_exact": False},  # fastest but wrong
        "D/129/full/float64": {"mi_all_zero": True},
        "B/11/boundary/float64": {"rowsums_ok": False},
        "A/11/full/float32": {"mi_max_abs_diff": 5e-6},
        "B/129/tree/float64": {"recompiles": 3},
    }
    recs = _grid(STEPS, **over)
    recs[[r.cell_id for r in recs].index("D-float64-11-full")] = _rec(
        "D", "float64", 11, "full", status="timeout"
    )
    recs[[r.cell_id for r in recs].index("C_triton-float64-129-full")] = _rec(
        "C_triton", "float64", 129, "full", status="error", error="boom"
    )
    out = camp.aggregate(recs)
    assert out["step_c_triton_k11"] == -1.0 and out["correct_c_triton_k11"] is False
    assert out["fastest_k11"] == "B"  # wrong C_triton and timed-out D are excluded
    assert out["best_kernel_step_k11"] == -1.0 and out["kernel_ok_k11"] is False
    assert out["ratio_b_over_kernel_k11"] == -1.0 and out["ratio_a_over_kernel_k11"] == -1.0
    assert out["step_d_k129"] == -1.0 and out["correct_d_k129"] is False
    assert out["step_c_triton_k129"] == -1.0 and out["best_kernel_step_k129"] == -1.0
    assert out["boundary_b_ok"] is False and out["a_b_correct"] is False
    assert out["a_b_full_ok"] is True
    assert out["f32_entropy_ok_a"] is False and out["f32_entropy_ok_b"] is True
    assert out["n_cells_timeout"] == 1 and out["n_cells_error"] == 1
    assert out["max_recompiles"] == 3


def test_aggregate_float32_diff_alone_does_not_fail_a_b_correct() -> None:
    """f32 cells are exempt from the 1e-9 diff gate (reported by f32_entropy_ok_*)."""
    out = camp.aggregate(_grid(STEPS, **{"A/11/full/float32": {"mi_max_abs_diff": 5e-6}}))
    assert out["a_b_correct"] is True and out["f32_entropy_ok_a"] is False


def test_aggregate_a_b_full_ok_requires_every_full_float64_cell() -> None:
    recs = _grid(STEPS)
    recs[[r.cell_id for r in recs].index("A-float64-129-full")] = _rec(
        "A", "float64", 129, "full", status="timeout"
    )
    assert camp.aggregate(recs)["a_b_full_ok"] is False
    assert camp.aggregate([])["a_b_full_ok"] is False


def test_dry_run_lists_the_requested_cells(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert camp.main(["--out-dir", str(tmp_path), "--dry-run"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4 * 2 * 3 * 2
    k129 = [ln for ln in lines if "--K 129" in ln]
    assert len(k129) == 24 and all("--n-used 23" in ln for ln in k129)
    assert all("--pair-chunk 2048" in ln for ln in lines if "--form B" in ln)


SIDECAR = Path(camp.__file__).with_suffix(".bth.toml")
D_BROKEN = {f"D/{k}/full/{e}": {"counts_exact": False, "rowsums_ok": False}
            for k in camp.KS for e in ("float64", "float32")}  # fmt: skip


def _matching_outcomes(out: dict[str, Any], tmp_path: Path) -> list[str]:
    """Evaluate every non-residual sidecar condition on ``out`` as bathos does (DuckDB)."""
    duckdb = pytest.importorskip("duckdb")
    import tomllib

    outcomes = tomllib.loads(SIDECAR.read_text())["outcomes"]
    path = tmp_path / "result.json"
    path.write_text(json.dumps(out))
    con = duckdb.connect()
    hits = []
    for name, spec in outcomes.items():
        if spec.get("is_residual"):
            continue
        sql = f"SELECT ({spec['condition']}) FROM read_json_auto('{path}')"
        if con.execute(sql).fetchone()[0]:
            hits.append(name)
    return hits


@pytest.mark.parametrize(
    ("steps", "over", "want"),
    [
        # Smoke-run shape (RTX PRO 6000, K=11): kernel 3.8x over B but only 1.3x over A.
        ({"A": 0.069, "B": 0.199, "C_triton": 0.052, "D": 0.039}, D_BROKEN, ["keep_a"]),
        ({"A": 4.0, "B": 2.0, "C_triton": 0.5, "D": 1.0}, {}, ["ship_kernel"]),
        ({"A": 4.0, "B": 2.0, "C_triton": 1.6, "D": 1.0}, D_BROKEN, ["ship_b"]),
        ({"A": 1.0, "B": 2.0, "C_triton": 0.5, "D": 1.0}, D_BROKEN, ["ship_kernel"]),
        ({"A": 1.0, "B": 2.0, "C_triton": 0.9, "D": 1.0}, D_BROKEN, ["keep_a"]),
        # Invalid: a recompile anywhere leaves only the residual.
        ({"A": 4.0, "B": 2.0, "C_triton": 0.5, "D": 1.0},
         {"A/11/tree/float64": {"recompiles": 1}}, []),
    ],
)  # fmt: skip
def test_sidecar_outcomes_are_exclusive_and_route_as_designed(
    steps: dict[str, float], over: dict[str, Any], want: list[str], tmp_path: Path
) -> None:
    out = camp.aggregate(_grid(steps, **over))
    assert _matching_outcomes(out, tmp_path) == want
