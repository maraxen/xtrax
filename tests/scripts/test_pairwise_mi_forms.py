"""CPU tests for ``scripts/benchmarks/pairwise_mi_forms.py`` (backlog #6099).

Every form runs at a tiny shape on CPU (Pallas in interpret mode) through the same
``run_one`` path the GPU job uses. The checks the benchmark reports are only worth
trusting if each one can fail, so the negative controls here corrupt a form on purpose and
require the matching check to turn red.

Not covered here, by design: GPU execution, Triton PTX compilation, real Mosaic GPU
lowering. ``test_triton_forms_lower_for_cuda`` runs the Pallas-to-Triton-IR lowering for an
abstract H200 on this host, which catches shape and unsupported-op errors only.
"""

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.benchmarks import pairwise_mi_forms as pmf

TINY = [
    "--shape", "custom", "--n-res", "10", "--n-frames", "150", "--n-pairs", "20",
    "--K", "6", "--n-used", "5", "--n-sample", "12", "--reps", "2", "--warmup", "1",
    "--pair-chunk", "8", "--pair-tile", "4", "--frame-tile", "64", "--a-batch-size", "8",
    "--interpret",
]  # fmt: skip


def _args(out_dir: Path, *extra: str) -> Any:
    return pmf.build_parser().parse_args([*TINY, "--out-dir", str(out_dir), *extra])


def _run(out_dir: Path, form: str, edtype: str = "float64", *extra: str) -> dict[str, Any]:
    return pmf.run_one(_args(out_dir, "--form", form, "--entropy-dtype", edtype, *extra))


@pytest.mark.parametrize("form", pmf.FORMS)
def test_form_counts_exact_and_mi_matches_f64(form: str, tmp_path: Path) -> None:
    r = _run(tmp_path, form)
    assert r["error"] == ""
    assert r["counts_exact"] is True
    assert r["rowsums_ok"] is True
    assert r["mi_max_abs_diff"] <= 1e-10
    assert r["recompiles"] == 0
    assert r["compile_events_first"] >= 1  # the compile counter saw the first call
    assert r["mi_all_zero"] is False and r["mi_nonfinite"] == 0
    assert r["n_pairs"] == 20 and r["n_sampled"] == 12
    assert (tmp_path / r["mi_npy"]).is_file()


@pytest.mark.parametrize("form", pmf.FORMS)
def test_form_float32_entropy_within_spec_threshold(form: str, tmp_path: Path) -> None:
    r = _run(tmp_path, form, "float32")
    assert r["error"] == ""
    assert r["counts_exact"] is True and r["rowsums_ok"] is True
    assert r["mi_max_abs_diff"] <= 1e-6  # the spec's float32 acceptance threshold


@pytest.mark.parametrize("form", pmf.FORMS)
def test_float32_cell_traces_no_float64(form: str) -> None:
    """The float32 entropy cell must not silently compute in float64 under x64."""
    with jax.enable_x64(True):
        cell = pmf.cell_from_args(_args(Path("."), "--form", form, "--entropy-dtype", "float32"))
        states = pmf.make_states(cell.n_res, cell.n_frames, cell.K, cell.n_used, cell.seed)
        built = pmf.build_form(cell, states)
        pairs = jnp.asarray(pmf.make_pairs(cell.n_res, cell.n_pairs, cell.seed))
        text = str(jax.make_jaxpr(built.mi_fn)(built.states_dev, pairs))
    assert "f64" not in text
    assert "f32" in text


@pytest.mark.parametrize("form", pmf.FORMS)
def test_negative_control_shifted_states_turns_counts_red(form: str, tmp_path: Path) -> None:
    r = _run(tmp_path, form, "float64", "--corrupt", "shift_state")
    assert r["error"] == ""
    assert r["counts_exact"] is False
    # A relabelling preserves MI: only the exact-count check can see this corruption.
    assert r["mi_max_abs_diff"] <= 1e-10


@pytest.mark.parametrize("form", ["A", "B", "C_triton"])
def test_negative_control_extra_count_turns_rowsums_red(form: str, tmp_path: Path) -> None:
    r = _run(tmp_path, form, "float64", "--corrupt", "extra_count")
    assert r["counts_exact"] is False
    assert r["rowsums_ok"] is False


@pytest.mark.parametrize("form", ["A", "B", "D"])
def test_negative_control_scaled_mi_turns_mi_diff_red(form: str, tmp_path: Path) -> None:
    r = _run(tmp_path, form, "float64", "--corrupt", "mi_scale")
    assert r["counts_exact"] is True  # the counts path is untouched
    assert r["mi_max_abs_diff"] > 1e-4


def test_negative_control_padding_would_break_counts() -> None:
    """Frames padded with state 0 instead of -1 would add counts to bin (0, 0)."""
    states = pmf.make_states(4, 100, 6, 5, 0)
    ref, _ = pmf.numpy_reference(states, np.array([[0, 1]]), 6)
    padded = np.pad(states, ((0, 0), (0, 28)), constant_values=0)
    bad, _ = pmf.numpy_reference(padded, np.array([[0, 1]]), 6)
    assert not np.array_equal(ref, bad)
    assert bad.sum() == 128 and ref.sum() == 100


def test_numpy_reference_matches_explicit_loop() -> None:
    states = pmf.make_states(5, 60, 4, 4, 3)
    pairs = np.array([[0, 4], [1, 2]])
    counts, mi = pmf.numpy_reference(states, pairs, 4)
    for n, (i, j) in enumerate(pairs):
        expect = np.zeros((4, 4), dtype=np.int64)
        for f in range(60):
            expect[states[i, f], states[j, f]] += 1
        assert np.array_equal(counts[n], expect)
        pij = expect / 60
        pi, pj = pij.sum(1), pij.sum(0)
        brute = sum(
            pij[a, b] * np.log(pij[a, b] / (pi[a] * pj[b]))
            for a in range(4)
            for b in range(4)
            if pij[a, b] > 0
        )
        assert abs(mi[n] - max(brute, 0.0)) < 1e-12


def test_stamp_reuse_and_invalidation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    real = pmf.measure_cell

    def counting(*a: Any, **k: Any) -> dict[str, Any]:
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(pmf, "measure_cell", counting)
    first = _run(tmp_path, "B")
    second = _run(tmp_path, "B")
    assert len(calls) == 1 and second == first
    reuse = (tmp_path / "reuse_log.jsonl").read_text().splitlines()
    assert len(reuse) == 1 and json.loads(reuse[0])["cell_id"].startswith("B_float64_custom")

    # Negative controls: a tampered artifact, a changed script hash and --force all re-run.
    art = next(tmp_path.glob("B_*.json"))
    if art.name.endswith(".stamp.json"):
        art = next(p for p in tmp_path.glob("B_*.json") if not p.name.endswith("stamp.json"))
    art.write_text(art.read_text() + " ")
    _run(tmp_path, "B")
    assert len(calls) == 2
    monkeypatch.setattr(pmf, "script_sha256", lambda: "0" * 64)
    _run(tmp_path, "B")
    assert len(calls) == 3
    _run(tmp_path, "B", "float64", "--force")
    assert len(calls) == 4


def test_errored_cell_is_recorded_not_raised(tmp_path: Path) -> None:
    r = _run(tmp_path, "C_triton", "float64", "--frame-tile", "48")  # not a power of two
    assert "power of two" in r["error"]
    assert r["counts_exact"] is False and r["step_median_s"] == -1.0
    # Without --interpret a Pallas form on CPU fails loudly instead of silently interpreting.
    args = pmf.build_parser().parse_args(
        [*[a for a in TINY if a != "--interpret"], "--out-dir", str(tmp_path / "x"), "--form", "D"]
    )
    assert pmf.run_one(args)["error"] != ""


def test_heartbeat_is_written_and_fsynced_phases(tmp_path: Path) -> None:
    _run(tmp_path, "A")
    beats = next(tmp_path.glob("*.heartbeat.jsonl")).read_text().splitlines()
    phases = [json.loads(line)["phase"] for line in beats]
    assert phases[:3] == ["data", "build", "compile_start"]
    assert phases.count("rep") == 2 and phases[-1] == "done"


def test_compile_counter_sees_recompiles() -> None:
    """Instrument control: a new input shape must increment the compile count."""
    pmf.CompileCounter.install()
    f = jax.jit(lambda x: x * 3 + 1)
    f(jnp.ones(5)).block_until_ready()
    n0, _ = pmf.CompileCounter.snapshot()
    f(jnp.ones(5)).block_until_ready()
    assert pmf.CompileCounter.snapshot()[0] == n0
    f(jnp.ones(7)).block_until_ready()
    assert pmf.CompileCounter.snapshot()[0] > n0


def test_plan_dedupes_irrelevant_tile_axes_and_wraps_timeout(tmp_path: Path) -> None:
    args = pmf.build_parser().parse_args(
        ["--out-dir", str(tmp_path), "--plan", "--Ks", "32,64", "--shapes", "full,tree",
         "--frame-tiles", "512,1024", "--pair-chunks", "2048", "--timeout-s", "99"]
    )  # fmt: skip
    cells = pmf.plan_cells(args)
    by_form = {f: [c for c in cells if c[1] == f] for f in pmf.FORMS}
    assert len(by_form["A"]) == 2 * 2 * 2  # entropy dtypes x shapes x Ks, tile axes ignored
    assert len(by_form["B"]) == 2 * 2 * 2 * 2  # x two frame tiles
    assert len(by_form["C_triton"]) == len(by_form["B"])
    assert by_form["C_mosaic"] == []  # real Mosaic lowering fails; opt in with --forms
    assert all("--n-frames" not in c for c in cells)
    pmf.print_plan(args)


def test_shapes_match_production_pair_and_boundary_counts() -> None:
    n_res, n_frames, _ = pmf.SHAPES["full"]
    assert n_res * (n_res - 1) // 2 == 89_253 and n_frames == 28_007
    n_res, n_frames, _ = pmf.SHAPES["boundary"]
    assert n_res * (n_res - 1) // 2 * n_frames > 2.45e9
    assert pmf.SHAPES["tree"][2] == 422


def test_synthetic_states_respect_n_used_and_sampling_includes_ends() -> None:
    states = pmf.make_states(30, 400, 16, 7, 0)
    assert states.dtype == np.int32 and states.min() >= 0 and states.max() < 7
    assert len(np.unique(states)) == 7
    assert np.array_equal(states, pmf.make_states(30, 400, 16, 7, 0))  # fixed seed
    idx = pmf.sample_indices(1000, 64, 0)
    assert len(idx) == 64 and idx[0] == 0 and idx[-1] == 999 and len(set(idx)) == 64


def test_float32_count_exactness_limit_is_enforced() -> None:
    cell = pmf.cell_from_args(_args(Path("."), "--form", "B", "--n-frames", str(2**24)))
    with pytest.raises(ValueError, match="not exact"):
        pmf.validate(cell)


@pytest.mark.parametrize("form", ["C_triton", "D"])
def test_triton_forms_lower_for_cuda(form: str, tmp_path: Path) -> None:
    args = pmf.build_parser().parse_args(
        ["--form", form, "--shape", "custom", "--n-res", "8", "--n-frames", "28007", "--K", "100",
         "--pair-chunk", "64", "--frame-tile", "1024", "--out-dir", str(tmp_path), "--lower-check"]
    )  # fmt: skip
    report = pmf.lower_check(args)
    assert report["error"] == "" and report["lowers"] and report["triton_custom_call"]


def test_lower_check_negative_control_goes_red(tmp_path: Path) -> None:
    """A non-power-of-two frame tile (validate bypassed) must not lower for Triton."""
    args = pmf.build_parser().parse_args(
        ["--form", "C_triton", "--shape", "custom", "--n-res", "8", "--n-frames", "64", "--K", "8",
         "--pair-chunk", "8", "--frame-tile", "48", "--out-dir", str(tmp_path), "--lower-check"]
    )  # fmt: skip
    report = pmf.lower_check(args, check_cell=False)
    assert report["lowers"] is False and report["error"] != ""
