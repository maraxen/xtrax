"""Tests for cross-commit compare, parity bridges, GPU/PTX probes, and triage."""

import dataclasses
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from xtrax.kernel_seam import VjpKernel
from xtrax.profiling.claims import ClaimClass, ClaimValidityError, assert_claim_supported
from xtrax.profiling.compare import (
    ParityContract,
    VerdictStatus,
    compare_probe_pair,
    update_baseline_pointer,
)
from xtrax.profiling.emitters import emit_probe_record, probe_record_to_bathos_metrics
from xtrax.profiling.kernel_probe import (
    collect_pass_c_ptx_stats,
    compute_capability_to_sm,
    compute_precision_ratio,
    find_ptxas,
    merge_xla_flags,
    parse_ptx_text,
    parse_ptxas_verbose,
    parse_triton_metadata_dir,
    probe_kernel_dual_pass,
    summarize_ptx_metrics,
)
from xtrax.profiling.record import ProbeRecord
from xtrax.profiling.trace import hlo_text_for
from xtrax.profiling.triage import (
    BottleneckCategory,
    TriageThresholds,
    triage_probe_bundle,
)
from xtrax.testing.kernel_parity import (
    ParityResult,
    build_probe_parity_payload,
    compare_outputs,
    evaluate_kernel_parity_for_probe,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SHA_A = "a" * 40
SHA_B = "b" * 40


def _rec(
    *,
    probe_id: str = "kernel",
    stage: int = 2,
    n_atoms: int = 1000,
    platform: str = "gpu",
    device_kind: str | None = "l40s",
    x64_enabled: bool = False,
    git_sha: str = SHA_A,
    metrics: dict[str, float] | None = None,
    config: dict[str, str] | None = None,
    scopes: dict[str, tuple[float, int] | None] | None = None,
    attribution_method: dict[str, str] | None = None,
    num_replicas: int = 1,
    xla_flags: str = "",
) -> ProbeRecord:
    base_metrics: dict[str, float] = {
        "n_executions": 10.0,
        "n_compilations": 0.0,
        "n_jit_traces": 10.0,
        "total_step_seconds": 1.0,
        "wall_mad_seconds": 0.01,
        "wall_iqr_seconds": 0.02,
        "peak_vram_bytes": 1000.0,
        "output_l2_norm": 4.2,
        "parity_rel_err": 1e-6,
        "parity_max_abs_err": 1e-6,
        "corruption_rel_err": 0.5,
        "work_active_count": 100.0,
    }
    if metrics:
        base_metrics.update(metrics)
    base_config: dict[str, str] = {
        "fn": "mod:step",
        "step_time_basis": "batch",
        "non_degeneracy_status": "pass",
        "corruption_control_status": "pass",
    }
    if config:
        base_config.update(config)
    return ProbeRecord(
        probe_id=probe_id,
        stage=stage,
        n_atoms=n_atoms,
        platform=platform,
        device_kind=device_kind,
        x64_enabled=x64_enabled,
        jax_version="0.10.2",
        jaxlib_version="0.10.2",
        xla_flags=xla_flags,
        git_sha=git_sha,
        metrics=base_metrics,
        config=base_config,
        scopes=scopes,
        attribution_method=attribution_method,
        num_replicas=num_replicas,
    )


# --- Cross-commit comparison & negative controls ----------------------------


def test_compare_bug_triton_atom1_work_drop_and_work_controls():
    base = _rec(git_sha=SHA_A)
    cand_drop = _rec(
        git_sha=SHA_B,
        metrics={"total_step_seconds": 0.25, "work_active_count": 50.0},
    )
    verdict = compare_probe_pair(base, cand_drop, hold_fixed_config=("fn",))
    assert verdict.status is VerdictStatus.REJECT_PARITY
    assert verdict.promotable is False
    assert "work_active_count" in verdict.metric_deltas

    cand_zero = _rec(git_sha=SHA_B, metrics={"total_step_seconds": 0.25, "work_active_count": 0.0})
    assert (
        compare_probe_pair(base, cand_zero, hold_fixed_config=("fn",)).status
        is VerdictStatus.REJECT_PARITY
    )

    base_zero = _rec(git_sha=SHA_A, metrics={"work_active_count": 0.0})
    cand_ok = _rec(git_sha=SHA_B, metrics={"total_step_seconds": 0.5})
    assert (
        compare_probe_pair(
            base_zero, cand_ok, hold_fixed_config=("fn",), require_parity=False
        ).status
        is VerdictStatus.REJECT_UNVERIFIABLE
    )


def test_compare_parameter_and_contract_validation():
    base = _rec(git_sha=SHA_A)
    cand = _rec(git_sha=SHA_B, metrics={"total_step_seconds": 0.5})

    with pytest.raises(ClaimValidityError):
        compare_probe_pair(
            base,
            cand,
            hold_fixed_config=(),
            require_parity=False,
            parity=ParityContract(),
        )
    with pytest.raises(ClaimValidityError):
        compare_probe_pair(base, cand, hold_fixed_config=(), min_speedup=0.9)
    with pytest.raises(ClaimValidityError):
        compare_probe_pair(base, cand, hold_fixed_config=(), max_vram_growth=0.9)
    with pytest.raises(ClaimValidityError):
        ParityContract(min_work_retention_ratio=1.5)
    with pytest.raises(ClaimValidityError):
        ParityContract(min_work_active_count=0.0)


def test_compare_unanimity_diagnostic_and_basis_refusals():
    base = _rec(git_sha=SHA_A)
    diag_rec = _rec(git_sha=SHA_B, config={"diagnostic_mode": "cmdbuf_off"})
    assert (
        compare_probe_pair(base, diag_rec, hold_fixed_config=()).status
        is VerdictStatus.REJECT_UNVERIFIABLE
    )
    with pytest.raises(ClaimValidityError, match="diagnostic"):
        assert_claim_supported([diag_rec], ClaimClass.END_TO_END, allow_no_target=True)
    assert (
        compare_probe_pair(
            base,
            _rec(git_sha=SHA_B, config={"step_time_basis": ""}),
            hold_fixed_config=(),
        ).status
        is VerdictStatus.REJECT_UNVERIFIABLE
    )
    assert (
        compare_probe_pair(
            base,
            _rec(git_sha=SHA_B, config={"step_time_basis": "per_replica"}),
            hold_fixed_config=(),
        ).status
        is VerdictStatus.REJECT_UNVERIFIABLE
    )
    assert (
        compare_probe_pair(
            base,
            dataclasses.replace(_rec(git_sha=SHA_B), contract_version="2.0"),
            hold_fixed_config=(),
        ).status
        is VerdictStatus.REJECT_UNVERIFIABLE
    )
    assert (
        compare_probe_pair(
            base,
            _rec(git_sha=SHA_B, metrics={"total_step_seconds": 0.5}),
            hold_fixed_config=("missing_key",),
        ).status
        is VerdictStatus.REJECT_UNVERIFIABLE
    )


def test_compare_dirty_sha_and_require_parity_false_behavior():
    base = _rec(git_sha=SHA_A)
    cand_dirty = _rec(git_sha=f"{SHA_B}-dirty", metrics={"total_step_seconds": 0.5})

    assert (
        compare_probe_pair(base, cand_dirty, hold_fixed_config=("fn",), allow_dirty=False).status
        is VerdictStatus.REJECT_UNVERIFIABLE
    )
    adv = compare_probe_pair(base, cand_dirty, hold_fixed_config=("fn",), allow_dirty=True)
    assert adv.status is VerdictStatus.ADVISORY_ONLY
    assert adv.promotable is False

    # Dirty allow_dirty=True still rejects regressions
    cand_dirty_slow = _rec(git_sha=f"{SHA_B}-dirty", metrics={"total_step_seconds": 2.0})
    verdict_slow = compare_probe_pair(
        base, cand_dirty_slow, hold_fixed_config=("fn",), allow_dirty=True
    )
    assert verdict_slow.status is VerdictStatus.REVERT_REGRESSION

    # require_parity=False waives missing parity as ADVISORY_ONLY, never explicit "fail"
    cand_unmeasured = _rec(
        git_sha=SHA_B,
        metrics={"total_step_seconds": 0.5},
        config={
            "non_degeneracy_status": "not_measured",
            "corruption_control_status": "not_measured",
        },
    )
    assert (
        compare_probe_pair(base, cand_unmeasured, hold_fixed_config=(), require_parity=False).status
        is VerdictStatus.ADVISORY_ONLY
    )

    cand_explicit_fail = _rec(
        git_sha=SHA_B,
        metrics={"total_step_seconds": 0.5},
        config={"non_degeneracy_status": "fail"},
    )
    assert (
        compare_probe_pair(
            base, cand_explicit_fail, hold_fixed_config=(), require_parity=False
        ).status
        is VerdictStatus.REJECT_PARITY
    )


def test_compare_dispatch_vram_dispersion_and_non_wall_guards():
    base = _rec(git_sha=SHA_A, metrics={"n_compilations": 1.0})
    cand = _rec(git_sha=SHA_B, metrics={"n_compilations": 1.0, "total_step_seconds": 0.5})
    assert (
        compare_probe_pair(base, cand, hold_fixed_config=()).status is VerdictStatus.REJECT_DISPATCH
    )

    base_clean = _rec(git_sha=SHA_A)
    assert (
        compare_probe_pair(
            base_clean,
            _rec(git_sha=SHA_B, metrics={"n_executions": 20.0, "total_step_seconds": 0.5}),
            hold_fixed_config=(),
        ).status
        is VerdictStatus.REJECT_DISPATCH
    )
    assert (
        compare_probe_pair(
            base_clean,
            _rec(git_sha=SHA_B, metrics={"peak_vram_bytes": 1500.0, "total_step_seconds": 0.5}),
            hold_fixed_config=(),
        ).status
        is VerdictStatus.REJECT_VRAM
    )

    # IQR noise floor (remove MAD so IQR path executes with 0.7413 factor)
    base_iqr = ProbeRecord(
        probe_id="b",
        stage=2,
        n_atoms=1000,
        platform="gpu",
        device_kind="l40s",
        git_sha=SHA_A,
        x64_enabled=False,
        jax_version="0.10.2",
        jaxlib_version="0.10.2",
        xla_flags="",
        metrics={
            "n_executions": 10.0,
            "n_compilations": 0.0,
            "total_step_seconds": 1.0,
            "wall_iqr_seconds": 0.2,
            "peak_vram_bytes": 1000.0,
            "output_l2_norm": 4.2,
            "parity_rel_err": 1e-6,
            "corruption_rel_err": 0.5,
        },
        config={
            "fn": "mod:step",
            "step_time_basis": "batch",
            "non_degeneracy_status": "pass",
            "corruption_control_status": "pass",
        },
    )
    cand_iqr = dataclasses.replace(
        base_iqr,
        probe_id="c",
        git_sha=SHA_B,
        metrics={**base_iqr.metrics, "total_step_seconds": 0.9},
    )
    v_noise = compare_probe_pair(base_iqr, cand_iqr, hold_fixed_config=())
    assert v_noise.status is VerdictStatus.REVERT_NOISE
    expected_nf = 1.0 + 2.0 * 0.7413 * math.hypot(0.2 / 1.0, 0.2 / 0.9)
    assert v_noise.noise_floor_x == pytest.approx(expected_nf)

    # Non-wall primary metric ignores wall_mad_seconds
    v_vram = compare_probe_pair(
        base_clean,
        _rec(git_sha=SHA_B, metrics={"peak_vram_bytes": 800.0}),
        hold_fixed_config=(),
        primary_metric="peak_vram_bytes",
    )
    assert v_vram.status is VerdictStatus.KEEP
    assert v_vram.noise_floor_x == pytest.approx(1.0)


def test_update_baseline_pointer_ratchet_and_sidecar(tmp_path):
    p_base = tmp_path / "base.json"
    p_cand = tmp_path / "cand.json"
    p_rej = tmp_path / "rej.json"
    base = _rec(probe_id="k1", git_sha=SHA_A)
    cand = _rec(probe_id="k1", git_sha=SHA_B, metrics={"total_step_seconds": 0.5})
    bad_cand = _rec(probe_id="k1", git_sha="c" * 40, metrics={"total_step_seconds": 2.0})
    base.write(p_base)
    cand.write(p_cand)
    bad_cand.write(p_rej)

    verdict_keep = compare_probe_pair(base, cand, hold_fixed_config=("fn",))
    assert verdict_keep.status is VerdictStatus.KEEP
    ptr = tmp_path / "ptr.json"
    update_baseline_pointer(ptr, verdict_keep, p_cand, initialize=True)
    ptr_data = json.loads(ptr.read_text())
    assert ptr_data["git_sha"] == SHA_B
    assert ptr_data["previous_sha"] == SHA_A

    # Chain continuity failure when pointer points to SHA_B but verdict baseline is SHA_A
    with pytest.raises(ClaimValidityError, match="continuity"):
        update_baseline_pointer(ptr, verdict_keep, p_cand, initialize=False)

    verdict_rev = compare_probe_pair(cand, bad_cand, hold_fixed_config=("fn",))
    assert verdict_rev.status is VerdictStatus.REVERT_REGRESSION
    with pytest.raises(ClaimValidityError, match="rejected_dir"):
        update_baseline_pointer(ptr, verdict_rev, p_rej)
    sidecar = update_baseline_pointer(ptr, verdict_rev, p_rej, rejected_dir=tmp_path / "rejected")
    assert sidecar.name.endswith(".verdict.json")
    assert json.loads(sidecar.read_text())["status"] == "revert_regression"


# --- Kernel Parity Payload & Non-Finite / Int Mismatch --------------------


def test_build_probe_parity_payload_and_nan_finiteness():
    got = {"val": jnp.array([1.0, 2.0]), "idx": jnp.array([1, 2], dtype=jnp.int32)}
    ref = {"val": jnp.array([1.0, 2.0]), "idx": jnp.array([1, 2], dtype=jnp.int32)}
    corr = {"val": jnp.array([1.0, 3.0]), "idx": jnp.array([1, 2], dtype=jnp.int32)}
    fwd = compare_outputs(got, ref, rel=1e-4)
    payload = build_probe_parity_payload(fwd, got, ref, rel=1e-4, corrupted_output=corr)
    assert payload.config_statuses["non_degeneracy_status"] == "pass"
    assert payload.config_statuses["corruption_control_status"] == "pass"
    assert payload.metrics["output_l2_norm"] > 0.0

    # Non-finite NaN output safely omits NaN from ProbeRecord metrics and triggers REJECT_PARITY
    nan_out = {"val": jnp.array([float("nan"), 2.0]), "idx": jnp.array([1, 2], dtype=jnp.int32)}
    nan_fwd = compare_outputs(nan_out, ref, rel=1e-4)
    nan_payload = build_probe_parity_payload(nan_fwd, nan_out, ref, rel=1e-4, corrupted_output=corr)
    assert nan_payload.config_statuses["non_degeneracy_status"] == "fail"
    assert "parity_nonfinite_metrics" in nan_payload.config_statuses
    for v in nan_payload.metrics.values():
        assert math.isfinite(v)

    # Integer mismatch without corruption produces fail without NaN crash
    bad_int = {"val": jnp.array([1.0, 2.0]), "idx": jnp.array([1, 99], dtype=jnp.int32)}
    bad_fwd = compare_outputs(bad_int, ref, rel=1e-4)
    int_payload = build_probe_parity_payload(bad_fwd, bad_int, ref, rel=1e-4)
    assert int_payload.config_statuses["non_degeneracy_status"] == "fail"
    assert int_payload.config_statuses["corruption_control_status"] == "not_measured"

    # Reject passing both corrupted_output and corruption_result
    with pytest.raises(ValueError, match="at most one"):
        build_probe_parity_payload(
            fwd,
            got,
            ref,
            corrupted_output=corr,
            corruption_result=ParityResult(False, 0.5, 0, "err"),
        )


def test_evaluate_kernel_parity_for_probe_with_vjp():
    def ref_fn(w, x):
        return jnp.sum(w * x)

    def fwd_fn(w, x):
        return ref_fn(w, x), (w, x)

    def bwd_fn(res, ct):
        w, x = res
        return (ct * x, ct * w)

    args = (jnp.array([2.0, 3.0]), jnp.array([4.0, 5.0]))
    vk = VjpKernel(fwd_fn, bwd_fn)
    with pytest.raises(ValueError, match="vjp_key"):
        evaluate_kernel_parity_for_probe(ref_fn, ref_fn, args, lambda fn: fn, vjp_kernel=vk)

    payload = evaluate_kernel_parity_for_probe(
        ref_fn,
        ref_fn,
        args,
        lambda fn: lambda w, x: fn(w, x) + 1.0,
        vjp_kernel=vk,
        vjp_key=jax.random.PRNGKey(0),
    )
    assert payload.config_statuses["non_degeneracy_status"] == "pass"
    assert payload.config_statuses["corruption_control_status"] == "pass"
    assert payload.config_statuses["vjp_parity_status"] == "pass"


# --- Precision Ratio, PTX/ptxas/Triton, Merge Flags, & Dual-Pass Probe ----


def test_compute_precision_ratio_prolix_and_error_guards():
    single = _rec(git_sha=SHA_A, x64_enabled=False, config={"precision": "single"})
    mixed = _rec(
        git_sha=SHA_A,
        x64_enabled=True,
        metrics={"total_step_seconds": 2.0},
        config={"precision": "mixed"},
    )
    assert compute_precision_ratio(single, mixed) == pytest.approx(2.0)

    # Prolix shared-process x64_enabled=True pattern
    single_x64 = _rec(git_sha=SHA_A, x64_enabled=True, config={"precision": "single"})
    assert compute_precision_ratio(single_x64, mixed) == pytest.approx(2.0)

    with pytest.raises(ClaimValidityError, match="x64_enabled"):
        compute_precision_ratio(
            single, _rec(git_sha=SHA_A, x64_enabled=False, config={"precision": "mixed"})
        )
    with pytest.raises(ClaimValidityError, match="precision"):
        compute_precision_ratio(_rec(git_sha=SHA_A, x64_enabled=True), mixed)


def test_merge_xla_flags_ptx_ptxas_triton_and_hlo_kwargs(tmp_path):
    merged = merge_xla_flags(
        "--xla_dump_to=/tmp/old --xla_dump_hlo_as_text true "
        "--xla_gpu_enable_command_buffer=FUSION --xla_foo=1",
        {"--xla_gpu_enable_command_buffer": ""},
    )
    assert "--xla_dump_to" not in merged
    assert "--xla_dump_hlo_as_text" not in merged
    assert "--xla_foo=1" in merged
    assert "--xla_gpu_enable_command_buffer=" in merged
    assert compute_capability_to_sm("8.9") == "sm_89"
    assert compute_capability_to_sm("12.0") == "sm_120"
    assert find_ptxas() is None or isinstance(find_ptxas(), str)

    # HLO text with kwargs
    fn = jax.jit(lambda x, *, scale=1.0: x * scale)
    hlo = hlo_text_for(fn, jnp.ones(4), scale=2.0)
    assert "HloModule" in hlo

    # PTX parser + dedup + ptxas + triton metadata
    ptx_sample = """
    // comment
    .visible .entry my_kern(
        .param .u64 ptr
    )
    .maxnreg 128
    {
        .reg .f64 %fd<4>;
        .reg .f32 %f0, %f1;
        .reg .pred %p<2>;
        .local .b8 __local_buf[16];
        ld.global.v2.f32 {%f0, %f1}, [%rd1];
        ret;
    }
    """
    parsed = parse_ptx_text(ptx_sample, ptx_file="k.ptx")
    assert len(parsed) == 1
    assert parsed[0].kernel_name == "my_kern"
    assert parsed[0].f64_vregs == 4
    assert parsed[0].ptx_local_bytes == 16
    assert parsed[0].vregs_total_32bit_equiv == 4 * 2 + 2 + 2

    ptxas_log = (
        "ptxas info    : Compiling entry function 'my_kern' for 'sm_89'\n"
        "ptxas info    : Used 168 registers, 256 bytes stack frame, "
        "64 bytes spill stores, 32 bytes spill loads, 4096 bytes smem\n"
    )
    pa = parse_ptxas_verbose(ptxas_log)
    assert pa["my_kern"].phys_regs == 168
    assert pa["my_kern"].spill_store_bytes + pa["my_kern"].spill_load_bytes == 96

    triton_dump = tmp_path / "tdump"
    triton_cache = tmp_path / "tcache"
    xla_dump = tmp_path / "xdump"
    triton_dump.mkdir()
    triton_cache.mkdir()
    xla_dump.mkdir()
    (triton_dump / "my_kern.ptx").write_text(ptx_sample)
    (triton_dump / "my_kern.cubin").write_bytes(b"ELF")
    meta_json = json.dumps({"name": "my_kern", "n_regs": 170, "n_spills": 3})
    (triton_dump / "my_kern.json").write_text(meta_json)
    (xla_dump / "dup.ptx").write_text(ptx_sample)  # duplicate entry

    t_meta = parse_triton_metadata_dir(triton_dump)
    assert t_meta["my_kern"].n_spills == 3

    col = collect_pass_c_ptx_stats(
        triton_dump, triton_cache, xla_dump, executed_kernels=("my_kern",)
    )
    assert len(col.stats) == 1
    assert col.dedup_count == 1
    assert col.filter_status == "matched"
    metrics = summarize_ptx_metrics(list(col.stats))
    assert metrics["ptx_f64_vregs_max"] == 4.0
    assert metrics["triton_spills_max"] == 3.0


def test_diag_worker_subprocess_isolation_and_dual_pass_cpu(tmp_path):
    # Verify _diag_worker has no top-level jax or xtrax.cli imports
    src_dir = REPO_ROOT / "src"
    code = (
        "import sys; import xtrax.profiling._diag_worker; "
        "bad = [m for m in ('jax', 'xtrax.cli') if m in sys.modules]; "
        "assert not bad, bad"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(src_dir), "PATH": "/usr/bin:/bin"},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    # End-to-end dual pass on CPU using importable fixture
    from tests.cli._profile_fixtures import vmapped_twin

    res = probe_kernel_dual_pass(
        vmapped_twin,
        (jnp.zeros((8, 16), jnp.float32),),
        probe_id="dual-test",
        n_atoms=8,
        stage=1,
        fn_spec="tests.cli._profile_fixtures:vmapped_twin",
        shapes="x=(8,16)f32",
        n_steady=2,
        n_wall=2,
        prod_path=tmp_path / "prod.json",
        diag_dir=tmp_path / "diags",
        isolate_passes=False,
    )
    assert res.prod_record.config["diagnostic_mode"] == "none"
    assert res.diag_record is not None
    assert res.diag_record.config["diagnostic_mode"] == "cmdbuf_off"
    assert res.diag_record_path is not None and res.diag_record_path.is_file()


# --- Deterministic Triage + Playbook Link Integrity -----------------------


def test_triage_all_six_rules_positive_negative_and_playbook_anchors():
    # Rule 1 + Rule 2 + Rule 3 + Rule 4 + Rule 5 + Rule 6
    prod = _rec(
        probe_id="dhfr",
        device_kind="rtx_pro_6000_blackwell",
        metrics={
            "n_compilations": 2.0,
            "exec_per_element": 0.6,
            "total_step_seconds": 1.0,
            "host_seconds": 0.4,
            "mixed_over_single_ratio": 1.6,
            "active_work_ratio": 0.1,
            "ptx_lmem_bytes_max": 128.0,
            "ctas_per_atomic_target": 128.0,
        },
        num_replicas=2,
    )
    diag = _rec(
        probe_id="dhfr:diag",
        scopes={"nl_kernel": (0.8, 1), "fft": (0.2, 1)},
        attribution_method={"nl_kernel": "op_name", "fft": "op_name"},
        config={"diagnostic_mode": "cmdbuf_off"},
        metrics={"ptx_f64_vregs_max": 12.0, "ptxas_regs_max": 192.0},
    )
    lo_b = _rec(probe_id="lo", git_sha=SHA_A, num_replicas=1, metrics={"total_step_seconds": 1.0})
    hi_b_contend = _rec(
        probe_id="hi", git_sha=SHA_A, num_replicas=4, metrics={"total_step_seconds": 9.0}
    )
    findings = triage_probe_bundle(prod, diag=diag, batch_scaling_pair=(lo_b, hi_b_contend))
    cats = [f.category for f in findings]
    assert cats == [
        BottleneckCategory.RECOMPILE_DRIFT,
        BottleneckCategory.HOST_BOUNDARY_SYNC,
        BottleneckCategory.FP64_PIPE_SATURATION,
        BottleneckCategory.SPATIAL_DEAD_WORK,
        BottleneckCategory.REGISTER_SPILL_PRESSURE,
        BottleneckCategory.ATOMIC_CONTENTION,
    ]

    # Positive control: ideal concurrent batch scaling (eta_rep = 1.0 / (2.0 / 4) = 2.0 >= 0.50)
    hi_b_good = _rec(
        probe_id="hi_good", git_sha=SHA_A, num_replicas=4, metrics={"total_step_seconds": 2.0}
    )
    clean_prod = _rec(metrics={"total_step_seconds": 1.0, "host_seconds": 0.05})
    assert triage_probe_bundle(clean_prod, batch_scaling_pair=(lo_b, hi_b_good)) == []
    strict = TriageThresholds(host_share_flag=0.01)
    assert len(triage_probe_bundle(clean_prod, thresholds=strict)) == 1

    # Verify every playbook_ref resolves to an existing markdown file and matching heading anchor
    for item in findings:
        rel_path, anchor = item.playbook_ref.split("#", 1)
        md_file = REPO_ROOT / rel_path
        assert md_file.is_file(), f"Missing playbook file: {md_file}"
        headings = []
        for line in md_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("#"):
                raw_h = line.lstrip("#").strip().lower()
                raw_h = raw_h.replace("`", "")
                slug = re.sub(r"[^a-z0-9\s_-]", "", raw_h).strip()
                slug = re.sub(r"\s+", "-", slug)
                headings.append(slug)
        assert anchor in headings, f"Missing anchor #{anchor} in {rel_path}; got {headings}"


def test_emitters_bathos_env_and_format_helper(tmp_path, monkeypatch):
    monkeypatch.setenv("BTH_RUN_ID", "run-123")
    monkeypatch.setenv("BTH_CAMPAIGN_ID", "camp-456")
    out = tmp_path / "bth.json"
    rec = emit_probe_record(
        path=out,
        probe_id="bth-probe",
        stage=1,
        n_atoms=64,
        platform="cpu",
        metrics={"total_step_seconds": 0.01},
        git_sha=SHA_A,
    )
    assert rec.config["bathos_run_id"] == "run-123"
    assert rec.config["bathos_campaign_id"] == "camp-456"
    verdict = compare_probe_pair(
        _rec(stage=1, platform="cpu", device_kind=None, git_sha=SHA_A),
        _rec(
            stage=1,
            platform="cpu",
            device_kind=None,
            git_sha=SHA_B,
            metrics={"total_step_seconds": 0.5},
        ),
        hold_fixed_config=(),
        max_vram_growth=None,
    )
    formatted = probe_record_to_bathos_metrics(rec, verdict=verdict)
    assert formatted["bathos_run_id"] == "run-123"
    assert formatted["verdict_status"] == verdict.status.value
    assert formatted["promotable"] in ("true", "false")


def test_emitters_field_names_and_minimal_bathos_metrics(tmp_path):
    from xtrax.profiling.emitters import probe_record_field_names

    names = probe_record_field_names()
    assert "probe_id" in names
    assert "contract_version" in names

    rec = _rec(stage=1, platform="cpu", device_kind=None, config={"fn": "m:f"})
    plain = probe_record_to_bathos_metrics(rec, verdict=None)
    assert "device_kind" not in plain
    assert "bathos_run_id" not in plain
    assert "verdict_status" not in plain

    unverif = compare_probe_pair(
        _rec(stage=1, platform="cpu", device_kind=None, git_sha=SHA_A, x64_enabled=False),
        _rec(stage=1, platform="cpu", device_kind=None, git_sha=SHA_B, x64_enabled=True),
        hold_fixed_config=(),
        max_vram_growth=None,
    )
    assert unverif.status is VerdictStatus.REJECT_UNVERIFIABLE
    with_unverif = probe_record_to_bathos_metrics(rec, verdict=unverif)
    assert with_unverif["verdict_status"] == "reject_unverifiable"
    assert with_unverif["promotable"] == "false"
    assert "speedup_x" not in with_unverif


def test_diag_worker_unit_helpers_and_in_process_modes(tmp_path, monkeypatch):
    from xtrax.profiling import _diag_worker as dw

    with pytest.raises(ValueError, match="module.path:symbol"):
        dw._load_symbol("no_colon")
    with pytest.raises(ValueError, match="Malformed import path"):
        dw._load_symbol(":attr")
    sym = dw._load_symbol("tests.cli._profile_fixtures:vmapped_twin")
    assert callable(sym)

    assert dw._dtype_lookup("f32", jnp) == jnp.float32
    with pytest.raises(ValueError, match="Unsupported shape dtype"):
        dw._dtype_lookup("c128", jnp)

    built = dw._build_args_from_shapes("x=(2,3)f32 y=(4,)<i32>", jnp)
    assert built[0].shape == (2, 3) and built[1].dtype == jnp.int32
    with pytest.raises(ValueError, match="Malformed shape entry"):
        dw._build_args_from_shapes("badtoken", jnp)
    with pytest.raises(ValueError, match="Malformed shape spec"):
        dw._build_args_from_shapes("x=bad", jnp)

    assert dw._split_factory_output(((1,), {"k": 2})) == ((1,), {"k": 2})
    assert dw._split_factory_output((1, 2)) == ((1, 2), {})
    assert dw._split_factory_output(42) == ((42,), {})

    res_in = dw._resolve_inputs(
        inputs_spec="tests.cli._profile_fixtures:make_inputs",
        scale=8,
        shapes=None,
    )
    assert res_in[1]["x"].shape[0] == 8
    res_shapes = dw._resolve_inputs(inputs_spec=None, scale=None, shapes="x=(2,4)f32")
    assert res_shapes[0][0].shape == (2, 4)
    with pytest.raises(ValueError, match="either --inputs or --shapes"):
        dw._resolve_inputs(inputs_spec=None, scale=None, shapes=None)

    class _LowerFails:
        def lower(self, *a, **kw):
            raise TypeError("cannot lower directly")

        def __call__(self, x):
            return x + 1.0

    hlo_fallback, skip1 = dw._resolve_pass_b_hlo_text(_LowerFails(), (jnp.zeros((2,)),), {}, jax)
    assert hlo_fallback is not None and skip1 is None

    def _unjittable(x):
        if len(x) > 0:
            raise ValueError("host error")
        return x

    # Use object without jax array conversion so jax.jit trace fails
    hlo_none, skip2 = dw._resolve_pass_b_hlo_text(_unjittable, ("not_an_array",), {}, jax)
    assert hlo_none is None and skip2 == "unjittable_wrapper"

    class _GpuDev:
        platform = "gpu"
        compute_capability = "8.9"

    monkeypatch.setattr(jax, "devices", lambda: [_GpuDev()])
    assert dw._device_compute_capability(jax) == "8.9"

    def _dev_err():
        raise RuntimeError("boom")

    monkeypatch.setattr(jax, "devices", _dev_err)
    assert dw._device_compute_capability(jax) is None
    monkeypatch.undo()

    # Op bucket classification + _collect_diag_scopes
    op_times = {
        "k1": (0.80, 2),
        "k2": (0.19, 1),
        "tiny": (0.005, 1),
        "custom": (0.005, 1),
    }
    scope_map = {
        "k1": "known_label",
        "k2": "triton:my_triton_k",
        "tiny": None,
        "custom": "custom_call:my_cc",
    }
    buckets, tritons = dw._accumulate_op_buckets(
        op_times,
        scope_map,
        frozenset({"known_label"}),
        {"custom_call:my_cc": [0.1, 1]},
    )
    assert "triton:my_triton_k" in buckets
    assert "custom_call:my_cc" not in buckets
    assert "tiny" not in buckets
    assert "my_triton_k" in tritons

    hlo_sample = (
        "HloModule m\n"
        "ENTRY %main (p: f32[4]) -> f32[4] {\n"
        '  %ns_op = f32[4] add(%p, %p), metadata={op_name="jit(fn)/my_named_scope/add"}\n'
        '  %t = f32[4] custom-call(), custom_call_target="triton_x", name="pallas_k"\n'
        "}"
    )
    ev = [
        {"ph": "X", "name": "k_op", "ts": 0, "dur": 1_000_000, "args": {"hlo_op": "k_op"}},
        {
            "ph": "X",
            "name": "ns_op",
            "ts": 1_000_000,
            "dur": 500_000,
            "args": {"hlo_op": "ns_op"},
        },
    ]
    scopes_d, attr_d, tk_d = dw._collect_diag_scopes(
        ev, hlo_sample, frozenset({"missing_lbl", "my_named_scope"}), n_steady=2
    )
    assert scopes_d["missing_lbl"] is None
    assert scopes_d["my_named_scope"] == [0.25, 1] and attr_d["my_named_scope"] == "named_scope"
    assert scopes_d["k_op"] == [0.5, 1] and attr_d["k_op"] == "op_name"
    assert tk_d == ["pallas_k"]

    with pytest.raises(SystemExit) as exc_ns:
        dw.main(["--mode", "prod", "--fn", "m:f", "--n-steady", "0", "--result-json", "r.json"])
    assert exc_ns.value.code != 0
    with pytest.raises(SystemExit) as exc_nw:
        dw.main(["--mode", "prod", "--fn", "m:f", "--n-wall", "0", "--result-json", "r.json"])
    assert exc_nw.value.code != 0

    for mode_name in ("prod", "diag", "ptx"):
        out_json = tmp_path / f"{mode_name}.json"
        rc = dw.main(
            [
                "--mode",
                mode_name,
                "--fn",
                "tests.cli._profile_fixtures:vmapped_twin",
                "--shapes",
                "x=(4,8)f32",
                "--n-steady",
                "1",
                "--n-wall",
                "1",
                "--result-json",
                str(out_json),
            ]
        )
        assert rc == 0
        assert json.loads(out_json.read_text())["mode"] == mode_name


def test_triage_validation_and_negative_branches():
    from xtrax.profiling.triage import _validate_batch_scaling_pair

    lo = _rec(probe_id="lo", num_replicas=1, metrics={"total_step_seconds": 1.0})
    bad_ver = _rec(probe_id="hi", num_replicas=4, metrics={"total_step_seconds": 4.0})
    bad_ver = dataclasses.replace(bad_ver, contract_version="99.0")
    with pytest.raises(ClaimValidityError, match="contract_version"):
        _validate_batch_scaling_pair(lo, bad_ver)

    with pytest.raises(ClaimValidityError, match="stage >= 1"):
        _validate_batch_scaling_pair(
            _rec(stage=0, platform="cpu", device_kind=None, num_replicas=1),
            _rec(stage=0, platform="cpu", device_kind=None, num_replicas=4),
        )
    with pytest.raises(ClaimValidityError, match="GPU batch_scaling_pair requires stage >= 2"):
        _validate_batch_scaling_pair(_rec(stage=1, num_replicas=1), _rec(stage=2, num_replicas=4))
    with pytest.raises(ClaimValidityError, match="stage mismatch"):
        _validate_batch_scaling_pair(
            _rec(stage=2, platform="gpu", device_kind="l40s", num_replicas=1),
            _rec(stage=3, platform="gpu", device_kind="l40s", num_replicas=4),
        )
    with pytest.raises(ClaimValidityError, match="git_sha mismatch"):
        _validate_batch_scaling_pair(
            _rec(git_sha=SHA_A, num_replicas=1), _rec(git_sha=SHA_B, num_replicas=4)
        )
    with pytest.raises(ClaimValidityError, match="mismatch on 'device_kind'"):
        _validate_batch_scaling_pair(
            _rec(device_kind="l40s", num_replicas=1),
            _rec(device_kind="h200", num_replicas=4),
        )
    with pytest.raises(ClaimValidityError, match="diagnostic record"):
        _validate_batch_scaling_pair(
            _rec(num_replicas=1, config={"diagnostic_mode": "cmdbuf_off"}),
            _rec(num_replicas=4),
        )
    with pytest.raises(ClaimValidityError, match="num_replicas"):
        _validate_batch_scaling_pair(_rec(num_replicas=4), _rec(num_replicas=2))
    with pytest.raises(ClaimValidityError, match="invalid step_time_basis"):
        _validate_batch_scaling_pair(
            _rec(num_replicas=1, config={"step_time_basis": "weird"}),
            _rec(num_replicas=2, config={"step_time_basis": "weird"}),
        )
    no_total = _rec(num_replicas=1)
    del no_total.metrics["total_step_seconds"]
    with pytest.raises(ClaimValidityError, match="missing total_step_seconds"):
        _validate_batch_scaling_pair(no_total, _rec(num_replicas=2))
    with pytest.raises(ClaimValidityError, match="finite > 0.0"):
        _validate_batch_scaling_pair(
            _rec(num_replicas=1, metrics={"total_step_seconds": 0.0}),
            _rec(num_replicas=2),
        )

    # precision_pair & high active_work_ratio & CTA-only vs scaling-only atomic contention
    s_prec = _rec(
        stage=2,
        platform="gpu",
        device_kind="l40s",
        x64_enabled=False,
        git_sha=SHA_A,
        metrics={"total_step_seconds": 1.0, "host_seconds": 0.05},
        config={"precision": "single"},
    )
    m_prec = _rec(
        stage=2,
        platform="gpu",
        device_kind="l40s",
        x64_enabled=True,
        git_sha=SHA_A,
        metrics={"total_step_seconds": 2.0, "host_seconds": 0.05},
        config={"precision": "mixed"},
    )
    assert [f.category for f in triage_probe_bundle(s_prec, precision_pair=(s_prec, m_prec))] == [
        BottleneckCategory.FP64_PIPE_SATURATION
    ]

    high_work = _rec(metrics={"active_work_ratio": 0.85, "total_step_seconds": 1.0})
    assert triage_probe_bundle(high_work) == []

    cta_only = _rec(metrics={"ctas_per_atomic_target": 128.0, "total_step_seconds": 1.0})
    assert [f.category for f in triage_probe_bundle(cta_only)] == [
        BottleneckCategory.ATOMIC_CONTENTION
    ]

    lo_s = _rec(probe_id="lo_s", git_sha=SHA_A, num_replicas=1, metrics={"total_step_seconds": 1.0})
    hi_s = _rec(probe_id="hi_s", git_sha=SHA_A, num_replicas=4, metrics={"total_step_seconds": 9.0})
    clean = _rec(metrics={"total_step_seconds": 1.0, "host_seconds": 0.05})
    assert [f.category for f in triage_probe_bundle(clean, batch_scaling_pair=(lo_s, hi_s))] == [
        BottleneckCategory.ATOMIC_CONTENTION
    ]

    # Rules negative/no-trigger guards
    h100_fp64 = _rec(
        device_kind="NVIDIA H100",
        metrics={"mixed_over_single_ratio": 2.0, "total_step_seconds": 1.0, "host_seconds": 0.05},
    )
    assert triage_probe_bundle(h100_fp64) == []
    bad_ratio = _rec(metrics={"mixed_over_single_ratio": 0.0, "total_step_seconds": 1.0})
    with pytest.raises(ClaimValidityError, match="mixed_over_single_ratio"):
        triage_probe_bundle(bad_ratio)

    low_share_dead = _rec(metrics={"active_work_ratio": 0.1, "total_step_seconds": 1.0})
    low_pct_diag = _rec(
        scopes={"a": (0.25, 1), "b": (0.25, 1), "c": (0.25, 1), "d": (0.25, 1)},
        attribution_method={"a": "op_name", "b": "op_name", "c": "op_name", "d": "op_name"},
        config={"diagnostic_mode": "cmdbuf_off"},
    )
    assert triage_probe_bundle(low_share_dead, diag=low_pct_diag) == []
    # Scopes all None triggers None-filtering path cleanly
    all_none_diag = _rec(
        scopes={"s": None},
        attribution_method={},
        config={"diagnostic_mode": "cmdbuf_off"},
    )
    assert len(triage_probe_bundle(low_share_dead, diag=all_none_diag)) == 1

    # Lmem only triggers register spill even without ptxas_regs_max
    lmem_only = _rec(metrics={"ptx_lmem_bytes_max": 64.0, "total_step_seconds": 1.0})
    cats_lmem = [f.category for f in triage_probe_bundle(lmem_only)]
    assert cats_lmem == [BottleneckCategory.REGISTER_SPILL_PRESSURE]


def test_compare_comprehensive_guard_branches(tmp_path):
    # Invalid compare params & ParityContract bounds
    with pytest.raises(ClaimValidityError):
        ParityContract(max_rel_err=-1.0)
    with pytest.raises(ClaimValidityError):
        ParityContract(min_work_active_count=0.0)
    with pytest.raises(ClaimValidityError):
        ParityContract(min_work_retention_ratio=1.5)

    b = _rec(git_sha=SHA_A)
    c = _rec(git_sha=SHA_B, metrics={"total_step_seconds": 0.5})
    for kw in (
        {"require_parity": False, "parity": ParityContract()},
        {"min_speedup": 0.9},
        {"noise_sigma": -0.1},
        {"max_compile_increase": -1.0},
        {"max_dispatch_growth": 0.9},
        {"max_vram_growth": 0.0},
    ):
        with pytest.raises(ClaimValidityError):
            compare_probe_pair(b, c, hold_fixed_config=(), **kw)

    # Contract & env failures
    bad_ver = dataclasses.replace(b, contract_version="99.0")
    assert (
        compare_probe_pair(bad_ver, c, hold_fixed_config=()).status
        is VerdictStatus.REJECT_UNVERIFIABLE
    )
    assert (
        compare_probe_pair(
            _rec(stage=0, platform="cpu", device_kind=None, git_sha=SHA_A),
            _rec(stage=0, platform="cpu", device_kind=None, git_sha=SHA_B),
            hold_fixed_config=(),
        ).status
        is VerdictStatus.REJECT_UNVERIFIABLE
    )
    assert (
        compare_probe_pair(
            _rec(stage=1, platform="gpu", git_sha=SHA_A),
            _rec(stage=1, platform="gpu", git_sha=SHA_B),
            hold_fixed_config=(),
        ).status
        is VerdictStatus.REJECT_UNVERIFIABLE
    )
    assert (
        compare_probe_pair(
            b,
            _rec(git_sha=SHA_B, config={"step_time_basis": "invalid"}),
            hold_fixed_config=(),
        ).status
        is VerdictStatus.REJECT_UNVERIFIABLE
    )
    assert (
        compare_probe_pair(
            b,
            _rec(git_sha=SHA_B, config={"step_time_basis": "per_replica"}),
            hold_fixed_config=(),
        ).status
        is VerdictStatus.REJECT_UNVERIFIABLE
    )
    assert (
        compare_probe_pair(b, c, hold_fixed_config=("missing_cfg",)).status
        is VerdictStatus.REJECT_UNVERIFIABLE
    )
    assert (
        compare_probe_pair(
            b,
            _rec(git_sha=SHA_B, config={"fn": "other:step"}),
            hold_fixed_config=("fn",),
        ).status
        is VerdictStatus.REJECT_UNVERIFIABLE
    )
    assert (
        compare_probe_pair(_rec(git_sha="not_a_sha"), c, hold_fixed_config=()).status
        is VerdictStatus.REJECT_UNVERIFIABLE
    )

    # Parity thresholds, work witnesses, export guardrails, custom bounds
    assert (
        compare_probe_pair(
            b,
            _rec(git_sha=SHA_B, config={"non_degeneracy_status": "fail"}),
            hold_fixed_config=(),
            require_parity=False,
        ).status
        is VerdictStatus.REJECT_PARITY
    )
    assert (
        compare_probe_pair(
            b,
            _rec(git_sha=SHA_B, metrics={"output_l2_norm": 0.0}),
            hold_fixed_config=(),
        ).status
        is VerdictStatus.REJECT_PARITY
    )
    assert (
        compare_probe_pair(
            b,
            _rec(
                git_sha=SHA_B,
                metrics={"total_step_seconds": 0.5, "parity_max_abs_err": 1.0},
            ),
            hold_fixed_config=(),
            parity=ParityContract(max_abs_err=1e-3),
        ).status
        is VerdictStatus.REJECT_PARITY
    )
    assert (
        compare_probe_pair(
            b,
            _rec(
                git_sha=SHA_B,
                metrics={"total_step_seconds": 0.5, "parity_vjp_rel_err": 1.0},
            ),
            hold_fixed_config=(),
            parity=ParityContract(max_vjp_rel_err=1e-3),
        ).status
        is VerdictStatus.REJECT_PARITY
    )
    assert (
        compare_probe_pair(
            b,
            _rec(
                git_sha=SHA_B,
                metrics={"total_step_seconds": 0.5},
                config={"vjp_parity_status": "not_measured"},
            ),
            hold_fixed_config=(),
        ).status
        is VerdictStatus.KEEP
    )
    assert (
        compare_probe_pair(
            b,
            _rec(
                git_sha=SHA_B,
                metrics={"total_step_seconds": 0.5, "parity_vjp_rel_err": 0.0},
                config={"vjp_parity_status": "not_measured"},
            ),
            hold_fixed_config=(),
            parity=ParityContract(max_vjp_rel_err=1e-3),
        ).status
        is VerdictStatus.REJECT_PARITY
    )
    assert (
        compare_probe_pair(
            b,
            _rec(git_sha=SHA_B, metrics={"total_step_seconds": 0.5}),
            hold_fixed_config=(),
            parity=ParityContract(require_export_ok=True),
        ).status
        is VerdictStatus.REJECT_PARITY
    )
    assert (
        compare_probe_pair(
            b,
            _rec(git_sha=SHA_B, metrics={"total_step_seconds": 0.5}),
            hold_fixed_config=(),
            parity=ParityContract(metric_ceilings={"missing_k": 1.0}),
        ).status
        is VerdictStatus.REJECT_PARITY
    )
    assert (
        compare_probe_pair(
            b,
            _rec(git_sha=SHA_B, metrics={"total_step_seconds": 0.5, "extra": 5.0}),
            hold_fixed_config=(),
            parity=ParityContract(metric_ceilings={"extra": 1.0}),
        ).status
        is VerdictStatus.REJECT_PARITY
    )
    assert (
        compare_probe_pair(
            b,
            _rec(git_sha=SHA_B, metrics={"total_step_seconds": 0.5}),
            hold_fixed_config=(),
            parity=ParityContract(metric_floors={"missing_floor": 1.0}),
        ).status
        is VerdictStatus.REJECT_PARITY
    )
    assert (
        compare_probe_pair(
            b,
            _rec(git_sha=SHA_B, metrics={"total_step_seconds": 0.5, "floor_k": 0.1}),
            hold_fixed_config=(),
            parity=ParityContract(metric_floors={"floor_k": 1.0}),
        ).status
        is VerdictStatus.REJECT_PARITY
    )
    assert (
        compare_probe_pair(
            _rec(git_sha=SHA_A, metrics={"work_active_count": 0.0}),
            c,
            hold_fixed_config=(),
        ).status
        is VerdictStatus.REJECT_UNVERIFIABLE
    )

    # Dispatch, compile, VRAM, dispersion, lower_is_better=False
    no_nc = _rec(git_sha=SHA_B)
    del no_nc.metrics["n_compilations"]
    assert (
        compare_probe_pair(b, no_nc, hold_fixed_config=()).status
        is VerdictStatus.REJECT_UNVERIFIABLE
    )
    no_ne = _rec(git_sha=SHA_B)
    del no_ne.metrics["n_executions"]
    assert (
        compare_probe_pair(b, no_ne, hold_fixed_config=()).status
        is VerdictStatus.REJECT_UNVERIFIABLE
    )

    b_comp = _rec(git_sha=SHA_A, metrics={"n_compilations": 1.0})
    c_comp = _rec(git_sha=SHA_B, metrics={"total_step_seconds": 0.5, "n_compilations": 1.0})
    assert (
        compare_probe_pair(b_comp, c_comp, hold_fixed_config=()).status
        is VerdictStatus.REJECT_DISPATCH
    )
    assert (
        compare_probe_pair(b_comp, c_comp, hold_fixed_config=(), max_compile_increase=1.0).status
        is VerdictStatus.KEEP
    )
    assert (
        compare_probe_pair(
            b,
            _rec(git_sha=SHA_B, metrics={"total_step_seconds": 0.5, "n_executions": 50.0}),
            hold_fixed_config=(),
        ).status
        is VerdictStatus.REJECT_DISPATCH
    )

    no_vram = _rec(git_sha=SHA_B, metrics={"total_step_seconds": 0.5})
    del no_vram.metrics["peak_vram_bytes"]
    assert (
        compare_probe_pair(b, no_vram, hold_fixed_config=()).status
        is VerdictStatus.REJECT_UNVERIFIABLE
    )
    assert (
        compare_probe_pair(
            b,
            _rec(git_sha=SHA_B, metrics={"total_step_seconds": 0.5, "peak_vram_bytes": 0.0}),
            hold_fixed_config=(),
        ).status
        is VerdictStatus.REJECT_UNVERIFIABLE
    )
    assert (
        compare_probe_pair(
            b,
            _rec(git_sha=SHA_B, metrics={"total_step_seconds": 0.5, "wall_mad_seconds": -0.1}),
            hold_fixed_config=(),
        ).status
        is VerdictStatus.REJECT_UNVERIFIABLE
    )

    # Zero MAD -> noise=0, IQR fallback, absent dispersion note, lower_is_better=False
    b_zero_mad = _rec(git_sha=SHA_A, metrics={"total_step_seconds": 1.0, "wall_mad_seconds": 0.0})
    c_zero_mad = _rec(git_sha=SHA_B, metrics={"total_step_seconds": 0.5, "wall_mad_seconds": 0.0})
    v_zmad = compare_probe_pair(b_zero_mad, c_zero_mad, hold_fixed_config=())
    assert v_zmad.status is VerdictStatus.KEEP and v_zmad.noise_floor_x == pytest.approx(1.0)

    b_iqr = _rec(git_sha=SHA_A, metrics={"total_step_seconds": 1.0, "wall_iqr_seconds": 0.1})
    c_iqr = _rec(git_sha=SHA_B, metrics={"total_step_seconds": 0.5, "wall_iqr_seconds": 0.05})
    del b_iqr.metrics["wall_mad_seconds"]
    del c_iqr.metrics["wall_mad_seconds"]
    v_iqr = compare_probe_pair(b_iqr, c_iqr, hold_fixed_config=())
    assert v_iqr.status is VerdictStatus.KEEP and v_iqr.noise_floor_x > 1.0
    assert "wall_dispersion_absent_using_min_speedup_floor" not in v_iqr.reasons

    b_nomad = _rec(git_sha=SHA_A, metrics={"throughput": 100.0})
    c_nomad = _rec(git_sha=SHA_B, metrics={"throughput": 150.0})
    del b_nomad.metrics["wall_mad_seconds"]
    del c_nomad.metrics["wall_mad_seconds"]
    v_tp = compare_probe_pair(
        b_nomad,
        c_nomad,
        hold_fixed_config=(),
        primary_metric="throughput",
        lower_is_better=False,
    )
    assert v_tp.status is VerdictStatus.KEEP and v_tp.speedup_x == pytest.approx(1.5)

    del b_nomad.metrics["wall_iqr_seconds"]
    del c_nomad.metrics["wall_iqr_seconds"]
    v_nodisp = compare_probe_pair(b_nomad, c_nomad, hold_fixed_config=())
    assert "wall_dispersion_absent_using_min_speedup_floor" in v_nodisp.reasons

    # Work retention ratio drop -> REJECT_PARITY; short SHA match & dirty allow_dirty
    from xtrax.profiling.compare import _shas_match

    assert _shas_match(SHA_A[:8], SHA_A) is True
    assert _shas_match(f"{SHA_A}-dirty", f"{SHA_A[:8]}-dirty") is True
    assert _shas_match("abc", SHA_A) is False

    v_dirty_ok = compare_probe_pair(
        _rec(git_sha=f"{SHA_A}-dirty"),
        _rec(git_sha=f"{SHA_B}-dirty", metrics={"total_step_seconds": 0.5}),
        hold_fixed_config=(),
        allow_dirty=True,
    )
    assert v_dirty_ok.status is VerdictStatus.ADVISORY_ONLY

    assert (
        compare_probe_pair(
            b,
            _rec(git_sha=SHA_B, metrics={"total_step_seconds": 0.5, "work_active_count": 50.0}),
            hold_fixed_config=(),
        ).status
        is VerdictStatus.REJECT_PARITY
    )

    # Ratchet pointer error cases
    cand_json = tmp_path / "cand_r.json"
    c.write(cand_json)
    unverif = compare_probe_pair(
        b,
        _rec(stage=2, platform="gpu", git_sha=SHA_B, x64_enabled=True),
        hold_fixed_config=(),
    )
    assert unverif.status is VerdictStatus.REJECT_UNVERIFIABLE
    with pytest.raises(ClaimValidityError, match="rejected_dir is required"):
        update_baseline_pointer(tmp_path / "ptr.json", unverif, cand_json)

    ptr_bad = tmp_path / "bad_ptr.json"
    ptr_bad.write_text("{corrupt")
    with pytest.raises(ClaimValidityError, match="unreadable"):
        update_baseline_pointer(ptr_bad, v_tp, cand_json)

    ptr_list = tmp_path / "list_ptr.json"
    ptr_list.write_text("[]")
    with pytest.raises(ClaimValidityError, match="JSON object"):
        update_baseline_pointer(ptr_list, v_tp, cand_json)

    ptr_mismatch = tmp_path / "mism_ptr.json"
    ptr_mismatch.write_text(json.dumps({"probe_id": "other", "git_sha": SHA_A}))
    with pytest.raises(ClaimValidityError, match="continuity broken"):
        update_baseline_pointer(ptr_mismatch, v_tp, cand_json)

    with pytest.raises(ClaimValidityError, match="provenance"):
        update_baseline_pointer(
            tmp_path / "p2.json",
            dataclasses.replace(v_tp, candidate_probe_id="wrong"),
            cand_json,
        )


def test_kernel_probe_ptx_dual_pass_and_precision_errors(tmp_path, monkeypatch):
    import importlib
    import shutil

    from xtrax.profiling import kernel_probe as kp

    # PTX balanced delimiter / unsized extern shared / summarize empty & optional fields
    assert kp._extract_entry_body("(.param .u64 x", 0) is None
    assert kp._extract_entry_body("() no brace", 0) is None
    assert kp._extract_entry_body("() { unclosed", 0) is None
    lmem, smem, dyn = kp._parse_mem_bytes(".extern .shared .b8 dyn[]; .local .f32 arr[4];")
    assert lmem == 16 and smem == 0 and dyn is True
    empty_sum = summarize_ptx_metrics([])
    assert empty_sum["ptx_kernel_count"] == 0.0
    stat = parse_ptx_text(".entry foo() { .reg .f32 %f<2>; ret; }")[0]
    multi_reg = parse_ptx_text(".entry k_multi() { .reg .f32 %f<4>; .reg .f64 %fd<2>; ret; }")[0]
    assert multi_reg.vregs_by_type == {"f32": 4, "f64": 2} and multi_reg.f64_vregs == 2
    rich_sum = summarize_ptx_metrics(
        [
            dataclasses.replace(
                stat,
                phys_regs=48,
                spill_store_bytes=8,
                spill_load_bytes=4,
                triton_spills=2,
            )
        ]
    )
    assert rich_sum["ptxas_regs_max"] == 48.0
    assert rich_sum["ptxas_spill_bytes_max"] == 12.0
    assert rich_sum["triton_spills_max"] == 2.0

    # merge_xla_flags space-separated value & trailing flag
    merged = merge_xla_flags(
        "--xla_gpu_enable_command_buffer foo --keep_flag=1 --xla_dump_to /tmp/d",
        {"--xla_gpu_enable_command_buffer": "BAR"},
    )
    assert "--keep_flag=1" in merged and "--xla_dump_to" not in merged

    # compute_capability_to_sm
    assert compute_capability_to_sm("12.0") == "sm_120"
    with pytest.raises(ValueError):
        compute_capability_to_sm("not_cc")

    # find_ptxas search paths via monkeypatch
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    fake_cuda = tmp_path / "cuda_home"
    (fake_cuda / "bin").mkdir(parents=True)
    ptxas_exe = fake_cuda / "bin" / "ptxas"
    ptxas_exe.write_text("#!/bin/sh\n")
    monkeypatch.setenv("CUDA_HOME", str(fake_cuda))
    assert find_ptxas() == str(ptxas_exe)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    orig_import = importlib.import_module

    def _imp_err(name):
        if name == "nvidia.cuda_nvcc":
            raise ImportError("missing")
        return orig_import(name)

    monkeypatch.setattr(importlib, "import_module", _imp_err)
    # Only None if /usr/local/cuda/bin/ptxas does not exist on host
    res_find = find_ptxas()
    assert res_find is None or Path(res_find).is_file()
    monkeypatch.undo()

    # parse_ptxas_verbose warnings and multi-kernel enrichment + collect_pass_c_ptx_stats
    verbose_log = "\n".join(
        [
            "ptxas info : Compiling entry function 'k_a' for 'sm_89'",
            "ptxas info : Function properties for k_a",
            "    128 bytes stack frame, 16 bytes spill stores, 8 bytes spill loads",
            "ptxas info : Used 72 registers, 4096 bytes smem",
        ]
    )
    pa = parse_ptxas_verbose(verbose_log)["k_a"]
    assert pa.phys_regs == 72 and pa.spill_store_bytes == 16 and pa.smem_bytes == 4096

    t_dump = tmp_path / "tdump"
    t_cache = tmp_path / "tcache"
    x_dump = tmp_path / "xdump"
    t_dump.mkdir()
    t_cache.mkdir()
    x_dump.mkdir()
    (t_dump / "k_a.ptx").write_text(".entry k_a() { .reg .f32 %f<4>; ret; }")
    (t_dump / "k_a.cubin").write_bytes(b"cubin")
    (t_dump / "k_a.json").write_text(json.dumps({"name": "k_a", "n_spills": 3, "n_regs": 64}))
    (t_cache / "stale_ka.json").write_text(
        json.dumps({"name": "k_a", "n_spills": 99, "n_regs": 99})
    )
    # Duplicate PTX inside x_dump exercises dedup skipping; extra subdir tests missing-dir skip
    (x_dump / "dup_ka.ptx").write_text(".entry k_a() { .reg .f32 %f<4>; ret; }")
    monkeypatch.setattr(kp, "find_ptxas", lambda: "/fake/ptxas")

    class _FakeProc:
        def __init__(self, rc, stderr=""):
            self.returncode = rc
            self.stdout = ""
            self.stderr = stderr

    monkeypatch.setattr(
        kp.subprocess,
        "run",
        lambda *a, **kw: _FakeProc(0, verbose_log),
    )
    art = collect_pass_c_ptx_stats(
        t_dump,
        t_cache,
        x_dump,
        executed_kernels=("other_name",),
        compute_capability="8.9",
    )
    assert art.filter_status == "unfiltered_name_mismatch"
    assert art.dedup_count >= 1
    assert art.spill_source == "ptxas" and art.stats[0].triton_spills == 3

    art_matched = collect_pass_c_ptx_stats(
        t_dump,
        t_cache,
        tmp_path / "missing_xdump",
        executed_kernels=("k_a",),
        compute_capability="bad_cc",
    )
    assert art_matched.filter_status == "matched"
    assert art_matched.spill_source == "triton_meta"
    assert art_matched.warnings

    monkeypatch.setattr(kp, "find_ptxas", lambda: None)
    art_noptxas = collect_pass_c_ptx_stats(t_dump, t_cache, x_dump, compute_capability="8.9")
    assert any("not found" in w for w in art_noptxas.warnings)

    monkeypatch.setattr(kp.subprocess, "run", lambda *a, **kw: _FakeProc(1, "err"))
    bad_parsed, bad_warns = kp._run_ptxas_on_files([t_dump / "k_a.ptx"], "sm_89", "/fake/ptxas")
    assert bad_parsed == {} and bad_warns

    def _raise_timeout(*_a, **_kw):
        raise subprocess.TimeoutExpired(cmd="ptxas", timeout=1)

    monkeypatch.setattr(kp.subprocess, "run", _raise_timeout)
    to_parsed, to_warns = kp._run_ptxas_on_files([t_dump / "k_a.ptx"], "sm_89", "/fake/ptxas")
    assert to_parsed == {} and any("invocation error" in w.lower() for w in to_warns)

    def _raise_oserr(*_a, **_kw):
        raise OSError("missing binary")

    monkeypatch.setattr(kp.subprocess, "run", _raise_oserr)
    os_parsed, os_warns = kp._run_ptxas_on_files([t_dump / "k_a.ptx"], "sm_89", "/fake/ptxas")
    assert os_parsed == {} and any("invocation error" in w.lower() for w in os_warns)

    # Same .entry name across two distinct PTX specializations retains separate ptxas stats;
    # Triton metadata containing only 'shared' does not claim spill_source='triton_meta'.
    t_dup = tmp_path / "tdup"
    t_dup.mkdir()
    (t_dup / "k_a.ptx").write_text(".entry k_same() { .reg .f32 %f<4>; ret; }")
    (t_dup / "k_b.ptx").write_text(".entry k_same() { .reg .f32 %f<8>; ret; }")

    def _ptxas_per_file(cmd, **_kw):
        regs = 48 if str(cmd[-1]).endswith("k_a.ptx") else 64
        log = (
            "ptxas info : Compiling entry function 'k_same' for 'sm_89'\n"
            f"ptxas info : Used {regs} registers, 0 bytes smem\n"
        )
        return _FakeProc(0, log)

    monkeypatch.setattr(kp, "find_ptxas", lambda: "/fake/ptxas")
    monkeypatch.setattr(kp.subprocess, "run", _ptxas_per_file)
    art_dup = collect_pass_c_ptx_stats(
        t_dup, tmp_path / "empty_c", tmp_path / "empty_x", compute_capability="8.9"
    )
    assert sorted(s.phys_regs for s in art_dup.stats if s.phys_regs is not None) == [48, 64]

    t_shared = tmp_path / "tshared"
    t_shared.mkdir()
    (t_shared / "k_sh.ptx").write_text(".entry k_sh() { .reg .f32 %f<2>; ret; }")
    (t_shared / "k_sh.json").write_text(json.dumps({"name": "k_sh", "shared": 128}))
    art_shared = collect_pass_c_ptx_stats(
        t_shared, tmp_path / "empty_c", tmp_path / "empty_x", compute_capability="bad_cc"
    )
    assert art_shared.spill_source == "ptx_only"

    # _apply_parent_gpu_preallocate honors user override and injects false when parent has jax
    env_empty: dict[str, str] = {}
    monkeypatch.setitem(sys.modules, "jax", jax)
    kp._apply_parent_gpu_preallocate(env_empty, isolate_passes=True)
    assert env_empty["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    env_explicit = {"XLA_PYTHON_CLIENT_PREALLOCATE": "true"}
    kp._apply_parent_gpu_preallocate(env_explicit, isolate_passes=True)
    assert env_explicit["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
    monkeypatch.undo()

    # Exercise mocked Pass A isolated + Pass B + Pass C + hlo_attribution_skipped end-to-end
    worker_envs: list[dict[str, str]] = []
    worker_cmds: list[list[str]] = []
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)

    def _mock_worker(cmd, *, env):
        worker_cmds.append(list(cmd))
        worker_envs.append(dict(env))
        res_file = Path(cmd[cmd.index("--result-json") + 1])
        mode_val = cmd[cmd.index("--mode") + 1]
        if mode_val == "prod":
            payload = {
                "mode": "prod",
                "platform": "cpu",
                "git_sha": SHA_A,
                "metrics": {
                    "n_executions": 1.0,
                    "n_compilations": 0.0,
                    "n_jit_traces": 1.0,
                    "total_step_seconds": 0.01,
                },
            }
        elif mode_val == "diag":
            payload = {
                "mode": "diag",
                "platform": "cpu",
                "git_sha": SHA_A,
                "hlo_attribution_skipped": "unjittable_wrapper",
                "scopes": {"k_a": [0.005, 1]},
                "attribution_method": {"k_a": "op_name"},
                "executed_triton_kernels": ["k_a"],
                "compute_capability": "8.9",
                "metrics": {"total_step_seconds": 0.02},
            }
        else:
            payload = {"mode": "ptx", "ok": True, "compute_capability": "8.9"}
        res_file.write_text(json.dumps(payload), encoding="utf-8")
        return 0, "", ""

    monkeypatch.setattr(kp, "_run_worker_subprocess", _mock_worker)
    monkeypatch.setattr(
        kp,
        "collect_pass_c_ptx_stats",
        lambda *_a, **_kw: kp._PassCArtifacts(
            stats=(stat,),
            dedup_count=0,
            filter_status="matched",
            spill_source="ptx_only",
            warnings=(),
        ),
    )
    mocked_dual = probe_kernel_dual_pass(
        lambda x: x,
        (jnp.zeros((2,)),),
        probe_id="mock-full",
        n_atoms=4,
        stage=1,
        fn_spec="m:f",
        shapes="x=(2,)f32",
        prod_path=tmp_path / "mock_prod.json",
        diag_dir=tmp_path / "mock_diags",
        isolate_passes=True,
    )
    assert mocked_dual.hlo_attribution_skipped == "unjittable_wrapper"
    assert mocked_dual.diag_record is not None
    assert mocked_dual.diag_record.metrics["ptx_kernel_count"] == 1.0
    assert mocked_dual.diag_record.config["hlo_attribution_skipped"] == "unjittable_wrapper"
    assert worker_envs and all(
        e.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "false" for e in worker_envs
    )
    assert all("--extent-axis" not in c for c in worker_cmds)

    worker_envs.clear()
    worker_cmds.clear()
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    mocked_explicit_env = probe_kernel_dual_pass(
        lambda x: x,
        (jnp.zeros((2,)),),
        probe_id="mock-explicit-env",
        n_atoms=4,
        stage=1,
        fn_spec="m:f",
        shapes="x=(2,)f32",
        extent_axis="x",
        prod_path=None,
        diag_dir=tmp_path / "mock_diags_2",
        isolate_passes=True,
    )
    assert mocked_explicit_env.prod_record.config["extent_axis"] == "x"
    assert worker_envs and all(
        e.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "true" for e in worker_envs
    )
    assert all("--extent-axis" not in c for c in worker_cmds)
    monkeypatch.undo()

    # probe_kernel_dual_pass skip modes & subprocess failure/success mocking
    no_spec = probe_kernel_dual_pass(
        lambda x: x,
        (jnp.zeros((2,)),),
        probe_id="no-spec",
        n_atoms=2,
        stage=1,
        n_steady=1,
        n_wall=1,
        isolate_passes=False,
    )
    assert no_spec.diag_skipped_reason == "no_importable_spec"
    assert no_spec.ptx_capture_status == "disabled"

    disabled = probe_kernel_dual_pass(
        lambda x: x,
        (jnp.zeros((2,)),),
        probe_id="diag-off",
        n_atoms=2,
        stage=1,
        fn_spec="m:f",
        shapes="x=(2,)f32",
        n_steady=1,
        n_wall=1,
        run_diag=False,
        isolate_passes=False,
    )
    assert disabled.diag_skipped_reason == "diag_disabled"

    monkeypatch.setattr(kp, "_run_worker_subprocess", lambda *_a, **_kw: (1, "", "boom"))
    diag_fail = probe_kernel_dual_pass(
        lambda x: x,
        (jnp.zeros((2,)),),
        probe_id="diag-fail",
        n_atoms=2,
        stage=1,
        fn_spec="tests.cli._profile_fixtures:vmapped_twin",
        shapes="x=(2,2)f32",
        n_steady=1,
        n_wall=1,
        isolate_passes=False,
    )
    assert diag_fail.diag_skipped_reason == "diag_subprocess_failed:exit_1"
    assert diag_fail.ptx_capture_status == "skipped:diag_failed"

    with pytest.raises(RuntimeError, match="Pass A isolated worker failed"):
        probe_kernel_dual_pass(
            lambda x: x,
            (jnp.zeros((2,)),),
            probe_id="pass-a-fail",
            n_atoms=2,
            stage=1,
            fn_spec="tests.cli._profile_fixtures:vmapped_twin",
            shapes="x=(2,2)f32",
            n_steady=1,
            n_wall=1,
            isolate_passes=True,
        )
    monkeypatch.undo()

    # compute_precision_ratio unanimity and error guards
    s_ok = _rec(stage=1, platform="cpu", device_kind=None, x64_enabled=False, git_sha=SHA_A)
    m_ok = _rec(stage=1, platform="cpu", device_kind=None, x64_enabled=True, git_sha=SHA_A)
    assert compute_precision_ratio(s_ok, m_ok) == pytest.approx(1.0)

    with pytest.raises(ClaimValidityError, match="contract_version"):
        compute_precision_ratio(dataclasses.replace(s_ok, contract_version="99.0"), m_ok)
    with pytest.raises(ClaimValidityError, match="stage >= 1"):
        compute_precision_ratio(
            dataclasses.replace(s_ok, stage=0), dataclasses.replace(m_ok, stage=0)
        )
    with pytest.raises(ClaimValidityError, match="GPU precision pair requires stage >= 2"):
        compute_precision_ratio(_rec(stage=1, x64_enabled=False), _rec(stage=1, x64_enabled=True))
    with pytest.raises(ClaimValidityError, match="stage mismatch"):
        compute_precision_ratio(
            _rec(stage=2, platform="gpu", device_kind="l40s", x64_enabled=False, git_sha=SHA_A),
            _rec(stage=3, platform="gpu", device_kind="l40s", x64_enabled=True, git_sha=SHA_A),
        )
    with pytest.raises(ClaimValidityError, match="diagnostic"):
        compute_precision_ratio(
            dataclasses.replace(s_ok, config={**s_ok.config, "diagnostic_mode": "cmdbuf_off"}),
            m_ok,
        )
    with pytest.raises(ClaimValidityError, match="git_sha mismatch"):
        compute_precision_ratio(s_ok, dataclasses.replace(m_ok, git_sha=SHA_B))
    with pytest.raises(ClaimValidityError, match="not unanimous on 'platform'"):
        compute_precision_ratio(
            s_ok, _rec(stage=1, platform="gpu", device_kind=None, x64_enabled=True)
        )
    with pytest.raises(ClaimValidityError, match="step_time_basis mismatch"):
        compute_precision_ratio(
            s_ok,
            dataclasses.replace(m_ok, config={**m_ok.config, "step_time_basis": "per_replica"}),
        )
    with pytest.raises(ClaimValidityError, match="missing hold_fixed"):
        compute_precision_ratio(s_ok, m_ok, hold_fixed_config=("absent_k",))
    with pytest.raises(ClaimValidityError, match="mismatch on 'fn'"):
        compute_precision_ratio(
            s_ok,
            dataclasses.replace(m_ok, config={**m_ok.config, "fn": "other:step"}),
            hold_fixed_config=("fn",),
        )
    with pytest.raises(ClaimValidityError, match="x64_enabled=True"):
        compute_precision_ratio(s_ok, dataclasses.replace(m_ok, x64_enabled=False))
    with pytest.raises(ClaimValidityError, match="config\\['precision'\\]"):
        compute_precision_ratio(dataclasses.replace(s_ok, x64_enabled=True), m_ok)
    with pytest.raises(ClaimValidityError, match="missing from pair"):
        compute_precision_ratio(s_ok, m_ok, metric="no_such_metric")
    with pytest.raises(ClaimValidityError, match="finite > 0.0"):
        compute_precision_ratio(
            dataclasses.replace(s_ok, metrics={**s_ok.metrics, "total_step_seconds": 0.0}),
            m_ok,
        )
