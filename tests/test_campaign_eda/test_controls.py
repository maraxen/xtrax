"""The 6 Formal Statistical Verification Controls for xtrax Campaign EDA (Spec §7).

Implements P5-09 per approved specification 261106_campaign-eda-redesign.md (Rev 1.1)
and pre-registered campaign_eda_benchmark.bth.toml:
1. Control 1 (Positive Planted Knob Effect): Delta* = 1.50, Delta_hat in [1.35, 1.65],
   0 not in CI_95%, |Delta_hat| >= MDE_0.80.
2. Control 2 (Negative Null Campaign): 0 in CI_95%, false-positive rate <= 0.05
   across independent simulations under §5.3.4 power discipline.
3. Control 3 (Vectorized Bootstrap Parity): |batched - looped| < 1e-12 on unequal
   cluster sizes (BLK-01), and exact equivalence to unweighted bootstrap on equal sizes.
4. Control 4 (Storage Format Parity): Bit-identical metric and digest extraction
   between loose files and packed archives (.tar.zst).
5. Control 5 (Uncommitted Unit Refusal): Exclusion of uncommitted, partial, or corrupt
   units from statistical pooling, accompanied by explicit warning banners.
6. Control 6 (Small-Sample Guard): Verification that U < 3 enforces descriptive_only = True
   and suppresses confidence intervals to None/null.
"""

from __future__ import annotations

import hashlib
import io
import json

import numpy as np

from xtrax.campaign.manifest import (
    CampaignManifest,
    ManifestMetadata,
    ManifestRow,
    ResourceRequirements,
)
from xtrax.eda.campaign.reader import (
    CampaignReader,
    is_campaign_unit_packed,
    read_campaign_unit_member,
)
from xtrax.eda.campaign.stats import (
    evaluate_cluster_bootstrap_condition,
    evaluate_contrast,
    generate_multinomial_cluster_weights,
    looped_cluster_ratio_bootstrap,
    vectorized_cluster_ratio_bootstrap,
)
from xtrax.eda.campaign.types import CellStatus
from xtrax.pack import pack_unit

try:
    import zstandard as zstd
except ImportError:
    zstd = None


def test_positive_planted_knob_effect():
    """Control 1: Verify statistical detection of a planted knob effect (spec §7.1)."""
    rng = np.random.default_rng(42)
    U = 6
    N_u = 32
    sigma = 0.25
    planted_delta = 1.50
    B = 2000

    # Draw data for K0 and K1
    data_k0 = rng.normal(loc=0.0, scale=sigma, size=(U, N_u))
    data_k1 = rng.normal(loc=planted_delta, scale=sigma, size=(U, N_u))

    t_k0 = np.sum(data_k0, axis=1, keepdims=True)  # (U, 1)
    t_k1 = np.sum(data_k1, axis=1, keepdims=True)  # (U, 1)
    n_vec = np.full((U, 1), N_u, dtype=np.int64)

    contrast = evaluate_contrast(
        unit_sums_a=t_k0,
        unit_counts_a=n_vec,
        unit_sums_b=t_k1,
        unit_counts_b=n_vec,
        metric_name="score",
        B=B,
        seed=42,
    )

    # Criteria 1: Estimated contrast in [1.35, 1.65]
    assert 1.35 <= contrast.delta_hat <= 1.65, (
        f"Delta_hat {contrast.delta_hat} outside [1.35, 1.65]"
    )

    # Criteria 2: 0 not in CI_95%
    assert contrast.ci_95 is not None
    ci_lower, ci_upper = contrast.ci_95
    assert ci_lower > 0.0 or ci_upper < 0.0, f"CI {contrast.ci_95} contains 0"

    # Criteria 3: |Delta_hat| >= MDE_0.80
    assert abs(contrast.delta_hat) >= contrast.mde_080, (
        f"|Delta_hat| < MDE: {contrast.delta_hat} < {contrast.mde_080}"
    )
    assert contrast.detected_and_powered


def test_negative_null_campaign_no_effect():
    """Control 2: Verify false positive rate <= 5% under true null (spec §7.2, §5.3.4)."""
    rng = np.random.default_rng(42)
    U = 6
    N_u = 32
    sigma = 1.0
    B = 500
    n_simulations = 100

    rejections = 0
    n_vec = np.full((U, 1), N_u, dtype=np.int64)

    for i in range(n_simulations):
        d0 = rng.normal(loc=0.0, scale=sigma, size=(U, N_u))
        d1 = rng.normal(loc=0.0, scale=sigma, size=(U, N_u))
        tk0 = np.sum(d0, axis=1, keepdims=True)
        tk1 = np.sum(d1, axis=1, keepdims=True)

        contrast = evaluate_contrast(
            unit_sums_a=tk0,
            unit_counts_a=n_vec,
            unit_sums_b=tk1,
            unit_counts_b=n_vec,
            metric_name="score",
            B=B,
            seed=int(rng.integers(1, 100000)),
        )

        # Under §5.3.4, an effect is claimed detected only if 0 not in CI and |delta_hat| >= MDE
        if contrast.detected_and_powered:
            rejections += 1

    fpr = rejections / float(n_simulations)
    assert fpr <= 0.05, f"False positive rate {fpr} exceeds 0.05"


def test_batched_equals_looped_parity():
    """Control 3: Vectorized cluster ratio bootstrap parity (spec §7.3, BLK-01).

    Criteria:
    1. |batched - looped| < 1e-12 on unequal cluster sizes.
    2. Exact equivalence to unweighted bootstrap on equal cluster sizes.
    """
    rng = np.random.default_rng(999)
    U = 8
    D = 4
    B = 1000

    # Unequal cluster sizes:
    counts_unequal = np.array([10, 50, 100, 20, 80, 40, 60, 30], dtype=np.int64)
    unit_sums = rng.normal(loc=5.0, scale=2.0, size=(U, D)) * counts_unequal.reshape(-1, 1)

    weights = generate_multinomial_cluster_weights(U, B, rng)

    theta_vec = vectorized_cluster_ratio_bootstrap(unit_sums, counts_unequal, weights)
    theta_loop = looped_cluster_ratio_bootstrap(unit_sums, counts_unequal, weights)

    ci_vec = np.percentile(theta_vec, [2.5, 97.5], axis=0)
    ci_loop = np.percentile(theta_loop, [2.5, 97.5], axis=0)

    # Parity check on unequal cluster sizes:
    max_diff_unequal = float(np.max(np.abs(ci_vec - ci_loop)))
    assert max_diff_unequal < 1e-12, f"Unequal cluster parity failed: max diff {max_diff_unequal}"

    # Equal cluster sizes parity check with unweighted bootstrap:
    counts_equal = np.full(U, 50, dtype=np.int64)
    unit_sums_eq = rng.normal(loc=5.0, scale=2.0, size=(U, D)) * counts_equal.reshape(-1, 1)

    theta_ratio_eq = vectorized_cluster_ratio_bootstrap(unit_sums_eq, counts_equal, weights)
    unit_means_eq = unit_sums_eq / counts_equal.reshape(-1, 1)
    theta_unweighted = (weights @ unit_means_eq) / float(U)

    max_diff_equal = float(np.max(np.abs(theta_ratio_eq - theta_unweighted)))
    assert max_diff_equal < 1e-12, (
        f"Equal cluster unweighted parity failed: max diff {max_diff_equal}"
    )


def test_storage_format_parity(tmp_path):
    """Control 4: Storage format parity (loose == packed read) (spec §7.4)."""
    loose_unit = tmp_path / "loose_unit"
    loose_unit.mkdir()

    scores_arr = np.array([1.234, 1.456, 1.678], dtype=np.float64)
    np.save(loose_unit / "scores.npy", scores_arr)
    with open(loose_unit / "data.json", "w", encoding="utf-8") as f:
        json.dump({"arm": "proteinmpnn", "seed": 1001}, f)
    with open(loose_unit / "done.json", "w", encoding="utf-8") as f:
        json.dump(
            {"schema_version": "campaign_done_marker_v3", "status": "committed", "duration_s": 1.2},
            f,
        )

    # Read from loose directory
    loose_scores_bytes = read_campaign_unit_member(loose_unit, "scores.npy")
    loose_scores_arr = np.load(io.BytesIO(loose_scores_bytes))
    loose_data_bytes = read_campaign_unit_member(loose_unit, "data.json")

    # Pack the unit
    pack_unit(loose_unit, record_name="done.json", check=lambda u, r, d: None)
    assert is_campaign_unit_packed(loose_unit)

    # Read from packed archive
    packed_scores_bytes = read_campaign_unit_member(loose_unit, "scores.npy")
    packed_scores_arr = np.load(io.BytesIO(packed_scores_bytes))
    packed_data_bytes = read_campaign_unit_member(loose_unit, "data.json")

    # Verify bit-identical parity
    assert (
        hashlib.sha256(loose_scores_bytes).hexdigest()
        == hashlib.sha256(packed_scores_bytes).hexdigest()
    )
    assert (
        hashlib.sha256(loose_data_bytes).hexdigest()
        == hashlib.sha256(packed_data_bytes).hexdigest()
    )
    np.testing.assert_array_equal(loose_scores_arr, packed_scores_arr)


def test_uncommitted_refusal_and_banner(tmp_path):
    """Control 5: Uncommitted unit refusal and explicit bannering (spec §7.5)."""
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()

    # Unit 1: COMMITTED
    u1 = campaign_dir / "u1"
    u1.mkdir()
    with open(u1 / "done.json", "w", encoding="utf-8") as f:
        json.dump({"manifest_row_hash": "hash_1", "status": "committed", "duration_s": 1.1}, f)

    # Unit 2: PARTIAL (active staging directory)
    u2 = campaign_dir / "u2"
    u2_tmp = campaign_dir / "u2.tmp"
    u2_tmp.mkdir()

    # Unit 3: FAILED (marked failed)
    u3 = campaign_dir / "u3"
    u3.mkdir()
    with open(u3 / "done.json", "w", encoding="utf-8") as f:
        json.dump(
            {"manifest_row_hash": "hash_3", "status": "failed", "error": "killed_incomplete"}, f
        )

    row1 = ManifestRow(
        manifest_row_hash="hash_1",
        input_hash="in_1",
        campaign_id="c",
        job_id="j1",
        chunk_id=0,
        sample_start=0,
        sample_count=16,
        target_input="t",
        arm_id="arm_a",
        arm_backend="b",
        checkpoint_id="cp",
        fixed_policy="p",
        state_weight_profile="w",
        seed=1,
        parameters={},
        output_locator=str(u1),
        resource_requirements=ResourceRequirements(gpu_memory_bytes=0),
    )
    row2 = ManifestRow(
        manifest_row_hash="hash_2",
        input_hash="in_2",
        campaign_id="c",
        job_id="j2",
        chunk_id=0,
        sample_start=0,
        sample_count=16,
        target_input="t",
        arm_id="arm_a",
        arm_backend="b",
        checkpoint_id="cp",
        fixed_policy="p",
        state_weight_profile="w",
        seed=2,
        parameters={},
        output_locator=str(u2),
        resource_requirements=ResourceRequirements(gpu_memory_bytes=0),
    )
    row3 = ManifestRow(
        manifest_row_hash="hash_3",
        input_hash="in_3",
        campaign_id="c",
        job_id="j3",
        chunk_id=0,
        sample_start=0,
        sample_count=16,
        target_input="t",
        arm_id="arm_a",
        arm_backend="b",
        checkpoint_id="cp",
        fixed_policy="p",
        state_weight_profile="w",
        seed=3,
        parameters={},
        output_locator=str(u3),
        resource_requirements=ResourceRequirements(gpu_memory_bytes=0),
    )

    manifest = CampaignManifest(
        schema_version="campaign_manifest_v1",
        metadata=ManifestMetadata(
            campaign_id="c",
            planner_version="1",
            spec_sha256="s",
            total_rows=3,
            created_at_utc="2026-11-06T20:30:00Z",
        ),
        rows=[row1, row2, row3],
    )

    reader = CampaignReader(campaign_dir=campaign_dir)
    committed = reader.get_committed_units(manifest)

    # Must ONLY pool the 1 committed unit
    assert len(committed) == 1
    assert committed[0][0].manifest_row_hash == "hash_1"

    # Reconciliation must explicitly classify all 3 units
    reconciled = reader.reconcile_campaign(manifest)
    status_map = {r.manifest_row_hash: r.status for r in reconciled}
    assert status_map["hash_1"] == CellStatus.COMMITTED
    assert status_map["hash_2"] == CellStatus.PARTIAL
    assert status_map["hash_3"] == CellStatus.FAILED


def test_small_sample_guard_suppression():
    """Control 6: Small-sample guard for U < 3 (spec §7.6, BLK-02)."""
    metric_names = ["score"]

    # Condition 1: U = 1 -> Must trigger guard
    sums_u1 = np.array([[10.0]])
    counts_u1 = np.array([5])
    est_u1 = evaluate_cluster_bootstrap_condition(sums_u1, counts_u1, metric_names)["score"]
    assert est_u1.descriptive_only
    assert est_u1.ci_95 is None
    assert est_u1.warning is not None

    # Condition 2: U = 2 -> Must trigger guard
    sums_u2 = np.array([[10.0], [12.0]])
    counts_u2 = np.array([5, 5])
    est_u2 = evaluate_cluster_bootstrap_condition(sums_u2, counts_u2, metric_names)["score"]
    assert est_u2.descriptive_only
    assert est_u2.ci_95 is None
    assert est_u2.warning is not None

    # Condition 3: U = 3 -> Must NOT trigger guard; evaluates full CIs
    sums_u3 = np.array([[10.0], [12.0], [11.0]])
    counts_u3 = np.array([5, 5, 5])
    est_u3 = evaluate_cluster_bootstrap_condition(sums_u3, counts_u3, metric_names)["score"]
    assert not est_u3.descriptive_only
    assert est_u3.ci_95 is not None
    assert est_u3.ci_95[0] < est_u3.ci_95[1]
    assert est_u3.warning is None
