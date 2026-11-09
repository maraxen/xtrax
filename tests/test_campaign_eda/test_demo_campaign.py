"""End-to-End Ubiquitin Demonstration Campaign Test (Spec §8, P5-10).

Validates full 18-unit aminx Ubiquitin campaign:
- 2 arms ('proteinmpnn', 'solublempnn')
- 3 temperatures (0.1, 0.3, 0.5)
- 3 replication seeds (1001, 1002, 1003; U=3 >= 3)
- 288 generated sequences
- Target: Ubiquitin (tests/fixtures/1ubq.pdb, 76 residues)
- Answering all 5 scientific questions within <= 60.0 seconds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from xtrax.campaign.manifest import (
    CampaignManifest,
    ManifestMetadata,
    ManifestRow,
    ResourceRequirements,
)
from xtrax.eda.campaign import run_campaign_eda
from xtrax.pack import pack_unit

UBIQUITIN_WT = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


def _simulate_ubiquitin_seq(rng: np.random.Generator, arm: str, temp: float) -> str:
    aa_alphabet = list("ACDEFGHIKLMNPQRSTVWY")
    seq_len = len(UBIQUITIN_WT)
    base_mutation_rate = 0.05 + 0.35 * temp
    if arm == "solublempnn":
        soluble_alphabet = list("DEGHKNQRST")
        weights = [1.5 if aa in soluble_alphabet else 0.8 for aa in aa_alphabet]
        pvals = np.array(weights) / sum(weights)
    else:
        pvals = [1.0 / len(aa_alphabet)] * len(aa_alphabet)

    seq_chars = list(UBIQUITIN_WT)
    for i in range(seq_len):
        if rng.uniform(0.0, 1.0) < base_mutation_rate:
            seq_chars[i] = rng.choice(aa_alphabet, p=pvals)
    return "".join(seq_chars)


def test_ubiquitin_demo_campaign_e2e(tmp_path):
    """Execute and validate end-to-end Ubiquitin demo campaign."""
    start_time = time.time()

    campaign_dir = tmp_path / "ubiquitin_demo_campaign"
    units_dir = campaign_dir / "units"
    units_dir.mkdir(parents=True)

    pdb_fixture = Path("tests/fixtures/1ubq.pdb").resolve()
    assert pdb_fixture.exists(), "Ubiquitin PDB fixture tests/fixtures/1ubq.pdb must exist"

    arms = ["proteinmpnn", "solublempnn"]
    temperatures = [0.1, 0.3, 0.5]
    seeds = [1001, 1002, 1003]
    samples_per_unit = 16

    manifest_rows: list[ManifestRow] = []
    unit_idx = 0

    for arm in arms:
        for temp in temperatures:
            for seed in seeds:
                unit_idx += 1
                row_hash = f"ubq_unit_{unit_idx:03d}_{arm}_t{temp}_s{seed}"
                unit_dir = units_dir / row_hash
                unit_dir.mkdir()

                unit_rng = np.random.default_rng(seed + int(temp * 1000))
                sequences = [
                    _simulate_ubiquitin_seq(unit_rng, arm, temp) for _ in range(samples_per_unit)
                ]
                base_nll = 1.20 if arm == "proteinmpnn" else 1.35
                scores = unit_rng.normal(
                    loc=base_nll + 0.6 * temp,
                    scale=0.15,
                    size=samples_per_unit,
                )
                duration_s = float(unit_rng.uniform(0.8, 1.8))

                # Write unit artifacts
                np.save(unit_dir / "sequences.npy", np.array(sequences))
                np.save(unit_dir / "scores.npy", scores)
                done_marker = {
                    "schema_version": "campaign_done_marker_v3",
                    "manifest_row_hash": row_hash,
                    "input_hash": row_hash,
                    "status": "committed",
                    "duration_s": duration_s,
                    "completed_at_utc": "2026-11-06T20:30:00Z",
                }
                with open(unit_dir / "done.json", "w", encoding="utf-8") as f:
                    json.dump(done_marker, f)

                # Pack half of the units to exercise storage format parity on real demo data
                if unit_idx % 2 == 0:
                    pack_unit(unit_dir, record_name="done.json", check=lambda u, r, d: None)

                manifest_rows.append(
                    ManifestRow(
                        manifest_row_hash=row_hash,
                        input_hash=row_hash,
                        campaign_id="ubiquitin_demo",
                        job_id=f"job_{unit_idx:03d}",
                        chunk_id=0,
                        sample_start=0,
                        sample_count=samples_per_unit,
                        target_input=UBIQUITIN_WT,
                        arm_id=arm,
                        arm_backend="cpu",
                        checkpoint_id=f"{arm}_v_48_020",
                        fixed_policy="none",
                        state_weight_profile="uniform",
                        seed=seed,
                        parameters={"temperature": temp},
                        output_locator=str(unit_dir),
                        resource_requirements=ResourceRequirements(gpu_memory_bytes=0),
                    )
                )

    manifest = CampaignManifest(
        schema_version="campaign_manifest_v1",
        metadata=ManifestMetadata(
            campaign_id="ubiquitin_demo",
            planner_version="1.1.0",
            spec_sha256="ubq_spec_sha256",
            total_rows=len(manifest_rows),
            created_at_utc="2026-11-06T20:30:00Z",
        ),
        rows=manifest_rows,
    )

    with open(campaign_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    # Execute full Campaign EDA pipeline
    results = run_campaign_eda(
        campaign_dir=campaign_dir,
        manifest=manifest,
        domain_name="aminx",
        export_artifacts=True,
        B=500,  # Fast execution for test suite
    )

    elapsed_time = time.time() - start_time

    # Verification of all 5 scientific questions:
    # --------------------------------------------------------------------------
    # Q1: Library Composition
    # --------------------------------------------------------------------------
    q1_rows = results["q1_composition"]["rows"]
    assert len(q1_rows) == 18
    assert all(r["diversity"] >= 0.0 for r in q1_rows)
    assert all(r["positional_entropy"] >= 0.0 for r in q1_rows)
    assert all(r["composition_entropy"] >= 0.0 for r in q1_rows)
    assert all(0.0 <= r["recovery"] <= 1.0 for r in q1_rows)
    assert len(results["q1_composition"]["cross_knob_collisions"]) > 0

    # --------------------------------------------------------------------------
    # Q2: Scores & Energetics
    # --------------------------------------------------------------------------
    q2_rows = results["q2_scores"]["rows"]
    assert len(q2_rows) == 18
    assert all(r["nll"] > 0.0 for r in q2_rows)
    assert len(results["q2_scores"]["best_of_n_curve"]) > 0
    assert len(results["q2_scores"]["pareto_frontier"]) > 0

    # --------------------------------------------------------------------------
    # Q3: Honest Uncertainty
    # --------------------------------------------------------------------------
    q3_rows = results["q3_uncertainty"]["rows"]
    # 2 arms x 3 temperatures = 6 condition groups
    assert len(q3_rows) == 6
    # U = 3 >= 3, so descriptive_only must be False and CIs must be active
    assert all(r["U"] == 3 for r in q3_rows)
    assert all(not r["descriptive_only"] for r in q3_rows)
    assert all(r["ci_lower"] is not None and r["ci_upper"] is not None for r in q3_rows)
    assert all(r["ci_lower"] < r["ci_upper"] for r in q3_rows)

    # --------------------------------------------------------------------------
    # Q4: Coverage Reconciliation
    # --------------------------------------------------------------------------
    q4 = results["q4_coverage"]
    assert q4["total_planned_rows"] == 18
    assert q4["committed_rows"] == 18
    assert q4["partial_rows"] == 0
    assert q4["failed_rows"] == 0
    assert q4["missing_rows"] == 0
    assert q4["is_complete"] is True
    assert q4["committed_visual_cells"] == 6

    # --------------------------------------------------------------------------
    # Q5: Secondary Run Telemetry
    # --------------------------------------------------------------------------
    q5 = results["q5_telemetry"]
    assert q5["total_wall_seconds"] > 0.0
    assert q5["pack_telemetry_status"] == "instrumented"
    assert "Pack Telemetry: instrumented (9 units" in q5["pack_telemetry_banner"]
    assert q5["pack_telemetry"]["total_raw_bytes"] > 0
    assert q5["pack_telemetry"]["overall_compression_ratio"] > 1.0
    assert q5["packed_units"] == 9
    assert q5["loose_units"] == 9

    # --------------------------------------------------------------------------
    # Neutral Artifact Export Bundle
    # --------------------------------------------------------------------------
    assert len(results["artifacts"]) > 0
    eda_dir = campaign_dir / "eda"
    assert (eda_dir / "manifest.json").exists()
    assert (eda_dir / "summary.json").exists()
    assert (eda_dir / "index.html").exists()
    assert (eda_dir / "specs" / "library_composition.vl.json").exists()
    assert (eda_dir / "specs" / "score_distributions.vl.json").exists()
    assert (eda_dir / "specs" / "uncertainty_contrasts.vl.json").exists()
    assert (eda_dir / "specs" / "sweep_coverage.vl.json").exists()
    assert (eda_dir / "specs" / "run_telemetry.vl.json").exists()
    assert (eda_dir / "static" / "library_composition.svg").exists()
    assert (eda_dir / "static" / "score_distributions.svg").exists()
    assert (eda_dir / "static" / "uncertainty_contrasts.svg").exists()
    assert (eda_dir / "static" / "sweep_coverage.svg").exists()
    assert (eda_dir / "static" / "run_telemetry.svg").exists()

    # Budget assertion: execution must complete within <= 60.0s
    assert elapsed_time <= 60.0, f"Demo runtime {elapsed_time:.2f}s exceeded 60.0s budget"
