"""xtrax.eda.campaign — Multi-knob exploratory data analysis for experimental campaigns.

Answers five foundational scientific questions across experimental campaigns:
1. Library Composition (diversity, positional & composition entropy, native recovery)
2. Scores & Energetics (NLL distributions, Best-of-N curves, score-diversity Pareto frontiers)
3. Honest Uncertainty (global MD-replicas rule, cluster ratio bootstrap, MDE power checks)
4. Sweep Coverage Reconciliation (COMMITTED, PARTIAL, FAILED, MISSING sweep grid)
5. Secondary Run Telemetry (wall times, reuse audit trail, pack telemetry seam banner)
"""

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from xtrax.campaign.manifest import CampaignManifest, ManifestRow
from xtrax.eda.campaign.artifacts import (
    dumps_canonical_json,
    export_campaign_eda,
    sanitize_for_rfc8259,
)
from xtrax.eda.campaign.coverage import reconcile_sweep_grid
from xtrax.eda.campaign.plugins import (
    DomainPlugin,
    clear_quarantined_units,
    discover_domain_plugins,
    enforce_cell_immutability,
    execute_domain_plugin,
    get_quarantined_units,
    quarantine_unit,
    register_domain_plugin,
)
from xtrax.eda.campaign.reader import (
    CampaignReader,
    is_campaign_unit_packed,
    read_campaign_unit_member,
    read_done_marker,
)
from xtrax.eda.campaign.stats import (
    calculate_pairwise_diversity,
    compute_best_of_n,
    compute_composition_entropy,
    compute_cross_knob_collisions,
    compute_duplicate_rates,
    compute_mde_080,
    compute_pareto_frontier_2d,
    compute_positional_entropy,
    compute_target_recovery,
    evaluate_cluster_bootstrap_condition,
    evaluate_contrast,
    generate_multinomial_cluster_weights,
    looped_cluster_ratio_bootstrap,
    safe_shannon_entropy,
    vectorized_cluster_ratio_bootstrap,
)
from xtrax.eda.campaign.telemetry import (
    PACK_TELEMETRY_BANNER,
    extract_campaign_telemetry,
)
from xtrax.eda.campaign.types import (
    BootstrapEstimate,
    CellStatus,
    ContrastEstimate,
    DomainMetricResult,
    LibraryCell,
    QuarantineRecord,
    ReconciliationRecord,
    ResidueTable,
    SinkArray,
    SinkSchema,
)

logger = logging.getLogger("xtrax.eda.campaign")


def _build_condition_key(row: ManifestRow) -> str:
    """Build condition key from arm_id and all sweep knob dimensions (F-XTX-01 / F-MAJ-02).

    Prevents variations in checkpoints, noise, or policies from being falsely pooled
    as stochastic replicates.
    """
    parts = [str(row.arm_id)]
    if row.parameters:
        for k in sorted(row.parameters.keys()):
            if k in (
                "seed",
                "rng_seed",
                "job_id",
                "chunk_id",
                "sample_start",
                "sample_count",
                "replicate",
            ):
                continue
            v = row.parameters[k]
            parts.append(f"{k}:{v}")
    if row.checkpoint_id and row.checkpoint_id not in ("none", "default"):
        parts.append(f"chk:{row.checkpoint_id}")
    if row.fixed_policy and row.fixed_policy not in ("none", "default"):
        parts.append(f"pol:{row.fixed_policy}")
    if row.state_weight_profile and row.state_weight_profile not in ("none", "uniform", "default"):
        parts.append(f"swp:{row.state_weight_profile}")
    return "_".join(parts)


def _primary_score_metric(plugin: Any, metrics: Mapping[str, float]) -> str:
    """Determine primary score observable for the domain (F-BLK-07 / F-XTX-02)."""
    if hasattr(plugin, "primary_observable") and callable(plugin.primary_observable):
        try:
            obs = plugin.primary_observable()
            if isinstance(obs, str) and obs in metrics:
                return obs
        except Exception:
            pass
    elif hasattr(plugin, "primary_observable") and isinstance(plugin.primary_observable, str):
        if plugin.primary_observable in metrics:
            return plugin.primary_observable

    for cand in ("nll", "score_nll", "total_entropy", "mean_residue_entropy", "score", "energy"):
        if cand in metrics:
            return cand
    if metrics:
        return next(iter(metrics))
    return "nll"


def run_campaign_eda(
    campaign_dir: Path | str,
    manifest: CampaignManifest | None = None,
    domain_name: str = "aminx",
    export_artifacts: bool = True,
    B: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute end-to-end Campaign EDA answering all 5 scientific questions.

    Args:
        campaign_dir: Root directory of the campaign containing manifest and units.
        manifest: Pre-loaded CampaignManifest, or loaded from <campaign_dir>/manifest.json.
        domain_name: Consumer domain plugin identifier (e.g. "aminx", "demistify").
        export_artifacts: Whether to compile and export neutral file bundle to <campaign_dir>/eda/.
        B: Number of bootstrap iterations for cluster ratio bootstrap.
        seed: Random seed for statistical reproducibility.

    Returns:
        Structured dictionary containing Q1, Q2, Q3, Q4, Q5 results and exported file manifest.
    """
    c_dir = Path(campaign_dir).resolve()
    reader = CampaignReader(campaign_dir=c_dir)

    # 1. Load manifest if not provided
    if manifest is None:
        manifest_path = c_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Campaign manifest not found at {manifest_path}")
        with open(manifest_path, encoding="utf-8") as f:
            manifest_raw = json.load(f)
        manifest = CampaignManifest.from_dict(manifest_raw)

    # Discover and select domain plugin
    plugins = discover_domain_plugins()
    if domain_name not in plugins:
        # Load builtin domains if needed
        import xtrax.eda.campaign.domains as _domains

        _ = _domains
        plugins = discover_domain_plugins()
    plugin = plugins.get(domain_name)
    if plugin is None:
        raise ValueError(
            f"Domain plugin {domain_name!r} not found. Available: {list(plugins.keys())}"
        )

    # 2. Reconcile coverage (Q4)
    reconciled_records = reader.reconcile_campaign(manifest)
    q4_coverage = reconcile_sweep_grid(manifest, reconciled_records)

    # 3. Extract committed units only (Refusing uncommitted / partial / corrupt / quarantined)
    quarantined_keys = [q.unit_key for q in get_quarantined_units()]
    committed_units = reader.get_committed_units(manifest, quarantined_keys=quarantined_keys)

    # Propagate any existing quarantined units to coverage banners (MAJ-02, MAJ-04)
    for q_rec in get_quarantined_units():
        banner_record = {
            "unit_id": q_rec.unit_key,
            "type": "quarantined",
            "reason": q_rec.reason,
        }
        if banner_record not in q4_coverage.setdefault("banners", []):
            q4_coverage["banners"].append(banner_record)

    # 4. Process committed units via domain plugin within quarantine boundary
    cells_and_metrics: list[
        tuple[LibraryCell, DomainMetricResult, ManifestRow, dict[str, Any]]
    ] = []
    default_schema = SinkSchema(arrays={})

    for row, unit_dir, marker_data in committed_units:
        before_q_count = len(get_quarantined_units())
        extracted = execute_domain_plugin(plugin, unit_dir, row, default_schema)
        after_q = get_quarantined_units()
        if len(after_q) > before_q_count:
            for new_rec in after_q[before_q_count:]:
                banner_record = {
                    "unit_id": new_rec.unit_key,
                    "type": "quarantined",
                    "reason": new_rec.reason,
                }
                q4_coverage.setdefault("banners", []).append(banner_record)
        for cell, metric_res in extracted:
            cells_and_metrics.append((cell, metric_res, row, marker_data))

    # --------------------------------------------------------------------------
    # Q1: Library Composition
    # --------------------------------------------------------------------------
    q1_rows: list[dict[str, Any]] = []
    condition_seqs: dict[str, list[str]] = {}

    for cell, metric_res, row, _ in cells_and_metrics:
        cond_str = _build_condition_key(row)
        if cond_str not in condition_seqs:
            condition_seqs[cond_str] = []
        condition_seqs[cond_str].extend(cell.sequences)

        q1_dict = {
            "manifest_row_hash": row.manifest_row_hash,
            "condition": cond_str,
            "arm": row.arm_id,
            "seed": row.seed,
            "diversity": metric_res.metrics.get("diversity", 0.0),
            "positional_entropy": metric_res.metrics.get("positional_entropy", 0.0),
            "composition_entropy": metric_res.metrics.get("composition_entropy", 0.0),
            "recovery": metric_res.metrics.get("recovery", 0.0),
            "recovery_margin": metric_res.metrics.get("recovery_margin", 0.0),
            "duplicate_rate": metric_res.metrics.get("duplicate_rate", 0.0),
            "flags": metric_res.flags,
        }
        for mk, mv in metric_res.metrics.items():
            q1_dict[mk] = mv
        q1_rows.append(q1_dict)

    # Cross-knob collision table (MIN-01)
    collision_matrix = compute_cross_knob_collisions(condition_seqs)

    # --------------------------------------------------------------------------
    # Q2: Scores & Energetics
    # --------------------------------------------------------------------------
    q2_rows: list[dict[str, Any]] = []
    all_scores: list[float] = []
    pareto_points: list[tuple[float, float, str]] = []

    for cell, metric_res, row, _ in cells_and_metrics:
        cond_str = _build_condition_key(row)
        primary_metric = _primary_score_metric(plugin, metric_res.metrics)
        primary_val = metric_res.metrics.get(primary_metric, metric_res.metrics.get("nll", 0.0))
        nll_val = metric_res.metrics.get("nll", primary_val)

        q2_dict = {
            "manifest_row_hash": row.manifest_row_hash,
            "condition": cond_str,
            "arm": row.arm_id,
            "seed": row.seed,
            "primary_metric": primary_metric,
            "primary_value": primary_val,
            "nll": nll_val,
            "perplexity": metric_res.metrics.get("perplexity", 1.0),
        }
        for mk, mv in metric_res.metrics.items():
            q2_dict.setdefault(mk, mv)
        q2_rows.append(q2_dict)

        if cell.scores is not None and len(cell.scores) > 0:
            all_scores.extend([float(s) for s in cell.scores])
            pid_val = 1.0 - metric_res.metrics.get("diversity", 0.0)
            pareto_points.append((float(np.mean(cell.scores)), pid_val, cond_str))
        elif primary_metric in metric_res.metrics:
            val = float(metric_res.metrics[primary_metric])
            all_scores.append(val)
            pid_val = 1.0 - metric_res.metrics.get("diversity", 0.0)
            pareto_points.append((val, pid_val, cond_str))

    best_of_n_curve = compute_best_of_n(all_scores) if all_scores else {}
    pareto_frontier = compute_pareto_frontier_2d(pareto_points) if pareto_points else []

    # --------------------------------------------------------------------------
    # Q3: Honest Uncertainty (Cluster Ratio Bootstrap)
    # --------------------------------------------------------------------------
    q3_rows: list[dict[str, Any]] = []
    # Group cells by condition
    cond_groups: dict[str, list[tuple[LibraryCell, DomainMetricResult]]] = {}
    for cell, metric_res, row, _ in cells_and_metrics:
        cond_str = _build_condition_key(row)
        if cond_str not in cond_groups:
            cond_groups[cond_str] = []
        cond_groups[cond_str].append((cell, metric_res))

    for cond_str, items in cond_groups.items():
        U = len(items)
        first_metric_res = items[0][1]
        primary_metric = _primary_score_metric(plugin, first_metric_res.metrics)

        unit_sums = []
        unit_counts = []
        for cell, metric_res in items:
            if (
                cell.tokens is not None
                and hasattr(cell.tokens, "shape")
                and len(cell.tokens.shape) > 0
            ):
                n_samples = int(cell.tokens.shape[0])
            else:
                n_samples = len(cell.sequences)
            unit_counts.append(n_samples)
            metric_val = metric_res.metrics.get(primary_metric, metric_res.metrics.get("nll", 0.0))
            unit_sums.append([n_samples * metric_val])

        estimates = evaluate_cluster_bootstrap_condition(
            unit_sums=np.array(unit_sums),
            unit_counts=np.array(unit_counts),
            metric_names=[primary_metric],
            B=B,
            seed=seed,
        )
        est = estimates[primary_metric]
        q3_rows.append(
            {
                "condition": cond_str,
                "metric": primary_metric,
                "U": U,
                "point_estimate": est.point_estimate,
                "ci_lower": est.ci_95[0] if est.ci_95 else None,
                "ci_upper": est.ci_95[1] if est.ci_95 else None,
                "se": est.se,
                "mde_080": est.mde_080,
                "descriptive_only": est.descriptive_only,
                "warning": est.warning,
            }
        )

    # --------------------------------------------------------------------------
    # Q5: Secondary Run Telemetry
    # --------------------------------------------------------------------------
    reuse_log_path = c_dir / "reuse_log.jsonl"
    q5_telemetry = extract_campaign_telemetry(manifest, reconciled_records, reuse_log_path)

    # --------------------------------------------------------------------------
    # Neutral Artifact Export
    # --------------------------------------------------------------------------
    artifacts_map: dict[str, str] = {}
    campaign_id = manifest.metadata.campaign_id if manifest.metadata else c_dir.name

    if export_artifacts:
        artifacts_map = export_campaign_eda(
            campaign_dir=c_dir,
            campaign_id=campaign_id,
            q1_data=q1_rows,
            q2_data=q2_rows,
            q3_data=q3_rows,
            q4_coverage=q4_coverage,
            q5_telemetry=q5_telemetry,
        )

    return {
        "campaign_id": campaign_id,
        "q1_composition": {
            "rows": q1_rows,
            "cross_knob_collisions": {
                f"{k[0]}__vs__{k[1]}": v for k, v in collision_matrix.items()
            },
        },
        "q2_scores": {
            "rows": q2_rows,
            "best_of_n_curve": best_of_n_curve,
            "pareto_frontier": pareto_frontier,
        },
        "q3_uncertainty": {
            "rows": q3_rows,
        },
        "q4_coverage": q4_coverage,
        "q5_telemetry": q5_telemetry,
        "artifacts": artifacts_map,
    }


__all__ = [
    "BootstrapEstimate",
    "CampaignReader",
    "CellStatus",
    "ContrastEstimate",
    "DomainMetricResult",
    "DomainPlugin",
    "LibraryCell",
    "PACK_TELEMETRY_BANNER",
    "QuarantineRecord",
    "ReconciliationRecord",
    "ResidueTable",
    "SinkArray",
    "SinkSchema",
    "calculate_pairwise_diversity",
    "clear_quarantined_units",
    "compute_best_of_n",
    "compute_composition_entropy",
    "compute_cross_knob_collisions",
    "compute_duplicate_rates",
    "compute_mde_080",
    "compute_pareto_frontier_2d",
    "compute_positional_entropy",
    "compute_target_recovery",
    "discover_domain_plugins",
    "dumps_canonical_json",
    "enforce_cell_immutability",
    "evaluate_cluster_bootstrap_condition",
    "evaluate_contrast",
    "execute_domain_plugin",
    "export_campaign_eda",
    "extract_campaign_telemetry",
    "generate_multinomial_cluster_weights",
    "get_quarantined_units",
    "is_campaign_unit_packed",
    "looped_cluster_ratio_bootstrap",
    "quarantine_unit",
    "read_campaign_unit_member",
    "read_done_marker",
    "reconcile_sweep_grid",
    "register_domain_plugin",
    "run_campaign_eda",
    "safe_shannon_entropy",
    "sanitize_for_rfc8259",
    "vectorized_cluster_ratio_bootstrap",
]
