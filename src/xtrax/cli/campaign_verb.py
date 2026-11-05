"""Campaign CLI verb implementation for xtrax (Track 4: T4-01).

Provides high-throughput multi-arm campaign planning, execution across
sequential/process-pool/SLURM backends, status inspection, and Tier-1/Tier-2
verification.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xtrax.campaign.executors.local import (
    LocalProcessPoolExecutor,
    LocalSequentialExecutor,
    _resolve_output_path,
)
from xtrax.campaign.executors.slurm import SlurmArrayExecutor
from xtrax.campaign.lease import CLOCK_DRIFT_GRACE_PERIOD_S, CampaignLock
from xtrax.campaign.manifest import (
    CampaignManifest,
    ManifestRow,
    ResourceRequirements,
    plan_campaign,
)
from xtrax.campaign.marker import validate_unit_done
from xtrax.campaign.spec import CampaignSpec
from xtrax.cli.errors import CLIError
from xtrax.cli.loader import load_fn

# tyro is bound dynamically in entrypoint.py:main() to keep imports tyro-free
tyro: Any = None

_ACTIONS = ("plan", "run", "status", "verify")


@dataclass
class CampaignArgs:
    """Arguments for the campaign verb.

    Attributes:
        action: One of 'plan', 'run', 'status', 'verify'.
        target: Optional positional path argument (spec_path for plan,
            output_dir for status/verify).
        spec_path: Path to campaign_spec.toml (for 'plan').
        manifest: Path to manifest.json (for 'run' and optional for 'verify').
        output_dir: Target output directory for campaign results.
        output: Destination manifest output path (for 'plan').
        row_index: Optional row index for single-unit / SLURM array task execution.
        executor: Execution backend ('local-seq', 'local-pool', 'slurm').
        require_approval: Whether to verify human approval gate attestation before run.
        deep: Whether to perform Tier-2 chunk digest audit during 'verify'.
        fn: Optional import path (module:fn) for custom unit compute callable.
    """

    action: tyro.conf.Positional[str] = "status"
    target: tyro.conf.Positional[str | None] = None
    spec_path: str | None = None
    manifest: str | None = None
    output_dir: str | None = None
    output: str | None = None
    row_index: int | None = None
    executor: str = "local-seq"
    require_approval: bool = False
    deep: bool = False
    fn: str | None = None


def _default_unit_fn(row: ManifestRow) -> dict[str, Any]:
    """Default unit execution callable when --fn is omitted."""
    return {"manifest_row_hash": row.manifest_row_hash, "input_hash": row.input_hash}


def _run_plan(args: CampaignArgs) -> CampaignManifest:
    """Execute 'xtrax campaign plan'."""
    spec_path_str = args.spec_path or args.target
    if not spec_path_str:
        raise CLIError(
            "xtrax campaign plan requires a specification file (e.g. 'campaign_spec.toml')"
        )

    spec_path = Path(spec_path_str)
    if not spec_path.is_file():
        raise CLIError(f"Campaign specification file not found: {spec_path_str}")

    try:
        spec = CampaignSpec.from_toml(spec_path)
    except Exception as exc:
        raise CLIError(f"Failed to parse campaign spec {spec_path_str}: {exc}") from exc

    output_path = args.output
    if output_path is None:
        output_path = str(Path(spec.output_dir) / "manifest.json")

    try:
        manifest = plan_campaign(spec, output_manifest_path=output_path)
    except Exception as exc:
        raise CLIError(f"Campaign planning failed: {exc}") from exc

    print(f"Planned campaign '{spec.campaign_id}' with {len(manifest.rows)} units -> {output_path}")
    return manifest


def _run_run(args: CampaignArgs) -> Any:
    """Execute 'xtrax campaign run'."""
    manifest_path_str = args.manifest or args.target
    if not manifest_path_str:
        raise CLIError("xtrax campaign run requires --manifest <path>")

    manifest_path = Path(manifest_path_str)
    if not manifest_path.is_file():
        raise CLIError(f"Campaign manifest file not found: {manifest_path_str}")

    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = CampaignManifest.from_dict(manifest_data)
    except Exception as exc:
        raise CLIError(f"Failed to load campaign manifest {manifest_path_str}: {exc}") from exc

    # Enforce approval gate (T2-32 / AC-12)
    if args.require_approval:
        from xtrax.loop.campaign_approval_gate import (
            DEFAULT_GATES_TOML,
            CampaignNotApprovedError,
            assert_campaign_approved,
        )

        gates_file = Path.cwd() / ".praxia" / "loop_human_gates.toml"
        toml_path = gates_file if gates_file.exists() else DEFAULT_GATES_TOML
        try:
            assert_campaign_approved(manifest.metadata.campaign_id, toml_path=toml_path)
        except CampaignNotApprovedError as exc:
            raise CLIError(f"Campaign start refused by approval gate: {exc}") from exc

    # Resolve output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(manifest_path.parent)

    # Resolve unit function
    if args.fn:
        unit_fn = load_fn(args.fn)
    else:
        unit_fn = _default_unit_fn

    # Resolve target row index
    row_indices = [int(args.row_index)] if args.row_index is not None else None

    # Instantiate chosen executor
    executor_kind = args.executor.lower()
    if executor_kind in ("local-seq", "local_seq", "sequential"):
        executor = LocalSequentialExecutor()
    elif executor_kind in ("local-pool", "local_pool", "pool"):
        executor = LocalProcessPoolExecutor()
    elif executor_kind in ("slurm", "slurm-array"):
        executor = SlurmArrayExecutor()
    else:
        raise CLIError(
            f"Unknown executor: {args.executor!r}. "
            "Expected one of 'local-seq', 'local-pool', 'slurm'"
        )

    try:
        summary = executor.run(
            manifest,
            unit_fn,
            output_dir=output_dir,
            row_indices=row_indices,
        )
    except Exception as exc:
        raise CLIError(f"Campaign execution failed: {exc}") from exc

    print(
        f"Campaign '{summary.campaign_id}' execution complete: "
        f"{summary.completed_units}/{summary.total_units} completed "
        f"({summary.skipped_units} skipped, {summary.failed_units} failed)"
    )
    return summary


def _run_status(args: CampaignArgs) -> dict[str, int]:
    """Execute 'xtrax campaign status'."""
    out_dir_str = args.output_dir
    if not out_dir_str and args.manifest:
        out_dir_str = str(Path(args.manifest).parent)
    if not out_dir_str:
        out_dir_str = args.target or "outputs"
    out_dir = Path(out_dir_str)
    if out_dir.is_file():
        out_dir = out_dir.parent
    if not out_dir.exists():
        raise CLIError(f"Output directory does not exist: {out_dir_str}")

    completed_count = 0
    running_count = 0
    failed_count = 0
    quarantined_count = 0

    import time

    now = time.time()

    # 1. Count completed done markers (deduplicated by manifest_row_hash)
    completed_hashes: set[str] = set()
    for done_file in out_dir.glob("*.done.json"):
        try:
            data = json.loads(done_file.read_text(encoding="utf-8"))
            row_hash = data.get("manifest_row_hash")
            if row_hash:
                completed_hashes.add(row_hash)
        except Exception:
            failed_count += 1
    completed_count = len(completed_hashes)

    # 2. Count active running locks (deduplicated by manifest_row_hash)
    running_hashes: set[str] = set()
    for lock_file in out_dir.glob("*.lock"):
        try:
            data = json.loads(lock_file.read_text(encoding="utf-8"))
            lock = CampaignLock.from_dict(data)
            row_hash = lock.manifest_row_hash
            if now < lock.lease_expires_at_unix_s + CLOCK_DRIFT_GRACE_PERIOD_S:
                if row_hash not in completed_hashes:
                    running_hashes.add(row_hash)
            else:
                failed_count += 1
        except Exception:
            failed_count += 1
    running_count = len(running_hashes)

    # 3. Count quarantined units
    quarantine_roots = [out_dir / ".quarantine", *out_dir.glob("*/.quarantine")]
    for q_root in quarantine_roots:
        if q_root.is_dir():
            quarantined_count += len(list(q_root.iterdir()))
    for corrupt_file in out_dir.glob("**/*.corrupt.*"):
        quarantined_count += 1

    print(f"Campaign Status: {out_dir}")
    print(f"  Completed:   {completed_count}")
    print(f"  Running:     {running_count}")
    print(f"  Failed:      {failed_count}")
    print(f"  Quarantined: {quarantined_count}")

    return {
        "completed": completed_count,
        "running": running_count,
        "failed": failed_count,
        "quarantined": quarantined_count,
    }


def _run_verify(args: CampaignArgs) -> dict[str, int]:
    """Execute 'xtrax campaign verify'."""
    out_dir_str = args.output_dir
    if not out_dir_str and args.manifest:
        out_dir_str = str(Path(args.manifest).parent)
    if not out_dir_str:
        out_dir_str = args.target or "outputs"
    out_dir = Path(out_dir_str)
    if out_dir.is_file():
        out_dir = out_dir.parent
    if not out_dir.exists():
        raise CLIError(f"Output directory does not exist: {out_dir_str}")

    passed_count = 0
    failed_count = 0
    errors: list[str] = []

    # If manifest is supplied, verify against planned rows
    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_file():
            raise CLIError(f"Campaign manifest file not found: {args.manifest}")
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = CampaignManifest.from_dict(manifest_data)
        except Exception as exc:
            raise CLIError(f"Failed to load campaign manifest: {exc}") from exc

        for row in manifest.rows:
            store_path = _resolve_output_path(out_dir, row)

            res = validate_unit_done(
                store_path,
                expected_row=row,
                deep_verify=args.deep,
                quarantine_on_failure=False,
            )
            if res.is_valid:
                passed_count += 1
            else:
                failed_count += 1
                errors.append(f"Row {row.manifest_row_hash}: {res.error_message}")
    else:
        # Scan done markers directly
        done_markers = list(out_dir.glob("*.done.json"))
        if not done_markers:
            print(f"No done markers found in {out_dir}")
            return {"passed": 0, "failed": 0}

        for marker_file in done_markers:
            try:
                marker_data = json.loads(marker_file.read_text(encoding="utf-8"))
                row_hash = marker_data.get("manifest_row_hash", "")
                input_hash = marker_data.get("input_hash", "")
                output_loc = marker_data.get("output_locator")

                if output_loc:
                    loc = Path(output_loc)
                    store_path = loc if loc.is_absolute() else out_dir / loc.name
                else:
                    store_name = marker_file.name.removesuffix(".done.json")
                    store_path = marker_file.with_name(store_name)

                synthetic_row = ManifestRow(
                    manifest_row_hash=row_hash,
                    input_hash=input_hash,
                    campaign_id="",
                    job_id="",
                    chunk_id=0,
                    sample_start=0,
                    sample_count=0,
                    target_input="",
                    arm_id="",
                    arm_backend="",
                    checkpoint_id="",
                    fixed_policy="",
                    state_weight_profile="",
                    seed=0,
                    parameters={},
                    output_locator=str(store_path),
                    resource_requirements=ResourceRequirements(0, 1, 60, None),
                )

                res = validate_unit_done(
                    store_path,
                    expected_row=synthetic_row,
                    deep_verify=args.deep,
                    quarantine_on_failure=False,
                )
                if res.is_valid:
                    passed_count += 1
                else:
                    failed_count += 1
                    errors.append(f"Unit {marker_file.name}: {res.error_message}")
            except Exception as exc:
                failed_count += 1
                errors.append(f"Marker {marker_file.name}: {exc}")

    total = passed_count + failed_count
    tier_label = "Tier-2 Deep" if args.deep else "Tier-1 Fast"
    print(
        f"Verified {total} units in {out_dir} ({tier_label}): "
        f"{passed_count} passed, {failed_count} failed"
    )

    if failed_count > 0:
        for err in errors[:10]:
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        raise CLIError(f"Campaign verification failed: {failed_count} unit(s) corrupted or invalid")

    return {"passed": passed_count, "failed": failed_count}


def run_campaign(args: CampaignArgs) -> Any:
    """Dispatch campaign subcommand."""
    action = args.action.lower()
    if action == "plan":
        return _run_plan(args)
    if action == "run":
        return _run_run(args)
    if action == "status":
        return _run_status(args)
    if action == "verify":
        return _run_verify(args)

    raise CLIError(f"error: unknown action {args.action!r}; expected one of {', '.join(_ACTIONS)}")


__all__ = ["CampaignArgs", "run_campaign"]
