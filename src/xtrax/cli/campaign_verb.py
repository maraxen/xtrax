"""Campaign CLI verb implementation for xtrax (Track 4: T4-01).

Provides high-throughput multi-arm campaign planning, execution across
sequential/process-pool/SLURM backends, status inspection, and Tier-1/Tier-2
verification.
"""

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xtrax.campaign.executors.local import (
    LocalProcessPoolExecutor,
    LocalSequentialExecutor,
    _resolve_output_path,
    _validate_resume_verify,
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
        fn: Import path (module:fn) of the unit compute callable. Required for 'run':
            the callable must write its output store, so there is no default.
    """

    # Quoted: ``tyro`` is None until entrypoint.main() binds it (AC2 keeps imports
    # tyro-free), so these annotations are resolved only when tyro reads the hints.
    action: "tyro.conf.Positional[str]" = "status"
    target: "tyro.conf.Positional[str | None]" = None
    spec_path: str | None = None
    manifest: str | None = None
    output_dir: str | None = None
    output: str | None = None
    row_index: int | None = None
    executor: str = "local-seq"
    require_approval: bool = False
    deep: bool = False
    fn: str | None = None
    #: 'run' resume check: 'deep' (default) recomputes each completed store's content digest
    #: before reusing it; 'tier1' trusts the done marker and root metadata only (fast path).
    resume_verify: str = "deep"


def _iter_campaign_files(out_dir: Path, pattern: str) -> list[Path]:
    """Recursively find campaign bookkeeping files, skipping quarantine and store internals.

    Planned locators nest stores under ``<campaign_id>/``, so a top-level glob misses
    them. Files inside ``.quarantine/`` or inside a ``*.zarr`` store are not live units.
    """
    found: list[Path] = []
    for path in sorted(out_dir.rglob(pattern)):
        rel_parts = path.relative_to(out_dir).parts[:-1]
        if any(part == ".quarantine" or part.endswith(".zarr") for part in rel_parts):
            continue
        found.append(path)
    return found


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


def _open_campaign_ledger(
    manifest: CampaignManifest,
    manifest_path: Path,
    executor_kind: str,
    row_indices: list[int] | None,
) -> Any:
    """Open the run-ledger entry for one `campaign run`, failing closed.

    The ledger's run kinds are train/eval/export; a campaign run executes unit
    functions without training, so it is recorded as ``eval`` and identified by
    ``context["verb"] == "campaign"``. Per-unit provenance stays in the done markers
    and ``reuse_log.jsonl``; this row records the run as a whole.
    """
    import hashlib

    from xtrax.run.ident import new_run_id
    from xtrax.telemetry.ledger import LedgerUnavailableError, RunLedger
    from xtrax.telemetry.record import KIND_EVAL

    context = {
        "verb": "campaign",
        "campaign_id": manifest.metadata.campaign_id,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "executor": executor_kind,
        "row_indices": "all" if row_indices is None else ",".join(map(str, row_indices)),
    }
    try:
        return RunLedger.open(new_run_id(), kind=KIND_EVAL, context=context)
    except LedgerUnavailableError as exc:
        raise CLIError(str(exc)) from exc


def _record_campaign_outcome(ledger: Any, summary: Any) -> None:
    """Add a campaign run's counts to its ledger row and mark failed or blocked runs."""
    from xtrax.telemetry.record import STATUS_FAILED

    ledger.context.update(
        {
            "total_units": str(summary.total_units),
            "completed_units": str(summary.completed_units),
            "skipped_units": str(summary.skipped_units),
            "failed_units": str(summary.failed_units),
            "locked_units": str(summary.locked_units),
        }
    )
    if summary.failed_units > 0:
        ledger.set_status(STATUS_FAILED, f"{summary.failed_units} unit(s) failed")
    elif summary.locked_units > 0:
        ledger.set_status(
            STATUS_FAILED, f"{summary.locked_units} unit(s) locked by another worker, not run"
        )


def _run_run(args: CampaignArgs) -> Any:
    """Execute 'xtrax campaign run'."""
    try:
        resume_mode = _validate_resume_verify(args.resume_verify)
    except ValueError as exc:
        raise CLIError(str(exc)) from exc
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

    # Resolve unit function. There is no default: a unit function that writes nothing
    # would otherwise be recorded as complete.
    if not args.fn:
        raise CLIError(
            "xtrax campaign run requires --fn <module:function>; the function must write "
            "the unit's output store"
        )
    unit_fn = load_fn(args.fn)

    # Resolve target row index
    row_indices = [int(args.row_index)] if args.row_index is not None else None

    # Instantiate chosen executor
    executor_kind = args.executor.lower()
    if executor_kind in ("local-seq", "local_seq", "sequential"):
        executor = LocalSequentialExecutor(resume_verify=resume_mode)
    elif executor_kind in ("local-pool", "local_pool", "pool"):
        executor = LocalProcessPoolExecutor(resume_verify=resume_mode)
    elif executor_kind in ("slurm", "slurm-array"):
        executor = SlurmArrayExecutor(
            manifest_path=manifest_path.resolve(),
            fn_spec=args.fn,
            resume_verify=resume_mode,
        )
    else:
        raise CLIError(
            f"Unknown executor: {args.executor!r}. "
            "Expected one of 'local-seq', 'local-pool', 'slurm'"
        )

    # A run that executes unit functions writes one run-ledger row (telemetry contract:
    # audit/telemetry_coverage.toml). The SLURM executor only writes a script; each
    # array task is its own local `campaign run` and records its own row. The ledger
    # is opened before any unit runs, so an unwritable ledger refuses the run up front.
    ledger = None
    if not isinstance(executor, SlurmArrayExecutor):
        ledger = _open_campaign_ledger(manifest, manifest_path, executor_kind, row_indices)

    try:
        with ledger if ledger is not None else contextlib.nullcontext():
            summary = executor.run(
                manifest,
                unit_fn,
                output_dir=output_dir,
                row_indices=row_indices,
            )
            if ledger is not None:
                _record_campaign_outcome(ledger, summary)
    except Exception as exc:
        raise CLIError(f"Campaign execution failed: {exc}") from exc
    if ledger is not None:
        print(f"Campaign run recorded in the run ledger as {ledger.run_id}")

    if summary.submitted_units:
        # The CLI builds the executor with submit=False: it writes the script only.
        script = Path(output_dir) / "submit_campaign.sh"
        print(
            f"Campaign '{summary.campaign_id}': wrote SLURM array script for "
            f"{summary.submitted_units}/{summary.total_units} units "
            f"({summary.skipped_units} already done). Nothing is queued yet; "
            f"submit with: sbatch {script}"
        )
    else:
        print(
            f"Campaign '{summary.campaign_id}' execution complete: "
            f"{summary.completed_units}/{summary.total_units} completed "
            f"({summary.skipped_units} skipped, {summary.locked_units} locked by another "
            f"worker, {summary.failed_units} failed)"
        )

    # A failed or lock-held unit must not exit 0: a SLURM array task would then look
    # COMPLETED to the scheduler. The script-only SLURM path has neither, so it passes.
    if summary.failed_units > 0:
        first_errors = [str(f.get("error", "unknown error")) for f in summary.failures[:3]]
        raise CLIError(
            f"Campaign '{summary.campaign_id}': {summary.failed_units} unit(s) failed; "
            f"first error(s): {'; '.join(first_errors)}"
        )
    if summary.locked_units > 0:
        raise CLIError(
            f"Campaign '{summary.campaign_id}': {summary.locked_units} unit(s) are locked by "
            "another worker and were not run"
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
    # Each store is counted once: a broken marker marks its store failed, and that store's
    # lock (expired or unreadable) is not counted again below.
    completed_hashes: set[str] = set()
    completed_stores: set[Path] = set()
    failed_stores: set[Path] = set()
    for done_file in _iter_campaign_files(out_dir, "*.done.json"):
        store_path = done_file.with_name(done_file.name.removesuffix(".done.json"))
        try:
            data = json.loads(done_file.read_text(encoding="utf-8"))
            row_hash = data.get("manifest_row_hash") if isinstance(data, dict) else None
        except Exception:
            row_hash = None
        if not isinstance(row_hash, str) or not row_hash.strip():
            # Unparseable marker, or one without a row hash: the unit is not done and
            # verify rejects it, so status reports it failed rather than hiding it.
            failed_stores.add(store_path)
        elif store_path.exists():
            completed_hashes.add(row_hash)
            completed_stores.add(store_path)
    failed_count += len(failed_stores)
    completed_count = len(completed_hashes)

    # 2. Count active running locks (deduplicated by manifest_row_hash)
    running_hashes: set[str] = set()
    for lock_file in _iter_campaign_files(out_dir, "*.lock"):
        lock_store = lock_file.with_name(lock_file.name.removesuffix(".lock"))
        if lock_store in failed_stores:
            continue
        try:
            data = json.loads(lock_file.read_text(encoding="utf-8"))
            lock = CampaignLock.from_dict(data)
            row_hash = lock.manifest_row_hash
            if row_hash in completed_hashes or lock_store in completed_stores:
                continue
            if now < lock.lease_expires_at_unix_s + CLOCK_DRIFT_GRACE_PERIOD_S:
                running_hashes.add(row_hash)
            else:
                failed_count += 1
        except Exception:
            if lock_store not in completed_stores:
                failed_count += 1
    running_count = len(running_hashes)

    # 3. Count quarantined units
    quarantine_roots = [out_dir / ".quarantine", *out_dir.glob("*/.quarantine")]
    for q_root in quarantine_roots:
        if q_root.is_dir():
            quarantined_count += len(list(q_root.iterdir()))
    corrupt_ids: set[str] = set()
    for corrupt_file in out_dir.glob("**/*.corrupt.*"):
        if ".quarantine" in corrupt_file.relative_to(out_dir).parts:
            continue
        _, sep, corrupt_id = corrupt_file.name.rpartition(".corrupt.")
        if sep and corrupt_id:
            corrupt_ids.add(corrupt_id)
    quarantined_count += len(corrupt_ids)

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
    manifest_str = args.manifest
    if not manifest_str and args.target and Path(args.target).is_file():
        manifest_str = args.target

    # Output dir: --output-dir, else a positional directory, else the manifest's parent.
    # A positional directory must win over the manifest's parent: in
    # `verify outputs --manifest manifest.json` the stores live under outputs/.
    out_dir_str = args.output_dir
    if not out_dir_str and args.target and Path(args.target).is_dir():
        out_dir_str = args.target
    if not out_dir_str and manifest_str:
        out_dir_str = str(Path(manifest_str).parent)
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
    if manifest_str:
        manifest_path = Path(manifest_str)
        if not manifest_path.is_file():
            raise CLIError(f"Campaign manifest file not found: {manifest_str}")
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = CampaignManifest.from_dict(manifest_data)
        except Exception as exc:
            raise CLIError(f"Failed to load campaign manifest: {exc}") from exc

        for row in manifest.rows:
            store_path = _resolve_output_path(out_dir, row)

            try:
                res = validate_unit_done(
                    store_path,
                    expected_row=row,
                    deep_verify=args.deep,
                    quarantine_on_failure=False,
                )
            except OSError as exc:
                failed_count += 1
                errors.append(f"Row {row.manifest_row_hash}: I/O error while verifying: {exc}")
                continue
            if res.is_valid:
                passed_count += 1
            else:
                failed_count += 1
                errors.append(f"Row {row.manifest_row_hash}: {res.error_message}")
    else:
        # Scan done markers directly. Each marker sits beside its store
        # (<store>.done.json), so the store is the marker's sibling.
        done_markers = _iter_campaign_files(out_dir, "*.done.json")
        if not done_markers:
            # Nothing verified is not a pass: an empty or fully quarantined campaign must
            # not satisfy a verify gate.
            quarantined = sum(
                len(list(q_root.iterdir()))
                for q_root in (out_dir / ".quarantine", *out_dir.glob("*/.quarantine"))
                if q_root.is_dir()
            )
            raise CLIError(
                f"Campaign verification failed: no done markers found in {out_dir} "
                f"({quarantined} quarantined unit(s)); nothing was verified"
            )

        seen_hashes: set[str] = set()
        for marker_file in done_markers:
            try:
                marker_data = json.loads(marker_file.read_text(encoding="utf-8"))
                row_hash = marker_data.get("manifest_row_hash", "")
                input_hash = marker_data.get("input_hash", "")
                if (
                    not isinstance(row_hash, str)
                    or not row_hash.strip()
                    or not isinstance(input_hash, str)
                    or not input_hash.strip()
                ):
                    failed_count += 1
                    errors.append(
                        f"Unit {marker_file.name}: done marker has empty "
                        "manifest_row_hash or input_hash"
                    )
                    continue
                if row_hash in seen_hashes:
                    continue
                seen_hashes.add(row_hash)

                store_path = marker_file.with_name(marker_file.name.removesuffix(".done.json"))

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
