"""Pure host SLURM array submission script generator and executor.

Covering [T3-04], GATE-08, GATE-09, and Section 6.3.3 of SPEC-261105-CAMPAIGN-RUNNER.
Zero imports of controller/ or bathos.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xtrax.campaign.executors.protocol import CampaignSummary

if TYPE_CHECKING:
    from xtrax.campaign.manifest import CampaignManifest, ManifestRow


def get_pending_row_indices(
    manifest: CampaignManifest,
    output_dir: Path | str,
) -> list[int]:
    """Identify uncompleted row indices eligible for scheduling.

    Per Section 6.3.3 of SPEC-261105-CAMPAIGN-RUNNER:
    Scans output_dir for valid .done.json markers and filters out completed units.
    """
    out_dir = Path(output_dir)
    pending = []
    for idx, row in enumerate(manifest.rows):
        done_candidates = [
            out_dir / f"{row.manifest_row_hash}.done.json",
            out_dir / f"{row.manifest_row_hash}.zarr.done.json",
            out_dir / f"{Path(row.output_locator).name}.done.json",
            out_dir / f"{row.output_locator}.done.json",
        ]
        if not any(cand.exists() for cand in done_candidates):
            pending.append(idx)
    return pending


def format_slurm_array_indices(
    indices: list[int],
    max_concurrent: int | None = None,
) -> str:
    """Format arbitrary sparse row indices into compact SLURM --array range string.

    Per Section 6.3.3 of SPEC-261105-CAMPAIGN-RUNNER:
    e.g. [12, 15, 44..50] with max_concurrent=20 -> '12,15,44-50%20'.
    """
    if not indices:
        return ""
    sorted_indices = sorted(set(indices))
    ranges: list[str] = []
    start = sorted_indices[0]
    end = start

    for idx in sorted_indices[1:]:
        if idx == end + 1:
            end = idx
        else:
            ranges.append(f"{start}-{end}" if start != end else f"{start}")
            start = end = idx

    ranges.append(f"{start}-{end}" if start != end else f"{start}")
    spec = ",".join(ranges)
    return f"{spec}%{max_concurrent}" if max_concurrent else spec


class SlurmArrayExecutor:
    """Pure host script generator emitting portable #SBATCH submission scripts for SLURM arrays.

    Per Section 6.3.3 of SPEC-261105-CAMPAIGN-RUNNER:
    - Pure host generator: 0 GPUs, 0 JIT compilations, 0 child processes during generation.
    - Sparse array resume: emits compact #SBATCH --array range scheduling only uncompleted rows.
    - Boundary control: if all units completed, returns early without generating an empty script.
    - Straggler cancellation: cancel_tasks issues scancel <job_id>_<task_id>.
    """

    def __init__(
        self,
        *,
        job_name: str = "xtrax_campaign",
        cpus_per_task: int = 4,
        gpus: int = 1,
        time_limit: str = "02:00:00",
        max_concurrent: int | None = None,
        partition: str | None = None,
        output_log: str = "logs/slurm_%A_%a.out",
        error_log: str = "logs/slurm_%A_%a.err",
        job_id: str | None = None,
        submit: bool = False,
    ) -> None:
        self.job_name = job_name
        self.cpus_per_task = cpus_per_task
        self.gpus = gpus
        self.time_limit = time_limit
        self.max_concurrent = max_concurrent
        self.partition = partition
        self.output_log = output_log
        self.error_log = error_log
        self.job_id = job_id
        self.submit = submit
        self._cancelled_tasks: set[Any] = set()

    def cancel_tasks(self, task_ids: list[Any], job_id: str | None = None) -> None:
        """Issue scancel <job_id>_<task_id> for straggler control.

        Args:
            task_ids: List of task identifiers or row indices to cancel.
            job_id: Optional job ID override (defaults to self.job_id).
        """
        target_job = job_id or self.job_id
        for tid in task_ids:
            task_ref = f"{target_job}_{tid}" if target_job else str(tid)
            subprocess.run(["scancel", task_ref], check=True)
            self._cancelled_tasks.add(tid)

    def generate_script(
        self,
        manifest: CampaignManifest,
        output_dir: Path | str,
        *,
        manifest_path: Path | str | None = None,
        resume: bool = True,
        max_concurrent: int | None = None,
        script_path: Path | str | None = None,
    ) -> str | None:
        """Generate a portable Bash #SBATCH array script.

        Args:
            manifest: Planned CampaignManifest.
            output_dir: Output directory containing results and done markers.
            manifest_path: Path to the canonical manifest.json on disk.
            resume: Whether to filter out completed units.
            max_concurrent: Optional concurrency throttle override (%M).
            script_path: Optional path to write the generated script file.

        Returns:
            Generated submission script text, or None if no pending tasks remain.
        """
        out_dir = Path(output_dir)
        if resume:
            pending_indices = get_pending_row_indices(manifest, out_dir)
        else:
            pending_indices = list(range(len(manifest.rows)))

        # Boundary Control (GATE-09): Exit cleanly if all units completed
        if not pending_indices:
            print(
                f"All {len(manifest.rows)} units in campaign "
                f"'{manifest.metadata.campaign_id}' are already completed. "
                f"No tasks to schedule."
            )
            return None

        concurrent_limit = max_concurrent or self.max_concurrent
        array_spec = format_slurm_array_indices(pending_indices, max_concurrent=concurrent_limit)

        resolved_manifest_path = (
            str(manifest_path)
            if manifest_path is not None
            else str(out_dir / "manifest.json")
        )

        directives = [
            "#!/usr/bin/env bash",
            f"#SBATCH --job-name={self.job_name}",
            f"#SBATCH --array={array_spec}",
            f"#SBATCH --cpus-per-task={self.cpus_per_task}",
        ]
        if self.gpus > 0:
            directives.append(f"#SBATCH --gres=gpu:{self.gpus}")
        directives.append(f"#SBATCH --time={self.time_limit}")
        if self.partition:
            directives.append(f"#SBATCH --partition={self.partition}")
        directives.append(f"#SBATCH --output={self.output_log}")
        directives.append(f"#SBATCH --error={self.error_log}")

        script_lines = directives + [
            "",
            "set -euo pipefail",
            "",
            f'MANIFEST_PATH="{resolved_manifest_path}"',
            "ROW_INDEX=${SLURM_ARRAY_TASK_ID}",
            "",
            'echo "Executing task ${SLURM_ARRAY_TASK_ID} on $(hostname)"',
            "xtrax campaign run \\",
            '  --manifest "${MANIFEST_PATH}" \\',
            '  --row-index "${ROW_INDEX}"',
            "",
        ]
        script_text = "\n".join(script_lines)

        if script_path is not None:
            path = Path(script_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(script_text, encoding="utf-8")
            os.chmod(path, 0o755)

        return script_text

    def run(
        self,
        manifest: CampaignManifest,
        unit_fn: Callable[[ManifestRow], Any],
        *,
        output_dir: str,
        row_indices: list[int] | None = None,
        max_failures: int | None = None,
    ) -> CampaignSummary:
        """Coordinate SLURM array generation or submission."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        pending = (
            list(row_indices)
            if row_indices is not None
            else get_pending_row_indices(manifest, out_dir)
        )
        total_units = len(manifest.rows)
        completed_units = total_units - len(pending)
        skipped_units = completed_units

        # Boundary Control (GATE-09): early return on empty pending
        if not pending:
            return CampaignSummary(
                campaign_id=manifest.metadata.campaign_id,
                total_units=total_units,
                completed_units=total_units,
                skipped_units=skipped_units,
                failed_units=0,
                elapsed_seconds=0.0,
                failures=[],
            )

        script_path = out_dir / "submit_campaign.sh"
        self.generate_script(
            manifest=manifest,
            output_dir=out_dir,
            resume=True,
            script_path=script_path,
        )

        if self.submit:
            res = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            # Parse Submitted batch job <job_id>
            output = res.stdout.strip()
            if "Submitted batch job" in output:
                self.job_id = output.split()[-1]

        return CampaignSummary(
            campaign_id=manifest.metadata.campaign_id,
            total_units=total_units,
            completed_units=completed_units,
            skipped_units=skipped_units,
            failed_units=0,
            elapsed_seconds=0.0,
            failures=[],
        )
