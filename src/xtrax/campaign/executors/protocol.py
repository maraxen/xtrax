"""CampaignExecutor protocol and CampaignSummary data representation.

Covering [T3-01] and Section 6.2 of SPEC-261105-CAMPAIGN-RUNNER.
Zero imports of controller/ or bathos.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from xtrax.campaign.manifest import CampaignManifest, ManifestRow


@dataclass(frozen=True, slots=True)
class CampaignSummary:
    """Execution outcome summary for a planned campaign manifest."""

    campaign_id: str
    total_units: int
    completed_units: int
    skipped_units: int
    failed_units: int
    elapsed_seconds: float
    failures: list[dict[str, Any]] = field(default_factory=list)
    #: Units not run because another worker holds their lease. Not completed.
    locked_units: int = 0
    #: Units handed to an external scheduler (SLURM) by this call. Not completed.
    submitted_units: int = 0

    def accounted_units(self) -> int:
        """Sum of every outcome bucket; equals ``total_units`` for a well-formed summary."""
        return (
            self.completed_units
            + self.skipped_units
            + self.failed_units
            + self.locked_units
            + self.submitted_units
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "total_units": self.total_units,
            "completed_units": self.completed_units,
            "skipped_units": self.skipped_units,
            "failed_units": self.failed_units,
            "elapsed_seconds": self.elapsed_seconds,
            "failures": self.failures,
            "locked_units": self.locked_units,
            "submitted_units": self.submitted_units,
        }


@runtime_checkable
class CampaignExecutor(Protocol):
    """Abstract execution harness for a planned campaign manifest.

    Enforces the Dispatch-Independence Invariant (Section 6.1):
    Universal Scheduler Equivalence where any planned CampaignManifest
    can be dispatched across local sequential, local process pool,
    or remote SLURM array executors without modifying the manifest.
    """

    def run(
        self,
        manifest: CampaignManifest,
        unit_fn: Callable[[ManifestRow], Any],
        *,
        output_dir: str,
        row_indices: list[int] | None = None,
        max_failures: int | None = None,
    ) -> CampaignSummary:
        """Execute units in manifest according to the scheduling strategy.

        Args:
            manifest: The planned CampaignManifest to execute.
            unit_fn: Unit compute callable taking a ManifestRow. Its output reaches disk in
                one of two ways: it writes a store at the row's resolved output path, or
                it returns a Mapping that the executor persists as a zarr v3 group
                (arrays as arrays, scalars and JSON values as attributes). A unit that
                does neither fails; it is never stamped done.
            output_dir: Directory where unit output stores and done markers reside.
            row_indices: Optional list of integer row indices to restrict execution to.
            max_failures: Optional threshold of allowable failed units before aborting.

        Returns:
            CampaignSummary describing execution results.
        """
        ...

    def cancel_tasks(self, task_ids: list[Any]) -> None:
        """Cancel or terminate active/pending tasks for straggler control.

        Args:
            task_ids: List of task identifiers or row indices to cancel.
        """
        ...
