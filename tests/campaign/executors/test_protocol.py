"""Tests for CampaignExecutor protocol and CampaignSummary dataclass.

Covering [T3-01] and Section 6.2 of SPEC-261105-CAMPAIGN-RUNNER.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from xtrax.campaign.executors.protocol import (
    CampaignExecutor,
    CampaignSummary,
)
from xtrax.campaign.manifest import CampaignManifest, ManifestRow


def test_campaign_summary_schema_and_immutability():
    """Verify CampaignSummary fields, types, and immutability."""
    summary = CampaignSummary(
        campaign_id="camp_test_001",
        total_units=10,
        completed_units=7,
        skipped_units=2,
        failed_units=1,
        elapsed_seconds=12.34,
        failures=[{"row_hash": "abc", "error": "test error"}],
    )
    assert summary.campaign_id == "camp_test_001"
    assert summary.total_units == 10
    assert summary.completed_units == 7
    assert summary.skipped_units == 2
    assert summary.failed_units == 1
    assert summary.elapsed_seconds == 12.34
    assert len(summary.failures) == 1
    assert summary.failures[0]["row_hash"] == "abc"

    # Immutability
    with pytest.raises((AttributeError, TypeError)):
        summary.completed_units = 8  # type: ignore[misc]

    # Slots check
    assert hasattr(summary, "__slots__")


def test_campaign_executor_protocol_conformance():
    """Verify CampaignExecutor is a runtime-checkable Protocol with run and cancel_tasks."""

    class MockValidExecutor:
        def run(
            self,
            manifest: CampaignManifest,
            unit_fn: Callable[[ManifestRow], Any],
            *,
            output_dir: str,
            row_indices: list[int] | None = None,
            max_failures: int | None = None,
        ) -> CampaignSummary:
            return CampaignSummary(
                campaign_id=manifest.metadata.campaign_id,
                total_units=len(manifest.rows),
                completed_units=0,
                skipped_units=0,
                failed_units=0,
                elapsed_seconds=0.0,
                failures=[],
            )

        def cancel_tasks(self, task_ids: list[Any]) -> None:
            pass

    executor = MockValidExecutor()
    assert isinstance(executor, CampaignExecutor)


def test_campaign_executor_protocol_rejection():
    """Verify non-conforming classes fail runtime protocol checks."""

    class MissingCancelExecutor:
        def run(
            self,
            manifest: CampaignManifest,
            unit_fn: Callable[[ManifestRow], Any],
            *,
            output_dir: str,
            row_indices: list[int] | None = None,
            max_failures: int | None = None,
        ) -> CampaignSummary:
            return CampaignSummary(
                campaign_id="",
                total_units=0,
                completed_units=0,
                skipped_units=0,
                failed_units=0,
                elapsed_seconds=0.0,
                failures=[],
            )

    class MissingRunExecutor:
        def cancel_tasks(self, task_ids: list[Any]) -> None:
            pass

    assert not isinstance(MissingCancelExecutor(), CampaignExecutor)
    assert not isinstance(MissingRunExecutor(), CampaignExecutor)
