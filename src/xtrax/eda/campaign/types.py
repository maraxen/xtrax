"""Core data types and schemas for xtrax Campaign EDA.

Implements P5-01 per approved specification 261106_campaign-eda-redesign.md (Rev 1.1).
Dependencies: strictly stdlib + numpy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np


class CellStatus(StrEnum):
    """Execution status of a planned manifest row or visual sweep grid cell."""

    COMMITTED = "COMMITTED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    MISSING = "MISSING"


@dataclass(frozen=True, slots=True)
class ResidueTable:
    """Canonical structural residue mapping."""

    native_tokens: str
    chain_ids: tuple[str, ...]
    res_numbers: tuple[int, ...]
    insertion_codes: tuple[str, ...]
    real_mask: np.ndarray
    parser_name: str
    parser_version: str


@dataclass(frozen=True, slots=True)
class LibraryCell:
    """Standardized extracted data cell for a single experimental unit."""

    knobs: Mapping[str, Any]
    seed: int
    target: str
    tokens: np.ndarray  # shape (N, L)
    sequences: tuple[str, ...]  # shape (N,)
    logits: np.ndarray | None  # shape (N, L, alphabet_size) or None
    scores: np.ndarray | None  # shape (N,) or None
    is_score_derived: bool
    residues: ResidueTable
    mask: np.ndarray  # shape (L,) boolean designable mask


@dataclass(frozen=True, slots=True)
class DomainMetricResult:
    """Named scalar metrics and tracks computed for a cell with strict type contracts."""

    metrics: Mapping[str, float]
    flags: Mapping[str, bool]
    tracks: Mapping[str, np.ndarray]

    def validate_schema(self) -> None:
        """Validate that all metric values and flags strictly adhere to schema (MAJ-02).

        Enforces that:
        - Every key in `metrics` is a string.
        - Every value in `metrics` is a strictly finite float or int (non-NaN and non-Inf).
        - Every key in `flags` is a string and every value is a boolean.

        Violations trigger immediate unit quarantine, guaranteeing that downstream PyArrow
        Parquet table conversion never fails with ArrowInvalid.
        """
        for k, v in self.metrics.items():
            if not isinstance(k, str):
                raise TypeError(f"Metric key {k!r} must be str, got {type(k).__name__}")
            if not isinstance(v, (int, float)) or not np.isfinite(v):
                raise TypeError(
                    f"Domain metric {k!r} has invalid value {v!r} (type {type(v).__name__}); "
                    "metrics must be strictly finite floats or ints."
                )
        for k, v in self.flags.items():
            if not isinstance(k, str) or not isinstance(v, bool):
                raise TypeError(
                    f"Flag {k!r}={v!r} invalid; keys must be str and values must be bool (got "
                    f"key type {type(k).__name__}, value type {type(v).__name__})"
                )
        for k, v in self.tracks.items():
            if not isinstance(k, str):
                raise TypeError(f"Track key {k!r} must be str, got {type(k).__name__}")
            if not isinstance(v, np.ndarray):
                raise TypeError(f"Track {k!r} must be np.ndarray, got {type(v).__name__}")
            if v.dtype.kind in ("f", "c") and not np.all(np.isfinite(v)):
                raise ValueError(f"Track {k!r} contains non-finite values (NaN or Inf)")


@dataclass(frozen=True, slots=True)
class SinkArray:
    """Descriptor for an array stored in a campaign sink."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    path: str


@dataclass(frozen=True, slots=True)
class SinkSchema:
    """Schema descriptor for a campaign sink."""

    arrays: Mapping[str, SinkArray]
    bind: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QuarantineRecord:
    """Record of an execution unit quarantined during plugin extraction or metric computation."""

    unit_key: str
    domain_name: str
    reason: str
    timestamp_utc: str


@dataclass(frozen=True, slots=True)
class ReconciliationRecord:
    """Reconciliation state for a planned campaign manifest row."""

    manifest_row_hash: str
    input_hash: str
    grid_coords: Mapping[str, Any]
    status: CellStatus
    failure_reason: str | None = None
    is_packed: bool = False
    duration_s: float | None = None
    completed_at_utc: str | None = None
    pack_telemetry: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class BootstrapEstimate:
    """Vectorized cluster ratio bootstrap result for a metric."""

    metric_name: str
    point_estimate: float
    ci_95: tuple[float, float] | None
    se: float
    mde_080: float
    descriptive_only: bool
    warning: str | None = None


@dataclass(frozen=True, slots=True)
class ContrastEstimate:
    """Contrast estimate between two knob conditions."""

    contrast_name: str
    metric_name: str
    delta_hat: float
    ci_95: tuple[float, float] | None
    se: float
    mde_080: float
    detected_and_powered: bool
    descriptive_only: bool
    warning: str | None = None
