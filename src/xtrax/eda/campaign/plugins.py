"""Domain plugin protocol, immutability isolation, and quarantine manager for xtrax Campaign EDA.

Implements P5-03 per approved specification 261106_campaign-eda-redesign.md (Rev 1.1).
Guarantees zero-copy array immutability (MAJ-01), defensive quarantine over (Exception, SystemExit)
(BLK-06), and strict schema validation (MAJ-02).
"""

from __future__ import annotations

import datetime
import importlib.metadata
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from xtrax.campaign.manifest import ManifestRow
from xtrax.eda.campaign.types import (
    DomainMetricResult,
    LibraryCell,
    QuarantineRecord,
    SinkSchema,
)

logger = logging.getLogger("xtrax.eda.campaign.plugins")


@runtime_checkable
class DomainPlugin(Protocol):
    """Protocol for consumer domain analytical plugins (spec §4.1)."""

    @property
    def domain_name(self) -> str:
        """Unique identifier of the domain (e.g. 'aminx', 'demistify')."""
        ...

    @property
    def version(self) -> str:
        """Semantic version of the plugin implementation."""
        ...

    def extract_cell(
        self,
        store_path: Path,
        manifest_row: ManifestRow,
        sink_schema: SinkSchema,
    ) -> Sequence[LibraryCell]:
        """Extract one or more LibraryCells from a committed unit store."""
        ...

    def compute_metrics(self, cell: LibraryCell) -> DomainMetricResult:
        """Compute domain-specific scalar and per-residue metrics."""
        ...


def enforce_cell_immutability(cell: LibraryCell) -> None:
    """Set writeable=False on all underlying NumPy arrays for zero-copy immutability (MAJ-01).

    Any in-place mutation attempts by third-party plugins will immediately raise
    ValueError: assignment destination is read-only at 0% memory copy overhead.
    """
    if isinstance(cell.tokens, np.ndarray):
        cell.tokens.flags.writeable = False
    if isinstance(cell.mask, np.ndarray):
        cell.mask.flags.writeable = False
    if cell.logits is not None and isinstance(cell.logits, np.ndarray):
        cell.logits.flags.writeable = False
    if cell.scores is not None and isinstance(cell.scores, np.ndarray):
        cell.scores.flags.writeable = False
    if hasattr(cell.residues, "real_mask") and isinstance(cell.residues.real_mask, np.ndarray):
        cell.residues.real_mask.flags.writeable = False


def enforce_metric_result_immutability(result: DomainMetricResult) -> None:
    """Set writeable=False on all underlying NumPy track arrays (F-LENB-06 / F-MAJ-05)."""
    if hasattr(result, "tracks") and isinstance(result.tracks, Mapping):
        for arr in result.tracks.values():
            if isinstance(arr, np.ndarray):
                arr.flags.writeable = False


def _safe_domain_name(plugin: DomainPlugin) -> str:
    """Defensively resolve domain name from plugin without crashing (F-LENB-09 / F-MAJ-06)."""
    try:
        name = getattr(plugin, "domain_name", None)
        if isinstance(name, str) and name:
            return name
    except (Exception, SystemExit):
        pass
    return type(plugin).__name__


# Global quarantine ledger
_QUARANTINED_UNITS: list[QuarantineRecord] = []
# Explicit plugin registry
_PLUGIN_REGISTRY: dict[str, DomainPlugin] = {}
# Plugin discovery cache (F-MIN-06)
_DOMAIN_PLUGIN_CACHE: dict[str, DomainPlugin] | None = None


def quarantine_unit(unit_key: str, domain_name: str, reason: str) -> QuarantineRecord:
    """Record an execution unit quarantined due to plugin failure or schema violation."""
    rec = QuarantineRecord(
        unit_key=unit_key,
        domain_name=domain_name,
        reason=reason,
        timestamp_utc=datetime.datetime.now(datetime.UTC).isoformat(),
    )
    _QUARANTINED_UNITS.append(rec)
    logger.warning("Quarantined unit %s under domain %s: %s", unit_key, domain_name, reason)
    return rec


def get_quarantined_units() -> list[QuarantineRecord]:
    """Retrieve all quarantined execution units recorded during the campaign analysis session."""
    return list(_QUARANTINED_UNITS)


def clear_quarantined_units() -> None:
    """Clear the quarantined units ledger."""
    _QUARANTINED_UNITS.clear()


def register_domain_plugin(plugin: DomainPlugin) -> None:
    """Explicitly register a domain plugin instance."""
    global _DOMAIN_PLUGIN_CACHE
    _DOMAIN_PLUGIN_CACHE = None
    _PLUGIN_REGISTRY[plugin.domain_name] = plugin


def discover_domain_plugins() -> dict[str, DomainPlugin]:
    """Discover registered and entry-point domain plugins under group xtrax.eda.domains."""
    global _DOMAIN_PLUGIN_CACHE
    if _DOMAIN_PLUGIN_CACHE is not None:
        return dict(_DOMAIN_PLUGIN_CACHE)

    plugins = dict(_PLUGIN_REGISTRY)
    try:
        eps = importlib.metadata.entry_points(group="xtrax.eda.domains")
        for ep in eps:
            try:
                plugin_cls = ep.load()
                plugin_inst = plugin_cls() if isinstance(plugin_cls, type) else plugin_cls
                if isinstance(plugin_inst, DomainPlugin):
                    plugins[ep.name] = plugin_inst
            except (Exception, SystemExit) as exc:
                logger.warning("Failed to load domain plugin %s: %s", ep.name, exc)
    except (Exception, SystemExit) as exc:
        logger.warning("Failed to inspect entry points for xtrax.eda.domains: %s", exc)

    _DOMAIN_PLUGIN_CACHE = plugins
    return dict(plugins)


def reload_domain_plugins() -> dict[str, DomainPlugin]:
    """Clear cache and rediscover domain plugins (F-MIN-06)."""
    global _DOMAIN_PLUGIN_CACHE
    _DOMAIN_PLUGIN_CACHE = None
    return discover_domain_plugins()


def execute_domain_plugin(
    plugin: DomainPlugin,
    store_path: Path,
    manifest_row: ManifestRow,
    sink_schema: SinkSchema,
) -> list[tuple[LibraryCell, DomainMetricResult]]:
    """Execute plugin extraction within defensive quarantine boundary (BLK-06, MAJ-01, MAJ-02).

    Catches both Exception and SystemExit to prevent malicious or buggy plugins from
    terminating the runner process. Validates schema and quarantines failing units.
    """
    results: list[tuple[LibraryCell, DomainMetricResult]] = []
    unit_key = manifest_row.manifest_row_hash
    domain_name = _safe_domain_name(plugin)

    try:
        cells = plugin.extract_cell(store_path, manifest_row, sink_schema)
        if not cells:
            quarantine_unit(
                unit_key,
                domain_name,
                "EmptyCells: extract_cell returned 0 cells",
            )
            return []

        for cell in cells:
            # Enforce zero-copy array immutability
            enforce_cell_immutability(cell)

            # Compute domain metrics
            metric_result = plugin.compute_metrics(cell)

            # Enforce zero-copy immutability on track arrays (F-LENB-06 / F-MAJ-05)
            enforce_metric_result_immutability(metric_result)

            # Validate strict numeric and finiteness schema (MAJ-02, F-MAJ-07)
            metric_result.validate_schema()

            results.append((cell, metric_result))
    except (Exception, SystemExit) as exc:
        reason = f"{type(exc).__name__}: {exc}"
        quarantine_unit(unit_key, domain_name, reason)
        return []

    return results
