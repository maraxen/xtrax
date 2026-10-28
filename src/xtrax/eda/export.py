"""Export functions for PlanStatsDict.

Provides bitwise-reproducible RFC 8259 canonical JSON serialization,
tidy long-format CSV and Parquet tabular export, neutral artifact bundle
generation with SHA-256 manifests, and backward-compatible DataFrame conversion.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import io
import json
import math
import os
import pathlib
import sys
import unicodedata
from typing import Any

import numpy as np

from xtrax.eda.spec import generate_vega_spec
from xtrax.eda.types import PlanStatsDict

if sys.modules.get("pandas") is None and "pandas" in sys.modules:
    raise ImportError(
        "xtrax.eda.export requires visualization extras. "
        "Install with: pip install xtrax[eda]"
    )

PathLike = str | os.PathLike[str]


def _normalize_floats(obj: Any) -> Any:
    """Normalize floats recursively: -0.0 -> 0.0, NaN/Inf -> None, format float."""
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        # Round / format to fixed 8 decimal places for cross-platform IEEE-754 consistency
        rounded = round(val, 8)
        if rounded == 0.0:
            return 0.0
        return rounded
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _normalize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_floats(v) for v in obj]
    return obj


def export_canonical_json(stats: PlanStatsDict, path: PathLike | None = None) -> bytes:
    """Serialize PlanStatsDict to bitwise-reproducible RFC 8259 canonical JSON bytes.

    Applies:
    - Normalization of -0.0 to 0.0
    - Float sanitization (NaN/+Inf/-Inf to null)
    - 8-decimal precision float stabilization
    - Lexicographic key sorting (sort_keys=True)
    - Minimal compact separators (',', ':')
    - UTF-8 NFC Unicode normalization

    Args:
        stats: Intermediate plan statistics.
        path: Optional file path to write to (uses atomic write).

    Returns:
        UTF-8 encoded JSON bytes.
    """
    normalized = _normalize_floats(stats)

    # Serialize with sorted keys and minimal separators per RFC 8259
    raw_str = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    # Normalize to Unicode NFC form
    nfc_str = unicodedata.normalize("NFC", raw_str)
    normalized_bytes = nfc_str.encode("utf-8")

    if path is not None:
        p = pathlib.Path(path)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(normalized_bytes)
        tmp.replace(p)

    return normalized_bytes


def export_axes_csv(stats: PlanStatsDict, path: PathLike | None = None) -> str:
    """Export plan axes to a tidy tabular CSV string with schema columns.

    Format: long-format tidy table where each row is an axis decision.
    Includes unit metadata column ("elements") and semicolon-delimited lists.

    Args:
        stats: Intermediate plan statistics.
        path: Optional file path to write to (uses atomic write).

    Returns:
        CSV formatted string.
    """
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "name",
            "strategy",
            "cardinality",
            "batch_size",
            "reasoning",
            "memory_estimate_bytes",
            "varying_inputs",
            "dedup_ratio",
            "unique_count",
            "padded_count",
            "padding_waste",
            "bucket_count",
            "bucket_boundaries",
            "unit",
        ],
    )
    writer.writeheader()

    dedup_map = {d["axis_name"]: d for d in stats.get("dedup_stats", [])}
    bucket_map = {b["axis_name"]: b for b in stats.get("bucket_stats", [])}

    for axis in stats.get("axes", []):
        name = axis["name"]
        dedup = dedup_map.get(name)
        bucket = bucket_map.get(name)

        varying = ";".join(axis.get("varying_inputs", []))
        dedup_ratio = dedup["dedup_ratio"] if dedup else ""
        unique_count = dedup["unique_count"] if dedup else ""
        padded_count = dedup["padded_count"] if dedup else ""
        padding_waste = dedup["padding_waste"] if dedup else ""

        bucket_count = bucket["bucket_count"] if bucket else ""
        if bucket:
            bucket_boundaries = ";".join(str(x) for x in bucket.get("bucket_boundaries", []))
        else:
            bucket_boundaries = ""

        row = {
            "name": name,
            "strategy": axis["strategy"],
            "cardinality": axis["cardinality"],
            "batch_size": axis["batch_size"],
            "reasoning": axis["reasoning"],
            "memory_estimate_bytes": axis.get("memory_estimate_bytes") or "",
            "varying_inputs": varying,
            "dedup_ratio": dedup_ratio,
            "unique_count": unique_count,
            "padded_count": padded_count,
            "padding_waste": padding_waste,
            "bucket_count": bucket_count,
            "bucket_boundaries": bucket_boundaries,
            "unit": "elements",
        }
        writer.writerow(row)

    csv_str = output.getvalue()
    if path is not None:
        p = pathlib.Path(path)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            f.write(csv_str)
        tmp.replace(p)

    return csv_str


def export_axes_parquet(stats: PlanStatsDict, path: PathLike | None = None) -> bytes:
    """Export axes to Parquet bytes with schema metadata.

    Requires pyarrow. Raises ImportError if pyarrow is unavailable.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "export_axes_parquet requires pyarrow. Install with: pip install pyarrow"
        ) from exc

    df = plan_to_dataframe(stats)
    table = pa.Table.from_pandas(df)
    metadata = {
        b"xtrax_schema_version": b"2.0",
        b"generator": b"xtrax.eda.export",
        b"total_axes": str(stats.get("total_axes", 0)).encode("utf-8"),
    }
    existing_meta = table.schema.metadata or {}
    existing_meta.update(metadata)
    table = table.replace_schema_metadata(existing_meta)

    buf = io.BytesIO()
    pq.write_table(table, buf)
    parquet_bytes = buf.getvalue()

    if path is not None:
        p = pathlib.Path(path)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(parquet_bytes)
        tmp.replace(p)

    return parquet_bytes


def export_plan_artifacts(
    stats: PlanStatsDict,
    output_dir: PathLike,
    prefix: str = "plan",
    context: str = "",
) -> dict[str, Any]:
    """Emit a neutral, byte-deterministic hashed artifact bundle.

    Emits:
    - <prefix>.canonical.json: Bitwise reproducible PlanStatsDict
    - <prefix>_axes.csv: Tidy tabular long-format data
    - <prefix>_axes.parquet: Parquet format with embedded schema metadata (if pyarrow available)
    - <prefix>.vl.json: Declarative Vega-Lite v5 specification
    - <prefix>_manifest.json: Cryptographic manifest with SHA-256 digests

    Args:
        stats: Plan statistics dictionary.
        output_dir: Directory to place emitted files.
        prefix: Prefix for artifact file names.
        context: Opaque caller context string passed through to manifest.

    Returns:
        Manifest dictionary.
    """
    out_dir = str(output_dir)
    os.makedirs(out_dir, exist_ok=True)

    artifacts: dict[str, dict[str, Any]] = {}

    def _record_file(filename: str, content: bytes) -> None:
        filepath = os.path.join(out_dir, filename)
        tmp_path = filepath + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(content)
        os.replace(tmp_path, filepath)
        sha256 = hashlib.sha256(content).hexdigest()
        artifacts[filename] = {"bytes": len(content), "sha256": sha256}

    # 1. Canonical JSON
    json_bytes = export_canonical_json(stats)
    _record_file(f"{prefix}.canonical.json", json_bytes)

    # 2. Tidy CSV
    csv_bytes = export_axes_csv(stats).encode("utf-8")
    _record_file(f"{prefix}_axes.csv", csv_bytes)

    # 3. Parquet (optional, only if pyarrow is installed)
    try:
        parquet_bytes = export_axes_parquet(stats)
        _record_file(f"{prefix}_axes.parquet", parquet_bytes)
    except ImportError:
        pass

    # 4. Vega-Lite Spec
    vl_spec = generate_vega_spec(stats)
    vl_bytes = json.dumps(vl_spec, indent=2).encode("utf-8")
    _record_file(f"{prefix}.vl.json", vl_bytes)

    # 5. Manifest
    manifest = {
        "schema_version": "2.0",
        "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "generator": "xtrax.eda.export",
        "context": context,
        "total_axes": stats.get("total_axes", 0),
        "artifacts": artifacts,
    }
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
    manifest_path = os.path.join(out_dir, f"{prefix}_manifest.json")
    tmp_manifest = manifest_path + ".tmp"
    with open(tmp_manifest, "wb") as f:
        f.write(manifest_bytes)
    os.replace(tmp_manifest, manifest_path)

    return manifest


def plan_to_dataframe(stats: PlanStatsDict) -> Any:
    """Convert PlanStatsDict to a pandas DataFrame (backward-compatibility wrapper)."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "xtrax.eda.export requires visualization extras. "
            "Install with: pip install xtrax[eda]"
        ) from exc

    rows = []
    for axis_entry in stats.get("axes", []):
        row: dict[str, Any] = {
            "name": axis_entry["name"],
            "strategy": axis_entry["strategy"],
            "cardinality": axis_entry["cardinality"],
            "batch_size": axis_entry["batch_size"],
            "reasoning": axis_entry["reasoning"],
            "memory_estimate_bytes": axis_entry.get("memory_estimate_bytes"),
            "dedup_ratio": pd.NA,
            "unique_count": pd.NA,
            "padded_count": pd.NA,
            "padding_waste": pd.NA,
            "bucket_count": pd.NA,
            "bucket_boundaries": pd.NA,
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "name",
                "strategy",
                "cardinality",
                "batch_size",
                "reasoning",
                "memory_estimate_bytes",
                "dedup_ratio",
                "unique_count",
                "padded_count",
                "padding_waste",
                "bucket_count",
                "bucket_boundaries",
            ]
        )

    dedup_by_name = {entry["axis_name"]: entry for entry in stats.get("dedup_stats", [])}
    for row in rows:
        axis_name = row["name"]
        if axis_name in dedup_by_name:
            dedup_entry = dedup_by_name[axis_name]
            row["dedup_ratio"] = dedup_entry["dedup_ratio"]
            row["unique_count"] = dedup_entry["unique_count"]
            row["padded_count"] = dedup_entry["padded_count"]
            row["padding_waste"] = dedup_entry["padding_waste"]

    bucket_by_name = {entry["axis_name"]: entry for entry in stats.get("bucket_stats", [])}
    for row in rows:
        axis_name = row["name"]
        if axis_name in bucket_by_name:
            bucket_entry = bucket_by_name[axis_name]
            row["bucket_count"] = bucket_entry["bucket_count"]
            row["bucket_boundaries"] = bucket_entry["bucket_boundaries"]

    return pd.DataFrame(rows)


__all__ = [
    "export_canonical_json",
    "export_axes_csv",
    "export_axes_parquet",
    "export_plan_artifacts",
    "plan_to_dataframe",
]
