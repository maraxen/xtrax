"""Manifest schemas, canonical JSON serialization, and pure host planner for xtrax.campaign.

Provides canonical hashing, aminx input hash parity, content-addressed unit identity,
and the pure host manifest planner (0 GPUs, deterministic canonical JSON output).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from xtrax.campaign.spec import (
    CampaignSpec,
    _check_no_callables,
)

if TYPE_CHECKING:
    pass

#: Keys owned by the campaign runner that must be mapped to normalized names
#: in aminx-compatible parameter hashing.
CAMPAIGN_OWNED_KEYS: frozenset[str] = frozenset(
    {
        "grid_mode",
        "samples_chunk_size",
        "job_id",
        "chunk_id",
        "sample_start",
        "sample_count",
        "output_path",
        "output_zarr_path",
        "output_h5_path",
    }
)


def normalize_json_value(value: Any) -> Any:  # noqa: ANN401
    """Recursively coerce value into JSON-safe, canonically-ordered form.

    - Floats (both Python float and NumPy floating) normalize -0.0 strictly to 0.0.
    - Dict keys are stringified and sorted via key=lambda kv: str(kv[0]) to avoid TypeError.
    - Strings are NFC-normalized; bytes are decoded as UTF-8.
    - NumPy scalars and arrays are converted to native Python primitives and lists.
    """
    if isinstance(value, float):
        return 0.0 if value == 0.0 else value
    if isinstance(value, np.floating):
        val = float(value.item())
        return 0.0 if val == 0.0 else val
    if isinstance(value, np.generic):
        return normalize_json_value(value.item())
    if isinstance(value, np.ndarray):
        return [normalize_json_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [normalize_json_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(k): normalize_json_value(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize value to deterministic, sorted-key, NFC-normalized JSON bytes.

    Separators are strictly (',', ':') with no extraneous whitespace.
    Non-finite floats (NaN, +inf, -inf) raise ValueError.
    """
    normalized = normalize_json_value(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class ResourceRequirements:
    """Resource budget calculated statically during host planning."""

    gpu_memory_bytes: int
    cpu_cores: int = 1
    timeout_seconds: int = 3600
    padding_bucket_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "gpu_memory_bytes": self.gpu_memory_bytes,
            "cpu_cores": self.cpu_cores,
            "timeout_seconds": self.timeout_seconds,
        }
        if self.padding_bucket_id is not None:
            d["padding_bucket_id"] = self.padding_bucket_id
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResourceRequirements:
        return cls(
            gpu_memory_bytes=int(data.get("gpu_memory_bytes", 0)),
            cpu_cores=int(data.get("cpu_cores", 1)),
            timeout_seconds=int(data.get("timeout_seconds", 3600)),
            padding_bucket_id=data.get("padding_bucket_id"),
        )


@dataclass(frozen=True, slots=True)
class ManifestRow:
    """An individual, content-addressed compute unit."""

    manifest_row_hash: str  # 64-character lowercase hex SHA-256 (global unit identity)
    input_hash: str  # 64-character lowercase hex SHA-256 (aminx parameter parity hash)
    campaign_id: str
    job_id: str
    chunk_id: int
    sample_start: int
    sample_count: int
    target_input: str
    arm_id: str
    arm_backend: str
    checkpoint_id: str
    fixed_policy: str
    state_weight_profile: str
    seed: int
    parameters: dict[str, Any]
    output_locator: str
    resource_requirements: ResourceRequirements

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_row_hash": self.manifest_row_hash,
            "input_hash": self.input_hash,
            "campaign_id": self.campaign_id,
            "job_id": self.job_id,
            "chunk_id": self.chunk_id,
            "sample_start": self.sample_start,
            "sample_count": self.sample_count,
            "target_input": self.target_input,
            "arm_id": self.arm_id,
            "arm_backend": self.arm_backend,
            "checkpoint_id": self.checkpoint_id,
            "fixed_policy": self.fixed_policy,
            "state_weight_profile": self.state_weight_profile,
            "seed": self.seed,
            "parameters": self.parameters,
            "output_locator": self.output_locator,
            "resource_requirements": self.resource_requirements.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ManifestRow:
        res = data.get("resource_requirements", {})
        if isinstance(res, dict):
            res_req = ResourceRequirements.from_dict(res)
        elif isinstance(res, ResourceRequirements):
            res_req = res
        else:
            res_req = ResourceRequirements(gpu_memory_bytes=0)

        return cls(
            manifest_row_hash=str(data["manifest_row_hash"]),
            input_hash=str(data["input_hash"]),
            campaign_id=str(data["campaign_id"]),
            job_id=str(data["job_id"]),
            chunk_id=int(data["chunk_id"]),
            sample_start=int(data["sample_start"]),
            sample_count=int(data["sample_count"]),
            target_input=str(data["target_input"]),
            arm_id=str(data["arm_id"]),
            arm_backend=str(data["arm_backend"]),
            checkpoint_id=str(data["checkpoint_id"]),
            fixed_policy=str(data.get("fixed_policy", "none")),
            state_weight_profile=str(data.get("state_weight_profile", "equal")),
            seed=int(data.get("seed", 42)),
            parameters=dict(data.get("parameters", {})),
            output_locator=str(data["output_locator"]),
            resource_requirements=res_req,
        )


@dataclass(frozen=True, slots=True)
class ManifestMetadata:
    """Metadata describing a generated campaign manifest."""

    campaign_id: str
    created_at_utc: str
    planner_version: str
    spec_sha256: str
    total_rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "created_at_utc": self.created_at_utc,
            "planner_version": self.planner_version,
            "spec_sha256": self.spec_sha256,
            "total_rows": self.total_rows,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ManifestMetadata:
        return cls(
            campaign_id=str(data["campaign_id"]),
            created_at_utc=str(data["created_at_utc"]),
            planner_version=str(data["planner_version"]),
            spec_sha256=str(data["spec_sha256"]),
            total_rows=int(data["total_rows"]),
        )


@dataclass(frozen=True, slots=True)
class CampaignManifest:
    """Complete planned campaign manifest containing all execution units."""

    schema_version: str  # Fixed: "campaign_manifest_v1"
    metadata: ManifestMetadata
    rows: list[ManifestRow]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metadata": self.metadata.to_dict(),
            "rows": [row.to_dict() for row in self.rows],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignManifest:
        schema_version = str(data.get("schema_version", "campaign_manifest_v1"))
        if schema_version != "campaign_manifest_v1":
            raise ValueError(f"Unsupported schema_version: {schema_version}")
        metadata = ManifestMetadata.from_dict(data["metadata"])
        rows = [ManifestRow.from_dict(r) for r in data.get("rows", [])]
        return cls(
            schema_version=schema_version,
            metadata=metadata,
            rows=rows,
        )


def aminx_compatible_row_payload(
    spec_dict: dict[str, Any],
    campaign_id: str,
    job_id: str,
    chunk_id: int,
    sample_start: int,
    sample_count: int,
    checkpoint_id: str,
    fixed_policy: str,
    state_weight_profile: str,
    seed: int,
) -> dict[str, Any]:
    """Construct normalized dictionary matching aminx compute_unit_input_hash exactly."""
    clean_params = {
        str(k): normalize_json_value(v)
        for k, v in spec_dict.items()
        if k not in CAMPAIGN_OWNED_KEYS
    }

    return {
        "campaign_id": campaign_id,
        "job_id": job_id,
        "chunk_id": chunk_id,
        "sample_start": sample_start,
        "sample_count": sample_count,
        "checkpoint_id": checkpoint_id,
        "fixed_policy": fixed_policy,
        "state_weight_profile": state_weight_profile,
        "seed": seed,
        "params": clean_params,
    }


def compute_unit_input_hash(spec_dict: dict[str, Any], **kwargs: Any) -> str:
    """Compute 64-char lowercase hex SHA-256 for aminx parameter parity."""
    payload = aminx_compatible_row_payload(spec_dict, **kwargs)
    payload_bytes = canonical_json_bytes(payload)
    return hashlib.sha256(payload_bytes).hexdigest()


def compute_manifest_row_hash(normalized_row_payload: dict[str, Any]) -> str:
    """Compute 64-char lowercase hex SHA-256 over full canonical unit payload."""
    payload_bytes = canonical_json_bytes(normalized_row_payload)
    return hashlib.sha256(payload_bytes).hexdigest()


def build_manifest_row(
    campaign_id: str,
    job_id: str,
    chunk_id: int,
    sample_start: int,
    sample_count: int,
    target_input: str,
    arm_id: str,
    arm_backend: str,
    checkpoint_id: str,
    fixed_policy: str = "none",
    state_weight_profile: str = "equal",
    seed: int = 42,
    parameters: dict[str, Any] | None = None,
    output_locator: str | None = None,
    resource_requirements: ResourceRequirements | None = None,
) -> ManifestRow:
    """Build a content-addressed ManifestRow calculating input_hash and manifest_row_hash."""
    clean_params = dict(parameters) if parameters else {}
    _check_no_callables(clean_params, path=f"ManifestRow({job_id}).parameters")
    normalized_params = normalize_json_value(clean_params)

    input_hash = compute_unit_input_hash(
        spec_dict=normalized_params,
        campaign_id=campaign_id,
        job_id=job_id,
        chunk_id=chunk_id,
        sample_start=sample_start,
        sample_count=sample_count,
        checkpoint_id=checkpoint_id,
        fixed_policy=fixed_policy,
        state_weight_profile=state_weight_profile,
        seed=seed,
    )

    row_identity_payload = {
        "campaign_id": campaign_id,
        "job_id": job_id,
        "chunk_id": chunk_id,
        "sample_start": sample_start,
        "sample_count": sample_count,
        "target_input": target_input,
        "arm_id": arm_id,
        "arm_backend": arm_backend,
        "checkpoint_id": checkpoint_id,
        "fixed_policy": fixed_policy,
        "state_weight_profile": state_weight_profile,
        "seed": seed,
        "parameters": normalized_params,
    }
    manifest_row_hash = compute_manifest_row_hash(row_identity_payload)

    if output_locator is None:
        output_locator = f"{campaign_id}/{manifest_row_hash}.zarr"

    if resource_requirements is None:
        resource_requirements = ResourceRequirements(
            gpu_memory_bytes=0,
            cpu_cores=1,
            timeout_seconds=3600,
        )

    return ManifestRow(
        manifest_row_hash=manifest_row_hash,
        input_hash=input_hash,
        campaign_id=campaign_id,
        job_id=job_id,
        chunk_id=chunk_id,
        sample_start=sample_start,
        sample_count=sample_count,
        target_input=target_input,
        arm_id=arm_id,
        arm_backend=arm_backend,
        checkpoint_id=checkpoint_id,
        fixed_policy=fixed_policy,
        state_weight_profile=state_weight_profile,
        seed=seed,
        parameters=normalized_params,
        output_locator=output_locator,
        resource_requirements=resource_requirements,
    )


def plan_campaign(
    spec: CampaignSpec,
    output_manifest_path: str | Path | None = None,
    created_at_utc: str | None = None,
    planner_version: str | None = None,
) -> CampaignManifest:
    """Plan a declarative CampaignSpec into a concrete CampaignManifest.

    This function performs pure host computation: 0 GPUs, 0 network side-effects,
    0 child processes, with deterministic canonical JSON output.
    """
    # Enforce pure-data partitioning across all spec inputs
    _check_no_callables(spec.parameters, path="CampaignSpec.parameters")
    _check_no_callables(spec.grid.axes, path="CampaignSpec.grid.axes")
    _check_no_callables(spec.metadata, path="CampaignSpec.metadata")
    for arm in spec.arms:
        _check_no_callables(arm.parameters, path=f"ArmSpec({arm.arm_id}).parameters")
        _check_no_callables(
            arm.resource_overrides, path=f"ArmSpec({arm.arm_id}).resource_overrides"
        )

    # Expand parameter grid combinations in deterministic sorted order
    axes_keys = sorted(spec.grid.axes.keys())
    if axes_keys:
        axes_values = [spec.grid.axes[k] for k in axes_keys]
        combinations = [dict(zip(axes_keys, p)) for p in itertools.product(*axes_values)]
    else:
        combinations = [{}]

    rows: list[ManifestRow] = []

    for t_idx, target_input in enumerate(spec.target_inputs):
        for a_idx, arm in enumerate(spec.arms):
            for c_idx, combo in enumerate(combinations):
                merged_params: dict[str, Any] = {}
                merged_params.update(spec.parameters)
                merged_params.update(arm.parameters)
                merged_params.update(combo)

                gpu_mem = int(arm.resource_overrides.get("gpu_memory_bytes", 0))
                cpu_cores = int(arm.resource_overrides.get("cpu_cores", 1))
                timeout_s = int(arm.resource_overrides.get("timeout_seconds", 3600))
                bucket_id = arm.resource_overrides.get("padding_bucket_id", None)
                res_req = ResourceRequirements(
                    gpu_memory_bytes=gpu_mem,
                    cpu_cores=cpu_cores,
                    timeout_seconds=timeout_s,
                    padding_bucket_id=bucket_id,
                )

                num_chunks = max(
                    1, (spec.samples_per_target + spec.chunk_size - 1) // spec.chunk_size
                )
                for chunk_id in range(num_chunks):
                    sample_start = chunk_id * spec.chunk_size
                    sample_count = min(spec.chunk_size, spec.samples_per_target - sample_start)
                    if sample_count <= 0:
                        continue

                    job_id = (
                        f"target_{t_idx:03d}_arm_{arm.arm_id}"
                        f"_combo_{c_idx:03d}_chunk_{chunk_id:03d}"
                    )

                    row = build_manifest_row(
                        campaign_id=spec.campaign_id,
                        job_id=job_id,
                        chunk_id=chunk_id,
                        sample_start=sample_start,
                        sample_count=sample_count,
                        target_input=target_input,
                        arm_id=arm.arm_id,
                        arm_backend=arm.backend,
                        checkpoint_id=arm.checkpoint_id,
                        fixed_policy=arm.fixed_policy,
                        state_weight_profile=arm.state_weight_profile,
                        seed=spec.seed,
                        parameters=merged_params,
                        output_locator=None,
                        resource_requirements=res_req,
                    )
                    rows.append(row)

    if created_at_utc is not None:
        timestamp = created_at_utc
    elif "created_at_utc" in spec.metadata:
        timestamp = str(spec.metadata["created_at_utc"])
    else:
        timestamp = "1970-01-01T00:00:00Z"

    if planner_version is None:
        try:
            import xtrax

            ver = getattr(xtrax, "__version__", "0.5.0")
        except ImportError:
            ver = "0.5.0"
        planner_version = f"xtrax@{ver}"

    spec_dict = spec.to_dict()
    spec_sha256 = hashlib.sha256(canonical_json_bytes(spec_dict)).hexdigest()

    metadata = ManifestMetadata(
        campaign_id=spec.campaign_id,
        created_at_utc=timestamp,
        planner_version=planner_version,
        spec_sha256=spec_sha256,
        total_rows=len(rows),
    )
    manifest = CampaignManifest(
        schema_version="campaign_manifest_v1",
        metadata=metadata,
        rows=rows,
    )

    if output_manifest_path is not None:
        p = Path(output_manifest_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(canonical_json_bytes(manifest.to_dict()))

    return manifest


plan = plan_campaign
