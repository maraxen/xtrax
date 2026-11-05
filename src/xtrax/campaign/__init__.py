"""xtrax.campaign — Distributed Campaign Runner & Execution Engine.

A standardized, distributed, scheduler-agnostic campaign execution engine
for high-throughput computational campaigns across multi-node clusters.
"""

from xtrax.campaign.manifest import (
    CAMPAIGN_OWNED_KEYS,
    CampaignManifest,
    ManifestMetadata,
    ManifestRow,
    ResourceRequirements,
    aminx_compatible_row_payload,
    build_manifest_row,
    canonical_json_bytes,
    compute_manifest_row_hash,
    compute_unit_input_hash,
    normalize_json_value,
    plan,
    plan_campaign,
)
from xtrax.campaign.spec import (
    ArmSpec,
    CampaignSpec,
    ParamGrid,
    SpecPartitionError,
)

__all__ = [
    "CAMPAIGN_OWNED_KEYS",
    "ArmSpec",
    "CampaignManifest",
    "CampaignSpec",
    "ManifestMetadata",
    "ManifestRow",
    "ParamGrid",
    "ResourceRequirements",
    "SpecPartitionError",
    "aminx_compatible_row_payload",
    "build_manifest_row",
    "canonical_json_bytes",
    "compute_manifest_row_hash",
    "compute_unit_input_hash",
    "normalize_json_value",
    "plan",
    "plan_campaign",
]
