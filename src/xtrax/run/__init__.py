"""xtrax.run — execution-time configuration layer."""

from xtrax.run.resolver import FeatureBatch, InputResolver, RuntimeBundle
from xtrax.run.sink import SinkSpec, make_sink
from xtrax.run.spec import RunSpec
from xtrax.run.zarr_integrity import (
    canonical_json_bytes,
    fsync_directory,
    fsync_file,
    fsync_tree,
    normalize_json_value,
    update_array_digest,
    update_zarr_node_digest,
    zarr_content_digest,
)
from xtrax.run.zarr_sink import ZarrStagingSink

__all__ = [
    "RunSpec",
    "InputResolver",
    "RuntimeBundle",
    "FeatureBatch",
    "SinkSpec",
    "make_sink",
    "ZarrStagingSink",
    "canonical_json_bytes",
    "normalize_json_value",
    "update_array_digest",
    "update_zarr_node_digest",
    "zarr_content_digest",
    "fsync_file",
    "fsync_directory",
    "fsync_tree",
]
