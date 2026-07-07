"""xtrax.run — execution-time configuration layer."""

from xtrax.run.resolver import FeatureBatch, InputResolver, RuntimeBundle
from xtrax.run.sink import SinkSpec, make_sink
from xtrax.run.spec import RunSpec
from xtrax.run.zarr_sink import ZarrStagingSink

__all__ = [
    "RunSpec",
    "InputResolver",
    "RuntimeBundle",
    "FeatureBatch",
    "SinkSpec",
    "make_sink",
    "ZarrStagingSink",
]
