"""Dataset loading and Grain input pipelines.

Grain is optional (the ``data`` extra). Importing ``xtrax.data`` does not
import it; building a pipeline raises ``ImportError`` with an install hint
when grain is missing.
"""

from xtrax.data.module import DataModule
from xtrax.data.pipeline import build_input_pipeline, create_distributed_pipeline
from xtrax.data.profile import (
    InputPipelineControlError,
    InputPipelineProfile,
    PipelineControls,
    profile_input_pipeline,
)

__all__ = [
    "DataModule",
    "InputPipelineControlError",
    "InputPipelineProfile",
    "PipelineControls",
    "build_input_pipeline",
    "create_distributed_pipeline",
    "profile_input_pipeline",
]
