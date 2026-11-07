"""xtrax.campaign.executors — Standardized campaign executors.

Providing scheduler-agnostic execution harnesses:
- LocalSequentialExecutor: single-process loop with JAX cache & memory hygiene
- LocalProcessPoolExecutor: multiprocessing pool with chunked worker recycling
- SlurmArrayExecutor: pure host script generator for SLURM arrays with sparse resume
"""

from xtrax.campaign.executors.local import (
    LocalProcessPoolExecutor,
    LocalSequentialExecutor,
)
from xtrax.campaign.executors.protocol import (
    CampaignExecutor,
    CampaignSummary,
)
from xtrax.campaign.executors.slurm import (
    SlurmArrayExecutor,
    format_slurm_array_indices,
    get_pending_row_indices,
)

__all__ = [
    "CampaignExecutor",
    "CampaignSummary",
    "LocalProcessPoolExecutor",
    "LocalSequentialExecutor",
    "SlurmArrayExecutor",
    "format_slurm_array_indices",
    "get_pending_row_indices",
]
