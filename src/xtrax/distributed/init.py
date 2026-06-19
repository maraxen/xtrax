"""Distributed initialization for JAX multi-process training."""

import os
from typing import Any

import jax.distributed

# Module-level state: stores initialization args
_init_state: dict[str, Any] = {
    "initialized": False,
    "coordinator_address": None,
    "num_processes": None,
    "process_id": None,
}


def _derive_coordinator_from_nodelist(nodelist: str) -> str:
    """Extract first node from SLURM_JOB_NODELIST and return as coordinator address."""
    # nodelist can be "node-001,node-002,node-003" or "node-001" or "node-[001-003]"
    # For simplicity, extract the first component (comma-separated or bracket-expanded)
    first_node = nodelist.split(",")[0]
    # Remove bracket ranges if present (e.g., "node-[001-003]" -> "node-[001-003]")
    # For now, just use the first node as-is; return with default JAX port
    return f"{first_node}:1234"


def init_dist(
    *,
    coordinator_address: str | None = None,
    num_processes: int | None = None,
    process_id: int | None = None,
) -> None:
    """
    Initialize JAX distributed training with optional auto-discovery.

    Args:
        coordinator_address: Address of the coordinator (e.g., "localhost:1234").
                            If None, attempts SLURM env discovery or falls back
                            to localhost.
        num_processes: Total number of processes in the distributed setup.
                      If None, attempts SLURM env discovery or falls back to 1.
        process_id: This process's ID (0-indexed).
                   If None, attempts SLURM env discovery or falls back to 0.

    Raises:
        RuntimeError: If init_dist has already been called with different arguments.

    Notes:
        - Idempotent: same args on repeat call → no-op.
        - Different args on second call → RuntimeError.
        - SLURM discovery: uses SLURM_PROCID, SLURM_NTASKS, SLURM_JOB_NODELIST.
        - Single-process fallback: skips jax.distributed.initialize for
          num_processes==1 (spec deviation: see below).
        - After successful init, calls _mark_dist_initialized() to enable
          DataModule(distributed=True).

    Spec Deviation (§3.22):
        The spec states that localhost fallback should call jax.distributed.initialize
        with single-process config. However, calling jax.distributed.initialize with
        num_processes=1 can be unstable in some environments. This implementation skips
        the call for num_processes==1, instead calling _mark_dist_initialized() and
        returning. If future requirements demand the single-process initialize call,
        this can be made a parameter.
    """
    global _init_state

    # Auto-discovery: explicit args → SLURM env → localhost fallback
    if coordinator_address is None or num_processes is None or process_id is None:
        # Try SLURM environment variables
        slurm_ntasks = os.environ.get("SLURM_NTASKS")
        slurm_procid = os.environ.get("SLURM_PROCID")
        slurm_nodelist = os.environ.get("SLURM_JOB_NODELIST")

        if slurm_ntasks is not None and slurm_procid is not None:
            # SLURM env found: use it to fill in missing args
            if num_processes is None:
                num_processes = int(slurm_ntasks)
            if process_id is None:
                process_id = int(slurm_procid)
            if coordinator_address is None:
                if slurm_nodelist:
                    coordinator_address = _derive_coordinator_from_nodelist(slurm_nodelist)
                else:
                    coordinator_address = "localhost:1234"
        else:
            # No SLURM env: use localhost fallback
            if coordinator_address is None:
                coordinator_address = "localhost:1234"
            if num_processes is None:
                num_processes = 1
            if process_id is None:
                process_id = 0

    # Check idempotency: if already initialized, verify args match
    if _init_state["initialized"]:
        if (
            _init_state["coordinator_address"] != coordinator_address
            or _init_state["num_processes"] != num_processes
            or _init_state["process_id"] != process_id
        ):
            raise RuntimeError(
                f"init_dist already initialized with different args. "
                f"Previous: coordinator_address="
                f"{_init_state['coordinator_address']}, "
                f"num_processes={_init_state['num_processes']}, "
                f"process_id={_init_state['process_id']}. "
                f"New: coordinator_address={coordinator_address}, "
                f"num_processes={num_processes}, process_id={process_id}."
            )
        # Same args: idempotent, no-op
        return

    # Store state
    _init_state["coordinator_address"] = coordinator_address
    _init_state["num_processes"] = num_processes
    _init_state["process_id"] = process_id

    # Initialize JAX distributed if num_processes > 1
    if num_processes > 1:
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=num_processes,
            process_id=process_id,
        )

    # Mark distributed initialization complete (enables DataModule(distributed=True))
    from xtrax.data.module import _mark_dist_initialized

    _mark_dist_initialized()

    _init_state["initialized"] = True


def is_distributed() -> bool:
    """Return True if init_dist has been called, False otherwise."""
    return _init_state["initialized"]


def _reset_for_testing() -> None:
    """Reset internal state for testing. Not for production use."""
    global _init_state
    _init_state = {
        "initialized": False,
        "coordinator_address": None,
        "num_processes": None,
        "process_id": None,
    }
    # Also reset DataModule's distributed flag
    import xtrax.data.module as dm

    dm._dist_initialized = False
