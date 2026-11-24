"""Optional bathos catalog bridge for kernel profiling runs (unshipped controller).

Lives exclusively under ``controller/`` so ``src/xtrax/`` keeps zero imports
or runtime coupling to ``bathos``. All ``bathos`` imports are deferred to call
time so this module stays importable when ``xtrax[controller]`` is absent.
"""

import os
from pathlib import Path
from typing import Any


def resolve_bathos_provenance(run_id: str, *, catalog_dir: str = "") -> dict[str, Any]:
    """Query ``bathos.query.get_run`` for ``run_id`` and return normalized provenance.

    Args:
        run_id: Bathos run identifier.
        catalog_dir: Optional bathos catalog directory override. When empty,
            resolves via ``BTH_CATALOG_DIR`` or ``bathos.config.default_catalog_dir()``.

    Returns:
        Dictionary containing catalog run metadata (empty dict when ``run_id``
        is not found).

    Raises:
        ImportError: When ``bathos`` (``xtrax[controller]``) is not installed.
    """
    try:
        import bathos.config
        import bathos.query
    except ImportError as exc:
        raise ImportError("Install xtrax[controller] to query bathos catalog") from exc

    env_dir = os.environ.get("BTH_CATALOG_DIR", "").strip()
    if catalog_dir.strip():
        resolved_dir = Path(catalog_dir.strip())
    elif env_dir:
        resolved_dir = Path(env_dir)
    else:
        resolved_dir = Path(bathos.config.default_catalog_dir())

    run_obj = bathos.query.get_run(run_id, resolved_dir)
    if run_obj is None:
        return {}
    return {
        "run_id": getattr(run_obj, "id", getattr(run_obj, "run_id", run_id)),
        "campaign_id": getattr(run_obj, "campaign_id", None),
        "git_sha": getattr(run_obj, "git_hash", getattr(run_obj, "git_sha", None)),
        "sidecar_sha256": getattr(run_obj, "sidecar_sha256", None),
        "status": getattr(run_obj, "status", None),
        "catalog_dir": str(resolved_dir),
    }


__all__ = ["resolve_bathos_provenance"]
