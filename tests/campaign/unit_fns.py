"""Importable unit functions for tests that drive ``xtrax campaign run --fn``.

Kept outside ``test_*`` modules so ``load_fn`` and spawned pool workers can import them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from xtrax.campaign.manifest import ManifestRow

#: ``--fn`` import path for :func:`array_unit`.
ARRAY_UNIT_FN = "tests.campaign.unit_fns:array_unit"


def array_unit(row: ManifestRow) -> dict[str, Any]:
    """Return one seed-dependent array and a scalar; the executor persists both."""
    values = np.arange(8, dtype=np.float32) + float(row.seed)
    return {"values": values, "seed": int(row.seed)}


#: Env var naming the campaign output dir for :func:`commit_sibling_then_die_unit`.
SIBLING_OUT_ENV = "XTRAX_TEST_SIBLING_OUT"


def commit_sibling_then_die_unit(row: ManifestRow) -> dict[str, Any]:
    """On chunk 0, commit row 1 of the same mock campaign, then SIGKILL this worker.

    Leaves row 1 valid on disk while the pool still has it queued, so the pool breaks
    with a unit whose result never reached the parent but whose store is committed.
    """
    import os
    import signal

    if row.chunk_id == 0:
        from tests.campaign.executors.test_local import _create_mock_manifest
        from xtrax.campaign.executors.local import LocalSequentialExecutor

        manifest = _create_mock_manifest(num_rows=2, campaign_id=row.campaign_id)
        LocalSequentialExecutor().run(
            manifest, array_unit, output_dir=os.environ[SIBLING_OUT_ENV], row_indices=[1]
        )
        os.kill(os.getpid(), signal.SIGKILL)
    return array_unit(row)
