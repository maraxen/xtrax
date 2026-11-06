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
