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


#: Env var naming a scratch dir that :func:`fail_fast_or_sleep_unit` uses to order its rows.
SYNC_DIR_ENV = "XTRAX_TEST_SYNC_DIR"


def fail_fast_or_sleep_unit(row: ManifestRow) -> dict[str, Any]:
    """Chunk 0 fails at once; any other chunk runs for 60 s and does not stop on SIGTERM.

    Ignoring SIGTERM models a unit blocked in native code (an XLA computation, a C
    loop), where Python signal handlers do not run until control returns. With two
    workers and ``max_failures=1``, the early exit SIGTERMs the idle worker, which dies
    and breaks the pool, while the busy worker keeps running.
    """
    import os
    import signal
    import time
    from pathlib import Path

    started = Path(os.environ[SYNC_DIR_ENV]) / "sleeper_started"
    if row.chunk_id == 0:
        # Fail only once the sleeper is running, so the early exit finds it mid-unit
        # instead of cancelling it before it starts.
        deadline = time.monotonic() + 30.0
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        msg = "fail fast"
        raise RuntimeError(msg)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    started.touch()
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        time.sleep(0.05)
    return array_unit(row)
