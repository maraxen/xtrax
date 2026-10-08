"""Shared SIGKILL harness for the zarr_commit kill tests (not itself collected)."""

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from xtrax.run import zarr_commit as zc

N_KEYS = 30
N_ARRAYS = 20
N_DELAYS = 12
IDENTITY = {"type": "kill-test"}
PREFIXES = (("chunks",),)

# --------------------------------------------------------------------------------------
# Kill harness
# --------------------------------------------------------------------------------------

_ATOMIC_CHILD = f"""
import sys
from pathlib import Path

import numpy as np

from xtrax.run import zarr_commit as zc

store = Path(sys.argv[1])
zc.create_store(
    store,
    identity_payload={IDENTITY!r},
    creator_run_id="creator",
    prefixes={PREFIXES!r},
    writer_id="child",
)
print("ready", flush=True)
for i in range({N_KEYS}):
    staged = zc.staging_root(store) / "child" / f"k{{i}}"
    arrays = {{f"a{{j}}": np.full(8, i * 100 + j, dtype=np.int64) for j in range({N_ARRAYS})}}
    zc.write_staged_group(staged, arrays)
    zc.commit_key(
        store,
        ("chunks", f"k{{i}}"),
        staged,
        input_digest=f"digest{{i}}",
        run_id="child-run",
        env={{}},
    )
"""

_INPLACE_CHILD = f"""
import sys
import time
from pathlib import Path

import numpy as np
import zarr

from xtrax.run import zarr_commit as zc

store = Path(sys.argv[1])
zc.create_store(
    store,
    identity_payload={IDENTITY!r},
    creator_run_id="creator",
    prefixes={PREFIXES!r},
    writer_id="child",
)
print("ready", flush=True)
for i in range({N_KEYS}):
    target = store / "chunks" / f"k{{i}}"
    group = zarr.open_group(str(target), mode="w")
    for j in range({N_ARRAYS}):
        arr = group.create_array(name=f"a{{j}}", shape=(8,), dtype="int64", chunks=(8,))
        arr[...] = np.full(8, i * 100 + j, dtype=np.int64)
        time.sleep(0.003)
    group.attrs["done"] = True
"""


def _spawn(script: Path, store: Path) -> subprocess.Popen[str]:
    env = {k: v for k, v in os.environ.items() if k != zc.FAULT_ENV}
    return subprocess.Popen(
        [sys.executable, str(script), str(store)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )


def _wait_ready(proc: subprocess.Popen[str]) -> None:
    assert proc.stdout is not None
    line = proc.stdout.readline()
    if line.strip() != "ready":
        proc.kill()
        err = proc.stderr.read() if proc.stderr else ""
        raise AssertionError(f"child never became ready: {line!r} {err}")


def _measure_duration(tmp_path: Path, script: Path) -> float:
    """Wall time from the child's 'ready' line to its clean exit."""
    store = tmp_path / "store_duration"
    proc = _spawn(script, store)
    _wait_ready(proc)
    t0 = time.monotonic()
    rc = proc.wait(timeout=300)
    duration = time.monotonic() - t0
    assert rc == 0, proc.stderr.read() if proc.stderr else ""
    return duration


def _kill_grid(tmp_path: Path, script: Path, duration: float) -> list[Path]:
    """Run the child N_DELAYS times, SIGKILLing at evenly spaced delays in (0, D).

    Runs are executed concurrently (each in its own store) to bound wall time. Each
    child's delay is measured from its own "ready" line, and concurrent contention
    can only slow a child relative to the solo duration D, so a kill at delay <= D
    still lands mid-run (never after completion).
    """

    def one(j: int) -> Path:
        delay = duration * (j + 1) / (N_DELAYS + 1)
        store = tmp_path / f"store_kill_{j}"
        proc = _spawn(script, store)
        _wait_ready(proc)
        time.sleep(delay)
        proc.kill()
        proc.wait(timeout=60)
        return store

    with ThreadPoolExecutor(max_workers=N_DELAYS) as pool:
        return list(pool.map(one, range(N_DELAYS)))


def _key_dirs(store: Path) -> list[Path]:
    chunks = store / "chunks"
    return sorted(p for p in chunks.iterdir() if p.is_dir()) if chunks.exists() else []
