"""Race tests for xtrax.run.zarr_commit.

* T-X2: two processes race ``commit_key`` on the same key.
* Store-creation race: many processes race ``create_store``.

(The SIGKILL tests live in ``test_zarr_commit_kill_atomic.py`` / ``..._control.py``.)
"""

import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from xtrax.run import zarr_commit as zc

IDENTITY = {"type": "kill-test"}
PREFIXES = (("chunks",),)

# --------------------------------------------------------------------------------------
# Commit race (T-X2)
# --------------------------------------------------------------------------------------

N_SAME = 50
N_DIFF = 10


def _race_worker(wid: int, store_str: str, barrier: Any, out_q: Any) -> None:
    """Stage a private copy of each key, meet at the barrier, then commit."""
    store = Path(store_str)
    arrays = {"arr": np.arange(16, dtype=np.int64), "m": np.ones((3, 3))}
    for phase, n_keys in ((1, N_SAME), (2, N_DIFF)):
        for i in range(n_keys):
            name = f"p{phase}k{i}"
            staged = zc.staging_root(store) / f"w{wid}" / name
            zc.write_staged_group(staged, arrays)
            digest = "same-digest" if phase == 1 else f"digest-w{wid}-{i}"
            barrier.wait(timeout=120)
            try:
                result = zc.commit_key(
                    store,
                    ("chunks", name),
                    staged,
                    input_digest=digest,
                    run_id=f"run-w{wid}",
                    env={},
                )
                outcome, record = type(result).__name__, result.record.to_dict()
            except zc.CommitConflictError as e:
                outcome, record = type(e).__name__, None
            out_q.put((phase, name, wid, outcome, record, str(staged)))


class TestCommitRace:
    """T-X2: exactly one process wins a contested key; the other sees a clean loss."""

    def test_two_process_commit_race(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        assert zc.create_store(
            store,
            identity_payload=IDENTITY,
            creator_run_id="creator",
            prefixes=PREFIXES,
            writer_id="parent",
        )
        ctx = mp.get_context("spawn")
        barrier = ctx.Barrier(2)
        out_q = ctx.Queue()
        procs = [
            ctx.Process(target=_race_worker, args=(wid, str(store), barrier, out_q))
            for wid in (0, 1)
        ]
        for p in procs:
            p.start()
        try:
            expected = 2 * (N_SAME + N_DIFF)
            results = [out_q.get(timeout=240) for _ in range(expected)]
        finally:
            for p in procs:
                p.join(timeout=60)
                if p.is_alive():
                    p.kill()
        assert [p.exitcode for p in procs] == [0, 0]

        by_key: dict[str, list[tuple[Any, ...]]] = {}
        for phase, name, wid, outcome, record, staged in results:
            by_key.setdefault(name, []).append((phase, wid, outcome, record, staged))
        assert len(by_key) == N_SAME + N_DIFF

        for name, rows in by_key.items():
            assert len(rows) == 2, name
            phase = rows[0][0]
            outcomes = sorted(r[2] for r in rows)
            if phase == 1:
                assert outcomes == ["Committed", "Duplicate"], (name, outcomes)
                committed = next(r for r in rows if r[2] == "Committed")
                duplicate = next(r for r in rows if r[2] == "Duplicate")
                # The duplicate reports the *winner's* record, not its own.
                assert committed[3] == duplicate[3], name
                assert committed[3]["run_id"] == f"run-w{committed[1]}"
                assert not Path(duplicate[4]).exists(), "loser's staged dir not cleaned up"
                final = zc.lookup(store, ("chunks", name), "same-digest", verify=True)
                assert isinstance(final, zc.Reuse)
                assert final.record.to_dict() == committed[3]
            else:
                assert outcomes == ["CommitConflictError", "Committed"], (name, outcomes)
                committed = next(r for r in rows if r[2] == "Committed")
                loser = next(r for r in rows if r[2] == "CommitConflictError")
                assert not Path(loser[4]).exists(), "loser's staged dir not cleaned up"
                i = name.removeprefix("p2k")
                win_digest = f"digest-w{committed[1]}-{i}"
                final = zc.lookup(store, ("chunks", name), win_digest, verify=True)
                assert isinstance(final, zc.Reuse)


# --------------------------------------------------------------------------------------
# Store-creation race
# --------------------------------------------------------------------------------------

N_CREATORS = 6
CREATE_PREFIXES = (("chunks",), ("meta",), ("a", "b"))


def _create_worker(wid: int, store_str: str, barrier: Any, out_q: Any) -> None:
    barrier.wait(timeout=120)
    try:
        created = zc.create_store(
            Path(store_str),
            identity_payload=IDENTITY,
            creator_run_id=f"creator-{wid}",
            prefixes=CREATE_PREFIXES,
            writer_id=f"w{wid}",
        )
        out_q.put((wid, created, None))
    except Exception as e:  # noqa: BLE001 - reported to the parent, which asserts none occur
        out_q.put((wid, None, f"{type(e).__name__}: {e}"))


_STAGE_THEN_COMMIT_CHILD = """
import sys
import time
from pathlib import Path

import numpy as np

from xtrax.run import zarr_commit as zc

store, release = Path(sys.argv[1]), Path(sys.argv[2])
staged = zc.staging_root(store) / "slow" / "k0"
zc.write_staged_group(staged, {"arr": np.arange(32)})
print("staged", flush=True)
deadline = time.monotonic() + 120
while not release.exists():
    if time.monotonic() > deadline:
        sys.exit(3)
    time.sleep(0.01)
result = zc.commit_key(
    store, ("chunks", "k0"), staged, input_digest="slow-digest", run_id="slow", env={}
)
print(type(result).__name__, flush=True)
"""


class TestCreateStoreRace:
    def test_six_way_create_race(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        ctx = mp.get_context("spawn")
        barrier = ctx.Barrier(N_CREATORS)
        out_q = ctx.Queue()
        procs = [
            ctx.Process(target=_create_worker, args=(wid, str(store), barrier, out_q))
            for wid in range(N_CREATORS)
        ]
        for p in procs:
            p.start()
        try:
            results = [out_q.get(timeout=240) for _ in range(N_CREATORS)]
        finally:
            for p in procs:
                p.join(timeout=60)
                if p.is_alive():
                    p.kill()
        errors = [r for r in results if r[2] is not None]
        assert not errors, errors
        outcomes = sorted(bool(r[1]) for r in results)
        assert outcomes == [False] * (N_CREATORS - 1) + [True], results

        record = zc.open_store(store, identity_payload=IDENTITY, prefixes=CREATE_PREFIXES)
        winner = next(r[0] for r in results if r[1])
        assert record["creator_run_id"] == f"creator-{winner}"
        root = zarr.open_group(str(store), mode="r")
        for prefix in CREATE_PREFIXES:
            assert zc.key_path(prefix) in root
        # No unrenamed root is left lying around.
        leftovers = list(zc.staging_root(store).rglob("__root__"))
        assert leftovers == []

    def test_create_store_loser_does_not_delete_inflight_staging(self, tmp_path: Path) -> None:
        """Regression: a losing create_store must not delete another writer's staged group."""
        store = tmp_path / "store"
        assert zc.create_store(
            store,
            identity_payload=IDENTITY,
            creator_run_id="creator",
            prefixes=PREFIXES,
            writer_id="first",
        )
        script = tmp_path / "slow_child.py"
        script.write_text(_STAGE_THEN_COMMIT_CHILD)
        release = tmp_path / "release"
        env = {k: v for k, v in os.environ.items() if k != zc.FAULT_ENV}
        child = subprocess.Popen(
            [sys.executable, str(script), str(store), str(release)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        try:
            assert child.stdout is not None
            assert child.stdout.readline().strip() == "staged"
            staged = zc.staging_root(store) / "slow" / "k0"
            assert (staged / "zarr.json").exists()

            # Another process attempts to create the (already existing) store and loses.
            ctx = mp.get_context("spawn")
            barrier = ctx.Barrier(1)
            out_q = ctx.Queue()
            p = ctx.Process(target=_create_worker, args=(99, str(store), barrier, out_q))
            p.start()
            wid, created, err = out_q.get(timeout=120)
            p.join(timeout=60)
            assert err is None, err
            assert created is False

            assert (staged / "zarr.json").exists(), "loser deleted another writer's staging"
        finally:
            release.write_text("go")
        out, err_text = child.communicate(timeout=120)
        assert child.returncode == 0, err_text
        assert out.strip() == "Committed"
        result = zc.lookup(store, ("chunks", "k0"), "slow-digest", verify=True)
        assert isinstance(result, zc.Reuse)
