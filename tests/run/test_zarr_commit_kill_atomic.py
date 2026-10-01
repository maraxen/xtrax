"""T-X1: SIGKILL a committing child at many points; the store never exposes a torn key."""

from pathlib import Path

from tests.run._zarr_commit_kill_harness import (
    _ATOMIC_CHILD,
    IDENTITY,
    N_DELAYS,
    N_KEYS,
    PREFIXES,
    _key_dirs,
    _kill_grid,
    _measure_duration,
)
from xtrax.run import zarr_commit as zc


class TestRandomKillAtomicity:
    """T-X1: SIGKILL at arbitrary times never leaves a torn or corrupt key."""

    def test_sigkill_never_tears_a_key(self, tmp_path: Path) -> None:
        script = tmp_path / "atomic_child.py"
        script.write_text(_ATOMIC_CHILD)
        duration = _measure_duration(tmp_path, script)
        stores = _kill_grid(tmp_path, script, duration)

        mid_run = 0
        for store in stores:
            zc.open_store(store, identity_payload=IDENTITY, prefixes=PREFIXES)
            reuse, missing = 0, 0
            for i in range(N_KEYS):
                key = ("chunks", f"k{i}")
                result = zc.lookup(store, key, f"digest{i}", verify=True)
                if (store / "chunks" / f"k{i}").exists():
                    assert isinstance(result, zc.Reuse), (store.name, key, result)
                    reuse += 1
                else:
                    assert isinstance(result, zc.Missing), (store.name, key, result)
                    missing += 1
            # Nothing but the committed keys is visible under the prefix.
            assert len(zc.committed_keys(store, ("chunks",))) == reuse
            assert len(_key_dirs(store)) == reuse
            if 0 < reuse < N_KEYS:
                mid_run += 1
        assert mid_run >= 1, (
            f"no kill landed mid-run (D={duration:.2f}s, delays over {N_DELAYS} runs): "
            "the kill grid never produced a partial store, so atomicity was not exercised"
        )
        print(f"T-X1: D={duration:.2f}s, mid-run kills={mid_run}/{N_DELAYS}")
