"""T-X3: N concurrent processes create-or-join one durable store through ZarrStagingSink.

Four long-lived spawn workers meet at a Barrier and construct a
``create_or_join`` sink on a FRESH store path, 50 times over. Per iteration the parent
checks the creation race had exactly one winner, nobody raised, every shard and writer
record committed, the root ``zarr.json`` was never rewritten, and every writer record
precedes that writer's shards.
"""

import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Any

from xtrax.run import zarr_commit as zc

IDENTITY = {"kind": "join-concurrency", "version": 1}
PREFIXES = (("shards",),)
N_WORKERS = 4
N_ITERS = 50
KEYS_PER_WORKER = 2


def _store_path(base: str, it: int) -> Path:
    return Path(base) / f"it{it:03d}" / "store"


def _worker(wid: int, base: str, n_iters: int, barrier: Any, out_q: Any) -> None:
    import numpy as np

    from xtrax.run import zarr_commit
    from xtrax.run.sink import SinkSpec
    from xtrax.run.zarr_sink import ZarrStagingSink

    # Spy on create_store so the parent can prove the creation race really happened.
    create_results: list[bool] = []
    real_create = zarr_commit.create_store

    def spy(*args: Any, **kwargs: Any) -> bool:  # noqa: ANN401
        created = real_create(*args, **kwargs)
        create_results.append(created)
        return created

    zarr_commit.create_store = spy  # type: ignore[assignment]

    for it in range(n_iters):
        store = _store_path(base, it)
        run_id = f"run-w{wid}-i{it}"
        result: dict[str, Any] = {"it": it, "wid": wid, "run_id": run_id, "error": None}
        before = len(create_results)
        try:
            barrier.wait(timeout=180)
            sink = ZarrStagingSink(
                SinkSpec(
                    run_id=run_id,
                    output_dir=store,
                    format="zarr",
                    flush_every=1000,
                    open_mode="create_or_join",
                    store_identity=IDENTITY,
                    prefixes=PREFIXES,
                )
            )
            result["creator_run_id"] = sink.store_record["creator_run_id"]
            result["root_after_open"] = (store / "zarr.json").read_bytes()
            for j in range(KEYS_PER_WORKER):
                sink.stage(
                    ("shards", f"w{wid}-k{j}"),
                    input_digest=f"digest-{wid}-{j}",
                    x=np.full(8, wid * 10 + j),
                )
            outcomes = sink.drain()
            result["outcomes"] = {"/".join(k): type(v).__name__ for k, v in outcomes.items()}
            result["root_after_drain"] = (store / "zarr.json").read_bytes()
            sink.close()
        except BaseException as e:  # noqa: BLE001 - reported to the parent, which asserts none occur
            result["error"] = f"{type(e).__name__}: {e}"
        result["create_results"] = create_results[before:]
        out_q.put(result)


def test_concurrent_create_or_join_fifty_fresh_stores(tmp_path: Path) -> None:
    base = str(tmp_path)
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(N_WORKERS)
    out_q = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(wid, base, N_ITERS, barrier, out_q))
        for wid in range(N_WORKERS)
    ]
    for p in procs:
        p.start()
    try:
        results = [out_q.get(timeout=300) for _ in range(N_WORKERS * N_ITERS)]
    finally:
        for p in procs:
            p.join(timeout=60)
            if p.is_alive():
                p.kill()

    errors = [r for r in results if r["error"] is not None]
    assert not errors, errors[:3]

    by_iter: dict[int, list[dict[str, Any]]] = {}
    for r in results:
        by_iter.setdefault(r["it"], []).append(r)
    assert sorted(by_iter) == list(range(N_ITERS))

    lost_races = 0
    for it, rs in sorted(by_iter.items()):
        store = _store_path(base, it)
        assert len(rs) == N_WORKERS, (it, len(rs))
        run_ids = {r["run_id"] for r in rs}

        # Exactly one creator, and every worker agrees on who it was.
        creators = [r for r in rs if r["creator_run_id"] == r["run_id"]]
        assert len(creators) == 1, (it, [r["creator_run_id"] for r in rs])
        assert {r["creator_run_id"] for r in rs} == {creators[0]["run_id"]}
        record = zc.open_store(store, identity_payload=IDENTITY, prefixes=PREFIXES)
        assert record["creator_run_id"] == creators[0]["run_id"], it
        created_flags = [flag for r in rs for flag in r["create_results"]]
        assert created_flags.count(True) == 1, (it, created_flags)
        lost_races += created_flags.count(False)

        # All shards and all writer records committed; each worker's outcomes were Committed.
        assert all(set(r["outcomes"].values()) == {"Committed"} for r in rs), (it, rs)
        shard_keys = zc.committed_keys(store, ("shards",))
        assert len(shard_keys) == N_WORKERS * KEYS_PER_WORKER, (it, shard_keys)
        writer_keys = zc.committed_keys(store, ("_xtrax_writers",))
        assert {k[1] for k in writer_keys} == run_ids, (it, writer_keys)

        # The root was never rewritten by anyone, at any point.
        final_root = (store / "zarr.json").read_bytes()
        for r in rs:
            assert r["root_after_open"] == final_root, (it, r["run_id"], "open")
            assert r["root_after_drain"] == final_root, (it, r["run_id"], "drain")

        # Writer record precedes that writer's shards; shards verify.
        for r in rs:
            writer = zc.read_record(store / "_xtrax_writers" / r["run_id"])
            assert writer is not None, (it, r["run_id"])
            for j in range(KEYS_PER_WORKER):
                key = ("shards", f"w{r['wid']}-k{j}")
                shard = zc.read_record(store / zc.key_path(key))
                assert shard is not None and shard.run_id == r["run_id"], (it, key)
                assert datetime.fromisoformat(writer.committed_at) <= datetime.fromisoformat(
                    shard.committed_at
                ), (it, key)
                result = zc.lookup(store, key, f"digest-{r['wid']}-{j}", verify=True)
                assert isinstance(result, zc.Reuse), (it, key, result)

        # Every worker closed: no staging directory survives.
        staging = zc.staging_root(store)
        assert not staging.exists() or list(staging.iterdir()) == [], (it, list(staging.iterdir()))

    # The creation race must actually have been contested, or the test proves nothing.
    assert lost_races >= 1, "no worker ever lost the create_store race in 50 iterations"
    print(f"T-X3: {N_ITERS} iterations, {lost_races} lost creation races")
