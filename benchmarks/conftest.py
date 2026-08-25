import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from xtrax.profiling.bench import (
    build_bench_record_plan,
    check_probe_id_collision,
    record_from_plan,
)
from xtrax.training.trainer import Trainer
from xtrax.training.types import ResumableState


class _TinyMLP(eqx.Module):
    layers: list

    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        self.layers = [
            eqx.nn.Linear(64, 64, key=k1),
            eqx.nn.Linear(64, 1, key=k2),
        ]

    def __call__(self, x):
        # eqx.nn.Linear operates on rank-1 input — vmap over the batch axis.
        def _forward(xi):
            xi = jax.nn.tanh(self.layers[0](xi))
            return self.layers[1](xi)

        return jax.vmap(_forward)(x)


@pytest.fixture(scope="module")
def tiny_model():
    return _TinyMLP(jax.random.key(0))


@pytest.fixture(scope="module")
def synthetic_batch():
    k1, k2 = jax.random.split(jax.random.key(42))
    return {
        "inputs": jax.random.normal(k1, (32, 64)),
        "targets": jax.random.normal(k2, (32, 1)),
    }


@pytest.fixture(scope="module")
def trainer(tiny_model):
    def mse(pred, target):
        return jnp.mean((pred - target) ** 2)

    return Trainer(loss_fn=mse, optimizer=optax.adam(1e-3))


@pytest.fixture(scope="module")
def trainer_state(tiny_model, trainer):
    # Trainer has no init_state method — build ResumableState explicitly.
    opt_state = trainer.optimizer.init(eqx.filter(tiny_model, eqx.is_array))
    return ResumableState(
        step=jnp.int32(0),
        key=jax.random.key(0),
        model=tiny_model,
        opt_state=opt_state,
    )


def pytest_sessionfinish(session, exitstatus):
    """Optionally persist bench results as ProbeRecords (opt-in, fail-closed).

    Off by default: without XTRAX_BENCH_RECORD_DIR set this hook does
    nothing at all, so local bench runs never dirty the tree. When set,
    each benchmark that DECLARED its stage/n_atoms (see
    xtrax.profiling.bench) is written as one record into that directory;
    undeclared or empty-stats benches are reported skipped-with-reason in
    a terminal summary rather than silently dropped. Per-fixture failures
    are contained -- one bad declaration must not lose the other benches'
    records.
    """
    out_dir = os.environ.get("XTRAX_BENCH_RECORD_DIR", "").strip()
    if not out_dir:
        return

    bench_session = getattr(session.config, "_benchmarksession", None)
    fixtures = list(getattr(bench_session, "benchmarks", None) or [])

    written: list[str] = []
    skipped: list[tuple[str, str]] = []
    claimed: dict[str, str] = {}
    for fixture in fixtures:
        fullname = str(getattr(fixture, "fullname", getattr(fixture, "name", "?")))
        try:
            stats = getattr(fixture, "stats", None)
            if not stats:
                raise RuntimeError("no samples recorded (empty stats)")
            plan = build_bench_record_plan(
                fullname=fullname,
                params=getattr(fixture, "params", None),
                extra_info=dict(getattr(fixture, "extra_info", None) or {}),
                stats_dict=stats.as_dict(),
            )
            collides_with = check_probe_id_collision(plan.probe_id, claimed)
            if collides_with is not None:
                raise RuntimeError(
                    f"probe_id {plan.probe_id!r} collides with already-written "
                    f"{collides_with!r} (node ids differing only in '_' runs "
                    "normalize identically) -- refusing to silently overwrite"
                )
            claimed[plan.probe_id] = fullname
            record_from_plan(plan).write(Path(out_dir) / f"{plan.probe_id}.json")
            written.append(plan.probe_id)
        except Exception as exc:  # noqa: BLE001 -- reported per-bench below
            skipped.append((fullname, str(exc)))

    reporter = session.config.pluginmanager.getplugin("terminalreporter")
    if reporter is not None:
        reporter.section("xtrax bench records", sep="=")
        if not fixtures:
            reporter.line(
                f"XTRAX_BENCH_RECORD_DIR={out_dir} set, but no benchmarks "
                "ran (wrong path? --benchmark-disable?)"
            )
        for probe_id in written:
            reporter.line(f"written: {probe_id}.json")
        for fullname, reason in skipped:
            reporter.line(f"skipped: {fullname}: {reason}")
