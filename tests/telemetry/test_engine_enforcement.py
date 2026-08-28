"""Engine-level enforcement: every fit and eval leaves a ledger row.

Engine is the chokepoint that matters. A CLI-level change instruments only runs
launched through the CLI; instrumenting Engine also covers direct library use
(``Trainer``/``Engine`` constructed in someone else's code), which previously
had no funnel at all.

Fixtures mirror tests/engine/test_engine.py rather than inventing new ones, so
these tests exercise the same shapes the engine suite does.
"""

from collections.abc import Iterator
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from xtrax.engine.engine import Engine
from xtrax.telemetry.ledger import LedgerUnavailableError, iter_rows
from xtrax.telemetry.record import (
    KIND_EVAL,
    KIND_TRAIN,
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_OPTED_OUT,
)
from xtrax.training.trainer import Trainer
from xtrax.training.types import ResumableState


class DummyLoss:
    def __call__(self, predictions: Any, targets: Any) -> jax.Array:
        return jnp.mean((predictions - targets) ** 2)


class DummyModel(eqx.Module):
    weight: jax.Array = eqx.field()

    def __init__(self, key):
        self.weight = jax.random.normal(key, (2,))

    def __call__(self, x):
        return x @ self.weight


class DummyDataModule:
    def __init__(self, batch_count: int = 2, batch_size: int = 2):
        self.batch_count = batch_count
        self.batch_size = batch_size

    def _batches(self, seed: int) -> Iterator[dict[str, Any]]:
        for i in range(self.batch_count):
            yield {
                "inputs": jax.random.normal(jax.random.PRNGKey(i + seed), (self.batch_size, 2)),
                "targets": jax.random.normal(
                    jax.random.PRNGKey(i + seed + 100), (self.batch_size,)
                ),
            }

    def train_iter(self) -> Iterator[dict[str, Any]]:
        return self._batches(0)

    def eval_iter(self) -> Iterator[dict[str, Any]]:
        return self._batches(200)


def _engine(**kwargs) -> Engine:
    return Engine(
        trainer=Trainer(loss_fn=DummyLoss(), optimizer=optax.adam(1e-3)),
        callbacks=kwargs.pop("callbacks", ()),
        validation_callbacks=kwargs.pop("validation_callbacks", ()),
    )


def _state() -> ResumableState:
    key = jax.random.PRNGKey(0)
    model = DummyModel(key)
    return ResumableState(
        step=jnp.array(0, dtype=jnp.int32),
        key=key,
        model=model,
        opt_state=optax.adam(1e-3).init(eqx.filter(model, eqx.is_array)),
    )


@pytest.fixture(autouse=True)
def _isolated_ledger(tmp_path, monkeypatch):
    """Point the ledger at a temp dir so tests never touch the repo's own."""
    monkeypatch.setenv("XTRAX_LEDGER_ROOT", str(tmp_path / "ledger"))
    monkeypatch.delenv("XTRAX_TELEMETRY_OPTOUT", raising=False)
    return tmp_path / "ledger"


# --- training ---------------------------------------------------------------


def test_fit_writes_a_ledger_row(_isolated_ledger):
    _engine().fit_sync(_state(), DummyDataModule(), num_epochs=1)
    rows = list(iter_rows(_isolated_ledger))
    assert len(rows) == 1
    assert rows[0].kind == KIND_TRAIN
    assert rows[0].telemetry_status == STATUS_COMPLETE
    assert rows[0].is_citable


def test_fit_captures_the_executed_ir(_isolated_ledger):
    """The point of the whole exercise: what actually ran is recoverable."""
    _engine().fit_sync(_state(), DummyDataModule(), num_epochs=1)
    row = next(iter(iter_rows(_isolated_ledger)))
    kinds = {ref.kind for ref in row.ir}
    assert "jaxpr" in kinds
    assert "stablehlo" in kinds
    assert all(ref.bytes > 0 for ref in row.ir), [r.reason for r in row.ir]


def test_fit_records_provenance(_isolated_ledger):
    _engine().fit_sync(_state(), DummyDataModule(), num_epochs=1)
    prov = next(iter(iter_rows(_isolated_ledger))).provenance
    assert prov.git_sha != "unknown"
    assert prov.jax_version
    assert prov.hostname


def test_ir_is_captured_once_not_once_per_step(_isolated_ledger):
    """Many steps and epochs must still yield one IR ref per artifact kind."""
    _engine().fit_sync(_state(), DummyDataModule(batch_count=5), num_epochs=3)
    row = next(iter(iter_rows(_isolated_ledger)))
    kinds = [ref.kind for ref in row.ir]
    assert len(kinds) == len(set(kinds)), f"IR captured more than once: {kinds}"


def test_an_engine_with_no_callbacks_is_still_instrumented(_isolated_ledger):
    """callbacks=() was hard-coded at both CLI call sites; it must not opt out."""
    engine = _engine(callbacks=())
    assert engine.callbacks == ()
    engine.fit_sync(_state(), DummyDataModule(), num_epochs=1)
    assert len(list(iter_rows(_isolated_ledger))) == 1


def test_user_callbacks_still_fire_alongside_telemetry(_isolated_ledger):
    """Telemetry is appended, never a replacement for the caller's callbacks."""
    seen: list[str] = []

    class Tracking:
        def on_train_start(self, state):
            seen.append("train_start")

        def on_train_end(self, state):
            seen.append("train_end")

        def on_resume(self, state):
            seen.append("resume")

        def on_epoch_start(self, state, epoch):
            seen.append("epoch_start")

        def on_epoch_end(self, state, epoch):
            seen.append("epoch_end")

        def on_step_start(self, state):
            seen.append("step_start")

        def on_step_end(self, state, metrics):
            seen.append("step_end")

    _engine(callbacks=(Tracking(),)).fit_sync(_state(), DummyDataModule(), num_epochs=1)
    assert "train_start" in seen
    assert "epoch_end" in seen
    assert len(list(iter_rows(_isolated_ledger))) == 1


# --- inference --------------------------------------------------------------


def test_eval_writes_a_ledger_row(_isolated_ledger):
    """Inference runs are runs. Before this, eval persisted nothing at all."""
    import asyncio

    asyncio.run(_engine().eval(_state(), DummyDataModule()))
    rows = list(iter_rows(_isolated_ledger))
    assert len(rows) == 1
    assert rows[0].kind == KIND_EVAL
    assert rows[0].is_citable


def test_eval_captures_ir(_isolated_ledger):
    import asyncio

    asyncio.run(_engine().eval(_state(), DummyDataModule()))
    row = next(iter(iter_rows(_isolated_ledger)))
    assert {ref.kind for ref in row.ir} >= {"jaxpr", "stablehlo"}


# --- fail-closed ------------------------------------------------------------


def test_fit_refuses_to_start_when_the_ledger_is_unwritable(tmp_path, monkeypatch):
    """The enforcement claim, tested directly."""
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0o500)
    monkeypatch.setenv("XTRAX_LEDGER_ROOT", str(blocked / "ledger"))
    try:
        with pytest.raises(LedgerUnavailableError):
            _engine().fit_sync(_state(), DummyDataModule(), num_epochs=1)
    finally:
        blocked.chmod(0o700)


def test_eval_refuses_to_start_when_the_ledger_is_unwritable(tmp_path, monkeypatch):
    import asyncio

    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0o500)
    monkeypatch.setenv("XTRAX_LEDGER_ROOT", str(blocked / "ledger"))
    try:
        with pytest.raises(LedgerUnavailableError):
            asyncio.run(_engine().eval(_state(), DummyDataModule()))
    finally:
        blocked.chmod(0o700)


# --- failure and opt-out ----------------------------------------------------


def test_a_crashed_run_is_recorded_and_marked_non_citable(_isolated_ledger):
    class Exploding(DummyDataModule):
        def train_iter(self):
            raise RuntimeError("data pipeline died")

    with pytest.raises(RuntimeError, match="data pipeline died"):
        _engine().fit_sync(_state(), Exploding(), num_epochs=1)

    rows = list(iter_rows(_isolated_ledger))
    assert len(rows) == 1
    assert rows[0].telemetry_status == STATUS_FAILED
    assert "data pipeline died" in rows[0].status_reason
    assert not rows[0].is_citable


def test_optout_still_records_a_row_and_makes_it_non_citable(_isolated_ledger, monkeypatch):
    monkeypatch.setenv("XTRAX_TELEMETRY_OPTOUT", "1")
    _engine().fit_sync(_state(), DummyDataModule(), num_epochs=1)
    rows = list(iter_rows(_isolated_ledger))
    assert len(rows) == 1
    assert rows[0].telemetry_status == STATUS_OPTED_OUT
    assert not rows[0].is_citable


def test_optout_skips_ir_capture(_isolated_ledger, monkeypatch):
    """Opting out must actually save the work, not just relabel the row."""
    monkeypatch.setenv("XTRAX_TELEMETRY_OPTOUT", "1")
    _engine().fit_sync(_state(), DummyDataModule(), num_epochs=1)
    assert next(iter(iter_rows(_isolated_ledger))).ir == ()


# --- caller-supplied ledgers ------------------------------------------------


def test_a_caller_supplied_ledger_is_not_closed_by_the_engine(_isolated_ledger):
    """A ledger spanning a sweep must survive each individual fit."""
    from xtrax.telemetry.ledger import RunLedger

    ledger = RunLedger.open("run-sweep-parent", root=_isolated_ledger)
    _engine().fit_sync(_state(), DummyDataModule(), num_epochs=1, ledger=ledger)
    assert not ledger.closed
    assert list(iter_rows(_isolated_ledger)) == []  # nothing written yet
    ledger.close()
    assert len(list(iter_rows(_isolated_ledger))) == 1


def test_an_explicit_run_id_is_used(_isolated_ledger):
    _engine().fit_sync(_state(), DummyDataModule(), num_epochs=1, run_id="run-explicit-0001")
    assert next(iter(iter_rows(_isolated_ledger))).run_id == "run-explicit-0001"


def test_each_run_gets_a_distinct_id_by_default(_isolated_ledger):
    engine = _engine()
    engine.fit_sync(_state(), DummyDataModule(), num_epochs=1)
    engine.fit_sync(_state(), DummyDataModule(), num_epochs=1)
    ids = [r.run_id for r in iter_rows(_isolated_ledger)]
    assert len(ids) == 2
    assert len(set(ids)) == 2


def test_repeated_runs_share_ir_blobs(_isolated_ledger):
    """Dedup across runs: identical code must not re-store its IR."""
    from xtrax.telemetry.ledger import blobs_dir
    from xtrax.telemetry.store import BlobStore

    engine = _engine()
    engine.fit_sync(_state(), DummyDataModule(), num_epochs=1)
    after_first = BlobStore(blobs_dir(_isolated_ledger)).count()
    engine.fit_sync(_state(), DummyDataModule(), num_epochs=1)
    assert BlobStore(blobs_dir(_isolated_ledger)).count() == after_first
