"""Regressions from code review: close containment and segment naming.

Both are failure-path bugs -- the kind that only bite when something else has
already gone wrong, which is exactly when a telemetry subsystem must not make
things worse.
"""

import pytest

from xtrax.telemetry.ledger import (
    LedgerUnavailableError,
    RunLedger,
    _active_segment,
    iter_rows,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("XTRAX_TELEMETRY_OPTOUT", raising=False)
    monkeypatch.delenv("XTRAX_LEDGER_ROOT", raising=False)


# --- close containment ------------------------------------------------------


def test_a_write_failure_does_not_mask_the_real_exception(tmp_path, monkeypatch):
    """The bug: a full disk while writing the row replaced the training error.

    The user then sees an OSError about the ledger and has no idea their run
    actually died of something else -- the telemetry hiding the failure it
    exists to record.
    """
    root = tmp_path / "ledger"
    ledger = RunLedger.open("run-1", root=root)

    def boom(*_args, **_kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr("xtrax.telemetry.ledger._append_line", boom)

    with pytest.warns(UserWarning, match="failed to write the ledger row"):
        with pytest.raises(ValueError, match="the real failure"):
            with ledger:
                raise ValueError("the real failure")


def test_a_write_failure_with_no_exception_in_flight_is_raised(tmp_path, monkeypatch):
    """With nothing else wrong, the write failure IS the failure -- raise it."""
    root = tmp_path / "ledger"
    ledger = RunLedger.open("run-1", root=root)

    def boom(*_args, **_kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr("xtrax.telemetry.ledger._append_line", boom)

    with pytest.raises(LedgerUnavailableError, match="could not write the ledger row"):
        with ledger:
            pass


def test_close_if_open_is_safe_to_call_twice(tmp_path):
    root = tmp_path / "ledger"
    ledger = RunLedger.open("run-1", root=root)
    ledger.close_if_open()
    ledger.close_if_open()
    assert len(list(iter_rows(root))) == 1


def test_close_itself_still_refuses_a_second_write(tmp_path):
    """Exactly one row per run is what dedup and last-wins rest on."""
    root = tmp_path / "ledger"
    ledger = RunLedger.open("run-1", root=root)
    ledger.close()
    with pytest.raises(Exception, match="already closed"):
        ledger.close()


def test_engine_write_failure_does_not_mask_a_training_error(tmp_path, monkeypatch):
    """The same containment, through Engine's hand-rolled try/finally."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax

    from xtrax.engine.engine import Engine
    from xtrax.training.trainer import Trainer
    from xtrax.training.types import ResumableState

    class Model(eqx.Module):
        w: jax.Array

        def __init__(self, key):
            self.w = jax.random.normal(key, (2,))

        def __call__(self, x):
            return x @ self.w

    class Broken:
        def train_iter(self):
            raise RuntimeError("dataset exploded")

        def eval_iter(self):
            raise RuntimeError("dataset exploded")

    monkeypatch.setenv("XTRAX_LEDGER_ROOT", str(tmp_path / "ledger"))
    key = jax.random.PRNGKey(0)
    model = Model(key)
    state = ResumableState(
        step=jnp.array(0, dtype=jnp.int32),
        key=key,
        model=model,
        opt_state=optax.adam(1e-3).init(eqx.filter(model, eqx.is_array)),
    )
    engine = Engine(
        trainer=Trainer(lambda p, t: jnp.mean((p - t) ** 2), optax.adam(1e-3)),
        callbacks=(),
    )

    def boom(*_args, **_kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr("xtrax.telemetry.ledger._append_line", boom)

    with pytest.warns(UserWarning, match="failed to write the ledger row"):
        # The dataset error must survive, not be replaced by the OSError.
        with pytest.raises(RuntimeError, match="dataset exploded"):
            engine.fit_sync(state, Broken(), num_epochs=1)


# --- segment naming ---------------------------------------------------------


def test_a_stray_segment_file_does_not_abort_the_run(tmp_path):
    """The bug: int(stem) on a non-numeric name raised, and under fail-closed
    semantics an uncaught ValueError here aborts the run entirely."""
    root = tmp_path / "ledger"
    with RunLedger.open("run-1", root=root):
        pass
    # A manual copy, an editor backup -- anything that sorts after the real one.
    (root / "segments" / "00001.bak.jsonl").write_text("", encoding="utf-8")

    with RunLedger.open("run-2", root=root):
        pass
    assert {r.run_id for r in iter_rows(root)} == {"run-1", "run-2"}


def test_segment_selection_ignores_non_numeric_names(tmp_path):
    root = tmp_path / "ledger"
    (root / "segments").mkdir(parents=True)
    (root / "segments" / "notes.jsonl").write_text("", encoding="utf-8")
    assert _active_segment(root).name == "00001.jsonl"


def test_segments_are_ordered_numerically_not_lexicographically(tmp_path):
    """'00010' sorts before '00009' lexicographically only by luck of padding;
    relying on that would misfile rows once the count crosses a digit width."""
    root = tmp_path / "ledger"
    seg = root / "segments"
    seg.mkdir(parents=True)
    for name in ("00001.jsonl", "00009.jsonl", "00010.jsonl"):
        (seg / name).write_text("", encoding="utf-8")
    assert _active_segment(root).name == "00010.jsonl"


def test_a_stray_file_is_still_read_as_history(tmp_path):
    """Not a write target, but its rows are data and must not be dropped."""
    root = tmp_path / "ledger"
    with RunLedger.open("run-1", root=root) as ledger:
        row = ledger.build_record()
    (root / "segments" / "archived.bak.jsonl").write_text(
        row.to_json_line().replace("run-1", "run-copied"), encoding="utf-8"
    )
    assert "run-copied" in {r.run_id for r in iter_rows(root)}
