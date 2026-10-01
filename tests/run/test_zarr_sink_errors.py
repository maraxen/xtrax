"""#5552: a failing drain names the staged key, array, shape and dtype -- also through
jax io_callback, which re-renders only the original exception's message line."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import pytest

from xtrax.run.sink import SinkSpec
from xtrax.run.zarr_sink import ZarrStagingSink

pytest.importorskip("zarr")

# zarr cannot resolve a numpy object dtype to a zarr data type.
_BAD = np.array([object(), object()], dtype=object)


def _sink(tmp_path: Path) -> ZarrStagingSink:
    # flush_every high enough that stage() never drains on its own.
    return ZarrStagingSink(
        SinkSpec(run_id="err-run", output_dir=tmp_path / "o.zarr", format="zarr", flush_every=99)
    )


def _raw_zarr_error_type(tmp_path: Path) -> type[BaseException]:
    """What THIS zarr raises for an object-dtype array (KeyError at the 3.0.8 floor,
    ValueError later) -- the drain must re-raise exactly that type, unwrapped."""
    import zarr

    group = zarr.open_group(str(tmp_path / "raw.zarr"), mode="w")
    try:
        group.create_array(name="x", shape=(2,), dtype=object)
    except Exception as e:  # noqa: BLE001 -- the type is the point
        return type(e)
    pytest.skip("this zarr accepts object dtype; no failing payload to drain")


def _assert_self_locating(text: str) -> None:
    assert "('batch', 7)" in text
    assert "'bad_payload'" in text
    assert "shape=(2,)" in text
    assert "dtype=object" in text


def test_direct_drain_error_keeps_type_and_names_the_payload(tmp_path: Path) -> None:
    sink = _sink(tmp_path)
    sink.stage(("batch", 7), bad_payload=_BAD)
    expected = _raw_zarr_error_type(tmp_path)
    with pytest.raises(expected) as info:  # zarr's own type, not a wrapper
        sink.drain()
    assert type(info.value) is expected
    _assert_self_locating(str(info.value))
    assert str(info.value).count("ZarrStagingSink.drain") == 1  # prefixed once, original kept


def test_drain_error_through_io_callback_is_self_locating(tmp_path: Path) -> None:
    sink = _sink(tmp_path)

    def cb(x):  # noqa: ANN001, ANN202
        sink.stage(("batch", 7), bad_payload=_BAD)
        sink.drain()
        return np.zeros((), np.int32)

    with pytest.raises(jax.errors.JaxRuntimeError) as info:
        jax.experimental.io_callback(cb, jax.ShapeDtypeStruct((), jnp.int32), jnp.ones(2))
    _assert_self_locating(str(info.value))


def test_oserror_through_io_callback_is_self_locating(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """OSError(errno, strerror) renders from strerror, not args[0]: disk-full is the
    likeliest real drain failure, and must not fall back to an invisible note."""
    import zarr

    def full_disk(*_a, **_k):  # noqa: ANN002, ANN003, ANN202
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(zarr.Group, "create_array", full_disk)
    sink = _sink(tmp_path)

    def cb(x):  # noqa: ANN001, ANN202
        sink.stage(("batch", 7), bad_payload=np.zeros((2,), np.float32))
        sink.drain()
        return np.zeros((), np.int32)

    with pytest.raises(jax.errors.JaxRuntimeError) as info:
        jax.experimental.io_callback(cb, jax.ShapeDtypeStruct((), jnp.int32), jnp.ones(2))
    text = str(info.value)
    assert "('batch', 7)" in text
    assert "'bad_payload'" in text
    assert "No space left on device" in text


def test_prefix_falls_back_to_a_note_when_str_ignores_args() -> None:
    from xtrax.run.zarr_sink import _prefix_message

    class Opaque(Exception):
        def __str__(self) -> str:
            return "fixed text"

    e = Opaque("anything")
    _prefix_message(e, "CTX")
    assert "CTX" in e.__notes__


def test_failed_drain_leaves_the_buffer_for_a_retry(tmp_path: Path) -> None:
    """The message says the buffer was not cleared; hold it to that."""
    sink = _sink(tmp_path)
    sink.stage(("batch", 7), bad_payload=_BAD)
    with pytest.raises(_raw_zarr_error_type(tmp_path), match="NOT cleared"):
        sink.drain()
    assert ("batch", 7) in sink._pending
