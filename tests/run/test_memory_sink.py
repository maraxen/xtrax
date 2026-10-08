"""MemorySink contract: spec checks, lifecycle, attr rules, append rules, receipt."""

import warnings

import numpy as np
import pytest

from xtrax.run import MemorySink
from xtrax.run.sink import SinkSpec

_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"level": {"type": "string"}, "depth": {"type": "integer"}},
    "required": ["level"],
}


def _sink(**kwargs: object) -> MemorySink:
    spec_kwargs: dict[str, object] = {"run_id": "mem", "format": "memory", "flush_every": 100}
    spec_kwargs.update(kwargs)
    return MemorySink(SinkSpec(**spec_kwargs))  # type: ignore[arg-type]


class TestSpecChecks:
    def test_non_memory_format_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="format == 'memory'"):
            MemorySink(SinkSpec(run_id="r", format="zarr"))

    @pytest.mark.parametrize("run_id", ["", "   "])
    def test_blank_run_id_is_rejected(self, run_id: str) -> None:
        with pytest.raises(ValueError, match="non-blank"):
            MemorySink(SinkSpec(run_id=run_id, format="memory"))

    def test_seed_is_recorded_in_provenance_only_when_set(self) -> None:
        assert _sink(seed=7).provenance["seed"] == 7
        assert "seed" not in _sink().provenance


class TestLifecycle:
    def test_stage_and_drain_after_close_raise(self) -> None:
        sink = _sink()
        sink.close()
        with pytest.raises(RuntimeError, match=r"stage\(\) after close\(\)"):
            sink.stage((0,), x=np.ones(2))
        with pytest.raises(RuntimeError, match=r"drain\(\) after close\(\)"):
            sink.drain()
        with pytest.raises(RuntimeError, match=r"stamp_reserved\(\) after close\(\)"):
            sink.stamp_reserved((0,), "note", {"a": 1})

    def test_stage_after_finalize_raises(self) -> None:
        sink = _sink()
        sink.finalize()
        with pytest.raises(RuntimeError, match=r"stage\(\) after finalize\(\)"):
            sink.stage((0,), x=np.ones(2))

    def test_finalize_runs_only_once(self) -> None:
        sink = _sink()
        sink.finalize()
        with pytest.raises(RuntimeError, match="only once"):
            sink.finalize()

    def test_finalize_refuses_undrained_keys(self) -> None:
        sink = _sink()
        sink.stage((0,), x=np.ones(2))
        with pytest.raises(RuntimeError, match="1 staged key"):
            sink.finalize()
        sink.drain()
        assert sink.finalize().run_id == "mem"

    def test_close_warns_and_discards_undrained_keys(self) -> None:
        sink = _sink()
        sink.stage((0,), x=np.ones(2))
        assert len(sink) == 1
        with pytest.warns(UserWarning, match="discarding 1 staged key"):
            sink.close()
        assert len(sink) == 0
        with pytest.raises(KeyError, match="no committed entry"):
            sink.read((0,))

    def test_close_is_idempotent_and_silent_when_empty(self) -> None:
        sink = _sink()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sink.close()
            sink.close()

    def test_context_manager_closes(self) -> None:
        with _sink() as sink:
            assert isinstance(sink, MemorySink)
        with pytest.raises(RuntimeError, match="after close"):
            sink.drain()

    def test_flush_every_drains_automatically(self) -> None:
        sink = _sink(flush_every=2)
        sink.stage((0,), x=np.ones(1))
        assert len(sink) == 1
        sink.stage((1,), x=np.ones(1))
        assert len(sink) == 0
        np.testing.assert_array_equal(sink.read((1,))["x"], np.ones(1))


class TestStageArguments:
    @pytest.mark.parametrize("arg", ["input_digest", "input_payload", "commit_meta", "env_extra"])
    def test_durable_only_arguments_are_rejected(self, arg: str) -> None:
        sink = _sink()
        with pytest.raises(ValueError, match="durable-only"):
            sink.stage((0,), x=np.ones(1), **{arg: "v"})
        assert len(sink) == 0

    def test_take_pops_payload_and_attrs(self) -> None:
        sink = _sink()
        sink.stage((0,), {"note": "a"}, x=np.arange(3))
        taken = sink.take((0,))
        np.testing.assert_array_equal(taken["x"], np.arange(3))
        sink.drain()
        with pytest.raises(KeyError, match="no committed entry"):
            sink.read_attrs((0,))

    def test_take_of_missing_key_raises(self) -> None:
        with pytest.raises(KeyError, match="no pending entry"):
            _sink().take((9,))


class TestAttrRules:
    def test_core_provenance_names_are_reserved(self) -> None:
        sink = _sink()
        with pytest.raises(ValueError, match="reserved core provenance"):
            sink.stage((0,), {"run_id": "x"}, v=np.ones(1))
        assert len(sink) == 0

    def test_reserved_namespace_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="reserved namespace"):
            _sink().stage((0,), {"xtrax.mine": 1}, v=np.ones(1))

    def test_schema_type_violation_is_rejected_at_stage(self) -> None:
        sink = _sink(extension_schema=_SCHEMA)
        with pytest.raises(ValueError, match="violate the SinkSpec"):
            sink.stage((0,), {"level": 3}, v=np.ones(1))
        assert len(sink) == 0

    def test_missing_required_attr_refuses_drain_and_keeps_buffer(self) -> None:
        sink = _sink(extension_schema=_SCHEMA)
        sink.stage((0,), {"depth": 1}, v=np.ones(1))
        with pytest.raises(ValueError, match="incomplete"):
            sink.drain()
        assert len(sink) == 1
        sink.stage((0,), {"level": "a"})
        sink.drain()
        attrs = sink.read_attrs((0,))
        assert attrs["level"] == "a"
        assert attrs["depth"] == 1
        assert attrs["run_id"] == "mem"

    def test_read_attrs_of_missing_key_raises(self) -> None:
        with pytest.raises(KeyError, match="no committed entry"):
            _sink().read_attrs((0,))


class TestStampReserved:
    @pytest.mark.parametrize(
        ("name", "match"),
        [
            ("", "non-empty"),
            ("a/b", "may not contain"),
            ("a.b", "may not contain"),
            ("commit", "reserved by the sink"),
            ("store", "reserved by the sink"),
        ],
    )
    def test_bad_names_are_rejected(self, name: str, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            _sink().stamp_reserved((0,), name, {"a": 1})

    def test_non_json_payload_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not JSON-safe"):
            _sink().stamp_reserved((0,), "note", {"a": object()})

    def test_stamp_does_not_change_the_receipt(self) -> None:
        plain = _sink()
        plain.stage((0,), v=np.arange(2))
        plain.drain()
        stamped = _sink()
        stamped.stage((0,), v=np.arange(2))
        stamped.drain()
        stamped.stamp_reserved((0,), "note", {"a": 1})
        assert stamped.finalize().digest == plain.finalize().digest


class TestAppendRules:
    def test_replace_mode_overwrites(self) -> None:
        sink = _sink()
        sink.stage((0,), v=np.array([1, 2]))
        sink.drain()
        sink.stage((0,), v=np.array([3]))
        sink.drain()
        np.testing.assert_array_equal(sink.read((0,))["v"], np.array([3]))

    @pytest.mark.parametrize(
        ("first", "second"),
        [(np.int32(1), np.array([2], dtype=np.int32)), (np.array([1]), np.int64(2))],
        ids=["scalar-stored", "scalar-incoming"],
    )
    def test_scalar_append_is_rejected_and_store_kept(self, first, second) -> None:
        sink = _sink(append=True)
        sink.stage((0,), v=first)
        sink.drain()
        sink.stage((0,), v=second)
        with pytest.raises(ValueError, match="scalar"):
            sink.drain()
        np.testing.assert_array_equal(sink.read((0,))["v"], np.asarray(first))

    def test_rank_mismatch_is_rejected(self) -> None:
        sink = _sink(append=True)
        sink.stage((0,), v=np.zeros((1, 2)))
        sink.drain()
        sink.stage((0,), v=np.zeros((1, 2, 1)))
        with pytest.raises(ValueError, match="trailing shape"):
            sink.drain()
