import ast
from pathlib import Path

import jax
import numpy as np
import pytest

from xtrax.data.pipeline import (
    DEFAULT_MP_WORKERS,
    DEFAULT_NUM_THREADS,
    build_input_pipeline,
    create_distributed_pipeline,
)


class TestCreateDistributedPipeline:
    """Tests for create_distributed_pipeline."""

    def test_create_distributed_pipeline_accepts_valid_divisible_batch(self):
        """create_distributed_pipeline accepts divisible global_batch_size."""
        dataset = [1, 2, 3, 4, 5, 6]
        pipeline = create_distributed_pipeline(
            dataset=dataset,
            global_batch_size=6,
            num_devices=2,
            seed=42,
        )
        # Should not raise; pipeline should be a dataset or iterable
        assert pipeline is not None

    def test_create_distributed_pipeline_raises_on_non_divisible_batch(self):
        """Raises ValueError if global_batch_size not divisible by num_devices."""
        dataset = [1, 2, 3, 4, 5]
        with pytest.raises(
            ValueError, match="global_batch_size=5 must be divisible by num_devices=2"
        ):
            create_distributed_pipeline(
                dataset=dataset,
                global_batch_size=5,
                num_devices=2,
                seed=42,
            )

    def test_create_distributed_pipeline_accepts_seed_param(self):
        """create_distributed_pipeline accepts and uses seed parameter."""
        dataset = [1, 2, 3, 4]
        pipeline = create_distributed_pipeline(
            dataset=dataset,
            global_batch_size=4,
            num_devices=2,
            seed=123,
        )
        assert pipeline is not None


def _flat(pipeline) -> list[int]:
    return [int(x) for batch in pipeline for x in np.asarray(batch).ravel()]


def _ints(n: int) -> list[np.int32]:
    return [np.int32(i) for i in range(n)]


class TestGrainPipeline:
    """Grain shuffle / pad / batch / shard / prefetch behaviour."""

    def test_in_memory_source_batches_with_np_stack(self):
        """A small in-memory source comes out as stacked batches, not the raw list."""
        source = _ints(6)
        pipeline = build_input_pipeline(
            source,
            batch_size=3,
            seed=0,
            num_epochs=1,
            shuffle=False,
        )
        batches = list(pipeline)
        assert len(batches) == 2
        assert isinstance(batches[0], np.ndarray)
        np.testing.assert_array_equal(batches[0], np.array([0, 1, 2], dtype=np.int32))
        np.testing.assert_array_equal(batches[1], np.array([3, 4, 5], dtype=np.int32))

    def test_shuffle_is_seeded(self):
        """The same seed repeats; a different seed does not."""
        kwargs = {"batch_size": 4, "num_epochs": 1, "shuffle": True}
        first = _flat(build_input_pipeline(_ints(16), seed=0, **kwargs))
        second = _flat(build_input_pipeline(_ints(16), seed=0, **kwargs))
        other = _flat(build_input_pipeline(_ints(16), seed=1, **kwargs))
        assert first == second
        assert first != other
        assert sorted(first) == list(range(16))

    def test_pad_fn_fixes_the_batch_shape(self):
        """Caller-supplied pad maps variable lengths onto one shape before batch."""

        def pad(example: np.ndarray) -> np.ndarray:
            out = np.zeros(4, dtype=np.int32)
            out[: example.shape[0]] = example
            return out

        source = [np.arange(i, dtype=np.int32) for i in range(1, 5)]
        batches = list(
            build_input_pipeline(
                source,
                batch_size=2,
                seed=0,
                num_epochs=1,
                shuffle=False,
                pad_fn=pad,
            )
        )
        assert [batch.shape for batch in batches] == [(2, 4), (2, 4)]

    def test_shard_options_split_the_source(self):
        """ShardOptions takes a consecutive even split before batching."""
        from grain.sharding import ShardOptions

        def shard(index: int, *, drop_remainder: bool, n: int = 8) -> list[int]:
            return _flat(
                build_input_pipeline(
                    _ints(n),
                    batch_size=1,
                    seed=0,
                    num_epochs=1,
                    shuffle=False,
                    drop_remainder=False,
                    shard_options=ShardOptions(
                        shard_index=index,
                        shard_count=2,
                        drop_remainder=drop_remainder,
                    ),
                )
            )

        assert shard(0, drop_remainder=True) == [0, 1, 2, 3]
        assert shard(1, drop_remainder=True) == [4, 5, 6, 7]
        assert shard(0, drop_remainder=False, n=7) == [0, 1, 2, 3]
        assert shard(1, drop_remainder=False, n=7) == [4, 5, 6]

    def test_create_distributed_pipeline_shards_by_process(self, monkeypatch):
        """Process index/count select the shard; the batch is per-device."""
        monkeypatch.setattr(jax, "process_index", lambda: 1)
        monkeypatch.setattr(jax, "process_count", lambda: 2)
        source = _ints(8)
        pipeline = create_distributed_pipeline(
            source,
            global_batch_size=4,
            num_devices=2,
            seed=0,
            shuffle=False,
        )
        assert pipeline is not source
        batches = list(pipeline)
        assert [batch.shape for batch in batches] == [(2,), (2,)]
        assert _flat(batches) == [4, 5, 6, 7]

    def test_repeat_none_yields_past_one_epoch(self):
        """num_epochs=None keeps yielding after the source is exhausted once."""
        pipeline = build_input_pipeline(
            _ints(2),
            batch_size=2,
            seed=0,
            num_epochs=None,
            shuffle=False,
        )
        iterator = iter(pipeline)
        try:
            first = next(iterator)
            second = next(iterator)
        finally:
            iterator.close()
        np.testing.assert_array_equal(first, np.array([0, 1], dtype=np.int32))
        np.testing.assert_array_equal(second, np.array([0, 1], dtype=np.int32))

    def test_device_put_moves_batches(self):
        """device_put is optional and places batches on the given device."""
        pipeline = build_input_pipeline(
            _ints(2),
            batch_size=2,
            seed=0,
            num_epochs=1,
            shuffle=False,
            device=jax.devices()[0],
        )
        iterator = iter(pipeline)
        try:
            batch = next(iterator)
        finally:
            iterator.close()
        assert isinstance(batch, jax.Array)
        np.testing.assert_array_equal(np.asarray(batch), np.array([0, 1], dtype=np.int32))

    def test_mp_prefetch_marks_unparsed_absl_flags(self):
        """First batch of mp_prefetch must not raise UnparsedFlagAccessError.

        ``FlagValues`` rejects attribute assignment, so the unparsed state is
        cleared on the private parsed bit. Building the pipeline must mark
        the flags parsed before the first batch is pulled.
        """
        from absl import flags

        flags.FLAGS.__dict__["__flags_parsed"] = False
        assert not flags.FLAGS.is_parsed()
        iterator = None
        try:
            pipeline = build_input_pipeline(
                _ints(4),
                batch_size=2,
                seed=0,
                num_epochs=1,
                shuffle=False,
                mp_workers=1,
                mp_buffer=1,
                prefetch_buffer_size=1,
            )
            assert flags.FLAGS.is_parsed()
            iterator = iter(pipeline)
            batch = next(iterator)
        finally:
            if iterator is not None:
                iterator.close()
            flags.FLAGS.mark_as_parsed()
        np.testing.assert_array_equal(np.asarray(batch), np.array([0, 1], dtype=np.int32))

    def test_defaults_are_one_thread_and_no_workers(self):
        """Cheap reads default to one read thread and no mp workers."""
        assert DEFAULT_NUM_THREADS == 1
        assert DEFAULT_MP_WORKERS == 0

    def test_missing_grain_raises_install_hint(self, monkeypatch):
        """The pipeline import fails only when a pipeline is built."""
        import builtins

        real_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name == "grain" or name.startswith("grain."):
                raise ImportError("grain missing")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded)
        with pytest.raises(ImportError, match=r"xtrax\[data\]"):
            build_input_pipeline(_ints(2), batch_size=2, seed=0)

    def test_data_package_does_not_import_grain_at_module_scope(self):
        """grain stays inside functions so importing xtrax.data is safe without it."""
        root = Path(__file__).resolve().parents[2] / "src" / "xtrax" / "data"
        for path in root.glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in tree.body:
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                assert not any(name == "grain" or name.startswith("grain.") for name in names), path
