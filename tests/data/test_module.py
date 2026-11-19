import importlib.util
import itertools

import numpy as np
import pytest

from xtrax.data.module import DataModule
from xtrax.data.pipeline import build_input_pipeline

requires_grain = pytest.mark.skipif(
    importlib.util.find_spec("grain") is None,
    reason="needs the optional grain dependency (xtrax[data])",
)


class TestDataModule:
    """Tests for DataModule iterator behavior and distributed guards."""

    def test_train_iter_yields_from_dataset(self):
        """train_iter yields items from the dataset."""
        dataset = [1, 2, 3]
        module = DataModule(
            dataset=dataset,
            batch_size=2,
            num_epochs=1,
            seed=42,
            distributed=False,
        )
        result = list(module.train_iter())
        assert result == [1, 2, 3]

    def test_eval_iter_yields_from_dataset(self):
        """eval_iter yields items from the dataset."""
        dataset = [4, 5, 6]
        module = DataModule(
            dataset=dataset,
            batch_size=2,
            num_epochs=1,
            seed=42,
            distributed=False,
        )
        result = list(module.eval_iter())
        assert result == [4, 5, 6]

    def test_train_iter_distributed_single_process_raises(self):
        """train_iter raises RuntimeError when distributed=True in a one-process runtime."""
        dataset = [1, 2, 3]
        module = DataModule(
            dataset=dataset,
            batch_size=2,
            num_epochs=1,
            seed=42,
            distributed=True,
        )
        with pytest.raises(RuntimeError, match="requires a multi-process JAX runtime"):
            list(module.train_iter())

    def test_eval_iter_distributed_single_process_raises(self):
        """eval_iter raises RuntimeError when distributed=True in a one-process runtime."""
        dataset = [1, 2, 3]
        module = DataModule(
            dataset=dataset,
            batch_size=2,
            num_epochs=1,
            seed=42,
            distributed=True,
        )
        with pytest.raises(RuntimeError, match="requires a multi-process JAX runtime"):
            list(module.eval_iter())

    def test_distributed_multi_process_clears_guard(self, monkeypatch):
        """jax.process_count() > 1 allows train_iter and eval_iter (#6107).

        Paired with the single-process tests above, this is the negative control: the same
        module raises at process_count() == 1 and proceeds at 2.
        """
        import jax

        monkeypatch.setattr(jax, "process_count", lambda: 2)
        dataset = [1, 2, 3]
        module = DataModule(
            dataset=dataset,
            batch_size=2,
            num_epochs=1,
            seed=42,
            distributed=True,
        )
        assert list(module.train_iter()) == [1, 2, 3]
        assert list(module.eval_iter()) == [1, 2, 3]

    def test_distributed_false_never_consults_process_count(self, monkeypatch):
        """distributed=False must not raise whatever the process count is."""
        import jax

        monkeypatch.setattr(jax, "process_count", lambda: 1)
        module = DataModule(dataset=[1, 2], batch_size=2, num_epochs=1, seed=0, distributed=False)
        assert list(module.train_iter()) == [1, 2]

    def test_data_module_does_not_depend_on_xtrax_distributed(self):
        """The deprecated package is not imported by DataModule or its guard (#6107)."""
        import subprocess
        import sys

        code = (
            "import sys; import xtrax.data.module, xtrax.engine; "
            "sys.exit(1 if 'xtrax.distributed' in sys.modules else 0)"
        )
        result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True)
        assert result.returncode == 0, result.stderr.decode()


class TestDataModuleGrainPipeline:
    """DataModule yields Grain batches when use_grain_pipeline is set."""

    @requires_grain
    def test_train_iter_batches_and_eval_iter_does_not_shuffle(self):
        """Train shuffles with the seed; eval keeps source order."""
        source = [np.int32(i) for i in range(8)]
        module = DataModule(
            dataset=source,
            batch_size=2,
            num_epochs=1,
            seed=0,
            distributed=False,
            use_grain_pipeline=True,
        )
        train = [int(x) for batch in module.train_iter() for x in np.asarray(batch).ravel()]
        eval_ids = [int(x) for batch in module.eval_iter() for x in np.asarray(batch).ravel()]
        assert eval_ids == list(range(8))
        assert sorted(train) == list(range(8))
        assert train != eval_ids
        # The second call is the next epoch: it reshuffles with seed + 1.
        again = [int(x) for batch in module.train_iter() for x in np.asarray(batch).ravel()]
        assert sorted(again) == list(range(8))
        assert again != train

    @requires_grain
    def test_pad_fn_is_applied(self):
        """pad_fn runs before batching so every batch has one shape."""

        def pad(example: np.ndarray) -> np.ndarray:
            out = np.zeros(4, dtype=np.int32)
            out[: example.shape[0]] = example
            return out

        module = DataModule(
            dataset=[np.arange(i, dtype=np.int32) for i in range(1, 5)],
            batch_size=2,
            num_epochs=1,
            seed=0,
            distributed=False,
            use_grain_pipeline=True,
            pad_fn=pad,
        )
        batches = list(module.eval_iter())
        assert [batch.shape for batch in batches] == [(2, 4), (2, 4)]

    def test_distributed_grain_still_requires_multi_process(self):
        """The process-count guard runs before the pipeline is built."""
        module = DataModule(
            dataset=[np.int32(i) for i in range(4)],
            batch_size=2,
            num_epochs=1,
            seed=0,
            distributed=True,
            use_grain_pipeline=True,
        )
        with pytest.raises(RuntimeError, match="requires a multi-process JAX runtime"):
            list(module.train_iter())

    @requires_grain
    def test_distributed_grain_shards_by_process(self, monkeypatch):
        """distributed=True shards the source with the JAX process index."""
        import jax

        monkeypatch.setattr(jax, "process_index", lambda: 0)
        monkeypatch.setattr(jax, "process_count", lambda: 2)
        module = DataModule(
            dataset=[np.int32(i) for i in range(8)],
            batch_size=2,
            num_epochs=1,
            seed=0,
            distributed=True,
            use_grain_pipeline=True,
        )
        got = [int(x) for batch in module.eval_iter() for x in np.asarray(batch).ravel()]
        assert got == [0, 1, 2, 3]

    @requires_grain
    def test_distributed_false_yields_the_full_source(self, monkeypatch):
        """distributed=False keeps every example even when this process is one of two."""
        import jax

        monkeypatch.setattr(jax, "process_index", lambda: 1)
        monkeypatch.setattr(jax, "process_count", lambda: 2)
        module = DataModule(
            dataset=[np.int32(i) for i in range(8)],
            batch_size=2,
            num_epochs=1,
            seed=0,
            distributed=False,
            use_grain_pipeline=True,
        )
        got = [int(x) for batch in module.eval_iter() for x in np.asarray(batch).ravel()]
        assert got == list(range(8))

    @requires_grain
    def test_train_iter_uses_the_module_seed(self):
        """Train order matches build_input_pipeline(seed=7), not a constant seed."""
        source = [np.int32(i) for i in range(16)]

        def order(seed: int) -> list[int]:
            pipeline = build_input_pipeline(
                source,
                batch_size=4,
                seed=seed,
                num_epochs=1,
                shuffle=True,
            )
            return [int(x) for batch in pipeline for x in np.asarray(batch).ravel()]

        module = DataModule(
            dataset=source,
            batch_size=4,
            num_epochs=1,
            seed=7,
            distributed=False,
            use_grain_pipeline=True,
        )
        train = [int(x) for batch in module.train_iter() for x in np.asarray(batch).ravel()]
        assert train == order(7)
        assert train != order(0)

    @requires_grain
    def test_eval_iter_is_one_pass_regardless_of_num_epochs(self):
        """eval_iter ignores the training num_epochs and ends after one pass."""
        module = DataModule(
            dataset=[np.int32(i) for i in range(8)],
            batch_size=2,
            num_epochs=None,
            seed=0,
            distributed=False,
            use_grain_pipeline=True,
        )
        batches = list(itertools.islice(module.eval_iter(), 10))
        got = [int(x) for batch in batches for x in np.asarray(batch).ravel()]
        assert len(batches) == 4
        assert got == list(range(8))

    @requires_grain
    def test_eval_iter_keeps_the_partial_last_batch(self):
        """eval_iter does not drop the trailing examples that do not fill a batch."""
        module = DataModule(
            dataset=[np.int32(i) for i in range(5)],
            batch_size=2,
            num_epochs=1,
            seed=0,
            distributed=False,
            use_grain_pipeline=True,
        )
        shapes = [np.asarray(batch).shape for batch in module.eval_iter()]
        assert shapes == [(2,), (2,), (1,)]

    @requires_grain
    def test_train_iter_yields_one_epoch_per_call(self):
        """train_iter yields a single epoch even when num_epochs asks for more."""
        module = DataModule(
            dataset=[np.int32(i) for i in range(8)],
            batch_size=2,
            num_epochs=5,
            seed=0,
            distributed=False,
            use_grain_pipeline=True,
        )
        batches = list(module.train_iter())
        assert len(batches) == 4
        assert sorted(int(x) for batch in batches for x in np.asarray(batch).ravel()) == list(
            range(8)
        )

    @requires_grain
    def test_train_iter_varies_the_seed_per_call(self):
        """Call k of train_iter shuffles with seed + k, so epochs differ."""
        source = [np.int32(i) for i in range(16)]

        def order(seed: int) -> list[int]:
            pipeline = build_input_pipeline(
                source, batch_size=4, seed=seed, num_epochs=1, shuffle=True
            )
            return [int(x) for batch in pipeline for x in np.asarray(batch).ravel()]

        module = DataModule(
            dataset=source,
            batch_size=4,
            num_epochs=1,
            seed=7,
            distributed=False,
            use_grain_pipeline=True,
        )
        first = [int(x) for batch in module.train_iter() for x in np.asarray(batch).ravel()]
        second = [int(x) for batch in module.train_iter() for x in np.asarray(batch).ravel()]
        assert first == order(7)
        assert second == order(8)

    @requires_grain
    def test_collate_fn_is_the_grain_batch_fn(self):
        """A DataModule collate_fn replaces the default np.stack batching."""

        def collate(examples: list[np.ndarray]) -> np.ndarray:
            return np.stack(examples) + 100

        module = DataModule(
            dataset=[np.int32(i) for i in range(4)],
            batch_size=2,
            num_epochs=1,
            seed=0,
            distributed=False,
            use_grain_pipeline=True,
            collate_fn=collate,
        )
        batches = [np.asarray(batch) for batch in module.eval_iter()]
        assert [int(x) for batch in batches for x in batch.ravel()] == [100, 101, 102, 103]
