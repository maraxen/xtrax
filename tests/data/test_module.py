import numpy as np
import pytest

import xtrax.data.module as _mod
from xtrax.data.module import DataModule, _mark_dist_initialized
from xtrax.data.pipeline import build_input_pipeline


class TestDataModule:
    """Tests for DataModule iterator behavior and distributed guards."""

    @pytest.fixture(autouse=True)
    def reset_dist_flag(self):
        """Reset the _dist_initialized flag before each test."""
        _mod._dist_initialized = False
        yield
        _mod._dist_initialized = False

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

    def test_train_iter_distributed_without_init_raises(self):
        """train_iter raises RuntimeError when distributed=True without init_dist()."""
        dataset = [1, 2, 3]
        module = DataModule(
            dataset=dataset,
            batch_size=2,
            num_epochs=1,
            seed=42,
            distributed=True,
        )
        with pytest.raises(RuntimeError, match="distributed=True requires init_dist"):
            list(module.train_iter())

    def test_eval_iter_distributed_without_init_raises(self):
        """eval_iter raises RuntimeError when distributed=True without init_dist()."""
        dataset = [1, 2, 3]
        module = DataModule(
            dataset=dataset,
            batch_size=2,
            num_epochs=1,
            seed=42,
            distributed=True,
        )
        with pytest.raises(RuntimeError, match="distributed=True requires init_dist"):
            list(module.eval_iter())

    def test_mark_dist_initialized_clears_guard(self):
        """_mark_dist_initialized() allows subsequent train_iter and eval_iter calls."""
        dataset = [1, 2, 3]
        module = DataModule(
            dataset=dataset,
            batch_size=2,
            num_epochs=1,
            seed=42,
            distributed=True,
        )
        _mark_dist_initialized()
        result_train = list(module.train_iter())
        result_eval = list(module.eval_iter())
        assert result_train == [1, 2, 3]
        assert result_eval == [1, 2, 3]


class TestDataModuleGrainPipeline:
    """DataModule yields Grain batches when use_grain_pipeline is set."""

    @pytest.fixture(autouse=True)
    def reset_dist_flag(self):
        """Reset the _dist_initialized flag before each test."""
        _mod._dist_initialized = False
        yield
        _mod._dist_initialized = False

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
        again = [int(x) for batch in module.train_iter() for x in np.asarray(batch).ravel()]
        assert again == train

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

    def test_distributed_grain_still_requires_init(self):
        """The init_dist guard runs before the pipeline is built."""
        module = DataModule(
            dataset=[np.int32(i) for i in range(4)],
            batch_size=2,
            num_epochs=1,
            seed=0,
            distributed=True,
            use_grain_pipeline=True,
        )
        with pytest.raises(RuntimeError, match="distributed=True requires init_dist"):
            list(module.train_iter())

    def test_distributed_grain_shards_by_process(self, monkeypatch):
        """distributed=True shards the source with the JAX process index."""
        import jax

        monkeypatch.setattr(jax, "process_index", lambda: 0)
        monkeypatch.setattr(jax, "process_count", lambda: 2)
        _mark_dist_initialized()
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
