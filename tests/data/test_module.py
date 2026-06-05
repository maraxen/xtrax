import pytest

import xtrax.data.module as _mod
from xtrax.data.module import DataModule, _mark_dist_initialized


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
