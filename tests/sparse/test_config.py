import pytest

from xtrax.sparse.config import SparseConfig


class TestSparseConfig:
    def test_frozen(self):
        cfg = SparseConfig(nse_budget=4, update_schedule=lambda s: s % 2 == 0)
        with pytest.raises((AttributeError, TypeError)):
            cfg.nse_budget = 8

    def test_schedule(self):
        cfg = SparseConfig(nse_budget=4, update_schedule=lambda s: s % 2 == 0)
        assert cfg.update_schedule(0) is True
        assert cfg.update_schedule(1) is False

    def test_default_fallback_mode(self):
        cfg = SparseConfig(nse_budget=4, update_schedule=lambda s: True)
        assert cfg.fallback_mode == "dense_mask"
