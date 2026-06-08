import logging

import jax.numpy as jnp

from xtrax.sparse.config import SparseConfig
from xtrax.sparse.manager import SparseMaskManager
from xtrax.sparse.policy import SparsePolicy


def _make_manager(budget=4, schedule=None):
    cfg = SparseConfig(nse_budget=budget, update_schedule=schedule or (lambda s: True))
    return SparseMaskManager(policy=SparsePolicy(config=cfg))


class TestSparseMaskManager:
    def test_first_call_always_masks(self):
        mgr = _make_manager(budget=4, schedule=lambda s: False)
        params = {"w": jnp.ones((3, 3))}
        mgr.step(params, step=0)
        assert mgr.current_masks() != {}

    def test_no_update_reuses_masks(self):
        mgr = _make_manager(budget=4, schedule=lambda s: s == 0)
        params = {"w": jnp.arange(9, dtype=jnp.float32).reshape(3, 3)}
        mgr.step(params, step=0)
        keys_after_0 = set(mgr.current_masks())
        mgr.step(params, step=1)
        assert set(mgr.current_masks()) == keys_after_0

    def test_update_step_recomputes(self):
        mgr = _make_manager(budget=4, schedule=lambda s: s % 2 == 0)
        params = {"w": jnp.ones((3, 3))}
        mgr.step(params, step=0)
        mgr.step(params, step=1)
        mgr.step(params, step=2)
        assert mgr._initialized

    def test_path_filter_excludes(self):
        mgr = _make_manager(budget=4)
        params = {"weight": jnp.ones((3, 3)), "bias": jnp.ones((3, 3))}
        mgr.step(params, step=0, path_filter=lambda p: "weight" in p)
        masks = mgr.current_masks()
        assert any("weight" in k for k in masks)
        assert not any("bias" in k for k in masks)

    def test_current_masks_is_copy(self):
        mgr = _make_manager(budget=4)
        params = {"w": jnp.ones((3, 3))}
        mgr.step(params, step=0)
        assert mgr.current_masks() is not mgr.current_masks()

    def test_step_logs_debug_for_skipped_1d_leaf(self, caplog):
        mgr = _make_manager(budget=8)
        params = {"weight": jnp.ones((4, 4)), "bias": jnp.ones((4,))}
        with caplog.at_level(logging.DEBUG, logger="xtrax.sparse.manager"):
            mgr.step(params, step=0)
        assert any("skipping leaf" in r.message for r in caplog.records)

    def test_step_logs_debug_when_path_filter_excludes_leaf(self, caplog):
        mgr = _make_manager(budget=8)
        params = {"weight": jnp.ones((4, 4)), "bias": jnp.ones((4,))}
        with caplog.at_level(logging.DEBUG, logger="xtrax.sparse.manager"):
            mgr.step(params, step=0, path_filter=lambda p: "weight" in p)
        assert any("skipping leaf" in r.message for r in caplog.records)
