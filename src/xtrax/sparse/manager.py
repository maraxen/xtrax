from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import jax

from xtrax.sparse.policy import SparsePolicy

PyTree = Any
logger = logging.getLogger(__name__)


def _path_str(path) -> str:
	"""Convert a JAX tree path to dotted string notation."""
	return ".".join(p.key if hasattr(p, "key") else str(p) for p in path)


class SparseMaskManager:
    """Python-side mutable mask tracker. NOT an eqx.Module."""

    def __init__(self, policy: SparsePolicy) -> None:
        self.policy = policy
        self._masks: dict[str, jax.Array] = {}
        self._initialized: bool = False

    def step(
        self,
        params: PyTree,
        step: int,
        path_filter: Callable[[str], bool] = lambda _: True,
    ) -> PyTree:
        should_update = not self._initialized or self.policy.should_update(step)
        if should_update:
            for path, leaf in jax.tree_util.tree_leaves_with_path(params):
                path_str = _path_str(path)
                if path_filter(path_str) and hasattr(leaf, "ndim") and leaf.ndim >= 2:
                    self._masks[path_str] = self.policy.make_mask(leaf, step)
                else:
                    logger.debug("SparseMaskManager: skipping leaf %s", path_str)
            self._initialized = True

        # Old masks applied to current (updated) params on no-update steps — by design.
        def apply_leaf(path, leaf):
            path_str = _path_str(path)
            if path_str in self._masks:
                return self.policy.apply_mask(leaf, self._masks[path_str])
            return leaf

        return jax.tree_util.tree_map_with_path(apply_leaf, params)

    def current_masks(self) -> dict[str, jax.Array]:
        return dict(self._masks)
