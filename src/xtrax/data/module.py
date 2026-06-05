from collections.abc import Callable, Iterator
from typing import Any

import equinox as eqx

# Module-level flag for distributed init state (real integration with
# init_dist deferred to Phase 5/6)
_dist_initialized: bool = False


def _mark_dist_initialized() -> None:
    """Call after init_dist() to allow DataModule iterators to proceed."""
    global _dist_initialized
    _dist_initialized = True


class DataModule(eqx.Module):
    dataset: Any
    batch_size: int = eqx.field(static=True)
    num_epochs: int | None = eqx.field(static=True)  # None = cycle indefinitely
    seed: int = eqx.field(static=True)
    distributed: bool = eqx.field(static=True)
    collate_fn: Callable | None = eqx.field(static=True, default=None)

    def train_iter(self) -> Iterator[Any]:
        if self.distributed and not _dist_initialized:
            raise RuntimeError(
                "DataModule: distributed=True requires init_dist() before train_iter()."
            )
        yield from self.dataset

    def eval_iter(self) -> Iterator[Any]:
        if self.distributed and not _dist_initialized:
            raise RuntimeError(
                "DataModule: distributed=True requires init_dist() before eval_iter()."
            )
        yield from self.dataset
