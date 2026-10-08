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
    """Batch source for ``Engine``.

    With ``use_grain_pipeline=False`` (the default) ``train_iter`` and
    ``eval_iter`` yield ``dataset`` unchanged. With ``use_grain_pipeline=True``
    they yield batches from :func:`xtrax.data.pipeline.build_input_pipeline`:
    train shuffles, eval does not, and ``distributed=True`` shards the source
    across JAX processes. ``pad_fn``, when set, is the caller-supplied
    fixed-length map applied before batching.
    """

    dataset: Any
    batch_size: int = eqx.field(static=True)
    num_epochs: int | None = eqx.field(static=True)  # None = cycle indefinitely
    seed: int = eqx.field(static=True)
    distributed: bool = eqx.field(static=True)
    collate_fn: Callable | None = eqx.field(static=True, default=None)
    use_grain_pipeline: bool = eqx.field(static=True, default=False)
    pad_fn: Callable | None = eqx.field(static=True, default=None)

    def train_iter(self) -> Iterator[Any]:
        """Yield train batches. Shuffles when the Grain pipeline is enabled."""
        yield from self._iter(shuffle=True, entry="train_iter")

    def eval_iter(self) -> Iterator[Any]:
        """Yield eval batches. Does not shuffle when the Grain pipeline is enabled."""
        yield from self._iter(shuffle=False, entry="eval_iter")

    def _iter(self, *, shuffle: bool, entry: str) -> Iterator[Any]:
        if self.distributed and not _dist_initialized:
            raise RuntimeError(
                f"DataModule: distributed=True requires init_dist() before {entry}()."
            )
        if not self.use_grain_pipeline:
            yield from self.dataset
            return
        from xtrax.data.pipeline import build_input_pipeline

        yield from build_input_pipeline(
            self.dataset,
            batch_size=self.batch_size,
            seed=self.seed,
            num_epochs=self.num_epochs,
            shuffle=shuffle,
            pad_fn=self.pad_fn,
            shard_by_process=self.distributed,
        )
