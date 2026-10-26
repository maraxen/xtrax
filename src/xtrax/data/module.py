import itertools
from collections.abc import Callable, Iterator
from typing import Any

from xtrax._tree import Struct, field

# Module-level flag for distributed init state (real integration with
# init_dist deferred to Phase 5/6)
_dist_initialized: bool = False


def _mark_dist_initialized() -> None:
    """Call after init_dist() to allow DataModule iterators to proceed."""
    global _dist_initialized
    _dist_initialized = True


class DataModule(Struct):
    """Batch source for ``Engine``.

    With ``use_grain_pipeline=False`` (the default) ``train_iter`` and
    ``eval_iter`` yield ``dataset`` unchanged. With ``use_grain_pipeline=True``
    they yield batches from :func:`xtrax.data.pipeline.build_input_pipeline`:
    train shuffles, eval does not, and ``distributed=True`` shards the source
    across JAX processes. ``pad_fn``, when set, is the caller-supplied
    fixed-length map applied before batching; ``collate_fn``, when set, is
    passed to Grain as the batch function.

    On the Grain path each ``train_iter()`` call yields exactly one epoch.
    Call ``k`` (counting from 0 on this instance) shuffles with
    ``seed + k``, so successive epochs see different orders and a fresh
    instance replays the same sequence. ``Engine.fit`` calls ``train_iter``
    once per epoch, so its ``num_epochs`` sets the number of passes;
    ``DataModule.num_epochs`` is not consulted by the iterators. ``eval_iter``
    is always one unshuffled pass that keeps the final partial batch.
    """

    dataset: Any
    batch_size: int = field(static=True)
    num_epochs: int | None = field(static=True)  # None = cycle indefinitely
    seed: int = field(static=True)
    distributed: bool = field(static=True)
    collate_fn: Callable | None = field(static=True, default=None)
    use_grain_pipeline: bool = field(static=True, default=False)
    pad_fn: Callable | None = field(static=True, default=None)
    # Epoch counter: each train_iter() call takes the next value. Immutable
    # modules cannot carry a plain int, so the counter object lives here.
    _epochs: Iterator[int] = field(static=True, init=False, default_factory=itertools.count)

    def train_iter(self) -> Iterator[Any]:
        """Yield one train epoch. Shuffles when the Grain pipeline is enabled."""
        epoch = next(self._epochs)
        yield from self._iter(
            shuffle=True, entry="train_iter", seed=self.seed + epoch, drop_remainder=True
        )

    def eval_iter(self) -> Iterator[Any]:
        """Yield one eval pass. Does not shuffle; keeps the final partial batch."""
        yield from self._iter(
            shuffle=False, entry="eval_iter", seed=self.seed, drop_remainder=False
        )

    def _iter(self, *, shuffle: bool, entry: str, seed: int, drop_remainder: bool) -> Iterator[Any]:
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
            seed=seed,
            num_epochs=1,
            shuffle=shuffle,
            pad_fn=self.pad_fn,
            batch_fn=self.collate_fn,
            drop_remainder=drop_remainder,
            shard_by_process=self.distributed,
        )
