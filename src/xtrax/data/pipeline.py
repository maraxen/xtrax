"""Grain input pipelines.

Grain is optional. This module imports it only when a pipeline is built.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

# Cheap in-memory reads got slower with more parallelism: 22.6k ex/s at 1
# read thread versus 2.1k ex/s at 8 (loop-e1 loader profile). Defaults stay
# at one thread and no multiprocessing workers.
DEFAULT_NUM_THREADS = 1
DEFAULT_PREFETCH_BUFFER_SIZE = 64
DEFAULT_MP_WORKERS = 0

__all__ = [
    "DEFAULT_MP_WORKERS",
    "DEFAULT_NUM_THREADS",
    "DEFAULT_PREFETCH_BUFFER_SIZE",
    "build_input_pipeline",
    "create_distributed_pipeline",
]


def _import_grain() -> Any:
    try:
        import grain  # ty: ignore[unresolved-import]
    except ImportError as exc:
        raise ImportError(
            "Building an xtrax data pipeline requires the optional 'grain' "
            "dependency. Install with: pip install 'xtrax[data]'"
        ) from exc
    return grain


def _ensure_absl_flags_parsed() -> None:
    """Mark absl flags parsed before Grain ``mp_prefetch``.

    Grain's worker pool reads absl flags on the first batch. Outside an absl
    app they are never parsed, and that batch raises ``UnparsedFlagAccessError``.
    Pytest masks the error, so the mark has to happen here.
    """
    from absl import flags

    if not flags.FLAGS.is_parsed():
        flags.FLAGS.mark_as_parsed()


def _shard_bounds(num_examples: int, options: Any) -> tuple[int, int]:
    """Even split for a public ``grain.sharding.ShardOptions``.

    Same intervals as Grain's ``even_split`` (not part of Grain's public
    API): ``drop_remainder`` keeps every shard the same length and drops the
    tail, otherwise the remainder is spread over the first shards.
    """
    per = num_examples // options.shard_count
    start = per * options.shard_index
    end = per * (options.shard_index + 1)
    unused = num_examples % options.shard_count
    if unused > 0 and not options.drop_remainder:
        start += min(options.shard_index, unused)
        end += min(options.shard_index + 1, unused)
    return start, end


def _resolve_shard_options(shard_by_process: bool, shard_options: Any) -> Any:
    if shard_options is not None:
        return shard_options
    if not shard_by_process:
        return None
    grain = _import_grain()
    return grain.sharding.ShardByJaxProcess(drop_remainder=True)


def build_input_pipeline(
    source: Any,
    *,
    batch_size: int,
    seed: int,
    num_epochs: int | None = 1,
    shuffle: bool = True,
    pad_fn: Callable[..., Any] | None = None,
    batch_fn: Callable[..., Any] | None = None,
    drop_remainder: bool = True,
    num_threads: int = DEFAULT_NUM_THREADS,
    prefetch_buffer_size: int = DEFAULT_PREFETCH_BUFFER_SIZE,
    mp_workers: int = DEFAULT_MP_WORKERS,
    mp_buffer: int = 2,
    device: Any = None,
    device_prefetch: int = 2,
    cpu_prefetch: int = 4,
    shard_by_process: bool = False,
    shard_options: Any = None,
) -> Any:
    """Build a Grain pipeline over a random-access source.

    Composition::

        MapDataset.source(source)
            [shard by ShardOptions]
            .seed(seed).shuffle()          # when shuffle
            .repeat(num_epochs)            # skipped when num_epochs == 1
            .map(pad_fn)                   # optional, caller-supplied
            .batch(batch_size, np.stack)
            .to_iter_dataset(ReadOptions(num_threads, prefetch_buffer_size))
            [.mp_prefetch]                 # when mp_workers > 0
            [device_put]                   # when device is set

    ``pad_fn`` is domain-free: the caller maps each example to one fixed
    shape so the train step compiles once. ``batch_fn`` defaults to
    ``numpy.stack``; elements must be stackable arrays or scalars.

    Process sharding uses ``shard_options`` when given, otherwise
    ``grain.sharding.ShardByJaxProcess`` (``jax.process_index`` /
    ``jax.process_count``) when ``shard_by_process`` is true. Sharding is
    a consecutive even split applied before shuffle, so each process
    shuffles only its own shard.

    ``num_threads`` defaults to 1 and ``mp_workers`` to 0. On cheap reads,
    more read threads and process workers made throughput worse (22.6k
    examples/s at 1 thread versus 2.1k at 8). Raise ``mp_workers`` only for
    CPU-bound decode. ``mp_workers > 0`` marks absl flags parsed first;
    otherwise the first batch raises ``UnparsedFlagAccessError`` outside
    an absl app.

    ``num_epochs=None`` repeats forever. ``device=None`` skips
    ``device_put``; ``device_prefetch=0`` skips it even when a device is
    set.

    Raises:
        ImportError: Grain is not installed (``xtrax[data]``).
        ValueError: ``batch_size`` is not positive, or ``mp_workers`` is negative.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if mp_workers < 0:
        raise ValueError(f"mp_workers must be >= 0, got {mp_workers}.")

    grain = _import_grain()
    ds = grain.MapDataset.source(source)
    resolved = _resolve_shard_options(shard_by_process, shard_options)
    if resolved is not None:
        start, end = _shard_bounds(len(ds), resolved)
        ds = ds[start:end]
    if shuffle:
        ds = ds.seed(seed).shuffle()
    if num_epochs != 1:
        ds = ds.repeat(num_epochs)
    if pad_fn is not None:
        ds = ds.map(pad_fn)
    ds = ds.batch(
        batch_size,
        drop_remainder=drop_remainder,
        batch_fn=np.stack if batch_fn is None else batch_fn,
    )
    it = ds.to_iter_dataset(
        grain.ReadOptions(num_threads=num_threads, prefetch_buffer_size=prefetch_buffer_size)
    )
    if mp_workers > 0:
        _ensure_absl_flags_parsed()
        it = it.mp_prefetch(
            grain.MultiprocessingOptions(num_workers=mp_workers, per_worker_buffer_size=mp_buffer)
        )
    if device is not None and device_prefetch > 0:
        it = grain.experimental.device_put(
            it,
            device,
            cpu_buffer_size=cpu_prefetch,
            device_buffer_size=device_prefetch,
        )
    return it


def create_distributed_pipeline(
    dataset: Any,
    global_batch_size: int,
    num_devices: int,
    seed: int,
    *,
    num_epochs: int | None = 1,
    shuffle: bool = True,
    pad_fn: Callable[..., Any] | None = None,
    batch_fn: Callable[..., Any] | None = None,
    drop_remainder: bool = True,
    num_threads: int = DEFAULT_NUM_THREADS,
    prefetch_buffer_size: int = DEFAULT_PREFETCH_BUFFER_SIZE,
    mp_workers: int = DEFAULT_MP_WORKERS,
    mp_buffer: int = 2,
    device: Any = None,
    device_prefetch: int = 2,
    cpu_prefetch: int = 4,
    shard_options: Any = None,
) -> Any:
    """Shard ``dataset`` across JAX processes and batch it.

    The per-device batch size is ``global_batch_size // num_devices``.
    Examples are split with ``ShardByJaxProcess`` (or ``shard_options``)
    before shuffle, so each process sees a disjoint shard.

    Raises:
        ValueError: ``global_batch_size`` is not divisible by ``num_devices``.
        ImportError: Grain is not installed (``xtrax[data]``).
    """
    if global_batch_size % num_devices != 0:
        raise ValueError(
            f"create_distributed_pipeline: global_batch_size={global_batch_size} "
            f"must be divisible by num_devices={num_devices}."
        )
    return build_input_pipeline(
        dataset,
        batch_size=global_batch_size // num_devices,
        seed=seed,
        num_epochs=num_epochs,
        shuffle=shuffle,
        pad_fn=pad_fn,
        batch_fn=batch_fn,
        drop_remainder=drop_remainder,
        num_threads=num_threads,
        prefetch_buffer_size=prefetch_buffer_size,
        mp_workers=mp_workers,
        mp_buffer=mp_buffer,
        device=device,
        device_prefetch=device_prefetch,
        cpu_prefetch=cpu_prefetch,
        shard_by_process=shard_options is None,
        shard_options=shard_options,
    )
