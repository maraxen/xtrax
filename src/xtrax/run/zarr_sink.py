"""Zarr-backed staging sink for JAX io_callback-driven streaming output.

Generic host-side sink: callers stage arbitrary named numpy-array payloads
under an opaque key (e.g. batch/chunk indices), then drain them into a
chunked, on-disk Zarr store. Domain-specific dispatch -- deciding WHICH
payload maps to which JAX op, and firing the actual
``jax.experimental.io_callback`` -- is the caller's responsibility; this
module only owns staging and Zarr storage.

Requires the optional ``zarr`` dependency: ``pip install xtrax[io]``. Zarr
itself is imported lazily inside :meth:`ZarrStagingSink.__init__`, so
importing this module (or ``xtrax.run``) never requires zarr to be
installed -- only constructing a sink does.
"""

from typing import Any

import numpy as np

from xtrax.run.sink import SinkSpec


class ZarrStagingSink:
    """Stages keyed numpy-array payloads for incremental drain into a chunked Zarr store.

    Each staged key maps to a nested Zarr group -- the key's components,
    stringified and joined by ``/``, become the group path -- and named
    arrays staged under that key become sibling arrays within the group.
    Writes are batched: :meth:`stage` buffers in memory and only touches
    disk once ``spec.flush_every`` stage calls have accumulated (or
    :meth:`drain` is called explicitly).
    """

    def __init__(self, spec: SinkSpec) -> None:
        if spec.format != "zarr":
            msg = f"ZarrStagingSink requires SinkSpec.format == 'zarr', got {spec.format!r}"
            raise ValueError(msg)
        if spec.output_dir is None:
            msg = "ZarrStagingSink requires SinkSpec.output_dir"
            raise ValueError(msg)

        try:
            import zarr
        except ImportError as e:
            msg = (
                "ZarrStagingSink requires the optional 'zarr' dependency. "
                "Install with: pip install xtrax[io]"
            )
            raise ImportError(msg) from e

        self._spec = spec
        self._root: zarr.Group = zarr.open_group(str(spec.output_dir), mode="a")
        self._pending: dict[tuple[Any, ...], dict[str, np.ndarray]] = {}
        self._staged_since_drain = 0

    def stage(self, key: tuple[Any, ...], **arrays: Any) -> None:  # noqa: ANN401
        """Buffer one or more named arrays under ``key`` for later drain.

        Args:
            key: Opaque hashable tuple identifying this payload, e.g.
                ``(batch_idx, chunk_start, chunk_count)``.
            **arrays: Named numpy-convertible arrays to stage under ``key``.
                Repeated ``stage`` calls for the same key merge: later names
                overwrite earlier ones with the same name, new names
                accumulate alongside existing ones.
        """
        entry = self._pending.setdefault(key, {})
        entry.update({name: np.asarray(value) for name, value in arrays.items()})
        self._staged_since_drain += 1
        if self._staged_since_drain >= self._spec.flush_every:
            self.drain()

    def take(self, key: tuple[Any, ...]) -> dict[str, np.ndarray]:
        """Pop and return a still-buffered (not yet drained) payload for ``key``.

        Raises:
            KeyError: If ``key`` has no pending (undrained) entry.
        """
        try:
            return self._pending.pop(key)
        except KeyError as e:
            msg = f"ZarrStagingSink: no pending entry for key={key!r}"
            raise KeyError(msg) from e

    def drain(self) -> None:
        """Write all pending payloads into the Zarr store and clear the buffer."""
        for key, arrays in self._pending.items():
            group_path = "/".join(str(part) for part in key)
            group = self._root.require_group(group_path) if group_path else self._root
            for name, array in arrays.items():
                arr = group.create_array(
                    name=name,
                    shape=array.shape,
                    dtype=array.dtype,
                    chunks=array.shape if array.shape else (1,),
                    overwrite=True,
                )
                arr[...] = array
        self._pending.clear()
        self._staged_since_drain = 0

    def __len__(self) -> int:
        """Number of keys currently buffered (not yet drained)."""
        return len(self._pending)
