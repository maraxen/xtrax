"""Content-digest and durability primitives for Zarr directory stores.

Generic building blocks for callers that need to verify a Zarr store's
content deterministically (e.g. a done-marker recording "this output is
exactly what I wrote") and durabilize a directory-of-many-files store to
disk before trusting that verification. Domain-specific orchestration --
locking, atomic promotion, done-marker schemas -- is the caller's
responsibility; this module only owns digesting and fsyncing.

``zarr_content_digest`` and ``update_zarr_node_digest`` require the optional
``zarr`` dependency: ``pip install xtrax[io]``. Zarr is imported lazily
inside those two functions, so importing this module (or ``xtrax.run``)
never requires zarr to be installed -- only calling Zarr-touching functions
does. The other functions here (JSON canonicalization, array digesting,
fsync) have no zarr dependency at all.
"""

import hashlib
import json
import os
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import zarr


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    """Serialize ``payload`` to deterministic, sorted-key, NFC-normalized JSON bytes."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def normalize_json_value(value: Any) -> Any:  # noqa: ANN401
    """Recursively coerce ``value`` into JSON-safe, canonically-ordered form.

    Numpy scalars/arrays become Python scalars/lists, dict keys are
    stringified and sorted, strings are NFC-normalized, bytes are decoded
    as UTF-8.
    """
    normalized: Any = value
    if isinstance(value, np.generic):
        normalized = normalize_json_value(value.item())
    elif isinstance(value, np.ndarray):
        normalized = [normalize_json_value(item) for item in value.tolist()]
    elif isinstance(value, (list, tuple)):
        normalized = [normalize_json_value(item) for item in value]
    elif isinstance(value, dict):
        normalized = {str(key): normalize_json_value(item) for key, item in sorted(value.items())}
    elif isinstance(value, bytes):
        normalized = value.decode("utf-8")
    elif isinstance(value, str):
        normalized = unicodedata.normalize("NFC", value)
    return normalized


def update_array_digest(digest: "hashlib._Hash", array: np.ndarray) -> None:
    """Fold ``array``'s dtype, shape, and little-endian-canonicalized bytes into ``digest``."""
    digest.update(array.dtype.name.encode("utf-8"))
    digest.update(b"|")
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(b"|")
    canonical = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
    digest.update(canonical.tobytes(order="C"))


def update_zarr_node_digest(
    digest: "hashlib._Hash",
    node: "zarr.Group | zarr.Array",
    path: str,
) -> None:
    """Recursively fold a Zarr group/array's path, attrs, and data into ``digest``.

    Requires the optional ``zarr`` dependency at call time (not import time).
    """
    import zarr

    digest.update(path.encode("utf-8"))
    digest.update(b"\n")
    attrs_payload = {
        str(key): normalize_json_value(value) for key, value in sorted(node.attrs.items())
    }
    digest.update(canonical_json_bytes(attrs_payload))
    digest.update(b"\n")
    if isinstance(node, zarr.Array):
        update_array_digest(digest, np.asarray(node[...]))
        digest.update(b"\n")
        return
    for key in sorted(node.keys()):
        child = node[key]
        update_zarr_node_digest(digest, child, f"{path}/{key}")


def zarr_content_digest(path: Path) -> str:
    """Compute a deterministic sha256 content digest of the Zarr store at ``path``.

    Covers every node's path, attrs, and (for arrays) data -- unaffected by
    filesystem metadata (mtimes, chunk-file layout) or which process/session
    wrote the store, only by the store's logical content.

    Raises:
        ImportError: If the optional ``zarr`` dependency is not installed.
    """
    try:
        import zarr
    except ImportError as e:
        msg = (
            "zarr_content_digest requires the optional 'zarr' dependency. "
            "Install with: pip install xtrax[io]"
        )
        raise ImportError(msg) from e

    digest = hashlib.sha256()
    root = zarr.open_group(str(path), mode="r")
    update_zarr_node_digest(digest, root, "/")
    return digest.hexdigest()


def fsync_file(path: Path) -> None:
    """Force a single file's data to disk."""
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    """Force a single directory's entries (not its children's contents) to disk."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_tree(path: Path) -> None:
    """Recursively durabilize a directory tree: every file's data, then every
    directory's entries bottom-up (deepest first), ending with ``path`` itself.

    Zarr stores are directories of many chunk/metadata files, unlike a single
    HDF5 file -- content-digest verification is only meaningful if every
    file's bytes are actually on disk first.
    """
    for child in path.rglob("*"):
        if child.is_file():
            fsync_file(child)
    dirs = sorted(
        (p for p in path.rglob("*") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    )
    for d in dirs:
        fsync_directory(d)
    fsync_directory(path)
