"""Utility functions for computing deterministic digests over payloads, arrays, and source trees.

Used by durability and reproducibility infrastructure to verify that logical content
has not changed (independent of when it was written, which session or process wrote it,
or operational provenance like git state).
"""

import hashlib
import os
import socket
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xtrax.run.zarr_integrity import canonical_json_bytes, normalize_json_value


def canonical_digest(payload: Mapping[str, Any]) -> str:
    """Compute a deterministic sha256 hexdigest of the payload.

    The payload is normalized (JSON-safe, sorted keys, NFC strings) and
    serialized to canonical JSON bytes, then hashed.

    Args:
        payload: A mapping (dict-like) to digest.

    Returns:
        A lowercase hexadecimal sha256 hash.
    """
    normalized = dict(normalize_json_value(dict(payload)))
    return hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()


def array_digest(x: Any) -> str:  # noqa: ANN401
    """Compute a deterministic sha256 hexdigest over an array.

    The array's dtype, shape, and little-endian-canonical bytes are folded
    into the digest. JAX arrays are block-waited, device-gotten, and
    converted to numpy; typed PRNG keys are replaced with their key_data
    before hashing.

    Args:
        x: A numpy or JAX array, or any array-like object convertible to numpy.

    Returns:
        A lowercase hexadecimal sha256 hash.
    """
    import jax
    import numpy as np

    digest = hashlib.sha256()
    # Handle JAX arrays: block_until_ready, replace typed keys, device_get.
    if isinstance(x, jax.Array):
        jax.block_until_ready(x)
        # If it's a typed PRNG key, replace it with its key_data.
        if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
            x = jax.random.key_data(x)
        x = jax.device_get(x)
    # Convert to numpy array.
    arr = np.asarray(x)
    # Use the existing update_array_digest helper from zarr_integrity.
    from xtrax.run.zarr_integrity import update_array_digest

    update_array_digest(digest, arr)
    return digest.hexdigest()


def source_fingerprint(paths: Mapping[str, Path]) -> str:
    """Compute a deterministic sha256 hexdigest over a collection of Python source files.

    Each label (e.g. "denxity", "xtrax.tiling") maps to a file or directory path.
    For directories, all ``*.py`` files are collected via ``rglob("*.py")``, skipping
    any path with a ``__pycache__`` component. For files, the file itself is included.

    Each collected file's entry is:
    - Entry name: ``"<label>/<relpath>"`` (relative to the directory; for a single
      file, ``"<label>/<filename>"``).
    - Digest contribution: ``<entry_name_utf8> + b"\\0" + <length_ascii> + b"\\0" + <file_bytes>``.

    All entries are sorted by entry name across all labels, then digested in that order.

    Args:
        paths: A mapping from label (str) to file or directory Path.

    Returns:
        A lowercase hexadecimal sha256 hash.

    Raises:
        FileNotFoundError: If any path does not exist.
        ValueError: If no .py files are found overall.
    """
    digest = hashlib.sha256()
    entries: dict[str, bytes] = {}
    for label, path in paths.items():
        path = Path(path)
        if not path.exists():
            msg = f"source_fingerprint: path for label {label!r} does not exist: {path}"
            raise FileNotFoundError(msg)
        if path.is_file():
            # Single file: entry name is label/filename, file bytes are the content.
            with path.open("rb") as f:
                file_bytes = f.read()
            entry_name = f"{label}/{path.name}"
            entries[entry_name] = file_bytes
        elif path.is_dir():
            # Directory: collect all .py files, build entries for each.
            py_files = sorted(p for p in path.rglob("*.py") if "__pycache__" not in p.parts)
            if not py_files:
                msg = f"source_fingerprint: no .py files found under {label}={path}"
                raise ValueError(msg)
            for py_file in py_files:
                relpath = py_file.relative_to(path)
                entry_name = f"{label}/{relpath.as_posix()}"
                with py_file.open("rb") as f:
                    file_bytes = f.read()
                entries[entry_name] = file_bytes
        else:
            msg = (
                f"source_fingerprint: path for label {label!r} is neither file nor "
                f"directory: {path}"
            )
            raise ValueError(msg)
    if not entries:
        msg = "source_fingerprint: no .py files found overall"
        raise ValueError(msg)
    # Digest all entries in sorted order.
    for entry_name in sorted(entries.keys()):
        file_bytes = entries[entry_name]
        digest.update(entry_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(file_bytes)).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_bytes)
    return digest.hexdigest()


def numerics_env() -> dict[str, Any]:
    """Capture the numerics environment: JAX backend, hardware, and XLA flags.

    Returns a dict with two top-level keys:
    - ``"numerics"``: JAX-specific config (backend, device_kind, XLA_FLAGS, matmul precision,
      deterministic ops flag).
    - ``"info"``: Deployment-specific info (hostname, SLURM job ID if set).

    Returns:
        A dict with structure::

            {
                "numerics": {
                    "platform": str,
                    "device_kind": str,
                    "XLA_FLAGS": str,
                    "jax_default_matmul_precision": str,
                    "xla_gpu_deterministic_ops": bool,
                },
                "info": {
                    "hostname": str,
                    "slurm_job_id": str | None,
                },
            }
    """
    import jax

    xla_flags = os.environ.get("XLA_FLAGS", "")
    return {
        "numerics": {
            "platform": jax.default_backend(),
            "device_kind": jax.devices()[0].device_kind,
            "XLA_FLAGS": xla_flags,
            "jax_default_matmul_precision": str(jax.config.jax_default_matmul_precision),
            "xla_gpu_deterministic_ops": "--xla_gpu_deterministic_ops=true" in xla_flags,
        },
        "info": {
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }
