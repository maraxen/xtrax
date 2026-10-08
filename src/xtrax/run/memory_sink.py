"""In-memory staging sink with the same stage/drain/finalize protocol as Zarr.

No optional dependency: constructing a memory sink never imports zarr. Payloads
stay in process. :meth:`MemorySink.finalize` returns a :class:`~xtrax.run.sink.SinkReceipt`
labeled :data:`~xtrax.run.zarr_integrity.MEMORY_DIGEST_ALGO_VERSION`.

The digest is sha256 over canonical JSON of ``{group_path: {array_name: array_digest}}``.
Array names and array bytes are inputs. Caller attrs, run id, seed, and git
provenance are not, so changing an attr does not change the digest and two sinks
with the same arrays and different run ids share one. That algorithm is not the
zarr node walk (:data:`~xtrax.run.zarr_integrity.DIGEST_ALGO_VERSION`); memory and
zarr digests are not comparable.

``stage`` copies arrays, so later mutation of the caller's input does not change
the buffer. ``read`` returns copies of the committed arrays.
"""

import warnings
from typing import Any

import numpy as np

from xtrax import __version__ as _XTRAX_VERSION
from xtrax.run._sink_names import CORE_PROVENANCE_FIELDS, PRODUCER_NAME, RESERVED_ATTR_PREFIX
from xtrax.run.digest import array_digest, canonical_digest
from xtrax.run.sink import SinkReceipt, SinkSpec
from xtrax.run.zarr_integrity import MEMORY_DIGEST_ALGO_VERSION


def _zarr_sink():
    """Lazy: zarr_sink imports ``xtrax.run``'s commit helpers, which are not ready mid-import."""
    from xtrax.run import zarr_sink

    return zarr_sink


class MemorySink:
    """Stages keyed numpy payloads in memory and commits them on :meth:`drain`.

    ``spec.append=False`` (the default) replaces an array of the same name on a
    later drain. ``spec.append=True`` concatenates along axis 0, matching
    :class:`~xtrax.run.zarr_sink.ZarrStagingSink`. ``read`` / ``read_attrs``
    return the committed view.
    """

    def __init__(self, spec: SinkSpec) -> None:
        if spec.format != "memory":
            msg = f"MemorySink requires SinkSpec.format == 'memory', got {spec.format!r}"
            raise ValueError(msg)
        if not spec.run_id or not spec.run_id.strip():
            msg = "MemorySink requires a non-blank SinkSpec.run_id"
            raise ValueError(msg)
        self._spec = spec
        self._closed = False
        self._finalized = False
        self._pending: dict[tuple[Any, ...], dict[str, np.ndarray]] = {}
        self._pending_attrs: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._committed: dict[tuple[Any, ...], dict[str, np.ndarray]] = {}
        self._committed_attrs: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._reserved: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._staged_since_drain = 0
        git_sha, git_branch, git_dirty = _zarr_sink()._resolve_git_provenance(spec.provenance)
        self.provenance: dict[str, Any] = {
            "run_id": spec.run_id,
            "git_sha": git_sha,
            "git_branch": git_branch,
            "git_dirty": git_dirty,
            "producer": PRODUCER_NAME,
            "xtrax_version": _XTRAX_VERSION,
        }
        if spec.seed is not None:
            self.provenance["seed"] = spec.seed

    def _require_open(self, what: str) -> None:
        if self._closed:
            msg = (
                f"MemorySink: {what}() after close() is not legitimate -- "
                "close() ends this sink's use; open a fresh sink."
            )
            raise RuntimeError(msg)
        if self._finalized and what != "finalize":
            msg = (
                f"MemorySink: {what}() after finalize() is not legitimate -- "
                "finalize() ends the run; use a fresh sink."
            )
            raise RuntimeError(msg)

    def _validate_stage_attrs(self, key: tuple[Any, ...], attrs: dict[str, Any]) -> None:
        collisions = sorted(CORE_PROVENANCE_FIELDS.intersection(attrs))
        if collisions:
            msg = (
                f"MemorySink: staged attrs for key={key!r} use reserved core provenance "
                f"field name(s) {collisions}"
            )
            raise ValueError(msg)
        reserved = sorted(name for name in attrs if str(name).startswith(RESERVED_ATTR_PREFIX))
        if reserved:
            msg = (
                f"MemorySink: staged attrs for key={key!r} use reserved namespace "
                f"{RESERVED_ATTR_PREFIX!r} (attr name(s) {reserved})"
            )
            raise ValueError(msg)
        zs = _zarr_sink()
        schema = zs._schema_for_key(self._spec, key)
        if schema is None:
            return
        merged = dict(self._pending_attrs.get(key, {}))
        merged.update(attrs)
        errors = zs._validate_attrs_against_schema(merged, schema, check_required=False)
        if errors:
            msg = (
                f"MemorySink: staged attrs for key={key!r} violate the SinkSpec "
                f"{zs._schema_label(self._spec, key)}: {'; '.join(errors)}"
            )
            raise ValueError(msg)

    def stage(
        self,
        key: tuple[Any, ...],
        attrs: dict[str, Any] | None = None,
        *,
        input_digest: str | None = None,
        input_payload: Any | None = None,  # noqa: ANN401
        commit_meta: Any | None = None,  # noqa: ANN401
        env_extra: Any | None = None,  # noqa: ANN401
        **arrays: Any,  # noqa: ANN401
    ) -> None:
        """Buffer named arrays under ``key``. See :meth:`ZarrStagingSink.stage`."""
        self._require_open("stage")
        if any(v is not None for v in (input_digest, input_payload, commit_meta, env_extra)):
            msg = (
                "MemorySink: input_digest/input_payload/commit_meta/env_extra are "
                "durable-only arguments"
            )
            raise ValueError(msg)
        if attrs:
            self._validate_stage_attrs(key, attrs)
        # Copy at stage so take() and a later drain cannot alias the caller.
        converted = {name: np.array(value, copy=True) for name, value in arrays.items()}
        self._pending.setdefault(key, {}).update(converted)
        if attrs:
            self._pending_attrs.setdefault(key, {}).update(attrs)
        self._staged_since_drain += 1
        if self._staged_since_drain >= self._spec.flush_every:
            self.drain()

    def take(self, key: tuple[Any, ...]) -> dict[str, np.ndarray]:
        """Pop a still-buffered payload. Discards pending attrs for ``key``."""
        self._pending_attrs.pop(key, None)
        try:
            return self._pending.pop(key)
        except KeyError as e:
            msg = f"MemorySink: no pending entry for key={key!r}"
            raise KeyError(msg) from e

    def _validate_required_before_drain(self) -> None:
        if self._spec.extension_schema is None and not self._spec.level_schemas:
            return
        failures: list[str] = []
        zs = _zarr_sink()
        for key, pending in self._pending_attrs.items():
            schema = zs._schema_for_key(self._spec, key)
            if schema is None or not schema.get("required"):
                continue
            merged = dict(self._committed_attrs.get(key, {}))
            merged.update(pending)
            missing = zs._missing_required_fields(merged, schema)
            if missing:
                failures.append(f"key={key!r}: {'; '.join(missing)}")
        if failures:
            msg = (
                "MemorySink: drain() refuses to persist attrs that are incomplete "
                f"under the SinkSpec schema ({' | '.join(failures)}); nothing was written "
                "and the buffer is intact"
            )
            raise ValueError(msg)

    def drain(self) -> dict[tuple[Any, ...], Any]:
        """Commit pending payloads into the in-memory store and clear the buffer.

        Returns an empty dict (there is no durable commit outcome).
        """
        self._require_open("drain")
        self._validate_required_before_drain()
        for key, arrays in self._pending.items():
            stored = self._committed.setdefault(key, {})
            # Plan every array before assigning so a rejected append leaves this key intact.
            planned = {
                name: _store_array(stored.get(name), array, append=self._spec.append)
                for name, array in arrays.items()
            }
            stored.update(planned)
            attrs = dict(self._committed_attrs.get(key, {}))
            attrs.update(self._pending_attrs.get(key, {}))
            attrs.update(
                {"run_id": self.provenance["run_id"], "git_sha": self.provenance["git_sha"]}
            )
            self._committed_attrs[key] = attrs
        self._pending.clear()
        self._pending_attrs.clear()
        self._staged_since_drain = 0
        return {}

    def read(self, key: tuple[Any, ...]) -> dict[str, np.ndarray]:
        """Return copies of the arrays committed under ``key``."""
        try:
            arrays = self._committed[key]
        except KeyError as e:
            msg = f"MemorySink: no committed entry for key={key!r}"
            raise KeyError(msg) from e
        return {name: value.copy() for name, value in arrays.items()}

    def read_attrs(self, key: tuple[Any, ...]) -> dict[str, Any]:
        """Return a copy of the attrs committed under ``key``."""
        if key not in self._committed and key not in self._committed_attrs:
            msg = f"MemorySink: no committed entry for key={key!r}"
            raise KeyError(msg)
        return dict(self._committed_attrs.get(key, {}))

    def stamp_reserved(self, key: tuple[Any, ...], name: str, payload: dict[str, Any]) -> None:
        """Record a reserved-namespace attr. Same name rules as the Zarr sink."""
        self._require_open("stamp_reserved")
        if not name:
            msg = "MemorySink.stamp_reserved: name must be non-empty"
            raise ValueError(msg)
        if "/" in name or "." in name:
            msg = f"MemorySink.stamp_reserved: name={name!r} may not contain '/' or '.'"
            raise ValueError(msg)
        if name in {"commit", "store"}:
            msg = f"MemorySink.stamp_reserved: name={name!r} is reserved by the sink itself"
            raise ValueError(msg)
        from xtrax.run.zarr_integrity import canonical_json_bytes
        from xtrax.run.zarr_integrity import normalize_json_value as nv

        try:
            normalized = dict(nv(dict(payload)))
            canonical_json_bytes(normalized)
        except (TypeError, ValueError) as e:
            msg = f"MemorySink.stamp_reserved: payload is not JSON-safe: {e}"
            raise ValueError(msg) from e
        self._reserved.setdefault(key, {})[RESERVED_ATTR_PREFIX + name] = normalized

    def finalize(self) -> SinkReceipt:
        """Hash committed arrays and return a receipt. May run only once."""
        if self._finalized:
            msg = "MemorySink: finalize() already ran for this sink; it may run only once."
            raise RuntimeError(msg)
        if self._pending:
            msg = (
                f"MemorySink: finalize() refuses to complete with {len(self._pending)} "
                "staged key(s) undrained -- call drain() or take() first."
            )
            raise RuntimeError(msg)
        payload: dict[str, Any] = {}
        for key in sorted(self._committed, key=lambda item: tuple(str(part) for part in item)):
            path = "/".join(str(part) for part in key)
            arrays = self._committed[key]
            payload[path] = {name: array_digest(arrays[name]) for name in sorted(arrays)}
        self._finalized = True
        return SinkReceipt(
            path=self._spec.output_dir,
            digest=canonical_digest(payload),
            digest_algo_version=MEMORY_DIGEST_ALGO_VERSION,
            run_id=self._spec.run_id,
            seed=self._spec.seed,
        )

    def close(self) -> None:
        """Discard undrained keys and refuse later stage/drain calls."""
        if self._closed:
            return
        self._closed = True
        if self._pending:
            warnings.warn(
                f"MemorySink.close(): discarding {len(self._pending)} staged key(s) that "
                "were never drained.",
                UserWarning,
                stacklevel=2,
            )
            self._pending.clear()
            self._pending_attrs.clear()

    def __enter__(self) -> "MemorySink":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __len__(self) -> int:
        """Number of keys currently buffered (not yet drained)."""
        return len(self._pending)


def _store_array(previous: np.ndarray | None, array: np.ndarray, *, append: bool) -> np.ndarray:
    """Copy ``array`` into the committed store, appending along axis 0 when asked."""
    incoming = np.array(array, copy=True)
    if previous is None or not append:
        return incoming
    if incoming.ndim == 0 or previous.ndim == 0:
        msg = "MemorySink.drain: cannot append a scalar array along a leading axis"
        raise ValueError(msg)
    if incoming.ndim != previous.ndim or incoming.shape[1:] != previous.shape[1:]:
        msg = (
            "MemorySink.drain: cannot append array: trailing shape "
            f"{incoming.shape[1:]} does not match stored {previous.shape[1:]}"
        )
        raise ValueError(msg)
    if incoming.dtype != previous.dtype:
        msg = (
            f"MemorySink.drain: cannot append array: dtype {incoming.dtype} "
            f"does not match stored {previous.dtype}"
        )
        raise ValueError(msg)
    return np.concatenate([previous, incoming], axis=0)
