"""Record adapters for consumers whose per-unit record is not a ``zarr_commit`` record.

Spec 261030 §3.1 (Record adapters). Each adapter reads a consumer's own record (the truth) and
maps it onto :class:`xtrax.run.unit_index.CoreRecord`. Formats were read from the consumers'
``origin/main`` on 2026-11-03; an adapter reads, it never writes a consumer's files.

* :class:`AminxDoneMarkerAdapter`: aminx ``<output>.done.json``, ``campaign_done_marker_v3``
  (aminx ``host/campaign.py`` ``_write_done_marker`` / ``_validate_done_marker``). ``unit_key``
  is ``manifest_row_hash``; ``input_hash`` → ``input_digest``; ``content_digest_sha256`` (a
  ``zarr_content_digest``) → ``content_digest``; ``attempt_id`` → ``run_id``. ``record_digest``
  is computed over that mapped core.
* :class:`AlynxrFragmentAdapter`: alynxr fold-stats ``parts/<part>.json`` fragments
  (``alynxr.fold_stats_part/1``, alynxr ``fold_stats/layout.py``). ``unit_key`` and ``locator``
  are the part path; ``inputs_sha256`` → ``input_digest``; ``logical_sha256`` →
  ``content_digest``. Verification is alynxr's own reuse rule: the part file's size and sha256
  match its fragment (the logical hash needs pyarrow, the transfer hash does not).

Stdlib only at import time; aminx content verification imports ``xtrax.run.zarr_integrity``
(and so zarr) when it runs.
"""

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

from xtrax.run.unit_index import (
    CoreRecord,
    IndexReuse,
    IndexStale,
    SchemaMismatch,
    UnitIndexError,
    canonical_store,
)
from xtrax.run.zarr_commit import Corrupt, Missing

AMINX_DONE_MARKER = "aminx_done_marker_v3"
AMINX_V3 = "campaign_done_marker_v3"
AMINX_V2 = "campaign_done_marker_v2"
ALYNXR_FRAGMENT_ADAPTER = "alynxr_fold_stats_fragment"
ALYNXR_FRAGMENT = "alynxr.fold_stats_part/1"
_DONE = ".done.json"


def _digest(adapter: str, fields: dict[str, Any]) -> str:
    """sha256 of canonical JSON of the mapped core, tagged with the adapter name."""
    payload = json.dumps({"adapter": adapter, **fields}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Any:
    from xtrax.run.unit_index import _read_bytes

    return json.loads(_read_bytes(path))


def _inside(store: Path, locator: str) -> bool:
    """``locator`` is a plain relative path whose target resolves strictly inside ``store``.

    Rows are not trusted: ``../x``, an absolute path, ``.``/empty parts, or a symlink that leads
    out of the store would make an adapter read (and reuse) an artifact the store does not hold.
    """
    parts = locator.split("/")
    if not locator or "\0" in locator or locator.startswith("/"):
        return False
    if any(p in ("", ".", "..") for p in parts):
        return False
    root = os.path.realpath(store)
    target = os.path.realpath(os.path.join(root, locator))
    return target.startswith(root + os.sep)


# ---- aminx ---------------------------------------------------------------------------------------


def _aminx_core(marker: dict[str, Any]) -> CoreRecord | None:
    """The mapped core of a v3 marker that carries an input hash; None otherwise."""
    if marker.get("schema_version") != AMINX_V3:
        return None
    row, inp = marker.get("manifest_row_hash"), marker.get("input_hash")
    content, attempt = marker.get("content_digest_sha256"), marker.get("attempt_id")
    if not all(isinstance(v, str) and v for v in (row, inp, content)):
        return None
    run_id = attempt if isinstance(attempt, str) else ""
    fields = {
        "manifest_row_hash": row,
        "input_hash": inp,
        "content_digest_sha256": content,
        "attempt_id": run_id,
        "completed_at_unix_s": marker.get("completed_at_unix_s"),
    }
    return CoreRecord(str(row), str(inp), str(content), _digest(AMINX_DONE_MARKER, fields), run_id)


class AminxDoneMarkerAdapter:
    """aminx ``campaign_done_marker_v3`` records; the locator is the output path in the store."""

    name = AMINX_DONE_MARKER

    def find(self, store: Path) -> Iterator[tuple[str, CoreRecord]]:
        store = Path(store)
        for dirpath, _dirs, files in os.walk(store):
            for f in sorted(files):
                if not f.endswith(_DONE):
                    continue
                marker_path = Path(dirpath) / f
                try:
                    marker = _read_json(marker_path)
                except (OSError, ValueError):
                    continue  # a marker is written last and atomically; skip what is not one
                if not isinstance(marker, dict):
                    continue
                locator = marker_path.relative_to(store).as_posix()[: -len(_DONE)]
                core = _aminx_core(marker)
                if core is not None:
                    yield locator, core

    def classify(self, store: Path, locator: str, input_digest: str, *, verify: bool) -> Any:
        store = Path(store)
        if not _inside(store, locator):
            return Corrupt(f"locator {locator!r} is not a path inside the store")
        output = store / locator
        try:
            marker = _read_json(output.with_name(output.name + _DONE))
        except FileNotFoundError:
            return Missing()
        except (OSError, ValueError) as e:
            return Corrupt(f"unreadable done marker: {type(e).__name__}: {e}")
        if not isinstance(marker, dict):
            return Corrupt("done marker is not a JSON object")
        version = marker.get("schema_version")
        if version != AMINX_V3:
            return SchemaMismatch(f"done marker schema {version!r}, expected {AMINX_V3!r}")
        core = _aminx_core(marker)
        if core is None:
            return Corrupt("v3 done marker without manifest_row_hash, input_hash or digest")
        if not output.exists():
            return Corrupt(f"done marker present but output store is missing: {locator}")
        if core.input_digest != input_digest:
            return IndexStale(canonical_store(store), locator, core)
        if verify:
            from xtrax.run.zarr_integrity import zarr_content_digest

            try:
                observed = zarr_content_digest(output)
            except OSError:
                raise  # environmental I/O says nothing about the unit (as zarr_commit.lookup)
            except Exception as e:  # noqa: BLE001 - classified, as zarr_commit.lookup does
                return Corrupt(f"failed to compute content digest ({type(e).__name__}): {e}")
            if observed != core.content_digest:
                return Corrupt("content digest mismatch")
        return IndexReuse(canonical_store(store), locator, core)


# ---- alynxr ------------------------------------------------------------------------------------


def _fragment_name(part: str) -> str:
    return part.removesuffix(".parquet").replace("/", "__") + ".json"


def _safe_part(part: str) -> bool:
    p = PurePosixPath(part)
    return (
        bool(part)
        and not p.is_absolute()
        and ".." not in p.parts
        and part.endswith(".parquet")
        and "\0" not in part
    )


def _alynxr_core(frag: dict[str, Any]) -> CoreRecord | None:
    if frag.get("schema") != ALYNXR_FRAGMENT:
        return None
    part, inp, logical = frag.get("part"), frag.get("inputs_sha256"), frag.get("logical_sha256")
    if not all(isinstance(v, str) and v for v in (part, inp, logical)):
        return None
    run_id = frag.get("run_id") if isinstance(frag.get("run_id"), str) else ""
    fields = {
        "part": part,
        "inputs_sha256": inp,
        "logical_sha256": logical,
        "file_sha256": frag.get("file_sha256"),
        "run_id": run_id,
    }
    return CoreRecord(
        str(part), str(inp), str(logical), _digest(ALYNXR_FRAGMENT_ADAPTER, fields), str(run_id)
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class AlynxrFragmentAdapter:
    """alynxr fold-stats parts, through their fragments; the locator is the part path."""

    name = ALYNXR_FRAGMENT_ADAPTER

    def find(self, store: Path) -> Iterator[tuple[str, CoreRecord]]:
        parts_dir = Path(store) / "parts"
        try:
            names = sorted(os.listdir(parts_dir))
        except FileNotFoundError:
            return
        for name in names:
            if not name.endswith(".json") or name.startswith("."):
                continue
            try:
                frag = _read_json(parts_dir / name)
            except FileNotFoundError:
                continue
            except (OSError, ValueError) as e:
                raise UnitIndexError(f"unreadable alynxr fragment {name}: {e}") from e
            if not isinstance(frag, dict):
                raise UnitIndexError(f"alynxr fragment {name} is not a JSON object")
            part = frag.get("part")
            if not isinstance(part, str) or not _safe_part(part) or _fragment_name(part) != name:
                raise UnitIndexError(f"alynxr fragment {name} does not belong to part {part!r}")
            core = _alynxr_core(frag)
            if core is not None:
                yield part, core

    def classify(self, store: Path, locator: str, input_digest: str, *, verify: bool) -> Any:
        if not _safe_part(locator):
            return Corrupt(f"bad part path {locator!r}")
        store = Path(store)
        try:
            frag = _read_json(store / "parts" / _fragment_name(locator))
        except FileNotFoundError:
            return Missing()
        except (OSError, ValueError) as e:
            return Corrupt(f"unreadable fragment: {type(e).__name__}: {e}")
        if not isinstance(frag, dict) or frag.get("part") != locator:
            return Corrupt(f"fragment does not name part {locator!r}")
        if frag.get("schema") != ALYNXR_FRAGMENT:
            return SchemaMismatch(
                f"fragment schema {frag.get('schema')!r}, expected {ALYNXR_FRAGMENT!r}"
            )
        core = _alynxr_core(frag)
        if core is None:
            return Corrupt("fragment lacks part, inputs_sha256 or logical_sha256")
        if core.input_digest != input_digest:
            return IndexStale(canonical_store(store), locator, core)
        part = store / locator
        if not _inside(store, locator):
            return Corrupt(f"part {locator!r} resolves outside the store")
        try:
            size = part.stat().st_size
        except FileNotFoundError:
            return Corrupt(f"part file is missing: {locator}")  # as aminx: checked without verify
        if verify:
            if size != frag.get("file_bytes") or _sha256_file(part) != frag.get("file_sha256"):
                return Corrupt("part file does not match its fragment's size and sha256")
        return IndexReuse(canonical_store(store), locator, core)


ADAPTERS = {
    AMINX_DONE_MARKER: AminxDoneMarkerAdapter(),
    ALYNXR_FRAGMENT_ADAPTER: AlynxrFragmentAdapter(),
}
