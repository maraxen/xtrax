"""Pack a completed loose-file unit into a few files, keeping random access to each member.

Spec 261030 §3.2, §3.3. A unit is a directory whose record, ``unit.json``, is written last and
carries the same fields as a ``zarr_commit`` ``CommitRecord``. Its ``content_digest`` is computed
over the uncompressed members, so a unit has the same identity whether it is loose or packed.
Packing adds a ``storage`` field outside ``record_digest``, so an index row stays valid.

The logic lives in :mod:`xtrax.pack._node`, which is stdlib-only and Python 3.9 compatible so a
cluster script can embed it (:func:`node_source`) where xtrax is not installed. The format is
alynxr's ``alynxr.unit_pack/1`` (one zstd frame per USTAR member, plus a frame index), and a pack
written by either implementation reads through the other.

Example:
    >>> from xtrax import pack  # doctest: +SKIP
    >>> pack.write_unit_record(unit, input_digest=digest, run_id="run-1")  # doctest: +SKIP
    >>> pack.pack_unit(unit)  # doctest: +SKIP
    >>> pack.read_unit_member(unit, "fold/scores.json")  # doctest: +SKIP
"""

from pathlib import Path

from xtrax.pack._node import (
    PACK_FILE,
    PACK_FORMAT,
    PACK_INDEX,
    PACK_TELEMETRY_SCHEMA,
    RECORD_SCHEMA,
    UNIT_RECORD,
    CodecUnavailable,
    LockUnavailable,
    UnitPackRefused,
    collect_loose_members,
    compute_record_digest,
    digest_of_members,
    find_packed_unit,
    is_packed,
    is_plain_relpath,
    list_unit_members,
    local_only_lock_reason,
    mount_of,
    pack_sweep,
    pack_unit,
    read_unit_member,
    record_problem,
    register_telemetry_hook,
    storage_files,
    unit_content_digest,
    write_unit_record,
)

__all__ = [
    "PACK_FILE",
    "PACK_FORMAT",
    "PACK_INDEX",
    "PACK_TELEMETRY_SCHEMA",
    "RECORD_SCHEMA",
    "UNIT_RECORD",
    "CodecUnavailable",
    "LockUnavailable",
    "UnitPackRefused",
    "collect_loose_members",
    "compute_record_digest",
    "digest_of_members",
    "find_packed_unit",
    "is_packed",
    "is_plain_relpath",
    "list_unit_members",
    "local_only_lock_reason",
    "mount_of",
    "node_source",
    "pack_sweep",
    "pack_unit",
    "read_unit_member",
    "record_problem",
    "register_telemetry_hook",
    "storage_files",
    "unit_content_digest",
    "write_unit_record",
]


def node_source() -> str:
    """Source text of the stdlib-only core, for embedding in a rendered cluster script."""
    return Path(__file__).with_name("_node.py").read_text(encoding="utf-8")
