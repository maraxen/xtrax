"""Tests for xtrax.run.zarr_commit durable atomic-commit primitives."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import zarr

from xtrax.run import zarr_commit as zc
from xtrax.run.zarr_integrity import zarr_content_digest


def _simple_staged_group(staged_dir: Path) -> None:
    """Create a simple staged group with two arrays."""
    zc.write_staged_group(
        staged_dir,
        {
            "data": np.arange(10, dtype=np.float32),
            "labels": np.array([0, 1, 1, 0, 1], dtype=np.int32),
        },
        attrs={"source": "test"},
    )


class TestKeyPath:
    """Tests for key_path helper."""

    def test_simple_key(self) -> None:
        assert zc.key_path(("a", "b", "c")) == "a/b/c"

    def test_single_element(self) -> None:
        assert zc.key_path(("x",)) == "x"

    def test_numeric_parts(self) -> None:
        assert zc.key_path(("0", "1", "2")) == "0/1/2"

    def test_rejects_empty_key(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            zc.key_path(())

    def test_rejects_empty_part(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            zc.key_path(("a", "", "b"))

    def test_rejects_dot(self) -> None:
        with pytest.raises(ValueError, match="'.', '..', or contains"):
            zc.key_path(("a", ".", "b"))

    def test_rejects_dotdot(self) -> None:
        with pytest.raises(ValueError, match="'.', '..', or contains"):
            zc.key_path(("a", "..", "b"))

    def test_rejects_slash_in_part(self) -> None:
        with pytest.raises(ValueError, match="contains"):
            zc.key_path(("a/b", "c"))


class TestStagingRoot:
    """Tests for staging_root helper."""

    def test_staging_root_path(self, tmp_path: Path) -> None:
        store = tmp_path / "store.zarr"
        expected = tmp_path / "store.zarr.staging"
        assert zc.staging_root(store) == expected


class TestWriteStagedGroup:
    """Tests for write_staged_group."""

    def test_creates_zarr_group_with_arrays(self, tmp_path: Path) -> None:
        staged_dir = tmp_path / "staged"
        zc.write_staged_group(
            staged_dir,
            {"arr1": np.array([1, 2, 3]), "arr2": np.ones((3, 2))},
            attrs={"name": "test"},
        )
        assert (staged_dir / "zarr.json").exists()
        group = zarr.open_group(str(staged_dir), mode="r")
        assert np.array_equal(group["arr1"][:], [1, 2, 3])
        assert np.array_equal(group["arr2"][:], np.ones((3, 2)))
        assert group.attrs.get("name") == "test"

    def test_handles_zero_d_array(self, tmp_path: Path) -> None:
        staged_dir = tmp_path / "staged"
        zc.write_staged_group(staged_dir, {"scalar": np.array(42.0)})
        group = zarr.open_group(str(staged_dir), mode="r")
        assert group["scalar"][()] == 42.0

    def test_handles_zero_size_array(self, tmp_path: Path) -> None:
        staged_dir = tmp_path / "staged"
        zc.write_staged_group(staged_dir, {"empty": np.zeros((0, 3))})
        group = zarr.open_group(str(staged_dir), mode="r")
        assert group["empty"].shape == (0, 3)


class TestReadRecord:
    """Tests for read_record."""

    def test_reads_valid_record(self, tmp_path: Path) -> None:
        staged_dir = tmp_path / "staged"
        _simple_staged_group(staged_dir)
        group = zarr.open_group(str(staged_dir), mode="r+")
        record_dict = {
            "input_digest": "abc123",
            "input_payload": {"x": 1},
            "meta": {},
            "content_digest": "def456",
            "run_id": "run1",
            "env": {},
            "committed_at": "2024-01-01T00:00:00+00:00",
        }
        record_dict["record_digest"] = zc._compute_record_digest(record_dict)
        group.attrs[zc.COMMIT_ATTR] = record_dict
        record = zc.read_record(staged_dir)
        assert record is not None
        assert record.input_digest == "abc123"

    def test_returns_none_for_missing_zarr_json(self, tmp_path: Path) -> None:
        assert zc.read_record(tmp_path / "nonexistent") is None

    def test_returns_none_for_missing_commit_attr(self, tmp_path: Path) -> None:
        staged_dir = tmp_path / "staged"
        _simple_staged_group(staged_dir)
        record = zc.read_record(staged_dir)
        assert record is None

    def test_returns_none_for_tampered_record_digest(self, tmp_path: Path) -> None:
        staged_dir = tmp_path / "staged"
        _simple_staged_group(staged_dir)
        group = zarr.open_group(str(staged_dir), mode="r+")
        record_dict = {
            "input_digest": "abc123",
            "input_payload": {},
            "meta": {},
            "content_digest": "def456",
            "run_id": "run1",
            "env": {},
            "committed_at": "2024-01-01T00:00:00+00:00",
            "record_digest": "wrong_digest",
        }
        group.attrs[zc.COMMIT_ATTR] = record_dict
        record = zc.read_record(staged_dir)
        assert record is None


_COMMIT_CHILD = """
import sys
from pathlib import Path

import numpy as np

from xtrax.run import zarr_commit as zc

store = Path(sys.argv[1])
zc.create_store(
    store,
    identity_payload={"type": "test"},
    creator_run_id="creator",
    prefixes=(("chunks",),),
    writer_id="child",
)
staged_dir = zc.staging_root(store) / "child" / "k0"
zc.write_staged_group(staged_dir, {"arr": np.arange(5)})
result = zc.commit_key(
    store,
    ("chunks", "k0"),
    staged_dir,
    input_digest="digest-k0",
    run_id="run_commit",
    env={"test": "env"},
)
print(type(result).__name__)
"""

_CREATE_CHILD = """
import sys
from pathlib import Path

from xtrax.run import zarr_commit as zc

zc.create_store(
    Path(sys.argv[1]),
    identity_payload={"type": "test"},
    creator_run_id="creator",
    prefixes=(("prefix1",),),
    writer_id="child",
)
"""


def _run_script(
    script: Path, body: str, *args: str, fault: str | None = None
) -> subprocess.CompletedProcess[str]:
    """Run ``body`` as a child python process, optionally with a fault step armed."""
    script.write_text(body)
    env = {k: v for k, v in os.environ.items() if k != zc.FAULT_ENV}
    if fault is not None:
        env[zc.FAULT_ENV] = fault
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )


class TestCommitKeyAtomicity:
    """Fault injection: a hard exit at every commit step leaves a valid store."""

    @pytest.mark.parametrize(
        ("step", "committed"),
        [
            ("staged_written", False),
            ("digested", False),
            ("record_written", False),
            ("fsynced", False),
            ("renamed", True),
        ],
    )
    def test_crash_at_step(self, tmp_path: Path, step: str, committed: bool) -> None:
        store = tmp_path / "store"
        script = tmp_path / "child.py"
        crashed = _run_script(script, _COMMIT_CHILD, str(store), fault=step)
        assert crashed.returncode == 137, crashed.stderr

        # The store is valid and the key is either fully there or fully absent.
        zc.open_store(store, identity_payload={"type": "test"}, prefixes=(("chunks",),))
        result = zc.lookup(store, ("chunks", "k0"), "digest-k0", verify=True)
        if committed:
            assert isinstance(result, zc.Reuse), result
        else:
            assert isinstance(result, zc.Missing), result
            # The partial staged group is left behind for gc, never in the store.
            assert (zc.staging_root(store) / "child" / "k0").exists()
            removed = zc.gc_staging(store, timedelta(seconds=0))
            assert zc.staging_root(store) / "child" in removed

        # Resume: an unfaulted rerun converges to exactly one committed key.
        rerun = _run_script(script, _COMMIT_CHILD, str(store))
        assert rerun.returncode == 0, rerun.stderr
        assert rerun.stdout.strip() == ("Duplicate" if committed else "Committed")
        final = zc.lookup(store, ("chunks", "k0"), "digest-k0", verify=True)
        assert isinstance(final, zc.Reuse)

    @pytest.mark.parametrize("step", ["root_staged", "root_renamed"])
    def test_create_store_crash(self, tmp_path: Path, step: str) -> None:
        store = tmp_path / "store"
        script = tmp_path / "create.py"
        crashed = _run_script(script, _CREATE_CHILD, str(store), fault=step)
        assert crashed.returncode == 137, crashed.stderr

        if step == "root_staged":
            # Crashed before the rename: no store at all, and a fresh create wins.
            assert not store.exists()
            assert zc.create_store(
                store,
                identity_payload={"type": "test"},
                creator_run_id="creator",
                prefixes=(("prefix1",),),
                writer_id="second",
            )
        # Either way the store is now complete and valid.
        zc.open_store(store, identity_payload={"type": "test"}, prefixes=(("prefix1",),))


class TestLookupClassification:
    """Tests for lookup() classification logic."""

    def test_lookup_missing(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        zc.create_store(
            store,
            identity_payload={"type": "test"},
            creator_run_id="creator",
            prefixes=(("chunks",),),
            writer_id="writer",
        )
        result = zc.lookup(store, ("chunks", "missing"), "digest1")
        assert isinstance(result, zc.Missing)

    def test_lookup_reuse(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        zc.create_store(
            store,
            identity_payload={"type": "test"},
            creator_run_id="creator",
            prefixes=(("chunks",),),
            writer_id="writer",
        )
        staged_dir = zc.staging_root(store) / "writer" / "test_key"
        zc.write_staged_group(staged_dir, {"arr": np.arange(5)})
        zc.commit_key(
            store,
            ("chunks", "k0"),
            staged_dir,
            input_digest="digest1",
            run_id="run1",
            env={},
        )
        result = zc.lookup(store, ("chunks", "k0"), "digest1")
        assert isinstance(result, zc.Reuse)
        assert result.record.input_digest == "digest1"

    def test_lookup_stale(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        zc.create_store(
            store,
            identity_payload={"type": "test"},
            creator_run_id="creator",
            prefixes=(("chunks",),),
            writer_id="writer",
        )
        staged_dir = zc.staging_root(store) / "writer" / "test_key"
        zc.write_staged_group(staged_dir, {"arr": np.arange(5)})
        zc.commit_key(
            store,
            ("chunks", "k0"),
            staged_dir,
            input_digest="digest1",
            run_id="run1",
            env={},
        )
        result = zc.lookup(store, ("chunks", "k0"), "different_digest")
        assert isinstance(result, zc.Stale)
        assert result.record.input_digest == "digest1"

    def test_lookup_corrupt_not_zarr(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        zc.create_store(
            store,
            identity_payload={"type": "test"},
            creator_run_id="creator",
            prefixes=(("chunks",),),
            writer_id="writer",
        )
        # Create a plain directory at the key location (not a zarr group)
        key_dir = store / "chunks" / "corrupt"
        key_dir.mkdir(parents=True)
        # Write a file so it's not empty, but not a zarr group
        (key_dir / "somefile.txt").write_text("not a zarr group")
        result = zc.lookup(store, ("chunks", "corrupt"), "digest1")
        assert isinstance(result, zc.Corrupt)
        assert "not a zarr" in result.reason

    def test_lookup_corrupt_uncommitted(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        zc.create_store(
            store,
            identity_payload={"type": "test"},
            creator_run_id="creator",
            prefixes=(("chunks",),),
            writer_id="writer",
        )
        # Create a plain zarr group without commit record
        key_dir = store / "chunks" / "uncommitted"
        zc.write_staged_group(key_dir, {"arr": np.arange(5)})
        result = zc.lookup(store, ("chunks", "uncommitted"), "digest1")
        assert isinstance(result, zc.Corrupt)
        assert "uncommitted" in result.reason

    def test_lookup_corrupt_content_digest_mismatch(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        zc.create_store(
            store,
            identity_payload={"type": "test"},
            creator_run_id="creator",
            prefixes=(("chunks",),),
            writer_id="writer",
        )
        staged_dir = zc.staging_root(store) / "writer" / "test_key"
        zc.write_staged_group(staged_dir, {"arr": np.arange(5, dtype=np.float32)})
        zc.commit_key(
            store,
            ("chunks", "k0"),
            staged_dir,
            input_digest="digest1",
            run_id="run1",
            env={},
        )
        # Modify the committed data on disk directly by rewriting a chunk file
        key_dir = store / "chunks" / "k0"
        # Find and modify the array chunk file
        chunk_files = list(key_dir.glob("arr/*"))
        if chunk_files:
            # Modify the first chunk file to corrupt it
            chunk_file = chunk_files[0]
            with chunk_file.open("r+b") as f:
                data = f.read()
                # Flip a byte in the middle
                mid = len(data) // 2
                modified = data[:mid] + bytes([data[mid] ^ 0xFF]) + data[mid + 1 :]
                f.seek(0)
                f.write(modified)

        result = zc.lookup(store, ("chunks", "k0"), "digest1", verify=True)
        assert isinstance(result, zc.Corrupt)
        assert "content digest" in result.reason

    def test_lookup_with_verify_false_skips_content_check(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        zc.create_store(
            store,
            identity_payload={"type": "test"},
            creator_run_id="creator",
            prefixes=(("chunks",),),
            writer_id="writer",
        )
        staged_dir = zc.staging_root(store) / "writer" / "test_key"
        zc.write_staged_group(staged_dir, {"arr": np.arange(5)})
        zc.commit_key(
            store,
            ("chunks", "k0"),
            staged_dir,
            input_digest="digest1",
            run_id="run1",
            env={},
        )
        # Modify data but verify=False should not detect it
        key_dir = store / "chunks" / "k0"
        group = zarr.open_group(str(key_dir), mode="r+")
        group["arr"][:] = np.arange(10)

        result = zc.lookup(store, ("chunks", "k0"), "digest1", verify=False)
        assert isinstance(result, zc.Reuse)


class TestCommittedKeys:
    """Tests for committed_keys."""

    def test_lists_all_committed_keys(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        zc.create_store(
            store,
            identity_payload={"type": "test"},
            creator_run_id="creator",
            prefixes=(("chunks",),),
            writer_id="writer",
        )
        # Commit 3 keys
        for i in range(3):
            staged_dir = zc.staging_root(store) / "writer" / f"key{i}"
            zc.write_staged_group(staged_dir, {"arr": np.arange(5)})
            zc.commit_key(
                store,
                ("chunks", f"k{i}"),
                staged_dir,
                input_digest=f"digest{i}",
                run_id="run1",
                env={},
            )
        keys = zc.committed_keys(store)
        assert len(keys) == 3
        assert ("chunks", "k0") in keys
        assert ("chunks", "k1") in keys
        assert ("chunks", "k2") in keys

    def test_committed_keys_with_prefix(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        zc.create_store(
            store,
            identity_payload={"type": "test"},
            creator_run_id="creator",
            prefixes=(("chunks",), ("other",)),
            writer_id="writer",
        )
        # Commit keys under different prefixes
        for prefix in ["chunks", "other"]:
            for i in range(2):
                staged_dir = zc.staging_root(store) / "writer" / f"{prefix}_{i}"
                zc.write_staged_group(staged_dir, {"arr": np.arange(5)})
                zc.commit_key(
                    store,
                    (prefix, f"k{i}"),
                    staged_dir,
                    input_digest=f"{prefix}_digest{i}",
                    run_id="run1",
                    env={},
                )
        # All keys (relative to store root)
        all_keys = zc.committed_keys(store)
        assert len(all_keys) == 4
        assert ("chunks", "k0") in all_keys
        assert ("chunks", "k1") in all_keys
        assert ("other", "k0") in all_keys
        assert ("other", "k1") in all_keys

        # Keys under prefix "chunks" (still relative to store root)
        chunk_keys = zc.committed_keys(store, ("chunks",))
        assert len(chunk_keys) == 2
        assert ("chunks", "k0") in chunk_keys
        assert ("chunks", "k1") in chunk_keys


class TestGcStaging:
    """Tests for gc_staging."""

    def test_removes_old_staging_dirs(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        store.mkdir()
        staging = zc.staging_root(store)
        staging.mkdir()

        # Create old and new staging dirs
        old_dir = staging / "old"
        new_dir = staging / "new"
        old_dir.mkdir()
        new_dir.mkdir()

        # Make old_dir appear old (100 seconds in the past)
        old_mtime = time.time() - 100
        os.utime(old_dir, (old_mtime, old_mtime))

        # Remove dirs older than 50 seconds
        removed = zc.gc_staging(store, timedelta(seconds=50))
        # old_dir is 100s old, so now - old_mtime = 100 > 50 -> should be removed
        assert old_dir in removed
        # new_dir is recent, so now - new_mtime < 50 -> should not be removed
        assert new_dir not in removed
        # old_dir should no longer exist
        assert not old_dir.exists()

    def test_excludes_specified_dirs(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        store.mkdir()
        staging = zc.staging_root(store)
        staging.mkdir()

        old_dir = staging / "old"
        excluded_dir = staging / "excluded"
        old_dir.mkdir()
        excluded_dir.mkdir()

        old_mtime = time.time() - 100
        os.utime(old_dir, (old_mtime, old_mtime))
        os.utime(excluded_dir, (old_mtime, old_mtime))

        removed = zc.gc_staging(store, timedelta(seconds=50), exclude=["excluded"])
        assert old_dir not in [r for r in removed if r.exists()]
        assert excluded_dir.exists()  # excluded should still exist

    def test_returns_empty_for_missing_staging(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        store.mkdir()
        removed = zc.gc_staging(store, timedelta(seconds=1))
        assert removed == []


class TestCreateStore:
    """Tests for create_store."""

    def test_creates_store_with_prefixes(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        result = zc.create_store(
            store,
            identity_payload={"type": "test"},
            creator_run_id="creator",
            prefixes=(("prefix1",), ("prefix2",)),
            writer_id="writer",
        )
        assert result is True
        assert (store / "zarr.json").exists()

        # Verify prefixes exist
        root = zarr.open_group(str(store), mode="r")
        assert "prefix1" in root
        assert "prefix2" in root

    def test_returns_false_if_store_exists(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        # Create once
        zc.create_store(
            store,
            identity_payload={"type": "test"},
            creator_run_id="creator",
            prefixes=(("prefix1",),),
            writer_id="writer1",
        )
        # Try to create again with different writer_id
        result = zc.create_store(
            store,
            identity_payload={"type": "test"},
            creator_run_id="creator",
            prefixes=(("prefix1",),),
            writer_id="writer2",
        )
        assert result is False


class TestOpenStore:
    """Tests for open_store."""

    def test_opens_valid_store(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        zc.create_store(
            store,
            identity_payload={"type": "test", "version": 1},
            creator_run_id="creator",
            prefixes=(("prefix1",),),
            writer_id="writer",
        )
        record = zc.open_store(
            store,
            identity_payload={"type": "test", "version": 1},
            prefixes=(("prefix1",),),
        )
        assert "identity_payload" in record
        assert record["identity_payload"]["type"] == "test"

    def test_raises_not_durable_for_plain_zarr(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        store.mkdir()
        zarr.open_group(str(store), mode="w")
        with pytest.raises(zc.NotADurableStoreError):
            zc.open_store(
                store,
                identity_payload={"type": "test"},
                prefixes=(),
            )

    def test_raises_identity_mismatch(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        zc.create_store(
            store,
            identity_payload={"type": "test", "version": 1},
            creator_run_id="creator",
            prefixes=(),
            writer_id="writer",
        )
        with pytest.raises(zc.StoreIdentityMismatch):
            zc.open_store(
                store,
                identity_payload={"type": "test", "version": 2},
                prefixes=(),
            )

    def test_raises_unknown_prefix(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        zc.create_store(
            store,
            identity_payload={"type": "test"},
            creator_run_id="creator",
            prefixes=(("prefix1",),),
            writer_id="writer",
        )
        with pytest.raises(zc.UnknownPrefixError):
            zc.open_store(
                store,
                identity_payload={"type": "test"},
                prefixes=(("prefix1",), ("missing",)),
            )


class TestCommitKeyDuplicate:
    """Tests for duplicate detection in commit_key."""

    def test_returns_duplicate_for_same_input_digest(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        zc.create_store(
            store,
            identity_payload={"type": "test"},
            creator_run_id="creator",
            prefixes=(("chunks",),),
            writer_id="writer",
        )
        staged_dir1 = zc.staging_root(store) / "writer" / "key1"
        zc.write_staged_group(staged_dir1, {"arr": np.arange(5)})
        result1 = zc.commit_key(
            store,
            ("chunks", "k0"),
            staged_dir1,
            input_digest="digest1",
            run_id="run1",
            env={},
        )
        assert isinstance(result1, zc.Committed)

        # Commit again with same digest
        staged_dir2 = zc.staging_root(store) / "writer" / "key2"
        zc.write_staged_group(staged_dir2, {"arr": np.arange(5)})
        result2 = zc.commit_key(
            store,
            ("chunks", "k0"),
            staged_dir2,
            input_digest="digest1",
            run_id="run1",
            env={},
        )
        assert isinstance(result2, zc.Duplicate)
        assert result2.record.input_digest == "digest1"

    def test_raises_conflict_for_different_input_digest(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        zc.create_store(
            store,
            identity_payload={"type": "test"},
            creator_run_id="creator",
            prefixes=(("chunks",),),
            writer_id="writer",
        )
        staged_dir1 = zc.staging_root(store) / "writer" / "key1"
        zc.write_staged_group(staged_dir1, {"arr": np.arange(5)})
        zc.commit_key(
            store,
            ("chunks", "k0"),
            staged_dir1,
            input_digest="digest1",
            run_id="run1",
            env={},
        )

        # Try to commit different digest to same key
        staged_dir2 = zc.staging_root(store) / "writer" / "key2"
        zc.write_staged_group(staged_dir2, {"arr": np.arange(10)})
        with pytest.raises(zc.CommitConflictError) as excinfo:
            zc.commit_key(
                store,
                ("chunks", "k0"),
                staged_dir2,
                input_digest="digest2",
                run_id="run1",
                env={},
            )
        assert "different input_digest" in str(excinfo.value)


class TestUnknownPrefixError:
    """Tests for UnknownPrefixError in commit_key."""

    def test_commit_key_raises_unknown_prefix(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        zc.create_store(
            store,
            identity_payload={"type": "test"},
            creator_run_id="creator",
            prefixes=(("chunks",),),
            writer_id="writer",
        )
        staged_dir = zc.staging_root(store) / "writer" / "key1"
        zc.write_staged_group(staged_dir, {"arr": np.arange(5)})

        # Try to commit to non-existent prefix
        with pytest.raises(zc.UnknownPrefixError):
            zc.commit_key(
                store,
                ("unknown", "k0"),
                staged_dir,
                input_digest="digest1",
                run_id="run1",
                env={},
            )


class TestDigestLocality:
    """Tests for content digest locality."""

    def test_staged_digest_matches_committed_digest(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        zc.create_store(
            store,
            identity_payload={"type": "test"},
            creator_run_id="creator",
            prefixes=(("chunks",),),
            writer_id="writer",
        )
        staged_dir = zc.staging_root(store) / "writer" / "key1"
        zc.write_staged_group(staged_dir, {"arr": np.arange(5), "labels": np.array([1, 0])})

        staged_digest = zarr_content_digest(staged_dir)

        result = zc.commit_key(
            store,
            ("chunks", "k0"),
            staged_dir,
            input_digest="digest1",
            run_id="run1",
            env={},
        )
        assert result.record.content_digest == staged_digest

        # Verify digest matches at the committed location
        committed_digest = zarr_content_digest(store / "chunks" / "k0")
        assert committed_digest == staged_digest


class TestReuseOpen:
    """Tests for Reuse.open()."""

    def test_reuse_open_returns_readonly_group(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        zc.create_store(
            store,
            identity_payload={"type": "test"},
            creator_run_id="creator",
            prefixes=(("chunks",),),
            writer_id="writer",
        )
        staged_dir = zc.staging_root(store) / "writer" / "key1"
        zc.write_staged_group(staged_dir, {"arr": np.arange(5)})
        zc.commit_key(
            store,
            ("chunks", "k0"),
            staged_dir,
            input_digest="digest1",
            run_id="run1",
            env={},
        )

        result = zc.lookup(store, ("chunks", "k0"), "digest1")
        assert isinstance(result, zc.Reuse)

        group = result.open()
        assert np.array_equal(group["arr"][:], np.arange(5))


def _make_store(tmp_path: Path) -> Path:
    store = tmp_path / "store"
    assert zc.create_store(
        store,
        identity_payload={"type": "test"},
        creator_run_id="creator",
        prefixes=(("chunks",),),
        writer_id="writer",
    )
    return store


def _commit_k0(store: Path, staged_name: str = "key", digest: str = "digest1") -> zc.CommitRecord:
    staged_dir = zc.staging_root(store) / "writer" / staged_name
    zc.write_staged_group(staged_dir, {"arr": np.arange(5)})
    result = zc.commit_key(
        store, ("chunks", "k0"), staged_dir, input_digest=digest, run_id="run1", env={}
    )
    return result.record


class TestFsyncOrdering:
    """The durability barriers must bracket the rename (fsync-before-rename)."""

    def test_fsync_before_and_after_rename(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = _make_store(tmp_path)
        staged_dir = zc.staging_root(store) / "writer" / "key"
        zc.write_staged_group(staged_dir, {"arr": np.arange(5)})
        target = store / "chunks" / "k0"

        events: list[tuple[str, str]] = []
        real_tree, real_dir, real_rename = zc.fsync_tree, zc.fsync_directory, os.rename

        def rec_tree(path: Path) -> None:
            events.append(("fsync_tree", str(path)))
            real_tree(path)

        def rec_dir(path: Path) -> None:
            events.append(("fsync_directory", str(path)))
            real_dir(path)

        def rec_rename(src: str, dst: str, *a: object, **kw: object) -> None:
            events.append(("rename", f"{src}->{dst}"))
            real_rename(src, dst, *a, **kw)  # type: ignore[arg-type]

        monkeypatch.setattr("xtrax.run.zarr_commit.fsync_tree", rec_tree)
        monkeypatch.setattr("xtrax.run.zarr_commit.fsync_directory", rec_dir)
        monkeypatch.setattr("xtrax.run.zarr_commit.os.rename", rec_rename)

        result = zc.commit_key(
            store, ("chunks", "k0"), staged_dir, input_digest="d", run_id="r", env={}
        )
        assert isinstance(result, zc.Committed)

        rename_event = ("rename", f"{staged_dir}->{target}")
        assert events.count(rename_event) == 1, events
        r = events.index(rename_event)
        before, after = events[:r], events[r + 1 :]
        assert ("fsync_tree", str(staged_dir)) in before
        assert ("fsync_directory", str(staged_dir.parent)) in before
        assert ("fsync_directory", str(target.parent)) in after
        # The parent-of-staged barrier must come after the tree barrier.
        assert before.index(("fsync_tree", str(staged_dir))) < before.index(
            ("fsync_directory", str(staged_dir.parent))
        )


class TestLookupReasons:
    """lookup() distinguishes uncommitted groups from tampered commit records."""

    def test_tampered_record_is_corrupt_tampered(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        _commit_k0(store)
        group = zarr.open_group(str(store / "chunks" / "k0"), mode="r+")
        tampered = dict(group.attrs[zc.COMMIT_ATTR])  # type: ignore[arg-type]
        tampered["run_id"] = "evil"
        group.attrs[zc.COMMIT_ATTR] = tampered
        result = zc.lookup(store, ("chunks", "k0"), "digest1")
        assert isinstance(result, zc.Corrupt)
        assert "tampered" in result.reason
        assert "uncommitted" not in result.reason
        assert zc.read_record(store / "chunks" / "k0") is None

    def test_malformed_record_is_corrupt_tampered(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        _commit_k0(store)
        group = zarr.open_group(str(store / "chunks" / "k0"), mode="r+")
        group.attrs[zc.COMMIT_ATTR] = {"input_digest": "digest1"}
        result = zc.lookup(store, ("chunks", "k0"), "digest1")
        assert isinstance(result, zc.Corrupt)
        assert "tampered" in result.reason

    def test_uncommitted_group_reason(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        zc.write_staged_group(store / "chunks" / "k0", {"arr": np.arange(3)})
        result = zc.lookup(store, ("chunks", "k0"), "digest1")
        assert isinstance(result, zc.Corrupt)
        assert "uncommitted" in result.reason
        assert "tampered" not in result.reason


class TestUnknownPrefixCleanup:
    """An unknown prefix fails fast and leaves no orphaned staging behind."""

    def test_no_staged_dir_left_behind(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        staged_dir = zc.staging_root(store) / "writer" / "key"
        zc.write_staged_group(staged_dir, {"arr": np.arange(5)})
        with pytest.raises(zc.UnknownPrefixError):
            zc.commit_key(
                store, ("unknown", "k0"), staged_dir, input_digest="d", run_id="r", env={}
            )
        assert not staged_dir.exists()
        assert not (store / "unknown").exists()

    def test_prefix_checked_before_digest_work(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = _make_store(tmp_path)
        staged_dir = zc.staging_root(store) / "writer" / "key"
        zc.write_staged_group(staged_dir, {"arr": np.arange(5)})

        def boom(path: Path) -> str:
            raise AssertionError("digest computed before prefix check")

        monkeypatch.setattr("xtrax.run.zarr_commit.zarr_content_digest", boom)
        with pytest.raises(zc.UnknownPrefixError):
            zc.commit_key(
                store, ("unknown", "k0"), staged_dir, input_digest="d", run_id="r", env={}
            )


class TestCreateStoreStagingIsolation:
    """create_store must only ever clean up its own writer's staging."""

    def test_winner_leaves_foreign_staging_intact(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        foreign = zc.staging_root(store) / "other_writer" / "k9"
        zc.write_staged_group(foreign, {"arr": np.arange(3)})
        assert zc.create_store(
            store,
            identity_payload={"type": "test"},
            creator_run_id="creator",
            prefixes=(("chunks",),),
            writer_id="me",
        )
        assert (foreign / "zarr.json").exists()
        assert not (zc.staging_root(store) / "me").exists()

    def test_loser_leaves_foreign_staging_intact(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        foreign = zc.staging_root(store) / "other_writer" / "k9"
        zc.write_staged_group(foreign, {"arr": np.arange(3)})
        assert not zc.create_store(
            store,
            identity_payload={"type": "test"},
            creator_run_id="creator",
            prefixes=(("chunks",),),
            writer_id="me",
        )
        assert (foreign / "zarr.json").exists()
        assert not (zc.staging_root(store) / "me" / "__root__").exists()
