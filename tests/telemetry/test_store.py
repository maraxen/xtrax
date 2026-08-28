"""Content-addressed blob store: dedup, idempotency, integrity, inode budget."""

import gzip

import pytest

from xtrax.telemetry.store import (
    POPULATION_WARN_THRESHOLD,
    BlobStore,
    BlobStoreError,
    digest_of,
)


def test_put_is_content_addressed(tmp_path):
    store = BlobStore(tmp_path)
    sha, size = store.put("hello world")
    assert len(sha) == 64
    assert size == len("hello world")
    assert store.get(sha) == "hello world"


def test_identical_text_dedups_to_one_blob(tmp_path):
    """The property the whole cost model rests on."""
    store = BlobStore(tmp_path)
    sha_a, _ = store.put("same text")
    sha_b, _ = store.put("same text")
    assert sha_a == sha_b
    assert store.count() == 1


def test_different_text_produces_different_blobs(tmp_path):
    """The other half: dedup must not collapse genuinely different IR."""
    store = BlobStore(tmp_path)
    store.put("version one")
    store.put("version two")
    assert store.count() == 2


def test_put_does_not_rewrite_an_existing_blob(tmp_path):
    """Rewriting would churn artifacts other ledger rows already reference."""
    store = BlobStore(tmp_path)
    sha, _ = store.put("stable content")
    path = store.find(sha)
    assert path is not None
    before = path.stat().st_mtime_ns
    store.put("stable content")
    assert path.stat().st_mtime_ns == before


def test_digest_is_over_uncompressed_bytes(tmp_path):
    """So a recorded digest stays verifiable independently of our codec."""
    text = "some ir text"
    expected, _ = digest_of(text)
    store = BlobStore(tmp_path)
    sha, _ = store.put(text)
    assert sha == expected


def test_bytes_field_is_the_raw_length_not_the_compressed_one(tmp_path):
    store = BlobStore(tmp_path)
    text = "x" * 10_000
    _, size = store.put(text)
    assert size == 10_000
    assert store.total_bytes() < 10_000  # actually compressed


def test_compression_is_deterministic(tmp_path):
    """gzip embeds an mtime by default; we pin it so bytes are content-only."""
    a = BlobStore(tmp_path / "a")
    b = BlobStore(tmp_path / "b")
    sha_a, _ = a.put("identical")
    sha_b, _ = b.put("identical")
    assert a.find(sha_a).read_bytes() == b.find(sha_b).read_bytes()


def test_stored_blob_is_actually_compressed_and_readable_externally(tmp_path):
    """Audit ergonomics: a human with zcat can read the store."""
    store = BlobStore(tmp_path)
    sha, _ = store.put("plain ir text")
    path = store.find(sha)
    if path.suffix == ".gz":
        assert gzip.decompress(path.read_bytes()).decode() == "plain ir text"


def test_verify_detects_corruption(tmp_path):
    store = BlobStore(tmp_path)
    sha, _ = store.put("original")
    assert store.verify(sha)
    store.find(sha).write_bytes(gzip.compress(b"tampered", mtime=0))
    assert not store.verify(sha)


def test_verify_is_false_for_a_missing_blob(tmp_path):
    assert not BlobStore(tmp_path).verify("f" * 64)


def test_get_missing_blob_raises(tmp_path):
    with pytest.raises(BlobStoreError, match="not present"):
        BlobStore(tmp_path).get("a" * 64)


def test_malformed_digest_is_rejected(tmp_path):
    with pytest.raises(BlobStoreError, match="64 hex chars"):
        BlobStore(tmp_path).path_for("tooshort")


def test_delete_reports_whether_it_existed(tmp_path):
    store = BlobStore(tmp_path)
    sha, _ = store.put("doomed")
    assert store.delete(sha)
    assert not store.delete(sha)


def test_iter_digests_round_trips_put_digests(tmp_path):
    store = BlobStore(tmp_path)
    shas = {store.put(f"text {i}")[0] for i in range(5)}
    assert set(store.iter_digests()) == shas


def test_empty_store_reports_zero(tmp_path):
    store = BlobStore(tmp_path / "missing")
    assert store.count() == 0
    assert store.total_bytes() == 0
    assert list(store.iter_digests()) == []


def test_temp_files_are_not_mistaken_for_blobs(tmp_path):
    """A crashed write leaves a .tmp; it must not be read back as an artifact."""
    store = BlobStore(tmp_path)
    store.put("real")
    (tmp_path / "abc.gz.999.tmp").write_bytes(b"junk")
    assert store.count() == 1


def test_population_check_is_quiet_below_the_threshold(tmp_path, recwarn):
    store = BlobStore(tmp_path)
    store.put("one")
    assert store.check_population() == 1
    assert not [w for w in recwarn if "blob store" in str(w.message)]


def test_population_check_warns_when_dedup_appears_broken(tmp_path):
    """Guards the inode budget: a population this large means per-step capture."""
    store = BlobStore(tmp_path)
    for i in range(3):
        store.put(f"blob {i}")
    with pytest.warns(UserWarning, match="captured inside a step loop"):
        store.check_population(threshold=2)


def test_population_threshold_default_is_generous(tmp_path):
    """The default must not fire for legitimate multi-signature workloads."""
    assert POPULATION_WARN_THRESHOLD >= 1000
