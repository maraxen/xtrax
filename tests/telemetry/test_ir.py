"""IR capture: determinism, dedup, degradation, and the capture-mode policy.

Deliberately small models. Per ~/.claude/rules/local-compute-limits.md these run
locally, so every function here traces in milliseconds; the point is the capture
contract, not the workload.
"""

import jax
import jax.numpy as jnp
import pytest

from xtrax.telemetry.ir import (
    CAPTURE_FULL,
    CAPTURE_FULL_OPTIMIZED,
    CAPTURE_HASH,
    CAPTURE_NONE,
    IR_KIND_JAXPR,
    IR_KIND_OPTIMIZED_HLO,
    IR_KIND_STABLEHLO,
    capture_ir,
    degraded_reason,
    resolve_capture_mode,
)
from xtrax.telemetry.record import IR_FULL, IR_HASH_ONLY, IR_SKIPPED
from xtrax.telemetry.store import BlobStore


def _fn(x, w):
    return jnp.tanh(x @ w).sum()


_X = jnp.ones((4, 8))
_W = jnp.ones((8, 2))


# --- the determinism invariant ---------------------------------------------


def test_identical_functions_produce_identical_digests(tmp_path):
    """The CAS invariant. Per BATHOS.md, verify the measurement pipeline itself:
    a store that always-hits or always-misses would invalidate every audit."""
    store = BlobStore(tmp_path)
    first = capture_ir(_fn, _X, _W, store=store)
    second = capture_ir(_fn, _X, _W, store=store)
    assert [r.sha256 for r in first] == [r.sha256 for r in second]


def test_a_changed_function_produces_different_digests(tmp_path):
    """The other half: the digest must be sensitive to a real change."""
    store = BlobStore(tmp_path)

    def other(x, w):
        return jnp.sin(x @ w).sum()

    a = {r.kind: r.sha256 for r in capture_ir(_fn, _X, _W, store=store)}
    b = {r.kind: r.sha256 for r in capture_ir(other, _X, _W, store=store)}
    assert a[IR_KIND_JAXPR] != b[IR_KIND_JAXPR]
    assert a[IR_KIND_STABLEHLO] != b[IR_KIND_STABLEHLO]


def test_recapturing_the_same_signature_adds_no_blobs(tmp_path):
    """One blob per shape signature -- the basis of the inode budget."""
    store = BlobStore(tmp_path)
    capture_ir(_fn, _X, _W, store=store)
    after_first = store.count()
    for _ in range(5):
        capture_ir(_fn, _X, _W, store=store)
    assert store.count() == after_first


def test_a_different_shape_signature_produces_new_blobs(tmp_path):
    store = BlobStore(tmp_path)
    capture_ir(_fn, _X, _W, store=store)
    before = store.count()
    capture_ir(_fn, jnp.ones((16, 8)), _W, store=store)
    assert store.count() > before


# --- what gets captured -----------------------------------------------------


def test_default_capture_is_jaxpr_and_stablehlo(tmp_path):
    refs = capture_ir(_fn, _X, _W, store=BlobStore(tmp_path))
    assert [r.kind for r in refs] == [IR_KIND_JAXPR, IR_KIND_STABLEHLO]
    assert all(r.mode == IR_FULL for r in refs)


def test_captured_text_is_retrievable_and_looks_like_ir(tmp_path):
    """The artifact must actually be usable at audit time, not just hashed."""
    store = BlobStore(tmp_path)
    refs = capture_ir(_fn, _X, _W, store=store)
    by_kind = {r.kind: r for r in refs}
    jaxpr_text = store.get(by_kind[IR_KIND_JAXPR].sha256)
    hlo_text = store.get(by_kind[IR_KIND_STABLEHLO].sha256)
    assert "lambda" in jaxpr_text or "tanh" in jaxpr_text
    assert "module" in hlo_text or "stablehlo" in hlo_text


def test_bytes_recorded_match_the_stored_text(tmp_path):
    store = BlobStore(tmp_path)
    for ref in capture_ir(_fn, _X, _W, store=store):
        assert ref.bytes == len(store.get(ref.sha256).encode("utf-8"))


def test_shape_dtype_struct_traces_like_a_concrete_array(tmp_path):
    """Abstract inputs must be accepted; only shape/dtype affect the IR."""
    store = BlobStore(tmp_path)
    concrete = capture_ir(_fn, _X, _W, store=store)
    abstract = capture_ir(
        _fn,
        jax.ShapeDtypeStruct(_X.shape, _X.dtype),
        jax.ShapeDtypeStruct(_W.shape, _W.dtype),
        store=store,
    )
    assert [r.sha256 for r in concrete] == [r.sha256 for r in abstract]


def test_optimized_hlo_is_opt_in(tmp_path):
    store = BlobStore(tmp_path)
    default_kinds = {r.kind for r in capture_ir(_fn, _X, _W, store=store)}
    assert IR_KIND_OPTIMIZED_HLO not in default_kinds

    opted = capture_ir(
        _fn, _X, _W, store=store, mode=resolve_capture_mode(CAPTURE_FULL_OPTIMIZED)
    )
    assert IR_KIND_OPTIMIZED_HLO in {r.kind for r in opted}


# --- degradation ------------------------------------------------------------


def test_an_unrenderable_function_is_skipped_with_a_reason_not_raised(tmp_path):
    """The observer must never kill the observed."""

    def explodes(x, w):
        raise RuntimeError("cannot trace this")

    refs = capture_ir(explodes, _X, _W, store=BlobStore(tmp_path))
    assert refs
    assert all(r.mode == IR_SKIPPED for r in refs)
    assert all("cannot trace this" in (r.reason or "") for r in refs)


def test_a_skipped_artifact_yields_a_degraded_reason(tmp_path):
    def explodes(x, w):
        raise RuntimeError("nope")

    refs = capture_ir(explodes, _X, _W, store=BlobStore(tmp_path))
    reason = degraded_reason(refs)
    assert reason is not None
    assert IR_KIND_JAXPR in reason


def test_full_capture_yields_no_degraded_reason(tmp_path):
    refs = capture_ir(_fn, _X, _W, store=BlobStore(tmp_path))
    assert degraded_reason(refs) is None


def test_oversize_ir_degrades_to_hash_only_with_the_digest_intact(tmp_path):
    """Exceeding the cap must still leave a verifiable fingerprint."""
    store = BlobStore(tmp_path)
    refs = capture_ir(_fn, _X, _W, store=store, max_bytes=1)
    assert all(r.mode == IR_HASH_ONLY for r in refs)
    assert all(len(r.sha256) == 64 for r in refs)
    assert all("exceeds" in (r.reason or "") for r in refs)
    assert store.count() == 0


def test_hash_only_digests_match_full_capture_digests(tmp_path):
    """Same digest function on both paths, so the two remain comparable."""
    full = capture_ir(_fn, _X, _W, store=BlobStore(tmp_path / "a"))
    hashed = capture_ir(_fn, _X, _W, store=BlobStore(tmp_path / "b"), max_bytes=1)
    assert [r.sha256 for r in full] == [r.sha256 for r in hashed]


def test_capture_without_a_store_records_fingerprints_only(tmp_path):
    refs = capture_ir(_fn, _X, _W, store=None)
    assert all(r.mode == IR_HASH_ONLY for r in refs)


# --- capture-mode policy ----------------------------------------------------


def test_capture_none_disables_capture_entirely(tmp_path):
    refs = capture_ir(
        _fn, _X, _W, store=BlobStore(tmp_path), mode=resolve_capture_mode(CAPTURE_NONE)
    )
    assert refs == ()


def test_capture_hash_mode_stores_nothing(tmp_path):
    store = BlobStore(tmp_path)
    refs = capture_ir(_fn, _X, _W, store=store, mode=resolve_capture_mode(CAPTURE_HASH))
    assert all(r.mode == IR_HASH_ONLY for r in refs)
    assert store.count() == 0


def test_capture_mode_reads_the_environment(monkeypatch):
    monkeypatch.setenv("XTRAX_CAPTURE_IR", CAPTURE_NONE)
    assert not resolve_capture_mode().enabled


def test_an_unrecognised_capture_mode_defaults_to_full_rather_than_raising(monkeypatch):
    """A typo in an env var must not stop a training run, and must fail safe."""
    monkeypatch.setenv("XTRAX_CAPTURE_IR", "definitely-not-a-mode")
    mode = resolve_capture_mode()
    assert mode.raw == CAPTURE_FULL
    assert mode.store_text


def test_capture_mode_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("XTRAX_CAPTURE_IR", "NONE")
    assert not resolve_capture_mode().enabled


@pytest.mark.parametrize(
    ("raw", "expected_kinds"),
    [
        (CAPTURE_FULL, 2),
        (CAPTURE_FULL_OPTIMIZED, 3),
        (CAPTURE_HASH, 2),
        (CAPTURE_NONE, 0),
    ],
)
def test_capture_mode_kind_counts(raw, expected_kinds):
    assert len(resolve_capture_mode(raw).kinds) == expected_kinds
