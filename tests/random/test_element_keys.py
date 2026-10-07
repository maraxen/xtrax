"""Chunk- and resume-invariant PRNG key stream (#2544).

Golden key_data (uint32) captured from aminx.host.plan.compute_sample_keys
at aminx commit d1210e4adca9be19dc6cf03e07c06832c6191607
(jax 0.10.2, default_prng_impl=threefry2x32). Base keys are
jax.random.key(seed); n=13; chunk_sample_start=0. The resume-7 slices
captured in the same session match these rows from index 7.
"""

import jax
import numpy as np
import pytest

from xtrax.random import (
    INDEX_DTYPE,
    ChunkSpan,
    element_keys,
    iter_chunk_keys,
    make_chunk_plan,
)

N = 13
CHUNK_SIZES = (1, 2, 5, 13)
RESUME_STARTS = (0, 4, 7)

# Rows are jax.random.key_data of compute_sample_keys(jax.random.key(seed), 13).
GOLDEN = {
    0: (
        (1797259609, 2579123966),
        (928981903, 3453687069),
        (4146024105, 2718843009),
        (2467461003, 3840466878),
        (2285895361, 433833334),
        (1524306142, 1887795613),
        (3792494674, 2909014575),
        (2716826189, 292468403),
        (2292653872, 931858003),
        (823147153, 450340679),
        (3668660785, 652965180),
        (255289810, 156206175),
        (2763346363, 124557586),
    ),
    12345: (
        (1214163296, 439912094),
        (867802714, 3762255628),
        (795667951, 2300365598),
        (709272294, 1661227809),
        (305750198, 730924241),
        (1545186055, 432531754),
        (3552431482, 1517125165),
        (3645526921, 1314572738),
        (3549796694, 2529333308),
        (2657012232, 1416960692),
        (1763606390, 309040335),
        (3052679535, 1314812838),
        (2650252694, 1226034348),
    ),
}


def _key_data(keys: jax.Array) -> np.ndarray:
    return np.asarray(jax.random.key_data(keys), dtype=np.uint32)


def _golden(seed: int, start: int = 0, count: int | None = None) -> np.ndarray:
    rows = GOLDEN[seed]
    stop = len(rows) if count is None else start + count
    return np.asarray(rows[start:stop], dtype=np.uint32)


def _concat_plan(seed: int, plan: tuple[ChunkSpan, ...]) -> np.ndarray:
    parts = [_key_data(keys) for _span, keys in iter_chunk_keys(seed, plan)]
    if not parts:
        return np.zeros((0, 2), dtype=np.uint32)
    return np.concatenate(parts, axis=0)


def _chunk_local_keys(base_key: jax.Array, count: int) -> jax.Array:
    """Negative control: fold_in the index within the chunk, ignoring the global start."""
    local = np.arange(count, dtype=np.int32)
    return jax.vmap(lambda idx: jax.random.fold_in(base_key, idx))(local)


class TestElementKeys:
    def test_index_dtype_contract(self):
        assert INDEX_DTYPE == np.dtype(np.int32)

    @pytest.mark.parametrize("seed", (0, 12345))
    def test_matches_aminx_golden(self, seed: int):
        got = _key_data(element_keys(seed, 0, N))
        np.testing.assert_array_equal(got, _golden(seed))

    @pytest.mark.parametrize("seed", (0, 12345))
    def test_int_seed_matches_explicit_key(self, seed: int):
        from_seed = _key_data(element_keys(seed, 0, N))
        from_key = _key_data(element_keys(jax.random.key(seed), 0, N))
        np.testing.assert_array_equal(from_seed, from_key)

    def test_numpy_integer_seed(self):
        got = _key_data(element_keys(np.int64(0), 0, N))
        np.testing.assert_array_equal(got, _golden(0))

    def test_empty_count(self):
        keys = element_keys(0, 4, 0)
        assert _key_data(keys).shape == (0, 2)

    def test_negative_seed_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            element_keys(-1, 0, 1)

    def test_negative_start_raises(self):
        with pytest.raises(ValueError, match="start"):
            element_keys(0, -1, 1)

    def test_start_outside_int32_raises(self):
        with pytest.raises(ValueError, match="int32"):
            element_keys(0, int(np.iinfo(np.int32).max) + 1, 1)


class TestChunkAndResumeInvariance:
    @pytest.mark.parametrize("seed", (0, 12345))
    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    @pytest.mark.parametrize("resume", RESUME_STARTS)
    def test_chunked_resume_matches_unchunked_and_golden(
        self, seed: int, chunk_size: int, resume: int
    ):
        count = N - resume
        plan = make_chunk_plan(count, chunk_size, start=resume)
        covered = sum(span.count for span in plan)
        assert covered == count
        assert plan[0].start == resume
        got = _concat_plan(seed, plan)
        unchunked = _key_data(element_keys(seed, resume, count))
        np.testing.assert_array_equal(got, unchunked)
        np.testing.assert_array_equal(got, _golden(seed, resume, count))

    def test_resume_plan_matches_suffix_of_original_plan(self):
        full = make_chunk_plan(N, 5, start=0)
        resumed = make_chunk_plan(N - 4, 5, start=4)
        full_data = _concat_plan(0, full)
        resumed_data = _concat_plan(0, resumed)
        np.testing.assert_array_equal(resumed_data, full_data[4:])

    def test_chunk_local_derivation_fails_invariance(self):
        """A fold_in of the in-chunk index changes with the chunk boundary."""
        base = jax.random.key(0)
        full = _key_data(element_keys(base, 0, N))
        local_parts = []
        for span in make_chunk_plan(N, 5, start=0):
            local_parts.append(_key_data(_chunk_local_keys(base, span.count)))
        local = np.concatenate(local_parts, axis=0)
        assert local.shape == full.shape
        assert not np.array_equal(local, full)
        # The first chunk starts at global 0, so only later chunks diverge.
        assert np.array_equal(local[:5], full[:5])
        assert not np.array_equal(local[5:], full[5:])

    def test_public_package_export(self):
        import xtrax

        assert xtrax.element_keys is element_keys
        assert xtrax.iter_chunk_keys is iter_chunk_keys
        assert xtrax.make_chunk_plan is make_chunk_plan
        assert xtrax.ChunkSpan is ChunkSpan


def test_iter_chunk_keys_yields_span_slices():
    plan = (ChunkSpan(0, 2), ChunkSpan(2, 1))
    seen = []
    for span, keys in iter_chunk_keys(12345, plan):
        seen.append(span)
        np.testing.assert_array_equal(_key_data(keys), _golden(12345, span.start, span.count))
        assert keys.shape[0] == span.count
    assert seen == list(plan)
