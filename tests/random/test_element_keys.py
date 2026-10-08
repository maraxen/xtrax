"""Chunk- and resume-invariant PRNG key stream (#2544).

Golden key_data (uint32) captured from aminx.host.plan.compute_sample_keys
at aminx commit d1210e4adca9be19dc6cf03e07c06832c6191607
(jax 0.10.2, default_prng_impl=threefry2x32). Base keys are
jax.random.key(seed); n=13; chunk_sample_start=0. The resume-7 slices
captured in the same session match these rows from index 7.
"""

import math
import sys

import jax
import numpy as np
import pytest

from xtrax.random import (
    ChunkSpan,
    element_keys,
    iter_chunk_keys,
    make_chunk_plan,
)

N = 13
CHUNK_SIZES = (1, 2, 5, 13)
RESUME_STARTS = (0, 4, 7)

# element_keys(0, 2**31 - 2, 3) from the same aminx commit. The third index
# is 2**31, which numpy int32 addition wraps to -2**31.
INT32_WRAP_GOLDEN = np.array(
    [
        [1120217530, 2425421979],
        [3380239599, 3343201961],
        [917456828, 636652529],
    ],
    dtype=np.uint32,
)

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


class _PlanDidNotFinish(TimeoutError):
    """``make_chunk_plan`` exceeded the test's iteration bound."""


# Line events inside make_chunk_plan, not wall time. A ``while <=`` off-by-one
# appends a zero-length chunk forever; a one-second alarm still lets that loop
# allocate until the machine stalls. A few thousand lines is enough for every
# plan this file builds and stops the runaway loop before it grows.
_PLAN_LINE_BUDGET = 5000


def _call_bounded(fn, /, *args, **kwargs):
    """Call ``fn`` and fail if ``make_chunk_plan`` exceeds ``_PLAN_LINE_BUDGET``."""
    events = sys.monitoring.events
    tool = sys.monitoring.PROFILER_ID
    sys.monitoring.use_tool_id(tool, "xtrax-plan-bound")
    seen = {"n": 0}

    def _on_line(code, line_number):
        if code.co_name != "make_chunk_plan":
            return None
        seen["n"] += 1
        if seen["n"] > _PLAN_LINE_BUDGET:
            msg = (
                "make_chunk_plan exceeded the iteration bound (possible while-condition off-by-one)"
            )
            raise _PlanDidNotFinish(msg)
        return None

    sys.monitoring.register_callback(tool, events.LINE, _on_line)
    sys.monitoring.set_events(tool, events.LINE)
    try:
        return fn(*args, **kwargs)
    finally:
        sys.monitoring.set_events(tool, events.NO_EVENTS)
        sys.monitoring.register_callback(tool, events.LINE, None)
        sys.monitoring.free_tool_id(tool)


def _chunk_plan(count: int, chunk_size: int, *, start: int = 0) -> tuple[ChunkSpan, ...]:
    try:
        plan = _call_bounded(make_chunk_plan, count, chunk_size, start=start)
    except _PlanDidNotFinish as exc:
        pytest.fail(str(exc))
    expected_len = 0 if count == 0 else math.ceil(count / chunk_size)
    # Length is checked before any walk of the spans. A planner that emits
    # extra empty chunks fails here instead of looping in the test.
    assert len(plan) == expected_len
    return plan


def _chunk_local_keys(base_key: jax.Array, count: int) -> jax.Array:
    """Negative control: fold_in the index within the chunk, ignoring the global start."""
    local = np.arange(count, dtype=np.int32)
    return jax.vmap(lambda idx: jax.random.fold_in(base_key, idx))(local)


class TestElementKeys:
    def test_int32_wrap_matches_aminx_golden(self):
        got = _key_data(element_keys(0, 2**31 - 2, 3))
        np.testing.assert_array_equal(got, INT32_WRAP_GOLDEN)

    def test_int32_overflow_wraps_like_numpy(self):
        """``start=2**31-1``, ``count=2`` wraps as numpy int32 addition does.

        The documented contract is that ``start + i`` is an ``int32`` sum:
        ``2**31 - 1`` stays the int32 maximum, and the next index wraps to
        the int32 minimum (``-2**31``). Keys are ``fold_in`` of those bits.
        """
        start = 2**31 - 1
        count = 2
        wrapped = np.arange(count, dtype=np.int32) + np.int32(start)
        assert wrapped.dtype == np.dtype(np.int32)
        assert int(wrapped[0]) == np.iinfo(np.int32).max
        assert int(wrapped[1]) == np.iinfo(np.int32).min
        base = jax.random.key(0)
        expected = _key_data(jax.vmap(lambda idx: jax.random.fold_in(base, idx))(wrapped))
        got = _key_data(element_keys(0, start, count))
        np.testing.assert_array_equal(got, expected)

    def test_legacy_prngkey_matches_typed_key(self):
        legacy = jax.random.PRNGKey(12345)
        typed = jax.random.key(12345)
        assert legacy.dtype == np.uint32
        assert legacy.shape == (2,)
        from_legacy = _key_data(element_keys(legacy, 0, N))
        from_typed = _key_data(element_keys(typed, 0, N))
        np.testing.assert_array_equal(from_legacy, from_typed)

    @pytest.mark.parametrize(
        ("base_key", "typed"),
        ((jax.random.key(0), True), (0, True), (jax.random.PRNGKey(0), False)),
        ids=("typed-key", "int-seed", "legacy-key"),
    )
    def test_output_key_kind_follows_the_input(self, base_key, typed: bool):
        keys = element_keys(base_key, 0, 3)
        if typed:
            assert jax.dtypes.issubdtype(keys.dtype, jax.dtypes.prng_key)
            assert keys.shape == (3,)
        else:
            assert keys.dtype == np.uint32
            assert keys.shape == (3, 2)

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

    def test_negative_count_raises(self):
        with pytest.raises(ValueError, match=r"count must be >= 0, got -1"):
            element_keys(0, 0, -1)

    def test_bool_seed_raises(self):
        with pytest.raises(TypeError, match=r"seed must be an int or a PRNG key, got bool"):
            element_keys(True, 0, 1)

    def test_non_integer_start_raises(self):
        with pytest.raises(TypeError, match=r"start must be an integer, got float"):
            element_keys(0, 1.5, 1)

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
        plan = _chunk_plan(count, chunk_size, start=resume)
        covered = sum(span.count for span in plan)
        assert covered == count
        assert plan[0].start == resume
        got = _concat_plan(seed, plan)
        unchunked = _key_data(element_keys(seed, resume, count))
        np.testing.assert_array_equal(got, unchunked)
        np.testing.assert_array_equal(got, _golden(seed, resume, count))

    def test_resume_plan_matches_suffix_of_original_plan(self):
        full = _chunk_plan(N, 5, start=0)
        resumed = _chunk_plan(N - 4, 5, start=4)
        full_data = _concat_plan(0, full)
        resumed_data = _concat_plan(0, resumed)
        np.testing.assert_array_equal(resumed_data, full_data[4:])

    def test_chunk_local_derivation_fails_invariance(self):
        """A fold_in of the in-chunk index changes with the chunk boundary."""
        base = jax.random.key(0)
        full = _key_data(element_keys(base, 0, N))
        local_parts = []
        for span in _chunk_plan(N, 5, start=0):
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

    def test_plan_spans_cover_range_without_hanging(self):
        cases = (
            (0, 5, 0),  # count=0 -> empty plan
            (3, 10, 4),  # chunk_size > count -> one chunk
            (13, 5, 0),
            (10, 3, 7),
            (8, 1, 2),
        )
        for count, chunk_size, start in cases:
            plan = _chunk_plan(count, chunk_size, start=start)
            if count == 0:
                assert plan == ()
                continue
            if chunk_size > count:
                assert len(plan) == 1
                assert plan[0] == ChunkSpan(start, count)
            cursor = start
            for span in plan:
                assert span.count >= 1
                assert span.count <= chunk_size
                assert span.start == cursor
                cursor += span.count
            assert cursor == start + count

    def test_make_chunk_plan_rejects_bad_size_and_count(self):
        with pytest.raises(ValueError, match=r"chunk_size must be >= 1, got 0"):
            _call_bounded(make_chunk_plan, 4, 0)
        with pytest.raises(ValueError, match=r"count must be >= 0, got -3"):
            _call_bounded(make_chunk_plan, -3, 2)


def test_iter_chunk_keys_yields_span_slices():
    plan = (ChunkSpan(0, 2), ChunkSpan(2, 1))
    seen = []
    for span, keys in iter_chunk_keys(12345, plan):
        seen.append(span)
        np.testing.assert_array_equal(_key_data(keys), _golden(12345, span.start, span.count))
        assert keys.shape[0] == span.count
    assert seen == list(plan)
