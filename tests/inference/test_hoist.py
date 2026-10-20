"""Tests for xtrax.inference.hoist (debt #2600): acceptance A1-A8 of spec 261020_hoisted-map."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend import core as jcore

import xtrax.inference as inference
from xtrax.inference import memo
from xtrax.inference.errors import HoistParityError, HoistVaryingInputError, XtraxInferenceError
from xtrax.inference.hoist import hoisted_map, make_hoisted
from xtrax.inference.memo import MemoPolicy
from xtrax.tiling.plan import AxisSpec

L = 5  # elements in the per-element structure
C = 3  # channels of the encoder output
_W = jnp.linspace(-1.0, 1.0, 3 * C).reshape(3, C)


def encoder(coords: jax.Array) -> jax.Array:
    """(L, 3) -> (L, L, C): the expensive structure encoder."""
    diff = coords[:, None, :] - coords[None, :, :]
    return jnp.tanh(diff @ _W)


def _coords() -> jax.Array:
    return jnp.linspace(0.0, 2.0, L * 3).reshape(L, 3) ** 2


def _keys(n: int, seed: int = 0) -> jax.Array:
    return jax.random.split(jax.random.key(seed), n)


# Monolithic per-element function. The key flows into the encoder input through a
# zero-scaled term: the value is key-independent, but the jaxpr reads the key. The
# non-zero key term makes each element's output differ, so ordering bugs show up.
def monolithic(args: dict[str, Any], key: jax.Array) -> jax.Array:
    noisy = args["coords"] + 0.0 * jax.random.normal(key, args["coords"].shape)
    return encoder(noisy).sum(axis=1) + 0.1 * jax.random.normal(key, (L, C))


# Hoisted split of `monolithic`: the encoder is shared, the key stays per element.
def shared_encoder(coords: jax.Array) -> jax.Array:
    return encoder(coords)


def per_element_key(enc: jax.Array, key: jax.Array) -> jax.Array:
    return enc.sum(axis=1) + 0.1 * jax.random.normal(key, (L, C))


def _reference_outputs(coords: jax.Array, keys: jax.Array) -> jax.Array:
    return jax.vmap(lambda k: monolithic({"coords": coords}, k))(keys)


def _sub_jaxprs(value: Any) -> list[Any]:
    """Jaxprs nested in one equation parameter (pjit, scan, while, cond branches)."""
    if isinstance(value, jcore.ClosedJaxpr):
        return [value.jaxpr]
    if isinstance(value, jcore.Jaxpr):
        return [value]
    if isinstance(value, (tuple, list)):
        return [sub for item in value for sub in _sub_jaxprs(item)]
    return []


def _jaxpr_shapes(jaxpr: Any) -> set[tuple[int, ...]]:
    """Shapes of every equation output, walking sub-jaxprs recursively."""
    shapes: set[tuple[int, ...]] = set()
    for eqn in jaxpr.eqns:
        shapes.update(tuple(v.aval.shape) for v in eqn.outvars)
        for param in eqn.params.values():
            for sub in _sub_jaxprs(param):
                shapes |= _jaxpr_shapes(sub)
    return shapes


def _traced_shapes(fn: Callable[..., Any], *args: Any) -> set[tuple[int, ...]]:
    return _jaxpr_shapes(jax.make_jaxpr(fn)(*args).jaxpr)


def _record_encoder_evals(counter: list[int], enc: jax.Array) -> None:
    """Add the encoder evaluations that ``enc`` stands for to ``counter`` (debug callback).

    A callback under vmap runs once with the batched value, so the count is the
    number of (L, L, C) encoder outputs in the batch, not the number of callbacks.
    """

    def bump(value: Any) -> None:
        counter[0] += int(np.size(value)) // (L * L * C)

    jax.debug.callback(bump, enc)


def _counting_shared(counter: list[int]) -> Callable[[jax.Array], jax.Array]:
    """Shared encoder that records its evaluations in ``counter`` (debug callback)."""

    def shared(coords: jax.Array) -> jax.Array:
        enc = encoder(coords)
        _record_encoder_evals(counter, enc)
        return enc

    return shared


def _is_typed_key(leaf: Any) -> bool:
    return hasattr(leaf, "dtype") and jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key)


# --- A1 + A2: runs once per call; parity on vmap, ragged chunks, and N=1 -------------

# N=9 with batch 4 is the remainder-1 branch of chunked_map.
PARITY_CASES = [(16, None), (10, 4), (9, 4), (1, None)]
PARITY_IDS = ["vmap-N16", "chunked-N10-b4", "remainder1-N9-b4", "single-N1"]


@pytest.mark.parametrize(("n", "batch_size"), PARITY_CASES, ids=PARITY_IDS)
def test_matches_reference_per_element(n: int, batch_size: int | None) -> None:
    coords = _coords()
    keys = _keys(n)
    hoisted = make_hoisted(shared_encoder, per_element_key, batch_size=batch_size)
    out = hoisted({"coords": coords}, keys)
    assert out.shape == (n, L, C)
    np.testing.assert_allclose(out, _reference_outputs(coords, keys), rtol=1e-5, atol=1e-8)
    if n > 1:
        # Distinct per-element outputs, so a misordered or misplaced chunk fails the check above.
        assert not np.allclose(out[0], out[-1])


@pytest.mark.parametrize(("n", "batch_size"), PARITY_CASES, ids=PARITY_IDS)
def test_shared_executes_once_per_call_uncached(n: int, batch_size: int | None) -> None:
    coords = _coords()
    keys = _keys(n)
    counter = [0]
    hoisted = make_hoisted(_counting_shared(counter), per_element_key, batch_size=batch_size)
    for call in range(2):
        out = hoisted({"coords": coords}, keys)
        jax.block_until_ready(out)
        jax.effects_barrier()
        assert counter[0] == call + 1
        assert hoisted.stats()["shared_calls"] == call + 1
    np.testing.assert_allclose(out, _reference_outputs(coords, keys), rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize(("n", "batch_size"), PARITY_CASES, ids=PARITY_IDS)
def test_traced_hoisted_call_has_encoder_once_and_no_batched_encoder(
    n: int, batch_size: int | None
) -> None:
    coords = _coords()
    keys = _keys(n)

    def program(c: jax.Array, ks: jax.Array) -> jax.Array:
        hoisted = make_hoisted(shared_encoder, per_element_key, batch_size=batch_size)
        return hoisted({"coords": c}, ks)

    shapes = _traced_shapes(program, coords, keys)
    assert (L, L, C) in shapes
    # Any leading axis on an (L, L, C) value means the encoder output was replicated
    # per element or per chunk: (N, ...) under vmap, (batch_size, ...) or (1, ...) otherwise.
    assert not [s for s in shapes if len(s) == 4 and s[1:] == (L, L, C)]


def test_counter_distinguishes_a_recomputing_mutant() -> None:
    """Mutant: the encoder moved into per_element, so it runs once per element."""
    n = 16
    counter = [0]

    def bumped_encoder(coords: jax.Array) -> jax.Array:
        enc = encoder(coords)
        _record_encoder_evals(counter, enc)
        return enc

    def mutant_per_element(coords: jax.Array, key: jax.Array) -> jax.Array:
        # The key reaches the encoder input (false dependence), so vmap batches the encoder.
        noisy = coords + 0.0 * jax.random.normal(key, coords.shape)
        return bumped_encoder(noisy).sum(axis=1) + 0.1 * jax.random.normal(key, (L, C))

    coords = _coords()
    keys = _keys(n)
    hoisted = make_hoisted(lambda coords: coords, mutant_per_element)
    out = hoisted({"coords": coords}, keys)
    jax.block_until_ready(out)
    jax.effects_barrier()
    # The measurement distinguishes: the mutant runs the encoder N times per call, not once.
    assert counter[0] == n
    shapes = _traced_shapes(
        lambda c, ks: make_hoisted(lambda coords: coords, mutant_per_element)({"coords": c}, ks),
        coords,
        keys,
    )
    assert (n, L, L, C) in shapes


def test_cached_shared_is_memoized_across_calls() -> None:
    coords = _coords()
    keys = _keys(4)
    hoisted = make_hoisted(shared_encoder, per_element_key, cache=MemoPolicy())
    for _ in range(3):
        hoisted({"coords": coords}, keys)
    stats = hoisted.stats()
    assert (stats["misses"], stats["hits"], stats["shared_calls"]) == (1, 2, 3)

    hoisted({"coords": coords + 1.0}, keys)
    stats = hoisted.stats()
    assert (stats["misses"], stats["hits"]) == (2, 2)


def test_cached_parity_matches_reference() -> None:
    coords = _coords()
    keys = _keys(10)
    hoisted = make_hoisted(shared_encoder, per_element_key, batch_size=4, cache=MemoPolicy())
    out = hoisted({"coords": coords}, keys)
    np.testing.assert_allclose(out, _reference_outputs(coords, keys), rtol=1e-5, atol=1e-8)


def test_uncached_stats_keys() -> None:
    hoisted = make_hoisted(shared_encoder, per_element_key)
    hoisted({"coords": _coords()}, _keys(2))
    assert hoisted.stats() == {"shared_calls": 1, "traced_bypass": 0}


# --- A3: a typed PRNG key in shared_args is part of the cache key -----------------


def shared_with_key(coords: jax.Array, key: jax.Array) -> jax.Array:
    """Reads the key's bits: a draw would be refused by the memo purity screen."""
    bits = jax.random.key_data(key).astype(jnp.float32)
    return encoder(coords) + 1e-2 * jnp.sin(bits).sum()


def element_scale(shared_out: jax.Array, x: jax.Array) -> jax.Array:
    return shared_out.sum(axis=1) * x


def test_typed_key_in_shared_args_changes_cache_key() -> None:
    coords = _coords()
    xs = jnp.arange(1.0, 4.0)
    hoisted = make_hoisted(shared_with_key, element_scale, cache=MemoPolicy())
    out_a = hoisted({"coords": coords, "key": jax.random.key(1)}, xs)
    out_b = hoisted({"coords": coords, "key": jax.random.key(2)}, xs)
    stats = hoisted.stats()
    assert (stats["misses"], stats["hits"]) == (2, 0)
    assert not np.allclose(out_a, out_b)


def test_dropping_typed_keys_from_digest_turns_key_change_into_stale_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative control for A3: without typed-key digests the second call is a stale hit."""
    original = memo._leaf_digest

    def skip_typed_keys(leaf: Any, sink: Any) -> None:
        if _is_typed_key(leaf):
            return
        original(leaf, sink)

    monkeypatch.setattr(memo, "_leaf_digest", skip_typed_keys)
    coords = _coords()
    xs = jnp.arange(1.0, 4.0)
    hoisted = make_hoisted(shared_with_key, element_scale, cache=MemoPolicy())
    out_a = hoisted({"coords": coords, "key": jax.random.key(1)}, xs)
    out_b = hoisted({"coords": coords, "key": jax.random.key(2)}, xs)
    stats = hoisted.stats()
    assert (stats["misses"], stats["hits"]) == (1, 1)
    np.testing.assert_array_equal(out_a, out_b)


# --- A4: the axis declaration catches a shared input that varies -------------------


def test_varying_shared_input_is_refused_by_name() -> None:
    axis = AxisSpec(name="pair", cardinality=3, default_batch_size=4, varying_inputs=("key",))
    hoisted = make_hoisted(shared_with_key, element_scale, axis=axis)
    with pytest.raises(HoistVaryingInputError, match="'key'") as info:
        hoisted({"coords": _coords(), "key": jax.random.key(0)}, jnp.arange(1.0, 4.0))
    assert info.value.axis == "pair"
    assert info.value.input_name == "key"
    assert isinstance(info.value, XtraxInferenceError)


def test_varying_input_only_in_xs_passes_the_axis_check() -> None:
    axis = AxisSpec(name="pair", cardinality=3, default_batch_size=4, varying_inputs=("key",))
    hoisted = make_hoisted(shared_encoder, per_element_key, axis=axis)
    out = hoisted({"coords": _coords()}, _keys(3))
    assert out.shape == (3, L, C)


# --- A5: the parity probe catches a per-element term that the hoist dropped -------


def shared_with_noise(coords: jax.Array, noise: jax.Array) -> tuple[jax.Array, jax.Array]:
    return encoder(coords).sum(axis=1), noise


def dropping_noise(shared_out: tuple[jax.Array, jax.Array], key: jax.Array) -> jax.Array:
    """Buggy per-element stage: omits the noise term that the reference includes."""
    enc_sum, _ = shared_out
    return enc_sum


def noisy_reference(args: dict[str, Any], key: jax.Array) -> jax.Array:
    enc_sum = encoder(args["coords"]).sum(axis=1)
    return enc_sum + args["noise"] * jax.random.normal(key, (L, C))


def test_parity_probe_catches_dropped_noise_term() -> None:
    hoisted = make_hoisted(shared_with_noise, dropping_noise)
    args = {"coords": _coords(), "noise": jnp.float32(0.1)}
    with pytest.raises(HoistParityError) as info:
        hoisted(args, _keys(3), reference=noisy_reference, check_elements=2)
    assert info.value.index == 0
    assert info.value.max_abs_diff > 0.0


def test_parity_probe_passes_when_the_dropped_term_is_zero() -> None:
    hoisted = make_hoisted(shared_with_noise, dropping_noise)
    args = {"coords": _coords(), "noise": 0.0}
    out = hoisted(args, _keys(3), reference=noisy_reference, check_elements=2)
    assert out.shape == (3, L, C)


# --- A6: the library's traced program has no batched encoder ----------------------


def _invariant_monolithic(args: dict[str, Any], key: jax.Array) -> jax.Array:
    """Control: the encoder reads no key at all."""
    return encoder(args["coords"]).sum(axis=1) + 0.1 * jax.random.normal(key, (L, C))


def test_false_dependence_is_removed_by_hoisting() -> None:
    coords = _coords()
    keys = _keys(4)
    n = keys.shape[0]

    def monolithic_program(c: jax.Array, ks: jax.Array) -> jax.Array:
        return jax.vmap(lambda k: monolithic({"coords": c}, k))(ks)

    def library_program(c: jax.Array, ks: jax.Array) -> jax.Array:
        return make_hoisted(shared_encoder, per_element_key)({"coords": c}, ks)

    def control_program(c: jax.Array, ks: jax.Array) -> jax.Array:
        return jax.vmap(lambda k: _invariant_monolithic({"coords": c}, k))(ks)

    mono = _traced_shapes(monolithic_program, coords, keys)
    hoisted = _traced_shapes(library_program, coords, keys)
    control = _traced_shapes(control_program, coords, keys)

    assert (n, L, L, C) in mono
    assert (n, L, L, C) not in hoisted
    assert (L, L, C) in hoisted
    # Control: a truly invariant encoder is not batched in either version, so the
    # shape check above can tell false dependence from real invariance.
    assert (L, L, C) in control
    assert (n, L, L, C) not in control


# --- A7: under an enclosing jit ----------------------------------------------------


@pytest.mark.parametrize("cache", [None, MemoPolicy()], ids=["uncached", "cached"])
def test_under_enclosing_jit_returns_correct_values(cache: MemoPolicy | None) -> None:
    coords = _coords()
    keys = _keys(5)
    hoisted = make_hoisted(shared_encoder, per_element_key, cache=cache)
    got = jax.jit(lambda c: hoisted({"coords": c}, keys))(coords)
    np.testing.assert_allclose(got, _reference_outputs(coords, keys), rtol=1e-5, atol=1e-8)
    if cache is None:
        assert hoisted.stats()["traced_bypass"] == 0
    else:
        assert hoisted.stats()["traced_bypass"] == 1


# --- Static Python arguments that fix a shape ----------------------------------------


def shared_sized(coords: jax.Array, n: int) -> jax.Array:
    """``n`` fixes a shape: the first n rows of the encoder's channel sums."""
    return encoder(coords).sum(axis=1)[:n]


def per_element_sized(shared_out: jax.Array, key: jax.Array) -> jax.Array:
    return shared_out.sum(axis=0) + 0.1 * jax.random.normal(key, (C,))


def monolithic_sized(args: dict[str, Any], key: jax.Array) -> jax.Array:
    return encoder(args["coords"]).sum(axis=1)[: args["n"]].sum(axis=0) + 0.1 * jax.random.normal(
        key, (C,)
    )


@pytest.mark.parametrize("cache", [None, MemoPolicy()], ids=["uncached", "cached"])
def test_static_python_arg_fixing_a_shape_works_in_every_path(cache: MemoPolicy | None) -> None:
    coords = _coords()
    keys = _keys(4)
    expected = jax.vmap(lambda k: monolithic_sized({"coords": coords, "n": 3}, k))(keys)
    hoisted = make_hoisted(shared_sized, per_element_sized, cache=cache)
    out = hoisted({"coords": coords, "n": 3}, keys)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-8)
    got = jax.jit(lambda c: hoisted({"coords": c, "n": 3}, keys))(coords)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-8)


# --- A8: argument errors ------------------------------------------------------------


@pytest.mark.parametrize(
    "xs",
    [jnp.zeros((0, 3)), {}, {"a": jnp.zeros((3, 2)), "b": jnp.zeros((4,))}],
    ids=["empty-axis", "no-leaves", "leaves-disagree"],
)
def test_bad_xs_is_a_value_error(xs: Any) -> None:
    hoisted = make_hoisted(shared_encoder, per_element_key)
    with pytest.raises(ValueError):
        hoisted({"coords": _coords()}, xs)


def test_probe_without_reference_is_a_value_error() -> None:
    hoisted = make_hoisted(shared_encoder, per_element_key)
    with pytest.raises(ValueError, match="reference"):
        hoisted({"coords": _coords()}, _keys(2), check_elements=1)


def test_probe_on_traced_xs_is_a_value_error() -> None:
    def run(ks: jax.Array) -> jax.Array:
        return hoisted_map(
            shared_encoder,
            per_element_key,
            {"coords": _coords()},
            ks,
            reference=monolithic,
            check_elements=1,
        )

    with pytest.raises(ValueError, match="concrete"):
        jax.jit(run)(_keys(2))


# --- one-shot form, exports -----------------------------------------------------


def test_hoisted_map_one_shot_matches_reference() -> None:
    coords = _coords()
    keys = _keys(6)
    out = hoisted_map(
        shared_encoder,
        per_element_key,
        {"coords": coords},
        keys,
        batch_size=4,
        reference=monolithic,
        check_elements=6,
    )
    np.testing.assert_allclose(out, _reference_outputs(coords, keys), rtol=1e-5, atol=1e-8)


def test_public_exports() -> None:
    for name in (
        "HoistedMap",
        "HoistParityError",
        "HoistVaryingInputError",
        "hoisted_map",
        "make_hoisted",
    ):
        assert name in inference.__all__
        assert hasattr(inference, name)
    assert issubclass(HoistVaryingInputError, XtraxInferenceError)
    assert issubclass(HoistParityError, XtraxInferenceError)
