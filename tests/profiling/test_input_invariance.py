"""Declared per-axis input invariance is checked against jaxpr dependence (#2599)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from xtrax.inference.cse import align_call_invars
from xtrax.tiling.plan import AxisSpec, declare_varying_inputs


def _specs(*varying: str) -> tuple[AxisSpec, ...]:
    return declare_varying_inputs(
        (AxisSpec(name="batch", cardinality=4, default_batch_size=4),),
        {"batch": varying},
        input_names=("params", "x"),
    )


def _encoder(params, x):
    """Encoder reads only the invariant input; the product is per-element."""
    hidden = jnp.sin(params)
    return hidden, hidden * x


def _mixed(params, x):
    """The invariant input combines directly with the varying one."""
    return params + x


def test_leading_callee_invar_without_a_caller_reads_no_input():
    """A callee invar paired with None depends on no top-level input."""
    from types import SimpleNamespace

    from xtrax.inference.cse import trace_input_dependence

    class _Var:
        pass

    caller = _Var()
    const_in = _Var()
    bound_in = _Var()
    out = _Var()
    sub = SimpleNamespace(
        invars=(const_in, bound_in),
        outvars=(const_in,),
        eqns=(),
        constvars=(),
    )
    eqn = SimpleNamespace(
        primitive=SimpleNamespace(name="call"),
        invars=(caller,),
        outvars=(out,),
        params={"jaxpr": sub},
    )
    closed = SimpleNamespace(
        invars=(caller,),
        outvars=(out,),
        eqns=(eqn,),
        constvars=(),
    )
    traced = trace_input_dependence(closed)
    assert traced.outputs == (frozenset(),)


def test_constvar_output_reads_no_input():
    """A closed-over constant is invariant along every axis."""
    from types import SimpleNamespace

    from xtrax.inference.cse import trace_input_dependence

    class _Var:
        pass

    const = _Var()
    arg = _Var()
    closed = SimpleNamespace(
        invars=(arg,),
        constvars=(const,),
        outvars=(const,),
        eqns=(),
    )
    traced = trace_input_dependence(closed)
    assert traced.outputs == (frozenset(),)


def test_align_call_invars_pairs_from_the_right():
    """Call invars line up with the callee; a cond predicate is the extra prefix."""
    assert align_call_invars(("p", "x"), ("a", "b")) == (("a", "p"), ("b", "x"))
    assert align_call_invars(("pred", "p", "x"), ("a", "b")) == (("a", "p"), ("b", "x"))
    assert align_call_invars(("x",), ("c", "a")) == (("c", None), ("a", "x"))


def test_encoder_on_invariant_input_is_invariant_subgraph():
    """sin(params) is invariant along batch; the product that reads x is not."""
    from xtrax.inference import verify_axis_invariance

    report = verify_axis_invariance(
        _encoder,
        {"params": jnp.float32(0.5), "x": jnp.float32(1.5)},
        _specs("x"),
    )
    axis = report.axis("batch")
    assert axis.varying_inputs == ("x",)
    assert "sin" in axis.invariant_intermediates
    assert axis.invariant_outputs == (0,)


def test_invariant_input_combined_with_varying_input_is_allowed():
    """Weights times per-element data is ordinary code, not a contradiction."""
    from xtrax.inference import verify_axis_invariance

    report = verify_axis_invariance(
        lambda w, x: w @ x,
        {"params": jnp.ones((3, 3)), "x": jnp.ones((3,))},
        _specs("x"),
    )
    assert report.axis("batch").invariant_outputs == ()
    report = verify_axis_invariance(
        _mixed, {"params": jnp.float32(0.5), "x": jnp.float32(1.5)}, _specs("x")
    )
    assert report.axis("batch").invariant_outputs == ()


def test_correct_invariance_claim_passes():
    """Claiming the encoder output invariant matches the jaxpr."""
    from xtrax.inference import verify_axis_invariance

    report = verify_axis_invariance(
        _encoder,
        {"params": jnp.float32(0.5), "x": jnp.float32(1.5)},
        _specs("x"),
        claimed_invariant_outputs={"batch": [0]},
    )
    assert report.axis("batch").invariant_outputs == (0,)


def test_claimed_invariant_output_that_reads_varying_input_raises():
    """Output 1 reads x, so claiming it invariant along batch is contradicted."""
    from xtrax.inference import InputInvarianceError, verify_axis_invariance

    with pytest.raises(InputInvarianceError, match="output 1") as exc:
        verify_axis_invariance(
            _encoder,
            {"params": jnp.float32(0.5), "x": jnp.float32(1.5)},
            _specs("x"),
            claimed_invariant_outputs={"batch": [0, 1]},
        )
    assert exc.value.axis == "batch"
    assert exc.value.output_index == 1
    assert exc.value.input_name == "x"


@pytest.mark.parametrize(
    ("claims", "match"),
    [
        ({"seq": [0]}, "unknown axis 'seq'"),
        ({"batch": [2]}, "out of range"),
        ({"batch": [-1]}, "out of range"),
    ],
    ids=["unknown-axis", "index-too-large", "negative-index"],
)
def test_malformed_claims_raise(claims, match):
    from xtrax.inference import verify_axis_invariance

    with pytest.raises(ValueError, match=match):
        verify_axis_invariance(
            _encoder,
            {"params": jnp.float32(0.5), "x": jnp.float32(1.5)},
            _specs("x"),
            claimed_invariant_outputs=claims,
        )


def test_verify_unknown_input_names_the_axis():
    """A varying name absent from the example inputs raises before tracing."""
    from xtrax.inference import verify_axis_invariance

    spec = AxisSpec(
        name="batch",
        cardinality=4,
        default_batch_size=4,
        varying_inputs=("nope",),
    )
    with pytest.raises(ValueError, match="unknown input 'nope'") as exc:
        verify_axis_invariance(
            _encoder,
            {"params": jnp.float32(0.5), "x": jnp.float32(1.5)},
            (spec,),
        )
    assert "batch" in str(exc.value)


def _wrap(kind: str, body):
    if kind == "checkpoint":
        return jax.checkpoint(body)
    if kind == "custom_jvp":
        fn = jax.custom_jvp(body)

        @fn.defjvp
        def _jvp(primals, tangents):
            value = body(*primals)
            tangent = jax.tree.map(jnp.zeros_like, value)
            return value, tangent

        return fn
    if kind == "pjit":
        # jax.jit lowers through the pjit call primitive.
        return jax.jit(body)
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["checkpoint", "custom_jvp", "pjit"])
def test_invariant_encoder_inside_callee_is_seen(kind: str):
    """sin that exists only inside checkpoint / custom_jvp / pjit is still invariant."""
    from xtrax.inference import verify_axis_invariance

    wrapped = _wrap(kind, _encoder)
    params = jnp.float32(0.5)
    x = jnp.float32(1.5)
    closed = jax.make_jaxpr(wrapped)(params, x)
    top = [eqn.primitive.name for eqn in closed.eqns]
    assert "sin" not in top

    report = verify_axis_invariance(
        wrapped,
        {"params": params, "x": x},
        _specs("x"),
    )
    axis = report.axis("batch")
    assert "sin" in axis.invariant_intermediates
    assert axis.invariant_outputs == (0,)


@pytest.mark.parametrize("kind", ["checkpoint", "custom_jvp", "pjit"])
def test_claim_contradicted_inside_callee_raises(kind: str):
    """The product that reads x is hidden in the callee; the claim still fails."""
    from xtrax.inference import InputInvarianceError, verify_axis_invariance

    wrapped = _wrap(kind, _encoder)
    with pytest.raises(InputInvarianceError, match="'x'") as exc:
        verify_axis_invariance(
            wrapped,
            {"params": jnp.float32(0.5), "x": jnp.float32(1.5)},
            _specs("x"),
            claimed_invariant_outputs={"batch": [1]},
        )
    assert exc.value.axis == "batch"
    assert exc.value.output_index == 1


def test_cond_branch_sees_invariant_encoder_and_not_the_predicate():
    """The cond predicate is not a branch input; sin(params) stays invariant."""
    from xtrax.inference import verify_axis_invariance

    def program(params, x):
        hidden = jnp.sin(params)

        def when_true(a, b):
            return a + b

        def when_false(a, b):
            return a - b

        mixed = jax.lax.cond(x > 0, when_true, when_false, hidden, x)
        return hidden, mixed

    report = verify_axis_invariance(
        program,
        {"params": jnp.float32(0.5), "x": jnp.float32(1.5)},
        _specs("x"),
    )
    axis = report.axis("batch")
    assert "sin" in axis.invariant_intermediates
    assert axis.invariant_outputs == (0,)


def test_cond_output_claimed_invariant_raises_on_the_predicate_input():
    """A cond whose predicate reads x is not invariant, even if branches ignore x."""
    from xtrax.inference import InputInvarianceError, verify_axis_invariance

    def program(params, x):
        return jax.lax.cond(x > 0, jnp.sin, jnp.cos, params)

    with pytest.raises(InputInvarianceError, match="'x'") as exc:
        verify_axis_invariance(
            program,
            {"params": jnp.float32(0.5), "x": jnp.float32(1.5)},
            _specs("x"),
            claimed_invariant_outputs={"batch": [0]},
        )
    assert exc.value.axis == "batch"


def test_verify_unknown_input_with_no_examples_lists_none():
    """An empty example catalog still names the unknown input."""
    from xtrax.inference import verify_axis_invariance

    spec = AxisSpec(
        name="batch",
        cardinality=4,
        default_batch_size=4,
        varying_inputs=("x",),
    )
    with pytest.raises(ValueError, match="unknown input 'x'") as exc:
        verify_axis_invariance(_encoder, {}, (spec,))
    assert "(none)" in str(exc.value)


def test_empty_declaration_reports_no_axis():
    """No specs yields an empty report, and lookup names that emptiness."""
    from xtrax.inference import verify_axis_invariance

    report = verify_axis_invariance(
        _encoder,
        {"params": jnp.float32(0.5), "x": jnp.float32(1.5)},
        (),
    )
    assert report.by_axis == ()
    with pytest.raises(KeyError, match="none"):
        report.axis("batch")


def test_unknown_report_axis_raises():
    """Looking up an axis the report does not contain names the known axes."""
    from xtrax.inference import verify_axis_invariance

    report = verify_axis_invariance(
        _encoder,
        {"params": jnp.float32(0.5), "x": jnp.float32(1.5)},
        _specs("x"),
    )
    with pytest.raises(KeyError, match="token"):
        report.axis("token")
