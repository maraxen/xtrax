"""The public jaxpr walker covers nested higher-order primitives."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from xtrax.profiling import iter_jaxpr_eqns, sub_jaxprs


def test_walker_is_exported_from_profiling() -> None:
    from xtrax.profiling.jaxpr import iter_jaxpr_eqns as defined

    assert iter_jaxpr_eqns is defined


def test_scan_inside_while_inside_cond_is_walked() -> None:
    """A mul that exists only in the scan body is invisible at the top level."""

    def program(x: jax.Array) -> jax.Array:
        def when_true(y: jax.Array) -> jax.Array:
            def cond(state: tuple[jax.Array, jax.Array]) -> jax.Array:
                return state[0] < 2

            def body(state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
                step, carry = state

                def scan_body(c: jax.Array, _: None) -> tuple[jax.Array, None]:
                    return c * 2.0, None

                updated = jax.lax.scan(scan_body, carry, None, length=3)[0]
                return step + 1, updated

            return jax.lax.while_loop(cond, body, (jnp.int32(0), y))[1]

        def when_false(y: jax.Array) -> jax.Array:
            return y - 1.0

        return jax.lax.cond(x > 0, when_true, when_false, x)

    closed = jax.make_jaxpr(program)(jnp.float32(1.0))
    names = [eqn.primitive.name for eqn in iter_jaxpr_eqns(closed)]
    top = [eqn.primitive.name for eqn in closed.eqns]
    assert "cond" in names
    assert "while" in names
    assert "scan" in names
    assert "mul" in names
    assert "scan" not in top
    assert "while" not in top
    assert "mul" not in top

    cond_eqn = next(eqn for eqn in closed.eqns if eqn.primitive.name == "cond")
    branches = list(sub_jaxprs(cond_eqn.params))
    assert len(branches) == 2
    assert any(eqn.primitive.name == "while" for branch in branches for eqn in branch.eqns)
