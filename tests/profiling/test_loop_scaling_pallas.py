"""Tests for detecting and flagging scan over pallas_call in loop_scaling (Debt #3196)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from xtrax.profiling.loop_scaling import (
    extent_scaling_report,
    loop_bodies,
)


def _pallas_scale_body(x_ref, o_ref):
    o_ref[...] = x_ref[...] * 2.0


def pallas_scale(x):
    return pl.pallas_call(
        _pallas_scale_body,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=True,
    )(x)


def plain_scan(x: jax.Array) -> jax.Array:
    """Standard scan without pallas."""

    def body(carry, elem):
        return carry + jnp.sum(elem), elem

    out, _ = jax.lax.scan(body, jnp.float32(0.0), x)
    return out


def pallas_in_scan(x: jax.Array) -> jax.Array:
    """Scan whose step calls a pallas kernel (or vmap-induced scan over pallas)."""

    def body(carry, elem):
        scaled = pallas_scale(elem)
        return carry + jnp.sum(scaled), scaled

    out, _ = jax.lax.scan(body, jnp.float32(0.0), x)
    return out


def test_plain_scan_has_no_pallas():
    """Positive control: regular scan body does not have pallas_call."""
    x = jnp.ones((4, 8))
    bodies = loop_bodies(plain_scan, x)
    assert len(bodies) == 1
    assert bodies[0].has_pallas_call is False

    report = extent_scaling_report(
        plain_scan,
        lambda n: (jnp.ones((n, 8)),),
        extent=4,
    )
    assert len(report.findings) == 1
    assert report.findings[0].has_pallas_call is False
    assert report.findings[0].vmap_induced_pallas_scan is False
    assert len(report.pallas_scan_findings) == 0


def test_pallas_in_scan_is_flagged():
    """Negative control: scan containing pallas_call is detected and flagged."""
    x = jnp.ones((4, 8))
    bodies = loop_bodies(pallas_in_scan, x)
    assert len(bodies) == 1
    assert bodies[0].has_pallas_call is True

    report = extent_scaling_report(
        pallas_in_scan,
        lambda n: (jnp.ones((n, 8)),),
        extent=4,
    )
    assert len(report.findings) == 1
    finding = report.findings[0]
    assert finding.has_pallas_call is True
    assert finding.vmap_induced_pallas_scan is True
    assert len(report.pallas_scan_findings) == 1
    assert report.pallas_scan_findings[0] == finding
