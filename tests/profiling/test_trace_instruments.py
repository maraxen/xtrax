"""Trace loading, HLO-text glue, and CPU fusion-to-scope attribution."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from xtrax.profiling import hlo_text_for, load_trace_events
from xtrax.profiling.trace import parse_scopes, scope_map_from_hlo_text


def test_load_trace_events_concatenates_sorted_gzip_traces(tmp_path: Path):
    first = tmp_path / "plugins" / "profile" / "early"
    second = tmp_path / "plugins" / "profile" / "late"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    _write_trace(first / "a.trace.json.gz", [{"ph": "X", "name": "early"}])
    _write_trace(second / "b.trace.json.gz", [{"ph": "X", "name": "late"}])

    events = load_trace_events(tmp_path)
    assert [event["name"] for event in events] == ["early", "late"]


def test_load_trace_events_missing_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no perfetto trace"):
        load_trace_events(tmp_path)


def test_load_trace_events_rejects_payload_without_events(tmp_path: Path):
    path = tmp_path / "bad.trace.json.gz"
    with gzip.open(path, "wt") as fh:
        json.dump({"other": []}, fh)
    with pytest.raises(ValueError, match="traceEvents"):
        load_trace_events(tmp_path)


def test_hlo_text_for_jax_jit_and_filter_jit():
    def add_one(x):
        return x + 1

    jitted = jax.jit(add_one)
    x = jnp.ones((4,))
    jit_text = hlo_text_for(jitted, x)
    assert "HloModule" in jit_text

    @eqx.filter_jit
    def step(y):
        return y + 1

    compiled = step.lower(x).compile()
    assert not hasattr(compiled, "as_text")
    eqx_text = hlo_text_for(step, x)
    assert "HloModule" in eqx_text


def test_hlo_text_for_raises_on_unsupported_object():
    with pytest.raises(TypeError, match="HLO text"):
        hlo_text_for(object(), jnp.ones((2,)))

    class _NoText:
        def lower(self, *_args):
            return self

        def compile(self):
            return self

    with pytest.raises(TypeError, match="as_text"):
        hlo_text_for(_NoText(), jnp.ones((2,)))


def test_fusion_instruction_maps_to_fused_body_scope():
    """CPU traces name the fusion instruction, not the ops inside it.

    ``add_add_fusion.3`` is what ``args['hlo_op']`` carries. The labeled op
    ``add.1914`` sits inside the fused computation, and the fusion's ROOT
    (a bitcast) has no ``op_name``, so following ``calls=`` to ROOT yields
    nothing. The fusion instruction has to inherit the body's scope.
    """
    hlo = """
%fused_add (p: f32[4]) -> f32[4] {
  %add.1914 = f32[4] add(%p, %p), metadata={op_name="jit(step)/inner_a/add"}
  ROOT %bitcast.1 = f32[4] bitcast(%add.1914)
}

ENTRY %main.1 (x: f32[4]) -> (s32[], f32[4]) {
  %x = f32[4] parameter(0)
  ROOT %add_add_fusion.3 = f32[4] fusion(%x), kind=kLoop, calls=%fused_add
}
"""
    scope_map = scope_map_from_hlo_text(hlo, frozenset({"inner_a"}))
    assert scope_map["add.1914"] == "inner_a"
    assert scope_map["add_add_fusion.3"] == "inner_a"
    scopes = parse_scopes(
        [
            {
                "ph": "X",
                "name": "add_add_fusion.3",
                "dur": 100.0,
                "args": {"hlo_op": "add_add_fusion.3"},
            }
        ],
        scope_map,
    )
    assert scopes["inner_a"][1] == 1


def test_fusion_instruction_uses_own_op_name_when_body_has_none():
    fusion = (
        "  ROOT %add_add_fusion.3 = f32[4] fusion(%x), kind=kLoop, "
        'calls=%fused_add, metadata={op_name="jit(step)/inner_b/add"}'
    )
    hlo = f"""
%fused_add (p: f32[4]) -> f32[4] {{
  ROOT %bitcast.1 = f32[4] bitcast(%p)
}}

ENTRY %main.1 (x: f32[4]) -> (s32[], f32[4]) {{
  %x = f32[4] parameter(0)
{fusion}
}}
"""
    scope_map = scope_map_from_hlo_text(hlo, frozenset({"inner_b"}))
    assert scope_map["add_add_fusion.3"] == "inner_b"


def test_cpu_profiler_attributes_named_scopes(tmp_path: Path):
    """End-to-end: a tiny filter_jit step under jax.profiler.trace on CPU.

    Executed events name the fusion instruction. Before fusion-to-scope
    attribution, ``parse_scopes`` is empty.
    """

    @eqx.filter_jit
    def step(x):
        with jax.named_scope("outer"):
            y = x + x
            with jax.named_scope("inner_a"):
                y = y * y
            with jax.named_scope("inner_b"):
                y = jnp.sin(y) + y
        return y

    x = jnp.ones((8,))
    jax.block_until_ready(step(x))
    hlo = hlo_text_for(step, x)
    with jax.profiler.trace(str(tmp_path), create_perfetto_trace=True):
        jax.block_until_ready(step(x))
    events = load_trace_events(tmp_path)
    known = frozenset({"outer", "inner_a", "inner_b"})
    scopes = parse_scopes(events, scope_map_from_hlo_text(hlo, known))
    assert scopes
    assert set(scopes) <= known


def _write_trace(path: Path, events: list[dict]) -> None:
    with gzip.open(path, "wt") as fh:
        json.dump({"traceEvents": events}, fh)
