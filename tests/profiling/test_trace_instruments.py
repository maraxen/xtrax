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
from xtrax.profiling.trace import parse_hlo_op_times, parse_scopes, scope_map_from_hlo_text


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


def test_fusion_own_op_name_when_callee_computation_is_absent():
    """Own ``op_name`` applies when the callee was never split out.

    An inline fused body closes the parent at the inner ``}``, so the callee
    is absent from ``computations`` and the fusion instruction is not an
    entry in that split. The unlabeled body contributes no votes.
    """
    hlo = """
ENTRY %main.1 (x: f32[4]) -> f32[4] {
  %body (p: f32[4]) -> f32[4] {
    ROOT %bitcast.1 = f32[4] bitcast(%p)
  }
  %x = f32[4] parameter(0)
  ROOT %fuse.1 = f32[4] fusion(%x), kind=kLoop, calls=%body, metadata={op_name="inner_b/add"}
}
"""
    scope_map = scope_map_from_hlo_text(hlo, frozenset({"inner_b"}))
    assert scope_map["fuse.1"] == "inner_b"


def test_fusion_body_majority_label():
    """Two ``scope_a`` votes and one ``scope_b`` vote resolve to ``scope_a``."""
    hlo = """
%fused_add (p: f32[4]) -> f32[4] {
  %add.1 = f32[4] add(%p, %p), metadata={op_name="scope_a/add"}
  %add.2 = f32[4] add(%p, %p), metadata={op_name="scope_a/mul"}
  ROOT %add.3 = f32[4] add(%p, %p), metadata={op_name="scope_b/add"}
}

ENTRY %main.1 (x: f32[4]) -> f32[4] {
  %x = f32[4] parameter(0)
  ROOT %add_add_fusion.3 = f32[4] fusion(%x), kind=kLoop, calls=%fused_add
}
"""
    scope_map = scope_map_from_hlo_text(hlo, frozenset({"scope_a", "scope_b"}))
    assert scope_map["add_add_fusion.3"] == "scope_a"
    assert scope_map["fused_add"] == "scope_a"


def test_fusion_body_tie_keeps_earliest_inserted_label():
    """A three-way tie keeps the earliest label, ``scope_m``.

    ``scope_a`` is the lexicographic minimum and ``scope_z`` the maximum,
    so either name-based tie break misses ``scope_m``.
    """
    hlo = """
%fused_tie (p: f32[4]) -> f32[4] {
  %m.1 = f32[4] add(%p, %p), metadata={op_name="scope_m/add"}
  %a.1 = f32[4] add(%p, %p), metadata={op_name="scope_a/add"}
  ROOT %z.1 = f32[4] add(%p, %p), metadata={op_name="scope_z/add"}
}

ENTRY %main.1 (x: f32[4]) -> f32[4] {
  %x = f32[4] parameter(0)
  ROOT %tie_fusion.1 = f32[4] fusion(%x), kind=kLoop, calls=%fused_tie
}
"""
    known = frozenset({"scope_m", "scope_a", "scope_z"})
    scope_map = scope_map_from_hlo_text(hlo, known)
    assert scope_map["tie_fusion.1"] == "scope_m"
    assert scope_map["fused_tie"] == "scope_m"

    inline = """
ENTRY %main.1 (x: f32[4]) -> f32[4] {
  %fused_tie (p: f32[4]) -> f32[4] {
    %m.1 = f32[4] add(%p, %p), metadata={op_name="scope_m/add"}
    %a.1 = f32[4] add(%p, %p), metadata={op_name="scope_a/add"}
    ROOT %z.1 = f32[4] add(%p, %p), metadata={op_name="scope_z/add"}
  }
  %x = f32[4] parameter(0)
  ROOT %tie_fusion.2 = f32[4] fusion(%x), kind=kLoop, calls=%fused_tie
}
"""
    inline_map = scope_map_from_hlo_text(inline, known)
    assert inline_map["tie_fusion.2"] == "scope_m"
    assert inline_map["fused_tie"] == "scope_m"


def test_entry_tuple_return_instruction_resolves():
    """An ENTRY instruction whose return type is a tuple still maps."""
    hlo = """
ENTRY %main.1 (x: f32[4]) -> (s32[], f32[4]) {
  %x = f32[4] parameter(0)
  ROOT %add.1 = f32[4] add(%x, %x), metadata={op_name="inner_a/add"}
}
"""
    scope_map = scope_map_from_hlo_text(hlo, frozenset({"inner_a"}))
    assert scope_map["add.1"] == "inner_a"


def test_fusion_uses_block_votes_when_callee_is_absent():
    """Inline fused bodies are invisible to computation splitting.

    Brace-stack votes still name the callee. Two ``scope_a`` instructions
    and one ``scope_b`` instruction both win for the fusion and the block.
    """
    hlo = """
ENTRY %main.1 (x: f32[4]) -> f32[4] {
  %fused_add (p: f32[4]) -> f32[4] {
    %add.1 = f32[4] add(%p, %p), metadata={op_name="scope_a/add"}
    %add.2 = f32[4] add(%p, %p), metadata={op_name="scope_a/mul"}
    ROOT %add.3 = f32[4] add(%p, %p), metadata={op_name="scope_b/add"}
  }
  %x = f32[4] parameter(0)
  ROOT %loop_fusion.1 = f32[4] fusion(%x), kind=kLoop, calls=%fused_add
}
"""
    scope_map = scope_map_from_hlo_text(hlo, frozenset({"scope_a", "scope_b"}))
    assert scope_map["loop_fusion.1"] == "scope_a"
    assert scope_map["fused_add"] == "scope_a"


def test_cpu_profiler_attributes_named_scopes(tmp_path: Path):
    """The traced fusion thunk takes the fused body's majority scope.

    Three ``inner_a`` ops and a trailing ``inner_b`` root fuse into one
    thunk. That thunk's own ``op_name`` is the root (``inner_b``); the
    body's majority is ``inner_a``.
    """

    @eqx.filter_jit
    def step(x):
        with jax.named_scope("inner_a"):
            y = x + x
            y = y * y
            y = jnp.sin(y)
        with jax.named_scope("inner_b"):
            y = y + y
        return y

    x = jnp.ones((8,))
    jax.block_until_ready(step(x))
    hlo = hlo_text_for(step, x)
    with jax.profiler.trace(str(tmp_path), create_perfetto_trace=True):
        jax.block_until_ready(step(x))
    events = load_trace_events(tmp_path)
    known = frozenset({"inner_a", "inner_b"})
    scope_map = scope_map_from_hlo_text(hlo, known)
    fusion_ops = [op for op in parse_hlo_op_times(events) if "fusion" in op]
    assert fusion_ops
    for op in fusion_ops:
        assert op in scope_map
        assert scope_map[op] == "inner_a"
    assert set(parse_scopes(events, scope_map)) == {"inner_a"}


def _write_trace(path: Path, events: list[dict]) -> None:
    with gzip.open(path, "wt") as fh:
        json.dump({"traceEvents": events}, fh)
