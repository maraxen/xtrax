"""Tests for auto-attribution of custom-call and Triton GPU kernels in trace (Debt #3196)."""

from __future__ import annotations

from xtrax.profiling.trace import scope_map_from_hlo_text


def test_custom_call_triton_with_op_name_metadata():
    """HLO custom-call with metadata op_name attributes to deepest known label."""
    hlo_text = (
        "HloModule test_triton_module\n"
        "ENTRY %main (%arg0: f32[16,16], %arg1: f32[16,16]) -> f32[16,16] {\n"
        "  %arg0 = f32[16,16]{1,0} parameter(0)\n"
        "  %arg1 = f32[16,16]{1,0} parameter(1)\n"
        "  %custom-call.1 = f32[16,16]{1,0} custom-call(%arg0, %arg1), "
        'custom_call_target="__gpu$xla.gpu.triton", '
        'backend_config={name = "matmul_kernel", num_warps = 4 : i32}, '
        'metadata={op_name="jit(model)/forward_pass/layer_dense/matmul"}\n'
        "  ROOT %tuple.1 = (f32[16,16]{1,0}) tuple(%custom-call.1)\n"
        "}\n"
    )
    known_labels = frozenset(["forward_pass", "layer_dense"])
    scopes = scope_map_from_hlo_text(hlo_text, known_labels)
    assert scopes.get("custom-call.1") == "layer_dense"


def test_custom_call_triton_kernel_name_matches_known_label():
    """HLO custom-call without op_name whose name matches known_label attributes to that label."""
    hlo_text = (
        "HloModule test_triton_module\n"
        "ENTRY %main (%arg0: f32[16,16]) -> f32[16,16] {\n"
        "  %arg0 = f32[16,16]{1,0} parameter(0)\n"
        "  %triton_kernel.42 = f32[16,16]{1,0} custom-call(%arg0), "
        'custom_call_target="__gpu$xla.gpu.triton", '
        'backend_config={name = "binned_scatter_reduce", num_warps = 4 : i32}\n'
        "  ROOT %tuple.1 = (f32[16,16]{1,0}) tuple(%triton_kernel.42)\n"
        "}\n"
    )
    known_labels = frozenset(["binned_scatter_reduce", "attention"])
    scopes = scope_map_from_hlo_text(hlo_text, known_labels)
    assert scopes.get("triton_kernel.42") == "binned_scatter_reduce"


def test_custom_call_triton_fallback_sanitized_name():
    """HLO custom-call without matching label attributes to sanitized target:kernel_name."""
    hlo_text = (
        "HloModule test_triton_module\n"
        "ENTRY %main (%arg0: f32[16,16]) -> f32[16,16] {\n"
        "  %arg0 = f32[16,16]{1,0} parameter(0)\n"
        "  %custom_call_thunk.7 = f32[16,16]{1,0} custom-call(%arg0), "
        'custom_call_target="__gpu$xla.gpu.triton", '
        'backend_config={name = "unlabeled_pallas_op"}\n'
        "  ROOT %tuple.1 = (f32[16,16]{1,0}) tuple(%custom_call_thunk.7)\n"
        "}\n"
    )
    known_labels = frozenset(["known_scope_a", "known_scope_b"])
    scopes = scope_map_from_hlo_text(hlo_text, known_labels)
    assert scopes.get("custom_call_thunk.7") == "triton:unlabeled_pallas_op"
