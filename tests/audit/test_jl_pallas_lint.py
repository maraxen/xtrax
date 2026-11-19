"""Tests for JL-PALLAS static lint rules (Debt #3197)."""

from __future__ import annotations

import textwrap
from pathlib import Path

from scripts.lint_pallas import lint_pallas_source


def test_clean_source_passes():
    source = textwrap.dedent("""
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import triton as plt
        import jax.numpy as jnp

        def kernel(x):
            def body(x_ref, o_ref):
                o_ref[...] = x_ref[...] + 1.0

            return pl.pallas_call(
                body,
                out_shape=x,
                in_specs=[pl.BlockSpec((16, 32), lambda i, j: (i, j))],
                compiler_params=plt.CompilerParams(num_warps=4),
            )(x)
    """)
    findings = lint_pallas_source(source, Path("src/xtrax/op/my_kernel.py"))
    assert findings == []


def test_jl_pallas01_interpret_in_src():
    source = textwrap.dedent("""
        from jax.experimental import pallas as pl

        def kernel(x):
            return pl.pallas_call(
                body,
                out_shape=x,
                interpret=True,
            )(x)
    """)
    findings = lint_pallas_source(source, Path("src/xtrax/op/my_kernel.py"))
    rule_ids = [f["rule_id"] for f in findings]
    assert "JL-PALLAS01" in rule_ids
    finding = next(f for f in findings if f["rule_id"] == "JL-PALLAS01")
    assert finding["severity"] == "major"


def test_jl_pallas01_allowed_in_tests():
    source = textwrap.dedent("""
        from jax.experimental import pallas as pl

        def test_kernel(x):
            return pl.pallas_call(
                body,
                out_shape=x,
                interpret=True,
            )(x)
    """)
    findings = lint_pallas_source(source, Path("tests/op/test_my_kernel.py"))
    rule_ids = [f["rule_id"] for f in findings]
    assert "JL-PALLAS01" not in rule_ids


def test_jl_pallas02_missing_compiler_params():
    source = textwrap.dedent("""
        from jax.experimental import pallas as pl

        def kernel(x):
            return pl.pallas_call(
                body,
                out_shape=x,
            )(x)
    """)
    findings = lint_pallas_source(source, Path("src/xtrax/op/my_kernel.py"))
    rule_ids = [f["rule_id"] for f in findings]
    assert "JL-PALLAS02" in rule_ids
    finding = next(f for f in findings if f["rule_id"] == "JL-PALLAS02")
    assert finding["severity"] == "minor"


def test_jl_pallas03_non_power_of_two_literal():
    source = textwrap.dedent("""
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import triton as plt

        def kernel(x):
            return pl.pallas_call(
                body,
                out_shape=x,
                in_specs=[pl.BlockSpec((16, 30), lambda i, j: (i, j))],
                compiler_params=plt.CompilerParams(),
            )(x)
    """)
    findings = lint_pallas_source(source, Path("src/xtrax/op/my_kernel.py"))
    rule_ids = [f["rule_id"] for f in findings]
    assert "JL-PALLAS03" in rule_ids
    finding = next(f for f in findings if f["rule_id"] == "JL-PALLAS03")
    assert "30" in finding["message"]


def test_jl_pallas03_symbolic_and_none_dimensions_accepted():
    """Finding M4: symbolic expressions (e.g. BLOCK_SIZE) and None must not crash."""
    source = textwrap.dedent("""
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import triton as plt

        BLOCK_SIZE = 32

        def kernel(x):
            return pl.pallas_call(
                body,
                out_shape=x,
                in_specs=[
                    pl.BlockSpec(
                        block_shape=(None, BLOCK_SIZE, TILE_K // 2),
                        index_map=lambda i, j, k: (i, j, k),
                    )
                ],
                compiler_params=plt.CompilerParams(),
            )(x)
    """)
    findings = lint_pallas_source(source, Path("src/xtrax/op/my_kernel.py"))
    rule_ids = [f["rule_id"] for f in findings]
    assert "JL-PALLAS03" not in rule_ids


def test_jl_pallas04_python_control_flow_in_kernel():
    source = textwrap.dedent("""
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import triton as plt

        def my_kernel_body(x_ref, o_ref):
            if True:
                o_ref[...] = x_ref[...]
            for i in range(10):
                o_ref[...] += 1.0

        def kernel(x):
            return pl.pallas_call(
                my_kernel_body,
                out_shape=x,
                compiler_params=plt.CompilerParams(),
            )(x)
    """)
    findings = lint_pallas_source(source, Path("src/xtrax/op/my_kernel.py"))
    rule_ids = [f["rule_id"] for f in findings]
    assert "JL-PALLAS04" in rule_ids
    pallas04_findings = [f for f in findings if f["rule_id"] == "JL-PALLAS04"]
    assert len(pallas04_findings) >= 2  # 'if' and 'for'
