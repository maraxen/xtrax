"""Tests for xtrax.cli.export module (E2.2).

Tests the run_export function which traces a JAX function to StableHLO MLIR
via jax.export.export and emits the result to stdout or a file.

Format contract:
- Default: StableHLO MLIR text to stdout (or --out file).
- --serialized: flatbuffers bytes to --out file (--out is required).
"""

from __future__ import annotations

import pytest

from xtrax.cli.errors import CLIError
from xtrax.cli.export import ExportArgs, run_export


# Module-level fixture function — no decorator needed for jax.export.export.
def fn(x):
    """A bare function for export tests.

    jax.export.export just traces the function to StableHLO — no @axis_config,
    no BatchPlanner, no AmbiguousAxisError. A plain lambda or bare function works.
    """
    return x * 2


class TestExportDefaultMlir:
    """Tests for the default MLIR text output path."""

    def test_default_mlir_to_stdout(self, capsys):
        """AC1: default (no --out) emits non-empty StableHLO MLIR text to stdout.

        The MLIR module string produced by jax.export.export().mlir_module()
        should contain "module" or "stablehlo" or "@main" as StableHLO markers.
        """
        args = ExportArgs(fn="tests.cli.test_export:fn", shapes="x=(4,)f32")
        run_export(args)

        captured = capsys.readouterr()
        output = captured.out

        assert len(output) > 0, "Expected non-empty MLIR text on stdout"
        assert any(marker in output for marker in ("module", "stablehlo", "@main")), (
            f"Expected StableHLO MLIR markers in output, got: {output[:200]}"
        )

    def test_out_writes_mlir_text_file(self, tmp_path):
        """AC2: --out <path> writes MLIR text to file and prints confirmation to stderr."""
        out_path = tmp_path / "traced.mlir"
        args = ExportArgs(
            fn="tests.cli.test_export:fn",
            shapes="x=(4,)f32",
            out=str(out_path),
        )
        run_export(args)

        assert out_path.exists(), "Expected MLIR text file to be written"
        content = out_path.read_text()
        assert len(content) > 0, "Expected non-empty MLIR text file"
        assert any(marker in content for marker in ("module", "stablehlo", "@main")), (
            f"Expected StableHLO MLIR markers in file, got: {content[:200]}"
        )


class TestExportSerialized:
    """Tests for the --serialized (flatbuffers bytes) output path."""

    def test_serialized_with_out_writes_bytes_file(self, tmp_path):
        """AC3: --serialized --out <path> writes non-empty bytes file.

        Requires flatbuffers (xtrax[cli] extra). With flatbuffers installed,
        exp.serialize() should produce valid flatbuffer bytes.
        """
        pytest.importorskip("flatbuffers")  # serialize() needs the flatbuffers (cli) extra
        out_path = tmp_path / "traced.bin"
        args = ExportArgs(
            fn="tests.cli.test_export:fn",
            shapes="x=(4,)f32",
            out=str(out_path),
            serialized=True,
        )
        run_export(args)

        assert out_path.exists(), "Expected serialized bytes file to be written"
        data = out_path.read_bytes()
        assert len(data) > 0, "Expected non-empty serialized bytes file"

    def test_serialized_without_out_raises_cli_error(self):
        """AC4: --serialized without --out raises CLIError mentioning --out.

        Binary output to stdout is a footgun, so --out is required for --serialized.
        The error message must mention '--out' to guide the user.
        """
        args = ExportArgs(
            fn="tests.cli.test_export:fn",
            shapes="x=(4,)f32",
            serialized=True,
            out=None,
        )

        with pytest.raises(CLIError) as exc_info:
            run_export(args)

        error_msg = str(exc_info.value)
        assert "--out" in error_msg, (
            f"Expected '--out' in error message, got: {error_msg}"
        )
