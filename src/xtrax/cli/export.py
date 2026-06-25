"""Export verb for xtrax CLI (E2.2).

This module provides the run_export() entry point: it loads a JAX function,
traces it to StableHLO MLIR via jax.export.export, and emits the result
in the requested format.

Format contract
---------------
- **Default (text):** StableHLO MLIR text is written to stdout, or to a file
  if ``--out`` is provided. The text is produced by ``Exported.mlir_module()``
  and is human-readable.
- **Serialized (--serialized):** flatbuffers bytes are produced by
  ``Exported.serialize()`` and written to ``--out`` (required; binary output
  to a terminal is a footgun). Requires the ``xtrax[cli]`` extra
  (``pip install xtrax[cli]``) which pulls in ``flatbuffers>=2``.

Note: ``jax.export.export`` just *traces* the function to StableHLO — it does
NOT invoke ``infer_bundle``, ``BatchPlanner``, or any xtrax-specific machinery.
A plain undecorated function works fine; no ``@axis_config`` is required.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import jax

from xtrax.cli.errors import CLIError
from xtrax.cli.loader import load_fn
from xtrax.cli.shapes import parse_shapes


@dataclass
class ExportArgs:
    """Arguments for the export verb.

    Attributes:
        fn: Import-path string to the function to trace.
            Format: ``'module.path:symbol'``.
        shapes: Space-separated shape specification string.
            Format: ``'name=(d0,d1,...)<dtype> ...'``.
            Each name corresponds to a positional argument of the function, in order.
        out: Optional file path for the output. Contracts:
            - Default (text): if ``out`` is None, MLIR text is printed to stdout;
              if ``out`` is given, the file is written and a confirmation line is
              printed to stderr.
            - Serialized (``--serialized``): ``out`` is REQUIRED. Binary output
              to a terminal is a footgun. If ``out`` is None, a ``CLIError`` is
              raised directing the user to supply ``--out``.
        serialized: If True, emit flatbuffers bytes via ``Exported.serialize()``
            instead of MLIR text. Requires ``flatbuffers>=2``
            (``pip install xtrax[cli]``).
    """

    fn: str
    shapes: str = ""
    out: str | None = field(default=None)
    serialized: bool = field(default=False)


def run_export(args: ExportArgs) -> None:
    """Execute the export verb: trace to StableHLO and emit in the requested format.

    Pipeline:
    1. Load the function from the import-path string via ``load_fn``.
    2. Parse the shapes string into ``ShapeDtypeStruct`` objects via ``parse_shapes``.
    3. Build abstract inputs as an ordered list of ``ShapeDtypeStruct`` values.
    4. Call ``jax.export.export(jax.jit(fn))(*abstract_inputs)`` to produce
       an ``Exported`` object (StableHLO trace, no xtrax inference machinery).
    5. Emit in the requested format:
       - Default: ``Exported.mlir_module()`` → text → stdout or ``--out`` file.
       - Serialized: ``Exported.serialize()`` → bytes → ``--out`` file (required).

    Args:
        args: ExportArgs with fn, shapes, out, and serialized.

    Raises:
        CLIError: On any user-facing error:
            - Bad import path or bad shapes (from load_fn / parse_shapes).
            - ``--serialized`` without ``--out`` (binary to stdout is a footgun).
            - Missing ``flatbuffers`` package when ``--serialized`` is requested.
              Install with ``pip install xtrax[cli]`` (adds ``flatbuffers>=2``).

    Example:
        >>> args = ExportArgs(fn="mylib:forward", shapes="x=(4,3)f32")
        >>> run_export(args)  # Prints StableHLO MLIR text to stdout

        >>> args = ExportArgs(
        ...     fn="mylib:forward",
        ...     shapes="x=(4,3)f32",
        ...     out="traced.mlir",
        ... )
        >>> run_export(args)  # Writes MLIR text to traced.mlir

        >>> args = ExportArgs(
        ...     fn="mylib:forward",
        ...     shapes="x=(4,3)f32",
        ...     out="traced.bin",
        ...     serialized=True,
        ... )
        >>> run_export(args)  # Writes flatbuffers bytes to traced.bin
    """
    # Step 1: Load the function from the import-path string.
    fn = load_fn(args.fn)

    # Step 2: Parse shape string to ShapeDtypeStruct dict.
    parsed = parse_shapes(args.shapes)

    # Step 3: Build ordered abstract inputs list (positional order matches fn args).
    abstract = list(parsed.values())

    # Step 4: Trace to StableHLO via jax.export.export.
    # jax.jit wraps the fn; export() takes abstract inputs (ShapeDtypeStruct).
    # This is pure tracing — no xtrax inference machinery, no AmbiguousAxisError.
    exp = jax.export.export(jax.jit(fn))(*abstract)

    # Step 5: Emit in the requested format.
    if args.serialized:
        # --serialized requires --out (binary to stdout is a footgun).
        if args.out is None:
            raise CLIError(
                "--serialized requires --out <path> (binary). "
                "Writing binary flatbuffers to stdout is unsupported."
            )

        # Serialize to flatbuffers bytes. Requires flatbuffers>=2 (xtrax[cli] extra).
        try:
            data = exp.serialize()
        except (ImportError, ModuleNotFoundError) as exc:
            raise CLIError("--serialized requires flatbuffers: pip install xtrax[cli]") from exc

        # Write bytes to the output file.
        with open(args.out, "wb") as f:
            f.write(data)
        print(f"wrote {args.out}", file=sys.stderr)

    else:
        # Default: emit StableHLO MLIR text.
        text = exp.mlir_module()

        if args.out is not None:
            # Write text to file and confirm to stderr.
            with open(args.out, "w") as f:
                f.write(text)
            print(f"wrote {args.out}", file=sys.stderr)
        else:
            # Print to stdout.
            print(text)


__all__ = ["ExportArgs", "run_export"]
