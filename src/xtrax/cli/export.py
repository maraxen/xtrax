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


def _record_export(fn: object, abstract: list, args: ExportArgs) -> None:
    """Write a kind="export" ledger row for this lowering.

    Never raises. An export is a developer-facing utility that must keep working
    even where a ledger cannot be written -- a read-only checkout, a scratch
    dir -- so unlike ``Engine.fit`` this path is fail-open. The asymmetry is
    deliberate: fit refuses to start because an unrecorded *execution* is
    unrecoverable, whereas an export can simply be run again.
    """
    from xtrax.run.ident import new_run_id
    from xtrax.telemetry.ir import capture_ir
    from xtrax.telemetry.ledger import LedgerUnavailableError, RunLedger
    from xtrax.telemetry.record import KIND_EXPORT
    from xtrax.telemetry.store import BlobStore

    context = {
        "fn": str(args.fn),
        "shapes": str(args.shapes),
        "serialized": str(bool(args.serialized)),
        "out": str(args.out) if args.out is not None else "<stdout>",
    }
    try:
        with RunLedger.open(new_run_id(), kind=KIND_EXPORT, context=context) as ledger:
            if ledger.opted_out:
                return
            refs = capture_ir(fn, *abstract, store=BlobStore(ledger.blob_root))
            ledger.record_ir(refs)
    except (LedgerUnavailableError, OSError) as exc:
        print(f"warning: could not record export telemetry ({exc})", file=sys.stderr)


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

    # Step 4b: Record the export in the run ledger.
    #
    # An export produces exactly the artifact a retrospective audit wants, and
    # until now discarded every trace of where it came from -- no run_id, no
    # manifest, no provenance. Recording it here means a StableHLO file found on
    # disk months later can be tied back to the commit, environment, and inputs
    # that produced it. kind="export" marks that nothing was executed, only
    # lowered.
    _record_export(fn, abstract, args)

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
