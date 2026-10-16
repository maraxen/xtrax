"""``xtrax profile``: measure one unit of work at its production shape (spec 261016 X4).

The verb runs :func:`xtrax.profiling.unit_probe.probe_unit` on a callable and
reports recompiles, dispatches per element, the device/host split, compile
seconds and wall time per unit. The steady-state flags are:

- ``steady_compile_count > 0``: the unit recompiles on every call.
- ``executions / cardinality >= EXEC_PER_ELEMENT_FLAG``: the unit dispatches
  about once per element, which is the host-loop signature. Dispatch count is
  the discriminator here; steady compiles stay at zero for a host loop.

Exit codes: 0 pass, 1 flags raised, 2 could not measure. Exit 2 is never a pass.

Inputs come from one of two sources:

- ``--shapes "x=(64,128)f32"`` builds zero-filled positional inputs.
- ``--inputs module:factory`` calls ``factory(scale)``. The factory returns
  ``(args, kwargs)``, or just ``args``.

``--extent-axis NAME`` names the scale axis. Its cardinality is the leading
dimension of the input called NAME, and the per-element metrics divide by it.
With ``--inputs`` the input must be a kwarg of the factory's result.
"""

import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from xtrax.cli.errors import CLIError
from xtrax.cli.loader import load_fn
from xtrax.cli.profile_dir import resolve_profile_dir
from xtrax.cli.shapes import parse_shapes
from xtrax.profiling.emitters import emit_probe_record
from xtrax.profiling.loop_scaling import extent_scaling_report
from xtrax.profiling.unit_probe import UnitProbeResult, probe_unit

EXIT_PASS = 0
EXIT_FLAGGED = 1
EXIT_NOT_MEASURED = 2

# PROVISIONAL and uncalibrated. The gap between host loops (>= 3.0) and vmapped
# or chunked units (<= 0.14) is wide, but X5 must calibrate this on real
# entrypoints (aminx sampler, scripts/prof_*) before it is treated as fixed.
EXEC_PER_ELEMENT_FLAG = 0.5


class _NotMeasured(Exception):
    """A measurement could not be made. The message explains why; the verb exits 2."""


@dataclass
class ProfileArgs:
    """Arguments for the ``profile`` verb.

    Args:
        fn: Import path of the unit to measure, as ``module:symbol``.
        shapes: Zero-filled inputs, ``name=(d0,...)dtype ...``. Give this or ``inputs``.
        inputs: Factory import path, ``module:factory``, called as ``factory(scale)``.
        extent_axis: Name of the scale axis input. Its leading dimension is the cardinality.
        scale: Integer passed to ``inputs``. Required with ``inputs``.
        n_steady: Traced steady calls per measurement.
        n_wall: Untraced calls for the wall-time median.
        n_units: Optional number of units. Gives an estimated total wall time.
        out: Directory for the stage-1 ProbeRecord. Empty or ``none`` disables it.
            When omitted, the resolver chain decides (see ``profile_dir``).
    """

    fn: str
    shapes: str | None = None
    inputs: str | None = None
    extent_axis: str | None = None
    scale: int | None = None
    n_steady: int = 3
    n_wall: int = 5
    n_units: int | None = None
    out: str | None = None


@dataclass
class _Measurement:
    fn: str
    platform: str
    extent_axis: str
    cardinality: int
    result: UnitProbeResult
    exec_per_element: float
    flags: list[str]
    scaling_lines: list[str]
    n_units: int | None


def run_profile(args: ProfileArgs) -> None:
    """Execute the profile verb, print a markdown report, and exit with the verdict code.

    Raises:
        SystemExit: Always, with code 0, 1 or 2 (see module docstring).
    """
    try:
        resolved, source = resolve_profile_dir(args.out)
    except ValueError as exc:
        _not_measured(f"profile dir configuration is malformed: {exc}")

    try:
        measurement = _measure(args)
    except _NotMeasured as exc:
        _not_measured(str(exc))

    if resolved is not None:
        try:
            record_path = _write_record(resolved, measurement, args)
        except Exception as exc:  # noqa: BLE001 - any write failure must surface as exit 2
            _not_measured(f"measured, but the ProbeRecord was not written: {exc}")
        record_line = f"record: {record_path}"
    else:
        record_line = "record: not written"

    flagged = bool(measurement.flags)
    code = EXIT_FLAGGED if flagged else EXIT_PASS
    print(_render(measurement, args, resolved, source, record_line, code))
    raise SystemExit(code)


def _not_measured(reason: str) -> NoReturn:
    print("verdict: could not measure (exit 2)")
    print(f"reason: {reason}", file=sys.stderr)
    raise SystemExit(EXIT_NOT_MEASURED)


def _measure(args: ProfileArgs) -> _Measurement:
    import jax

    if (args.shapes is None) == (args.inputs is None):
        raise _NotMeasured("give exactly one of --shapes or --inputs")
    if not args.extent_axis:
        raise _NotMeasured("--extent-axis NAME is required")
    if args.n_units is not None and args.n_units < 1:
        raise _NotMeasured(f"--n-units must be >= 1, got {args.n_units}")

    fn = _load(args.fn, "--fn")

    make_args: Any = None
    if args.shapes is not None:
        cardinality, call_args, kwargs, make_args = _from_shapes(args)
    else:
        call_args, kwargs, cardinality = _from_factory(args)

    try:
        result = probe_unit(fn, call_args, kwargs, n_steady=args.n_steady, n_wall=args.n_wall)
    except Exception as exc:  # noqa: BLE001 - user callable failures are exit 2, not crashes
        raise _NotMeasured(f"fn or probe failed: {type(exc).__name__}: {exc}") from exc

    if result.executions_per_unit <= 0:
        raise _NotMeasured(
            "the trace recorded zero executions per unit, so there is nothing to judge"
        )

    exec_per_element = result.executions_per_element(cardinality)
    flags: list[str] = []
    if result.steady_compile_count > 0:
        flags.append(
            f"steady recompiles: {result.steady_compile_count} in the steady window "
            "(expected 0; the unit rebuilds its executable on each call)"
        )
    if exec_per_element >= EXEC_PER_ELEMENT_FLAG:
        flags.append(
            f"exec/element {exec_per_element:.3f} >= {EXEC_PER_ELEMENT_FLAG} "
            "(provisional threshold, uncalibrated until X5; host-loop signature)"
        )

    scaling_lines = _scaling_lines(fn, make_args, cardinality, args)

    return _Measurement(
        fn=args.fn,
        platform=jax.default_backend(),
        extent_axis=args.extent_axis,
        cardinality=cardinality,
        result=result,
        exec_per_element=exec_per_element,
        flags=flags,
        scaling_lines=scaling_lines,
        n_units=args.n_units,
    )


def _load(path: str, label: str) -> Any:
    try:
        return load_fn(path)
    except CLIError as exc:
        raise _NotMeasured(f"{label} {path!r}: {exc}") from exc


def _from_shapes(args: ProfileArgs) -> tuple[int, tuple, dict, Any]:
    try:
        specs = parse_shapes(args.shapes or "")
    except CLIError as exc:
        raise _NotMeasured(f"--shapes: {exc}") from exc
    axis: str = args.extent_axis or ""
    if axis not in specs:
        raise _NotMeasured(
            f"--extent-axis {axis!r} is not an input name in --shapes (have: {', '.join(specs)})"
        )
    shape = specs[axis].shape
    if len(shape) < 1:
        raise _NotMeasured(f"extent input {axis!r} is a scalar; it needs a leading axis")
    cardinality = int(shape[0])
    if cardinality < 1:
        raise _NotMeasured(f"extent input {axis!r} has zero leading extent")

    def make_args(extent: int) -> tuple[Any, ...]:
        return _zero_args(specs, axis, extent)

    return cardinality, make_args(cardinality), {}, make_args


def _from_factory(args: ProfileArgs) -> tuple[tuple, dict, int]:
    if args.scale is None:
        raise _NotMeasured("--inputs needs --scale (the integer passed to the factory)")
    factory = _load(args.inputs or "", "--inputs")
    try:
        produced = factory(args.scale)
    except Exception as exc:  # noqa: BLE001 - factory failures are exit 2
        raise _NotMeasured(f"--inputs factory raised: {type(exc).__name__}: {exc}") from exc

    call_args, kwargs = _split_factory_output(produced)
    if args.extent_axis not in kwargs:
        raise _NotMeasured(
            f"--extent-axis {args.extent_axis!r} is not a kwarg of the factory result "
            f"(have: {', '.join(kwargs) or 'none'})"
        )
    shape = getattr(kwargs[args.extent_axis], "shape", None)
    if not shape:
        raise _NotMeasured(f"extent kwarg {args.extent_axis!r} has no leading axis")
    cardinality = int(shape[0])
    if cardinality < 1:
        raise _NotMeasured(f"extent kwarg {args.extent_axis!r} has zero leading extent")
    return call_args, kwargs, cardinality


def _split_factory_output(produced: Any) -> tuple[tuple, dict]:
    if (
        isinstance(produced, tuple)
        and len(produced) == 2
        and isinstance(produced[0], tuple)
        and isinstance(produced[1], dict)
    ):
        return produced[0], dict(produced[1])
    if isinstance(produced, tuple):
        return produced, {}
    return (produced,), {}


def _zero_args(specs: dict, extent_axis: str, extent: int) -> tuple[Any, ...]:
    import jax.numpy as jnp

    out = []
    for name, spec in specs.items():
        shape = (extent, *spec.shape[1:]) if name == extent_axis else tuple(spec.shape)
        out.append(jnp.zeros(shape, spec.dtype))
    return tuple(out)


def _scaling_lines(fn: Any, make_args: Any, cardinality: int, args: ProfileArgs) -> list[str]:
    """Loop-scaling report at ``L`` and ``2L``. Informational; it does not set the exit code."""
    if make_args is None:
        return [
            "skipped: --inputs gives kwargs the extent probe cannot pass; "
            "use --shapes to include it"
        ]
    try:
        report = extent_scaling_report(fn, make_args, cardinality)
    except Exception as exc:  # noqa: BLE001 - the report is optional; say why it is missing
        return [f"skipped: {type(exc).__name__}: {exc}"]
    lines = [f"loop bodies compared at {cardinality} and {2 * cardinality}: {len(report.findings)}"]
    for finding in report.flagged:
        lines.append(f"- flagged `{finding.path}` ({finding.primitive}): ratio {finding.ratio:.2f}")
    return lines


def _write_record(directory: Path, m: _Measurement, args: ProfileArgs) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", m.fn).strip("_")
    stamp = time.strftime("%Y%m%dT%H%M%S") + f"{time.time_ns() % 1_000_000:06d}"
    path = directory / f"profile_{slug}_{stamp}.json"
    config = {
        "fn": m.fn,
        "extent_axis": m.extent_axis,
        "n_steady": str(args.n_steady),
        "n_wall": str(args.n_wall),
    }
    if args.shapes is not None:
        config["shapes"] = args.shapes
    if args.inputs is not None:
        config["inputs"] = args.inputs
    if args.scale is not None:
        config["scale"] = str(args.scale)
    metrics: dict[str, float | int | str] = dict(m.result.as_metrics())
    metrics["exec_per_element"] = float(m.exec_per_element)
    emit_probe_record(
        path=path,
        probe_id=f"xtrax-profile:{m.fn}",
        stage=1,
        n_atoms=m.cardinality,
        platform=m.platform,
        metrics=metrics,
        config=config,
    )
    return path


def _render(
    m: _Measurement,
    args: ProfileArgs,
    resolved: Path | None,
    source: str,
    record_line: str,
    code: int,
) -> str:
    r = m.result
    lines = [
        f"## xtrax profile: `{m.fn}`",
        "",
        "| metric | value |",
        "|---|---|",
        f"| platform | {m.platform}"
        f"{' (timings are CPU-relative only)' if m.platform == 'cpu' else ''} |",
        f"| extent axis | {m.extent_axis} = {m.cardinality} |",
        f"| steady compiles (window) | {r.steady_compile_count} |",
        f"| executions / unit | {r.executions_per_unit:.2f} |",
        f"| exec/element | {m.exec_per_element:.4f} |",
        f"| jit traces / unit | {r.jit_traces_per_unit:.2f} |",
        f"| warm-up compile s | {r.warmup_compile_seconds:.3f} "
        f"({r.warmup_compile_count} compiles) |",
        f"| wall / unit s (median) | {r.wall_seconds_per_unit:.6f} |",
        f"| device s / unit | {r.device_seconds_per_unit:.6f} |",
        f"| host s / unit | {r.host_seconds_per_unit:.6f} |",
    ]
    if m.n_units is not None:
        total = r.wall_seconds_per_unit * m.n_units
        qualifier = ", CPU-relative" if m.platform == "cpu" else ""
        lines.append(f"| est. total wall s ({m.n_units} units{qualifier}) | {total:.3f} |")
    lines += [
        "",
        f"Thresholds (provisional, uncalibrated until X5): exec/element < {EXEC_PER_ELEMENT_FLAG}; "
        "steady compiles = 0.",
        "",
        "### Flags",
    ]
    lines += [f"- FLAG: {flag}" for flag in m.flags] or ["- none"]
    lines += ["", "### Extent scaling (informational)"]
    lines += [f"- {line}" for line in m.scaling_lines]
    lines += [
        "",
        f"profile dir: {resolved if resolved is not None else 'none'} (source: {source})",
        record_line,
        "",
        f"verdict: {'FLAGGED' if code == EXIT_FLAGGED else 'PASS'} (exit {code})",
    ]
    return "\n".join(lines)


__all__ = ["EXEC_PER_ELEMENT_FLAG", "ProfileArgs", "run_profile"]
