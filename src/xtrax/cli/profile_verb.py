"""``xtrax profile``: measure one unit of work at its production shape (spec 261016 X4, X6, X9).

The verb runs :func:`xtrax.profiling.unit_probe.probe_unit` on a callable and
reports recompiles, dispatches per element, the device/host split, compile
seconds and wall time per unit. The steady-state flags are:

- ``steady_compile_count > 0``: the unit recompiles on every call.
- ``executions / cardinality >= EXEC_PER_ELEMENT_FLAG``: the unit dispatches
  about once per element, which is the host-loop signature. Dispatch count is
  the discriminator here; steady compiles stay at zero for a host loop.

Exit codes: 0 pass, 1 flags raised (or, with ``--check``, a stale record),
2 could not measure. Exit 2 is never a pass.

Inputs come from one of two sources:

- ``--shapes "x=(64,128)f32"`` builds zero-filled positional inputs.
- ``--inputs module:factory`` calls ``factory(scale)``. The factory returns
  ``(args, kwargs)``, or just ``args``.

``--extent-axis NAME`` names the scale axis. Its cardinality is the leading
dimension of the input called NAME, and the per-element metrics divide by it.
With ``--inputs`` the input must be a kwarg of the factory's result.

Fit and extrapolation (X6). Before measuring, the verb lowers and compiles the
unit on shape structs and reads the device memory limit (``fits: yes / no /
unknown``). A unit that does not fit at the measured extent is refused (exit 2),
even with ``--target-extent``. ``--target-extent N`` only extrapolates a unit that
fits where it is measured: the wall estimate is linear in N, and the verb also
runs the fit check at N (``fits at target extent``). A target that does not fit
withholds the estimate (``does not fit at target extent``). An unknown target fit
is reported as unknown. The estimate is printed only if
``assert_claim_supported(END_TO_END)`` accepts it (scale ratio within
``SCALE_EXTRAPOLATION_LIMIT``, clean git sha); otherwise the line reads
``wall estimate withheld``. Withholding never changes the exit code: the dispatch
ratio and compile count are scale-free.

Record check (X9). ``--check PATH --sha REV`` reads a ProbeRecord instead of
measuring. Exit 0 only when the record is internally consistent, its sha matches
``REV`` and its flags pass. Exit 1 when it is stale or flagged. Exit 2 when it is
unreadable, lacks or has invalid metrics, is internally inconsistent, has an
unverifiable or unresolvable sha, or when ``--sha`` is omitted and no flag was
raised (freshness is unknown, so the record cannot pass). ``--check`` and ``--fn``
are mutually exclusive.
"""

import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

from xtrax.cli.errors import CLIError
from xtrax.cli.loader import load_fn
from xtrax.cli.profile_dir import resolve_profile_dir
from xtrax.cli.shapes import parse_shapes
from xtrax.profiling.claims import (
    ClaimClass,
    ClaimValidityError,
    _reject_unverifiable_git_sha,
    assert_claim_supported,
)
from xtrax.profiling.emitters import emit_probe_record
from xtrax.profiling.fit import FitCheck, fit_precheck
from xtrax.profiling.loop_scaling import extent_scaling_report
from xtrax.profiling.record import ProbeRecord
from xtrax.profiling.rules import EXEC_PER_ELEMENT_FLAG, THRESHOLD_STATUS, steady_state_flags
from xtrax.profiling.unit_probe import UnitProbeResult, probe_unit

EXIT_PASS = 0
EXIT_FLAGGED = 1
EXIT_NOT_MEASURED = 2


class _NotMeasured(Exception):
    """A measurement could not be made. The message explains why; the verb exits 2.

    ``fit_line`` carries the fit-check row when the fit was already decided, so the
    report can show it even though nothing was measured.
    """

    def __init__(self, message: str, *, fit_line: str | None = None) -> None:
        super().__init__(message)
        self.fit_line = fit_line


@dataclass
class ProfileArgs:
    """Arguments for the ``profile`` verb.

    Args:
        fn: Import path of the unit to measure, as ``module:symbol``. Exactly one of
            ``fn`` or ``check``.
        shapes: Zero-filled inputs, ``name=(d0,...)dtype ...``. Give this or ``inputs``.
        inputs: Factory import path, ``module:factory``, called as ``factory(scale)``.
        extent_axis: Name of the scale axis input. Its leading dimension is the cardinality.
        scale: Integer passed to ``inputs``. Required with ``inputs``.
        n_steady: Traced steady calls per measurement.
        n_wall: Untraced calls for the wall-time median.
        n_units: Optional number of units. Gives an estimated total wall time.
        out: Directory for the stage-1 ProbeRecord. Empty or ``none`` disables it.
            When omitted, the resolver chain decides (see ``profile_dir``).
        target_extent: Production extent of the scale axis. The unit must fit at the
            measured extent. The wall estimate is extrapolated linearly to this extent,
            and withheld when the unit does not fit here. Flags: ``--target-extent``.
        check: Path of a ProbeRecord to check instead of measuring. Flags: ``--check``.
        sha: Revision the ``check`` record must match (``HEAD``, a sha, ...). Only with
            ``check``. Flags: ``--sha``.
    """

    fn: str | None = None
    shapes: str | None = None
    inputs: str | None = None
    extent_axis: str | None = None
    scale: int | None = None
    n_steady: int = 3
    n_wall: int = 5
    n_units: int | None = None
    out: str | None = None
    target_extent: int | None = None
    check: str | None = None
    sha: str | None = None


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
    metrics: dict[str, float] = field(default_factory=dict)
    config: dict[str, str] = field(default_factory=dict)
    fit: FitCheck | None = None
    target_extent: int | None = None
    extrapolation_lines: list[str] = field(default_factory=list)


def run_profile(args: ProfileArgs) -> None:
    """Execute the profile verb, print a markdown report, and exit with the verdict code.

    Raises:
        SystemExit: Always, with code 0, 1 or 2 (see module docstring).
    """
    if (args.fn is None) == (args.check is None):
        _not_measured("give exactly one of --fn or --check")
    if args.sha is not None and args.check is None:
        _not_measured("--sha is only valid with --check")
    if args.check is not None:
        _check_record(args)

    try:
        resolved, source = resolve_profile_dir(args.out)
    except ValueError as exc:
        _not_measured(f"profile dir configuration is malformed: {exc}")

    try:
        measurement = _measure(args)
    except _NotMeasured as exc:
        if exc.fit_line is not None:
            print(exc.fit_line)
        _not_measured(str(exc))
    except Exception as exc:  # noqa: BLE001 - any failure to measure is exit 2, never exit 1
        # Exit 1 means a flag or a stale record. An uncaught exception would exit 1
        # through the interpreter, so every failure in the measure path lands here.
        _not_measured(f"measurement raised {type(exc).__name__}: {exc}")

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


# --- --check (X9) ----------------------------------------------------------


def _check_record(args: ProfileArgs) -> NoReturn:
    path = args.check or ""
    try:
        record = ProbeRecord.read(path)
    except Exception as exc:  # noqa: BLE001 - any unreadable record is exit 2
        _not_measured(f"could not read ProbeRecord {path!r}: {type(exc).__name__}: {exc}")

    try:
        assert_claim_supported([record], ClaimClass.DISPATCH_COUNT)
        _reject_unverifiable_git_sha([record], ClaimClass.DISPATCH_COUNT)
    except ClaimValidityError as exc:
        _not_measured(f"record cannot back a dispatch check: {exc}")

    values: dict[str, float] = {}
    for name in ("n_executions", "exec_per_element", "n_compilations"):
        value = record.metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _not_measured(f"record lacks the numeric metric {name}")
        values[name] = float(value)
    for name, value in values.items():
        if value < 0:
            _not_measured(f"record metric {name} is negative ({value}); not a measurement")
    if values["n_executions"] == 0:
        _not_measured("record shows zero executions; not a measurement")
    n_executions = values["n_executions"]
    exec_per_element = values["exec_per_element"]
    n_compilations = values["n_compilations"]

    # run_profile writes exec_per_element = n_executions / n_atoms; a record that
    # disagrees with itself was edited or corrupted, and cannot back a verdict.
    if record.n_atoms <= 0:
        _not_measured(f"record n_atoms must be > 0, got {record.n_atoms}")
    expected_exec = n_executions / record.n_atoms
    if not math.isclose(exec_per_element, expected_exec, rel_tol=1e-6, abs_tol=1e-12):
        _not_measured(
            "record metrics are inconsistent: exec_per_element "
            f"{exec_per_element} != n_executions / n_atoms = {expected_exec} "
            f"(n_executions {n_executions}, n_atoms {record.n_atoms})"
        )

    stale: list[str] = []
    if args.sha is None:
        sha_line = "sha check: not performed (no --sha given)"
    else:
        try:
            resolved = _resolve_sha(args.sha)
        except _NotMeasured as exc:
            _not_measured(str(exc))
        record_full = _full_record_sha(record.git_sha)
        sha_line = f"--sha {args.sha} resolves to {resolved}"
        if record_full != resolved:
            stale.append(
                f"STALE: record git_sha {record.git_sha} differs from {args.sha} ({resolved})"
            )

    flags = [f"FLAG: {flag}" for flag in steady_state_flags(exec_per_element, n_compilations)]

    findings = stale + flags
    if args.sha is None and not findings:
        _not_measured(
            "no --sha given, so freshness is unknown and the record cannot pass; "
            "pass --sha <rev> to check freshness"
        )
    labels = []
    if stale:
        labels.append("STALE")
    if flags:
        labels.append("FLAGGED")
    code = EXIT_FLAGGED if findings else EXIT_PASS
    verdict = ", ".join(labels) if labels else "PASS"
    lines = [
        f"## xtrax profile --check: `{path}`",
        "",
        "| record | value |",
        "|---|---|",
        f"| probe_id | {record.probe_id} |",
        f"| record git_sha | {record.git_sha} |",
        f"| sha check | {sha_line} |",
        f"| n_atoms | {record.n_atoms} |",
        f"| executions | {n_executions:.2f} |",
        f"| exec/element | {exec_per_element:.4f} |",
        f"| steady compiles / unit | {n_compilations:.4f} |",
        "",
        "### Findings",
    ]
    lines += [f"- {finding}" for finding in findings] or ["- none"]
    lines += ["", f"verdict: {verdict} (exit {code})"]
    print("\n".join(lines))
    raise SystemExit(code)


_FULL_SHA = re.compile(r"[0-9a-f]{40}")
_HEX_SHA = re.compile(r"[0-9a-f]{4,40}")


def _full_record_sha(git_sha: str) -> str:
    """The record's commit as a full sha. A short sha is resolved in the cwd repository."""
    if _FULL_SHA.fullmatch(git_sha):
        return git_sha
    if not _HEX_SHA.fullmatch(git_sha):
        _not_measured(f"record git_sha {git_sha!r} is not a commit sha (unknown sha)")
    try:
        return _resolve_sha(git_sha)
    except _NotMeasured as exc:
        _not_measured(f"record git_sha {git_sha!r} does not resolve (unknown sha): {exc}")


def _resolve_sha(sha: str) -> str:
    cwd = Path.cwd()
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise _NotMeasured(f"cannot run git to resolve --sha {sha!r}: {exc}") from exc
    resolved = out.stdout.strip()
    if out.returncode != 0 or not resolved:
        raise _NotMeasured(
            f"cannot resolve --sha {sha!r} in {cwd} (not a git checkout, or no such revision)"
        )
    return resolved


# --- measurement -----------------------------------------------------------


def _measure(args: ProfileArgs) -> _Measurement:
    import jax

    if (args.shapes is None) == (args.inputs is None):
        raise _NotMeasured("give exactly one of --shapes or --inputs")
    if not args.extent_axis:
        raise _NotMeasured("--extent-axis NAME is required")
    if args.n_units is not None and args.n_units < 1:
        raise _NotMeasured(f"--n-units must be >= 1, got {args.n_units}")
    if args.target_extent is not None and args.target_extent < 1:
        raise _NotMeasured(f"--target-extent must be >= 1, got {args.target_extent}")

    fn = _load(args.fn or "", "--fn")

    make_args: Any = None
    if args.shapes is not None:
        cardinality, specs, axis = _from_shapes(args)
        fit = fit_precheck(fn, _shape_structs(specs, axis, cardinality), {})
        fit_line = _fit_line(fit)
        _refuse_if_unfit(fit, fit_line, cardinality)
        call_args: tuple = _zero_args(specs, axis, cardinality)
        kwargs: dict = {}
        make_args = lambda extent: _zero_args(specs, axis, extent)  # noqa: E731
        at_extent = lambda extent: (_shape_structs(specs, axis, extent), {})  # noqa: E731
    else:
        call_args, kwargs, cardinality = _from_factory(args)
        fit = fit_precheck(fn, call_args, kwargs)
        fit_line = _fit_line(fit)
        _refuse_if_unfit(fit, fit_line, cardinality)
        at_extent = lambda extent: _factory_at_extent(  # noqa: E731
            call_args, kwargs, args.extent_axis or "", extent
        )

    target_fit: FitCheck | None = None
    if args.target_extent is not None:
        target_args, target_kwargs = at_extent(args.target_extent)
        target_fit = fit_precheck(fn, target_args, target_kwargs)

    try:
        result = probe_unit(fn, call_args, kwargs, n_steady=args.n_steady, n_wall=args.n_wall)
    except Exception as exc:  # noqa: BLE001 - user callable failures are exit 2, not crashes
        raise _NotMeasured(f"fn or probe failed: {type(exc).__name__}: {exc}") from exc

    if result.executions_per_unit <= 0:
        raise _NotMeasured(
            "the trace recorded zero executions per unit, so there is nothing to judge"
        )

    exec_per_element = result.executions_per_element(cardinality)
    flags = steady_state_flags(
        float(exec_per_element),
        float(result.steady_compile_count) / result.n_steady,
    )

    scaling_lines = _scaling_lines(fn, make_args, cardinality, args)

    platform = jax.default_backend()
    metrics: dict[str, float] = dict(result.as_metrics())
    metrics["exec_per_element"] = float(exec_per_element)
    if args.target_extent is not None:
        metrics["target_extent"] = float(args.target_extent)
    config = _config(args)

    extrapolation_lines = _extrapolation_lines(
        args, cardinality, result, platform, metrics, config, target_fit
    )

    return _Measurement(
        fn=args.fn or "",
        platform=platform,
        extent_axis=args.extent_axis,
        cardinality=cardinality,
        result=result,
        exec_per_element=exec_per_element,
        flags=flags,
        scaling_lines=scaling_lines,
        n_units=args.n_units,
        metrics=metrics,
        config=config,
        fit=fit,
        target_extent=args.target_extent,
        extrapolation_lines=extrapolation_lines,
    )


def _fit_line(fit: FitCheck) -> str:
    if fit.fits is True:
        return f"fits: yes ({fit.reason})"
    if fit.fits is False:
        return f"fits: no ({fit.reason})"
    return f"fits: unknown ({fit.reason})"


def _refuse_if_unfit(fit: FitCheck, fit_line: str, cardinality: int) -> None:
    # --target-extent never rescues a unit that does not fit where it is measured.
    if fit.fits is False:
        raise _NotMeasured(
            f"the unit does not fit device memory at measured extent {cardinality} "
            f"({fit.reason}); --target-extent only extrapolates a unit that fits, so "
            "re-run at a smaller extent and set --target-extent to the production extent",
            fit_line=fit_line,
        )


def _factory_at_extent(
    call_args: tuple, kwargs: dict, extent_axis: str, extent: int
) -> tuple[tuple, dict]:
    """The factory's inputs with the extent kwarg's leading axis replaced, as structs."""
    import jax

    leaf = kwargs[extent_axis]
    shape = tuple(leaf.shape)
    out = dict(kwargs)
    out[extent_axis] = jax.ShapeDtypeStruct((extent, *shape[1:]), leaf.dtype)
    return call_args, out


def _target_fit_lines(target: int, target_fit: FitCheck | None) -> list[str]:
    if target_fit is None:
        return []
    if target_fit.fits is True:
        return [f"fits at target extent: yes at {target} ({target_fit.reason})"]
    if target_fit.fits is False:
        return [f"does not fit at target extent {target}: {target_fit.reason}"]
    return [f"fits at target extent: unknown at {target} ({target_fit.reason})"]


def _extrapolation_lines(
    args: ProfileArgs,
    cardinality: int,
    result: UnitProbeResult,
    platform: str,
    metrics: dict[str, float],
    config: dict[str, str],
    target_fit: FitCheck | None,
) -> list[str]:
    target = args.target_extent
    if target is None:
        return []
    fit_lines = _target_fit_lines(target, target_fit)
    if target_fit is not None and target_fit.fits is False:
        return [
            *fit_lines,
            f"wall estimate withheld: the unit does not fit at extent {target}",
        ]
    if target <= cardinality:
        return [
            *fit_lines,
            f"target extent {target} <= measured extent {cardinality}: "
            "no extrapolation; wall is reported at the measured extent",
        ]
    # The in-memory record is what the claim is checked against; it carries the same
    # metrics the verb writes, so the claim and the artifact cannot diverge.
    record = ProbeRecord(
        probe_id=f"xtrax-profile:{args.fn}",
        stage=1,
        n_atoms=cardinality,
        platform=platform,
        metrics=dict(metrics),
        config=config,
    )
    label = "extrapolated, CPU-relative" if platform == "cpu" else "extrapolated"
    try:
        assert_claim_supported([record], ClaimClass.END_TO_END, target_n_atoms=target)
    except ClaimValidityError as exc:
        return [*fit_lines, f"wall estimate withheld: {exc}"]

    ratio = target / cardinality
    per_unit = result.wall_seconds_per_unit * ratio
    lines = [
        *fit_lines,
        f"est. wall per unit s at extent {target} ({label}, linear x{ratio:.2f}): {per_unit:.6f}",
    ]
    if args.n_units is not None:
        lines.append(
            f"est. total wall s ({args.n_units} units at extent {target}, {label}): "
            f"{per_unit * args.n_units:.3f}"
        )
    return lines


def _config(args: ProfileArgs) -> dict[str, str]:
    config = {
        "fn": args.fn or "",
        "extent_axis": args.extent_axis or "",
        "n_steady": str(args.n_steady),
        "n_wall": str(args.n_wall),
    }
    if args.shapes is not None:
        config["shapes"] = args.shapes
    if args.inputs is not None:
        config["inputs"] = args.inputs
    if args.scale is not None:
        config["scale"] = str(args.scale)
    return config


def _load(path: str, label: str) -> Any:
    try:
        return load_fn(path)
    except CLIError as exc:
        raise _NotMeasured(f"{label} {path!r}: {exc}") from exc


def _from_shapes(args: ProfileArgs) -> tuple[int, dict, str]:
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
    return cardinality, specs, axis


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


def _shape_structs(specs: dict, extent_axis: str, extent: int) -> tuple[Any, ...]:
    import jax

    out = []
    for name, spec in specs.items():
        shape = (extent, *spec.shape[1:]) if name == extent_axis else tuple(spec.shape)
        out.append(jax.ShapeDtypeStruct(shape, spec.dtype))
    return tuple(out)


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
    emit_probe_record(
        path=path,
        probe_id=f"xtrax-profile:{m.fn}",
        stage=1,
        n_atoms=m.cardinality,
        platform=m.platform,
        metrics=dict(m.metrics),
        config=dict(m.config),
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
        f"Thresholds ({THRESHOLD_STATUS}): exec/element < {EXEC_PER_ELEMENT_FLAG}; "
        "steady compiles = 0.",
        "",
        "### Fit",
        _fit_line(m.fit) if m.fit is not None else "fits: unknown (not checked)",
        "",
        "### Extrapolation",
    ]
    lines += [f"- {line}" for line in m.extrapolation_lines] or [
        "- none (measured at production extent)"
    ]
    lines += [
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
