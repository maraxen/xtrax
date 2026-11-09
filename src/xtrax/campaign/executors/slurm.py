"""Pure host SLURM array submission script generator and executor.

Covering [T3-04], GATE-08, GATE-09, and Section 6.3.3 of SPEC-261105-CAMPAIGN-RUNNER.
Zero imports of controller/ or bathos.
"""

import importlib.util
import os
import re
import shlex
import site
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xtrax.campaign.executors.protocol import CampaignSummary

if TYPE_CHECKING:
    from xtrax.campaign.manifest import CampaignManifest, ManifestRow


def _validate_row_indices(
    manifest: "CampaignManifest",
    row_indices: list[int] | None,
) -> list[int]:
    """Return deduplicated row indices in manifest order, or raise ``ValueError``."""
    n_rows = len(manifest.rows)
    if row_indices is None:
        return list(range(n_rows))
    valid_range = range(n_rows)
    selected: set[int] = set()
    for idx in row_indices:
        if idx not in valid_range:
            msg = (
                f"Row index {idx} is out of range for manifest with {n_rows} row(s) "
                f"(expected 0..{n_rows - 1})"
                if n_rows > 0
                else f"Row index {idx} is out of range for empty manifest"
            )
            raise ValueError(msg)
        selected.add(idx)
    return sorted(selected)


def get_pending_row_indices(
    manifest: "CampaignManifest",
    output_dir: Path | str,
    resume_verify: str = "deep",
    row_indices: list[int] | None = None,
) -> list[int]:
    """Identify uncompleted row indices eligible for scheduling.

    Per Section 6.3.3 of SPEC-261105-CAMPAIGN-RUNNER. A row counts as completed only
    when it passes the same resume verification the local executors use to skip it
    (matching done marker, a present, readable store, and by default a matching content
    digest), so a deleted or corrupt store is rescheduled rather than trusted because a
    marker file survived. Each row skipped as done is recorded in ``reuse_log.jsonl``.
    """
    from xtrax.campaign.executors.local import (
        _check_resume_done,
        _record_reuse,
        _resolve_output_path,
        _validate_resume_verify,
    )

    mode = _validate_resume_verify(resume_verify)
    candidate_indices = _validate_row_indices(manifest, row_indices)
    out_dir = Path(output_dir)
    pending: list[int] = []
    for idx in candidate_indices:
        row = manifest.rows[idx]
        if _check_resume_done(out_dir, row, resume_verify=mode):
            _record_reuse(
                out_dir,
                row,
                _resolve_output_path(out_dir, row),
                mode,
                reused_by="SlurmArrayExecutor.get_pending",
            )
        else:
            pending.append(idx)
    return pending


#: SLURM's default ``MaxArraySize``; array task IDs must be strictly below it.
DEFAULT_MAX_ARRAY_SIZE = 1001


def format_slurm_array_indices(
    indices: list[int],
    max_concurrent: int | None = None,
) -> str:
    """Format arbitrary sparse row indices into compact SLURM --array range string.

    Per Section 6.3.3 of SPEC-261105-CAMPAIGN-RUNNER:
    e.g. [12, 15, 44..50] with max_concurrent=20 -> '12,15,44-50%20'.
    """
    if not indices:
        return ""
    sorted_indices = sorted(set(indices))
    ranges: list[str] = []
    start = sorted_indices[0]
    end = start

    for idx in sorted_indices[1:]:
        if idx == end + 1:
            end = idx
        else:
            ranges.append(f"{start}-{end}" if start != end else f"{start}")
            start = end = idx

    ranges.append(f"{start}-{end}" if start != end else f"{start}")
    spec = ",".join(ranges)
    return f"{spec}%{max_concurrent}" if max_concurrent else spec


def _sbatch_log_path(log_path: str, base_dir: str) -> str:
    """Return ``log_path`` as written in an ``#SBATCH --output/--error`` directive.

    A relative log path is anchored under ``base_dir`` like :func:`_anchor_log_path`, but
    any ``%`` in ``base_dir`` is escaped as ``%%``: sbatch reads ``%`` as a filename
    pattern (``%j``, ``%A``, ``%a``), and the output directory is literal. The configured
    log path keeps its patterns. A path with whitespace, a quote or a newline would be
    split or misread in the directive, so it is refused rather than written.
    """
    anchored = _anchor_log_path(log_path, base_dir)
    if any(ch.isspace() or ch in "\"'" for ch in anchored):
        msg = (
            f"SLURM log path {anchored!r} contains whitespace or a quote, which an #SBATCH "
            "directive cannot carry; use an output directory and log name without them"
        )
        raise ValueError(msg)
    if Path(log_path).is_absolute():
        return log_path
    return str(Path(base_dir.replace("%", "%%")) / log_path)


_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _check_directive_value(name: str, value: str) -> None:
    """Refuse a resource value an ``#SBATCH`` directive cannot carry intact."""
    if not value or any(ch.isspace() or ch in "\"'#" for ch in value):
        msg = (
            f"SLURM {name} {value!r} is empty or contains whitespace, a quote or '#', "
            "which an #SBATCH directive cannot carry"
        )
        raise ValueError(msg)


#: Runs this environment's xtrax CLI with the generating interpreter. ``-P`` keeps the
#: task's working directory (sbatch's cwd) off ``sys.path``, as a console script would.
_CLI_BOOTSTRAP = "import sys; from xtrax.cli import main; sys.exit(main())"


def fn_import_roots(fn_spec: str) -> list[str]:
    """Directories an array task must put on ``PYTHONPATH`` to import ``fn_spec``.

    The top-level package of ``fn_spec`` (imported, or located without importing) says
    where it was found. A location inside the running environment or the base
    interpreter (``sys.prefix``, ``sys.exec_prefix``, ``sys.base_prefix``) or a site
    directory (including user site-packages) is importable in every task already;
    anything else, such as a source tree found through the cwd or ``PYTHONPATH``, is
    returned so the task does not depend on the submitting shell's environment.
    """
    top = fn_spec.rsplit(":", 1)[0].split(".")[0]
    mod = sys.modules.get(top)
    if mod is not None:
        locations = list(getattr(mod, "__path__", None) or [])
        origin = getattr(mod, "__file__", None)
    else:
        try:
            spec = importlib.util.find_spec(top)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            return []
        locations = list(spec.submodule_search_locations or [])
        origin = spec.origin if spec.has_location else None
    if locations:
        parents = [Path(loc).absolute().parent for loc in locations]
    elif origin:
        parents = [Path(str(origin)).absolute().parent]
    else:
        return []
    covered = {Path(p).absolute() for p in (sys.prefix, sys.exec_prefix, sys.base_prefix)}
    covered |= {Path(p).absolute() for p in site.getsitepackages()}
    user_site = site.getusersitepackages()
    if user_site:
        covered.add(Path(user_site).absolute())
    roots: list[str] = []
    for parent in parents:
        if any(parent == c or c in parent.parents for c in covered):
            continue
        if str(parent) not in roots:
            roots.append(str(parent))
    return roots


def task_environment(
    fn_spec: str,
) -> tuple[list[str], list[str], dict[str, str], list[str]]:
    """What each array task needs to run as the generating process would.

    Returns the launcher (the generating interpreter, by absolute path, running the
    xtrax CLI), the ``PYTHONPATH`` roots for ``fn_spec``, the exported variables (the
    resolved, absolute run-ledger root and the telemetry opt-out) and comment lines that
    name the layer each value came from. Array tasks start in sbatch's working
    directory, possibly with no exported environment (``--export=NONE``), so nothing
    here may depend on ``PATH``, the cwd or the submitting shell.
    """
    from xtrax.telemetry.ledger import DEFAULT_LEDGER_ROOT, resolve_root
    from xtrax.telemetry.record import telemetry_opted_out

    # Not .resolve(): a venv's bin/python is a symlink to the base interpreter, which
    # cannot import this environment's packages.
    python = str(Path(sys.executable).absolute())
    pythonpath = fn_import_roots(fn_spec)
    ledger_root = resolve_root().absolute()
    if (os.environ.get("XTRAX_LEDGER_ROOT") or "").strip():
        ledger_source = "XTRAX_LEDGER_ROOT at generation"
    else:
        ledger_source = f"the default {DEFAULT_LEDGER_ROOT} under the generating directory"
    env = {"XTRAX_LEDGER_ROOT": str(ledger_root)}
    comments = [
        "Baked at generation; tasks need no PATH, cwd or exported shell",
        "(works with sbatch --export=NONE).",
        f"python: interpreter of the generating environment ({sys.prefix})",
        f"XTRAX_LEDGER_ROOT: decided by {ledger_source}",
    ]
    if pythonpath:
        comments.append(f"PYTHONPATH: import root of {fn_spec}, outside the environment")
    if telemetry_opted_out():
        env["XTRAX_TELEMETRY_OPTOUT"] = "1"
        comments.append("XTRAX_TELEMETRY_OPTOUT: set at generation")
    return [python, "-P", "-c", _CLI_BOOTSTRAP], pythonpath, env, comments


def _anchor_log_path(log_path: str, base_dir: str) -> str:
    """Return an absolute SLURM log path: absolute input is kept, relative is under ``base_dir``."""
    if Path(log_path).is_absolute():
        return log_path
    return str(Path(base_dir) / log_path)


class SlurmArrayExecutor:
    """Pure host script generator emitting portable #SBATCH submission scripts for SLURM arrays.

    Per Section 6.3.3 of SPEC-261105-CAMPAIGN-RUNNER:
    - Pure host generator: 0 GPUs, 0 JIT compilations, 0 child processes during generation.
    - Sparse array resume: emits compact #SBATCH --array range scheduling only uncompleted rows.
    - Boundary control: if all units completed, returns early without generating an empty script.
    - Straggler cancellation: cancel_tasks issues scancel <job_id>_<task_id>.
    """

    def __init__(
        self,
        *,
        job_name: str = "xtrax_campaign",
        cpus_per_task: int = 4,
        gpus: int = 0,
        time_limit: str = "02:00:00",
        max_concurrent: int | None = None,
        partition: str | None = None,
        mem: str | None = None,
        output_log: str = "logs/slurm_%A_%a.out",
        error_log: str = "logs/slurm_%A_%a.err",
        job_id: str | None = None,
        submit: bool = False,
        manifest_path: str | Path | None = None,
        fn_spec: str | None = None,
        max_array_size: int = DEFAULT_MAX_ARRAY_SIZE,
        resume_verify: str = "deep",
        launcher: Sequence[str] | None = None,
        pythonpath: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        env_comments: Sequence[str] = (),
    ) -> None:
        """Configure the array script.

        Args:
            gpus: GPUs per array task; 0 (the default) emits no ``--gres`` line.
            partition: ``--partition`` value (one name or a comma-separated list).
            mem: ``--mem`` value (e.g. ``"4G"``); None emits no ``--mem`` line.
            launcher: Command words that run the xtrax CLI in each task. None (the
                default) uses the generating interpreter by absolute path (see
                :func:`task_environment`), so tasks do not depend on ``PATH`` or on
                sbatch exporting the submitting shell.
            pythonpath: Directories prepended to ``PYTHONPATH`` in each task. None (the
                default) uses :func:`fn_import_roots` of the function spec.
            env: Variables exported in each task (after every ``#SBATCH`` line). None
                (the default) exports the resolved run-ledger root and the telemetry
                opt-out of the generating process.
            env_comments: Comment lines written above the exports, e.g. which layer
                decided a baked value.

        Raises:
            ValueError: If a resource value would break an ``#SBATCH`` directive, a
                count is out of range, an env name is not a shell identifier, or a
                ``pythonpath`` entry contains ``:``.
        """
        from xtrax.campaign.executors.local import _validate_resume_verify

        self.resume_verify = _validate_resume_verify(resume_verify)
        if gpus < 0:
            raise ValueError(f"gpus must be >= 0, got {gpus}")
        if cpus_per_task < 1:
            raise ValueError(f"cpus_per_task must be >= 1, got {cpus_per_task}")
        if max_concurrent is not None and max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}")
        for name, value in (
            ("job_name", job_name),
            ("time_limit", time_limit),
            ("partition", partition),
            ("mem", mem),
        ):
            if value is not None:
                _check_directive_value(name, value)
        if launcher is not None and not launcher:
            raise ValueError("launcher must name at least one command word")
        for entry in pythonpath or ():
            if not entry or ":" in entry or "\n" in entry:
                raise ValueError(f"pythonpath entry {entry!r} is empty or contains ':'")
        for key, value in (env or {}).items():
            if not _ENV_NAME.fullmatch(key):
                raise ValueError(f"env name {key!r} is not a shell identifier")
            if "\n" in value or "\x00" in value:
                raise ValueError(f"env value for {key} contains a newline or NUL")
        self.job_name = job_name
        self.cpus_per_task = cpus_per_task
        self.gpus = gpus
        self.time_limit = time_limit
        self.max_concurrent = max_concurrent
        self.partition = partition
        self.mem = mem
        self.launcher = tuple(launcher) if launcher is not None else None
        self.pythonpath = tuple(pythonpath) if pythonpath is not None else None
        self.env = dict(env) if env is not None else None
        self.env_comments = tuple(line for c in env_comments for line in c.splitlines())
        self.output_log = output_log
        self.error_log = error_log
        self.job_id = job_id
        self.submit = submit
        #: Manifest the array tasks read; required by ``run``.
        self.manifest_path = manifest_path
        #: ``module:function`` import path each array task runs; required by ``run``.
        self.fn_spec = fn_spec
        #: Cluster ``MaxArraySize``; a row index at or above it cannot be a task ID.
        self.max_array_size = max_array_size
        self._cancelled_tasks: set[Any] = set()

    def cancel_tasks(self, task_ids: list[Any], job_id: str | None = None) -> None:
        """Issue scancel <job_id>_<task_id> for straggler control.

        Args:
            task_ids: List of task identifiers or row indices to cancel.
            job_id: Optional job ID override (defaults to self.job_id).
        """
        target_job = job_id or self.job_id
        for tid in task_ids:
            task_ref = f"{target_job}_{tid}" if target_job else str(tid)
            subprocess.run(["scancel", task_ref], check=True)
            self._cancelled_tasks.add(tid)

    def generate_script(
        self,
        manifest: "CampaignManifest",
        output_dir: Path | str,
        *,
        manifest_path: Path | str | None = None,
        fn_spec: str | None = None,
        row_indices: list[int] | None = None,
        resume: bool = True,
        max_concurrent: int | None = None,
        script_path: Path | str | None = None,
    ) -> str | None:
        """Generate a portable Bash #SBATCH array script.

        Each array task runs ``xtrax campaign run --executor local-seq`` on one row, so
        the task ID is the manifest row index.

        Args:
            manifest: Planned CampaignManifest.
            output_dir: Output directory containing results and done markers.
            manifest_path: Path to the manifest.json the tasks read (default: the
                executor's ``manifest_path``). Required.
            fn_spec: ``module:function`` each task runs (default: the executor's
                ``fn_spec``). Required.
            row_indices: Optional subset of rows to schedule.
            resume: Whether to filter out completed units.
            max_concurrent: Optional concurrency throttle override (%M).
            script_path: Optional path to write the generated script file.

        Returns:
            Generated submission script text, or None if no pending tasks remain.

        Raises:
            ValueError: If the manifest path or function spec is missing, or a pending
                row index is not below ``max_array_size``.
        """
        out_dir = Path(output_dir)
        resolved_manifest = manifest_path if manifest_path is not None else self.manifest_path
        resolved_fn = fn_spec if fn_spec is not None else self.fn_spec

        candidate_indices = _validate_row_indices(manifest, row_indices)
        pending_indices = (
            get_pending_row_indices(
                manifest,
                out_dir,
                resume_verify=self.resume_verify,
                row_indices=candidate_indices,
            )
            if resume
            else candidate_indices
        )

        # Boundary Control (GATE-09): Exit cleanly if all units completed
        if not pending_indices:
            print(
                f"All selected units in campaign '{manifest.metadata.campaign_id}' are "
                "already completed. No tasks to schedule."
            )
            return None

        if resolved_manifest is None:
            msg = "SlurmArrayExecutor needs manifest_path: array tasks must read the manifest"
            raise ValueError(msg)
        if not resolved_fn:
            msg = "SlurmArrayExecutor needs fn_spec ('module:function') for array tasks to run"
            raise ValueError(msg)

        if max(pending_indices) >= self.max_array_size:
            msg = (
                f"Row index {max(pending_indices)} is not below the cluster MaxArraySize "
                f"({self.max_array_size}); sbatch would reject the array. Split the "
                "manifest or pass max_array_size matching the cluster's configuration."
            )
            raise ValueError(msg)

        concurrent_limit = max_concurrent or self.max_concurrent
        array_spec = format_slurm_array_indices(pending_indices, max_concurrent=concurrent_limit)

        # Array tasks may start in a different working directory: pass absolute paths.
        abs_manifest = str(Path(resolved_manifest).resolve())
        abs_output_dir = str(out_dir.resolve())

        # sbatch resolves relative log paths against its own cwd, and SLURM does not
        # create missing log directories: anchor relative logs under the output directory
        # and create their parent so every array task can open its log at launch.
        out_log = _anchor_log_path(self.output_log, abs_output_dir)
        err_log = _anchor_log_path(self.error_log, abs_output_dir)
        # Checked before any directory is created, so a refused path leaves nothing behind.
        out_directive = _sbatch_log_path(self.output_log, abs_output_dir)
        err_directive = _sbatch_log_path(self.error_log, abs_output_dir)
        for log_path in (out_log, err_log):
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)

        directives = [
            "#!/usr/bin/env bash",
            f"#SBATCH --job-name={self.job_name}",
            f"#SBATCH --array={array_spec}",
            f"#SBATCH --cpus-per-task={self.cpus_per_task}",
        ]
        if self.gpus > 0:
            directives.append(f"#SBATCH --gres=gpu:{self.gpus}")
        directives.append(f"#SBATCH --time={self.time_limit}")
        if self.partition:
            directives.append(f"#SBATCH --partition={self.partition}")
        if self.mem:
            directives.append(f"#SBATCH --mem={self.mem}")
        directives.append(f"#SBATCH --output={out_directive}")
        directives.append(f"#SBATCH --error={err_directive}")

        # sbatch stops reading directives at the first command, so every export comes
        # after the last #SBATCH line.
        launcher_words, pythonpath, env = self.launcher, self.pythonpath, self.env
        comments: list[str] = []
        if launcher_words is None or pythonpath is None or env is None:
            d_launcher, d_pythonpath, d_env, comments = task_environment(resolved_fn)
            launcher_words = d_launcher if launcher_words is None else launcher_words
            pythonpath = d_pythonpath if pythonpath is None else pythonpath
            env = d_env if env is None else env
        env_lines: list[str] = [f"# {line}" for line in [*comments, *self.env_comments]]
        env_lines += [f"export {key}={shlex.quote(val)}" for key, val in env.items()]
        if pythonpath:
            joined = shlex.quote(":".join(pythonpath))
            env_lines.append(f"export PYTHONPATH={joined}" + "${PYTHONPATH:+:${PYTHONPATH}}")
        launcher = " ".join(shlex.quote(word) for word in launcher_words)

        script_lines = directives + [
            "",
            "set -euo pipefail",
            "",
            *env_lines,
            *([""] if env_lines else []),
            f"MANIFEST_PATH={shlex.quote(abs_manifest)}",
            f"OUTPUT_DIR={shlex.quote(abs_output_dir)}",
            f"UNIT_FN={shlex.quote(resolved_fn)}",
            "ROW_INDEX=${SLURM_ARRAY_TASK_ID}",
            "",
            'echo "Executing task ${SLURM_ARRAY_TASK_ID} on $(hostname)"',
            f"{launcher} campaign run \\",
            '  --manifest "${MANIFEST_PATH}" \\',
            '  --output-dir "${OUTPUT_DIR}" \\',
            '  --fn "${UNIT_FN}" \\',
            "  --executor local-seq \\",
            f"  --resume-verify {self.resume_verify} \\",
            '  --row-index "${ROW_INDEX}"',
            "",
        ]
        script_text = "\n".join(script_lines)

        if script_path is not None:
            path = Path(script_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(script_text, encoding="utf-8")
            os.chmod(path, 0o755)

        return script_text

    def run(
        self,
        manifest: "CampaignManifest",
        unit_fn: "Callable[[ManifestRow], Any]",
        *,
        output_dir: str,
        row_indices: list[int] | None = None,
        max_failures: int | None = None,
    ) -> CampaignSummary:
        """Generate (and optionally submit) the array script for the pending rows.

        ``unit_fn`` cannot cross into array tasks; they import ``fn_spec`` instead, so
        both ``manifest_path`` and ``fn_spec`` must be set on the executor. Nothing is
        completed by this call: pending rows are reported as ``submitted_units``.
        """
        del unit_fn, max_failures  # tasks import fn_spec; failures are per-task
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        candidate_indices = _validate_row_indices(manifest, row_indices)
        pending = get_pending_row_indices(
            manifest,
            out_dir,
            resume_verify=self.resume_verify,
            row_indices=candidate_indices,
        )
        total_units = len(candidate_indices)
        skipped_units = total_units - len(pending)

        # Boundary Control (GATE-09): early return on empty pending
        if not pending:
            return CampaignSummary(
                campaign_id=manifest.metadata.campaign_id,
                total_units=total_units,
                completed_units=0,
                skipped_units=skipped_units,
                failed_units=0,
                elapsed_seconds=0.0,
                failures=[],
            )

        script_path = out_dir / "submit_campaign.sh"
        self.generate_script(
            manifest=manifest,
            output_dir=out_dir,
            row_indices=pending,
            resume=False,
            script_path=script_path,
        )

        if self.submit:
            res = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            # Parse Submitted batch job <job_id>
            output = res.stdout.strip()
            if "Submitted batch job" in output:
                self.job_id = output.split()[-1]

        return CampaignSummary(
            campaign_id=manifest.metadata.campaign_id,
            total_units=total_units,
            completed_units=0,
            skipped_units=skipped_units,
            failed_units=0,
            elapsed_seconds=0.0,
            failures=[],
            submitted_units=len(pending),
        )
