"""Debt #3318: CLI SLURM resource flags, and array scripts that need no submitting shell.

Found by the PR #267 sbatch smoke on Engaging (job 28152909): the CLI could not set
partition, GPUs, time or memory (the script always asked for a GPU), and the script ran
bare ``xtrax``, so it worked only because sbatch's default ``--export=ALL`` carried the
venv ``PATH`` and ``PYTHONPATH`` from the submitting shell.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.campaign.executors.test_local import _create_mock_manifest
from tests.campaign.test_r3_cli_slurm import _write_manifest
from tests.campaign.unit_fns import ARRAY_UNIT_FN
from xtrax.campaign.executors.local import _resolve_output_path
from xtrax.campaign.executors.slurm import SlurmArrayExecutor
from xtrax.campaign.marker import validate_unit_done
from xtrax.cli.campaign_verb import CampaignArgs, _fn_import_roots, run_campaign
from xtrax.cli.errors import CLIError

REPO_ROOT = Path(__file__).resolve().parents[2]


def _generate(tmp_path: Path, campaign_id: str, num_rows: int = 2, **slurm: object) -> Path:
    manifest = _create_mock_manifest(num_rows=num_rows, campaign_id=campaign_id)
    manifest_path = _write_manifest(tmp_path / campaign_id / "manifest.json", manifest)
    out = tmp_path / campaign_id / "outputs"
    run_campaign(
        CampaignArgs(
            action="run",
            fn=ARRAY_UNIT_FN,
            manifest=str(manifest_path),
            output_dir=str(out),
            executor="slurm",
            **slurm,  # type: ignore[arg-type]
        )
    )
    return out / "submit_campaign.sh"


def _directives(script: Path) -> list[str]:
    return [ln for ln in script.read_text().splitlines() if ln.startswith("#SBATCH")]


# ==============================================================================
# The script runs with an empty environment from another directory
# ==============================================================================


def test_generated_script_runs_with_no_exported_environment(
    tmp_path: Path, _isolated_run_ledger: Path
) -> None:
    """Like ``sbatch --export=NONE``: no venv on PATH, no PYTHONPATH, a foreign cwd."""
    from xtrax.telemetry.ledger import iter_rows

    script = _generate(tmp_path, "camp_3318_env", num_rows=2)
    elsewhere = tmp_path / "submit_dir"
    elsewhere.mkdir()
    proc = subprocess.run(
        [
            "env",
            "-i",
            "PATH=/usr/bin:/bin",
            f"HOME={os.environ.get('HOME', str(tmp_path))}",
            "SLURM_ARRAY_TASK_ID=1",
            "bash",
            str(script),
        ],
        cwd=elsewhere,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    manifest = _create_mock_manifest(num_rows=2, campaign_id="camp_3318_env")
    out = script.parent
    row = manifest.rows[1]
    assert validate_unit_done(_resolve_output_path(out, row), row, deep_verify=True).is_valid
    assert not _resolve_output_path(out, manifest.rows[0]).exists(), "ran the wrong row"

    rows = [r for r in iter_rows(_isolated_run_ledger) if r.context.get("verb") == "campaign"]
    assert len(rows) == 1, "the task's ledger row did not land in the baked root"
    assert not (elsewhere / ".xtrax").exists(), "ledger followed the task's cwd"


# ==============================================================================
# Resource flags
# ==============================================================================


def test_default_script_requests_no_gpu(tmp_path: Path) -> None:
    directives = _directives(_generate(tmp_path, "camp_3318_default"))
    assert not any("--gres" in d for d in directives)
    assert "#SBATCH --time=02:00:00" in directives
    assert "#SBATCH --cpus-per-task=4" in directives
    assert not any("--partition" in d or "--mem" in d for d in directives)


def test_slurm_flags_reach_the_directives(tmp_path: Path) -> None:
    directives = _directives(
        _generate(
            tmp_path,
            "camp_3318_flags",
            num_rows=3,
            slurm_partition="pi_so3,mit_preemptable",
            slurm_gpus=2,
            slurm_time="00:10:00",
            slurm_cpus=2,
            slurm_mem="4G",
            slurm_job_name="smoke",
            slurm_max_concurrent=2,
        )
    )
    for expected in (
        "#SBATCH --partition=pi_so3,mit_preemptable",
        "#SBATCH --gres=gpu:2",
        "#SBATCH --time=00:10:00",
        "#SBATCH --cpus-per-task=2",
        "#SBATCH --mem=4G",
        "#SBATCH --job-name=smoke",
        "#SBATCH --array=0-2%2",
    ):
        assert expected in directives


def test_exports_follow_every_directive(tmp_path: Path) -> None:
    lines = _generate(tmp_path, "camp_3318_order").read_text().splitlines()
    last_directive = max(i for i, ln in enumerate(lines) if ln.startswith("#SBATCH"))
    first_export = min(i for i, ln in enumerate(lines) if ln.startswith("export "))
    assert first_export > last_directive


@pytest.mark.parametrize(
    ("field", "value"),
    [("slurm_partition", "a b"), ("slurm_mem", "4G\n#SBATCH --x"), ("slurm_time", "")],
)
def test_directive_breaking_value_is_refused(tmp_path: Path, field: str, value: str) -> None:
    with pytest.raises(CLIError, match="#SBATCH directive"):
        _generate(tmp_path, f"camp_3318_bad_{field}", **{field: value})


@pytest.mark.parametrize(("field", "value"), [("slurm_gpus", -1), ("slurm_cpus", 0)])
def test_out_of_range_count_is_refused(tmp_path: Path, field: str, value: int) -> None:
    with pytest.raises(CLIError, match=">="):
        _generate(tmp_path, f"camp_3318_range_{field}", **{field: value})


def test_slurm_flag_with_local_executor_is_refused(tmp_path: Path) -> None:
    manifest = _create_mock_manifest(num_rows=1, campaign_id="camp_3318_local")
    manifest_path = _write_manifest(tmp_path / "manifest.json", manifest)
    with pytest.raises(CLIError, match="--slurm-partition only apply to --executor slurm"):
        run_campaign(
            CampaignArgs(
                action="run",
                fn=ARRAY_UNIT_FN,
                manifest=str(manifest_path),
                executor="local-seq",
                slurm_partition="mit_quicktest",
            )
        )
    assert not list(tmp_path.rglob("*.done.json")), "units ran despite the refused flag"


# ==============================================================================
# Baked environment
# ==============================================================================


def test_launcher_is_the_generating_environments_console_script(tmp_path: Path) -> None:
    text = _generate(tmp_path, "camp_3318_launcher").read_text()
    expected = Path(sys.executable).absolute().parent / "xtrax"
    assert f"{expected} campaign run \\" in text


def test_missing_console_script_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_bin = tmp_path / "fakeenv" / "bin"
    fake_bin.mkdir(parents=True)
    monkeypatch.setattr(sys, "executable", str(fake_bin / "python"))
    with pytest.raises(CLIError, match="no xtrax console script"):
        _generate(tmp_path, "camp_3318_noscript")


def test_ledger_root_default_is_baked_absolute_with_its_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("XTRAX_LEDGER_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    text = _generate(tmp_path, "camp_3318_ledger_default").read_text()
    assert f"export XTRAX_LEDGER_ROOT={tmp_path / '.xtrax' / 'ledger'}" in text
    assert "# XTRAX_LEDGER_ROOT: decided by the default" in text


def test_ledger_root_from_env_is_baked_with_its_source(
    tmp_path: Path, _isolated_run_ledger: Path
) -> None:
    text = _generate(tmp_path, "camp_3318_ledger_env").read_text()
    assert f"export XTRAX_LEDGER_ROOT={_isolated_run_ledger}" in text
    assert "# XTRAX_LEDGER_ROOT: decided by XTRAX_LEDGER_ROOT at generation" in text


def test_telemetry_optout_is_baked_only_when_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XTRAX_TELEMETRY_OPTOUT", "0")
    assert "XTRAX_TELEMETRY_OPTOUT" not in _generate(tmp_path, "camp_3318_opt0").read_text()
    monkeypatch.setenv("XTRAX_TELEMETRY_OPTOUT", "yes")
    assert "export XTRAX_TELEMETRY_OPTOUT=1" in _generate(tmp_path, "camp_3318_opt1").read_text()


def test_fn_import_roots() -> None:
    assert _fn_import_roots(ARRAY_UNIT_FN) == [str(REPO_ROOT)]
    import json  # noqa: F401  (stdlib: inside the base interpreter)

    assert _fn_import_roots("json:dumps") == []
    assert _fn_import_roots("never_imported_module_3318:fn") == []


def test_executor_refuses_bad_env_and_pythonpath() -> None:
    with pytest.raises(ValueError, match="shell identifier"):
        SlurmArrayExecutor(env={"BAD-NAME": "x"})
    with pytest.raises(ValueError, match="contains ':'"):
        SlurmArrayExecutor(pythonpath=["/a:/b"])
    with pytest.raises(ValueError, match="launcher"):
        SlurmArrayExecutor(launcher=())
