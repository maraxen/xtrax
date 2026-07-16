"""Tests for the T2-25 sidecar-drift reaction gate (#2181, AC-18)."""

import pytest

from xtrax.loop.sidecar_drift_gate import (
    SidecarDriftGateInputError,
    SidecarDriftSignal,
    SidecarHashMismatchError,
    assert_sidecar_drift_reaction,
)


def _signal(*, drifted, script_id="run_nvt.py", first_run_sha256="", current_sha256=""):
    return SidecarDriftSignal(
        drifted=drifted,
        script_id=script_id,
        first_run_sha256=first_run_sha256,
        current_sha256=current_sha256,
    )


class TestNoDriftNeverReactsRegardlessOfMode:
    def test_collaborative(self):
        decision = assert_sidecar_drift_reaction(_signal(drifted=False), agent_mode="collaborative")
        assert decision.drifted is False
        assert decision.should_warn is False
        assert decision.reason == ""

    def test_autonomous(self):
        decision = assert_sidecar_drift_reaction(_signal(drifted=False), agent_mode="autonomous")
        assert decision.drifted is False
        assert decision.should_warn is False
        assert decision.reason == ""


class TestNoDriftCoversZeroPriorRunsCase:
    """bathos's own check_sidecar_drift returns False both when hashes match AND when a script
    has no prior run carrying a sidecar hash yet -- either way, drifted=False and no reaction."""

    def test_no_baseline_yet_is_indistinguishable_from_matching(self):
        # A caller with zero prior runs to compare against passes drifted=False, same as a
        # caller whose hashes genuinely matched -- this module does not need to tell them apart.
        no_baseline = assert_sidecar_drift_reaction(
            _signal(drifted=False, first_run_sha256="", current_sha256="abc123"),
            agent_mode="autonomous",
        )
        matching = assert_sidecar_drift_reaction(
            _signal(drifted=False, first_run_sha256="abc123", current_sha256="abc123"),
            agent_mode="autonomous",
        )
        assert no_baseline == matching


class TestDriftCollaborativeWarnsWithoutRaising:
    def test_warns_and_names_script_and_hashes(self):
        decision = assert_sidecar_drift_reaction(
            _signal(
                drifted=True,
                script_id="scripts/experiments/run_nvt.py",
                first_run_sha256="original_sha_abc",
                current_sha256="edited_sha_xyz",
            ),
            agent_mode="collaborative",
        )
        assert decision.drifted is True
        assert decision.should_warn is True
        assert "scripts/experiments/run_nvt.py" in decision.reason
        assert "original_sha_abc" in decision.reason
        assert "edited_sha_xyz" in decision.reason


class TestDriftAutonomousDenies:
    def test_raises_naming_script_and_hashes(self):
        with pytest.raises(SidecarHashMismatchError) as exc_info:
            assert_sidecar_drift_reaction(
                _signal(
                    drifted=True,
                    script_id="scripts/experiments/run_nvt.py",
                    first_run_sha256="original_sha_abc",
                    current_sha256="edited_sha_xyz",
                ),
                agent_mode="autonomous",
            )
        message = str(exc_info.value)
        assert "scripts/experiments/run_nvt.py" in message
        assert "original_sha_abc" in message
        assert "edited_sha_xyz" in message


class TestSidecarDriftSignal:
    def test_is_frozen(self):
        signal = _signal(drifted=True)
        with pytest.raises(AttributeError):
            signal.drifted = False


class TestInvalidAgentModeRaises:
    def test_unrecognized_mode_raises_when_not_drifted(self):
        with pytest.raises(SidecarDriftGateInputError, match="semi-autonomous"):
            assert_sidecar_drift_reaction(_signal(drifted=False), agent_mode="semi-autonomous")

    def test_unrecognized_mode_raises_when_drifted(self):
        with pytest.raises(SidecarDriftGateInputError, match="semi-autonomous"):
            assert_sidecar_drift_reaction(_signal(drifted=True), agent_mode="semi-autonomous")
