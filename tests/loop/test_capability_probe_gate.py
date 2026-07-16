"""Tests for the T2-27 capability-probe gate (#2181, AC-20, PM-4)."""

import pytest

from xtrax.loop.capability_probe_gate import (
    CapabilityNotLiveError,
    CapabilityProbeInputError,
    CapabilityProbeResult,
    assert_capability_live,
)


class TestExplorationAlwaysProceeds:
    def test_both_live_does_not_raise(self):
        result = CapabilityProbeResult(seed_live=True, stats_battery_live=True)
        assert_capability_live(result, campaign_mode="exploration")

    def test_neither_live_does_not_raise(self):
        result = CapabilityProbeResult(seed_live=False, stats_battery_live=False)
        assert_capability_live(result, campaign_mode="exploration")

    def test_seed_not_live_does_not_raise(self):
        result = CapabilityProbeResult(seed_live=False, stats_battery_live=True)
        assert_capability_live(result, campaign_mode="exploration")

    def test_stats_not_live_does_not_raise(self):
        result = CapabilityProbeResult(seed_live=True, stats_battery_live=False)
        assert_capability_live(result, campaign_mode="exploration")


class TestConfirmationModeGating:
    def test_both_live_does_not_raise(self):
        result = CapabilityProbeResult(seed_live=True, stats_battery_live=True)
        assert_capability_live(result, campaign_mode="confirmation")

    def test_seed_not_live_raises_naming_seed(self):
        result = CapabilityProbeResult(seed_live=False, stats_battery_live=True)
        with pytest.raises(CapabilityNotLiveError, match="seed_live"):
            assert_capability_live(result, campaign_mode="confirmation")

    def test_stats_not_live_raises_naming_stats(self):
        result = CapabilityProbeResult(seed_live=True, stats_battery_live=False)
        with pytest.raises(CapabilityNotLiveError, match="stats_battery_live"):
            assert_capability_live(result, campaign_mode="confirmation")

    def test_neither_live_raises_naming_both(self):
        result = CapabilityProbeResult(seed_live=False, stats_battery_live=False)
        with pytest.raises(CapabilityNotLiveError) as exc_info:
            assert_capability_live(result, campaign_mode="confirmation")
        message = str(exc_info.value)
        assert "seed_live" in message
        assert "stats_battery_live" in message


class TestSequentialModeGating:
    def test_both_live_does_not_raise(self):
        result = CapabilityProbeResult(seed_live=True, stats_battery_live=True)
        assert_capability_live(result, campaign_mode="sequential")

    def test_seed_not_live_raises_naming_seed(self):
        result = CapabilityProbeResult(seed_live=False, stats_battery_live=True)
        with pytest.raises(CapabilityNotLiveError, match="seed_live"):
            assert_capability_live(result, campaign_mode="sequential")

    def test_stats_not_live_raises_naming_stats(self):
        result = CapabilityProbeResult(seed_live=True, stats_battery_live=False)
        with pytest.raises(CapabilityNotLiveError, match="stats_battery_live"):
            assert_capability_live(result, campaign_mode="sequential")

    def test_neither_live_raises_naming_both(self):
        result = CapabilityProbeResult(seed_live=False, stats_battery_live=False)
        with pytest.raises(CapabilityNotLiveError) as exc_info:
            assert_capability_live(result, campaign_mode="sequential")
        message = str(exc_info.value)
        assert "seed_live" in message
        assert "stats_battery_live" in message


class TestCapabilityProbeResult:
    def test_is_frozen(self):
        result = CapabilityProbeResult(seed_live=True, stats_battery_live=True)
        with pytest.raises(AttributeError):
            result.seed_live = False


class TestInvalidCampaignModeRaises:
    def test_unrecognized_mode_raises_regardless_of_liveness(self):
        result = CapabilityProbeResult(seed_live=True, stats_battery_live=True)
        with pytest.raises(CapabilityProbeInputError, match="confirmatory"):
            assert_capability_live(result, campaign_mode="confirmatory")

    def test_unrecognized_mode_raises_even_when_both_not_live(self):
        result = CapabilityProbeResult(seed_live=False, stats_battery_live=False)
        with pytest.raises(CapabilityProbeInputError, match="confirmatory"):
            assert_capability_live(result, campaign_mode="confirmatory")
