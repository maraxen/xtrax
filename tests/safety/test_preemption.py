"""Tests for PreemptionHandler: signal handling and state management."""

import os
import signal
import time

import pytest

from xtrax.safety.preemption import PreemptionHandler


@pytest.fixture(autouse=True)
def reset_signal_handlers():
    """Save and restore signal handlers before/after each test.

    This ensures test isolation and prevents order-dependent failures.
    """
    sigusr1_old = signal.signal(signal.SIGUSR1, signal.SIG_DFL)
    sigterm_old = signal.signal(signal.SIGTERM, signal.SIG_DFL)
    yield
    signal.signal(signal.SIGUSR1, sigusr1_old)
    signal.signal(signal.SIGTERM, sigterm_old)


class TestPreemptionHandler:
    """Test PreemptionHandler signal registration and callback behavior."""

    def test_sigusr1_triggers_save_fn(self):
        """SIGUSR1 should trigger save_fn exactly once."""
        call_count = [0]

        def save_fn():
            call_count[0] += 1

        handler = PreemptionHandler(save_fn, rank=0)
        handler.register()

        # Send SIGUSR1 to self
        os.kill(os.getpid(), signal.SIGUSR1)
        # Small delay to allow signal handler to execute
        time.sleep(0.01)

        assert call_count[0] == 1, f"Expected save_fn called once, got {call_count[0]}"

    def test_sigterm_triggers_save_fn(self):
        """SIGTERM should also trigger save_fn exactly once."""
        call_count = [0]

        def save_fn():
            call_count[0] += 1

        handler = PreemptionHandler(save_fn, rank=0)
        handler.register()

        # Send SIGTERM to self
        os.kill(os.getpid(), signal.SIGTERM)
        # Small delay to allow signal handler to execute
        time.sleep(0.01)

        assert call_count[0] == 1, f"Expected save_fn called once, got {call_count[0]}"

    def test_rank_non_zero_does_not_call_save_fn(self):
        """rank != 0 should NOT call save_fn on SIGUSR1 or SIGTERM."""
        call_count = [0]

        def save_fn():
            call_count[0] += 1

        # Install SIG_IGN to prevent OS default signal handling (kill)
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

        handler = PreemptionHandler(save_fn, rank=1)
        handler.register()

        # Send SIGUSR1 to self
        os.kill(os.getpid(), signal.SIGUSR1)
        time.sleep(0.01)

        # Send SIGTERM to self
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.01)

        msg = (
            f"Expected save_fn never called with rank=1, "
            f"got {call_count[0]} calls"
        )
        assert call_count[0] == 0, msg

    def test_register_idempotent(self):
        """Calling register() twice should register handler only once."""
        call_count = [0]

        def save_fn():
            call_count[0] += 1

        handler = PreemptionHandler(save_fn, rank=0)
        handler.register()
        handler.register()  # Second call should be no-op

        # Send SIGUSR1
        os.kill(os.getpid(), signal.SIGUSR1)
        time.sleep(0.01)

        # Handler should still be called exactly once (not twice)
        msg = (
            f"Expected save_fn called once with idempotent register, "
            f"got {call_count[0]}"
        )
        assert call_count[0] == 1, msg

    def test_preempted_property_false_before_signal(self):
        """preempted should be False before any signal is received."""
        handler = PreemptionHandler(lambda: None, rank=0)
        handler.register()

        assert handler.preempted is False, "preempted should be False before signal"

    def test_preempted_property_true_after_signal(self):
        """preempted should be True after signal is received."""
        handler = PreemptionHandler(lambda: None, rank=0)
        handler.register()

        # Send SIGUSR1
        os.kill(os.getpid(), signal.SIGUSR1)
        time.sleep(0.01)

        assert handler.preempted is True, "preempted should be True after signal"

    def test_save_fn_called_at_most_once_with_multiple_signals(self):
        """save_fn should be called at most once even if multiple signals fire."""
        call_count = [0]

        def save_fn():
            call_count[0] += 1

        handler = PreemptionHandler(save_fn, rank=0)
        handler.register()

        # Send both SIGUSR1 and SIGTERM
        os.kill(os.getpid(), signal.SIGUSR1)
        time.sleep(0.01)
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.01)

        msg = f"Expected save_fn called at most once, got {call_count[0]} calls"
        assert call_count[0] == 1, msg
