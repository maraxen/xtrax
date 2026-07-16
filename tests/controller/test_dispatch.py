"""Tests for DispatchBackend Protocol and MockDispatchBackend (AC-2, #3611).

Exercises:
1. CandidateHandoff and CandidateHandoffFailure contract shapes (finding 10).
2. MockDispatchBackend's deterministic mode (mode=NONE, default).
3. MockDispatchBackend's failure-injection modes individually (finding 11):
   - TIMEOUT mode: raises TimeoutError.
   - MALFORMED_COMPLETION mode: raises ValueError.
   - CANDIDATE_HANDOFF_FAILURE mode: raises CandidateHandoffFailure.
"""

import hashlib
from pathlib import Path

import pytest

from controller.dispatch import (
    CandidateHandoff,
    CandidateHandoffFailure,
    DispatchBackend,
    MockDispatchBackend,
    MockFailureMode,
)

_VALID_SHA256 = hashlib.sha256(b"placeholder").hexdigest()


class TestCandidateHandoff:
    """Test CandidateHandoff contract (finding 10: FULL hash, not truncated)."""

    def test_candidate_handoff_is_frozen(self, tmp_path: Path) -> None:
        """CandidateHandoff is frozen; no field mutation."""
        path = tmp_path / "candidate.py"
        handoff = CandidateHandoff(
            path=path,
            content_sha256=_VALID_SHA256,
        )

        # Verify frozen: attempting to mutate raises FrozenInstanceError (dataclass machinery).
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError, match="cannot assign to field"):
            handoff.path = tmp_path / "other.py"  # type: ignore

        with pytest.raises(FrozenInstanceError, match="cannot assign to field"):
            handoff.content_sha256 = "def456"  # type: ignore

    def test_candidate_handoff_rejects_truncated_hash(self, tmp_path: Path) -> None:
        """content_sha256 shorter than 64 chars is rejected at construction (finding 10).

        This is the exact failure shape finding 10 warns against: an 8-char-prefix
        hash reintroducing collision exposure in a long-running loop. The invariant
        must be enforced structurally, not just by convention.
        """
        with pytest.raises(ValueError, match="64-character"):
            CandidateHandoff(path=tmp_path / "candidate.py", content_sha256="ab12cd34")

    def test_candidate_handoff_rejects_non_hex_hash(self, tmp_path: Path) -> None:
        """content_sha256 must be valid hex, even if 64 characters long."""
        with pytest.raises(ValueError, match="valid hex"):
            CandidateHandoff(path=tmp_path / "candidate.py", content_sha256="g" * 64)

    def test_candidate_handoff_full_hash_not_truncated(self, tmp_path: Path) -> None:
        """CandidateHandoff.content_sha256 must hold the FULL hex digest, not truncated.

        Requirement from finding 10: protect against hash collisions in long-running
        loops (unlike older 8-char-prefix approaches).
        """
        path = tmp_path / "candidate.py"
        source = "def foo():\n    return 42\n"

        # Compute the full SHA-256 hex digest (64 hex chars, 256 bits).
        full_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        assert len(full_hash) == 64, "SHA-256 hex digest must be 64 chars (256 bits)"

        handoff = CandidateHandoff(
            path=path,
            content_sha256=full_hash,
        )

        # Verify the hash is stored in full, not truncated to 8 chars (the old error pattern).
        assert handoff.content_sha256 == full_hash
        assert len(handoff.content_sha256) == 64


class TestCandidateHandoffFailure:
    """Test CandidateHandoffFailure exception.

    Finding 10 requirement: explicit staging-write failure variant, distinct from
    other dispatch errors (timeout, malformed completion).
    """

    def test_candidate_handoff_failure_is_an_exception(self) -> None:
        """CandidateHandoffFailure is an Exception, not a union type variant."""
        assert issubclass(CandidateHandoffFailure, Exception)

    def test_candidate_handoff_failure_can_be_raised_and_caught(self) -> None:
        """CandidateHandoffFailure can be raised and caught normally."""
        msg = "staging write failed: disk full"
        with pytest.raises(CandidateHandoffFailure, match="disk full"):
            raise CandidateHandoffFailure(msg)


class TestDispatchBackendProtocolConformance:
    """Verify MockDispatchBackend structurally satisfies the DispatchBackend Protocol."""

    def test_mock_dispatch_backend_is_a_dispatch_backend(self, tmp_path: Path) -> None:
        """MockDispatchBackend conforms to DispatchBackend (runtime_checkable Protocol)."""
        backend = MockDispatchBackend(
            candidate_path=tmp_path / "candidate.py",
            candidate_content="x = 1",
        )
        assert isinstance(backend, DispatchBackend)


class TestMockDispatchBackendDeterministic:
    """Test MockDispatchBackend in deterministic mode (mode=NONE, default)."""

    def test_mock_deterministic_returns_candidate_handoff(self, tmp_path: Path) -> None:
        """mode=NONE returns a valid CandidateHandoff with full SHA-256."""
        candidate_file = tmp_path / "candidate.py"
        source = "def foo():\n    return 42\n"
        expected_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()

        backend = MockDispatchBackend(
            candidate_path=candidate_file,
            candidate_content=source,
            mode=MockFailureMode.NONE,
        )

        handoff = backend.dispatch_candidate()

        assert isinstance(handoff, CandidateHandoff)
        assert handoff.path == candidate_file
        assert handoff.content_sha256 == expected_hash
        assert len(handoff.content_sha256) == 64  # Full SHA-256 digest

    def test_mock_deterministic_default_mode_is_none(self, tmp_path: Path) -> None:
        """Default mode is NONE; explicit mode kwarg is optional."""
        candidate_file = tmp_path / "candidate.py"
        source = "x = 1"
        expected_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()

        # No mode specified; should default to NONE.
        backend = MockDispatchBackend(
            candidate_path=candidate_file,
            candidate_content=source,
        )

        handoff = backend.dispatch_candidate()

        assert handoff.content_sha256 == expected_hash

    def test_mock_hash_stability_across_calls(self, tmp_path: Path) -> None:
        """Hash is stable: same content always produces same SHA-256 (deterministic)."""
        candidate_file = tmp_path / "candidate.py"
        source = "y = 2"

        backend = MockDispatchBackend(
            candidate_path=candidate_file,
            candidate_content=source,
            mode=MockFailureMode.NONE,
        )

        handoff_a = backend.dispatch_candidate()
        handoff_b = backend.dispatch_candidate()

        assert handoff_a.content_sha256 == handoff_b.content_sha256


class TestMockDispatchBackendTimeoutMode:
    """Test MockDispatchBackend in TIMEOUT failure-injection mode (finding 11)."""

    def test_mock_timeout_raises_timeout_error(self, tmp_path: Path) -> None:
        """mode=TIMEOUT raises TimeoutError."""
        backend = MockDispatchBackend(
            candidate_path=tmp_path / "candidate.py",
            candidate_content="x = 1",
            mode=MockFailureMode.TIMEOUT,
            timeout_delay=0.01,
        )

        with pytest.raises(TimeoutError, match="dispatch timed out"):
            backend.dispatch_candidate()

    def test_mock_timeout_respects_delay_parameter(self, tmp_path: Path) -> None:
        """mode=TIMEOUT sleeps for the specified delay before raising."""
        import time

        backend = MockDispatchBackend(
            candidate_path=tmp_path / "candidate.py",
            candidate_content="x = 1",
            mode=MockFailureMode.TIMEOUT,
            timeout_delay=0.1,
        )

        start = time.time()
        with pytest.raises(TimeoutError):
            backend.dispatch_candidate()
        elapsed = time.time() - start

        # Allow generous margin (0.05s) for test system variation.
        assert elapsed >= 0.05, f"Expected at least 0.05s delay, got {elapsed}s"


class TestMockDispatchBackendMalformedMode:
    """Test MockDispatchBackend in MALFORMED_COMPLETION failure-injection mode (finding 11)."""

    def test_mock_malformed_raises_value_error(self, tmp_path: Path) -> None:
        """mode=MALFORMED_COMPLETION raises ValueError."""
        backend = MockDispatchBackend(
            candidate_path=tmp_path / "candidate.py",
            candidate_content="x = 1",
            mode=MockFailureMode.MALFORMED_COMPLETION,
        )

        with pytest.raises(ValueError, match="malformed dispatch completion"):
            backend.dispatch_candidate()

    def test_mock_malformed_message_indicates_parsing_failure(self, tmp_path: Path) -> None:
        """ValueError message clarifies parsing failure (not just any error)."""
        backend = MockDispatchBackend(
            candidate_path=tmp_path / "candidate.py",
            candidate_content="x = 1",
            mode=MockFailureMode.MALFORMED_COMPLETION,
        )

        with pytest.raises(ValueError) as exc_info:
            backend.dispatch_candidate()

        assert "could not parse" in str(exc_info.value)


class TestMockDispatchBackendHandoffFailureMode:
    """Test MockDispatchBackend in CANDIDATE_HANDOFF_FAILURE failure-injection mode (finding 11)."""

    def test_mock_handoff_failure_raises_candidate_handoff_failure(self, tmp_path: Path) -> None:
        """mode=CANDIDATE_HANDOFF_FAILURE raises CandidateHandoffFailure."""
        backend = MockDispatchBackend(
            candidate_path=tmp_path / "candidate.py",
            candidate_content="x = 1",
            mode=MockFailureMode.CANDIDATE_HANDOFF_FAILURE,
        )

        with pytest.raises(CandidateHandoffFailure, match="staging write failed"):
            backend.dispatch_candidate()

    def test_mock_handoff_failure_message_indicates_write_failure(self, tmp_path: Path) -> None:
        """CandidateHandoffFailure message clarifies the write-path failure."""
        backend = MockDispatchBackend(
            candidate_path=tmp_path / "candidate.py",
            candidate_content="x = 1",
            mode=MockFailureMode.CANDIDATE_HANDOFF_FAILURE,
        )

        with pytest.raises(CandidateHandoffFailure) as exc_info:
            backend.dispatch_candidate()

        # Message should indicate staging/write semantics, not a generic error.
        error_msg = str(exc_info.value)
        assert "staging" in error_msg.lower() or "write" in error_msg.lower()


class TestMockDispatchBackendAllModesDistinct:
    """Verify all failure-injection modes produce distinct behaviors (finding 11 validation)."""

    def test_all_four_modes_produce_different_outcomes(self, tmp_path: Path) -> None:
        """Each mode (NONE, TIMEOUT, MALFORMED, HANDOFF_FAILURE) produces distinct behavior."""

        # Helper to classify the outcome of a dispatch call.
        def classify_outcome(mode: MockFailureMode):
            backend = MockDispatchBackend(
                candidate_path=tmp_path / "candidate.py",
                candidate_content="x = 1",
                mode=mode,
                timeout_delay=0.001,  # Minimal to speed up test.
            )
            try:
                result = backend.dispatch_candidate()
                return ("success", type(result).__name__)
            except TimeoutError:
                return ("timeout", "TimeoutError")
            except ValueError:
                return ("malformed", "ValueError")
            except CandidateHandoffFailure:
                return ("handoff_failure", "CandidateHandoffFailure")

        outcomes = {
            MockFailureMode.NONE: classify_outcome(MockFailureMode.NONE),
            MockFailureMode.TIMEOUT: classify_outcome(MockFailureMode.TIMEOUT),
            MockFailureMode.MALFORMED_COMPLETION: classify_outcome(
                MockFailureMode.MALFORMED_COMPLETION
            ),
            MockFailureMode.CANDIDATE_HANDOFF_FAILURE: classify_outcome(
                MockFailureMode.CANDIDATE_HANDOFF_FAILURE
            ),
        }

        # Verify all outcomes are distinct.
        outcome_values = list(outcomes.values())
        assert len(outcome_values) == len(set(outcome_values)), (
            f"Expected all four modes to produce distinct outcomes; got {outcomes}"
        )

        # Spot-check each mode produces the expected classification.
        assert outcomes[MockFailureMode.NONE][0] == "success"
        assert outcomes[MockFailureMode.TIMEOUT][0] == "timeout"
        assert outcomes[MockFailureMode.MALFORMED_COMPLETION][0] == "malformed"
        assert outcomes[MockFailureMode.CANDIDATE_HANDOFF_FAILURE][0] == "handoff_failure"


__all__ = [
    "TestCandidateHandoff",
    "TestCandidateHandoffFailure",
    "TestMockDispatchBackendAllModesDistinct",
    "TestMockDispatchBackendDeterministic",
    "TestMockDispatchBackendHandoffFailureMode",
    "TestMockDispatchBackendMalformedMode",
    "TestMockDispatchBackendTimeoutMode",
]
