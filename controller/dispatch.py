"""DispatchBackend Protocol and MockDispatchBackend (AC-2, #3611).

This module defines the abstract interface for a dispatch backend that proposes
candidate mutations and hands them off to the controller. A concrete backend
(e.g., PraxiaDispatchBackend) implements DispatchBackend; this module also
provides MockDispatchBackend for testing, with configurable failure-injection
modes for exercising error/retry paths without real infrastructure.

Key design choices:
- DispatchBackend is a typing.Protocol for pluggable backends.
- CandidateHandoff includes the FULL SHA-256 hash (not truncated), protecting
  against collision exposure in long-running loops (finding 10).
- MockDispatchBackend supports deterministic candidate return (default) and
  configurable failure-injection modes: timeout, malformed completion, and
  CandidateHandoffFailure (finding 11).
- Zero praxia dependency; CandidateHandoffFailure is an exception, not a
  union type variant, following Python idiom for write-path failures.
"""

import hashlib
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable


class CandidateHandoffFailure(Exception):
    """Raised when a dispatch backend fails to hand off a candidate (e.g., staging write fails).

    Finding 10 revision: explicit failure variant for staging-write failures, distinct
    from other kinds of dispatch errors (e.g., timeout, malformed completion).
    """


class MockFailureMode(Enum):
    """Configurable failure-injection modes for MockDispatchBackend (finding 11)."""

    NONE = "none"  # Default: deterministic successful return
    TIMEOUT = "timeout"  # Simulate a dispatch timeout
    MALFORMED_COMPLETION = "malformed"  # Return a completion that cannot be parsed
    CANDIDATE_HANDOFF_FAILURE = "handoff_failure"  # Raise CandidateHandoffFailure


@runtime_checkable
class DispatchBackend(Protocol):
    """Abstract interface for a dispatch backend.

    A concrete backend implementing this protocol proposes the next candidate
    mutation given some loop state context, performs any necessary staging
    (writing source files, computing hashes), and returns a CandidateHandoff
    contract confirming successful hand-off.

    Raises:
        CandidateHandoffFailure: if staging-write or other hand-off mechanics fail.
        TimeoutError: if dispatch takes too long (backend-specific timeout semantics).
    """

    def dispatch_candidate(self) -> "CandidateHandoff":
        """Propose and hand off the next candidate mutation.

        Returns:
            CandidateHandoff: a frozen contract with path and content_sha256,
                ready for the controller to read and verify.

        Raises:
            CandidateHandoffFailure: if staging or hand-off fails.
            TimeoutError: if dispatch exceeds backend timeout (if applicable).
        """
        ...


@dataclass(frozen=True, slots=True)
class CandidateHandoff:
    """Hand-off contract from dispatch backend to controller.

    Attributes:
        path: Path to the staged candidate source file (must be readable by controller).
        content_sha256: Full SHA-256 hex digest of the candidate source. Computed from
            the FULL hash, never truncated (finding 10: protects against hash collisions
            in long-running loops, unlike older 8-char-prefix approaches).
    """

    path: Path
    content_sha256: str


@dataclass
class MockDispatchBackend:
    """Mock DispatchBackend for testing, with deterministic and failure-injection modes.

    Finding 11 revision: supports both deterministic candidate return (default) and
    configurable failure-injection modes (timeout, malformed completion,
    CandidateHandoffFailure), so AC-8c's error/retry paths can be unit-tested without
    real infrastructure.

    Attributes:
        candidate_path: Path to return in CandidateHandoff (used when mode is NONE).
        candidate_content: Source content to hash and include in CandidateHandoff
            (used when mode is NONE).
        mode: MockFailureMode controlling dispatch behavior (default: NONE for
            deterministic return).
        timeout_delay: Seconds to sleep before raising TimeoutError (used when mode
            is TIMEOUT).
    """

    candidate_path: Path
    candidate_content: str
    mode: MockFailureMode = MockFailureMode.NONE
    timeout_delay: float = 0.1

    def dispatch_candidate(self) -> CandidateHandoff:
        """Propose a candidate, optionally injecting a failure.

        Returns:
            CandidateHandoff: path and content_sha256 (deterministic on mode=NONE).

        Raises:
            TimeoutError: if mode is TIMEOUT.
            ValueError: if mode is MALFORMED_COMPLETION.
            CandidateHandoffFailure: if mode is CANDIDATE_HANDOFF_FAILURE.
        """
        match self.mode:
            case MockFailureMode.NONE:
                # Deterministic success: compute full SHA-256 and return.
                sha256_hex = hashlib.sha256(self.candidate_content.encode("utf-8")).hexdigest()
                return CandidateHandoff(
                    path=self.candidate_path,
                    content_sha256=sha256_hex,
                )

            case MockFailureMode.TIMEOUT:
                # Simulate a dispatch timeout.
                time.sleep(self.timeout_delay)
                msg = f"dispatch timed out after {self.timeout_delay}s"
                raise TimeoutError(msg)

            case MockFailureMode.MALFORMED_COMPLETION:
                # Return a response that cannot be parsed as a CandidateHandoff.
                # This is simulated via ValueError to indicate parsing failure.
                msg = "malformed dispatch completion: could not parse candidate metadata"
                raise ValueError(msg)

            case MockFailureMode.CANDIDATE_HANDOFF_FAILURE:
                # Raise the explicit staging-write failure variant.
                msg = "staging write failed: permission denied or disk full"
                raise CandidateHandoffFailure(msg)


__all__ = [
    "CandidateHandoff",
    "CandidateHandoffFailure",
    "DispatchBackend",
    "MockDispatchBackend",
    "MockFailureMode",
]
