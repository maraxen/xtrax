"""Tests for the single-parent `derived_from` lineage interim (LC-08, epic #3611, AC-7).

AC-7's own measurable criterion: "explicit fail-loud behavior -- if the controller ever
attempts to record a genuinely multi-parent merge ... under this single-parent interim, it
MUST raise a named exception, never silently pick one parent or silently drop lineage."

Exercises:
1. `CandidateParentage.distinct_real_parents`/`is_multi_parent` -- the dedup/blank-filter rule
   that decides what counts as "genuinely" multi-parent (finding: a repeated ID is NOT
   multi-parent; a blank entry is NOT a real parent).
2. `resolve_derived_from` -- single-parent and root (no-parent) cases resolve correctly;
   a genuine multi-parent case raises `MultiParentLineageUnsupportedError` naming the
   offending parents in its message.
3. `record_candidate_run` -- threads the resolved `derived_from` through to
   `BathosCampaignAdapter.run` (mocked, no live bathos infrastructure) for the single-parent
   case, and -- the fail-loud requirement's sharpest test -- for the multi-parent case,
   raises BEFORE `adapter.run` is ever called (asserted by a call-recording spy, not just
   "raises somewhere").

No live bathos infrastructure is required for any test in this module.
"""

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from controller.bathos_campaign_adapter import CandidateRunResult
from controller.lineage_interim import (
    CandidateParentage,
    MultiParentLineageUnsupportedError,
    record_candidate_run,
    resolve_derived_from,
)


class _RecordingAdapter:
    """A `BathosCampaignAdapter`-shaped spy: records every `run` call it receives and
    returns a pre-programmed `CandidateRunResult`. Used to prove `record_candidate_run`
    either does or does not reach `adapter.run`, without any real bathos infrastructure.
    """

    def __init__(self) -> None:
        self.run_calls: list[dict[str, Any]] = []

    def run(self, script_path: str, **kwargs: Any) -> CandidateRunResult:
        self.run_calls.append({"script_path": script_path, **kwargs})
        return CandidateRunResult(script_path=script_path, exit_code=0, success=True)


# ---------------------------------------------------------------------------
# CandidateParentage: distinct_real_parents / is_multi_parent
# ---------------------------------------------------------------------------


class TestCandidateParentage:
    def test_is_frozen(self) -> None:
        parentage = CandidateParentage(parent_run_ids=("run-a",))
        with pytest.raises(FrozenInstanceError):
            parentage.parent_run_ids = ("run-b",)  # type: ignore

    def test_no_parents_is_not_multi_parent(self) -> None:
        parentage = CandidateParentage()
        assert parentage.distinct_real_parents == ()
        assert parentage.is_multi_parent is False

    def test_single_parent(self) -> None:
        parentage = CandidateParentage(parent_run_ids=("run-a",))
        assert parentage.distinct_real_parents == ("run-a",)
        assert parentage.is_multi_parent is False

    def test_same_parent_listed_twice_is_not_multi_parent(self) -> None:
        """The same parent ID repeated is one real parent, not a multi-parent merge --
        exactly the distinction the AC's own self-audit warns against getting wrong."""
        parentage = CandidateParentage(parent_run_ids=("run-a", "run-a", "run-a"))
        assert parentage.distinct_real_parents == ("run-a",)
        assert parentage.is_multi_parent is False

    def test_blank_entries_are_not_real_parents(self) -> None:
        parentage = CandidateParentage(parent_run_ids=("", "   ", "run-a"))
        assert parentage.distinct_real_parents == ("run-a",)
        assert parentage.is_multi_parent is False

    def test_all_blank_resolves_to_no_parents(self) -> None:
        parentage = CandidateParentage(parent_run_ids=("", "  \t"))
        assert parentage.distinct_real_parents == ()
        assert parentage.is_multi_parent is False

    def test_genuinely_distinct_parents_is_multi_parent(self) -> None:
        """A real, actual multi-parent scenario: two genuinely distinct parent run IDs
        (a ratcheted best-so-far candidate merging two prior candidates)."""
        parentage = CandidateParentage(parent_run_ids=("run-a", "run-b"))
        assert parentage.distinct_real_parents == ("run-a", "run-b")
        assert parentage.is_multi_parent is True

    def test_distinct_real_parents_preserves_first_seen_order_and_dedups(self) -> None:
        parentage = CandidateParentage(
            parent_run_ids=("run-b", "run-a", "run-b", "", "run-c", "run-a")
        )
        assert parentage.distinct_real_parents == ("run-b", "run-a", "run-c")
        assert parentage.is_multi_parent is True


# ---------------------------------------------------------------------------
# resolve_derived_from
# ---------------------------------------------------------------------------


class TestResolveDerivedFrom:
    def test_root_candidate_resolves_to_empty_string(self) -> None:
        """No parent at all -- matches BathosCampaignAdapter.run's own derived_from
        default/pass-through convention (empty string, not None)."""
        assert resolve_derived_from(CandidateParentage()) == ""

    def test_single_parent_resolves_to_that_parent(self) -> None:
        parentage = CandidateParentage(parent_run_ids=("parent-run-uuid-123",))
        assert resolve_derived_from(parentage) == "parent-run-uuid-123"

    def test_duplicate_parent_resolves_to_the_single_id_not_an_error(self) -> None:
        """A real, non-error case: the same parent listed twice must resolve cleanly,
        not raise -- proving the multi-parent check is genuinely about distinctness."""
        parentage = CandidateParentage(
            parent_run_ids=("parent-run-uuid-123", "parent-run-uuid-123")
        )
        assert resolve_derived_from(parentage) == "parent-run-uuid-123"

    def test_blank_entries_ignored_when_resolving_single_real_parent(self) -> None:
        parentage = CandidateParentage(parent_run_ids=("", "parent-run-uuid-123", "   "))
        assert resolve_derived_from(parentage) == "parent-run-uuid-123"

    def test_genuine_multi_parent_raises_named_exception(self) -> None:
        """The AC's flagship case: a real, actual multi-parent merge (two genuinely
        distinct parent run IDs, not the same one twice) must raise the named exception,
        never silently pick one parent or drop lineage."""
        parentage = CandidateParentage(
            parent_run_ids=("parent-run-uuid-AAA", "parent-run-uuid-BBB")
        )

        with pytest.raises(MultiParentLineageUnsupportedError) as exc_info:
            resolve_derived_from(parentage)

        # The exception must name the offending parents, not just say "multi-parent".
        assert exc_info.value.parent_run_ids == ("parent-run-uuid-AAA", "parent-run-uuid-BBB")
        message = str(exc_info.value)
        assert "parent-run-uuid-AAA" in message
        assert "parent-run-uuid-BBB" in message

    def test_multi_parent_with_three_distinct_parents_names_all_three(self) -> None:
        parentage = CandidateParentage(parent_run_ids=("run-a", "run-b", "run-c"))

        with pytest.raises(MultiParentLineageUnsupportedError) as exc_info:
            resolve_derived_from(parentage)

        assert exc_info.value.parent_run_ids == ("run-a", "run-b", "run-c")
        message = str(exc_info.value)
        assert "run-a" in message
        assert "run-b" in message
        assert "run-c" in message

    def test_multi_parent_masked_by_duplicates_and_blanks_still_raises(self) -> None:
        """Two genuinely distinct real parents, padded with duplicates and blanks --
        must still raise, proving the dedup/blank-filter logic doesn't accidentally
        mask a real multi-parent case down to one."""
        parentage = CandidateParentage(
            parent_run_ids=("run-a", "", "run-a", "run-b", "  ", "run-b")
        )

        with pytest.raises(MultiParentLineageUnsupportedError) as exc_info:
            resolve_derived_from(parentage)

        assert exc_info.value.parent_run_ids == ("run-a", "run-b")


# ---------------------------------------------------------------------------
# record_candidate_run: threading + fail-loud-before-any-call
# ---------------------------------------------------------------------------


class TestRecordCandidateRun:
    def test_single_parent_threads_derived_from_to_adapter_run(self) -> None:
        adapter = _RecordingAdapter()
        parentage = CandidateParentage(parent_run_ids=("parent-run-uuid-123",))

        result = record_candidate_run(
            adapter,  # type: ignore[arg-type]
            "candidate.py",
            parentage,
            campaign_id="camp-1",
        )

        assert isinstance(result, CandidateRunResult)
        assert len(adapter.run_calls) == 1
        call = adapter.run_calls[0]
        assert call["script_path"] == "candidate.py"
        assert call["derived_from"] == "parent-run-uuid-123"
        assert call["campaign_id"] == "camp-1"

    def test_root_candidate_threads_empty_string_derived_from(self) -> None:
        adapter = _RecordingAdapter()

        record_candidate_run(adapter, "candidate.py", CandidateParentage())  # type: ignore[arg-type]

        assert adapter.run_calls[0]["derived_from"] == ""

    def test_duplicate_parent_threads_the_single_id(self) -> None:
        adapter = _RecordingAdapter()
        parentage = CandidateParentage(parent_run_ids=("run-a", "run-a"))

        record_candidate_run(adapter, "candidate.py", parentage)  # type: ignore[arg-type]

        assert adapter.run_calls[0]["derived_from"] == "run-a"

    def test_genuine_multi_parent_raises_before_adapter_run_is_ever_called(self) -> None:
        """The sharpest fail-loud test: a genuine multi-parent case must raise BEFORE
        `adapter.run` is invoked at all -- not after a partial call, not after `adapter.run`
        raises something else. The spy adapter proves `run` was never reached: fail loud
        and early, not after a partial/corrupted write."""
        adapter = _RecordingAdapter()
        parentage = CandidateParentage(parent_run_ids=("run-a", "run-b"))

        with pytest.raises(MultiParentLineageUnsupportedError):
            record_candidate_run(adapter, "candidate.py", parentage)  # type: ignore[arg-type]

        assert adapter.run_calls == [], (
            "adapter.run must never be called for a genuine multi-parent parentage -- "
            "the exception must fire before any bathos call, not after"
        )

    def test_exception_is_not_swallowed_by_record_candidate_run(self) -> None:
        """record_candidate_run must not catch-and-suppress MultiParentLineageUnsupportedError
        anywhere in its own code path -- it must propagate to the caller unmodified."""
        adapter = _RecordingAdapter()
        parentage = CandidateParentage(parent_run_ids=("run-a", "run-b", "run-c"))

        with pytest.raises(MultiParentLineageUnsupportedError) as exc_info:
            record_candidate_run(adapter, "candidate.py", parentage)  # type: ignore[arg-type]

        # The real exception instance, with its full parent list intact -- not a generic
        # Exception, not a re-wrapped/truncated message.
        assert exc_info.value.parent_run_ids == ("run-a", "run-b", "run-c")

    def test_other_kwargs_forwarded_unchanged(self) -> None:
        adapter = _RecordingAdapter()
        parentage = CandidateParentage(parent_run_ids=("run-a",))

        record_candidate_run(  # type: ignore[arg-type]
            adapter,
            "candidate.py",
            parentage,
            args=["--seed", "1"],
            campaign_id="camp-9",
            output_paths=["out.json"],
            tags=["loop-controller"],
            agent_mode="autonomous",
            no_sidecar=True,
        )

        call = adapter.run_calls[0]
        assert call["args"] == ["--seed", "1"]
        assert call["campaign_id"] == "camp-9"
        assert call["output_paths"] == ["out.json"]
        assert call["tags"] == ["loop-controller"]
        assert call["agent_mode"] == "autonomous"
        assert call["no_sidecar"] is True
