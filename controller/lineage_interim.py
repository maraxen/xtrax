"""Single-parent `derived_from` lineage interim (epic #3611 loop-controller, LC-08, AC-7).

AC-7's text (verbatim): "lineage interim wiring -- single-parent `derived_from` usage via
`AC-5`'s adapter for candidate parentage. AC: explicit fail-loud behavior -- if the
controller ever attempts to record a genuinely multi-parent merge (a ratcheted best-so-far
candidate with more than one real parent) under this single-parent interim, it MUST raise a
named exception, never silently pick one parent or silently drop lineage. Revisit trigger:
re-evaluate migrating off this interim the first time that exception actually fires in a real
campaign (not 'any time,' per the pre-mortem's own warning against an unprioritized backlog
item). Also file the non-blocking bathos-side backlog item for a true `campaign_edges`/
`run_edges` MCP tool." See
`.praxia/docs/specs/260716_loop-controller-epic-architecture-resolved.md` ("AC-7", "lineage
interim", "derived_from", "Finding 2").

## Why this is an interim, not the final mechanism

bathos supports true multi-parent lineage at the database level (`campaign_edges`/
`run_edges`), but neither has an MCP tool or CLI verb today, and writing those edges directly
would bypass MCP's validation/audit surface -- a different, and worse, risk category than
LC-07's pure-read gaps (b)/(c) (see the architecture spec's Finding 2, "Fork 2 -- (d)"). The
only lineage mechanism this epic can go through MCP for right now is `run`'s own
`derived_from: str = ""` parameter (a *single* parent run ID), already wired end-to-end by
LC-06's `BathosCampaignAdapter.run` (`controller/bathos_campaign_adapter.py`). This module is
the controller-side policy layer sitting in front of that adapter: it decides what single
`derived_from` value (if any) to pass, and it is the one place responsible for refusing --
loudly, before any bathos call happens -- to silently collapse a genuine multi-parent merge
into a single parent.

## Fail-loud, not fail-soft

The AC is explicit that this must never "silently pick one parent or silently drop lineage."
`resolve_derived_from` (and `record_candidate_run`, which calls it) raise
`MultiParentLineageUnsupportedError` the moment more than one *distinct, real* parent run ID
is present -- before `BathosCampaignAdapter.run` (and therefore before any bathos MCP call, any
write, any partial state) is ever invoked. "Real" deliberately excludes blank/whitespace-only
entries (not an identifier at all) and de-duplicates identical IDs (the same parent listed
twice is one parent, not a multi-parent merge) -- see `CandidateParentage.distinct_real_parents`
for the exact rule.

## Revisit trigger (per the AC, not "any time")

Migrating off this interim onto a real `campaign_edges`/`run_edges` MCP tool should be
re-evaluated the first time `MultiParentLineageUnsupportedError` actually fires in a real
campaign -- not on an unprioritized "someday" backlog item. The session's own pre-mortem
(architecture spec, "Pre-mortem Record") named exactly that failure mode: an unprioritized
backlog item for the real mechanism sitting unaddressed while the interim silently
misrepresents lineage. Raising loudly here is what makes the trigger observable at all.

## Backlog follow-up (LC-13, not filed by this module)

The AC also asks that a non-blocking bathos-side backlog item be filed for a true
`campaign_edges`/`run_edges` MCP tool. That is exactly epic #3611's `LC-13` ("AC-10 cross-repo
gap backlog filing"), filed separately by the orchestrating session -- this module does not
call any backlog-filing tool itself, to avoid a duplicate/conflicting entry. This docstring is
the record of that cross-reference.
"""

from dataclasses import dataclass, field

from controller.bathos_campaign_adapter import BathosCampaignAdapter, CandidateRunResult


class MultiParentLineageUnsupportedError(Exception):
    """A candidate's parentage names more than one distinct, real parent run -- unsupported
    by the single-parent `derived_from` interim (AC-7).

    Raised by `resolve_derived_from`/`record_candidate_run` *before* any
    `BathosCampaignAdapter` call happens, so lineage is never silently collapsed to one parent
    or silently dropped, and no bathos write can be left in a partial state as a side effect of
    this failure. See the module docstring's "Revisit trigger" section: the first real-campaign
    firing of this exception is the signal to re-evaluate migrating off this interim onto a true
    multi-parent lineage mechanism (tracked as LC-13, not "any time").
    """

    def __init__(self, parent_run_ids: tuple[str, ...]) -> None:
        self.parent_run_ids = parent_run_ids
        msg = (
            f"genuinely multi-parent lineage attempted under the single-parent derived_from "
            f"interim (AC-7): {len(parent_run_ids)} distinct real parent run ID(s) "
            f"{list(parent_run_ids)!r} -- bathos has no campaign_edges/run_edges MCP tool "
            f"today (tracked as LC-13), so this cannot be recorded without either silently "
            f"picking one parent or silently dropping lineage, both of which this interim "
            f"refuses to do. Raise this to a human/backlog decision rather than retrying."
        )
        super().__init__(msg)


@dataclass(frozen=True, slots=True)
class CandidateParentage:
    """A candidate's proposed parentage: the run ID(s) it was derived from.

    Attributes:
        parent_run_ids: Every parent run ID the caller believes contributed to this
            candidate, in whatever order the caller has them. May be empty (a root
            candidate with no lineage), contain one entry (the common case), contain the
            same ID repeated (not genuinely multi-parent -- see `distinct_real_parents`),
            or contain more than one genuinely distinct ID (a true multi-parent merge, e.g.
            a ratcheted best-so-far candidate combining two prior candidates -- exactly the
            case `resolve_derived_from` refuses to collapse silently).

    This type intentionally does not restrict `parent_run_ids` to length <= 1 at
    construction time: the controller's main-loop logic may legitimately *construct* a
    multi-parent parentage (e.g. when proposing a merge of two best-so-far candidates) --
    the single-parent interim's job is to refuse to *record* it via `derived_from`, not to
    make it unrepresentable.
    """

    parent_run_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def distinct_real_parents(self) -> tuple[str, ...]:
        """Distinct, non-blank parent run IDs, in first-seen order.

        Blank/whitespace-only entries are filtered out as not being a real parent
        identifier at all (an empty slot, not a zeroth parent). Duplicate IDs collapse to
        one entry -- the same parent listed twice is one real parent, not a multi-parent
        merge (see the module docstring's self-audit note on this exact distinction).
        """
        seen: dict[str, None] = {}
        for parent_run_id in self.parent_run_ids:
            stripped = parent_run_id.strip()
            if stripped:
                seen.setdefault(stripped, None)
        return tuple(seen.keys())

    @property
    def is_multi_parent(self) -> bool:
        """True iff this parentage names more than one distinct, real parent."""
        return len(self.distinct_real_parents) > 1


def resolve_derived_from(parentage: CandidateParentage) -> str:
    """Resolve the single `derived_from` value to pass to `BathosCampaignAdapter.run`.

    Args:
        parentage: the candidate's proposed parentage.

    Returns:
        `""` if `parentage` has no real parent (a root candidate -- matches
        `BathosCampaignAdapter.run`'s own `derived_from: str = ""` default/pass-through
        convention), or the single distinct real parent run ID if there is exactly one.

    Raises:
        MultiParentLineageUnsupportedError: `parentage` names more than one distinct, real
            parent run ID. Raised here, before any bathos call -- callers must not catch
            this and fall back to picking `distinct_real_parents[0]` or similar; that is
            precisely the silent-degradation behavior the AC forbids.
    """
    distinct = parentage.distinct_real_parents
    if len(distinct) > 1:
        raise MultiParentLineageUnsupportedError(distinct)
    return distinct[0] if distinct else ""


def record_candidate_run(
    adapter: BathosCampaignAdapter,
    script_path: str,
    parentage: CandidateParentage,
    *,
    args: list[str] | None = None,
    campaign_id: str = "",
    output_paths: list[str] | None = None,
    tags: list[str] | None = None,
    agent_mode: str = "",
    no_sidecar: bool = False,
) -> CandidateRunResult:
    """Resolve `parentage` to a single `derived_from` value and record the run via LC-06's
    `BathosCampaignAdapter.run` -- the controller-side glue this AC exists to provide.

    `resolve_derived_from` is called first and unconditionally: a
    `MultiParentLineageUnsupportedError` therefore always surfaces *before*
    `adapter.run` (and so before any bathos MCP call, any write, any campaign-attachment
    side effect) is ever invoked -- fail loud and early, not after a partial/corrupted
    write. Every other argument is forwarded to `adapter.run` unchanged; this function adds
    no policy beyond the `derived_from` resolution itself.

    Args:
        adapter: the `BathosCampaignAdapter` (LC-06) to record the run through. Inject a
            test double here for unit tests -- this function never constructs its own
            adapter or talks to bathos directly.
        script_path: forwarded to `adapter.run`.
        parentage: this candidate's proposed parentage, resolved via `resolve_derived_from`.
        args: forwarded to `adapter.run`.
        campaign_id: forwarded to `adapter.run`.
        output_paths: forwarded to `adapter.run`.
        tags: forwarded to `adapter.run`.
        agent_mode: forwarded to `adapter.run`.
        no_sidecar: forwarded to `adapter.run`.

    Returns:
        The `CandidateRunResult` from `adapter.run`.

    Raises:
        MultiParentLineageUnsupportedError: `parentage` names more than one distinct, real
            parent run ID -- raised before `adapter.run` is called.
        BathosMcpToolError: `adapter.run` itself failed (bathos-side).
        BathosMcpTransportError: `adapter.run` itself failed (transport-side).
        BathosTokenMissingError: `adapter.run` itself failed (no local write token).
    """
    derived_from = resolve_derived_from(parentage)
    return adapter.run(
        script_path,
        args=args,
        derived_from=derived_from,
        campaign_id=campaign_id,
        output_paths=output_paths,
        tags=tags,
        agent_mode=agent_mode,
        no_sidecar=no_sidecar,
    )


__all__ = [
    "CandidateParentage",
    "MultiParentLineageUnsupportedError",
    "record_candidate_run",
    "resolve_derived_from",
]
