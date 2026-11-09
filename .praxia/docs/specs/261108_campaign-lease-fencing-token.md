---
title: Campaign lease fencing token (#3312)
description: Spec for a fencing token on xtrax campaign leases so a stalled holder cannot clobber a stealer's commit, abort cleanup or lock.
status: reviewed
task_id: 261108_campaign-debt-specs
date: '261108'
backlog_ids: ''
adversarial_review: 'round 1 (261109): Antigravity challenger, surface_dispatch f1e7eb4e, SOUND_WITH_FIXES; 11 findings, all verified against code and folded in (section 7)'
---
# Campaign lease fencing token (#3312)

Debt #3312. Base: main d406ade (PR #267). All `file:line` refs are to `src/xtrax/campaign/` and were read at HEAD 63f3a60, where `lease.py`, `executors/local.py` and `marker.py` are unchanged from d406ade.

## 1. Problem

A lease is a time promise. The stealer's test is `now >= lease_expires + 300 s` AND `mtime_age >= lease_seconds + 300 s` (`lease.py:340-343`). A holder that stalls longer than that (SIGSTOP, VM pause, hung NFS write, swap storm) loses the lease without knowing. When it wakes, nothing it holds identifies it as stale: every destructive step trusts a check made before the stall.

### Interleaving A: stall between commit step 5 and step 6

1. Holder H passes the step-5 re-verification: lock still H's, unexpired, no abort (`lease.py:545-560`).
2. H stalls for more than lease + 300 s.
3. Stealer S steals: `os.replace(steal_tmp, lock_path)` (`lease.py:382`), then `_clear_stale_commit_token` deletes H's `.commit` because the owner differs (`lease.py:386`, `420-442`). S runs the unit and commits steps 3-7 with its own `.commit`.
4. H wakes at step 6. `output_path.exists()` is true (S's store), so H calls `quarantine_output(output_path)` (`lease.py:574-577`). That moves S's store and S's `.done.json` aside (`marker.py:309-311`).
5. H then does `os.replace(partial, output_path)` (`lease.py:578`) and `os.replace(tmp_done, done_marker_path)` (`lease.py:584`). If S was between its own steps 6 and 7 when H ran, S's step 7 then overwrites H's marker with a marker for S's digest over H's store. Tier-1 resume (marker plus root metadata only, `marker.py:370-380`) accepts the torn pair; only `deep` resume catches the digest mismatch.

H's `.commit` unlink at `lease.py:592` is unconditional, so it can also delete S's token and let a third committer in.

### Interleaving B: stall before step 8 or release

After step 7, H unlinks `.lock` with no owner check (`lease.py:600-602`). If H stalled before it, H deletes S's live lock, so a third worker acquires with S still running. `release()` reads the owner, then unlinks (`lease.py:463-468`): a check-then-act with a stall allowed between.

### Interleaving C: abort cleanup

`_execute_single_unit`'s `finally` runs while H may no longer hold the lease (`executors/local.py:484-502`). If H failed before staging (`not staged`), it quarantines whatever `output_path` holds that fails validation (`local.py:490-498`). S's in-progress store (a unit that writes straight to `output_path`, `_stage_unit_output` case 1 at `local.py:335-338`) has no marker yet, so it is invalid and gets quarantined mid-write. `lock.release()` follows (`local.py:500-501`).

### Interleaving D: heartbeat replace

The renewal checks its time guard (`lease.py:205`), reads and CAS-checks the lock (`lease.py:209-217`), writes and fsyncs a tmp file (`lease.py:231-233`), then `os.replace` (`lease.py:242`). A stall between the guard and the replace (the existing test parks exactly in `fsync_file`, `tests/campaign/test_heartbeat.py:20-57`) lets H overwrite S's lock with H's, with a passing post-verify (`lease.py:246-254`). S's own renewal then fails its CAS, S's abort event sets, and S's commit step 5 raises `LeaseLostError`. The two workers hold contradictory views, and H's lock hides S.

### Related: reclaim

`_reclaim_dead_worker_leases` deletes all `<out>.partial.*` (`local.py:734-738`) before re-reading the lock bytes (`local.py:740`). The re-read guards only the lock unlink. The window is tiny and it runs after the pool has shut down, but it is a destructive step with no fence.

## 2. Design

### Token

Add `fence: int` to the lock file (`CampaignLock`, `lease.py:55-90`; `HeartbeatThread` rewrite dict at `lease.py:222-230`). A holder's identity becomes the triple `(owner_token, fence, acquired_at_unix_s)`. Heartbeat CAS already compares owner and acquired_at (`lease.py:212-216`); it gains `fence`.

Bumping. A persistent high-water file `<out>.fence` (plain integer, never unlinked by release) survives the lock's unlink on release, so the counter does not reset.

- Fresh acquire (`lease.py:295`, serialized by `O_EXCL` on `.lock`): `fence = read(.fence, default 0) + 1`, written into the lock body, then the holder writes `.fence = fence` by tmp plus `os.replace`. Only the lock holder writes `.fence`.
- Steal (`lease.py:353-389`, serialized by the `.steal` token plus byte-CAS): `fence = max(read(.fence), stolen_lock.fence) + 1`.
- `.fence` only rises (finding 2). The write is: re-read the lock and confirm it is still ours (owner, fence, acquired_at); re-read `.fence`; replace only if our fence is greater. A holder that stalled between creating the lock and writing `.fence` therefore cannot roll back a stealer's higher value. The read-compare-replace is itself check-then-act, so `.fence` can still lag in the R1 window; it never moves backward through this path except in that window.

If a holder dies between writing the lock and `.fence`, the next acquirer can reuse the same fence value. That is acceptable because every check compares the whole triple including `owner_token`. The fence supplies ordering where order is needed (markers, forensics, the `.fence` check in `_assert_holder`); owner uniqueness supplies identity.

### Check before each destructive step

One helper, `FileLock._assert_holder()`: read the lock; require `owner_token`, `fence`, `acquired_at` equal to ours; require our fence >= `read(.fence)` (finding 8: catches a stale heartbeat that clobbered a stealer's lock, since the stealer raised `.fence`); no `abort_event`; and the local lease deadline not passed. Raise `LeaseLostError` otherwise.

Local deadline (finding 1). It is NOT fixed at acquire. It starts at `boottime_at_acquire + lease_seconds` and each successful heartbeat renewal advances it to `boottime_sampled_before_the_renewal_write + lease_seconds`, set only after the post-verify passes (`lease.py:246-254`). Sampling before the write keeps it conservative. `boottime` is `time.clock_gettime(time.CLOCK_BOOTTIME)` on Linux and `time.monotonic()` elsewhere, so neither a wall-clock step nor a suspend extends it. A healthy holder whose heartbeat renews keeps passing however long the unit runs (test 9).

Call `_assert_holder()` immediately before:

| Step | Site | Change |
|---|---|---|
| steal replace | `lease.py:382` | re-read our `.steal` token body (owner) and redo the lock byte-CAS immediately before `os.replace`; abort the steal otherwise (finding 6: a stealer stalled past the token-reclaim age must not clobber a later stealer) |
| lost-lease `.commit` unlink | `lease.py:563-565` | unlink only if the token body's `owner_token` and `fence` are ours (finding 3) |
| quarantine of orphan | `lease.py:574-577` | check first |
| promote store | `lease.py:578` | check first |
| marker replace | `lease.py:584` | check first |
| `.commit` unlink | `lease.py:592` | unlink only if the token body's `owner_token` and `fence` are ours |
| `.lock` unlink | `lease.py:600-602`, `463-468` | unlink only if the body matches the triple |
| orphan set-aside before the unit runs | `local.py:246` (`_set_aside_orphaned_output` → `quarantine_output`) | pass the lock in; check first (finding 4) |
| staging move of a unit-written store | `local.py:337` (`os.replace(output_path, partial_path)` in `_stage_unit_output`) | pass the lock in; check first (finding 5: otherwise a stale holder moves the stealer's store into its own partial and abort cleanup deletes it) |
| abort cleanup | `local.py:490-498` | skip quarantine unless `_assert_holder()` passes |
| heartbeat replace | `lease.py:242` | re-read and compare the triple immediately before `os.replace` |
| reclaim | `local.py:734-740` | compare the lock bytes before removing partials AND again immediately before the lock unlink (finding 7; the existing re-read at `:740` stays) |

`.commit` bodies gain `fence` (written at `lease.py:532-536`). `_clear_stale_commit_token` (`lease.py:439`) keeps its owner test.

### Make the residual harmless where possible

1. The done marker gains optional `lease_fence: int | None` and `owner_token: str | None` fields on `CampaignDoneMarker` (frozen, slots: `marker.py:82-109`), set in `commit_output` with `dataclasses.replace` before serializing (finding 9). `from_dict` drops unknown keys (`marker.py:114-130`), so the fields must be declared there too. The marker schema version is NOT bumped: `validate_unit_done` requires the exact schema string (`marker.py:427-428`), so a bump would invalidate every existing marker.
2. After step 7 the finisher re-reads the lock; if it is no longer the holder it logs `lease_lost_after_commit` with both fences and raises `LeaseLostError` instead of reporting success. Store and marker stay for the next resume to verify with `deep`.
3. `StolenAttemptDiscarded` on the stealer's side stays as is.

### What the token does and does not guarantee

- Does: close every window except the gap between a check and the following `rename`/`unlink`. Today the windows are lease + 300 s wide; after this they are one syscall pair wide, except step 6, where `quarantine_output` is several syscalls (`marker.py:288-312`), so its window is that whole call (finding 10).
- Relies on NFS close-to-open consistency: each check opens and reads the lock file, which revalidates cached attributes on open. A mount with `nocto` or a long `actimeo` can serve a stale lock without any stall; such mounts are unsupported for campaign output (finding 10).
- Does not: make check-then-act atomic. POSIX has no conditional rename on NFS or Lustre, and `flock` is unreliable there. A stall in exactly the check-to-rename gap, longer than lease + 300 s, still reproduces interleaving A step 5. This is residual risk R1: detectable (marker `lease_fence` older than the current `.fence`, or a `deep` digest mismatch) but not preventable without per-attempt final paths.
- Does not: fence the unit function. It protects the commit steps only; a unit that writes outside `.partial.<uuid>` or straight to `output_path` is unprotected until commit.
- Gives no cross-host clock guarantee. `time.monotonic()` bounds the local deadline; the stealer's wall-clock test is unchanged.

### Rejected alternative: token-named final paths

Promote to `<out>.attempt.<fence>` and make readers resolve through a pointer. A stale rename becomes harmless, but the output layout that consumers (aminx adapters, `xtrax.pack`) read at a fixed name changes. Out of scope; revisit if R1 is ever observed.

## 3. Backward compatibility

| Case | Behavior |
|---|---|
| Lock without `fence` (0.4.1b7-era) | `from_dict` already uses `.get` defaults (`lease.py:81-90`); default `fence = 0`. Steal sets `max(.fence or 0, 0) + 1`. |
| New code reads old lock | works, as above |
| Old code renews a new lock | old `HeartbeatThread` rewrites the dict without `fence` (`lease.py:222-230`). New code reads 0 and compares `fence` only when both sides carry one. Documented mixed-fleet gap. |
| Marker without `lease_fence` | `from_dict` default None; `validate_unit_done` ignores the field. Schema string unchanged. |
| No `.fence` file | treated as 0 |
| `.fence` visible to old code | new sibling name; the only `<out>.*` glob in `local.py`/`lease.py` is `.partial.*` (`local.py:734`) |

No on-disk campaign needs migration.

## 4. Test plan (red before green)

Fixtures reuse the park-in-`fsync_file` pattern (`test_heartbeat.py:20-57`) and the hand-written lock payloads of `test_lease.py:273-362` and `test_commit_protocol.py:187-229`. No sleeps: events plus an injectable clock (`clock`/`monotonic` callables on `FileLock` and `HeartbeatThread`, defaults unchanged).

1. `test_stalled_holder_cannot_quarantine_stealer_store` (A). Holder acquires, stages a partial, parks (event) at the start of step 6 via a monkeypatched hook. Advance the injected clock past lease + 300 s and age the lock mtime with `os.utime`. Stealer acquires and commits fully. Release the holder. Assert: holder raises `LeaseLostError`; stealer's store and marker digest are unchanged; no quarantine directory exists. Red today: store and marker are quarantined.
2. `test_stalled_holder_does_not_unlink_stealer_lock_or_commit_token` (B, finding 3). Two cases, because a holder parked after step 7 has already stamped `.done.json`, so a stealer discards at step 1 and never creates `.commit` (finding 11). (a) `.lock`: holder parks before step 8; stealer holds a live lock; after resume `.lock` still carries the stealer's body. (b) `.commit`: holder parks between step 4 and step 5; stealer steals and creates its own `.commit`; holder resumes, takes the lost-lease path (`lease.py:563-565`); the stealer's `.commit` survives.
3. `test_abort_cleanup_does_not_quarantine_stealer_output` (C). Drive `_execute_single_unit` with a unit function that parks; steal meanwhile; stealer writes a partial output at `output_path`; holder raises. Assert the stealer's directory is untouched.
4. `test_heartbeat_replace_after_steal_is_refused` (D). Extend the parked-fsync test: steal while the renewal is parked, resume, assert the lock still carries the stealer's owner and fence.
5. `test_fence_monotonic_across_release_and_steal`. acquire, release, acquire gives fence 1 then 2. Steal from a fence-7 lock gives at least 8. Lock without `fence` gives 1.
6. `test_old_format_lock_and_marker_accepted`. Locks and markers without the new fields acquire, steal, validate and resume.
7. `test_reclaim_rereads_lock_before_removing_partials`. Rewrite the lock between read and partial removal; the other owner's partials survive. Also rewrite it during partial removal; the lock is not unlinked.
9. `test_renewing_holder_passes_assert_holder_past_lease_seconds` (finding 1). Injected clock: acquire with a short lease, run renewals past `lease_seconds`, then commit succeeds. Red against an acquire-time deadline.
10. `test_stale_holder_cannot_roll_back_fence` (finding 2). Holder parks between lock create and `.fence` write; stealer raises `.fence`; holder resumes; `.fence` keeps the stealer's value and the holder's next `_assert_holder` fails.
11. `test_clobbering_heartbeat_fails_assert_holder` (finding 8). Force the residual: holder's heartbeat replace lands over a stealer's lock (bypassing the replace-time check via a hook); the holder's next `_assert_holder` raises because its fence < `.fence`.
12. `test_stalled_holder_cannot_set_aside_or_stage_stealer_store` (findings 4, 5). Park the holder after acquire before `_set_aside_orphaned_output`, and separately after `unit_fn` returns before `_stage_unit_output`; stealer writes its store at `output_path`; the store survives both.
13. `test_stalled_stealer_cannot_clobber_later_stealer` (finding 6). Stealer S1 parks before the steal `os.replace`; age its `.steal` token past the reclaim age; S2 steals; S1 resumes and aborts; the lock carries S2.
14. Regression: `test_heartbeat.py`, `test_lease.py`, `test_r3_lease.py`, `test_commit_protocol.py`, `test_concurrency_stress.py` stay green. Narrow files only; never the whole suite on this box.

For 1-4 and 12-13, commit each test against unmodified code and show the failing assertion before the fix. 9-11 test new mechanisms and are red against a naive implementation of the first draft.

## 5. Assumptions

| # | Assumption | Status | Changes design if wrong? |
|---|---|---|---|
| 1 | Steal needs expiry + 300 s and mtime age >= lease + 300 s | verified (`lease.py:340-343`) | no |
| 2 | Step 6 quarantines an existing `output_path` | verified (`lease.py:574-577`, `marker.py:296-311`) | no |
| 3 | `.commit` and `.lock` unlinks are unconditional | verified (`lease.py:592`, `600-602`) | no |
| 4 | Abort cleanup quarantines an invalid `output_path` with no lease check | verified (`local.py:490-498`) | no |
| 5 | Heartbeat replace follows the CAS read with no recheck | verified (`lease.py:209-242`) | no |
| 6 | `validate_unit_done` requires the exact schema string | verified (`marker.py:427-428`) | yes: no bump |
| 7 | No fencing code to reuse | verified by grep of `~/projects`, `~/repos` (`*.py`, `*.rs`): only unrelated "context fencing" in hermes-agent and pydantic-ai | no |
| 8 | Marker consumers (aminx/alynxr adapters, `xtrax.pack`) tolerate extra marker keys | verified 261109: `CampaignDoneMarker.from_dict` reads named keys with `.get` and drops the rest (`marker.py:114-130`); `run/unit_index_adapters.py:106-147` and aminx `host/campaign.py:969-1010` likewise; no strict parse in `xtrax.pack` (Haiku recon). alynxr `unit_pack.py` not read | no; new fields must be added to `CampaignDoneMarker` or they are dropped on read |
| 9 | Nothing outside `local.py`/`lease.py` globs `<out>.*` and would pick up `.fence` | verified 261109 for xtrax: `cli/campaign_verb.py:113` rglobs `*.done.json`/`*.lock`, `:417`/`:422` `.quarantine`/`*.corrupt.*`; `local.py:734` `.partial.*`; `pack/_node.py:316,744` walk the unit dir only. aminx/alynxr globs not audited | no |
| 10 | `os.replace` is atomic within one directory on Lustre and NFS | unverified (inherited from the existing protocol) | no |
| 11 | `time.monotonic()` advances across SIGSTOP (CLOCK_MONOTONIC does; VM suspend may not) | verified (Linux semantics): `time.monotonic()` is CLOCK_MONOTONIC, which advances across SIGSTOP and stops during system suspend; CLOCK_BOOTTIME advances across both | yes: the local deadline uses `time.clock_gettime(time.CLOCK_BOOTTIME)` on Linux, `monotonic()` elsewhere |
| 12 | A stalled process freezes its heartbeat thread too | no longer load-bearing (261109): with the renewal-advanced local deadline, a heartbeat-only stall stops renewals, the deadline lapses and the main thread's next `_assert_holder` fails; a main-thread-only stall keeps the lease renewed, so nothing is stolen | no |
| 13 | Mixed-version fleets mid-campaign are not a supported mode | unverified | yes: otherwise add a lock schema version |
| 14 | `executors/slurm.py` takes locks only through `FileLock` | verified 261109: array tasks run `campaign run --executor local-seq --row-index` (`slurm.py:504-510`) into `_execute_single_unit`; the only `FileLock(` and `.lock` O_EXCL sites are `lease.py:295` and `local.py:413` | no |

## 6. Out of scope, risks, size

Out of scope: fencing the unit function's own writes; per-attempt final output paths; changing the 300 s grace constant.

Risks:
- R1: residual check-to-rename window (section 2), accepted and detectable.
- R2: the injectable clock touches `HeartbeatThread` and `FileLock` signatures; keep defaults identical so existing tests do not move.
- R3: `_assert_holder` adds 3-5 small reads per commit on NFS; negligible against unit runtime.
- R4: a `.fence` write failure after the lock create; handled by the triple compare, covered by test 5.

Size (revised 261109): about 250-300 lines in `lease.py` and `local.py`, about 15 in `marker.py`, about 550 lines of tests. One PR. No bathos pre-registration: a correctness fix with deterministic tests, no finding.

## 7. Adversarial review log

Round 1, 261109. Challenger: Antigravity headless via `surface_dispatch` (dispatch `f1e7eb4e-de6a-4941-bc88-b72b3cf15cad`, role reviewer, document mode, admitted, containment `agent_sandbox`), on commit d23f195. Verdict SOUND_WITH_FIXES. Each finding was checked against the code at the cited lines by the orchestrator before it was accepted. No defender round: no blocker was disputed.

| # | Sev | Finding | Disposition |
|---|---|---|---|
| 1 | blocker | Acquire-time local deadline fails a healthy renewing holder | accepted; deadline advances on each verified renewal (section 2), test 9 |
| 2 | blocker | Stalled fresh acquirer rolls `.fence` back after a steal | accepted; rise-only, holder-checked `.fence` write, test 10 |
| 3 | blocker | Lost-lease path unlinks `.commit` unconditionally (`lease.py:563-565`) | accepted; owner+fence-checked unlink, test 2(b) |
| 4 | major | `_set_aside_orphaned_output` quarantines with no holder check (`local.py:246`) | accepted; table row, test 12 |
| 5 | major | `_stage_unit_output` moves a unit-written store with no holder check (`local.py:337`) | accepted; table row, test 12 |
| 6 | major | Stalled stealer clobbers a later stealer at `lease.py:382` | accepted; steal-token and byte-CAS recheck before replace, test 13 |
| 7 | major | Reclaim wording left the lock unlink unguarded | accepted as a wording fix: check before removal and before unlink, test 7 |
| 8 | major | `_assert_holder` never reads `.fence` | accepted; fence >= `.fence` check, test 11 |
| 9 | minor | `CampaignDoneMarker` is frozen | accepted; `dataclasses.replace` |
| 10 | minor | R1 wider than one syscall pair at step 6; NFS attribute caching | accepted; stated in section 2 |
| 11 | minor | Test 2 cannot produce a stealer `.commit` with the holder parked after step 7 | accepted; test 2 split |

Open decision carried to implementation: assumption 13 (mixed-version fleets). Default: unsupported, documented in the changelog; old-format locks still acquire and steal (test 6).

