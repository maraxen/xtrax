---
title: Campaign row hash covers output_locator (#3308)
description: Spec to bind a manifest row's output_locator to its hash without touching input_hash, so aminx parity (GATE-11) is unaffected.
status: draft
task_id: 261108_campaign-debt-specs
date: '261108'
backlog_ids: ''
adversarial_review: ''
---
# Campaign row hash covers output_locator (#3308)


Status: draft for a later implementation session. Base: main d406ade. aminx read at origin/main 86042c0e.

## Key finding: GATE-11 does not constrain manifest_row_hash

- GATE-11 (`tests/campaign/test_aminx_parity.py:27-160`) compares only `input_hash` against golden vectors. For `manifest_row_hash` it asserts 64 hex chars and non-zero (lines 158-159). Nothing in it pins a row hash.
- `input_hash` is computed by `compute_unit_input_hash` (manifest.py) from `params` + identity fields. `output_locator` is not in it and must stay out.
- aminx's `manifest_row_hash` is a different payload (aminx `host/campaign.py:287-305`: `schema_version`, `model_family`, `temperature`, `backbone_noise`, ...). xtrax's row hash (manifest.py:341-356) already differs. The two row hashes are not comparable today.
- aminx also keeps the locator outside its hash: `_row_output_path` (campaign.py:517) derives `<base>/<campaign_id>/<row_hash>.h5` and it is set on the row after hashing (campaign.py:1521).

So the debt's premise ("hashing it breaks GATE-11 parity") holds only for `input_hash`. A change confined to `manifest_row_hash` needs no aminx PR and no golden-vector change.

## 1. Problem and exploit

`output_locator` is a free field on `ManifestRow` (manifest.py:`ManifestRow`), loaded by `from_dict` (manifest.py:~190) and checked by `verify_manifest_row_hashes` (manifest.py:359) over `_compute_unit_hashes`, whose payload (manifest.py:341-356) omits it. `_resolve_output_path` (executors/local.py:141-151) trusts it: absolute paths are returned as is, relative ones join `output_dir`, `..` is not rejected. Default locator is `f"{campaign_id}/{manifest_row_hash}.zarr"` (manifest.py:435), so the locator is derived from the hash but never checked against it.

Exploit A (redirect, destructive): edit row A's `output_locator` to row B's store path in manifest.json. Row hashes still verify. `run`/resume resolves A to B's store. `validate_unit_done` (marker.py:448-465) sees B's marker with B's `manifest_row_hash` != A's, so A is "invalid", then `_set_aside_orphaned_output` (local.py:220-246, called at local.py:435 under A's lease) calls `quarantine_output` on B's valid store. B's finished work is moved aside and recomputed. An absolute locator does the same to any path the user can write.
Exploit B (collision): two rows share one locator; whichever commits second sees a foreign marker and quarantines the first's output, in a loop.
Not an exploit (verified): a hash-mismatched marker is never accepted as valid, so "wrong store trusted" does not occur through `validate_unit_done`. The failure is wrong-store destruction and shared stores. The done marker records `output_locator` as an absolute `str(output_path)` (local.py:471) and `validate_unit_done` never compares it (marker.py:448-480 check only the two hashes). The debt text should be corrected on this point.

## 2. Options

**a. Always hash the locator, both repos, `hash_version` field.** Circular: the default locator contains the row hash. Needs a "locator template" field or a two-pass scheme. Changes every row hash on disk, so all existing markers (keyed by the hash) are orphaned and every unit recomputes. aminx would need a matching change only if parity of row hashes were wanted; it is not (see above). Rejected: maximal churn, no gain over (c).

**b. Marker records locator; refuse mismatch.** Marker already stores it (absolute). Compare resolved locator to marker's. Absolute value breaks when the output dir moves; a relative field needs `campaign_done_marker_v4` and rejects every existing marker. It catches the problem only after a store exists, and `_set_aside_orphaned_output` runs before any marker comparison could help, so exploit A still quarantines B. Insufficient alone.

**c. (Recommended) Cover the locator in the row hash only when it is non-default, and verify at load.**
- Define `default_locator(campaign_id, h) = f"{campaign_id}/{h}.zarr"`.
- Planner rows (all of them: manifest.py:558 passes `output_locator=None`) keep today's hash byte for byte. No on-disk change, existing markers still validate.
- Custom locator (`build_manifest_row(output_locator=...)`): the hash payload gains `"output_locator": <str>`, so the locator is covered.
- `verify_manifest_row_hashes` accepts a row iff either (i) stored hash == hash without locator AND locator == `default_locator(cid, stored_hash)`, or (ii) stored hash == hash with the locator. Editing A's locator to B's fails both: not default for H_A, and H_A != hash-with-locator. Sound because the two payloads differ.
- Hardening, independent of hash: in `_resolve_output_path` reject a resolved path that escapes `output_dir` unless the locator is absolute AND covered by the hash. Flag as optional (risk below).
- `input_hash` untouched, so GATE-11 passes unchanged.
- Disk: manifest.json format unchanged (same fields). Markers unchanged. Only hand-built rows with custom locators get new hashes.

## 3. Migration and back-compat

None released. Planner-produced manifests and their markers validate unchanged under (c). Rows built with custom locators (tests, `bench_out.zarr` in test_p7_challenger_stress.py:414, `unit_NNN.zarr` fixtures in test_r3_cli_slurm.py:48, executors/test_slurm.py:47, executors/test_local.py:82, test_p7_remediation_regression.py:290) change hash; any hardcoded hash or pre-written marker in those fixtures must be regenerated. PR-era on-disk campaigns made by hand-built custom-locator rows fail `from_dict` with `ManifestIntegrityError`, message "re-plan the campaign" (existing text). Do not add a hash_version field; (i)/(ii) is self-describing.

## 4. Test plan

Red first (new tests in tests/campaign/test_manifest.py):
1. Plan a campaign, set row A's `output_locator` to row B's, `CampaignManifest.from_dict` must raise `ManifestIntegrityError`. Fails on d406ade (the field is unhashed).
2. Same for an absolute path and for `../x`.
3. Executor-level (executors/test_local.py): finish B, tamper A in manifest.json, run: loading refuses, B's store and marker untouched (assert no `.quarantine` dir). Fails on d406ade (B quarantined).
4. Custom locator round trip: `build_manifest_row(output_locator="u/x.zarr")` loads; editing that locator fails; two custom rows with different locators have different hashes; same params, different locator gives different row hash but identical `input_hash`.
5. Default-locator hash stability: golden `manifest_row_hash` for the two GOLDEN_VECTORS planner-style rows (computed on d406ade before the change) must not move. This is the back-compat guard.
Parity: GATE-11 file unchanged; add one assertion that `input_hash` is independent of `output_locator`. Optionally add a comment that the gate pins `input_hash` only.

## 5. Assumptions

| Assumption | Tag | Changes recommendation? |
|---|---|---|
| Row hash payload omits locator | verified manifest.py:341-356 | no |
| Default locator derived from row hash | verified manifest.py:435 | no |
| Planner always passes `output_locator=None` | verified manifest.py:558 | no |
| `_resolve_output_path` trusts abs/relative/`..` | verified local.py:141-151 | no |
| Marker validation compares only the two hashes, not the locator | verified marker.py:448-480 | no |
| Quarantine of a foreign valid store under A's lease (exploit A) | read, not executed (local.py:220-246, 435) | no, but step 3 of the test plan must confirm it red |
| GATE-11 pins only `input_hash` | verified test_aminx_parity.py:131-160 | yes, if wrong (a) becomes the only fix |
| aminx row hash is a different payload and keeps locator unhashed | verified aminx campaign.py:287-305, 517, 1521 (origin/main 86042c0e) | yes, if parity of row hashes were required |
| GOLDEN_VECTORS are real aminx production outputs and match aminx's `compute_unit_input_hash` (aminx signature differs: takes a sampling spec payload, campaign.py:902) | unverified | not for this fix; matters to the parity claim itself |
| Nobody outside this repo consumes xtrax `manifest_row_hash` (aminx, alynxr, tev_design) | unverified | yes: an external consumer pinning planner-row hashes is safe under (c) but would break under (a) |
| No consumer relies on absolute or escaping locators | unverified | affects only the optional containment hardening |
| Slurm executor/CLI paths all go through `_resolve_output_path` and `from_dict` | partly verified (slurm.py:61-75, campaign_verb.py:19,566); other `from_dict` bypasses unchecked | no |

## 6. Out of scope, risks, size

- Out of scope: changing `input_hash`; marker schema bump; aminx changes; locator policy for non-zarr sinks.
- Risks: the (i)/(ii) verification must run in one shared function for planning and loading (as `_compute_unit_hashes` does today) so they cannot drift; containment hardening could break a user with a legitimate absolute locator, so gate it on hash coverage.
- Size: ~40 lines in manifest.py, ~10 in local.py, ~120 lines of tests. One xtrax PR. No aminx PR required.
