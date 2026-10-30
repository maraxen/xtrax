# Spec: unit index, output packing and file-count guard (DRAFT v2)

- **task_id**: `261030_xtrax-ledger-pack`
- **status**: DRAFT v2. v1 was adversarially reviewed against the code on 2026-10-30, and §8 lists every
  objection with its evidence and resolution. Two owner decisions are open (§7). No code has been written.
- **consumers**: alynxr, aminx, aminx-ensemble, demistify, prolix
- **origin**: owner request 2026-10-30 (upstream the alynxr provenance/consolidation design into xtrax).
  Owner decisions already taken: (1) recon the five consumers first; (2) the index is **lock-free
  per-writer segments**, not `flock`.

Tags: **V** = read in code at the cited line in this session; **U** = unverified. Every U must be
probed before it changes the design.

## 1. Problem

Each consumer builds its own unit record, completion marker and resume-by-input-hash. Only alynxr packs
outputs, and only for search products. Only alynxr guards file count at run time. On Engaging scratch the
binding limit is file count, not bytes: alynxr hit 1,000,000 files at about 46% of bytes. The inode cost
comes from per-unit loose directories, and nothing consolidates them.

What v1 got wrong: most of the "build" already exists in xtrax and alynxr (§2.2). The upstream work is
mostly **lifting and closing gaps**, not new mechanisms.

## 2. Recon facts

### 2.1 Consumers

| Consumer | Per-unit footprint | Completion record / resume | Packing | Inode guard | xtrax use |
|---|---|---|---|---|---|
| alynxr | fold unit: ~5 + 2R loose files (R U, A7); `fold.py:183`, `readers.py:44-45` | search: `product.json` per product. Fold stats: immutable Parquet parts + one fragment per part, derived `MANIFEST.json` (V, `fold_stats/layout.py:1-30`) | `pack_members`: search products only, `PACK_THRESHOLD = 64`, stdlib-only node module (V, `product_node.py:1-8,50,290`); fold-unit "Archive Stage B" not implemented (V, `fold_stats/readers.py:19-21`) | `FileCapGuard`: `lfs quota` on Lustre, `df -i` elsewhere, re-reads every 1000 files and **refuses mid-run** (V, `storage.py:1-10,270-330`); worker exit 5 (V, `fold_stats/worker.py`) | **yes**, lazily: `fold_stats/array_commit.py` commits array shards through `xtrax.run.zarr_commit` (V) |
| aminx | one zarr store per campaign row + `.lock` + done marker + `.partial.<attempt>` (`host/campaign.py`, origin/main `ef8810c2`) | `O_EXCL` lease lock with CAS steal, 1800 s, or a `DistributedLockBackend`. Done marker `campaign_done_marker_v3`: `manifest_row_hash`, `input_hash`, `input_hash_components`, `content_digest_sha256` (= `xtrax.run.zarr_content_digest`), `attempt_id`, `producer` (V, `campaign.py:1015-1045`). Outcomes: reuse; legacy-v2 → quarantine + recompute; input-hash mismatch → quarantine + recompute; schema stale → invalidate + recompute; content mismatch → **raise** (V, `campaign.py:969-1012,1789-1880`) | none (U) | input dirs over 5000 files are not hashed (V, `campaign.py:65,820`); no output guard (U) | imports `xtrax.run` fsync/digest primitives (V) |
| demistify | residue shards in one durable zarr store; phase5 grid ~144 files | `DurableShardSink` is **demistify's** class (`demistify/shard_sink.py`), wrapping `xtrax.run.ZarrStagingSink` `create_or_join` (V) | none | disk quota only (`phase0_preflight.py:35-39`) | `>=0.4.1b3` |
| prolix | 1 json + optional 1 xtc per run | existence check only (`fit_bonded_hp4.py:257`) | none | none | `>=0.4.1b2` |
| aminx-ensemble | stub code. Its epic spec §3.3 resumes per chunk through `ZarrStagingSink` `create_or_join` + `lookup`, **Stale/Corrupt raise**, and `chunk_size` keeps a store under 100,000 files (V, `aminx-ensemble/.praxia/docs/specs/261030_aminx-ensemble-epic.md` §3.3) | planned on `zarr_commit`, **not** on a separate ledger | not applicable (zarr keys, no loose dirs) | planned: stdlib `_resolve.py` / `_config.check_location`, which records inodes, group and setgid (epic §3.5) | planned |

### 2.2 What xtrax and alynxr already provide (V)

- `xtrax.run.zarr_commit`: staged, fsynced, renamed commit. The commit record (`input_digest`,
  `content_digest`, `run_id`, `env`, `record_digest`) lives **inside the artifact** as the `xtrax.commit`
  attr (`zarr_commit.py:34,364-377`). `lookup` returns `Reuse | Missing | Stale | Corrupt`
  (`:460-512`), and `committed_keys` finds completions by walking the tree (`:515-550`).
  **Gap:** `commit_key` returns `Duplicate` on the input digest alone and never compares content
  (`:399-400`). alynxr patches that gap locally (`array_commit.py` header: lookup(verify) before every
  write, a content compare on Duplicate, quarantine and never delete).
- `xtrax.run.sink`: `SinkSpec` / `make_sink` / `ZarrStagingSink`, with a `create_or_join`
  multi-writer durable mode (`sink.py:59-190`). This is the sink every zarr consumer already uses.
- `xtrax.telemetry.{ledger,compact,store}`: **run-level** telemetry. `flock` JSONL segments
  (`ledger.py:19-24,48-63`) are compacted into `sealed/` and moved to `archive/` (`compact.py:1-20`). The
  dedup rule is last row per `(run_id, kind)` (`compact.py:78-90`). The blob store is content-addressed
  over uncompressed bytes, uses a pluggable codec (gzip, or zstd only when `zstandard` is installed)
  and has a population check (`store.py:1-35,60-75`). The CLI verb is `xtrax ledger`
  (`cli/ledger_verb.py`). `resolve_root` falls back to a default path, `.xtrax/ledger`
  (`ledger.py:162-167`). That conflicts with the derived-data-directory rule, so it must not be copied.
- alynxr `fold_stats/layout.py`: an existing lock-free per-writer index. Each writer adds an immutable
  part and then its own fragment (each by temp + `os.replace`). `MANIFEST.json` is derived and rebuilt
  idempotently by anyone. Identity is logical (sorted rows, canonical JSON), not file bytes, because
  Parquet bytes are not stable across pyarrow versions. R-S6 refuses one key with two logical hashes.
  DuckDB is the read side (`table.py:349-360`).
- Import weight: `import xtrax.telemetry.store` loads none of jax, zarr or equinox (V, probe this session).
  The cluster node that runs alynxr's packer has **no xtrax installed** (V, `product_node.py:1-8`).

## 3. Design

Principle: **the artifact's own record is the source of truth; the index is derived** (owner decision
D1, default shown, §7). This is the rule alynxr's fold-stats store already follows, and it is the only
reading that leaves no second completion state that can disagree with the first.

### 3.1 Unit index (`xtrax.run.unit_index`)

The name avoids `xtrax.ledger`, which collides with `xtrax.telemetry.ledger` and the `xtrax ledger`
verb. The index is an **index** over per-unit commit records. It never certifies completion on its own.

- **Truth.** A unit is complete when its artifact carries a verifying commit record. For a zarr unit,
  that is `zarr_commit`'s `xtrax.commit` attr. For a non-zarr unit (a loose dir, a pack), it is a
  `unit.json` record written last, by temp + `os.replace`, in the unit directory with the same fields
  as `CommitRecord`. The record lives with the artifact, so it moves and packs with it.
- **Index rows.** One JSON row per record:
  `{schema_version, unit_key, input_digest, content_digest, record_digest, writer_id, run_id, ts, reused_from}`.
  `content_digest` is defined in 3.2 and survives packing. `reused_from` is null, or names the store and
  `record_digest` that a Reuse decision read. This satisfies the "record what was reused from where" rule.
- **Write order.** The record is committed first and the index row appended second. If a crash lands
  between the two, the result is a record with no row. `rebuild` (below) adds the row back, so nothing
  is lost and nothing is double-counted.
- **Segments.** `<root>/segments/<writer_id>.jsonl`, with `writer_id = <host>-<pid>-<SLURM_JOB_ID>-<SLURM_RESTART_COUNT>-<uuid8>`.
  Requeue (alynxr `search.py:75-78`) and pid reuse therefore never share a segment. Each segment has a
  single writer and is append-only, with `fsync` after each row (one row per finished unit). A torn
  last line is skipped on read and reported, never repaired in place. The segment file count is bounded
  by **writers, not units, and not batches**. That is A2, now a design property, but still probed.
- **Lookup.** `lookup(unit_key, input_digest)` returns the **existing** `zarr_commit` types `Reuse |
  Missing | Stale | Corrupt` (imported, not redefined), plus `SchemaMismatch` for a record or row whose
  `schema_version` is not current. Lookup always goes to the artifact's record. The index only narrows
  which artifacts to open. The return value is a **classification only**. What to do with it (aminx
  quarantines and recomputes; aminx-ensemble raises) stays with the consumer.
- **Rebuild.** `rebuild(root, stores)` walks the records (`committed_keys`, plus `unit.json` for non-zarr
  units) and writes a derived `sealed/index.jsonl`. Anyone can run it, and it is idempotent, following
  alynxr's derived-`MANIFEST.json` rule.
- **Compact.** Folds segments into `sealed/` and **moves** them to `archive/`, never deleting. The dedup
  key is `unit_key`, not timestamp order, so clock skew between hosts cannot reorder rows. Two rows for
  one `unit_key` that agree on `(input_digest, content_digest)` collapse to one. If they disagree, the
  compaction is refused, and the refusal lists both rows (the R-S6 rule). Timestamps are informational.
- **Root.** The location is resolved as: explicit argument > `XTRAX_UNIT_INDEX_ROOT` (`none` or empty
  disables) > `[tool.xtrax]` in the nearest `pyproject.toml` > `${XDG_CONFIG_HOME:-~/.config}/xtrax/config.toml`
  > off. There is no default path, and the resolver exposes which layer decided.
- **Locks.** Mutual exclusion while a unit runs stays with the consumer (aminx's lease lock and
  `DistributedLockBackend`). The index has no locks.
- **Read side.** DuckDB `read_json` over `segments/` + `sealed/`, as an optional extra
  (`xtrax[index]`). DuckDB is not a core dependency, and it is never a writer.

### 3.2 Content identity that survives packing

`content_digest` of a non-zarr unit = sha256 of canonical JSON `[[relpath, sha256(uncompressed bytes)], ...]`
sorted by relpath. This is the same choice `telemetry/store.py` makes (addressing over uncompressed bytes)
and the same check `product_node.read_member` applies. A unit therefore has the same digest whether it
is loose or packed, and whichever codec or library version wrote the pack. The pack file's own sha256 only
verifies a transfer (alynxr's identity rule). For zarr units, `content_digest` stays
`zarr_content_digest`.

### 3.3 Pack (`xtrax.pack`)

- **Format.** The format is alynxr's existing one, generalized: USTAR tar, one compressed member per file,
  `mtime=0`, and an offset index `{relpath: {offset, length, sha256_uncompressed, codec}}`. The order is
  alynxr's five steps (`product_node.py:290-370`): write `.partial`; verify every member against the record;
  `os.replace`; write the index and then the record (the record now names `pack`, `pack_sha256`,
  `index_sha256`); delete the loose files last. At every step, a record that names a pack names a verified
  pack, and a record without one means every loose file exists.
- **Node-embeddable core.** `xtrax/pack/_node.py` is stdlib-only, written in Python 3.9 syntax, and
  imports nothing from xtrax. Two tests enforce this: an `ast` import check and a standalone `-I` run
  under 3.9. Its source text is exported (`xtrax.pack.node_source()`) so that a renderer can embed it in
  a cluster script, as alynxr does today. The library side (`xtrax.pack`) adds typed wrappers, the
  reader, and refusals.
- **Codec.** The codec registry is `telemetry/store.py`'s: gzip by default, and zstd only when `zstandard`
  is importable. There is **no Python floor change**. Reads are suffix- or codec-field-agnostic. The node
  core supports gzip only, because the stdlib on 3.9 has no zstd.
- **Opt-in.** Consumers opt in per unit class. A unit whose record names a pack is read through
  `xtrax.pack.read_member`. alynxr's fold ingest must learn to do this: today a packed unit fails as
  "no completion stamp" (V, `fold_stats/readers.py:19-21`).
- **Scope.** Pack applies to **loose-file units** only (alynxr fold units and search products, prolix
  runs if wanted). For zarr consumers, file count is controlled by chunking (aminx-ensemble's
  `chunk_size`) or by the zarr v3 sharding codec. Neither xtrax sink uses sharding today (V, grep). That
  is a separate item (§6), not a pack.

### 3.4 File guard (`xtrax.fileguard`)

This lifts alynxr's `FileCapGuard` (`storage.py`) **with its semantics intact**:
- Plan-time admission **and** in-run checks: re-read the real quota every `reread_every` files and
  interpolate between reads. The guard refuses at the check before a write, and the process is never
  killed mid-write. v1's "never mid-run" contradicted the code it generalized, and is withdrawn.
- On Lustre, the per-user figure comes from `lfs quota`. When `lfs` cannot give it, the read is
  refused, never replaced by `df -i` (which is filesystem-wide on Lustre). An unknown filesystem type is
  refused.
- The guard also records group quota and setgid inheritance of the target directory (global rule:
  `so3_shared/marielle/` is setgid to a full-quota group).
- Events go through an injectable emitter, defaulting to `logging`. `cisternal.emit_event` is used when
  the `plugin` extra is installed. `storage.py` hard-imports cisternal, and xtrax must not, because
  cisternal is only an optional extra here (`pyproject.toml:124`).
- A stdlib-only, node-embeddable check (`xtrax/fileguard/_node.py`) follows the same 3.9 rule as the
  pack core. aminx-ensemble's `_resolve.check_location` then becomes a consumer of it rather than a
  third copy.

### 3.5 Close the `Duplicate` gap in `zarr_commit` (new, higher value than v1's ledger)

On `ENOTEMPTY`/`EEXIST`, `commit_key` should compare the staged `content_digest` with the existing
record's, and return a new `Conflict(reason="same input, different content")` instead of `Duplicate`. It
should also offer `quarantine(store, key)`, which moves a key to `<store>.quarantine/` and never
deletes it. These are lifted from alynxr's `array_commit.py`, which can then drop its local copy. This is
a behaviour change, so it ships behind an argument
(`commit_key(..., verify_duplicate=False)` default this minor, `True` next minor).

## 4. Assumptions to probe (before build)

Each probe whose number will be cited (A2, A3, A5, A7) runs under a committed bathos sidecar.

| # | Assumption | Tag | Probe |
|---|---|---|---|
| A1 | `os.replace` of a sibling temp file is atomic on Engaging scratch (Lustre) and on NFS home | U | contention probe on both filesystems; a torn-read count must be 0 |
| A2 | segment file count = writers per campaign; per consumer the count is < 1% of the units | U | model from §2.1 and one real campaign |
| A3 | DuckDB `read_json` over segments + sealed at 1e6 rows returns in < 5 s | U | spike, with sidecar |
| A4 | DuckDB 2.0 reads the same files, and wheels exist for 3.13 and 3.14 | U | throwaway venv; does not block, since DuckDB is an optional read extra |
| A5 | zstd beats gzip on real fold unit dirs by enough to matter. **No floor change either way.** | U | measure on fold unit dirs, with sidecar |
| A6 | aminx done marker v3 maps onto the §3.1 row (`input_hash` → `input_digest`, `content_digest_sha256` → `content_digest`) | **V** (`campaign.py:1015-1045`) | — |
| A6b | demistify zarr internal file count per residue shard | U | measure one finished store |
| A7 | R = models × seeds per ColabFold unit | U | `storage_report` on a finished Engaging batch, with sidecar |
| A8 | Python 3.9 on Engaging and titanix nodes runs the `_node.py` cores | U | `-I` run on both |
| A9 | `lfs quota -u` gives a per-user file count on Engaging scratch | V for alynxr's parser (`storage.py:87-127`), U for current cluster output | one live read |

## 5. Rollout

1. Additive minor release. `zarr_commit`'s duplicate check ships off by default (§3.5). No floor changes.
   Each consumer gets a stated floor.
2. **alynxr fold units adopt first** (owner decision D2, default shown, §7). They are the only consumer
   whose loose-file units are the measured inode problem, and they already have the pack format, the guard
   and an xtrax dependency. Order within alynxr: fileguard (replace `storage.py`), pack + ingest-from-pack,
   then the unit index.
3. Then: aminx-ensemble adopts **fileguard only**. Its resume stays on `zarr_commit` as its own spec
   designs. aminx maps done marker v3 into index rows (read only). demistify is read-only indexing over its
   durable store. prolix adopts unit records + pack if it wants them.
4. Each stage is tests-first, with a negative control per refusal: corrupt member, stale input digest,
   schema mismatch, torn segment tail, two rows with the same key and different content, over-budget
   admission, Lustre without `lfs`, and a 3.9 standalone run of each `_node.py`.

## 6. Out of scope

Changing alynxr's logical-hash or R-S6 semantics. Making DuckDB a writer or a core dependency. Deleting
archived segments or quarantined units. Quack as a backend (revisit after A4). Zarr v3 sharding codec
for `ZarrStagingSink` (worth its own spec: it is the inode lever for zarr consumers). Merging run-level
telemetry (`xtrax.telemetry`) or bathos's catalog into the unit index. The three keep separate concerns
and join on `run_id`.

## 7. Open owner decisions

| # | Decision | Default in this draft | Alternative |
|---|---|---|---|
| D1 | Is the index derived from per-artifact records, or the source of truth? | **derived**: one completion truth, crash-safe by rebuild | source of truth: a cheaper lookup, but a second completion state, plus a write-order and orphan rule to specify |
| D2 | First adopter | **alynxr fold units**: the measured problem, with existing pack/guard code | aminx-ensemble: no legacy code, but its spec needs neither index nor pack |

## 8. Adversarial review of v1 (2026-10-30)

| # | Objection to v1 | Evidence | Resolution in v2 | Blocked v1? |
|---|---|---|---|---|
| 1 | Two completion truths: the ledger row vs the record in the artifact (`xtrax.commit` attr, aminx done marker, alynxr fragment). A crash between the artifact commit and the row write gives Missing over a finished artifact | `zarr_commit.py:364-386`; `campaign.py:1015-1045`; `layout.py:17-22` | index is derived (3.1); D1 | yes |
| 2 | "temp + `os.replace` per sealed batch" bounds files by batches, not writers, so A2 is false as stated. Sealing per unit (the preemption rule) means one file per unit, which is the problem itself | spec v1 §3.1 vs A2 | single-writer append + fsync per row + torn-tail skip (3.1) | yes |
| 3 | `writer_id` was undefined; a requeued job would reuse its predecessor's segment | alynxr `search.py:75-78,602` uses `SLURM_JOB_ID` + `SLURM_RESTART_COUNT` | `writer_id` includes job, restart, pid and uuid (3.1) | yes |
| 4 | `xtrax.ledger` collides with `xtrax.telemetry.ledger`, the `xtrax ledger` verb, and identical dir names | `cli/ledger_verb.py:1-15`; `ledger.py:56-58` | renamed to `xtrax.run.unit_index`; outcome types imported from `zarr_commit` | yes |
| 5 | The compaction dedup rule was unstated. Telemetry's "last row per key" is wrong across hosts with skewed clocks | `compact.py:78-90` | dedup by `unit_key`; conflicting content refused, R-S6 style (3.1) | yes |
| 6 | Packing would turn every Reuse into Corrupt if `artifact_sha256` hashes loose files or pack bytes | `product_node.py:377-410`; `store.py:1-10`; `layout.py:24-29` | `content_digest` over uncompressed members (3.2) | yes |
| 7 | `pack_members` is stdlib-only, Python 3.9, embedded in node scripts; an xtrax library module cannot run there | `product_node.py:1-8` | stdlib 3.9 `_node.py` core + `node_source()` (3.3) | yes |
| 8 | "Refuses before the job, never mid-run" contradicts `FileCapGuard`, which re-reads and refuses mid-run. v1 also dropped the Lustre `lfs` rule and the cisternal coupling | `storage.py:1-10,270-330`; `worker.py` exit 5; `pyproject.toml:124` | semantics kept; Lustre rule kept; injectable emitter (3.4) | yes |
| 9 | The closest prior art was missing: alynxr's fold-stats store already is a lock-free per-writer index with a derived manifest | `layout.py:1-30`; `table.py:349-360` | cited in 2.2; the design follows it | no |
| 10 | alynxr's xtrax dependency was listed as "none"; it commits through `zarr_commit` | `array_commit.py` header | table corrected | no |
| 11 | `DurableShardSink` was attributed to xtrax; it is demistify's | `demistify/shard_sink.py:1-10` | table corrected | no |
| 12 | `commit_key` returns `Duplicate` on input digest alone; alynxr patches this locally | `zarr_commit.py:399-400`; `array_commit.py` header | upstream fix (3.5) | no |
| 13 | aminx-ensemble as first adopter: its spec already resumes on `zarr_commit`, with Stale/Corrupt raising and a `chunk_size` file budget, so it needs neither ledger nor pack | epic §3.3, §3.5 | alynxr first; aminx-ensemble fileguard only (5); D2 | yes (rollout) |
| 14 | "Same four outcomes" hides consumer policy. aminx has legacy-v2, schema-stale, quarantine-on-mismatch, and raise-on-content-mismatch | `campaign.py:969-1012,1789-1880` | classification only + `SchemaMismatch` (3.1) | no |
| 15 | A5's "3.14 floor" contradicts "additive", and `store.py` already handles zstd as optional | `store.py:24-35,60-75`; `pyproject.toml:6` | no floor change (3.3, A5) | no |
| 16 | The ledger root was unspecified; copying telemetry's default path breaks the derived-data rule | `ledger.py:162-167` | layered resolver, no default (3.1) | no |
| 17 | A3, A5 and A7 produce cited numbers with no pre-registration | global bathos rule | sidecars required (4) | no |
| 18 | The global "record what was reused from where" rule is not met | global preemption rule | `reused_from` field (3.1) | no |
| 19 | A6 was tagged U but is readable | `campaign.py:1015-1045` | retagged V | no |
