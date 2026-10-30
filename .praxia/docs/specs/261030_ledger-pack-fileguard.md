# Spec: shared run ledger, output packing and file-count guard (DRAFT v1)

- **task_id**: `261030_xtrax-ledger-pack`
- **status**: DRAFT v1 — recon-grounded, not adversarially reviewed. No code written.
- **consumers**: alynxr, aminx, aminx-ensemble, demistify, prolix
- **origin**: owner request 2026-10-30 (upstream the alynxr provenance/consolidation design into xtrax).
  Decisions already made by the owner: (1) recon the five consumers first; (2) the ledger defaults to
  **lock-free per-writer segments**, not `flock`.

Tags: **V** = read in code at the cited line by this session or by a recon agent whose report was
accepted into the praxia record; **U** = unverified (not read, or only claimed by an agent whose
dispatch carried no agent_result). Every U must be probed before it changes the design.

## 1. Problem

Each consumer builds its own "unit record, completion marker, resume by input hash". None packs run
outputs, and none guards file count except as a plan. On Engaging scratch the file cap, not bytes, is
the binding limit (alynxr hit 1,000,000 files at ~46% of bytes). The per-unit directories are the
inode cost, and nothing consolidates them.

## 2. Recon facts

| Consumer | Per-unit footprint | Ledger / resume | Packing | Inode guard | xtrax dep |
|---|---|---|---|---|---|
| alynxr | ~5 + 2R loose files, R = models x seeds (U for R); `fold.py:183`, `ingest.py:53-56`, `readers.py:44-45` | fragment per part, logical hash; ingest reads loose files only (`ingest.py:539`) | `pack_members` is search products only, `PACK_THRESHOLD = 64` (V, `product_node.py:50`); fold-unit "Archive Stage B" not implemented (V, `readers.py:19-21`) | `storage.py` admission guard, scratch tools only (V) | none |
| aminx | one zarr dir per campaign row + `.lock`, `.done.json`, transient `.partial.*` (`host/campaign.py:315-329`) | lease lock `O_EXCL` + CAS steal, 1800 s (V, `campaign.py:49,335`); done marker with content digest (U) | none (U) | none (U) | >=0.4.0a11,<0.5 |
| demistify | residue shards per chunk in one zarr; phase5 grid ~144 files (`phase5_nvi_sweep.py:267-332`); zarr internal count U | summary.json then set_latest; config-sha256 + `os.replace` for posterior blocks (`posterior_block_coverage.py:135-175`) | none | disk quota only (`phase0_preflight.py:35-39`) | >=0.4.1b3, locked 0.4.1b5; uses `DurableShardSink` |
| prolix | 1 json + optional 1 xtc per run; no per-frame files (`ensemble_plan.py:356-358`) | none; resume is existence check (`fit_bonded_hp4.py:257`) | none | none | >=0.4.1b2 |
| aminx-ensemble | none: `hello()` stub; layout is spec-only (`261030_aminx-ensemble-epic.md:304-320`) | planned per-chunk input digest, Reuse/Missing/Stale (spec :281-295) | planned zarr | planned, budget < 100k files/store (spec :355-358) | not a dependency |

Cross-cutting (V unless noted): no consumer uses `flock`; xtrax's own `telemetry/ledger.py` does
(`ledger.py:20,123`). xtrax already provides `telemetry/{ledger,compact,store}.py` (append-only JSONL
segments, sealed compaction that moves rather than deletes, content-addressed gzip blob store with a
population check) and `run/zarr_commit.py` (staged fsynced commit; Reuse/Stale/Corrupt/Missing).
xtrax pin floors differ: prolix `check_xtrax_pin.py` FLOOR 0.4.0a6 vs pyproject 0.4.1b2 (U, agent claim).

## 3. Design

### 3.1 Lock-free per-writer ledger (`xtrax.ledger`)
- One writer owns one segment file: `<root>/segments/<writer_id>/<seq>.jsonl`, written
  temp + `os.replace` per sealed batch, never shared-appended. No `flock` dependency.
- A **unit record** is a JSON row `{unit_key, input_sha256, status, artifact_sha256, writer_id, ts}`;
  completion is the existence of a row with `status = complete`. Resume = lookup returns
  Reuse (input hash matches, artifact verifies), Stale, Corrupt or Missing, the same four outcomes as
  `run/zarr_commit.py`.
- `compact` merges sealed segments into `sealed/`, moves originals to `archive/`, never deletes.
- Compatible with aminx's lease lock where a unit needs mutual exclusion while running; the lock is a
  per-unit claim file, not the ledger.
- Read side: a DuckDB `read_json` / Parquet view over the segments (3.4). The ledger is never a shared
  writable database file.

### 3.2 Pack (`xtrax.pack`)
Generalise alynxr `pack_members`: loose unit directory -> one archive with an offset index. Order:
write `.partial`, verify every member sha256 from the pack, rename, then remove loose files.
Random-access member read by index. Codec pluggable (gzip default; zstd when Python >= 3.14 or
`zstandard` is installed, because xtrax floors at 3.13). Consumers opt in per arm / per unit class.
Alynxr's ingest must learn to read a packed unit (today a packed unit fails as "no completion stamp",
V, `readers.py:19-21`, `ingest.py:86-112`).

### 3.3 File guard (`xtrax.fileguard`)
Admission check (`df -i` / quota) before a job writes N new files, and a population check. Generalise
alynxr `storage.py` (`parse_df_inodes`, `admit`). Refuses before the job, never mid-run.

### 3.4 DuckDB
Read side only. alynxr locks 1.5.6 (`>=1.1`, V). Search found DuckDB 2.0 docs describing in-process
MVCC writers and multi-process writes through a client-server "Quack" protocol, while older docs say
multi-process writes are unsupported. These are search results, not read docs (U). It gives no
multi-process write to one file on a shared filesystem, so it does not change 3.1.

## 4. Assumptions to probe (before build)

| # | Assumption | Tag | Probe |
|---|---|---|---|
| A1 | `os.replace` per segment is atomic on Engaging scratch and NFS | U | contention probe on both filesystems |
| A2 | one JSONL segment per writer keeps file count bounded by writers, not units | U | model count per consumer from section 2 |
| A3 | DuckDB reads JSONL and Parquet segments at ledger scale | U | spike at ~1e6 rows |
| A4 | DuckDB 2.0 (pre-release?) reads the same files, wheels exist for 3.13/3.14, format compatible | U | throwaway venv; pin range explicit |
| A5 | zstd member codec beats gzip enough to justify a 3.14 floor | U | measure on real fold unit dirs |
| A6 | demistify zarr internal file count and aminx done-marker schema | U | read / measure |
| A7 | R = models x seeds per ColabFold unit | U | count a finished Engaging batch with `storage_report` |

## 5. Rollout
1. Additive API, new minor, floor stated per consumer; no existing xtrax behaviour changes.
2. aminx-ensemble adopts first (no legacy code). Then alynxr (pack + ingest-from-pack), aminx,
   demistify, prolix.
3. Each stage needs tests-first with a negative control per refusal (corrupt member, stale input hash,
   torn segment, over-budget admission).

## 6. Out of scope
Changing alynxr's logical-hash or R-S6 semantics; making DuckDB a shared writer; deleting archived
segments; Quack as a backend (revisit after A4).
