---
name: xtrax-io-on-shared-filesystems
description: Use when a job that reads large files (MD trajectories, XTC/DCD/NetCDF, zarr/HDF5 shards, datasets) from NFS, Lustre, GPFS or another shared/network filesystem is slow, hits walltime, shows CPU time far below elapsed time, sits in process state D, or when choosing between a seek/strided/random-access reader and a sequential read or local staging copy.
xtrax_version: 0.4.0a12
triggers:
  - job slow / hits walltime reading trajectories on NFS / shared storage
  - sstat AveCPU << elapsed, ps STAT D, iowait
  - seek reader vs sequential read vs copy-to-local
  - strided read (stride=k) of XTC / DCD / large binary files
  - stage to node-local /tmp before analysis
  - cache a slim / strided copy once, read many times
---

# xtrax-io-on-shared-filesystems

## Purpose

Pick the read pattern from a measurement of the filesystem it runs on. On a
shared network filesystem the cost of a read is **requests x latency**, not
bytes: a small read that misses the client cache costs a round trip, and
seeking defeats read-ahead. A reader that touches 5x fewer bytes through many
more small, non-sequential requests can be slower there, however well it
benchmarks on a laptop SSD. One anecdotal measurement (2026-10-01, contended
NFS pool, not a pre-registered run): seek+link at ~0.2 MB/s vs several MB/s for
a 64 MiB-block copy, small reads ~130-200 ms each.

## Facts agents get wrong

Checked against mdtraj 1.11.1 source (`formats/xtc/xtc.pyx`) and `/proc/self/io` probes.

| Claim | Reality |
|---|---|
| "`md.iterload(stride=k)` seeks past skipped frames" | `iterload` never builds the offset table, and without it `read(stride=k)` decodes **every** frame (xtc.pyx `efficient_striding = stride > 1 and self._offsets is not None`): all bytes, sequentially. With offsets already built, `read(stride=k)` seeks instead. DCD likewise reads every frame (skips only the decode). |
| "XTC has no random access" | `XTCTrajectoryFile.offsets` builds a frame-offset table, then `seek(i)` works -- but building it walks **every frame header**: one seek + small read per frame (`_calc_len_and_offsets`), ~4 KB of stdio buffer per header, so for small frames the walk reads most of the file. |
| "fewer bytes = faster" | True on local disk only. On NFS count requests: the header walk (1 per frame) plus ~3 per **kept** frame (seek, read, any per-frame check). |
| "parallel readers will speed it up" | Not when the pool's aggregate bandwidth is the limit -- N readers share it, and each one's latency rises. |
| "local benchmark shows Nx" | Says nothing about the target filesystem under production concurrency. Locally, seek vs sequential at stride 50 measured ~12x, not 50x: the header walk is not free. |

## Diagnose (on a compute node, against the real path)

Rule out CPU-bound first: XTC decode alone measured ~37 MB/s per core from page cache (mdtraj 1.11.1, 5000 atoms).

```bash
sstat -j $JOB.batch --format=AveCPU,MaxDiskRead   # .batch for sbatch scripts; AveCPU is CPU TIME -- compare to elapsed
ps -o stat,%cpu,cmd -p $PID                       # ~100% CPU => decode/compute-bound, stop here; D = blocked in I/O
stat -f -c %T /path/to/file                       # nfs / lustre / gpfs / fuse / v9fs => network; "ext2/ext3" (= ext4), xfs, tmpfs => local
nfsiostat 5 /mount/point                          # PRIMARY signal on NFS: per-op RTT and kB/op (or /proc/self/mountstats)
dd if=/path/to/big_unread_file of=/dev/null bs=64M count=16 iflag=direct   # pool's sequential MB/s now, page cache bypassed
```

High RTT with small kB/op and low MB/s => latency-bound: cut request count
(sequential, large blocks, stage locally). MB/s near the `dd` figure with
large kB/op => bandwidth-bound: cut total bytes or concurrency. Add parallel
readers only if `dd` shows headroom. `/proc/$PID/io` Δrchar/Δsyscr is the
**application's** call size (stdio buffers make even a good sequential reader
show ~4-9 KB/call; NFS read-ahead aggregates below it), and rchar counts
page-cache hits -- use it to compare readers, not to classify the filesystem.
`py-spy` once the process is CPU-bound.

## Patterns that work

1. **Stage by one large sequential copy** to node-local disk (`dd bs=64M`,
   `rsync -W`), then use any access pattern locally.
2. **Reduce once, read many**: write a slim copy (strided, solute-only) once,
   keyed by source identity (size + head/tail hash), to a configured cache
   root; every later pass reads the small local-or-cached file at stride 1.
3. **Choose the reader by filesystem**: seek/strided only when the file is on a
   type known to be local; default to sequential for anything else, including
   unknown types (wrong that way costs at most the stride factor, ~12x at stride 50 locally; the
   other way can cost more than that on a contended pool).
4. **Log per-phase timings** (stage, header scan, decode, compute) with MB/s, so
   the next slow run tells you which phase.

## Gate before shipping an I/O-path change

Same-conditions A/B **on the target filesystem**: same file class, production-like
concurrency, both arms in one window. Users cannot drop the page cache: give each arm
files neither has read (or `dd iflag=nocache count=0` to advise them out), and
alternate arm order so a warm second arm cannot win by caching. Pre-register the pass
criterion (see `xtrax-probing` / bathos); the local benchmark is a hypothesis.
