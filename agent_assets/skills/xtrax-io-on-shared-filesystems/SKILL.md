---
name: xtrax-io-on-shared-filesystems
description: Use when a job that reads large files (MD trajectories, XTC/DCD/NetCDF, zarr/HDF5 shards, datasets) from NFS, Lustre, GPFS or another shared/network filesystem is slow, hits walltime, shows CPU time far below elapsed time, sits in process state D, or when choosing between a seek/strided/random-access reader and a sequential read or local staging copy.
xtrax_version: 0.4.0a11
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
bytes: one small read costs a round trip (measured 130-200 ms on a contended
NFS pool) and seeking defeats read-ahead. A reader that touches 5x fewer bytes
through many more small, non-sequential requests can be slower there, however
well it benchmarks on a laptop SSD. Measured once (2026-10-01): seek on the
contended pool ran at ~0.2 MB/s vs several MB/s for a 64 MiB-block copy.

## Facts agents get wrong

| Claim | Reality |
|---|---|
| "`md.iterload(stride=k)` seeks past skipped frames" | It decodes **every** frame and drops k-1 of k: reads all bytes, sequentially. |
| "XTC has no random access" | `XTCTrajectoryFile.offsets` builds a frame-offset table, then `seek(i)` works -- but building it walks **every frame header** (one small read per frame). |
| "fewer bytes = faster" | True on local disk only. On NFS, count requests: header walk + per-frame checks + per-kept-frame seek can be ~2-3 requests per frame. |
| "parallel readers will speed it up" | Not when the pool's aggregate bandwidth is the limit -- N readers share it, and each one's latency rises. |
| "local benchmark shows Nx" | Says nothing about the target filesystem under production concurrency. |

## Diagnose (on a compute node, against the real path)

```bash
stat -f -c %T /path/to/file                       # nfs / lustre / gpfs => network; ext4/xfs/tmpfs => local
sstat -j $JOB --format=AveCPU,MaxDiskRead          # CPU << elapsed => waiting on I/O
ps -o stat,cmd -p $PID                             # D = blocked in I/O
cat /proc/$PID/io; sleep 60; cat /proc/$PID/io     # Δrchar/Δsyscr = bytes per request
nfsiostat 5 /mount/point                          # NFS only: kB/op, ops/s, RTT per mount (or /proc/self/mountstats)
dd if=/path/to/big_unread_file of=/dev/null bs=64M count=64   # pool's sequential MB/s right now
```

Small bytes-per-request (or kB/op) with high RTT and low MB/s => latency-bound:
cut request count (sequential, large blocks, stage locally) -- adding parallel
readers only helps if the pool still has bandwidth headroom (check `dd` first).
Large bytes-per-request with low MB/s => bandwidth-bound: cut total bytes or
concurrency, not request size. `py-spy` only once the process is CPU-bound.

## Patterns that work

1. **Stage by one large sequential copy** to node-local disk (`dd bs=64M`,
   `rsync -W`), then use any access pattern locally.
2. **Reduce once, read many**: write a slim copy (strided, solute-only) once,
   keyed by source identity (size + head/tail hash), to a configured cache
   root; every later pass reads the small local-or-cached file at stride 1.
3. **Choose the reader by filesystem**: seek/strided only when the file is on a
   type known to be local; default to sequential for anything else, including
   unknown types (wrong that way costs ~stride x; the other way cost >10x).
4. **Log per-phase timings** (stage, header scan, decode, compute) with MB/s, so
   the next slow run tells you which phase.

## Gate before shipping an I/O-path change

Same-conditions A/B **on the target filesystem**: same file class, cold cache,
production-like concurrency, both arms in one window. Pre-register the pass
criterion (see `xtrax-probing` / bathos); the local benchmark is a hypothesis.
