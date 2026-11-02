#!/bin/bash
# A1 wrapper (task 261030_xtrax-ledger-pack; spec 261030_ledger-pack-fileguard.md section 4, row A1).
# Runs INSIDE a 3-node SLURM allocation (nodes=3, ntasks=3): task 0 writes, tasks 1-2 read, one task
# per node. Launches the stdlib harness with plain python3.11 (not uv): Engaging's default python3 is
# 3.6.8 (spec A8), so the interpreter is explicit. Produces raw JSONL logs only; the outcome comes
# from a1_analyze.py under `bth run`, later, locally.
set -euo pipefail

PY=${A1_PYTHON:-/usr/bin/python3.11}
JOB=${SLURM_JOB_ID:?run inside a SLURM allocation}
SCRATCH_DIR=${A1_SCRATCH_DIR:-/orcd/scratch/orcd/011/maarxaru/a1_probe}
POOL_DIR=${A1_POOL_DIR:-/orcd/pool/007/maarxaru/a1_probe}
HOME_DIR=${A1_HOME_DIR:-$HOME/a1_probe}
OUT=${A1_OUT_DIR:-outputs/probes/a1/raw_$JOB}
CONTROL_S=${A1_CONTROL_S:-30}
REAL_S=${A1_REAL_S:-180}
START=$(( $(date +%s) + ${A1_LEAD_S:-60} ))

"$PY" -c 'import sys; assert sys.version_info >= (3, 9), sys.version'
[ "${SLURM_JOB_NUM_NODES:-0}" -ge 3 ] || { echo "need 3 nodes, have ${SLURM_JOB_NUM_NODES:-0}" >&2; exit 2; }
mkdir -p "$SCRATCH_DIR" "$POOL_DIR" "$HOME_DIR" "$OUT"

echo "a1 job=$JOB nodes=$SLURM_JOB_NODELIST start_epoch=$START out=$OUT"
srun --nodes=3 --ntasks=3 --ntasks-per-node=1 "$PY" scripts/experiments/a1_replace_atomicity.py \
  --fs "scratch=$SCRATCH_DIR" --fs "pool=$POOL_DIR" --fs "home=$HOME_DIR" \
  --control-s "$CONTROL_S" --real-s "$REAL_S" --start-epoch "$START" --out-dir "$OUT" --tag "$JOB"

# Remove only this job's probe subdirectories.
for d in "$SCRATCH_DIR" "$POOL_DIR" "$HOME_DIR"; do rm -rf "$d/a1_$JOB"; done
