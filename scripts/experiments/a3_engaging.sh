#!/bin/bash
# A3 on Engaging (sidecar: a3_analyze.bth.toml). Run from the repo root inside a SLURM job.
#   setup                 create the duckdb==2.0.0 venv on so3/002 if missing (wheel is cp312-abi3)
#   build <rows> <tag>    build the steady + busy fixtures on scratch
#   read  <tag>           time them (run in a separate job, on a different node from build)
#   clean <tag>           remove a fixture left by a failed read
set -euo pipefail
PHASE=$1
VENV=/orcd/compute/so3/002/maarxaru/venvs/a3-duckdb-2.0.0
SCRATCH=/orcd/scratch/orcd/011/maarxaru
PY=$VENV/bin/python
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} phase=$PHASE"
case "$PHASE" in
  setup)
    if [ ! -x "$PY" ]; then
      /usr/bin/python3.12 -m venv "$VENV"
      "$VENV/bin/pip" install -q duckdb==2.0.0
    fi
    "$PY" -I -c 'import duckdb, sys; print("duckdb", duckdb.__version__, "python", sys.version.split()[0])'
    ;;
  build)
    "$PY" scripts/experiments/a3_duckdb_read.py --phase build \
      --root "$SCRATCH/a3_probe_$3" --layouts steady,busy --rows "$2"
    ;;
  read)
    "$PY" scripts/experiments/a3_duckdb_read.py --phase read --env engaging_scratch \
      --root "$SCRATCH/a3_probe_$2" --layouts steady,busy --repeats 3 --threads 4 \
      --out "outputs/probes/a3/raw_engaging_$2.json"
    ;;
  clean)
    rm -rf "${SCRATCH:?}/a3_probe_$2"
    ;;
  *)
    echo "unknown phase $PHASE" >&2
    exit 2
    ;;
esac
