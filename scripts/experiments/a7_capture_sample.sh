#!/bin/bash
# A7 amendment 2 input capture (task 261030_xtrax-ledger-pack; sidecar a7_fold_unit_inodes.bth.toml).
#
# Runs as a SLURM job on Engaging (never the login node). Writes a listing in the format
# a7_fold_unit_inodes.py reads: `find -printf "%y\t%P\t%s\n" | sort`, relative to ROOT, for a
# deterministic subsample of fold units.
#
# Selection, fixed before any unit's contents are listed:
#   1. list unit DIRECTORY NAMES only (fold_campaigns/<campaign>/<arm>/<unit>, -maxdepth 4,
#      -mindepth 4, -type d): names reveal nothing about a unit's file count;
#   2. per arm, keep the K units whose sha256("<campaign>/<arm>/<unit>") is lowest;
#   3. list each kept unit in full, plus the campaign and arm directory entries above it.
# The script does not look at completion: a kept unit without a *.done.txt is dropped by the
# probe's completed-unit rule, so n_units can be below K x arms.
#
# Usage: a7_capture_sample.sh ROOT K OUT CAMPAIGN [CAMPAIGN ...]
set -euo pipefail
ROOT=$1; K=$2; OUT=$3; shift 3
cd "$ROOT"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
for c in "$@"; do test -d "fold_campaigns/$c" || { echo "missing campaign $c" >&2; exit 2; }; done
: > "$tmp/units"
for c in "$@"; do
  find "fold_campaigns/$c" -mindepth 1 -maxdepth 1 -type d | sort > "$tmp/arms.$c"
  while read -r arm; do
    find "$arm" -mindepth 1 -maxdepth 1 -type d | sort | while read -r unit; do
      key=${unit#fold_campaigns/}
      printf '%s\t%s\n' "$(printf '%s' "$key" | sha256sum | cut -c1-64)" "$unit"
    done | sort | awk -v k="$K" 'NR <= k' | cut -f2 >> "$tmp/units"  # awk reads all input: no SIGPIPE under pipefail
  done < "$tmp/arms.$c"
done
sort -u "$tmp/units" > "$OUT.units"
{
  printf 'd\t\t0\n'
  printf 'd\tfold_campaigns\t0\n'
  for c in "$@"; do printf 'd\tfold_campaigns/%s\t0\n' "$c"; done
  cut -d/ -f1-3 "$OUT.units" | sort -u | sed 's/^/d\t/; s/$/\t0/'
  while read -r unit; do find "$unit" -printf '%y\t%p\t%s\n'; done < "$OUT.units"
} | sort > "$OUT"
echo "units_selected=$(wc -l < "$OUT.units") listing_lines=$(wc -l < "$OUT") sha256=$(sha256sum "$OUT" | cut -c1-64)"
