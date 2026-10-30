"""A7: inodes per alynxr fold unit, from a captured file listing (spec 261030 §4).

Question: is packing alynxr fold units worth building? A packed unit costs about
``PACKED_INODES`` inodes (unit dir, pack, index, record). Packing pays when a unit's loose
footprint is well above that.

Input: a TSV listing captured on Engaging with ``find . -printf "%y\\t%P\\t%s\\n" | sort`` from a
``folds`` directory laid out as ``fold_campaigns/<campaign>/<arm>/<unit>/...``. A unit is a
directory at depth 4; every entry below it (files and subdirectories) is one inode of that unit,
plus the unit directory itself. Entries outside any unit are counted as ``unassigned``.

Model count per unit (R = models x seeds in the spec) is the number of ``*_rank_*.cif`` files.

The script first runs a synthetic positive and negative control; ``control_ok`` gates the outcome.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("a7_fold_unit_inodes")

PACKED_INODES = 4
UNIT_DEPTH = 4  # fold_campaigns / <campaign> / <arm> / <unit>
_MODEL = re.compile(r"_rank_\d{3}_.*\.cif$")


def per_unit(lines: list[str]) -> tuple[dict[str, dict[str, int]], int]:
    """``({unit: {"inodes", "files", "models"}}, unassigned)`` from listing lines."""
    units: dict[str, dict[str, int]] = defaultdict(lambda: {"inodes": 0, "files": 0, "models": 0})
    unassigned = 0
    for line in lines:
        if not line.strip():
            continue
        kind, rel, _size = line.rstrip("\n").split("\t")
        if rel == "":
            continue  # the listing root itself
        parts = rel.split("/")
        if parts[0] != "fold_campaigns":
            unassigned += 1
            continue
        if len(parts) < UNIT_DEPTH:
            continue  # campaign / arm directories: shared, not per unit
        unit = "/".join(parts[:UNIT_DEPTH])
        u = units[unit]
        u["inodes"] += 1  # the unit dir itself (len == UNIT_DEPTH) or an entry below it
        if kind == "f":
            u["files"] += 1
            if _MODEL.search(parts[-1]):
                u["models"] += 1
    return dict(units), unassigned


def summarize(lines: list[str]) -> dict[str, object]:
    units, unassigned = per_unit(lines)
    inodes = sorted(u["inodes"] for u in units.values())
    models = sorted(u["models"] for u in units.values())
    arms = sorted({"/".join(k.split("/")[:3]) for k in units})
    med = statistics.median(inodes) if inodes else 0.0
    return {
        "n_units": len(units),
        "n_arms": len(arms),
        "arms": arms,
        "median_inodes_per_unit": float(med),
        "min_inodes_per_unit": int(inodes[0]) if inodes else 0,
        "max_inodes_per_unit": int(inodes[-1]) if inodes else 0,
        "median_models_per_unit": float(statistics.median(models)) if models else 0.0,
        "total_unit_inodes": int(sum(inodes)),
        "unassigned_entries": unassigned,
        "packed_inodes_per_unit": PACKED_INODES,
        "inode_saving_fraction_at_median": (1.0 - PACKED_INODES / med) if med else 0.0,
    }


def controls() -> dict[str, bool]:
    """Synthetic listings with known answers: one must match exactly, one must differ."""
    base = "fold_campaigns/c/arm-1"
    synth = [
        "d\t\t0",
        "d\tfold_campaigns\t0",
        "d\tfold_campaigns/c\t0",
        f"d\t{base}\t0",
        f"d\t{base}/u1\t0",
        f"f\t{base}/u1/a.json\t1",
        f"d\t{base}/u1/fold\t0",
        f"f\t{base}/u1/fold/q_rank_001_m_seed_000_sample_0.cif\t1",
        f"f\t{base}/u1/fold/q_rank_002_m_seed_001_sample_0.cif\t1",
        f"d\t{base}/u2\t0",
        f"f\t{base}/u2/a.json\t1",
        "f\tstray.txt\t1",
    ]
    units, unassigned = per_unit(synth)
    positive = (
        units
        == {
            f"{base}/u1": {"inodes": 5, "files": 3, "models": 2},
            f"{base}/u2": {"inodes": 2, "files": 1, "models": 0},
        }
        and unassigned == 1
    )
    # Negative control: moving a unit's file to the campaign level must change the answer.
    moved = [ln.replace(f"{base}/u2/a.json", "fold_campaigns/c/a.json") for ln in synth]
    units_moved, _ = per_unit(moved)
    negative_detected = units_moved != units
    return {"positive_control_ok": positive, "negative_control_detected": negative_detected}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--listing", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

    ctrl = controls()
    raw = args.listing.read_bytes()
    result = {
        **summarize(raw.decode("utf-8").splitlines()),
        **ctrl,
        "control_ok": ctrl["positive_control_ok"] and ctrl["negative_control_detected"],
        "listing": str(args.listing),
        "listing_sha256": hashlib.sha256(raw).hexdigest(),
    }
    log.info("controls %s; units %s", ctrl, result["n_units"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results_path = os.environ.get("BTH_RESULTS_PATH")
    if results_path:
        Path(results_path).write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
