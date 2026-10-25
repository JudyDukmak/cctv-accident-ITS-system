"""Make the fixed, scenario-grouped synthetic train/val split.

Synthetic clips are re-runs of ~14 scripted crashes (one per map + accident type),
filmed from many cameras and weathers (see check_synthetic_runs.py). A per-clip
random split would put the same crash site in train and val, so clips are grouped by
scenario_id = f"{map}_{type}" and whole scenarios are held out per fold.

The folds are fixed by ACTIVE_FOLDS below (no randomness, no sklearn). For fold k the
validation set is every row with val_fold == k and the training set is the rest.
split_dev is the default quick split (fold 0 as validation).

The split is frozen by split_manifest.json (SHA-256 of synthetic_split.csv). A run
that would produce a different file stops unless --unfreeze is given. --verify only
checks the existing file against the manifest and writes nothing.

READ-ONLY on the dataset: reads only metadata-synthetic.csv. No video, annotation
or real-data file is opened.

Usage:
    py make_synthetic_split.py [--dataset-path PATH] [--unfreeze]
    py make_synthetic_split.py --verify
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# fold index = the fold in which these scenarios are held out as validation.
# Historical record only (not used): the first, hand-made assignment (cost 13.06).
FOLDS_V1_HANDMADE = {
    0: ["Town05_head-on", "Town04_rear-end", "Town05_sideswipe", "Town10HD_t-bone", "Town07_single"],
    1: ["Town03_head-on", "Town07_rear-end", "Town04_sideswipe", "Town03_t-bone", "Town10HD_single"],
    2: ["Town06_head-on", "Town05_rear-end", "Town06_sideswipe", "Town07_t-bone"],
}
# Best scenario-level assignment of the seed-0 random search (rebalance_folds.py, cost 7.395).
FOLDS_V2_REBALANCED = {
    0: ["Town04_rear-end", "Town04_sideswipe", "Town06_head-on", "Town10HD_single", "Town10HD_t-bone"],
    1: ["Town03_t-bone", "Town05_head-on", "Town05_rear-end", "Town05_sideswipe", "Town07_single"],
    2: ["Town03_head-on", "Town06_sideswipe", "Town07_rear-end", "Town07_t-bone"],
}
ACTIVE_FOLDS = FOLDS_V2_REBALANCED
SPLIT_VERSION = "v2_rebalanced"

DEV_VAL_FOLD = 0
MULTI_VEHICLE_TYPES = ["head-on", "rear-end", "sideswipe", "t-bone"]

# Expected numbers for ACTIVE_FOLDS, compared as warnings only.
EXPECTED_VAL_SIZE = {0: 770, 1: 755, 2: 686}
EXPECTED_TOTAL = 2211
EXPECTED_VAL_TYPES = {
    0: {"rear-end": 245, "head-on": 202, "sideswipe": 139, "t-bone": 131, "single": 53},
    1: {"rear-end": 277, "t-bone": 184, "head-on": 174, "sideswipe": 107, "single": 13},
    2: {"rear-end": 272, "head-on": 212, "sideswipe": 159, "t-bone": 43, "single": 0},
}

# Provenance written to split_manifest.json.
MANIFEST_INFO = {
    "method": "Week 3 evaluation Section 5.4 random search, seed 0, 200000 draws, best of run 1 (scenario-level)",
    "grouping_rule": (
        "Scenario-level grouping. Same-map scenarios are separate crash sites (76-758 m apart). "
        "Map-level split rejected: the best of all 540 map-level splits costs 12.71 "
        "(timing term 6.75, one fold with median accident time 11.5 s) vs 7.395 for the "
        "scenario-level split, so maps are not enough for 3 balanced folds. "
        "Map sharing is stated as a limitation."
    ),
    "global_median_definition": "median of the 14 scenario medians (6.675 s)",
    "cost": {"total": 7.395, "size": 1.02, "timing": 1.375, "type_penalty": 5.0},
    "previous_version": {"name": "v1_handmade", "cost": 13.06, "archived_in": "archive/split_v1_handmade/"},
}

DEFAULT_DATASET = Path(
    r"C:\Users\ASUS\Documents\Uni\Senior 2\cctv-accident-ITS-system\ACCIDENT\dataset"
)
REPO_ROOT = Path(__file__).resolve().parent
OUT_CSV = REPO_ROOT / "synthetic_split.csv"
OUT_MANIFEST = REPO_ROOT / "split_manifest.json"
OUT_SUMMARY = REPO_ROOT / "synthetic_check" / "split_summary.txt"
OUT_COLUMNS = ["rgb_path", "scenario_id", "val_fold", "split_dev", "map", "type", "weather",
               "camera_position", "split_version"]


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_manifest():
    if not OUT_MANIFEST.is_file():
        return None
    try:
        return json.loads(OUT_MANIFEST.read_text(encoding="ascii"))
    except (OSError, ValueError) as e:
        fail(f"cannot read {OUT_MANIFEST}: {e}")


def build_split(df):
    if df["rgb_path"].duplicated().any():
        dup = df.loc[df["rgb_path"].duplicated(keep=False), "rgb_path"].head(10).tolist()
        fail(f"rgb_path is not unique, e.g. {dup}")

    scenario_to_fold = {}
    for k, scenarios in ACTIVE_FOLDS.items():
        for s in scenarios:
            if s in scenario_to_fold:
                fail(f"scenario {s} is listed in fold {scenario_to_fold[s]} and fold {k}")
            scenario_to_fold[s] = k
    if len(scenario_to_fold) != 14:
        fail(f"ACTIVE_FOLDS lists {len(scenario_to_fold)} scenarios, expected 14")

    out = df.copy()
    out["scenario_id"] = out["map"].astype(str) + "_" + out["type"].astype(str)
    in_data = set(out["scenario_id"])
    in_folds = set(scenario_to_fold)
    if in_data != in_folds:
        fail(
            "scenario_ids in the data and in ACTIVE_FOLDS differ. "
            f"Only in data: {sorted(in_data - in_folds)}; only in ACTIVE_FOLDS: {sorted(in_folds - in_data)}"
        )

    out["val_fold"] = out["scenario_id"].map(scenario_to_fold)
    if out["val_fold"].isna().any():
        fail(f"{int(out['val_fold'].isna().sum())} rows got no fold")
    out["val_fold"] = out["val_fold"].astype(int)

    for k in ACTIVE_FOLDS:
        present = set(out.loc[out["val_fold"] == k, "type"])
        missing = [t for t in MULTI_VEHICLE_TYPES if t not in present]
        if missing:
            fail(f"fold {k} validation is missing multi-vehicle types {missing}")

    out["split_dev"] = (out["val_fold"] == DEV_VAL_FOLD).map({True: "val", False: "train"})
    out["camera_position"] = out["camera_position"].astype(int)
    out["split_version"] = SPLIT_VERSION
    return out[OUT_COLUMNS].sort_values("rgb_path", kind="mergesort").reset_index(drop=True)


def count_table(series, name):
    t = series.value_counts().rename_axis(name).rename("n").to_frame()
    t["pct"] = (100 * t["n"] / t["n"].sum()).map("{:.1f}".format)
    return t.to_string()


def summarize(split):
    lines = []
    p = lines.append
    n = len(split)
    p(f"1. Total clips: {n}")
    p("")
    p("2. Folds (fold k: val = val_fold == k, train = the rest)")
    for k in sorted(ACTIVE_FOLDS):
        val = split[split["val_fold"] == k]
        p(f"-- fold {k}: train {n - len(val)}, val {len(val)} ({100 * len(val) / n:.1f}%)"
          f"{'   <- split_dev' if k == DEV_VAL_FOLD else ''}")
        p("   val types:")
        p("   " + count_table(val["type"], "type").replace("\n", "\n   "))
        p("   val weathers:")
        p("   " + count_table(val["weather"], "weather").replace("\n", "\n   "))
        p("")
    p("3. Scenarios")
    sc = split.groupby("scenario_id").agg(n_clips=("rgb_path", "size"), val_fold=("val_fold", "first"))
    p(sc.sort_values(["val_fold", "n_clips"], ascending=[True, False]).to_string())
    p("")
    p("4. EXPECTED comparison (warnings only)")

    def check(label, got, exp):
        p(f"   {'OK      ' if got == exp else 'MISMATCH'} {label}: got {got}, expected {exp}")

    check("total clips", n, EXPECTED_TOTAL)
    for k in sorted(ACTIVE_FOLDS):
        check(f"fold {k} val size", int((split["val_fold"] == k).sum()), EXPECTED_VAL_SIZE[k])
    for k in sorted(ACTIVE_FOLDS):
        counts = split.loc[split["val_fold"] == k, "type"].value_counts()
        for t, exp in EXPECTED_VAL_TYPES[k].items():
            check(f"fold {k} val {t}", int(counts.get(t, 0)), exp)
    return "\n".join(lines)


def verify():
    """Compare synthetic_split.csv with the manifest hash; writes nothing."""
    manifest = load_manifest()
    if manifest is None:
        fail(f"{OUT_MANIFEST} not found")
    if not OUT_CSV.is_file():
        fail(f"{OUT_CSV} not found")
    got = sha256_bytes(OUT_CSV.read_bytes())
    exp = manifest.get("split_sha256")
    ok = got == exp
    print(f"split_version (manifest): {manifest.get('split_version')}, frozen: {manifest.get('frozen')}")
    print(f"manifest split_sha256 : {exp}")
    print(f"file     sha256       : {got}")
    print("OK" if ok else "MISMATCH")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--unfreeze", action="store_true",
                    help="allow replacing a frozen split whose hash differs from the manifest")
    ap.add_argument("--verify", action="store_true",
                    help="only check synthetic_split.csv against split_manifest.json; write nothing")
    args = ap.parse_args()

    if args.verify:
        sys.exit(0 if verify() else 1)

    dataset_root = args.dataset_path.resolve()
    if (dataset_root / "dataset" / "metadata-synthetic.csv").is_file():
        dataset_root = dataset_root / "dataset"
    meta_csv = dataset_root / "metadata-synthetic.csv"
    if not meta_csv.is_file():
        fail(f"metadata-synthetic.csv not found in {dataset_root}")

    df = pd.read_csv(meta_csv)
    missing = {"rgb_path", "map", "type", "weather", "camera_position"} - set(df.columns)
    if missing:
        fail(f"metadata is missing columns: {sorted(missing)}")

    split = build_split(df)
    new_bytes = split.to_csv(index=False, lineterminator="\n").encode("ascii")
    new_sha = sha256_bytes(new_bytes)

    # Freeze guard: checked before anything is written.
    manifest = load_manifest()
    frozen_same = False
    if manifest is not None and manifest.get("frozen") is True:
        if manifest.get("split_sha256") != new_sha:
            if not args.unfreeze:
                fail(f"{OUT_MANIFEST.name} is frozen (split_sha256 {manifest.get('split_sha256')}) but this run "
                     f"would produce {new_sha}. Nothing written. Rerun with --unfreeze to replace the split.")
            print("--unfreeze given: replacing the frozen split")
        else:
            frozen_same = True

    if OUT_CSV.exists() and OUT_CSV.read_bytes() == new_bytes:
        print(f"{OUT_CSV.name} already exists with identical content (unchanged).")
    else:
        OUT_CSV.write_bytes(new_bytes)
        print(f"wrote {OUT_CSV}")

    if frozen_same:
        print(f"{OUT_MANIFEST.name} is frozen with the same split_sha256 (unchanged).")
    else:
        val_sizes = {str(k): int((split["val_fold"] == k).sum()) for k in sorted(ACTIVE_FOLDS)}
        out = {
            "split_version": SPLIT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **MANIFEST_INFO,
            "folds": {str(k): v for k, v in ACTIVE_FOLDS.items()},
            "val_sizes": val_sizes,
            "split_sha256": new_sha,
            "frozen": True,
        }
        OUT_MANIFEST.write_text(json.dumps(out, indent=2) + "\n", encoding="ascii", newline="\n")
        print(f"wrote {OUT_MANIFEST}")
    print(f"split_sha256: {new_sha}")

    text = (
        f"Synthetic scenario-grouped split ({SPLIT_VERSION})\nmetadata: {meta_csv}\noutput  : {OUT_CSV}\n"
        f"sha256  : {new_sha}\n\n"
        + summarize(split)
        + "\n"
    )
    print()
    print(text)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.write_text(text, encoding="ascii", newline="\n")
    print(f"saved {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
