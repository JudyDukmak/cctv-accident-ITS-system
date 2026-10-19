"""Make the fixed, scenario-grouped synthetic train/val split.

Synthetic clips are re-runs of ~14 scripted crashes (one per map + accident type),
filmed from many cameras and weathers (see check_synthetic_runs.py). A per-clip
random split would put the same crash site in train and val, so clips are grouped by
scenario_id = f"{map}_{type}" and whole scenarios are held out per fold.

The folds are fixed by FOLDS below (no randomness, no sklearn). For fold k the
validation set is every row with val_fold == k and the training set is the rest.
split_dev is the default quick split (fold 0 as validation).

READ-ONLY on the dataset: reads only metadata-synthetic.csv. No video, annotation
or real-data file is opened.

Usage:
    py make_synthetic_split.py [--dataset-path PATH] [--overwrite]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# fold index = the fold in which these scenarios are held out as validation
FOLDS = {
    0: ["Town05_head-on", "Town04_rear-end", "Town05_sideswipe", "Town10HD_t-bone", "Town07_single"],
    1: ["Town03_head-on", "Town07_rear-end", "Town04_sideswipe", "Town03_t-bone", "Town10HD_single"],
    2: ["Town06_head-on", "Town05_rear-end", "Town06_sideswipe", "Town07_t-bone"],
}
DEV_VAL_FOLD = 0
MULTI_VEHICLE_TYPES = ["head-on", "rear-end", "sideswipe", "t-bone"]

# Expected numbers, compared as warnings only.
EXPECTED_VAL_SIZE = {0: 670, 1: 860, 2: 681}
EXPECTED_TOTAL = 2211
EXPECTED_VAL_TYPES = {
    0: {"rear-end": 245, "head-on": 174, "t-bone": 131, "sideswipe": 107, "single": 13},
    1: {"rear-end": 272, "head-on": 212, "t-bone": 184, "sideswipe": 139, "single": 53},
    2: {"rear-end": 277, "head-on": 202, "sideswipe": 159, "t-bone": 43, "single": 0},
}

DEFAULT_DATASET = Path(
    r"C:\Users\ASUS\Documents\Uni\Senior 2\cctv-accident-ITS-system\ACCIDENT\dataset"
)
REPO_ROOT = Path(__file__).resolve().parent
OUT_CSV = REPO_ROOT / "synthetic_split.csv"
OUT_SUMMARY = REPO_ROOT / "synthetic_check" / "split_summary.txt"
OUT_COLUMNS = ["rgb_path", "scenario_id", "val_fold", "split_dev", "map", "type", "weather", "camera_position"]


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def build_split(df):
    if df["rgb_path"].duplicated().any():
        dup = df.loc[df["rgb_path"].duplicated(keep=False), "rgb_path"].head(10).tolist()
        fail(f"rgb_path is not unique, e.g. {dup}")

    scenario_to_fold = {}
    for k, scenarios in FOLDS.items():
        for s in scenarios:
            if s in scenario_to_fold:
                fail(f"scenario {s} is listed in fold {scenario_to_fold[s]} and fold {k}")
            scenario_to_fold[s] = k
    if len(scenario_to_fold) != 14:
        fail(f"FOLDS lists {len(scenario_to_fold)} scenarios, expected 14")

    out = df.copy()
    out["scenario_id"] = out["map"].astype(str) + "_" + out["type"].astype(str)
    in_data = set(out["scenario_id"])
    in_folds = set(scenario_to_fold)
    if in_data != in_folds:
        fail(
            "scenario_ids in the data and in FOLDS differ. "
            f"Only in data: {sorted(in_data - in_folds)}; only in FOLDS: {sorted(in_folds - in_data)}"
        )

    out["val_fold"] = out["scenario_id"].map(scenario_to_fold)
    if out["val_fold"].isna().any():
        fail(f"{int(out['val_fold'].isna().sum())} rows got no fold")
    out["val_fold"] = out["val_fold"].astype(int)

    for k in FOLDS:
        present = set(out.loc[out["val_fold"] == k, "type"])
        missing = [t for t in MULTI_VEHICLE_TYPES if t not in present]
        if missing:
            fail(f"fold {k} validation is missing multi-vehicle types {missing}")

    out["split_dev"] = (out["val_fold"] == DEV_VAL_FOLD).map({True: "val", False: "train"})
    out["camera_position"] = out["camera_position"].astype(int)
    return out[OUT_COLUMNS]


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
    for k in sorted(FOLDS):
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
    for k in sorted(FOLDS):
        check(f"fold {k} val size", int((split["val_fold"] == k).sum()), EXPECTED_VAL_SIZE[k])
    for k in sorted(FOLDS):
        counts = split.loc[split["val_fold"] == k, "type"].value_counts()
        for t, exp in EXPECTED_VAL_TYPES[k].items():
            check(f"fold {k} val {t}", int(counts.get(t, 0)), exp)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing synthetic_split.csv whose content differs")
    args = ap.parse_args()

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
    new_csv = split.to_csv(index=False, lineterminator="\n")

    if OUT_CSV.exists():
        old_csv = OUT_CSV.read_text(encoding="ascii")
        if old_csv == new_csv:
            print(f"{OUT_CSV.name} already exists with identical content (unchanged).")
        elif not args.overwrite:
            fail(f"{OUT_CSV} exists with different content; rerun with --overwrite to replace it")
        else:
            OUT_CSV.write_text(new_csv, encoding="ascii", newline="")
            print(f"overwrote {OUT_CSV}")
    else:
        OUT_CSV.write_text(new_csv, encoding="ascii", newline="")
        print(f"wrote {OUT_CSV}")

    text = (
        f"Synthetic scenario-grouped split\nmetadata: {meta_csv}\noutput  : {OUT_CSV}\n\n"
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
