"""One row per synthetic scenario (map + accident type), plus a per-map table.

READ-ONLY, SYNTHETIC DATA ONLY. Reads metadata-synthetic.csv, synthetic_split.csv
and (if present) synthetic_check/run_check.csv, whose crash_x / crash_y are the
estimated crash world positions of the clips sampled by check_synthetic_runs.py.
Never opens real data. No randomness. synthetic_split.csv is only read.

Outputs: synthetic_check/scenario_table.csv and synthetic_check/scenario_table.md.

Usage:
    py make_scenario_table.py [--dataset-path PATH]
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATASET = Path(
    r"C:\Users\ASUS\Documents\Uni\Senior 2\cctv-accident-ITS-system\ACCIDENT\dataset"
)
REPO_ROOT = Path(__file__).resolve().parent
SPLIT_CSV = REPO_ROOT / "synthetic_split.csv"
OUT_DIR = REPO_ROOT / "synthetic_check"
RUN_CHECK = OUT_DIR / "run_check.csv"
EXPECTED_CLIPS = 2211
COLUMNS = ["scenario_id", "map", "crash_location", "collision_type", "n_views", "n_clips",
           "median_accident_time_s", "current_val_fold", "n_sampled", "spread_m"]


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def clip_name(p):
    return Path(str(p).replace("\\", "/")).stem


def crash_locations(meta):
    """Per scenario: median crash (x, y), n sampled clips and spread_m, from run_check.csv.

    Returns (DataFrame indexed by scenario_id, number of joined rows), or (None, 0).
    """
    if not RUN_CHECK.is_file():
        print(f"{RUN_CHECK.name}: not found -> crash_location = not available")
        return None, 0
    rc = pd.read_csv(RUN_CHECK)
    print(f"{RUN_CHECK.name} columns ({len(rc.columns)}): {list(rc.columns)}")
    if not {"clip", "crash_x", "crash_y"} <= set(rc.columns):
        print("  needs columns clip, crash_x, crash_y -> crash_location = not available")
        return None, 0

    print("  join: run_check.clip (video file stem, e.g. Town05_sideswipe_rain_44)")
    print("        == metadata clip = stem of rgb_path; scenario_id = metadata map + '_' + metadata type")
    j = rc.merge(meta[["clip", "scenario_id", "map", "type"]], on="clip", how="left",
                 suffixes=("_rc", ""), validate="many_to_one")
    unmatched = j["scenario_id"].isna()
    no_xy = ~unmatched & j[["crash_x", "crash_y"]].isna().any(axis=1)
    if {"map_rc", "type_rc"} <= set(j.columns):
        disagree = ~unmatched & ((j["map_rc"] != j["map"]) | (j["type_rc"] != j["type"]))
        print(f"  rows whose run_check map/type disagree with metadata: {int(disagree.sum())}")
    joined = j[~unmatched & ~no_xy]
    print(f"  rows: {len(rc)} total, {len(joined)} joined with crash_x/crash_y, "
          f"{int(unmatched.sum())} not joined (clip not in metadata), "
          f"{int(no_xy.sum())} joined but without crash coordinates")
    for c in j.loc[unmatched, "clip"].head(10):
        print(f"    not joined: {c}")

    rows = {}
    for sid, g in joined.groupby("scenario_id"):
        xy = g[["crash_x", "crash_y"]].to_numpy(dtype=float)
        med = np.median(xy, axis=0)
        rows[sid] = {"x": med[0], "y": med[1], "n": len(g),
                     "spread_m": float(np.max(np.linalg.norm(xy - med, axis=1)))}
    return pd.DataFrame.from_dict(rows, orient="index"), len(joined)


def to_markdown(df):
    cells = [[str(v) for v in row] for row in df.astype(object).values]
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    return "\n".join([head, sep] + ["| " + " | ".join(r) + " |" for r in cells])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    args = ap.parse_args()

    dataset_root = args.dataset_path.resolve()
    if (dataset_root / "dataset" / "metadata-synthetic.csv").is_file():
        dataset_root = dataset_root / "dataset"
    meta_csv = dataset_root / "metadata-synthetic.csv"
    if not meta_csv.is_file():
        fail(f"metadata-synthetic.csv not found in {dataset_root}")
    if not SPLIT_CSV.is_file():
        fail(f"{SPLIT_CSV} not found")
    OUT_DIR.mkdir(exist_ok=True)

    meta = pd.read_csv(meta_csv)
    split = pd.read_csv(SPLIT_CSV, usecols=["rgb_path", "val_fold"])
    need = {"rgb_path", "map", "type", "camera_position", "accident_time"} - set(meta.columns)
    if need:
        fail(f"metadata is missing columns {sorted(need)}")
    meta["scenario_id"] = meta["map"].astype(str) + "_" + meta["type"].astype(str)
    meta["clip"] = meta["rgb_path"].map(clip_name)
    df = meta.merge(split, on="rgb_path", how="left", validate="one_to_one")
    if df["val_fold"].isna().any():
        fail(f"{int(df['val_fold'].isna().sum())} clips have no fold in synthetic_split.csv")
    folds_per_scenario = df.groupby("scenario_id")["val_fold"].nunique()
    if (folds_per_scenario > 1).any():
        fail(f"scenarios split over several folds: {folds_per_scenario[folds_per_scenario > 1].index.tolist()}")

    print(f"metadata: {meta_csv}")
    print(f"split   : {SPLIT_CSV}")
    loc, n_joined = crash_locations(meta)

    tab = df.groupby("scenario_id").agg(
        map=("map", "first"),
        collision_type=("type", "first"),
        n_views=("camera_position", "nunique"),
        n_clips=("rgb_path", "size"),
        median_accident_time_s=("accident_time", "median"),
        current_val_fold=("val_fold", "first"),
    ).reset_index()
    tab["current_val_fold"] = tab["current_val_fold"].astype(int)
    tab["median_accident_time_s"] = tab["median_accident_time_s"].round(2)
    if loc is None:
        loc = pd.DataFrame(columns=["x", "y", "n", "spread_m"])
    has = tab["scenario_id"].isin(loc.index)
    tab["crash_location"] = tab["scenario_id"].map(
        lambda s: f"x={loc.at[s, 'x']:.1f}, y={loc.at[s, 'y']:.1f} (n={int(loc.at[s, 'n'])})"
        if s in loc.index else "not available")
    tab["n_sampled"] = tab["scenario_id"].map(lambda s: int(loc.at[s, "n"]) if s in loc.index else 0)
    tab["spread_m"] = tab["scenario_id"].map(
        lambda s: round(loc.at[s, "spread_m"], 1) if s in loc.index else np.nan)
    tab = tab[COLUMNS]

    n_sampled_total = int(tab["n_sampled"].sum())
    print(f"\nn_sampled sum: {n_sampled_total} vs joined rows {n_joined} -> "
          f"{'OK' if n_sampled_total == n_joined else 'MISMATCH'}")
    if n_sampled_total != n_joined:
        fail(f"n_sampled sums to {n_sampled_total}, but {n_joined} rows were joined")

    pairs = []
    for mp, g in tab[has].groupby("map"):
        for a, b in itertools.combinations(g["scenario_id"], 2):
            d = float(np.hypot(loc.at[a, "x"] - loc.at[b, "x"], loc.at[a, "y"] - loc.at[b, "y"]))
            pairs.append({"map": mp, "scenario_A": a, "scenario_B": b, "distance_m": round(d, 1),
                          "spread_m_A": round(loc.at[a, "spread_m"], 1),
                          "spread_m_B": round(loc.at[b, "spread_m"], 1)})
    pairs = pd.DataFrame(pairs, columns=["map", "scenario_A", "scenario_B", "distance_m",
                                         "spread_m_A", "spread_m_B"])

    total = int(tab["n_clips"].sum())
    print(f"\nn_clips sum: {total} ({'OK' if total == EXPECTED_CLIPS else 'MISMATCH'}, expected {EXPECTED_CLIPS})")
    if total != EXPECTED_CLIPS:
        fail(f"n_clips sums to {total}, expected {EXPECTED_CLIPS}")

    maps = tab.groupby("map").agg(
        n_scenarios=("scenario_id", "size"),
        scenario_ids=("scenario_id", lambda s: ", ".join(s)),
        n_clips=("n_clips", "sum"),
        val_folds=("current_val_fold", lambda f: "{" + ", ".join(str(v) for v in sorted(set(f))) + "}"),
    ).reset_index()

    with pd.option_context("display.width", 250, "display.max_colwidth", 120):
        print("\nScenario table")
        print(tab.to_string(index=False))
        print("\nMap table")
        print(maps.to_string(index=False))
        print("\nScenario pairs on the same map (distance between median crash points, metres)")
        print(pairs.to_string(index=False) if len(pairs) else "  none (no crash locations available)")

    tab.to_csv(OUT_DIR / "scenario_table.csv", index=False)
    md = ("# Synthetic scenarios\n\n" + to_markdown(tab)
          + "\n\n# Maps\n\n" + to_markdown(maps) + "\n")
    (OUT_DIR / "scenario_table.md").write_text(md, encoding="ascii")
    print(f"\nsaved {OUT_DIR / 'scenario_table.csv'}")
    print(f"saved {OUT_DIR / 'scenario_table.md'}")


if __name__ == "__main__":
    main()
