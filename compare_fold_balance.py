"""Compare window-label balance of the archived v1 and the active v2 synthetic fold assignment.

READ-ONLY, SYNTHETIC DATA ONLY. Reads synthetic_windows.csv, synthetic_split.csv,
archive/split_v1_handmade/synthetic_split_v1_handmade.csv and metadata-synthetic.csv.
Never opens real data and modifies no file except the report
synthetic_check/fold_balance_check.txt.

Labels come from the current synthetic_windows.csv; each window's validation fold is
taken from the assignment being evaluated (via scenario_id), not from its val_fold
column. Label-1 share = label-1 windows / all windows (definition (a)).

Usage:
    py compare_fold_balance.py [--dataset-path PATH]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

DEFAULT_DATASET = Path(
    r"C:\Users\ASUS\Documents\Uni\Senior 2\cctv-accident-ITS-system\ACCIDENT\dataset"
)
REPO_ROOT = Path(__file__).resolve().parent
WINDOWS_CSV = REPO_ROOT / "synthetic_windows.csv"
SPLIT_CSV = REPO_ROOT / "synthetic_split.csv"
V1_SPLIT_CSV = REPO_ROOT / "archive" / "split_v1_handmade" / "synthetic_split_v1_handmade.csv"
OUT_TXT = REPO_ROOT / "synthetic_check" / "fold_balance_check.txt"

# Known label-1 share per validation fold (percent, definition (a)).
KNOWN_SHARE = {
    "V1_HANDMADE": {0: 75.5, 1: 68.3, 2: 41.0},
    "V2_ACTIVE": {0: 64.4, 1: 63.4, 2: 59.1},
}
FOLDS = [0, 1, 2]


class Report:
    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(str(text))


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def label1_share(g):
    """Definition (a): label-1 windows / all windows (labels -1, 0, 1), in percent."""
    return 100 * (g["label"] == 1).sum() / len(g)


def fold_assignment(path):
    """scenario_id -> val_fold from a split CSV; each scenario must have exactly one fold."""
    if not path.is_file():
        fail(f"{path} not found")
    s = pd.read_csv(path, usecols=["scenario_id", "val_fold"])
    per = s.groupby("scenario_id")["val_fold"].agg(["nunique", "first"])
    if (per["nunique"] > 1).any():
        fail(f"{path.name}: scenarios with several folds: {per.index[per['nunique'] > 1].tolist()}")
    return per["first"].astype(int).to_dict()


def fold_table(win, clips, fold_of):
    rows = []
    for k in FOLDS:
        scen = {s for s, f in fold_of.items() if f == k}
        w = win[win["scenario_id"].isin(scen)]
        c = clips[clips["scenario_id"].isin(scen)]
        n0 = w[w["label"] == 0].groupby("rgb_path").size().reindex(c["rgb_path"], fill_value=0)
        rows.append({
            "val_fold": k,
            "n_clips": len(c),
            "median_accident_time_s": round(float(c["accident_time"].median()), 2),
            "win_-1": int((w["label"] == -1).sum()),
            "win_0": int((w["label"] == 0).sum()),
            "win_1": int((w["label"] == 1).sum()),
            "label1_share_pct": round(float(label1_share(w)), 1),
            "mean_label0_per_clip": round(float(n0.mean()), 2),
            "clips_zero_label0": int((n0 == 0).sum()),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    args = ap.parse_args()

    dataset_root = args.dataset_path.resolve()
    if (dataset_root / "dataset" / "metadata-synthetic.csv").is_file():
        dataset_root = dataset_root / "dataset"
    meta_csv = dataset_root / "metadata-synthetic.csv"
    for f in (meta_csv, WINDOWS_CSV, SPLIT_CSV):
        if not f.is_file():
            fail(f"{f} not found")

    win = pd.read_csv(WINDOWS_CSV)
    split = pd.read_csv(SPLIT_CSV, usecols=["rgb_path", "scenario_id", "val_fold"])
    meta = pd.read_csv(meta_csv, usecols=["rgb_path", "map", "type", "accident_time"])
    meta["scenario_meta"] = meta["map"].astype(str) + "_" + meta["type"].astype(str)

    r = Report()
    r("Fold balance check (synthetic only)")
    r(f"synthetic_windows.csv columns: {list(win.columns)}")
    r("join: each window carries rgb_path and scenario_id; scenario_id is used directly and")
    r("      cross-checked against metadata map + '_' + type via rgb_path.")
    r("      The evaluated fold of a window = assignment[scenario_id] (its val_fold column is not used).")

    chk = win[["rgb_path", "scenario_id", "val_fold"]].drop_duplicates().merge(
        meta[["rgb_path", "scenario_meta"]], on="rgb_path", how="left")
    n_missing = int(chk["scenario_meta"].isna().sum())
    n_bad = int((chk["scenario_id"] != chk["scenario_meta"]).sum()) - n_missing
    r(f"  windows: {len(win)}, clips with windows: {win['rgb_path'].nunique()}, "
      f"clips not in metadata: {n_missing}, scenario_id mismatches vs metadata: {n_bad}")
    if n_missing or n_bad:
        fail("windows do not join cleanly to metadata")

    stale = chk[["rgb_path", "val_fold"]].merge(split[["rgb_path", "val_fold"]], on="rgb_path",
                                                how="left", suffixes=("_win", "_split"))
    n_stale = int((stale["val_fold_win"] != stale["val_fold_split"]).sum())
    r(f"  clips whose val_fold in synthetic_windows.csv differs from synthetic_split.csv: {n_stale}")

    assignments = {
        "V1_HANDMADE": fold_assignment(V1_SPLIT_CSV),
        "V2_ACTIVE": fold_assignment(SPLIT_CSV),
    }
    scen_windows = set(win["scenario_id"])
    for name, a in assignments.items():
        if set(a) != scen_windows:
            fail(f"{name}: scenarios differ from synthetic_windows.csv; missing "
                 f"{sorted(scen_windows - set(a))}, extra {sorted(set(a) - scen_windows)}")
    r(f"  V1_HANDMADE from {V1_SPLIT_CSV.relative_to(REPO_ROOT)}")
    r(f"  V2_ACTIVE   from {SPLIT_CSV.relative_to(REPO_ROOT)}")

    clips = meta.rename(columns={"scenario_meta": "scenario_id"})[["rgb_path", "scenario_id", "accident_time"]]

    r("")
    r("1. Self-checks: label-1 share per validation fold, definition (a) = label-1 / all windows")
    n_mismatch = 0
    tables = {}
    for name, a in assignments.items():
        t = fold_table(win, clips, a)
        if int(t["n_clips"].sum()) != len(clips):
            fail(f"{name}: folds cover {int(t['n_clips'].sum())} clips, expected {len(clips)}")
        tables[name] = t
        got = dict(zip(t["val_fold"], t["label1_share_pct"]))
        exp = KNOWN_SHARE[name]
        ok = all(abs(got[k] - exp[k]) < 0.05 for k in FOLDS)
        n_mismatch += not ok
        r(f"   {'OK      ' if ok else 'MISMATCH'} {name:<12} got "
          + " / ".join(f"{got[k]:.1f}" for k in FOLDS)
          + "   expected " + " / ".join(f"{exp[k]:.1f}" for k in FOLDS))

    r("")
    r("2. Per validation fold")
    with pd.option_context("display.width", 200):
        for name, t in tables.items():
            r(f"   {name}")
            r("   " + t.to_string(index=False).replace("\n", "\n   "))
            r("")

    r("3. label-1 share spread across folds (max - min, definition (a))")
    for name, t in tables.items():
        r(f"   {name:<12} {t['label1_share_pct'].max() - t['label1_share_pct'].min():.1f} percentage points")

    OUT_TXT.parent.mkdir(exist_ok=True)
    OUT_TXT.write_text("\n".join(r.lines) + "\n", encoding="ascii")
    print(f"\nsaved {OUT_TXT}")
    if n_mismatch:
        fail(f"{n_mismatch} self-check(s) MISMATCH (report saved)")


if __name__ == "__main__":
    main()
