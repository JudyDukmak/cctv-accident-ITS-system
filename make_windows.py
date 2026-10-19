"""Cut every synthetic clip into fixed-length sliding windows with accident labels.

SYNTHETIC DATA ONLY. Reads synthetic_split.csv (repo root) and metadata-synthetic.csv.
No video or annotation file is opened; real data is never touched. No randomness.

Window j: t_start = j * stride_s; sample times t_i = t_start + i / model_fps for
i = 0..window_frames-1; t_end = last t_i. Frame index of time t is
floor(t * fps_video + 0.5). A window is kept while its last frame index <= no_frames - 1.

Labels (t_star = accident_time, m = ignore_margin_s):
    t_end <  t_star - m                    -> 0,  "pre"
    t_star - m <= t_end < t_star           -> -1, "gap"   (ignored in training)
    t_end >= t_star and t_start < t_star   -> 1,  "onset"
    t_start >= t_star                      -> 1,  "post"

window_frame_indices() is importable by the training code.

Usage:
    py make_windows.py [--dataset-path PATH] [--model-fps 8] [--window-frames 16]
                       [--stride-s 0.5] [--ignore-margin-s 1.0] [--overwrite]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATASET = Path(
    r"C:\Users\ASUS\Documents\Uni\Senior 2\cctv-accident-ITS-system\ACCIDENT\dataset"
)
REPO_ROOT = Path(__file__).resolve().parent
SPLIT_CSV = REPO_ROOT / "synthetic_split.csv"
OUT_CSV = REPO_ROOT / "synthetic_windows.csv"
OUT_CONFIG = REPO_ROOT / "windows_config.json"
OUT_SUMMARY = REPO_ROOT / "synthetic_check" / "windows_summary.txt"
EXPECTED_ROWS = 2211
FRAME_TIME_TOL = 0.026  # s, about half a frame at 20 fps
TIME_DECIMALS = 6  # window times are rounded to remove float noise (e.g. 9.55 - 1.0)
LABEL_ORDER = {0: 0, -1: 1, 1: 2}  # labels may only move forward in this order
OUT_COLUMNS = ["rgb_path", "scenario_id", "val_fold", "split_dev", "window_idx", "t_start", "t_end",
               "frame_first", "frame_last", "t_star", "label", "phase"]
EXPECTED_CLIPS = {
    "Town05_sideswipe_rain_44": dict(windows=36, label0=14, label_m1=2, label1=20, onset=4, post=16),
    "Town05_sideswipe_clear_00": dict(windows=38, label0=12, label_m1=2, label1=24, onset=4, post=20),
}
TIMELINE_CLIP = "Town05_sideswipe_rain_44"


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def frame_index(t, fps_video):
    """floor(t * fps + 0.5); the product is rounded to 9 decimals so float noise
    (e.g. 2.4999999999) cannot shift a frame."""
    return int(math.floor(round(t * fps_video, 9) + 0.5))


def window_frame_indices(t_start, fps_video, model_fps, window_frames):
    """Frame indices of the window starting at t_start (seconds)."""
    return [frame_index(t_start + i / model_fps, fps_video) for i in range(window_frames)]


def label_window(t_start, t_end, t_star, margin):
    if t_end < round(t_star - margin, TIME_DECIMALS):
        return 0, "pre"
    if t_end < t_star:
        return -1, "gap"
    if t_start < t_star:
        return 1, "onset"
    return 1, "post"


def clip_windows(row, cfg):
    """All windows of one clip, as a list of dicts."""
    fps_video = row["fps_video"]
    last_valid = int(row["no_frames"]) - 1
    t_star = round(float(row["accident_time"]), TIME_DECIMALS)
    out = []
    j = 0
    while True:
        t_start = j * cfg["stride_s"]
        frames = window_frame_indices(t_start, fps_video, cfg["model_fps"], cfg["window_frames"])
        if frames[-1] > last_valid:
            break
        ts = round(t_start, TIME_DECIMALS)
        te = round(t_start + (cfg["window_frames"] - 1) / cfg["model_fps"], TIME_DECIMALS)
        label, phase = label_window(ts, te, t_star, cfg["ignore_margin_s"])
        out.append({
            "rgb_path": row["rgb_path"], "scenario_id": row["scenario_id"],
            "val_fold": row["val_fold"], "split_dev": row["split_dev"], "window_idx": j,
            "t_start": ts, "t_end": te, "frame_first": frames[0], "frame_last": frames[-1],
            "t_star": t_star, "label": label, "phase": phase,
        })
        j += 1
    return out


def load_inputs(dataset_root):
    meta_csv = dataset_root / "metadata-synthetic.csv"
    if not meta_csv.is_file():
        fail(f"metadata-synthetic.csv not found in {dataset_root}")
    if not SPLIT_CSV.is_file():
        fail(f"{SPLIT_CSV} not found - run make_synthetic_split.py first")
    split = pd.read_csv(SPLIT_CSV)
    meta = pd.read_csv(meta_csv)
    need = ["rgb_path", "accident_time", "accident_frame", "no_frames", "duration"]
    missing = [c for c in need if c not in meta.columns]
    if missing:
        fail(f"metadata is missing columns {missing}")
    for name, d in (("synthetic_split.csv", split), ("metadata", meta)):
        if d["rgb_path"].duplicated().any():
            fail(f"rgb_path is not unique in {name}")
    df = split.merge(meta[need], on="rgb_path", how="inner", validate="one_to_one")
    if not (len(df) == len(split) == len(meta) == EXPECTED_ROWS):
        fail(f"merge is not 1:1 on {EXPECTED_ROWS} rows: split {len(split)}, "
             f"metadata {len(meta)}, merged {len(df)}")
    bad = df[(df["duration"] <= 0) | (df["no_frames"] <= 0)]
    if len(bad):
        fail(f"{len(bad)} clips have non-positive duration or no_frames, e.g. {bad['rgb_path'].iloc[0]}")
    df["fps_video"] = df["no_frames"] / df["duration"]
    df["clip"] = df["rgb_path"].map(lambda p: Path(p.replace("\\", "/")).stem)
    return df, meta_csv


def check_windows(win, df, margin):
    """Hard checks; stops on the first failure."""
    counts = win.groupby("rgb_path").size()
    no_win = sorted(set(df["rgb_path"]) - set(counts.index))
    if no_win:
        fail(f"{len(no_win)} clips have no window, e.g. {no_win[:5]}")
    for path, g in win.groupby("rgb_path", sort=False):
        if not np.array_equal(g["window_idx"].to_numpy(), np.arange(len(g))):
            fail(f"windows are not consecutive for {path}")
        if (np.diff(g["label"].map(LABEL_ORDER).to_numpy()) < 0).any():
            fail(f"labels go backwards (not 0, -1, 1 order) for {path}")
    bad1 = win[(win["label"] == 1) & (win["t_end"] < win["t_star"])]
    if len(bad1):
        fail(f"{len(bad1)} label-1 windows have t_end < t_star, e.g. {bad1['rgb_path'].iloc[0]}")
    bad0 = win[(win["label"] == 0) & (win["t_end"] >= (win["t_star"] - margin).round(TIME_DECIMALS))]
    if len(bad0):
        fail(f"{len(bad0)} label-0 windows have t_end >= t_star - margin, e.g. {bad0['rgb_path'].iloc[0]}")


def pct_table(s, name):
    t = s.value_counts().sort_index().rename_axis(name).rename("n").to_frame()
    t["pct"] = (100 * t["n"] / t["n"].sum()).map("{:.1f}".format)
    return t.to_string()


def summarize(win, df, cfg):
    lines = []
    p = lines.append
    w = win.merge(df[["rgb_path", "clip", "type"]], on="rgb_path")

    p("0. Input checks")
    p("   fps_video = no_frames / duration, rounded to 0.1:")
    p("   " + df["fps_video"].round(1).value_counts().sort_index().rename_axis("fps").rename("clips")
      .to_string().replace("\n", "\n   "))
    err = (df["accident_frame"] / df["fps_video"] - df["accident_time"]).abs()
    off = df[err > FRAME_TIME_TOL]
    p(f"   clips with |accident_frame/fps_video - accident_time| > {FRAME_TIME_TOL}: {len(off)} (kept)")
    for _, r in off.head(20).iterrows():
        p(f"     {r['clip']}: frame {r['accident_frame']} / fps {r['fps_video']:.3f} = "
          f"{r['accident_frame'] / r['fps_video']:.3f}s vs accident_time {r['accident_time']:.3f}s")
    if len(off) > 20:
        p(f"     ... ({len(off) - 20} more)")
    p("")

    p(f"1. Total windows: {len(win)}")
    p("   per label:")
    p("   " + pct_table(win["label"], "label").replace("\n", "\n   "))
    p("   per phase:")
    p("   " + pct_table(win["phase"], "phase").replace("\n", "\n   "))
    p("")

    p("2. val_fold x label")
    p(pd.crosstab(w["val_fold"], w["label"], margins=True).to_string())
    p("   type x label")
    p(pd.crosstab(w["type"], w["label"], margins=True).to_string())
    p("")

    per_clip = win.groupby("rgb_path").size()
    p("3. Windows per clip: min {} / median {:g} / max {}".format(
        per_clip.min(), per_clip.median(), per_clip.max()))
    p("   accident_time (s): min {:.2f} / median {:.2f} / max {:.2f}".format(
        df["accident_time"].min(), df["accident_time"].median(), df["accident_time"].max()))
    p("")

    n0 = w[w["label"] == 0].groupby("rgb_path").size()
    n1 = w[w["label"] == 1].groupby("rgb_path").size()
    zero0 = df[~df["rgb_path"].isin(n0.index)]
    zero1 = df[~df["rgb_path"].isin(n1.index)]
    p(f"4. Clips with ZERO label-0 windows: {len(zero0)}")
    p("   5 clips with the smallest accident_time:")
    for _, r in df.nsmallest(5, "accident_time").iterrows():
        p(f"     {r['clip']}: accident_time {r['accident_time']:.2f}s, "
          f"label-0 windows {int(n0.get(r['rgb_path'], 0))}")
    p(f"   Clips with ZERO label-1 windows: {len(zero1)} (should be 0)")
    for c in zero1["clip"].head(10):
        p(f"     {c}")
    p("")

    p(f"5. Timeline of {TIMELINE_CLIP}")
    tl = w[w["clip"] == TIMELINE_CLIP]
    if tl.empty:
        p("   clip not found")
    else:
        r = df[df["clip"] == TIMELINE_CLIP].iloc[0]
        p(f"   t_star {r['accident_time']:.2f}s, fps_video {r['fps_video']:.3f}, no_frames {r['no_frames']}")
        p(tl[["window_idx", "t_start", "t_end", "label", "phase"]].to_string(index=False))
        onset = tl[tl["phase"] == "onset"]
        ex = onset.iloc[0] if len(onset) else tl.iloc[0]
        idx = window_frame_indices(ex["t_start"], r["fps_video"], cfg["model_fps"], cfg["window_frames"])
        p(f"   frame indices of window {ex['window_idx']} (t_start {ex['t_start']}s, {ex['phase']}):")
        p(f"     {idx}")
        p(f"     spacing: {np.diff(idx).tolist()}")
    p("")

    p("6. EXPECTED comparison (warnings only)")
    for clip, exp in EXPECTED_CLIPS.items():
        g = w[w["clip"] == clip]
        got = dict(windows=len(g), label0=int((g["label"] == 0).sum()), label_m1=int((g["label"] == -1).sum()),
                   label1=int((g["label"] == 1).sum()), onset=int((g["phase"] == "onset").sum()),
                   post=int((g["phase"] == "post").sum()))
        for k, v in exp.items():
            p(f"   {'OK      ' if got[k] == v else 'MISMATCH'} {clip} {k}: got {got[k]}, expected {v}")
    return "\n".join(lines)


def write_guarded(path, text, overwrite):
    """Write text; never silently replace a different existing file."""
    if path.exists():
        if path.read_text(encoding="ascii") == text:
            print(f"{path.name} already exists with identical content (unchanged).")
            return
        if not overwrite:
            fail(f"{path} exists with different content; rerun with --overwrite to replace it")
        print(f"overwriting {path}")
    path.write_text(text, encoding="ascii", newline="")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--model-fps", type=float, default=8)
    ap.add_argument("--window-frames", type=int, default=16)
    ap.add_argument("--stride-s", type=float, default=0.5)
    ap.add_argument("--ignore-margin-s", type=float, default=1.0)
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing synthetic_windows.csv / windows_config.json if they differ")
    args = ap.parse_args()
    if args.model_fps <= 0 or args.window_frames < 1 or args.stride_s <= 0 or args.ignore_margin_s < 0:
        fail("invalid parameters")

    dataset_root = args.dataset_path.resolve()
    if (dataset_root / "dataset" / "metadata-synthetic.csv").is_file():
        dataset_root = dataset_root / "dataset"
    cfg = {"model_fps": args.model_fps, "window_frames": args.window_frames,
           "stride_s": args.stride_s, "ignore_margin_s": args.ignore_margin_s}

    df, meta_csv = load_inputs(dataset_root)
    rows = []
    for _, r in df.iterrows():
        rows.extend(clip_windows(r, cfg))
    win = pd.DataFrame(rows, columns=OUT_COLUMNS)
    check_windows(win, df, cfg["ignore_margin_s"])

    config = {
        **cfg,
        "frame_index_rule": "floor(t * fps_video + 0.5), fps_video = no_frames / duration",
        "label_rule": {"0 pre": "t_end < t_star - margin", "-1 gap": "t_star - margin <= t_end < t_star",
                       "1 onset": "t_end >= t_star and t_start < t_star", "1 post": "t_start >= t_star"},
        "inputs": {"split": SPLIT_CSV.name, "metadata": str(meta_csv)},
        "output": OUT_CSV.name,
        "n_clips": len(df),
        "n_windows": len(win),
    }
    write_guarded(OUT_CSV, win.to_csv(index=False, lineterminator="\n"), args.overwrite)
    write_guarded(OUT_CONFIG, json.dumps(config, indent=2) + "\n", args.overwrite)

    text = (f"Synthetic sliding windows\nconfig: {json.dumps(cfg)}\nmetadata: {meta_csv}\n\n"
            + summarize(win, df, cfg) + "\n")
    print()
    print(text)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.write_text(text, encoding="ascii", newline="\n")
    print(f"saved {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
