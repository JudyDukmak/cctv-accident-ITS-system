"""How large is the crash box after preprocessing, and is it visible?

READ-ONLY, SYNTHETIC DATA ONLY. Reads metadata-synthetic.csv and only the synthetic
videos listed in it. Never opens real data. No randomness.

STEP 0: column and coordinate checks.
PART A: crash box size (metadata only, all clips) at full resolution, after
        short-side resize and after squash to S x S (S = 224, 336, 448), and how
        often a central square crop cuts the box. Saved to
        synthetic_check/crash_size_summary.txt.
PART B: zoomed crops around the box at accident_frame + [-10, -5, -2, 0, 2, 5, 10],
        read sequentially from frame 0. Saved as synthetic_check/zoom_<clip>.png.

Usage:
    py check_crash_visibility.py [--dataset-path PATH] [--clips CLIP [CLIP ...]]
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULT_DATASET = Path(
    r"C:\Users\ASUS\Documents\Uni\Senior 2\cctv-accident-ITS-system\ACCIDENT\dataset"
)
DEFAULT_CLIPS = ["Town05_sideswipe_rain_44", "Town03_t-bone_rain_04", "Town10HD_t-bone_clear_37"]
OUT_DIR = Path(__file__).resolve().parent / "synthetic_check"
SIZES = [224, 336, 448]
SMALL_PX = 16
OFFSETS = [-10, -5, -2, 0, 2, 5, 10]
MIN_CROP = 256
FALLBACK_W, FALLBACK_H = 1920, 1080
REQUIRED = ["path", "map", "type", "accident_frame", "accident_time", "x1", "y1", "x2", "y2"]
PATH_ALIASES = ["path", "rgb_path"]  # the synthetic metadata names the video column rgb_path


class Report:
    """Print lines and keep them for saving."""

    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(str(text))


def stats_line(name, s):
    q = np.percentile(s, [0, 10, 50, 90, 100])
    return f"  {name:<22} min {q[0]:8.1f}  p10 {q[1]:8.1f}  median {q[2]:8.1f}  p90 {q[3]:8.1f}  max {q[4]:8.1f}"


# --------------------------------------------------------------------------- #
# STEP 0
# --------------------------------------------------------------------------- #
def step0(df):
    print("STEP 0 - checks")
    print(f"  columns ({len(df.columns)}): {list(df.columns)}")
    path_col = next((c for c in PATH_ALIASES if c in df.columns), None)
    if path_col is None:
        print(f"  available columns: {list(df.columns)}")
        sys.exit("ERROR: required column 'path' (or 'rgb_path') is missing")
    if path_col != "path":
        print(f"  note: no 'path' column; using '{path_col}' as the video path column")
        df = df.rename(columns={path_col: "path"})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        print(f"  available columns: {list(df.columns)}")
        sys.exit(f"ERROR: required columns missing: {missing}")
    if "width" not in df.columns or "height" not in df.columns:
        print(f"  note: width/height columns missing; FALLBACK {FALLBACK_W} x {FALLBACK_H} used for all clips")
        df["width"], df["height"] = FALLBACK_W, FALLBACK_H
    else:
        sizes = df.groupby(["width", "height"]).size()
        print(f"  frame sizes (width x height: clips): "
              + ", ".join(f"{w} x {h}: {n}" for (w, h), n in sizes.items()))
    for c in ["x1", "y1", "x2", "y2"]:
        print(f"  {c}: min {df[c].min():.4f}  max {df[c].max():.4f}")
    coords = df[["x1", "y1", "x2", "y2"]]
    in_unit = bool(((coords >= 0) & (coords <= 1)).all().all())
    print(f"  all box coordinates in [0, 1]: {in_unit}")
    bad = df[(df["x2"] <= df["x1"]) | (df["y2"] <= df["y1"])]
    print(f"  boxes with x2 <= x1 or y2 <= y1: {len(bad)}")
    df["scenario_id"] = df["map"].astype(str) + "_" + df["type"].astype(str)
    df["clip"] = df["path"].map(lambda p: Path(str(p).replace("\\", "/")).stem)
    return df


# --------------------------------------------------------------------------- #
# PART A
# --------------------------------------------------------------------------- #
def part_a(df, out_txt):
    r = Report()
    r("PART A - crash box size (metadata only, all clips)")
    r(f"  clips: {len(df)}")
    w, h = df["width"].astype(float), df["height"].astype(float)
    df["box_w"] = (df["x2"] - df["x1"]) * w
    df["box_h"] = (df["y2"] - df["y1"]) * h
    df["box_long"] = np.maximum(df["box_w"], df["box_h"])
    df["cx"] = (df["x1"] + df["x2"]) / 2
    short, long_ = np.minimum(w, h), np.maximum(w, h)

    r("")
    r("1. Box size at full resolution (pixels)")
    for c in ["box_w", "box_h", "box_long"]:
        r(stats_line(c, df[c]))

    r("")
    r(f"2. Box long side after preprocessing (pixels); share < {SMALL_PX} px")
    for s in SIZES:
        short_side = df["box_long"] * s / short
        squash = np.maximum(df["box_w"] * s / w, df["box_h"] * s / h)
        r(f"  S = {s}")
        r(stats_line("a) short-side resize", short_side)
          + f"  <{SMALL_PX}px {100 * (short_side < SMALL_PX).mean():5.1f}%")
        r(stats_line("b) squash SxS", squash)
          + f"  <{SMALL_PX}px {100 * (squash < SMALL_PX).mean():5.1f}%")
        df[f"long_short_{s}"] = short_side

    r("")
    r("3. Centre-crop coverage (short-side resize, then central square crop)")
    keep_lo = (1 - short / long_) / 2
    keep_hi = 1 - keep_lo
    if (w < h).any():
        r(f"  note: {int((w < h).sum())} portrait clips; their crop is vertical but only cx is tested")
    r(f"  kept x-range: [{keep_lo.min():.5f}, {keep_hi.max():.5f}]"
      + ("" if keep_lo.nunique() == 1 else " (varies by clip)"))
    df["centre_out"] = (df["cx"] < keep_lo) | (df["cx"] > keep_hi)
    df["box_cut"] = (df["x1"] < keep_lo) | (df["x2"] > keep_hi)
    r(f"  box centre outside the crop: {int(df['centre_out'].sum())} / {len(df)} "
      f"({100 * df['centre_out'].mean():.1f}%)")
    r(f"  box partly cut by the crop : {int(df['box_cut'].sum())} / {len(df)} "
      f"({100 * df['box_cut'].mean():.1f}%)")

    r("")
    r("4. Per scenario_id (S = 224, short-side resize)")
    tab = df.groupby("scenario_id").agg(
        n_clips=("clip", "size"),
        median_long_224=("long_short_224", "median"),
        pct_lt16_224=("long_short_224", lambda x: 100 * (x < SMALL_PX).mean()),
        pct_centre_out=("centre_out", lambda x: 100 * x.mean()),
    )
    r(tab.to_string(float_format=lambda x: f"{x:.1f}"))

    out_txt.write_text("\n".join(r.lines) + "\n", encoding="ascii")
    print(f"\nsaved {out_txt}")
    return df


# --------------------------------------------------------------------------- #
# PART B
# --------------------------------------------------------------------------- #
def read_sequential(video, wanted):
    """Decode from frame 0 without seeking; return ({index: BGR frame}, frame_count_read, fps)."""
    cap = cv2.VideoCapture(str(video))
    frames, idx = {}, 0
    try:
        if not cap.isOpened():
            return None, 0, None
        fps = cap.get(cv2.CAP_PROP_FPS)
        last = max(wanted)
        while idx <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in wanted:
                frames[idx] = frame
            idx += 1
    finally:
        cap.release()
    return frames, idx, fps


def zoom_crop(frame, row):
    """Square crop centred on the box (clipped to the image), x2 nearest upscale, box drawn."""
    H, W = frame.shape[:2]
    bx1, by1, bx2, by2 = row["x1"] * W, row["y1"] * H, row["x2"] * W, row["y2"] * H
    side = max(MIN_CROP, 4 * max(bx2 - bx1, by2 - by1))
    cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
    x0, x1 = int(max(0, round(cx - side / 2))), int(min(W, round(cx + side / 2)))
    y0, y1 = int(max(0, round(cy - side / 2))), int(min(H, round(cy + side / 2)))
    crop = cv2.resize(frame[y0:y1, x0:x1], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    p1 = (int(round((bx1 - x0) * 2)), int(round((by1 - y0) * 2)))
    p2 = (int(round((bx2 - x0) * 2)), int(round((by2 - y0) * 2)))
    cv2.rectangle(crop, p1, p2, (0, 0, 255), 1)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (x0, y0, x1, y1)


def part_b(df, dataset_root, clips):
    print("\nPART B - zoomed crops around the crash box")
    for clip in clips:
        print(f"\n== {clip}")
        m = df[df["clip"] == clip]
        if m.empty:
            print("  not in metadata-synthetic.csv, skipped")
            continue
        row = m.iloc[0]
        video = dataset_root / str(row["path"]).replace("\\", "/")
        if not video.is_file():
            print(f"  video missing: {video}")
            continue
        af = int(row["accident_frame"])
        targets = {k: af + k for k in OFFSETS}
        frames, n_read, fps = read_sequential(video, {f for f in targets.values() if f >= 0})
        if frames is None:
            print(f"  cannot open video: {video}")
            continue
        if not fps or fps <= 0:
            print("  file fps unavailable; times below use no fps")
        fps_txt = f"{fps:.3f}" if fps else "n/a"
        rt = int(round(row["accident_time"] * fps)) if fps else None
        print(f"  accident_frame {af}, accident_time {row['accident_time']:.3f}s, file fps {fps_txt}, "
              f"round(accident_time*fps) = {rt} -> {'agree' if rt == af else 'DIFFER'}")
        print(f"  box (px): w {row['box_w']:.1f}, h {row['box_h']:.1f}, long {row['box_long']:.1f}")
        kept = [(k, f) for k, f in targets.items() if f in frames]
        skipped = [k for k, f in targets.items() if f not in frames]
        print(f"  frames decoded sequentially: {n_read}")
        print(f"  kept offsets: {[k for k, _ in kept]}; skipped (outside video): {skipped if skipped else 'none'}")
        if not kept:
            continue

        fig, axes = plt.subplots(1, len(kept), figsize=(3.2 * len(kept), 3.8), squeeze=False)
        for ax, (k, f) in zip(axes[0], kept):
            img, box = zoom_crop(frames[f], row)
            ax.imshow(img)
            t = f"{f / fps:.2f}" if fps else "n/a"
            ax.set_title(f"f={f} | t={t}s | offset={k}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        print(f"  crop region (x0, y0, x1, y1) px: {box}")
        fig.suptitle(f"{clip}  (accident_frame {af}, red box = metadata box, x2 nearest upscale)", fontsize=11)
        fig.tight_layout()
        out_png = OUT_DIR / f"zoom_{clip}.png"
        fig.savefig(out_png, dpi=100)
        plt.close(fig)
        print(f"  saved {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--clips", nargs="+", default=DEFAULT_CLIPS)
    args = ap.parse_args()

    dataset_root = args.dataset_path.resolve()
    if (dataset_root / "dataset" / "metadata-synthetic.csv").is_file():
        dataset_root = dataset_root / "dataset"
    meta_csv = dataset_root / "metadata-synthetic.csv"
    if not meta_csv.is_file():
        sys.exit(f"ERROR: metadata-synthetic.csv not found in {dataset_root}")
    OUT_DIR.mkdir(exist_ok=True)
    print(f"metadata: {meta_csv}")

    df = step0(pd.read_csv(meta_csv))
    print()
    df = part_a(df, OUT_DIR / "crash_size_summary.txt")
    part_b(df, dataset_root, args.clips)
    print("\nDone.")


if __name__ == "__main__":
    main()
