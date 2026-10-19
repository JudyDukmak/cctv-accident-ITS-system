"""Visual check of synthetic windows: 4x4 montages of the 16 sampled frames.

READ-ONLY. Reads synthetic_windows.csv, metadata-synthetic.csv and only synthetic
videos. Never touches real data. No randomness.

For each clip, four windows are shown: the last label-0 window, the first and last
onset windows, and the first post window. Frame indices come from
make_windows.window_frame_indices (imported, not re-implemented) and are checked
against frame_first / frame_last in the CSV. The sampled frame nearest to
accident_frame gets a red border.

Usage:
    py preview_windows.py [--dataset-path PATH] [--clips CLIP [CLIP ...]]
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from make_windows import window_frame_indices  # noqa: E402

DEFAULT_DATASET = Path(
    r"C:\Users\ASUS\Documents\Uni\Senior 2\cctv-accident-ITS-system\ACCIDENT\dataset"
)
DEFAULT_CLIPS = ["Town05_sideswipe_rain_44", "Town03_t-bone_rain_04"]
WINDOWS_CSV = REPO_ROOT / "synthetic_windows.csv"
CONFIG_JSON = REPO_ROOT / "windows_config.json"
OUT_DIR = REPO_ROOT / "synthetic_check"
TILE_INCH = 4.8  # 4.8 in at 100 dpi = 480 px per tile
DPI = 100


def pick_windows(w):
    """Last pre, first onset, last onset, first post (deduplicated, in window order)."""
    picks = []
    for phase, which in (("pre", -1), ("onset", 0), ("onset", -1), ("post", 0)):
        sub = w[w["phase"] == phase]
        if sub.empty:
            print(f"  no '{phase}' window")
            continue
        picks.append(sub.iloc[which])
    seen, out = set(), []
    for r in picks:
        if r["window_idx"] not in seen:
            seen.add(r["window_idx"])
            out.append(r)
    return out


def read_frames(video_file, indices):
    """Read the requested frame indices by decoding sequentially from the first one."""
    cap = cv2.VideoCapture(str(video_file))
    frames = {}
    try:
        if not cap.isOpened():
            return frames, "cannot open video"
        first, last = min(indices), max(indices)
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        wanted = set(indices)
        for idx in range(first, last + 1):
            ok, frame = cap.read()
            if not ok:
                return frames, f"read failed at frame {idx}"
            if idx in wanted:
                frames[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
    return frames, None


def save_montage(clip, row, indices, frames, fps, accident_frame, out_png):
    in_window = indices[0] <= accident_frame <= indices[-1]
    red_i = min(range(len(indices)), key=lambda i: abs(indices[i] - accident_frame)) if in_window else None

    fig, axes = plt.subplots(4, 4, figsize=(4 * TILE_INCH, 4 * TILE_INCH * 9 / 16 + 1.2))
    for i, ax in enumerate(axes.flat):
        idx = indices[i]
        if idx in frames:
            ax.imshow(frames[idx])
        else:
            ax.text(0.5, 0.5, "unreadable", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"i={i}  frame {idx}  t={idx / fps:.2f}s", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == red_i:
            for s in ax.spines.values():
                s.set_edgecolor("red")
                s.set_linewidth(5)
    fig.suptitle(
        f"{clip}  window {row['window_idx']}  label {row['label']}  phase {row['phase']}  "
        f"(t {row['t_start']:.3f}-{row['t_end']:.3f}s, accident_frame {accident_frame}, "
        f"t* {row['t_star']:.2f}s)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)
    return red_i


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
        sys.exit(f"metadata-synthetic.csv not found in {dataset_root}")
    if not WINDOWS_CSV.is_file():
        sys.exit(f"{WINDOWS_CSV} not found - run make_windows.py first")
    OUT_DIR.mkdir(exist_ok=True)

    import json
    cfg = json.loads(CONFIG_JSON.read_text()) if CONFIG_JSON.is_file() else {}
    model_fps = cfg.get("model_fps", 8)
    window_frames = cfg.get("window_frames", 16)
    print(f"model_fps {model_fps}, window_frames {window_frames} "
          f"({'from ' + CONFIG_JSON.name if cfg else 'defaults, config not found'})")

    win = pd.read_csv(WINDOWS_CSV)
    meta = pd.read_csv(meta_csv, usecols=["rgb_path", "accident_time", "accident_frame", "no_frames", "duration"])
    meta["clip"] = meta["rgb_path"].map(lambda p: Path(p.replace("\\", "/")).stem)
    meta["fps_video"] = meta["no_frames"] / meta["duration"]

    n_mismatch = 0
    for clip in args.clips:
        print(f"\n== {clip}")
        m = meta[meta["clip"] == clip]
        if m.empty:
            print("  not in metadata-synthetic.csv, skipped")
            continue
        m = m.iloc[0]
        w = win[win["rgb_path"] == m["rgb_path"]].sort_values("window_idx")
        if w.empty:
            print("  no windows in synthetic_windows.csv, skipped")
            continue
        video = dataset_root / m["rgb_path"].replace("\\", "/")
        if not video.is_file():
            print(f"  video missing: {video}")
            continue
        fps = m["fps_video"]
        print(f"  fps_video {fps:.3f}, accident_frame {m['accident_frame']}, accident_time {m['accident_time']:.2f}s")

        for row in pick_windows(w):
            j = int(row["window_idx"])
            indices = window_frame_indices(row["t_start"], fps, model_fps, window_frames)
            if indices[0] != row["frame_first"] or indices[-1] != row["frame_last"]:
                n_mismatch += 1
                print(f"  MISMATCH window {j}: computed {indices[0]}..{indices[-1]}, "
                      f"CSV {row['frame_first']}..{row['frame_last']}")
            frames, err = read_frames(video, indices)
            if err:
                print(f"  window {j}: {err}")
            out_png = OUT_DIR / f"preview_{clip}_j{j}.png"
            red_i = save_montage(clip, row, indices, frames, fps, int(m["accident_frame"]), out_png)
            where = f"position i={red_i} (frame {indices[red_i]})" if red_i is not None else "not in window"
            print(f"  window {j:>2} label {row['label']:>2} {row['phase']:<5} frames {indices[0]}..{indices[-1]}"
                  f"  accident_frame: {where}  -> {out_png.name}")
    print(f"\nframe index mismatches vs CSV: {n_mismatch}")

    clips = win[["rgb_path", "scenario_id"]].drop_duplicates().merge(meta, on="rgb_path", how="left")
    tab = clips.groupby("scenario_id")["accident_time"].agg(n_clips="size", median_accident_time_s="median")
    print("\nMedian accident_time per scenario_id:")
    print(tab.to_string(float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
