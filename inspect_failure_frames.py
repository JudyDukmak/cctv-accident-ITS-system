"""Frame sheets and weather tables for the fold-0 diagnostics of a run (SYNTHETIC ONLY: reads the frame
cache, synthetic videos, metadata-synthetic.csv and the diagnostics CSVs; never opens real videos,
metadata-real.csv or real_test_set.csv). Read-only on data and runs; no training.

1. One PNG per clip (the 6 largest-error clips at the last epoch and the 2 Town04_rear-end clips in clear
   weather with the smallest error), 3 rows x 4 frames:
     row 1  cached frames at t* - 1, t* - 0.5, t*, t* + 0.5 s
     row 2  cached frames at t_hat - 1, t_hat - 0.5, t_hat, t_hat + 0.5 s (t_hat of the last epoch)
     row 3  full-resolution crops of the original video around the metadata crash box, at the row-1 times
   Cached frames are on the 8 fps grid, so each panel shows the nearest grid frame and its time. Titles give
   the raw score of the window whose centre is nearest that time, at the best and the last epoch.
2. Town04_rear-end: within_2s, median error and T@1 by weather. 3. All validation clips: the same by weather.
Outputs: results/diagnostics/<run>_fold<k>/frames/ (PNGs and tables.txt).

Usage:  .venv-ml\\Scripts\\python.exe inspect_failure_frames.py [--run std_seed0] [--fold 0]
"""

import argparse
import sys
from pathlib import Path

import av
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from accs_scorer import temporal  # noqa: E402
from build_frame_cache import read_cached_frame  # noqa: E402

DATASET = REPO_ROOT / "ACCIDENT" / "dataset"
METADATA_SYN = DATASET / "metadata-synthetic.csv"
CACHE_DIR = REPO_ROOT / "cache" / "frames224_v1"
DIAG_ROOT = REPO_ROOT / "results" / "diagnostics"
MODEL_FPS = 8
OFFSETS = (-1.0, -0.5, 0.0, 0.5)
N_WORST, N_GOOD = 6, 2
CROP_SCALE, CROP_MIN = 3.0, 360  # crop = 3x the box, at least 360 px per side


def fail(msg: str) -> None:
    sys.exit(f"ERROR: {msg}")


def cached_frame(stem: str, t: float) -> tuple[np.ndarray, float]:
    """Nearest 8 fps grid frame to time t (clamped to the cache), and its time."""
    with np.load(CACHE_DIR / f"{stem}.npz") as z:
        item = {"offsets": z["offsets"], "data": z["data"]}
    k = int(np.clip(np.floor(t * MODEL_FPS + 0.5), 0, len(item["offsets"]) - 2))
    return read_cached_frame(item, k), k / MODEL_FPS


def video_frames(path: Path, times: list[float], fps: float) -> list[tuple[np.ndarray, int]]:
    """Full-resolution RGB frames at index floor(t * fps + 0.5), decoded sequentially (no seeking)."""
    want = {t: int(np.floor(t * fps + 0.5)) for t in times}
    got: dict[int, np.ndarray] = {}
    with av.open(str(path)) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i in want.values():
                got[i] = fr.to_ndarray(format="rgb24")
            if i >= max(want.values()):
                break
    return [(got[want[t]], want[t]) for t in times]


def score_at(win: pd.DataFrame, t: float) -> float:
    return float(win.iloc[(win["t_centre"] - t).abs().argmin()]["score"])


def crop_box(img: np.ndarray, box: tuple[float, float, float, float]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop around the normalised box; returns the crop and the box in crop pixels."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side_w = min(w, max(CROP_MIN, CROP_SCALE * (x2 - x1)))
    side_h = min(h, max(CROP_MIN, CROP_SCALE * (y2 - y1)))
    left = int(np.clip(cx - side_w / 2, 0, w - side_w))
    top = int(np.clip(cy - side_h / 2, 0, h - side_h))
    crop = img[top:top + int(side_h), left:left + int(side_w)]
    return crop, (int(x1 - left), int(y1 - top), int(x2 - left), int(y2 - top))


def sheet(out: Path, clip: pd.Series, win_b: pd.DataFrame, win_l: pd.DataFrame, lab_b: str, lab_l: str) -> None:
    stem = clip["stem"]
    t_star, t_hat = float(clip["t_star"]), float(clip["t_hat_l"])
    fig, axes = plt.subplots(3, 4, figsize=(16, 11.5), dpi=100)
    rows = [("around t*", t_star), (f"around t_hat ({lab_l})", t_hat)]
    for r, (name, centre) in enumerate(rows):
        for c, off in enumerate(OFFSETS):
            img, tg = cached_frame(stem, centre + off)
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_title(f"{name} {off:+.1f} s: grid t = {tg:.3f} s\nscore {lab_b} {score_at(win_b, tg):.3f}, "
                         f"{lab_l} {score_at(win_l, tg):.3f}", fontsize=8)
            ax.axis("off")
    times = [t_star + off for off in OFFSETS]
    frames = video_frames(DATASET / clip["rgb_path"], times, clip["no_frames"] / clip["duration"])
    box = (clip["x1"], clip["y1"], clip["x2"], clip["y2"])
    for c, ((img, idx), t) in enumerate(zip(frames, times)):
        crop, (bx1, by1, bx2, by2) = crop_box(img, box)
        ax = axes[2, c]
        ax.imshow(crop)
        ax.add_patch(plt.Rectangle((bx1, by1), bx2 - bx1, by2 - by1, fill=False, ec="red", lw=1.5))
        ax.set_title(f"video t* {OFFSETS[c]:+.1f} s: frame {idx} (t = {idx * clip['duration'] / clip['no_frames']:.3f} s)"
                     f"\nscore {lab_b} {score_at(win_b, t):.3f}, {lab_l} {score_at(win_l, t):.3f}; "
                     f"crop {crop.shape[1]}x{crop.shape[0]} px", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"{stem} ({clip['weather']}): t* {t_star:.2f} s; t_hat {lab_b} {clip['t_hat_b']:.2f} s "
                 f"(err {clip['t_hat_b'] - t_star:+.2f}), {lab_l} {t_hat:.2f} s (err {t_hat - t_star:+.2f}); "
                 f"rows: cached 224 frames around t*, around t_hat; video crops around the crash box at t* "
                 f"(box from metadata, drawn at all 4 times)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out)
    plt.close(fig)


def weather_table(df: pd.DataFrame, labels: tuple[str, str]) -> str:
    rows = []
    for w, g in df.groupby("weather", sort=True):
        row = {"weather": w, "n": len(g)}
        for lab, col in zip(labels, ("t_hat_b", "t_hat_l")):
            e = g[col] - g["t_star"]
            row |= {f"within_2s {lab}": (e.abs() <= 2).mean(), f"median_err {lab}": e.median(),
                    f"T@1 {lab}": temporal(g[col], g["t_star"], 1.0)}
        rows.append(row)
    return pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="std_seed0")
    ap.add_argument("--fold", type=int, default=0, choices=[0, 1, 2])
    args = ap.parse_args()
    run_dir = REPO_ROOT / "runs" / args.run / f"fold_{args.fold}"
    diag = DIAG_ROOT / f"{args.run}_fold{args.fold}"
    out_dir = diag / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(run_dir / "metrics.csv")
    best_ep = int(metrics[metrics["best"].astype(str) == "True"]["epoch"].iloc[-1])
    last_ep = int(metrics["epoch"].iloc[-1])
    lab_b, lab_l = f"ep{best_ep}", f"ep{last_ep}"
    pred_b = pd.read_csv(run_dir / "val_clip_predictions_best.csv")
    pred_l = pd.read_csv(diag / f"val_clip_predictions_epoch{last_ep}.csv")
    win_b = pd.read_csv(run_dir / "val_window_scores_best.csv")
    win_l = pd.read_csv(diag / f"val_window_scores_epoch{last_ep}.csv")
    split = pd.read_csv(REPO_ROOT / "synthetic_split.csv", usecols=["rgb_path", "scenario_id", "val_fold"])
    meta = pd.read_csv(METADATA_SYN, usecols=["rgb_path", "weather", "x1", "y1", "x2", "y2", "no_frames", "duration",
                                              "accident_time"])

    clips = (pred_b[["rgb_path", "t_star", "t_hat"]].rename(columns={"t_hat": "t_hat_b"})
             .merge(pred_l[["rgb_path", "t_hat"]].rename(columns={"t_hat": "t_hat_l"}), on="rgb_path", validate="1:1")
             .merge(split, on="rgb_path", validate="1:1").merge(meta, on="rgb_path", validate="1:1"))
    if len(clips) != len(pred_b) or (clips["val_fold"] != args.fold).any():
        fail("clip predictions do not match the fold's validation clips")
    if not np.allclose(clips["t_star"], clips["accident_time"]):
        fail("t_star differs from metadata accident_time")
    clips["stem"] = clips["rgb_path"].map(lambda p: Path(p).stem)
    clips["abs_err_l"] = (clips["t_hat_l"] - clips["t_star"]).abs()

    srt = clips.sort_values(["abs_err_l", "rgb_path"])
    worst = srt.tail(N_WORST).iloc[::-1]
    good = srt[(srt["scenario_id"] == "Town04_rear-end") & (srt["weather"] == "clear")].head(N_GOOD)

    files = []
    for kind, sel in (("worst", worst), ("good_rear-end_clear", good)):
        for _, clip in sel.iterrows():
            out = out_dir / f"{kind}_{clip['stem']}.png"
            sheet(out, clip, win_b[win_b["rgb_path"] == clip["rgb_path"]],
                  win_l[win_l["rgb_path"] == clip["rgb_path"]], lab_b, lab_l)
            files.append(f"{out.name}  (t* {clip['t_star']:.2f}, err {lab_b} {clip['t_hat_b'] - clip['t_star']:+.2f}, "
                         f"{lab_l} {clip['t_hat_l'] - clip['t_star']:+.2f}, weather {clip['weather']})")

    labels = (f"{lab_b} (best)", f"{lab_l} (last)")
    rep = [f"run {args.run}, fold {args.fold}; best epoch {best_ep}, last epoch {last_ep}; synthetic validation only", "",
           "2. Town04_rear-end by weather",
           weather_table(clips[clips["scenario_id"] == "Town04_rear-end"], labels), "",
           f"3. all {len(clips)} validation clips by weather", weather_table(clips, labels), "",
           "frame sheets (worst = largest |error| at the last epoch; good = Town04_rear-end, clear, smallest |error| "
           "at the last epoch):"] + [f"  {f}" for f in files]
    text = "\n".join(rep) + "\n"
    (out_dir / "tables.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
