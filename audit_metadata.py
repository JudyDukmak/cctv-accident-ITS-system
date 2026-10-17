"""
audit_metadata.py - Week 3 dataset audit for "From CCTV to Grounded Incident Reports".

What it answers (mentor's request, 14 Oct):
  1. Class counts (collision types) per split
  2. fps and clip durations
  3. Pre-impact footage per clip (frames and seconds before the accident)
  4. How many clips have very few pre-impact frames (< 8, the mentor's threshold)
  5. How much "normal" (pre-impact) footage exists in total, i.e. the
     denominator for false alarms per hour

It works on metadata-synthetic.csv, metadata-real.csv and (optionally) your
frozen real_test_set.csv. It never modifies any input file.

Usage (Windows, from the folder that contains this script):
  python audit_metadata.py --only synthetic          # uses the default dataset folder below
  python audit_metadata.py --dataset-path "C:\\path\\to\\Dataset\\archive"
  python audit_metadata.py --dataset-path "...\\archive" --test-set "...\\real_test_set.csv"
  python audit_metadata.py --dataset-path "...\\archive" --probe-videos   # read fps/frames from the video files (slower)
  python audit_metadata.py --dataset-path "...\\archive" --model-fps 4    # count pre-impact frames at 4 fps

Outputs (in ./audit_output/):
  audit_summary.md          - tables to paste into the Week 3 summary
  per_clip_synthetic.csv    - one row per clip with fps, pre/post frames and seconds
  per_clip_real.csv
  pre_impact_hist.png       - histogram of pre-impact seconds, synthetic vs real

Requirements: pandas, numpy, matplotlib; opencv-python only with --probe-videos.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW = 16          # classifier window length in frames (proposal 7.1)
MODEL_FPS = 8.0      # frame rate the classifier samples at (overridden by --model-fps)
FEW_FRAMES = 8       # mentor's threshold
FRAME_BINS = [0, 8, 16, 32, 64, 10**9]
FRAME_LABELS = ["<8", "8-15", "16-31", "32-63", ">=64"]


def probe_video(path: Path):
    """Return (fps, frame_count) from the file, or (nan, nan) if unreadable."""
    import cv2  # imported only when needed
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return np.nan, np.nan
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return (fps if fps > 0 else np.nan), (n if n > 0 else np.nan)


def build_per_clip(df: pd.DataFrame, videos_dir: Path, probe: bool) -> pd.DataFrame:
    out = pd.DataFrame()
    df = df.reset_index(drop=True)
    out["path"] = df["path"] if "path" in df.columns else df.index.astype(str)
    out["type"] = df["type"] if "type" in df.columns else "unknown"
    for extra in ["split", "map", "weather", "camera_position"]:
        if extra in df.columns:
            out[extra] = df[extra]

    def col(name):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").reset_index(drop=True)
        return pd.Series(np.nan, index=range(len(df)))

    t = col("accident_time")
    f = col("accident_frame")
    dur = col("duration")

    nf = col("no_frames")

    # fps: metadata column > no_frames / duration (repo convention) > accident_frame / accident_time
    if "fps" in df.columns:
        fps = col("fps")
        fps_source = "fps column"
    elif nf.notna().any() and dur.notna().any():
        fps = nf / dur.replace(0, np.nan)
        fps_source = "no_frames / duration"
    else:
        fps = f / t.replace(0, np.nan)
        fps_source = "accident_frame / accident_time"

    n_frames = nf if nf.notna().any() else dur * fps
    if f.isna().all():                     # real CSV may have no accident_frame
        f = (t * fps).round()
    if probe:
        probed = [probe_video(videos_dir / p) for p in out["path"]]
        pf = pd.Series([p[0] for p in probed])
        pn = pd.Series([p[1] for p in probed])
        fps = pf.fillna(fps)
        n_frames = pn.fillna(n_frames)
        fps_source = "video files (fallback: " + fps_source + ")"
        if dur.isna().all():
            dur = n_frames / fps

    out["fps"] = fps.round(3)
    out["duration_s"] = dur.round(3)
    out["n_frames"] = n_frames.round()
    out["accident_time_s"] = t
    out["pre_frames"] = f                     # frames before the impact frame
    out["pre_s"] = t
    out["post_s"] = (dur - t).round(3)
    out["post_frames"] = (n_frames - f).round()
    out["has_accident_label"] = t.notna() & (t >= 0)
    out["pre_bin"] = pd.cut(out["pre_frames"], FRAME_BINS, right=False, labels=FRAME_LABELS)
    # how many full non-overlapping 16-frame "normal" windows fit before impact
    out["normal_windows_16"] = np.floor(out["pre_frames"] / WINDOW)
    # pre-impact frames as the MODEL sees them after resampling to MODEL_FPS
    out["pre_frames_model"] = np.floor(t * MODEL_FPS)
    out.attrs["fps_source"] = fps_source
    return out


def describe(series: pd.Series) -> str:
    s = series.dropna()
    if s.empty:
        return "n/a"
    return (f"min {s.min():.2f} | median {s.median():.2f} | mean {s.mean():.2f} | "
            f"max {s.max():.2f}")


def section(name: str, pc: pd.DataFrame) -> list[str]:
    n = len(pc)
    lines = [f"## {name} ({n} clips)", ""]
    lines.append(f"- fps source: {pc.attrs.get('fps_source', '?')}")
    lines.append(f"- Clips with an accident label: {int(pc['has_accident_label'].sum())} / {n}"
                 f" -> clips with NO accident: {n - int(pc['has_accident_label'].sum())}")
    lines += ["", "**Collision types**", "", "| type | clips | share |", "|---|---|---|"]
    for k, v in pc["type"].value_counts().items():
        lines.append(f"| {k} | {v} | {v / n:.1%} |")
    if "split" in pc.columns:
        lines += ["", "**Splits**", "", "| split | clips |", "|---|---|"]
        for k, v in pc["split"].value_counts().items():
            lines.append(f"| {k} | {v} |")

    for extra in ["map", "weather", "camera_position"]:
        if extra in pc.columns:
            lines += ["", f"**{extra}**", "", f"| {extra} | clips |", "|---|---|"]
            for k, v in pc[extra].value_counts().head(15).items():
                lines.append(f"| {k} | {v} |")

    fps_counts = pc["fps"].round().value_counts().head(8)
    lines += ["", "**fps (rounded)**", "", "| fps | clips |", "|---|---|"]
    for k, v in fps_counts.items():
        lines.append(f"| {k:g} | {v} |")

    lines += ["", "**Timing**", "",
              f"- Duration (s): {describe(pc['duration_s'])}",
              f"- Pre-impact (s): {describe(pc['pre_s'])}",
              f"- Pre-impact (frames): {describe(pc['pre_frames'])}",
              f"- Post-impact (s): {describe(pc['post_s'])}"]

    few = int((pc["pre_frames"] < FEW_FRAMES).sum())
    lt_win = int((pc["pre_frames"] < WINDOW).sum())
    lines += ["", "**Pre-impact frames per clip**", "", "| bin | clips | share |", "|---|---|---|"]
    for k, v in pc["pre_bin"].value_counts().reindex(FRAME_LABELS).fillna(0).items():
        lines.append(f"| {k} | {int(v)} | {v / n:.1%} |")
    lines += ["",
              f"- Clips with fewer than {FEW_FRAMES} pre-impact frames: **{few} ({few / n:.1%})**",
              f"- Clips that cannot fit one full {WINDOW}-frame window before impact: "
              f"**{lt_win} ({lt_win / n:.1%})**"]

    fm = int((pc["pre_frames_model"] < FEW_FRAMES).sum())
    wm = int((pc["pre_frames_model"] < WINDOW).sum())
    lines += [f"- At the model's {MODEL_FPS:g} fps: fewer than {FEW_FRAMES} pre-impact frames: "
              f"**{fm} ({fm / n:.1%})**; fewer than {WINDOW}: **{wm} ({wm / n:.1%})**"]
    for name_t, thr in [("<0.5 s", 0.5), ("<1 s", 1.0), ("<2 s", 2.0)]:
        k = int((pc["pre_s"] < thr).sum())
        lines.append(f"- Clips with pre-impact {name_t}: {k} ({k / n:.1%})")

    pre_hours = pc["pre_s"].sum() / 3600
    post_hours = pc["post_s"].sum() / 3600
    lines += ["",
              "**Footage totals (for false alarms per hour)**", "",
              f"- Total pre-impact (normal) footage: **{pre_hours:.2f} h** "
              f"({pc['pre_s'].sum() / 60:.1f} min)",
              f"- Total post-impact footage: {post_hours:.2f} h",
              f"- Full non-overlapping {WINDOW}-frame normal windows: "
              f"{int(pc['normal_windows_16'].sum())}",
              ""]
    return lines


def save_hist(per_clip: dict, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, pc in per_clip.items():
        s = pc["pre_s"].dropna()
        if not s.empty:
            ax.hist(s.clip(upper=s.quantile(0.99)), bins=50, alpha=0.55, label=f"{name} (n={len(s)})")
    ax.set_xlabel("Seconds of footage before the accident")
    ax.set_ylabel("Clips")
    ax.set_title("Pre-impact footage per clip")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "pre_impact_hist.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", type=Path,
                    default=Path(r"C:\Users\ASUS\Documents\Uni\Senior 2\cctv-accident-ITS-system\ACCIDENT\dataset"),
                    help="folder with metadata-real.csv, metadata-synthetic.csv, real_videos/, synthetic_videos/")
    ap.add_argument("--test-set", type=Path, default=None,
                    help="your frozen real_test_set.csv (read-only; optional)")
    ap.add_argument("--probe-videos", action="store_true",
                    help="read fps and frame counts from the video files with OpenCV")
    ap.add_argument("--only", choices=["synthetic", "real", "both"], default="both",
                    help="which metadata file(s) to audit")
    ap.add_argument("--model-fps", type=float, default=8.0,
                    help="fps your classifier samples at (default 8); used for model-frame counts")
    ap.add_argument("--out", type=Path, default=Path("audit_output"))
    args = ap.parse_args()
    global MODEL_FPS
    MODEL_FPS = args.model_fps

    root = args.dataset_path.expanduser().resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    report = ["# Dataset audit — pre-impact footage and false-alarm budget", ""]
    per_clip = {}

    for name, csv, vids in [("Synthetic", "metadata-synthetic.csv", "synthetic_videos"),
                            ("Real", "metadata-real.csv", "real_videos")]:
        if args.only != "both" and name.lower() != args.only:
            continue
        p = root / csv
        if not p.is_file():
            report += [f"## {name}", "", f"Missing {p}", ""]
            continue
        df = pd.read_csv(p)
        report += [f"_Columns in {csv}: {', '.join(df.columns)}_", ""]
        pc = build_per_clip(df, root / vids, args.probe_videos)
        pc.to_csv(args.out / f"per_clip_{name.lower()}.csv", index=False)
        per_clip[name] = pc
        report += section(name, pc)

    if args.test_set and args.test_set.is_file() and "Real" in per_clip:
        ts = pd.read_csv(args.test_set)
        key = "path" if "path" in ts.columns else ts.columns[0]
        sub = per_clip["Real"][per_clip["Real"]["path"].isin(ts[key])].copy()
        sub.attrs = per_clip["Real"].attrs
        report += section(f"Frozen real test subset ({args.test_set.name})", sub)

    hist = save_hist(per_clip, args.out)
    if hist:
        report += [f"![pre-impact histogram]({hist.name})", ""]

    out_md = args.out / "audit_summary.md"
    out_md.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nSaved: {out_md.resolve()}")


if __name__ == "__main__":
    main()
