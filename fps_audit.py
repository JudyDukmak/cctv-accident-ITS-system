"""Audit: which CSV time fields agree with the real video files?

For every clip it reads ONLY the video header (fps, frame count) with OpenCV (fast, no decoding) and compares:
  A) accident_frame / accident_time   with the file's fps   (do the accident time and accident frame agree?)
  B) duration                          with the file's duration (frames / fps)
  C) no_frames                         with the file's frame count
Writes fps_audit.csv and prints a summary. Nothing is modified.

Usage:
  python fps_audit.py metadata-real.csv path/to/folder_with_videos
"""
import argparse, glob, os
import cv2
import pandas as pd

TOL = 0.05   # 5 % tolerance


def index_videos(root):
    return {f: os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs if f.lower().endswith(".mp4")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("root")
    ap.add_argument("--out", default="fps_audit.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    vids = index_videos(a.root)
    rows = []
    for r in df.itertuples():
        name = os.path.basename(r.path.replace("\\", "/"))
        p = vids.get(name)
        if p is None:
            rows.append(dict(path=r.path, found=False)); continue
        cap = cv2.VideoCapture(p)
        fps = cap.get(cv2.CAP_PROP_FPS); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        fdur = n / fps if fps else float("nan")
        rows.append(dict(
            path=r.path, found=True, file_fps=fps, file_frames=n, file_dur=fdur,
            csv_fps_from_acc=r.accident_frame / r.accident_time if r.accident_time else float("nan"),
            csv_fps_from_dur=r.no_frames / r.duration,
            acc_ratio=(r.accident_frame / r.accident_time) / fps if r.accident_time and fps else float("nan"),
            dur_ratio=r.duration / fdur if fdur else float("nan"),
            frames_diff=r.no_frames - n,
        ))
    au = pd.DataFrame(rows)
    ok = au[au.found]
    if len(ok) < len(au):
        print(f"WARNING: {len(au) - len(ok)} videos not found under {a.root}")

    acc_ok = (ok.acc_ratio - 1).abs() <= TOL
    dur_ok = (ok.dur_ratio - 1).abs() <= TOL
    ok = ok.assign(category=pd.Series("both agree", index=ok.index)
                   .mask(~acc_ok & dur_ok, "accident time/frame disagree with file fps")
                   .mask(acc_ok & ~dur_ok, "duration disagrees with file")
                   .mask(~acc_ok & ~dur_ok, "both disagree"))
    print(f"\nclips checked: {len(ok)}   (tolerance {TOL:.0%})")
    print(ok.category.value_counts().to_string())
    print("\nno_frames minus file frame count (value counts):")
    print(ok.frames_diff.value_counts().head(8).to_string())
    bad = ok[ok.category != "both agree"].copy()
    if len(bad):
        bad["file_fps_rounded"] = bad.file_fps.round()
        print("\nmismatching clips by the file's fps:")
        print(bad.groupby(["category", "file_fps_rounded"]).size().to_string())
        print("\nacc_ratio summary for clips where accident time/frame disagree "
              "(1.0 = agree; 0.4 = frame index is 0.4x what the time implies):")
        print(bad[bad.category.str.startswith("accident") | (bad.category == "both disagree")].acc_ratio.describe().round(2).to_string())
    ok.to_csv(a.out, index=False)
    print("\nsaved", a.out)


if __name__ == "__main__":
    main()