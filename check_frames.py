"""Visual check: does the label line up with the video?

For a few clips, saves an image with two frames side by side:
  left  = frame found by TIME  (accident_time, seconds)
  right = frame found by INDEX (accident_frame)
The bounding box from the CSV is drawn on both.
Also prints what the video file itself says (fps, frame count) next to the CSV.

Usage:
  python check_frames.py metadata-real.csv path/to/dataset_root --n 5 --out check_out
Picks the clips whose two fps estimates differ the most, plus 2 random control clips.
"""
import argparse, os, glob
import cv2
import numpy as np
import pandas as pd


def find_video(root, rel):
    for p in (os.path.join(root, rel),
              os.path.join(root, "real_videos", "videos", os.path.basename(rel)),
              os.path.join(root, "videos", os.path.basename(rel))):
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(root, "**", os.path.basename(rel)), recursive=True)
    return hits[0] if hits else None


def grab(cap, mode, value):
    cap.set(cv2.CAP_PROP_POS_MSEC if mode == "time" else cv2.CAP_PROP_POS_FRAMES, value)
    ok, frame = cap.read()
    return frame if ok else None


def draw(frame, row, label):
    if frame is None:
        frame = np.zeros((360, 640, 3), np.uint8)
        cv2.putText(frame, "could not read frame", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    h, w = frame.shape[:2]
    p1 = (int(row.x1 * w), int(row.y1 * h))
    p2 = (int(row.x2 * w), int(row.y2 * h))
    cv2.rectangle(frame, p1, p2, (0, 0, 255), max(2, w // 400))
    cv2.circle(frame, (int(row.center_x * w), int(row.center_y * h)), max(3, w // 300), (0, 255, 0), -1)
    cv2.putText(frame, label, (10, max(30, h // 15)), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.6, w / 1600), (255, 255, 0), max(1, w // 800))
    return cv2.resize(frame, (640, int(640 * h / w)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("root")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="check_out")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    df["fps_dur"] = df.no_frames / df.duration
    df["fps_acc"] = df.accident_frame / df.accident_time
    df["gap"] = (df.fps_dur - df.fps_acc).abs()
    worst = df.sort_values("gap", ascending=False).head(a.n)
    ctrl = df[df.gap < 0.1].sample(min(2, len(df)), random_state=a.seed)
    os.makedirs(a.out, exist_ok=True)

    for tag, part in (("WORST", worst), ("CONTROL", ctrl)):
        for _, r in part.iterrows():
            path = find_video(a.root, r.path)
            if path is None:
                print(f"[{tag}] {r.path}: video file not found"); continue
            cap = cv2.VideoCapture(path)
            vfps = cap.get(cv2.CAP_PROP_FPS)
            vframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"[{tag}] {os.path.basename(path)} gap={r.gap:.2f} | "
                  f"CSV: frames={r.no_frames} dur={r.duration:.2f}s acc_t={r.accident_time:.2f}s acc_f={r.accident_frame} | "
                  f"FILE: fps={vfps:.2f} frames={vframes} dur={vframes / vfps if vfps else 0:.2f}s")
            left = draw(grab(cap, "time", r.accident_time * 1000), r, f"by TIME {r.accident_time:.2f}s")
            right = draw(grab(cap, "frame", r.accident_frame), r, f"by INDEX {int(r.accident_frame)}")
            cap.release()
            hh = min(left.shape[0], right.shape[0])
            img = np.hstack([left[:hh], right[:hh]])
            cv2.imwrite(os.path.join(a.out, f"{tag}_{os.path.splitext(os.path.basename(path))[0]}.jpg"), img)
    print("images saved in", a.out)


if __name__ == "__main__":
    main()
