"""Cache the 8-fps frame grid of every synthetic clip as 224 x 224 JPEGs.

SYNTHETIC ONLY. Read-only inputs: metadata-synthetic.csv, synthetic_windows.csv,
split_manifest.json, windows_config.json. Never opens real videos or real metadata.

Per clip: fps_video = no_frames / duration (as in make_windows.py). Grid k = 0..K-1,
t_k = k / 8, frame index floor(t_k * fps_video + 0.5), K = 8 * max(t_start) + 16.
Every window's frames (make_windows.window_frame_indices) must equal
grid[8*t_start : 8*t_start + 16], so a window is a contiguous slice of the cache.

Each video is decoded SEQUENTIALLY with PyAV (no seeking); only grid frames are kept,
resized (full frame, no crop) to 224 x 224 with cv2.INTER_AREA, stored as RGB JPEG
quality 95 in cache/frames224_v1/<clip_stem>.npz:
    k (int32), frame_idx (int32), offsets (int64, K+1), data (uint8 JPEG bytes)
Files are written to a temporary name and renamed; existing files with the right grid
are skipped, so an interrupted run can be restarted.

Usage (training environment):
    .venv-ml\\Scripts\\python.exe build_frame_cache.py [--limit N] [--workers W] [--dataset-path PATH]
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from make_windows import window_frame_indices  # noqa: E402

DEFAULT_DATASET = Path(
    r"C:\Users\ASUS\Documents\Uni\Senior 2\cctv-accident-ITS-system\ACCIDENT\dataset"
)
WINDOWS_CSV = REPO_ROOT / "synthetic_windows.csv"
MANIFEST = REPO_ROOT / "split_manifest.json"
CONFIG = REPO_ROOT / "windows_config.json"
CACHE_DIR = REPO_ROOT / "cache" / "frames224_v1"
SIZE = 224
JPEG_QUALITY = 95
MODEL_FPS = 8
WINDOW_FRAMES = 16
VERIFY_CLIPS = 5
PROGRESS_EVERY = 50


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #
def grid_frame_indices(fps_video, K):
    """floor(t_k * fps_video + 0.5) for t_k = k / MODEL_FPS, k = 0..K-1 (written out independently)."""
    return np.array([math.floor((k / MODEL_FPS) * fps_video + 0.5) for k in range(K)], dtype=np.int64)


def build_grid(clip_windows, fps_video, no_frames, name):
    """Grid of one clip; checks every window is a contiguous slice and fits in the video."""
    t_starts = clip_windows["t_start"].to_numpy(float)
    k0s = np.round(t_starts * MODEL_FPS).astype(int)
    if not np.allclose(k0s / MODEL_FPS, t_starts):
        fail(f"{name}: t_start values are not multiples of 1/{MODEL_FPS} s")
    K = int(k0s.max()) + WINDOW_FRAMES
    grid = grid_frame_indices(fps_video, K)
    for t_start, k0 in zip(t_starts, k0s):
        w = window_frame_indices(t_start, fps_video, MODEL_FPS, WINDOW_FRAMES)
        if list(grid[k0:k0 + WINDOW_FRAMES]) != list(w):
            fail(f"{name}: window t_start={t_start} frames {w} != grid[{k0}:{k0 + WINDOW_FRAMES}] "
                 f"{grid[k0:k0 + WINDOW_FRAMES].tolist()}")
    if grid.max() > no_frames - 1:
        fail(f"{name}: largest grid frame index {grid.max()} > no_frames - 1 = {no_frames - 1}")
    return K, grid


# --------------------------------------------------------------------------- #
# Decoding and cache files
# --------------------------------------------------------------------------- #
def resize_rgb(rgb):
    return cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def encode_jpeg(rgb224):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb224, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.reshape(-1)


def read_cached_frame(npz, i):
    """RGB uint8 (224, 224, 3) frame at grid position i of a loaded cache .npz."""
    off = npz["offsets"]
    buf = npz["data"][off[i]:off[i + 1]]
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot decode cached frame {i}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def cache_path(stem):
    return CACHE_DIR / f"{stem}.npz"


def existing_ok(path, grid):
    if not path.is_file():
        return False
    try:
        with np.load(path) as z:
            return len(z["k"]) == len(grid) and np.array_equal(z["frame_idx"], grid) \
                and len(z["offsets"]) == len(grid) + 1
    except Exception:
        return False


def process_clip(video_path, stem, grid):
    """Decode one video sequentially and write its cache file. Returns (stem, status, frames, bytes, error)."""
    out = cache_path(stem)
    if existing_ok(out, grid):
        return stem, "skipped", len(grid), out.stat().st_size, None
    wanted = {}
    for pos, f in enumerate(grid.tolist()):
        wanted.setdefault(f, []).append(pos)
    last = int(grid.max())
    jpegs = [None] * len(grid)
    try:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            idx = -1
            for frame in container.decode(stream):
                idx += 1
                if idx in wanted:
                    buf = encode_jpeg(resize_rgb(frame.to_ndarray(format="rgb24")))
                    for pos in wanted[idx]:
                        jpegs[pos] = buf
                if idx >= last:
                    break
        missing = [p for p, b in enumerate(jpegs) if b is None]
        if missing:
            raise RuntimeError(f"video ended after frame {idx}; {len(missing)} grid frames missing "
                               f"(first grid position {missing[0]}, frame {grid[missing[0]]})")
        offsets = np.zeros(len(grid) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(b) for b in jpegs])
        tmp = out.with_name(out.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez(fh, k=np.arange(len(grid), dtype=np.int32), frame_idx=grid.astype(np.int32),
                     offsets=offsets, data=np.concatenate(jpegs).astype(np.uint8))
        os.replace(tmp, out)
        return stem, "written", len(grid), out.stat().st_size, None
    except Exception as e:  # report and continue with the other clips
        return stem, "error", 0, 0, f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------- #
# Verification against an independent decoder (OpenCV, sequential, no seeking)
# --------------------------------------------------------------------------- #
def decode_cv2_sequential(video_path, indices):
    cap = cv2.VideoCapture(str(video_path))
    frames, idx, last = {}, 0, max(indices)
    try:
        while idx <= last:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx in indices:
                frames[idx] = resize_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            idx += 1
    finally:
        cap.release()
    return frames


def verify(clips, log):
    log("")
    log(f"6. Verification on the first {len(clips)} clips: cached JPEG vs the same frame decoded again with "
        f"OpenCV (sequential), resized the same way. MAD = mean absolute pixel difference (0-255).")
    log("   MAD at frame-1 / frame+1 are shown as a reference: the cached frame should match its own index best.")
    for c in clips:
        path = cache_path(c["stem"])
        if not path.is_file():
            log(f"   {c['stem']}: no cache file")
            continue
        with np.load(path) as z:
            K = len(z["k"])
            positions = sorted({0, K // 2, K - 1})
            idx = {int(z["frame_idx"][p]) for p in positions}
            want = idx | {i - 1 for i in idx if i > 0} | {i + 1 for i in idx}
            ref = decode_cv2_sequential(c["video"], want)
            for p in positions:
                f = int(z["frame_idx"][p])
                cached = read_cached_frame(z, p).astype(np.int16)

                def mad(i):
                    return f"{np.abs(cached - ref[i]).mean():6.3f}" if i in ref else "   n/a"
                log(f"   {c['stem']:<32} pos {p:>3} frame {f:>4}  MAD {mad(f)}   "
                    f"(frame-1 {mad(f - 1)}, frame+1 {mad(f + 1)})")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips in sorted rgb_path order")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    lines = []

    def log(text=""):
        print(text, flush=True)
        lines.append(str(text))

    dataset_root = args.dataset_path.resolve()
    if (dataset_root / "dataset" / "metadata-synthetic.csv").is_file():
        dataset_root = dataset_root / "dataset"
    meta_csv = dataset_root / "metadata-synthetic.csv"
    for f in (meta_csv, WINDOWS_CSV, MANIFEST, CONFIG):
        if not f.is_file():
            fail(f"{f} not found")

    manifest = json.loads(MANIFEST.read_text(encoding="ascii"))
    windows_sha = sha256_file(WINDOWS_CSV)
    if windows_sha != manifest.get("windows_sha256"):
        fail(f"{WINDOWS_CSV.name} SHA-256 {windows_sha} != windows_sha256 {manifest.get('windows_sha256')}")
    config = json.loads(CONFIG.read_text(encoding="ascii"))
    if config["model_fps"] != MODEL_FPS or config["window_frames"] != WINDOW_FRAMES:
        fail(f"windows_config.json has model_fps {config['model_fps']}, window_frames {config['window_frames']}; "
             f"expected {MODEL_FPS}, {WINDOW_FRAMES}")

    meta = pd.read_csv(meta_csv, usecols=["rgb_path", "no_frames", "duration"])
    win = pd.read_csv(WINDOWS_CSV, usecols=["rgb_path", "t_start"])
    paths = sorted(win["rgb_path"].unique())
    missing = sorted(set(paths) - set(meta["rgb_path"]))
    if missing:
        fail(f"{len(missing)} window clips are not in metadata-synthetic.csv, e.g. {missing[:3]}")
    meta = meta.set_index("rgb_path")
    if args.limit is not None:
        paths = paths[:args.limit]

    log("Frame cache build (synthetic only)")
    log(f"time: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    log(f"metadata: {meta_csv}")
    log(f"windows_sha256: {windows_sha} (matches manifest)")
    log(f"cache dir: {CACHE_DIR}")
    log(f"clips: {len(paths)}" + (f" (--limit {args.limit})" if args.limit is not None else " (all)")
        + f", workers {args.workers}; PyAV {av.__version__}, OpenCV {cv2.__version__}")

    # Grids and window checks for the selected clips (all checks before any decoding).
    clips = []
    grouped = win[win["rgb_path"].isin(paths)].groupby("rgb_path")
    for p in paths:
        no_frames, duration = int(meta.at[p, "no_frames"]), float(meta.at[p, "duration"])
        fps_video = no_frames / duration
        stem = Path(p.replace("\\", "/")).stem
        K, grid = build_grid(grouped.get_group(p), fps_video, no_frames, stem)
        video = dataset_root / p.replace("\\", "/")
        if not video.is_file():
            fail(f"video missing: {video}")
        clips.append({"rgb_path": p, "stem": stem, "video": video, "K": K, "grid": grid, "fps": fps_video})
    log(f"2. grid checks passed: every window equals grid[8*t_start : 8*t_start+16]; "
        f"max grid index <= no_frames-1; total grid frames {sum(c['K'] for c in clips)}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    done = errors = frames = nbytes = written = skipped = 0
    err_list = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(process_clip, c["video"], c["stem"], c["grid"]) for c in clips]
        for fut in as_completed(futs):
            stem, status, nf, nb, err = fut.result()
            done += 1
            if status == "error":
                errors += 1
                err_list.append((stem, err))
                log(f"   ERROR {stem}: {err}")
            else:
                frames += nf
                nbytes += nb
                written += status == "written"
                skipped += status == "skipped"
            if done % PROGRESS_EVERY == 0 or done == len(clips):
                log(f"   progress {done}/{len(clips)} clips, {errors} errors, "
                    f"{(time.perf_counter() - t0) / 60:.1f} min")
    minutes = (time.perf_counter() - t0) / 60

    verify(clips[:VERIFY_CLIPS], log)

    # Manifest and summary describe the whole cache directory, not only this run.
    all_paths = sorted(win["rgb_path"].unique())
    present = [Path(p.replace("\\", "/")).stem for p in all_paths
               if cache_path(Path(p.replace("\\", "/")).stem).is_file()]
    dir_bytes = sum(cache_path(s).stat().st_size for s in present)
    complete = len(present) == len(all_paths) and errors == 0
    cache_manifest = {
        "complete": complete,
        "limit": args.limit,
        "clips_in_cache": len(present),
        "clips_total": len(all_paths),
        "clips_this_run": len(clips),
        "written_this_run": written,
        "skipped_this_run": skipped,
        "errors_this_run": errors,
        "frames_this_run": frames,
        "bytes_in_cache": dir_bytes,
        "resize_rule": f"full frame (no crop) resized to {SIZE}x{SIZE} with cv2.INTER_AREA, RGB",
        "jpeg_quality": JPEG_QUALITY,
        "model_fps": MODEL_FPS,
        "grid_rule": "t_k = k/8, frame = floor(t_k * no_frames/duration + 0.5), K = 8*max(t_start) + 16",
        "decoder": f"PyAV {av.__version__}, sequential, no seeking",
        "windows_sha256": windows_sha,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (CACHE_DIR / "cache_manifest.json").write_text(json.dumps(cache_manifest, indent=2) + "\n", encoding="ascii")

    log("")
    log(f"cache: {len(present)}/{len(all_paths)} clips present, {dir_bytes / 2**30:.3f} GB, complete {complete}")
    if err_list:
        log("errors: " + "; ".join(f"{s}: {e}" for s, e in err_list))
    log(f"FINAL: clips {len(clips)} (written {written}, skipped {skipped}), errors {errors}, frames {frames}, "
        f"{nbytes / 2**30:.3f} GB, {minutes:.2f} minutes")
    (CACHE_DIR / "cache_summary.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"saved {CACHE_DIR / 'cache_manifest.json'} and cache_summary.txt")


if __name__ == "__main__":
    main()
