"""Cache the 8-fps frame grid of every synthetic clip as 224 x 224 JPEGs.

Default mode: SYNTHETIC ONLY. Read-only inputs: metadata-synthetic.csv, synthetic_windows.csv,
split_manifest.json, windows_config.json. Never opens real videos or real metadata (only --real does; see below).

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

REAL TEST CACHE (--real; opt-in, the default above is unchanged): the 2,027 clips of metadata-real.csv, read with
usecols path, no_frames, duration, height, width ONLY (no label column is read). Checks metadata-real.csv and
real_test_set.csv (path column only) against real_test_set_manifest.json. Windows by the synthetic rule
(make_windows): t_start = j * 0.5 s, kept while the last frame index floor((t_start + 15/8) * fps + 0.5) <= no_frames - 1,
fps = no_frames / duration; window time t_centre = t_start + 0.9375 s. Before any decoding, the same window function is
applied to the 2,211 synthetic clips and must reproduce synthetic_windows.csv exactly. Grid, decoder, resize and JPEG
as above, into cache/frames224_real_v1/ (real_windows.csv, cache_manifest.json, cache_summary.txt there); clips with no
window get no cache file and are listed. --report-dir also receives cache_summary.txt and real_clips.csv.
    .venv-ml\\Scripts\\python.exe build_frame_cache.py --real --report-dir DIR [--limit N] [--workers W]
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
from make_windows import frame_index, window_frame_indices  # noqa: E402

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
# real test cache (--real)
REAL_CACHE_DIR = REPO_ROOT / "cache" / "frames224_real_v1"
REAL_WINDOWS_CSV = REAL_CACHE_DIR / "real_windows.csv"
REAL_TEST_CSV = REPO_ROOT / "real_test_set.csv"
REAL_TEST_MANIFEST = REPO_ROOT / "real_test_set_manifest.json"
REAL_META_COLUMNS = ["path", "no_frames", "duration", "height", "width"]  # never a label column
SYN_WINDOWS_CSV = REPO_ROOT / "synthetic_windows.csv"
STRIDE_S = 0.5
CENTRE_S = (WINDOW_FRAMES - 1) / MODEL_FPS / 2  # 0.9375 s: t_centre = t_start + CENTRE_S
EXPECTED_REAL_CLIPS = 2027


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


def grid_frame_indices_window_rule(fps_video, K):
    """make_windows.frame_index(k / MODEL_FPS, fps): floor(round(t_k * fps, 9) + 0.5), the window rule with its float
    noise guard. Equal to grid_frame_indices on every synthetic clip (fps 20.0); used by --real, where fps =
    no_frames / duration can fall within 1e-9 of a .5 boundary (see real_main)."""
    return np.array([frame_index(k / MODEL_FPS, fps_video) for k in range(K)], dtype=np.int64)


def build_grid(clip_windows, fps_video, no_frames, name, grid_fn=grid_frame_indices):
    """Grid of one clip; checks every window is a contiguous slice and fits in the video."""
    t_starts = clip_windows["t_start"].to_numpy(float)
    k0s = np.round(t_starts * MODEL_FPS).astype(int)
    if not np.allclose(k0s / MODEL_FPS, t_starts):
        fail(f"{name}: t_start values are not multiples of 1/{MODEL_FPS} s")
    K = int(k0s.max()) + WINDOW_FRAMES
    grid = grid_fn(fps_video, K)
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


def cache_path(stem, cache_dir=CACHE_DIR):
    return cache_dir / f"{stem}.npz"


def existing_ok(path, grid):
    if not path.is_file():
        return False
    try:
        with np.load(path) as z:
            return len(z["k"]) == len(grid) and np.array_equal(z["frame_idx"], grid) \
                and len(z["offsets"]) == len(grid) + 1
    except Exception:
        return False


def process_clip(video_path, stem, grid, cache_dir=CACHE_DIR):
    """Decode one video sequentially and write its cache file. Returns (stem, status, frames, bytes, error)."""
    out = cache_path(stem, cache_dir)
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


def verify(clips, log, cache_dir=CACHE_DIR):
    log("")
    log(f"6. Verification on the first {len(clips)} clips: cached JPEG vs the same frame decoded again with "
        f"OpenCV (sequential), resized the same way. MAD = mean absolute pixel difference (0-255).")
    log("   MAD at frame-1 / frame+1 are shown as a reference: the cached frame should match its own index best.")
    for c in clips:
        path = cache_path(c["stem"], cache_dir)
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
# Real test cache (--real)
# --------------------------------------------------------------------------- #
def windows_table(meta):
    """Windows of every clip in `meta` (columns rgb_path, no_frames, duration) by the synthetic rule of
    make_windows.clip_windows, without labels: t_start = j * STRIDE_S while the last frame index <= no_frames - 1.
    Columns rgb_path, stem, window_idx, t_start, t_end, t_centre (= t_start + CENTRE_S), k0, frame_first, frame_last."""
    rows = []
    for p, nf, dur in zip(meta["rgb_path"], meta["no_frames"], meta["duration"]):
        fps_video = int(nf) / float(dur)
        stem = Path(p.replace("\\", "/")).stem
        j = 0
        while True:
            t_start = j * STRIDE_S
            frames = window_frame_indices(t_start, fps_video, MODEL_FPS, WINDOW_FRAMES)
            if frames[-1] > int(nf) - 1:
                break
            ts = round(t_start, 6)
            rows.append({"rgb_path": p, "stem": stem, "window_idx": j, "t_start": ts,
                         "t_end": round(t_start + (WINDOW_FRAMES - 1) / MODEL_FPS, 6),
                         "t_centre": round(ts + CENTRE_S, 6), "k0": int(round(ts * MODEL_FPS)),
                         "frame_first": frames[0], "frame_last": frames[-1]})
            j += 1
    return pd.DataFrame(rows, columns=["rgb_path", "stem", "window_idx", "t_start", "t_end", "t_centre", "k0",
                                       "frame_first", "frame_last"])


def check_window_rule_on_synthetic(dataset_root, log):
    """windows_table applied to the 2,211 synthetic clips must reproduce synthetic_windows.csv exactly."""
    manifest = json.loads(MANIFEST.read_text(encoding="ascii"))
    if sha256_file(SYN_WINDOWS_CSV) != manifest.get("windows_sha256"):
        fail(f"{SYN_WINDOWS_CSV.name} SHA-256 differs from windows_sha256 in {MANIFEST.name}")
    cols = ["rgb_path", "window_idx", "t_start", "t_end", "frame_first", "frame_last"]
    ref = pd.read_csv(SYN_WINDOWS_CSV, usecols=cols).sort_values(["rgb_path", "window_idx"]).reset_index(drop=True)
    syn = pd.read_csv(dataset_root / "metadata-synthetic.csv", usecols=["rgb_path", "no_frames", "duration"])
    new = windows_table(syn)[cols].sort_values(["rgb_path", "window_idx"]).reset_index(drop=True)
    same = len(new) == len(ref) and new.equals(ref)
    if not same:
        fail(f"window rule check: windows_table on the synthetic clips gives {len(new)} windows, synthetic_windows.csv "
             f"{len(ref)}; they differ")
    log(f"2. window rule check: windows_table() on the {syn['rgb_path'].nunique()} synthetic clips reproduces "
        f"synthetic_windows.csv exactly ({len(ref)} windows; t_start, t_end, frame_first, frame_last equal)")


def write_guarded(path, df):
    """Write a CSV; an existing file must have identical content (never silently replaced)."""
    text = df.to_csv(index=False, lineterminator="\n")
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            fail(f"{path} exists with different content; not overwritten")
        return "unchanged (identical)"
    path.write_text(text, encoding="utf-8", newline="")
    return "written"


def real_main(args):
    lines = []

    def log(text=""):
        print(text, flush=True)
        lines.append(str(text))

    dataset_root = args.dataset_path.resolve()
    if (dataset_root / "dataset" / "metadata-real.csv").is_file():
        dataset_root = dataset_root / "dataset"
    meta_csv = dataset_root / "metadata-real.csv"
    for f in (meta_csv, REAL_TEST_CSV, REAL_TEST_MANIFEST, SYN_WINDOWS_CSV, MANIFEST):
        if not f.is_file():
            fail(f"{f} not found")
    report_dir = args.report_dir
    if report_dir is None:
        fail("--real needs --report-dir")

    # 1. inputs: hashes against the frozen test-set manifest; non-label columns only
    tm = json.loads(REAL_TEST_MANIFEST.read_text(encoding="utf-8"))
    meta_sha, test_sha = sha256_file(meta_csv), sha256_file(REAL_TEST_CSV)
    if meta_sha != tm["source_csv_sha256"]:
        fail(f"{meta_csv.name} SHA-256 {meta_sha} != source_csv_sha256 in {REAL_TEST_MANIFEST.name}")
    if test_sha != tm["output_sha256"]:
        fail(f"{REAL_TEST_CSV.name} SHA-256 {test_sha} != output_sha256 in {REAL_TEST_MANIFEST.name}")
    meta = pd.read_csv(meta_csv, usecols=REAL_META_COLUMNS).rename(columns={"path": "rgb_path"})
    test_paths = set(pd.read_csv(REAL_TEST_CSV, usecols=["path"])["path"])
    if len(meta) != EXPECTED_REAL_CLIPS or len(meta) != tm["source_csv_rows"]:
        fail(f"{meta_csv.name} has {len(meta)} rows, expected {EXPECTED_REAL_CLIPS}")
    if not meta["rgb_path"].is_unique:
        fail("path is not unique in metadata-real.csv")
    if len(test_paths) != tm["n"] or not test_paths <= set(meta["rgb_path"]):
        fail(f"{REAL_TEST_CSV.name}: {len(test_paths)} distinct paths (expected {tm['n']}), all must be in the metadata")
    if ((meta["no_frames"] <= 0) | (meta["duration"] <= 0)).any():
        fail("non-positive no_frames or duration in metadata-real.csv")
    meta["stem"] = meta["rgb_path"].map(lambda p: Path(p.replace("\\", "/")).stem)
    if not meta["stem"].is_unique:
        fail("clip file names (stems) are not unique; cache file names would collide")
    meta["fps"] = meta["no_frames"] / meta["duration"]
    meta["in_test400"] = meta["rgb_path"].isin(test_paths)
    meta = meta.sort_values("rgb_path").reset_index(drop=True)

    log("Frame cache build (REAL test clips; no label column read)")
    log(f"time: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    log(f"1. inputs: {meta_csv} (sha256 {meta_sha[:12]}... = test-set manifest), columns read: "
        f"{', '.join(REAL_META_COLUMNS)}")
    log(f"   {REAL_TEST_CSV.name} sha256 {test_sha[:12]}... = manifest output_sha256; {len(test_paths)} clips, column "
        f"read: path; all in metadata-real.csv")
    log(f"   cache dir: {REAL_CACHE_DIR}; PyAV {av.__version__}, OpenCV {cv2.__version__}; workers {args.workers}")
    log(f"   rules: fps = no_frames / duration; grid t_k = k/{MODEL_FPS}, frame floor(round(t_k * fps, 9) + 0.5); windows "
        f"{WINDOW_FRAMES} frames, stride {STRIDE_S} s, t_centre = t_start + {CENTRE_S} s; resize full frame to "
        f"{SIZE}x{SIZE} INTER_AREA, JPEG {JPEG_QUALITY}")

    # 2. the window function reproduces the synthetic windows
    check_window_rule_on_synthetic(dataset_root, log)

    # 3. real windows and grids
    win = windows_table(meta)
    n_win = win.groupby("rgb_path").size()
    meta["n_windows"] = meta["rgb_path"].map(n_win).fillna(0).astype(int)
    no_win = meta[meta["n_windows"] == 0]
    clips = []
    grids = {}
    rule_diff = []  # clips where the unrounded synthetic grid formula would pick another frame
    for p, g in win.groupby("rgb_path", sort=True):
        r = meta.loc[meta["rgb_path"] == p].iloc[0]
        K, grid = build_grid(g, float(r["fps"]), int(r["no_frames"]), r["stem"], grid_frame_indices_window_rule)
        n_diff = int((grid_frame_indices(float(r["fps"]), K) != grid).sum())
        if n_diff:
            rule_diff.append((p, n_diff))
        grids[p] = (K, grid)
        clips.append({"rgb_path": p, "stem": r["stem"], "video": dataset_root / p.replace("\\", "/"), "K": K,
                      "grid": grid})
    missing_video = [c["rgb_path"] for c in clips if not c["video"].is_file()]
    if missing_video:
        fail(f"{len(missing_video)} videos missing, e.g. {missing_video[:3]}")
    meta["grid_frames"] = meta["rgb_path"].map(lambda p: grids[p][0] if p in grids else 0)
    meta["grid_repeats_frames"] = meta["rgb_path"].map(
        lambda p: bool(p in grids and len(set(grids[p][1].tolist())) < len(grids[p][1])))
    t4 = meta[meta["in_test400"]]
    log(f"3. clips: {len(meta)} (frozen 400 subset: {len(t4)}); with >= 1 window: {len(clips)} "
        f"(subset {int((t4['n_windows'] > 0).sum())}); windows {len(win)} (subset "
        f"{int(t4['n_windows'].sum())}); windows per clip min / median / max (clips with windows) "
        f"{int(n_win.min())} / {float(n_win.median()):g} / {int(n_win.max())}; grid frames {int(meta['grid_frames'].sum())}")
    log(f"   clips shorter than one window (no window; no cache file): {len(no_win)} "
        f"(in the 400 subset: {int(no_win['in_test400'].sum())})")
    for _, r in no_win.iterrows():
        need = math.floor(round((WINDOW_FRAMES - 1) / MODEL_FPS * r["fps"], 9) + 0.5) + 1
        log(f"     {r['rgb_path']}: duration {r['duration']:.3f} s, no_frames {int(r['no_frames'])}, fps "
            f"{r['fps']:.3f}; one window needs {need} frames; in 400 subset: {bool(r['in_test400'])}")
    rep = meta[meta["grid_repeats_frames"]]
    log(f"   clips whose 8-fps grid repeats video frames (fps < 8, same rule as synthetic): {len(rep)} "
        f"(subset {int(rep['in_test400'].sum())}); fps of these {rep['fps'].min():.2f}-{rep['fps'].max():.2f}"
        if len(rep) else "   clips whose 8-fps grid repeats video frames: 0")
    log(f"   grid formula: frame = floor(round(t_k * fps, 9) + 0.5) (make_windows.frame_index, the window rule). The "
        f"synthetic cache formula without the 9-decimal rounding gives the same grid on all synthetic clips but differs "
        f"here in {len(rule_diff)} clips, {sum(n for _, n in rule_diff)} grid positions (t_k * fps within 1e-9 of a .5 "
        f"boundary, by one video frame); with it the windows would not be grid slices:")
    for p, n in rule_diff:
        log(f"     {p}: {n} grid positions")
    log(f"   fps (no_frames / duration): min {meta['fps'].min():.3f}, median {meta['fps'].median():.3f}, max "
        f"{meta['fps'].max():.3f}; duration min {meta['duration'].min():.3f} s, median {meta['duration'].median():.3f} s, "
        f"max {meta['duration'].max():.3f} s; height min {int(meta['height'].min())}, median "
        f"{float(meta['height'].median()):g}, max {int(meta['height'].max())}")

    REAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    windows_status = write_guarded(REAL_WINDOWS_CSV, win)
    windows_sha = sha256_file(REAL_WINDOWS_CSV)
    log(f"   {REAL_WINDOWS_CSV.name}: {windows_status}, sha256 {windows_sha}")

    # 4. decode
    run_clips = clips if args.limit is None else clips[:args.limit]
    log(f"4. decoding {len(run_clips)} clips" + (f" (--limit {args.limit})" if args.limit is not None else " (all)"))
    t0 = time.perf_counter()
    done = errors = frames = nbytes = written = skipped = 0
    err_list, status = [], {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(process_clip, c["video"], c["stem"], c["grid"], REAL_CACHE_DIR): c["rgb_path"]
                for c in run_clips}
        for fut in as_completed(futs):
            stem, st, nf, nb, err = fut.result()
            status[futs[fut]] = st
            done += 1
            if st == "error":
                errors += 1
                err_list.append((futs[fut], err))
                log(f"   ERROR {futs[fut]}: {err}")
            else:
                frames += nf
                nbytes += nb
                written += st == "written"
                skipped += st == "skipped"
            if done % (4 * PROGRESS_EVERY) == 0 or done == len(run_clips):
                log(f"   progress {done}/{len(run_clips)} clips, {errors} errors, "
                    f"{(time.perf_counter() - t0) / 60:.1f} min")
    minutes = (time.perf_counter() - t0) / 60

    verify(run_clips[:VERIFY_CLIPS], log, REAL_CACHE_DIR)

    present = [c["stem"] for c in clips if cache_path(c["stem"], REAL_CACHE_DIR).is_file()]
    dir_bytes = sum(cache_path(s, REAL_CACHE_DIR).stat().st_size for s in present)
    complete = len(present) == len(clips) and errors == 0
    meta["cache_status"] = meta["rgb_path"].map(lambda p: status.get(p, "not_run" if p in grids else "no_window"))
    cache_manifest = {
        "complete": complete,
        "data": "REAL (metadata-real.csv; non-label columns only)",
        "limit": args.limit,
        "clips_metadata": len(meta),
        "clips_with_windows": len(clips),
        "clips_without_window": no_win["rgb_path"].tolist(),
        "clips_in_cache": len(present),
        "clips_this_run": len(run_clips),
        "written_this_run": written,
        "skipped_this_run": skipped,
        "errors_this_run": errors,
        "frames_this_run": frames,
        "bytes_in_cache": dir_bytes,
        "resize_rule": f"full frame (no crop) resized to {SIZE}x{SIZE} with cv2.INTER_AREA, RGB",
        "jpeg_quality": JPEG_QUALITY,
        "model_fps": MODEL_FPS,
        "grid_rule": "t_k = k/8, frame = floor(round(t_k * no_frames/duration, 9) + 0.5) (make_windows.frame_index), "
                     "K = 8*max(t_start) + 16",
        "grid_rule_differs_from_unrounded": {p: n for p, n in rule_diff},
        "window_rule": f"t_start = j*{STRIDE_S} s while the last frame index <= no_frames-1; t_centre = t_start + "
                       f"{CENTRE_S} s",
        "decoder": f"PyAV {av.__version__}, sequential, no seeking",
        "metadata_real_sha256": meta_sha,
        "real_test_set_sha256": test_sha,
        "real_windows_sha256": windows_sha,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (REAL_CACHE_DIR / "cache_manifest.json").write_text(json.dumps(cache_manifest, indent=2) + "\n", encoding="ascii")

    log("")
    log(f"cache: {len(present)}/{len(clips)} clips with windows present, {dir_bytes / 2**30:.3f} GB, complete {complete}")
    if err_list:
        log("errors: " + "; ".join(f"{s}: {e}" for s, e in err_list))
    log(f"FINAL: clips {len(run_clips)} (written {written}, skipped {skipped}), errors {errors}, frames {frames}, "
        f"{nbytes / 2**30:.3f} GB, {minutes:.2f} minutes")
    text = "\n".join(lines) + "\n"
    (REAL_CACHE_DIR / "cache_summary.txt").write_text(text, encoding="utf-8")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "cache_summary.txt").write_text(text, encoding="utf-8")
    meta[["rgb_path", "stem", "no_frames", "duration", "fps", "height", "width", "n_windows", "grid_frames",
          "grid_repeats_frames", "in_test400", "cache_status"]].rename(columns={"rgb_path": "path"}).to_csv(
        report_dir / "real_clips.csv", index=False)
    (report_dir / "cache_manifest.json").write_text(json.dumps(cache_manifest, indent=2) + "\n", encoding="ascii")
    print(f"saved {REAL_CACHE_DIR / 'cache_manifest.json'}, cache_summary.txt; copies and real_clips.csv in {report_dir}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips in sorted rgb_path order")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--real", action="store_true", help="real test cache (see module doc); no label column is read")
    ap.add_argument("--report-dir", type=Path, default=None, help="with --real: copy of the summary + real_clips.csv")
    args = ap.parse_args()
    if args.real:
        real_main(args)
        return
    if args.report_dir is not None:
        fail("--report-dir only applies with --real")

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
