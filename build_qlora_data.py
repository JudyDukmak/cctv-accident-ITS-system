"""QLoRA training data for the narrow call: label audit, window manifest and frame cache. SYNTHETIC ONLY.

A new script because no existing script builds training examples for the VLM: it imports the frozen frame sampling and
decoding of vlm_zero_shot.py (load_clips, window_times, to_indices, decode; not changed) and the narrow-v2 settings of
narrow_prompt_v2.py, so a training window is sampled exactly as the narrow call samples its input. No model, no
training, no real clip or real label.

--audit (CPU): checks the collision labels of all 2211 synthetic clips (metadata-synthetic.csv): type, t*
  (accident_time), accident_frame, the box x1, y1, x2, y2 and the centre center_x, center_y (fractions of the image),
  frame size; writes results/qlora_data/label_audit.txt.
--manifest (CPU): results/qlora_data/manifest.csv, two windows per clip (refuses to overwrite):
  A  centre = the classifier's out-of-fold t_hat (C_cctv seed 0, frozen post-processing; vlm_zero_shot.CLASSIFIER_PRED)
  B  centre = t* + offset, offset ~ uniform[-1, +1] s (numpy default_rng(B_SEED), one draw per clip in rgb_path order)
  each 16 frames evenly over centre +- 2 s, shifted to stay inside the clip (vlm_zero_shot.window_times/to_indices,
  as the narrow call). Raw labels only (box in 0-1000, point in 0-1, motion from type); the target text format is
  chosen at training time.
--cache (CPU, resumable): the 16 frames of every window as the model receives them in the narrow call: decoded with
  vlm_zero_shot.decode (448 x 256 pixel area, aspect kept: 452 x 254 for 1920 x 1080), then resized to the processor's
  448 x 256 (PIL bicubic, which reproduces the processor's own resize; checked by --verify), JPEG quality JPEG_Q, one
  .npz per window under cache/qlora_frames448_v1/ (frame_idx, offsets, data = JPEG bytes; written to a temporary name
  and renamed; an existing file with the same frame indices is skipped).
--verify (CPU): for 3 windows, the cached frames vs the frames the real processor gives the model (narrow v1 and v2
  layouts), and that the processor leaves 448 x 256 inputs unresized.
--report (CPU): counts, t* inside the window, box sizes, cache size, an image grid of 4 examples.
Usage (from cctv-accident-ITS-system/):
    .venv-vlm\\Scripts\\python.exe build_qlora_data.py --audit
    .venv-vlm\\Scripts\\python.exe build_qlora_data.py --manifest
    .venv-vlm\\Scripts\\python.exe build_qlora_data.py --cache --workers 8
    .venv-vlm\\Scripts\\python.exe build_qlora_data.py --verify
    .venv-vlm\\Scripts\\python.exe build_qlora_data.py --report
"""

import argparse
import collections
import csv
import io
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
import vlm_zero_shot as vz  # noqa: E402  (frozen protocol code: imported, not changed)
import narrow_prompt_v2 as n2  # noqa: E402

OUT = REPO_ROOT / "results" / "qlora_data"
CACHE = REPO_ROOT / "cache" / "qlora_frames448_v1"
MANIFEST = OUT / "manifest.csv"
NARROW_TEST = REPO_ROOT / "results" / "narrow_v1_v2_200" / "test_clips.csv"
B_SEED, B_OFFSET_S = 0, 1.0
HALF_S, N_FRAMES = n2.HALF_S_V2, n2.N_FRAMES_V2  # 2.0 s, 16 frames (narrow v1 and v2)
MODEL_W, MODEL_H = 448, 256  # what the processor gives the model for the 452 x 254 decoded frames
JPEG_Q = 95
MOTION = {"head-on": "opposite", "rear-end": "same direction", "sideswipe": "parallel", "t-bone": "crossing",
          "single": "one vehicle"}
COLUMNS = ["rgb_path", "stem", "scenario_id", "fold", "type", "weather", "day_night", "window", "centre_source",
           "centre_s", "offset_s", "win_start_s", "win_end_s", "frame_indices", "t_star", "t_star_inside",
           "t_star_to_nearest_frame_s", "x1_1000", "y1_1000", "x2_1000", "y2_1000", "center_x", "center_y", "motion",
           "design_clip", "narrow_test_clip", "frames_path", "frame_w", "frame_h", "fps", "duration"]


def fnum(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- 1. label audit

def audit():
    meta = vz.synthetic_meta()
    cols = list(next(iter(meta.values())).keys())
    issues = collections.defaultdict(list)
    sizes = collections.Counter()
    frame_diff, centre_diff = [], []
    for p, m in meta.items():
        if m["type"] not in vz.TYPES:
            issues["type missing or not one of the 5"].append(p)
        t, nf, dur = fnum(m["accident_time"]), fnum(m["no_frames"]), fnum(m["duration"])
        if t is None or nf is None or dur is None or nf <= 0 or dur <= 0:
            issues["t*, no_frames or duration missing"].append(p)
            continue
        fps = nf / dur
        if not 0 <= t <= (nf - 1) / fps:
            issues["t* outside [0, last frame time]"].append(p)
        af = fnum(m["accident_frame"])
        if af is None:
            issues["accident_frame missing"].append(p)
        else:
            frame_diff.append(af - t * fps)
        box = [fnum(m[k]) for k in ("x1", "y1", "x2", "y2")]
        c = [fnum(m[k]) for k in ("center_x", "center_y")]
        if any(v is None for v in box):
            issues["box missing"].append(p)
            continue
        x1, y1, x2, y2 = box
        if not all(0 <= v <= 1 for v in box):
            issues["box outside the image [0, 1]"].append(p)
        if not (x1 < x2 and y1 < y2):
            issues["box empty or reversed"].append(p)
        if any(v is None for v in c):
            issues["centre missing"].append(p)
        else:
            centre_diff.append(max(abs(c[0] - (x1 + x2) / 2), abs(c[1] - (y1 + y2) / 2)))
            if not (x1 <= c[0] <= x2 and y1 <= c[1] <= y2):
                issues["centre outside the box"].append(p)
        sizes[(m["width"], m["height"])] += 1
    fd = np.array(frame_diff)
    cd = np.array(centre_diff)
    L = [f"Label audit of the synthetic collision labels; {vz.now()}; {vz.META_SYN.relative_to(REPO_ROOT)} "
         f"({len(meta)} clips). Synthetic only.",
         f"columns: {', '.join(cols)}",
         "collision location columns: center_x, center_y (collision point, fractions of width / height), x1, y1, x2, "
         "y2 (collision box, fractions), accident_time (t*, s), accident_frame (frame index of t*); also "
         "annotations_path (per-frame annotation file, not used here)",
         f"frame sizes (width x height): {dict(sizes)}",
         f"accident_frame - t* x fps (fps = no_frames / duration): min {fd.min():.4f}, max {fd.max():.4f}, "
         f"|diff| > 0.5 frame: {int((np.abs(fd) > 0.5).sum())}",
         f"centre vs box midpoint, max |difference| per clip: max {cd.max():.2e}, clips > 1e-6: {int((cd > 1e-6).sum())}",
         "checks (count, first examples):"]
    for name in ("type missing or not one of the 5", "t*, no_frames or duration missing", "t* outside [0, last frame time]",
                 "accident_frame missing", "box missing", "box outside the image [0, 1]", "box empty or reversed",
                 "centre missing", "centre outside the box"):
        L.append(f"  {name}: {len(issues[name])}" + (f"  e.g. {issues[name][:3]}" if issues[name] else ""))
    L.append(f"types: {dict(collections.Counter(m['type'] for m in meta.values()))}")
    text = "\n".join(L) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "label_audit.txt").write_text(text, encoding="utf-8")
    print(text)
    return sum(len(v) for v in issues.values())


# ---------------------------------------------------------------- 2. manifest

def frames_rel(stem, w):
    return (CACHE / f"{stem}_{w}.npz").relative_to(REPO_ROOT).as_posix()


def manifest():
    if MANIFEST.exists():
        vz.fail(f"{MANIFEST} exists; refusing to overwrite")
    clips = sorted(vz.load_clips(), key=lambda c: c["rgb_path"])  # all 2211, t_hat out-of-fold (checked by load_clips)
    split = {vz.norm(r["rgb_path"]): r for r in vz.read_csv(vz.SPLIT_CSV)}
    meta = vz.synthetic_meta()
    design = {vz.norm(r["rgb_path"]) for r in vz.read_csv(vz.DEV_CSV)}
    test = {vz.norm(r["rgb_path"]) for r in vz.read_csv(NARROW_TEST)}
    if len(clips) != 2211 or len(design) != 32 or len(test) != 200:
        vz.fail("expected 2211 clips, 32 design clips and 200 narrow-test clips")
    rng = np.random.default_rng(B_SEED)
    offsets = rng.uniform(-B_OFFSET_S, B_OFFSET_S, size=len(clips))
    rows = []
    for c, off in zip(clips, offsets):
        m, s = meta[c["rgb_path"]], split[c["rgb_path"]]
        stem = Path(c["rgb_path"]).stem
        for w, centre, src, o in (("A", c["t_hat"], "t_hat C_cctv seed 0 out-of-fold", ""),
                                  ("B", c["t_star"] + off, "t* + offset", float(off))):
            idx = vz.to_indices(c, vz.window_times(c, centre, HALF_S, N_FRAMES))
            ft = np.array(idx) / c["fps"]
            rows.append({"rgb_path": c["rgb_path"], "stem": stem, "scenario_id": s["scenario_id"], "fold": c["val_fold"],
                         "type": c["type"], "weather": c["weather"],
                         "day_night": "night" if c["weather"] == "night" else "day", "window": w, "centre_source": src,
                         "centre_s": centre, "offset_s": o, "win_start_s": ft[0], "win_end_s": ft[-1],
                         "frame_indices": json.dumps(idx), "t_star": c["t_star"],
                         "t_star_inside": bool(ft[0] <= c["t_star"] <= ft[-1]),
                         "t_star_to_nearest_frame_s": float(np.min(np.abs(ft - c["t_star"]))),
                         **{f"{k}_1000": round(1000 * float(m[k]), 1) for k in ("x1", "y1", "x2", "y2")},
                         "center_x": c["center_x"], "center_y": c["center_y"], "motion": MOTION[c["type"]],
                         "design_clip": c["rgb_path"] in design, "narrow_test_clip": c["rgb_path"] in test,
                         "frames_path": frames_rel(stem, w), "frame_w": MODEL_W, "frame_h": MODEL_H, "fps": c["fps"],
                         "duration": c["duration"]})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=COLUMNS)
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {MANIFEST}: {len(rows)} windows of {len(clips)} clips; sha256 {vz.sha256(MANIFEST)}")


# ---------------------------------------------------------------- 3. frame cache

def model_frames(got, idx):
    """The decoded frames (vz.decode, 452 x 254) resized to the processor's 448 x 256, uint8 (T, H, W, 3)."""
    from PIL import Image

    out = []
    for i in idx:
        img = got[i]
        if img.size != (452, 254):
            raise RuntimeError(f"decoded size {img.size}, expected 452 x 254")
        out.append(np.asarray(img.resize((MODEL_W, MODEL_H), Image.BICUBIC)))
    return np.stack(out)


def jpeg_bytes(frame):
    from PIL import Image

    b = io.BytesIO()
    Image.fromarray(frame).save(b, format="JPEG", quality=JPEG_Q)
    return np.frombuffer(b.getvalue(), dtype=np.uint8)


def read_npz(path):
    """(frame_idx, frames uint8 (T, H, W, 3)) of a cached window."""
    from PIL import Image

    with np.load(path) as z:
        off, data, fi = z["offsets"], z["data"], z["frame_idx"]
        frames = [np.asarray(Image.open(io.BytesIO(data[off[k]:off[k + 1]].tobytes())).convert("RGB"))
                  for k in range(len(fi))]
    return fi.tolist(), np.stack(frames)


def existing_ok(path, idx):
    if not path.is_file():
        return False
    try:
        with np.load(path) as z:
            return z["frame_idx"].tolist() == idx and len(z["offsets"]) == len(idx) + 1
    except Exception:
        return False


def cache_clip(rgb_path, windows):
    """windows: [(window, frame indices)]. Decode the clip once, write one .npz per window. Returns (status, bytes)."""
    c = CLIP_INFO[rgb_path]
    stem = Path(rgb_path).stem
    todo = [(w, idx) for w, idx in windows if not existing_ok(CACHE / f"{stem}_{w}.npz", idx)]
    if not todo:
        return "skipped", sum((CACHE / f"{stem}_{w}.npz").stat().st_size for w, _ in windows)
    got = vz.decode(c, sorted({i for _, idx in todo for i in idx}))
    for w, idx in todo:
        fr = model_frames(got, idx)
        jp = [jpeg_bytes(f) for f in fr]
        off = np.zeros(len(jp) + 1, dtype=np.int64)
        off[1:] = np.cumsum([len(b) for b in jp])
        out = CACHE / f"{stem}_{w}.npz"
        tmp = out.with_name(out.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez(fh, frame_idx=np.array(idx, dtype=np.int32), offsets=off, data=np.concatenate(jp),
                     size=np.array([MODEL_W, MODEL_H], dtype=np.int32))
        os.replace(tmp, out)
    return "written", sum((CACHE / f"{stem}_{w}.npz").stat().st_size for w, _ in windows)


CLIP_INFO = {}


def _init_worker():
    CLIP_INFO.update({c["rgb_path"]: c for c in vz.load_clips()})


def _work(args):
    rgb_path, windows = args
    try:
        return rgb_path, *cache_clip(rgb_path, windows), ""
    except Exception as e:  # report and continue
        return rgb_path, "error", 0, f"{type(e).__name__}: {e}"


def cache(workers):
    rows = vz.read_csv(MANIFEST)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["rgb_path"]].append((r["window"], json.loads(r["frame_indices"])))
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = sorted(by.items())
    t0 = time.perf_counter()
    stat = collections.Counter()
    errors, total = [], 0
    log = open(OUT / "cache_log.txt", "a", encoding="utf-8")
    log.write(f"{vz.now()} START: {len(jobs)} clips, {len(rows)} windows, {workers} workers\n")
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        futs = [ex.submit(_work, j) for j in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            p, st, nbytes, err = f.result()
            stat[st] += 1
            total += nbytes
            if err:
                errors.append((p, err))
                log.write(f"{vz.now()} ERROR {p}: {err}\n")
            if k % 100 == 0 or k == len(jobs):
                msg = (f"{vz.now()} {k}/{len(jobs)} clips ({dict(stat)}), {total / 2 ** 30:.2f} GB so far, "
                       f"{time.perf_counter() - t0:.0f} s")
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
    wall = time.perf_counter() - t0
    files = list(CACHE.glob("*.npz"))
    size = sum(f.stat().st_size for f in files)
    msg = (f"{vz.now()} DONE: {dict(stat)}; {len(errors)} errors; {len(files)} window files, {size / 2 ** 30:.3f} GB "
           f"({size / max(len(files), 1) / 2 ** 10:.0f} KB per window); wall {wall:.0f} s with {workers} workers")
    print(msg)
    log.write(msg + "\n")
    log.close()
    (OUT / "cache_summary.json").write_text(json.dumps(
        {"windows_expected": len(rows), "window_files": len(files), "bytes": size, "wall_s_this_run": wall,
         "workers": workers, "status_this_run": dict(stat), "errors": errors, "jpeg_quality": JPEG_Q,
         "frame_size": [MODEL_W, MODEL_H], "finished": vz.now()}, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- verify

def verify():
    from transformers import AutoProcessor

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    proc = AutoProcessor.from_pretrained(vz.smoke.MODELS[vz.MODEL_KEY])
    rows = vz.read_csv(MANIFEST)
    clips = {c["rgb_path"]: c for c in vz.load_clips()}
    picks = [rows[0], rows[1], next(r for r in rows if r["day_night"] == "night" and r["window"] == "B")]
    L = [f"Cache check through the real processor (no model); {vz.now()}"]
    for r in picks:
        c = clips[r["rgb_path"]]
        idx = json.loads(r["frame_indices"])
        fi, cached = read_npz(REPO_ROOT / r["frames_path"])
        dec = vz.stack(vz.decode(c, idx), idx)  # 452 x 254, as the narrow call
        for name, frames, ids in (("narrow v1 layout (pairs)", dec, idx),
                                  ("narrow v2 layout (duplicated)", np.repeat(dec, 2, axis=0), [i for i in idx for _ in range(2)])):
            inputs, _, _ = vz.smoke.build_inputs(proc, frames, ids, c["fps"], c, prompt_text_in=n2.PROMPT_V2)
            mv = vz.model_view(inputs, proc)
            mv = mv[::2] if "duplicated" in name else mv[:len(idx)]
            d = np.abs(mv.astype(int) - cached.astype(int))
            L.append(f"  {r['stem']} window {r['window']}, {name}: model input {mv.shape[2]} x {mv.shape[1]}; cached vs "
                     f"model input |diff| mean {d.mean():.2f}, 99th pct {np.percentile(d, 99):.0f}, max {d.max()} (0-255; "
                     f"JPEG q{JPEG_Q})")
        unc = model_frames(vz.decode(c, idx), idx)
        inputs, _, _ = vz.smoke.build_inputs(proc, dec, idx, c["fps"], c, prompt_text_in=n2.PROMPT_V2)
        mv = vz.model_view(inputs, proc)[:len(idx)]
        L.append(f"    PIL bicubic 448 x 256 (before JPEG) vs model input: max |diff| {np.abs(mv.astype(int) - unc.astype(int)).max()}")
        i2, _, _ = vz.smoke.build_inputs(proc, cached, idx, c["fps"], c, prompt_text_in=n2.PROMPT_V2)
        L.append(f"    cached 448 x 256 frames given to the processor: grid_thw {i2['video_grid_thw'][0].tolist()} "
                 f"(16 x 28 patches = 448 x 256, i.e. not resized again); frame indices match: {fi == idx}")
    text = "\n".join(L) + "\n"
    (OUT / "cache_verify.txt").write_text(text, encoding="utf-8")
    print(text)


# ---------------------------------------------------------------- 4. report

def pct(a, qs=(0, 5, 25, 50, 75, 95, 100)):
    return ", ".join(f"p{q} {np.percentile(a, q):.3f}" for q in qs)


def report():
    from PIL import Image, ImageDraw

    rows = vz.read_csv(MANIFEST)
    clips = [r for r in rows if r["window"] == "A"]
    L = [f"QLoRA training data (synthetic only; no model, no training); report {vz.now()}",
         f"manifest {MANIFEST.relative_to(REPO_ROOT)} sha256 {vz.sha256(MANIFEST)}: {len(rows)} windows = {len(clips)} "
         f"clips x 2 (A: centre t_hat C_cctv seed 0 OOF; B: t* + U[-1, 1] s, seed {B_SEED}); {N_FRAMES} frames over "
         f"centre +- {HALF_S:g} s, shifted inside the clip; frames {MODEL_W} x {MODEL_H}", ""]
    L.append("1. COUNTS (clips; each clip gives windows A and B)")
    L.append(f"  per fold: {dict(sorted(collections.Counter(r['fold'] for r in clips).items()))}")
    L.append(f"  {'type':10s} {'day':>5s} {'night':>6s} {'total':>6s}   per fold (0 / 1 / 2)")
    for t in vz.TYPES:
        cc = [r for r in clips if r["type"] == t]
        dn = collections.Counter(r["day_night"] for r in cc)
        fo = collections.Counter(r["fold"] for r in cc)
        L.append(f"  {t:10s} {dn['day']:5d} {dn['night']:6d} {len(cc):6d}   {fo['0']} / {fo['1']} / {fo['2']}")
    sg = [r for r in clips if r["type"] == "single"]
    L.append(f"  single has few clips: {len(sg)} of {len(clips)} ({len(sg) / len(clips):.1%}), "
             f"{sum(r['day_night'] == 'night' for r in sg)} at night; fold counts "
             f"{dict(sorted(collections.Counter(r['fold'] for r in sg).items()))} (none in fold 2).")
    L.append(f"  flags: design clips {sum(r['design_clip'] == 'True' for r in clips)}, narrow-test clips "
             f"{sum(r['narrow_test_clip'] == 'True' for r in clips)} (overlap "
             f"{sum(r['design_clip'] == 'True' and r['narrow_test_clip'] == 'True' for r in clips)})")
    L.append("")
    L.append("2. t* INSIDE THE WINDOW (first to last frame time)")
    for w in ("A", "B"):
        ww = [r for r in rows if r["window"] == w]
        ins = [r["t_star_inside"] == "True" for r in ww]
        near = np.array([float(r["t_star_to_nearest_frame_s"]) for r in ww])
        dist = np.array([float(r["t_star"]) - (float(r["win_start_s"]) + float(r["win_end_s"])) / 2 for r in ww])
        L.append(f"  window {w}: t* inside {sum(ins)}/{len(ww)} = {np.mean(ins):.4f}; t* - window middle: {pct(dist)}; "
                 f"t* to nearest sampled frame: median {np.median(near):.3f} s, max {near.max():.3f} s")
        for t in vz.TYPES:
            tw = [r["t_star_inside"] == "True" for r in ww if r["type"] == t]
            L.append(f"    {t:10s} {np.mean(tw):.4f}")
    off = np.array([float(r["offset_s"]) for r in rows if r["window"] == "B"])
    L.append(f"  window B offsets: {pct(off)}")
    L.append("")
    L.append("3. BOX SIZE (collision box; fractions of the image and pixels of the 448 x 256 model frame)")
    bw = np.array([(float(r["x2_1000"]) - float(r["x1_1000"])) / 1000 for r in clips])
    bh = np.array([(float(r["y2_1000"]) - float(r["y1_1000"])) / 1000 for r in clips])
    L.append(f"  width  (fraction): {pct(bw)}")
    L.append(f"  height (fraction): {pct(bh)}")
    L.append(f"  width  (px of 448): {pct(bw * MODEL_W)}")
    L.append(f"  height (px of 256): {pct(bh * MODEL_H)}")
    L.append(f"  area (fraction of the image): {pct(bw * bh)}")
    L.append(f"  boxes under 10 x 10 px in the model frame: {int(((bw * MODEL_W < 10) | (bh * MODEL_H < 10)).sum())}")
    L.append(f"  mean width / height (fraction): {bw.mean():.4f} / {bh.mean():.4f}")
    for t in vz.TYPES:
        m = np.array([r["type"] == t for r in clips])
        L.append(f"    {t:10s} median width {np.median(bw[m]):.3f}, height {np.median(bh[m]):.3f}")
    L.append("")
    s = json.loads((OUT / "cache_summary.json").read_text(encoding="utf-8"))
    files = list(CACHE.glob("*.npz"))
    missing = [r["frames_path"] for r in rows if not (REPO_ROOT / r["frames_path"]).is_file()]
    L.append("4. FRAME CACHE")
    L.append(f"  {CACHE.relative_to(REPO_ROOT)}: {len(files)} window files ({len(missing)} missing), "
             f"{sum(f.stat().st_size for f in files) / 2 ** 30:.3f} GB; JPEG q{JPEG_Q}, {MODEL_W} x {MODEL_H}; last run "
             f"{s['wall_s_this_run']:.0f} s with {s['workers']} workers, status {s['status_this_run']}, errors "
             f"{len(s['errors'])} (cache_log.txt has every run)")
    L.append("")
    picks = []
    for t, dn in (("head-on", "day"), ("rear-end", "night"), ("sideswipe", "day"), ("t-bone", "night")):
        picks.append(next(r for r in rows if r["window"] == "B" and r["type"] == t and r["day_night"] == dn
                          and r["design_clip"] == "False" and r["narrow_test_clip"] == "False"))
    tiles = []
    L.append("5. GRID (grid_examples.png): window B, the cached frame nearest t*, box (red) and point (yellow) from "
             "the labels")
    for r in picks:
        fi, fr = read_npz(REPO_ROOT / r["frames_path"])
        ft = np.array(fi) / float(r["fps"])
        k = int(np.argmin(np.abs(ft - float(r["t_star"]))))
        img = Image.fromarray(fr[k]).convert("RGB")
        d = ImageDraw.Draw(img)
        x1, y1, x2, y2 = (float(r[f"{a}_1000"]) / 1000 for a in ("x1", "y1", "x2", "y2"))
        d.rectangle([x1 * MODEL_W, y1 * MODEL_H, x2 * MODEL_W, y2 * MODEL_H], outline=(255, 0, 0), width=2)
        cx, cy = float(r["center_x"]) * MODEL_W, float(r["center_y"]) * MODEL_H
        d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(255, 255, 0))
        d.rectangle([0, 0, MODEL_W, 14], fill=(0, 0, 0))
        d.text((3, 1), f"{r['stem']} | frame t {ft[k]:.2f} s, t* {float(r['t_star']):.2f} s", fill=(255, 255, 255))
        tiles.append(np.asarray(img))
        L.append(f"  {r['stem']} ({r['type']}, {r['day_night']}): frame {fi[k]} at {ft[k]:.2f} s, t* "
                 f"{float(r['t_star']):.2f} s; box ({x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f})")
    vz.save_grid(np.stack(tiles), OUT / "grid_examples.png", cols=2)
    text = "\n".join(L) + "\n"
    (OUT / "report.txt").write_text(text, encoding="utf-8")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    for m in ("audit", "manifest", "cache", "verify", "report"):
        g.add_argument(f"--{m}", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")
    if args.audit:
        audit()
    elif args.manifest:
        manifest()
    elif args.cache:
        cache(args.workers)
    elif args.verify:
        verify()
    else:
        report()


if __name__ == "__main__":
    main()
