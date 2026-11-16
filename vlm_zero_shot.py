"""Zero-shot VLM protocols (Qwen3-VL-4B-Instruct, 4-bit NF4): full scan (time) and narrow call (type, point).

SYNTHETIC ONLY in this version: reads ACCIDENT/dataset/metadata-synthetic.csv, synthetic_split.csv, the saved
out-of-fold classifier predictions (narrow-window centre t_hat) and synthetic videos. Every clip path must start with
synthetic_videos/; no real clip, real label or real_test_set.csv is opened.

--make-dev-list (CPU): writes vlm_dev_clips.csv once (refuses to overwrite) = the prompt-development subset:
  30 clips drawn with numpy default_rng(DEV_SEED), DEV_PER_STRATUM per stratum (collision type x day/night;
  night = weather "night", day = every other weather), each stratum's pool sorted by rgb_path, the two stage-test
  clips excluded from the draw and then added (32 clips).

--run (GPU): on the clips of --clips, the full-scan protocol of each prompt version in --versions and the narrow
  call; writes --out-dir (PROTOCOL.txt before the first model call, per-clip CSVs, raw outputs, summary.txt).

Full scan (time only), version v:
  coarse  the whole clip at COARSE_FPS (t_k = k / COARSE_FPS), at most MAX_FRAMES frames (longer clips: MAX_FRAMES
          frames evenly from 0 to the last frame time); answer -> t_c.
  fine    MAX_FRAMES frames evenly over t_c +- FINE_HALF_S (shifted to stay inside the clip; a clip shorter than the
          window: the whole clip); answer -> t_fine.
  v1      the frame number k (0 .. n-1) drawn top-left on each frame; answer {"frame": k}; time = frame index / fps.
  v2      no drawn number; the frames' time comes from Qwen's own timestamp text, which the processor puts before
          every 2-frame token group ("<T.T seconds>", the mean time of the two frames; one per 2 frames, not per
          frame, because Qwen3-VL merges 2 consecutive frames into one group); answer {"time": seconds}.
  Both versions also carry the native timestamps (the processor always adds them).
  Valid answer: JSON (a ``` fence allowed) with the key; an integer (v1) / number (v2), or a string of digits
  (v1: "12"; v2: "12" or "12.5"), counted and flagged; v1 in 0 .. n-1, v2 in [first - 0.1, last + 0.1] s of the
  sampled frame times.
  Rule A (Saha et al., arXiv 2608.08867): an answer at t < MIN_T_S is invalid.
  Fallbacks: invalid coarse answer -> t_full = duration / 2, no fine pass; invalid fine answer -> t_full = t_c.
  Rule B (gate, Huang, arXiv 2605.01512): a valid fine answer at the first or last frame of the fine window (v1:
  frame 0 or n-1; v2: the time nearest to the first or last timestamp group) -> t_full = t_c (gated).
  Otherwise t_full = t_fine.
Narrow call (type and point; time stays t_hat): NARROW_FRAMES frames evenly over t_hat +- NARROW_HALF_S (shifted
  inside the clip), no drawn number, prompt NARROW_PROMPT; invalid -> type FALLBACK_TYPE, point FALLBACK_POINT.
  The narrow prompt has no frame numbers, so it is the same in versions 1 and 2 and runs once.
Frames: full frame (no crop), resized to a pixel budget of PIXEL_BUDGET (448 x 256 area, aspect kept, no upscaling),
  PIL bicubic; greedy decoding, MAX_NEW_TOKENS. GPU seconds = synchronised generate time (CPU decoding and the
  processor not included).

Usage (from cctv-accident-ITS-system/):
    .venv-vlm\\Scripts\\python.exe vlm_zero_shot.py --make-dev-list
    .venv-vlm\\Scripts\\python.exe -u vlm_zero_shot.py --run --clips vlm_dev_clips.csv --versions 1 2 \\
        --out-dir results\\vlm_dev
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "gpu_check"))
import vlm_smoke_test as smoke  # noqa: E402  (frame labels, prompt inputs, model loading)


def temporal(t_hat, t_star, sigma_t):
    """T@sigma, the same formula as accs_scorer.temporal (mean of exp(-err^2 / 2 sigma^2)); accs_scorer itself imports
    pandas, which .venv-vlm does not have (no dependency added)."""
    e = np.asarray(t_hat, dtype=float) - np.asarray(t_star, dtype=float)
    return float(np.mean(np.exp(-(e ** 2) / (2.0 * sigma_t ** 2))))

DATASET = REPO_ROOT / "ACCIDENT" / "dataset"
META_SYN = DATASET / "metadata-synthetic.csv"
SPLIT_CSV = REPO_ROOT / "synthetic_split.csv"
DEV_CSV = REPO_ROOT / "vlm_dev_clips.csv"
CLASSIFIER_PRED = REPO_ROOT / "results" / "diagnostics" / "C_cctv_seeds012" / "seed0_frozen" / "tta_clip_predictions_flip.csv"
DEV_SEED, DEV_PER_STRATUM = 0, 3
STAGE_TEST_CLIPS = smoke.STAGE_CLIPS

MODEL_KEY, PRECISION = "4B", "nf4"
MAX_FRAMES = 32
PIXEL_BUDGET = 448 * 256
COARSE_FPS = 1.0
FINE_HALF_S = 4.0
NARROW_HALF_S, NARROW_FRAMES = 2.0, 16
MIN_T_S = 0.3
MAX_NEW_TOKENS = 64
FALLBACK_TYPE, FALLBACK_POINT = "rear-end", (0.5, 0.5)  # most frequent synthetic type (794 of 2211); frame centre
TYPES = smoke.TYPES

FULL_PROMPT = {
    1: ("These {n} frames come from a fixed traffic camera, in time order, one every {dt} seconds. "
        "Each frame shows its number in the top-left corner, from 0 to {last}. "
        "A traffic collision happens in this video. Find the first frame in which the vehicles touch "
        "(for a single-vehicle crash: the first frame in which the vehicle hits something). "
        "Answer with one JSON object and nothing else, in this exact form: "
        '{{"frame": <integer from 0 to {last}>}}.'),
    2: ("These {n} frames come from a fixed traffic camera, in time order, one every {dt} seconds. "
        "The text before the frames gives their time in seconds from the start of the video. "
        "A traffic collision happens in this video. Find the first moment at which the vehicles touch "
        "(for a single-vehicle crash: the first moment at which the vehicle hits something). "
        "Answer with one JSON object and nothing else, in this exact form: "
        '{{"time": <time in seconds, a number from {t0} to {t1}>}}.'),
}
FULL_PROMPT[3] = FULL_PROMPT[2]  # v3 = v2 except the input layout (one frame per temporal token group)
PER_FRAME_GROUP = {3}  # versions whose sampled frames are each duplicated, so a 2-frame token group holds one frame
V3_FRAMES = 16  # distinct frames per pass in v3 (32 frame slots, as in v1 / v2)
NARROW_PROMPT = (
    "These 16 frames come from a fixed traffic camera, in time order, and cover about 4 seconds around a traffic "
    "collision. Answer with one JSON object and nothing else, in this exact form: "
    '{"collision_type": "<one of head-on, rear-end, sideswipe, single, t-bone>", "x": <number between 0 and 1>, '
    '"y": <number between 0 and 1>}. '
    "Collision types: head-on = the fronts of two vehicles hit each other; rear-end = the front of one vehicle hits "
    "the back of another; sideswipe = the sides of two vehicles scrape along each other; t-bone = the front of one "
    "vehicle hits the side of another; single = only one vehicle is involved (for example it hits a barrier or a "
    "pole, or leaves the road). x and y give the point of the collision in the image as fractions of the image width "
    "(x, from the left) and height (y, from the top)."
)


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stamp(msg):
    print(f"{now()} {msg}", flush=True)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def norm(p):
    return p.replace("\\", "/")


def synthetic_meta():
    rows = {norm(r["rgb_path"]): r for r in read_csv(META_SYN)}
    if len(rows) != 2211 or not all(p.startswith("synthetic_videos/") for p in rows):
        fail(f"{META_SYN.name}: expected 2211 synthetic clips")
    return rows


# ---------------------------------------------------------------- development subset

def make_dev_list():
    if DEV_CSV.exists():
        fail(f"{DEV_CSV.name} exists; the development subset is drawn once")
    meta = synthetic_meta()
    split = {norm(r["rgb_path"]): r for r in read_csv(SPLIT_CSV)}
    rng = np.random.default_rng(DEV_SEED)
    rows = []
    for typ in TYPES:
        for dn in ("day", "night"):
            pool = sorted(p for p, r in meta.items() if r["type"] == typ
                          and (r["weather"] == "night") == (dn == "night") and p not in STAGE_TEST_CLIPS)
            if len(pool) < DEV_PER_STRATUM:
                fail(f"stratum {typ} / {dn} has {len(pool)} clips")
            for p in rng.choice(pool, size=DEV_PER_STRATUM, replace=False):
                rows.append((p, typ, dn, "stratified draw"))
    for p in STAGE_TEST_CLIPS:
        r = meta[p]
        rows.append((p, r["type"], "night" if r["weather"] == "night" else "day", "stage-test clip"))
    with open(DEV_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rgb_path", "type", "day_night", "weather", "val_fold", "source"])
        for p, typ, dn, src in rows:
            w.writerow([p, typ, dn, meta[p]["weather"], split[p]["val_fold"], src])
    print(f"wrote {DEV_CSV} ({len(rows)} clips; seed {DEV_SEED}, {DEV_PER_STRATUM} per type x day/night stratum + "
          f"{len(STAGE_TEST_CLIPS)} stage-test clips), sha256 {sha256(DEV_CSV)}")


# ---------------------------------------------------------------- frames

def load_clips(clips_csv=None):
    """The clips of clips_csv; clips_csv None: all 2211 synthetic clips in metadata order."""
    meta = synthetic_meta()
    pred = {norm(r["rgb_path"]): r for r in read_csv(CLASSIFIER_PRED)}
    split = {norm(r["rgb_path"]): r for r in read_csv(SPLIT_CSV)}
    out = []
    rows = read_csv(clips_csv) if clips_csv is not None else [
        {"rgb_path": p, "day_night": "night" if m["weather"] == "night" else "day"} for p, m in meta.items()]
    for r in rows:
        p = norm(r["rgb_path"])
        if not p.startswith("synthetic_videos/") or p not in meta:
            fail(f"{p}: not a synthetic clip of {META_SYN.name}")
        m = meta[p]
        if int(pred[p]["fold"]) != int(split[p]["val_fold"]):
            fail(f"{p}: the classifier prediction is not out-of-fold")
        nf, dur = int(m["no_frames"]), float(m["duration"])
        out.append({"rgb_path": p, "video": str(DATASET / p), "type": m["type"], "weather": m["weather"],
                    "day_night": r.get("day_night", ""), "no_frames": nf, "duration": dur, "fps": nf / dur,
                    "t_star": float(m["accident_time"]), "center_x": float(m["center_x"]),
                    "center_y": float(m["center_y"]), "t_hat": float(pred[p]["t_hat"]),
                    "width": int(m["width"]), "height": int(m["height"]), "val_fold": int(split[p]["val_fold"])})
    return out


def last_time(clip):
    return (clip["no_frames"] - 1) / clip["fps"]


def to_indices(clip, times):
    return np.clip(np.round(np.asarray(times) * clip["fps"]).astype(int), 0, clip["no_frames"] - 1).tolist()


def coarse_times(clip):
    t_last = last_time(clip)
    n = int(math.floor(t_last * COARSE_FPS + 1e-9)) + 1
    return np.linspace(0.0, t_last, MAX_FRAMES) if n > MAX_FRAMES else np.arange(n) / COARSE_FPS


def window_times(clip, centre, half, n):
    t_last = last_time(clip)
    if t_last <= 2 * half:
        return np.linspace(0.0, t_last, n)
    lo = min(max(centre - half, 0.0), t_last - 2 * half)
    return np.linspace(lo, lo + 2 * half, n)


def decode(clip, indices, budget=PIXEL_BUDGET):
    """Frames (full frame, no crop) resized to the pixel budget (aspect kept, no upscaling), by frame index."""
    import av
    from PIL import Image

    wanted, got = set(indices), {}
    with av.open(clip["video"]) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                img = frame.to_image().convert("RGB")
                s = min(1.0, math.sqrt(budget / (img.width * img.height)))
                if s < 1.0:
                    img = img.resize((round(img.width * s), round(img.height * s)), Image.BICUBIC)
                got[i] = img
            if i >= max(wanted):
                break
    missing = sorted(wanted - set(got))
    if missing:
        fail(f"{clip['rgb_path']}: could not decode frames {missing[:5]}")
    return got


def stack(got, indices, labels=None):
    out = []
    for j, i in enumerate(indices):
        img = got[i].copy()
        if labels is not None:
            img = smoke.draw_label(img, labels[j])
        out.append(np.asarray(img))
    return np.stack(out)


# ---------------------------------------------------------------- answers

def parse_json(text):
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_full(text, version, times):
    """(answer time or None, value, digit_string flag, reason if invalid, edge flag)."""
    obj = parse_json(text)
    key = "frame" if version == 1 else "time"
    if obj is None:
        return None, None, False, "no JSON object", False
    if key not in obj:
        return None, None, False, f"no key {key}", False
    v, digit = obj[key], False
    if isinstance(v, str):
        if (version == 1 and re.fullmatch(r"\d+", v.strip())) or (version != 1 and re.fullmatch(r"\d+(\.\d+)?", v.strip())):
            digit = True
            v = int(v) if version == 1 else float(v)
        else:
            return None, obj[key], False, "not a number", False
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None, obj[key], digit, "not a number", False
    n = len(times)
    if version == 1:
        if not float(v).is_integer() or not 0 <= int(v) < n:
            return None, v, digit, "frame out of range", False
        k = int(v)
        t, edge = float(times[k]), k in (0, n - 1)
    else:
        if not times[0] - 0.1 <= v <= times[-1] + 0.1:
            return None, v, digit, "time out of range", False
        t = float(v)
        # timestamp groups the model saw: v2 one per 2 frames (pair mean); v3 one per frame (each frame duplicated)
        groups = np.asarray(times, dtype=float) if version in PER_FRAME_GROUP else group_times(times)
        g = int(np.argmin(np.abs(groups - t)))
        edge = g in (0, len(groups) - 1)
    if t < MIN_T_S:
        return None, v, digit, f"rule A: t < {MIN_T_S} s", edge
    return t, v, digit, "", edge


def group_times(times):
    """Qwen's per-group timestamps: the mean time of each 2-frame group (an odd count repeats the last frame)."""
    t = list(times) + ([times[-1]] if len(times) % 2 else [])
    return np.array([(t[i] + t[i + 1]) / 2 for i in range(0, len(t), 2)])


# ---------------------------------------------------------------- model calls

class Vlm:
    def __init__(self):
        import torch
        from transformers import AutoProcessor

        self.torch = torch
        if not torch.cuda.is_available():
            fail("CUDA is not available")
        self.device = torch.device("cuda", 0)
        stamp(f"loading {smoke.MODELS[MODEL_KEY]} ({PRECISION})")
        t0 = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(smoke.MODELS[MODEL_KEY])
        self.model = smoke.load_model({"model": MODEL_KEY, "precision": PRECISION})
        self.model.eval()
        torch.cuda.synchronize()
        self.load_s = time.perf_counter() - t0
        self.gpu = torch.cuda.get_device_name(0)
        stamp(f"model loaded in {self.load_s:.1f} s on {self.gpu}")

    def __call__(self, clip, frames, indices, prompt):
        torch = self.torch
        inputs, _, _ = smoke.build_inputs(self.processor, frames, indices, clip["fps"], clip, prompt_text_in=prompt)
        n_in = inputs["input_ids"].shape[1]
        inputs = inputs.to(self.device)
        inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(torch.bfloat16)
        self.last_visual_tokens = int((inputs["input_ids"][0] == self.processor.video_token_id).sum())
        self.last_grid_thw = inputs["video_grid_thw"][0].tolist()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(self.device)  # peak memory of this call (weights included)
        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                          temperature=None, top_p=None, top_k=None)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            return f"OOM: {str(e).splitlines()[0]}", time.perf_counter() - t0
        torch.cuda.synchronize()
        gen_s = time.perf_counter() - t0
        self.last_peak_alloc_gb = torch.cuda.max_memory_allocated(self.device) / 2 ** 30
        self.last_peak_reserved_gb = torch.cuda.max_memory_reserved(self.device) / 2 ** 30
        text = self.processor.batch_decode(out[0, n_in:][None], skip_special_tokens=True)[0]
        return text, gen_s


def full_prompt(version, clip, times):
    n = len(times)
    dt = (times[-1] - times[0]) / (n - 1) if n > 1 else 0.0
    return FULL_PROMPT[version].format(n=n, dt=f"{dt:.2f}", last=n - 1, t0=f"{times[0]:.1f}", t1=f"{times[-1]:.1f}")


def run(clips_csv, versions, out_dir, previous_dir=None):
    clips = load_clips(clips_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "summary.txt").exists():
        fail(f"{out_dir / 'summary.txt'} exists; refusing to overwrite a finished run")
    protocol = [
        f"Zero-shot VLM development run; written {now()} BEFORE the first model call.",
        f"clips: {clips_csv} (sha256 {sha256(clips_csv)}), {len(clips)} synthetic clips; no real clip or label opened",
        f"model: {smoke.MODELS[MODEL_KEY]}, {PRECISION} (NF4, double quant, bf16 compute); greedy, max_new_tokens "
        f"{MAX_NEW_TOKENS}; frames full frame resized to {PIXEL_BUDGET} pixels (448 x 256 area, aspect kept), bicubic",
        f"full scan: coarse {COARSE_FPS:g} fps over the whole clip, at most {MAX_FRAMES} frames; fine {MAX_FRAMES} frames "
        f"over t_c +- {FINE_HALF_S:g} s (shifted inside the clip); versions {versions}",
        "v1: frame number drawn top-left (0 .. n-1), answer {\"frame\": k}, time = frame index / fps",
        "v2: no drawn number; Qwen's native timestamp text '<T.T seconds>' before every 2-frame token group (mean of "
        "the two frame times; Qwen3-VL merges 2 frames per group, so one timestamp per 2 frames, not per frame: "
        "per-frame timestamps would need one group per frame = 2x the tokens, beyond the tested 32-frame memory); "
        "answer {\"time\": seconds}. v1 inputs carry the same native timestamps.",
        f"valid answers: integer / number or a string of digits (counted, flagged); range v1 0..n-1, v2 [first-0.1, "
        f"last+0.1] s; rule A: t < {MIN_T_S} s invalid; coarse invalid -> t_full = duration / 2, no fine pass; fine "
        "invalid -> t_c; rule B gate: fine answer at the first or last frame (v1) / nearest the first or last "
        "timestamp group (v2) -> t_c",
        f"narrow: {NARROW_FRAMES} frames over t_hat +- {NARROW_HALF_S:g} s (shifted inside), t_hat = {CLASSIFIER_PRED} "
        f"(C_cctv_seed0, frozen post-processing, out-of-fold); invalid -> {FALLBACK_TYPE}, {FALLBACK_POINT}; same "
        "prompt for v1 and v2 (no frame numbers), run once",
        "v3 (reserved slot; pre-registered change, written here before running): one frame per temporal token group "
        f"(each sampled frame duplicated), so each group carries one frame and one timestamp '<T.T seconds>' equal to "
        f"that frame's time; {V3_FRAMES} distinct frames per pass (coarse: evenly over the whole clip, from 0 to the "
        f"last frame time; fine: evenly over t_c +- {FINE_HALF_S:g} s, shifted inside the clip); everything else as "
        "in v2 (no drawn number, the v2 prompt with n = 16, answer {\"time\": seconds}, the same validity rules, rule A, "
        "fallbacks; rule B gate: fine answer nearest the first or last of the 16 frame times -> t_c)",
        "bug checks before v3 (results/vlm_dev/inspect/inspect.txt and grids): v2 timestamp text = true pair means "
        "(max 0.025 s, one decimal), v1 numbers readable at 448 x 256, narrow-call vehicles visible: no bug found",
        "selection criterion (pre-registered by the author): share of the 32 clips with |t_full - t*| <= 2 s; ties go "
        "to the simpler version",
        "", "prompts:", *[f" full v{v}: {FULL_PROMPT[v]}" for v in versions], f" narrow: {NARROW_PROMPT}"]
    (out_dir / "PROTOCOL.txt").write_text("\n".join(protocol) + "\n", encoding="utf-8")
    stamp(f"wrote {out_dir / 'PROTOCOL.txt'}")

    vlm = Vlm()
    full_rows = {v: [] for v in versions}
    narrow_rows = []
    raw = open(out_dir / "raw_outputs.jsonl", "w", encoding="utf-8")
    def call(v, clip, frames, idx, prompt):
        if v in PER_FRAME_GROUP:  # duplicate every frame: each 2-frame token group = one frame, timestamp = its time
            frames, idx = np.repeat(frames, 2, axis=0), [i for i in idx for _ in range(2)]
        return vlm(clip, frames, idx, prompt)

    def n_pass(v):
        return V3_FRAMES if v in PER_FRAME_GROUP else MAX_FRAMES

    for ci, clip in enumerate(clips, 1):
        stamp(f"clip {ci}/{len(clips)} {Path(clip['rgb_path']).stem}: coarse")
        c_idx = {v: to_indices(clip, np.linspace(0.0, last_time(clip), V3_FRAMES) if v in PER_FRAME_GROUP
                                else coarse_times(clip)) for v in versions}
        got = decode(clip, sorted(set().union(*c_idx.values())))
        res = {}
        for v in versions:
            ct = np.array(c_idx[v]) / clip["fps"]  # actual frame times
            frames = stack(got, c_idx[v], [str(k) for k in range(len(c_idx[v]))] if v == 1 else None)
            text, s = call(v, clip, frames, c_idx[v], full_prompt(v, clip, ct))
            raw.write(json.dumps({"clip": clip["rgb_path"], "version": v, "pass": "coarse", "output": text}) + "\n")
            t_c, val, digit, why, edge = parse_full(text, v, ct)
            res[v] = {"n_coarse": len(c_idx[v]), "coarse_raw": text, "coarse_value": val, "coarse_digit_string": digit,
                      "coarse_invalid": why, "coarse_edge": edge, "t_c": t_c, "gpu_s_coarse": s}
        # fine windows (one per version) and the narrow window, decoded together
        fine = {v: window_times(clip, res[v]["t_c"], FINE_HALF_S, n_pass(v)) for v in versions if res[v]["t_c"] is not None}
        fine_idx = {v: to_indices(clip, t) for v, t in fine.items()}
        n_idx = to_indices(clip, window_times(clip, clip["t_hat"], NARROW_HALF_S, NARROW_FRAMES))
        got = decode(clip, sorted(set(n_idx).union(*fine_idx.values())))
        stamp(f"clip {ci}/{len(clips)}: fine ({', '.join(f'v{v}' for v in fine_idx)}) and narrow")
        for v in versions:
            r = res[v]
            r.update(n_fine=0, fine_raw="", fine_value=None, fine_digit_string=False, fine_invalid="", fine_edge=False,
                     t_fine=None, gpu_s_fine=0.0, gate=False)
            if v not in fine_idx:
                r["t_full"], r["fallback"] = clip["duration"] / 2, "coarse"
            else:
                fi = fine_idx[v]
                ft = np.array(fi) / clip["fps"]
                frames = stack(got, fi, [str(k) for k in range(len(fi))] if v == 1 else None)
                text, s = call(v, clip, frames, fi, full_prompt(v, clip, ft))
                raw.write(json.dumps({"clip": clip["rgb_path"], "version": v, "pass": "fine", "output": text}) + "\n")
                t_f, val, digit, why, edge = parse_full(text, v, ft)
                r.update(n_fine=len(fi), fine_raw=text, fine_value=val, fine_digit_string=digit, fine_invalid=why,
                         fine_edge=edge, t_fine=t_f, gpu_s_fine=s, fine_window=f"{ft[0]:.2f}-{ft[-1]:.2f}")
                if t_f is None:
                    r["t_full"], r["fallback"] = r["t_c"], "fine"
                elif edge:
                    r["t_full"], r["fallback"], r["gate"] = r["t_c"], "", True
                else:
                    r["t_full"], r["fallback"] = t_f, ""
            r["error"] = r["t_full"] - clip["t_star"]
            full_rows[v].append({**{k: clip[k] for k in ("rgb_path", "type", "weather", "day_night", "duration",
                                                          "t_star", "t_hat")}, **r})
            stamp(f"  v{v}: t_c {r['t_c']}, t_fine {r['t_fine']}, t_full {r['t_full']:.2f}, fallback "
                  f"{r['fallback'] or '-'}, gate {r['gate']}")
        frames = stack(got, n_idx)
        text, s = vlm(clip, frames, n_idx, NARROW_PROMPT)
        raw.write(json.dumps({"clip": clip["rgb_path"], "pass": "narrow", "output": text}) + "\n")
        raw.flush()
        p = smoke.parse_output(text)
        ok = bool(p["values_ok"])
        typ = p["parsed"]["collision_type"] if ok else FALLBACK_TYPE
        x, y = (float(p["parsed"]["x"]), float(p["parsed"]["y"])) if ok else FALLBACK_POINT
        narrow_rows.append({**{k: clip[k] for k in ("rgb_path", "type", "weather", "day_night", "t_star", "t_hat",
                                                     "center_x", "center_y")},
                            "narrow_raw": text, "narrow_valid": ok, "pred_type": typ, "type_correct": typ == clip["type"],
                            "x": x, "y": y, "point_error": math.hypot(x - clip["center_x"], y - clip["center_y"]),
                            "gpu_s_narrow": s})
    raw.close()
    summarise(clips, versions, full_rows, narrow_rows, vlm, out_dir, previous_dir)


# ---------------------------------------------------------------- report

def write_rows(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def f(v, fmt="{:.2f}"):
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else fmt.format(v)


def summarise(clips, versions, full_rows, narrow_rows, vlm, out_dir, previous_dir=None):
    n = len(clips)
    L = [f"Zero-shot VLM development run on {n} synthetic clips ({out_dir}); {now()}; {smoke.MODELS[MODEL_KEY]} "
         f"{PRECISION} on {vlm.gpu}, load {vlm.load_s:.1f} s. Protocol: PROTOCOL.txt. Synthetic labels only.", ""]
    tot = []
    for v in versions:
        rows = full_rows[v]
        write_rows(out_dir / f"full_scan_v{v}.csv", rows)
        err = np.array([r["error"] for r in rows])
        tf = np.array([r["t_full"] for r in rows])
        ts = np.array([r["t_star"] for r in rows])
        d = np.array([r["duration"] for r in rows])
        tot.append({
            "version": f"v{v}", "n_clips": n,
            "within_2s": float(np.mean(np.abs(err) <= 2)),
            "T@1": temporal(tf, ts, 1.0),
            "coarse at first/last frame": sum(r["coarse_edge"] for r in rows) / n,
            "t_full within 10% of midpoint": float(np.mean(np.abs(tf - d / 2) <= 0.1 * d)),
            "coarse invalid (parse/range)": sum(r["coarse_invalid"] not in ("", f"rule A: t < {MIN_T_S} s") for r in rows),
            "coarse rule A": sum(r["coarse_invalid"].startswith("rule A") for r in rows),
            "fine invalid (parse/range)": sum(r["fine_invalid"] not in ("", f"rule A: t < {MIN_T_S} s") for r in rows),
            "fine rule A": sum(r["fine_invalid"].startswith("rule A") for r in rows),
            "coarse fallbacks": sum(r["fallback"] == "coarse" for r in rows),
            "fine fallbacks": sum(r["fallback"] == "fine" for r in rows),
            "gated (rule B)": sum(r["gate"] for r in rows),
            "digit strings": sum(r["coarse_digit_string"] + r["fine_digit_string"] for r in rows),
            "GPU s per clip (full scan)": float(np.mean([r["gpu_s_coarse"] + r["gpu_s_fine"] for r in rows])),
            "median |error| s": float(np.median(np.abs(err))),
        })
        L += [f"FULL SCAN v{v} per clip (t*, coarse answer -> t_c, fine answer -> t_fine, t_full, error = t_full - t*)",
              f"{'clip':34s} {'type':9s} {'dur':>5s} {'t*':>6s} {'coarse':>7s} {'t_c':>6s} {'fine':>7s} {'t_fine':>6s} "
              f"{'t_full':>6s} {'error':>7s} flags"]
        for r in rows:
            flags = [x for x in (f"fallback {r['fallback']}" if r["fallback"] else "", "GATE" if r["gate"] else "",
                                 "coarse edge" if r["coarse_edge"] else "",
                                 "digit string" if r["coarse_digit_string"] or r["fine_digit_string"] else "",
                                 f"coarse: {r['coarse_invalid']}" if r["coarse_invalid"] else "",
                                 f"fine: {r['fine_invalid']}" if r["fine_invalid"] else "") if x]
            L.append(f"{Path(r['rgb_path']).stem:34s} {r['type']:9s} {r['duration']:5.1f} {r['t_star']:6.2f} "
                     f"{str(r['coarse_value']):>7s} {f(r['t_c']):>6s} {str(r['fine_value']):>7s} {f(r['t_fine']):>6s} "
                     f"{r['t_full']:6.2f} {r['error']:+7.2f} {'; '.join(flags)}")
        L.append("")
    write_rows(out_dir / "narrow.csv", narrow_rows)
    write_rows(out_dir / "totals.csv", tot)
    L += [f"TOTALS per version (n = {n}; selection criterion = within_2s, ties to the simpler version)"]
    keys = list(tot[0])
    w = max(len(k) for k in keys)
    L.append(f"  {'':{w}s} " + "  ".join(f"{t['version']:>8}" for t in tot))
    for k in keys[1:]:
        L.append(f"  {k:{w}s} " + "  ".join(
            f"{t[k]:>8.4f}" if isinstance(t[k], float) else f"{t[k]:>8}" for t in tot))
    nv = sum(r["narrow_valid"] for r in narrow_rows)
    pe = np.array([r["point_error"] for r in narrow_rows])
    L += ["", f"NARROW CALL (t_hat +- {NARROW_HALF_S:g} s; same input and prompt for v1 and v2, run once): valid "
          f"{nv}/{n}; type accuracy {np.mean([r['type_correct'] for r in narrow_rows]):.4f} "
          f"({sum(r['type_correct'] for r in narrow_rows)}/{n}, fallbacks counted as {FALLBACK_TYPE}); point error "
          f"(Euclidean, normalised image coordinates) mean {pe.mean():.4f}, median {np.median(pe):.4f}; GPU s per clip "
          f"{np.mean([r['gpu_s_narrow'] for r in narrow_rows]):.2f}",
          "  predicted types: " + ", ".join(f"{t} {sum(r['pred_type'] == t for r in narrow_rows)}" for t in TYPES),
          "  true types:      " + ", ".join(f"{t} {sum(r['type'] == t for r in narrow_rows)}" for t in TYPES),
          f"{'clip':34s} {'type':9s} {'pred':9s} {'x':>5s} {'y':>5s} {'cx':>5s} {'cy':>5s} {'err':>5s} valid"]
    for r in narrow_rows:
        L.append(f"{Path(r['rgb_path']).stem:34s} {r['type']:9s} {r['pred_type']:9s} {r['x']:5.2f} {r['y']:5.2f} "
                 f"{r['center_x']:5.2f} {r['center_y']:5.2f} {r['point_error']:5.3f} {r['narrow_valid']}")
    if previous_dir is not None:
        prev = read_csv(previous_dir / "totals.csv")
        allv = [(p["version"], float(p["within_2s"]), float(p["T@1"])) for p in prev] + \
               [(t["version"], t["within_2s"], t["T@1"]) for t in tot]
        best = max(w for _, w, _ in allv)
        top = [v for v, w, _ in allv if w == best]
        L += ["", f"SELECTION (pre-registered: share within 2 s on the {n} clips; ties go to the simpler version); "
                  f"{previous_dir / 'totals.csv'} for the earlier versions"]
        L += [f"  {v}: within_2s {w:.4f}, T@1 {t1:.4f}" for v, w, t1 in allv]
        L.append(f"  selected: {top[0]}" if len(top) == 1 else
                 f"  TIE between {', '.join(top)} at {best:.4f}: the tie rule (simpler version) applies; not decided here")
        pn = {norm(p["rgb_path"]): p["narrow_raw"] for p in read_csv(previous_dir / "narrow.csv")}
        same = sum(pn.get(r["rgb_path"]) == r["narrow_raw"] for r in narrow_rows)
        L.append(f"  narrow call re-run (same input and prompt): {same}/{n} raw outputs identical to {previous_dir / 'narrow.csv'}")
    text = "\n".join(L) + "\n"
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


# ---------------------------------------------------------------- --inspect (bug checks; CPU, processor only, no model)

def model_view(inputs, processor):
    """The frames exactly as the model receives them: pixel_values_videos un-patchified (inverse of
    Qwen3VLVideoProcessor.patchify) and de-normalised, uint8 (T, H, W, 3)."""
    vp = processor.video_processor
    gt, gh, gw = inputs["video_grid_thw"][0].tolist()
    ps, m, tp = vp.patch_size, vp.merge_size, vp.temporal_patch_size
    x = inputs["pixel_values_videos"].float().view(gt, gh // m, gw // m, m, m, 3, tp, ps, ps)
    x = x.permute(0, 6, 5, 1, 3, 7, 2, 4, 8).reshape(gt * tp, 3, gh * ps, gw * ps)
    mean, std = (np.array(v, dtype=np.float32).reshape(1, 3, 1, 1) for v in (vp.image_mean, vp.image_std))
    img = (x.numpy() * std + mean) * 255.0
    return np.clip(np.round(img), 0, 255).astype(np.uint8).transpose(0, 2, 3, 1)


def save_grid(frames, path, cols=4, gap=4):
    from PIL import Image

    t, h, w, _ = frames.shape
    rows = -(-t // cols)
    grid = np.full((rows * h + (rows - 1) * gap, cols * w + (cols - 1) * gap, 3), 255, np.uint8)
    for k, f in enumerate(frames):
        r, c = divmod(k, cols)
        grid[r * (h + gap):r * (h + gap) + h, c * (w + gap):c * (w + gap) + w] = f
    Image.fromarray(grid).save(path)


def inspect(out_dir, ts_clip, narrow_clips, tc_by_version):
    """1. v2 timestamp text vs true sampling times; 2. v1 coarse + fine inputs as a grid; 3. narrow inputs of
    narrow_clips as grids. Frames in the grids are the model's own input (model_view), at its resolution."""
    from transformers import AutoProcessor

    out_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(smoke.MODELS[MODEL_KEY])
    clips = {c["rgb_path"]: c for c in load_clips(DEV_CSV)}
    L = [f"Bug checks (no model; processor {smoke.MODELS[MODEL_KEY]}); {now()}"]

    def build(clip, frames, idx, prompt):
        inputs, _, text = smoke.build_inputs(processor, frames, idx, clip["fps"], clip, prompt_text_in=prompt)
        full = processor.tokenizer.decode(inputs["input_ids"][0])
        return inputs, full

    c = clips[ts_clip]
    L.append(f"\n1. VERSION 2 TIMESTAMPS: {ts_clip} (duration {c['duration']} s, no_frames {c['no_frames']}, "
             f"fps given to the processor = no_frames / duration = {c['fps']:.6f})")
    for name, times in (("coarse", coarse_times(c)),
                        (f"fine (t_c {tc_by_version[2]:.2f} s from the v2 run)",
                         window_times(c, tc_by_version[2], FINE_HALF_S, MAX_FRAMES))):
        idx = to_indices(c, times)
        st = np.array(idx) / c["fps"]
        got = decode(c, idx)
        inputs, text = build(c, stack(got, idx), idx, full_prompt(2, c, st))
        stamps = [float(s) for s in re.findall(r"<(\d+(?:\.\d+)?) seconds>", text)]
        groups = group_times(st)
        md = inputs.get("video_metadata")
        L.append(f"  {name}: {len(idx)} frames, grid_thw {inputs['video_grid_thw'][0].tolist()}, "
                 f"{len(stamps)} timestamp tokens for {len(groups)} 2-frame groups")
        L.append(f"  {'group':>5s} {'frame idx':>11s} {'true times s':>15s} {'true mean':>9s} {'prompt text':>14s} {'diff':>6s}")
        for g in range(max(len(stamps), len(groups))):
            pair = idx[2 * g:2 * g + 2] or [idx[-1]]
            tt = st[2 * g:2 * g + 2] if 2 * g < len(st) else st[-1:]
            s = stamps[g] if g < len(stamps) else float("nan")
            gm = groups[g] if g < len(groups) else float("nan")
            L.append(f"  {g:5d} {str(pair):>11s} {' / '.join(f'{v:.2f}' for v in tt):>15s} {gm:9.3f} "
                     f"{'<%.1f seconds>' % s:>14s} {s - gm:+6.2f}")
        diff = np.abs(np.array(stamps[:len(groups)]) - groups)
        L.append(f"  max |text - true pair mean| = {diff.max():.3f} s (text has one decimal); count match "
                 f"{len(stamps) == len(groups)}; frames_indices given = sampled indices: True by construction "
                 f"(metadata {md is not None and 'returned' or 'not returned by the processor'})")
        if name == "coarse":
            L.append(f"  first characters of the video part of the prompt: {text[text.find('<0'):text.find('<0') + 70]!r}")

    L.append(f"\n2. VERSION 1 FRAME NUMBERS: {ts_clip}, coarse and fine inputs as the model receives them")
    for name, times in (("coarse", coarse_times(c)),
                        ("fine", window_times(c, tc_by_version[1], FINE_HALF_S, MAX_FRAMES))):
        idx = to_indices(c, times)
        got = decode(c, idx)
        inputs, _ = build(c, stack(got, idx, [str(k) for k in range(len(idx))]), idx,
                          full_prompt(1, c, np.array(idx) / c["fps"]))
        mv = model_view(inputs, processor)[:len(idx)]
        p = out_dir / f"v1_{name}_{Path(ts_clip).stem}.png"
        save_grid(mv, p)
        L.append(f"  {name}: {len(idx)} frames, model resolution {mv.shape[2]} x {mv.shape[1]} (W x H), grid {p.name} "
                 f"(4 per row, numbers 0..{len(idx) - 1} in order)")

    L.append("\n3. NARROW CALL INPUTS (t_hat +- 2 s, 16 frames), as the model receives them")
    for cp in narrow_clips:
        c2 = clips[cp]
        idx = to_indices(c2, window_times(c2, c2["t_hat"], NARROW_HALF_S, NARROW_FRAMES))
        got = decode(c2, idx)
        inputs, _ = build(c2, stack(got, idx), idx, NARROW_PROMPT)
        mv = model_view(inputs, processor)[:len(idx)]
        p = out_dir / f"narrow_{Path(cp).stem}.png"
        save_grid(mv, p)
        L.append(f"  {Path(cp).stem} ({c2['day_night']}, {c2['weather']}): t_hat {c2['t_hat']:.2f} s, t* {c2['t_star']:.2f} s, "
                 f"frames {idx[0]}..{idx[-1]} ({idx[0] / c2['fps']:.2f}-{idx[-1] / c2['fps']:.2f} s), model resolution "
                 f"{mv.shape[2]} x {mv.shape[1]}, grid {p.name}; true point ({c2['center_x']:.2f}, {c2['center_y']:.2f})")
    text = "\n".join(L) + "\n"
    (out_dir / "inspect.txt").write_text(text, encoding="utf-8")
    print(text)


# ---------------------------------------------------------------- narrow v2 (addition to the plan, 2026-11-16)

NARROW2_DIR = REPO_ROOT / "results" / "vlm_dev" / "narrow_v2"
NARROW2_FRAMES, NARROW2_BUDGET = 8, 896 * 512
NARROW2_PROMPT = NARROW_PROMPT.replace("These 16 frames", "These 8 frames")
NARROW_V1_CORRECT = 4  # narrow v1 type accuracy on the 32 development clips (results/vlm_dev/narrow.csv): 4 of 32
NARROW_SPECS = {1: (NARROW_FRAMES, PIXEL_BUDGET, NARROW_PROMPT), 2: (NARROW2_FRAMES, NARROW2_BUDGET, NARROW2_PROMPT)}
NARROW2_CLIPS = ("synthetic_videos/videos/head-on/Town05_head-on_wet_24.mp4",
                 "synthetic_videos/videos/head-on/Town03_head-on_night_03.mp4")  # inspection: one day, one night


def narrow2_protocol():
    p = NARROW2_DIR / "PROTOCOL.txt"
    if p.exists():
        fail(f"{p} exists; the narrow v2 protocol is written once")
    if NARROW2_PROMPT == NARROW_PROMPT:
        fail("the narrow v2 prompt must differ from v1 only in the frame count")
    NARROW2_DIR.mkdir(parents=True, exist_ok=True)
    L = [f"Narrow call v2 (addition to the plan; the only narrow version besides v1); written {now()} BEFORE any model "
         "call (and before the CPU inspection).",
         f"clips: the 32 development clips, {DEV_CSV.name} sha256 {sha256(DEV_CSV)}; synthetic only, no real clip or label",
         f"narrow v2: {NARROW2_FRAMES} frames evenly over t_hat +- {NARROW_HALF_S:g} s (same window as v1, shifted inside "
         f"the clip), t_hat = {CLASSIFIER_PRED.relative_to(REPO_ROOT)} (C_cctv_seed0, frozen post-processing, "
         f"out-of-fold); each frame resized to a fixed pixel area of {NARROW2_BUDGET} (896 x 512, aspect kept, bicubic, "
         "no upscaling); frames paired as Qwen3-VL does natively (2 frames per temporal token group: 4 groups, about "
         "1,792 visual tokens)",
         f"prompt: narrow v1 text with '16 frames' replaced by '8 frames': {NARROW2_PROMPT}",
         f"unchanged: {smoke.MODELS[MODEL_KEY]} {PRECISION} (NF4, double quant, bf16 compute), greedy, max_new_tokens "
         f"{MAX_NEW_TOKENS}, JSON format and validity (collision_type in the 5 types, x and y in [0, 1]), fallback "
         f"{FALLBACK_TYPE} / {FALLBACK_POINT}",
         f"decision rule (fixed now): adopt narrow v2 if its type accuracy on the 32 clips is higher than narrow v1 "
         f"({NARROW_V1_CORRECT} of 32); a tie keeps narrow v1 (simpler). Point error is reported, does not decide.",
         "after this attempt the narrow call is frozen; no further zero-shot narrow versions.",
         "measurement only: narrow v1 is re-run in the same process (same input and prompt as before) to measure its "
         "GPU seconds and peak GPU memory on mains; its answers must equal results/vlm_dev/narrow.csv (checked); the "
         f"decision uses the recorded v1 result ({NARROW_V1_CORRECT} of 32).",
         "run on mains power only (checked and recorded at start and end)."]
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"wrote {p}")


def narrow_input(clip, version, budget=None):
    n, b, prompt = NARROW_SPECS[version]
    idx = to_indices(clip, window_times(clip, clip["t_hat"], NARROW_HALF_S, n))
    return idx, stack(decode(clip, idx, budget or b), idx), prompt


def narrow2_inspect():
    from transformers import AutoProcessor

    if not (NARROW2_DIR / "PROTOCOL.txt").exists():
        fail("write the narrow v2 PROTOCOL.txt first (--narrow2 protocol)")
    out = NARROW2_DIR / "inspect"
    out.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(smoke.MODELS[MODEL_KEY])
    clips = {c["rgb_path"]: c for c in load_clips(DEV_CSV)}
    L = [f"Narrow v2 CPU check through the real processor (no model); {now()}"]
    for cp in NARROW2_CLIPS:
        c = clips[cp]
        for v in (1, 2):
            idx, frames, prompt = narrow_input(c, v)
            inputs, _, _ = smoke.build_inputs(processor, frames, idx, c["fps"], c, prompt_text_in=prompt)
            ids = inputs["input_ids"][0]
            mv = model_view(inputs, processor)[:len(idx)]
            L.append(f"{Path(cp).stem} ({c['day_night']}) narrow v{v}: {len(idx)} frames {idx[0]}..{idx[-1]} "
                     f"({idx[0] / c['fps']:.2f}-{idx[-1] / c['fps']:.2f} s), resized input {frames.shape[2]} x "
                     f"{frames.shape[1]}, model receives {mv.shape[2]} x {mv.shape[1]} (W x H), grid_thw "
                     f"{inputs['video_grid_thw'][0].tolist()}, visual tokens {int((ids == processor.video_token_id).sum())}, "
                     f"input tokens {int(ids.numel())}")
            if v == 2:
                p = out / f"narrow_v2_{Path(cp).stem}.png"
                save_grid(mv, p)
                L.append(f"   grid: {p.name} (the {len(idx)} frames as the model receives them, 4 per row)")
    text = "\n".join(L) + "\n"
    (out / "inspect.txt").write_text(text, encoding="utf-8")
    print(text)


def narrow_answer(text):
    p = smoke.parse_output(text)
    ok = bool(p["values_ok"])
    typ = p["parsed"]["collision_type"] if ok else FALLBACK_TYPE
    x, y = (float(p["parsed"]["x"]), float(p["parsed"]["y"])) if ok else FALLBACK_POINT
    return ok, typ, x, y


def narrow2_run(clips_csv):
    if not (NARROW2_DIR / "PROTOCOL.txt").exists() or not (NARROW2_DIR / "inspect" / "inspect.txt").exists():
        fail("PROTOCOL.txt and the CPU inspection must exist before the run")
    if (NARROW2_DIR / "summary.txt").exists():
        fail(f"{NARROW2_DIR / 'summary.txt'} exists; refusing to overwrite a finished run")
    clips = load_clips(clips_csv)
    vlm = Vlm()
    rows = {1: [], 2: []}
    with open(NARROW2_DIR / "raw_outputs.jsonl", "w", encoding="utf-8") as raw:
        for ci, c in enumerate(clips, 1):
            stamp(f"clip {ci}/{len(clips)} {Path(c['rgb_path']).stem}")
            for v in (1, 2):
                idx, frames, prompt = narrow_input(c, v)
                text, s = vlm(c, frames, idx, prompt)
                raw.write(json.dumps({"clip": c["rgb_path"], "narrow_version": v, "output": text}) + "\n")
                ok, typ, x, y = narrow_answer(text)
                rows[v].append({"rgb_path": c["rgb_path"], "type": c["type"], "day_night": c["day_night"],
                                "t_star": c["t_star"], "t_hat": c["t_hat"], "center_x": c["center_x"],
                                "center_y": c["center_y"], "narrow_raw": text, "narrow_valid": ok, "pred_type": typ,
                                "type_correct": typ == c["type"], "x": x, "y": y,
                                "point_error": math.hypot(x - c["center_x"], y - c["center_y"]), "gpu_s": s,
                                "peak_alloc_gb": vlm.last_peak_alloc_gb, "peak_reserved_gb": vlm.last_peak_reserved_gb,
                                "visual_tokens": vlm.last_visual_tokens, "n_frames": len(idx)})
    write_rows(NARROW2_DIR / "narrow_v1_rerun.csv", rows[1])
    write_rows(NARROW2_DIR / "narrow_v2.csv", rows[2])
    n = len(clips)
    prev = {norm(p["rgb_path"]): p["narrow_raw"] for p in read_csv(REPO_ROOT / "results" / "vlm_dev" / "narrow.csv")}
    same = sum(prev.get(r["rgb_path"]) == r["narrow_raw"] for r in rows[1])
    L = [f"Narrow call v1 vs v2 on {n} synthetic development clips; {now()}; {smoke.MODELS[MODEL_KEY]} {PRECISION} on "
         f"{vlm.gpu}, load {vlm.load_s:.1f} s. Protocol: PROTOCOL.txt. Synthetic labels only.", "",
         f"PER CLIP (narrow v2; v1 = the re-run, {same}/{n} raw outputs identical to results/vlm_dev/narrow.csv)",
         f"{'clip':34s} {'true':9s} {'v1 pred':9s} {'v2 pred':9s} {'x':>5s} {'y':>5s} {'cx':>5s} {'cy':>5s} "
         f"{'v2 err':>6s} valid"]
    for r1, r2 in zip(rows[1], rows[2]):
        L.append(f"{Path(r2['rgb_path']).stem:34s} {r2['type']:9s} {r1['pred_type']:9s} {r2['pred_type']:9s} "
                 f"{r2['x']:5.2f} {r2['y']:5.2f} {r2['center_x']:5.2f} {r2['center_y']:5.2f} {r2['point_error']:6.3f} "
                 f"{r2['narrow_valid']}")
    tot = []
    for v in (1, 2):
        R = rows[v]
        pe = np.array([r["point_error"] for r in R])
        k = sum(r["type_correct"] for r in R)
        tot.append({"narrow": f"v{v}", "frames": R[0]["n_frames"], "visual tokens": R[0]["visual_tokens"],
                    "valid answers": f"{sum(r['narrow_valid'] for r in R)}/{n}", "type accuracy": f"{k / n:.4f} ({k}/{n})",
                    "point error mean": f"{pe.mean():.4f}", "point error median": f"{np.median(pe):.4f}",
                    "GPU s per clip": f"{np.mean([r['gpu_s'] for r in R]):.2f}",
                    "peak GPU memory allocated GB": f"{max(r['peak_alloc_gb'] for r in R):.2f}",
                    "peak GPU memory reserved GB": f"{max(r['peak_reserved_gb'] for r in R):.2f}"})
    write_rows(NARROW2_DIR / "totals.csv", tot)
    L += ["", "TOTALS (peak memory: max over the calls, per call, model weights included)"]
    w = max(len(k) for k in tot[0])
    L.append(f"  {'':{w}s} " + "  ".join(f"{t['narrow']:>14s}" for t in tot))
    for key in list(tot[0])[1:]:
        L.append(f"  {key:{w}s} " + "  ".join(f"{str(t[key]):>14s}" for t in tot))
    corner = "true / pred"
    for v in (1, 2):
        L += ["", f"CONFUSION narrow v{v} (rows = true type, columns = predicted type)",
              f"  {corner:12s}" + "".join(f"{t:>10s}" for t in TYPES)]
        for tt in TYPES:
            L.append(f"  {tt:12s}" + "".join(
                f"{sum(r['type'] == tt and r['pred_type'] == pt for r in rows[v]):>10d}" for pt in TYPES))
    true_types = [r["type"] for r in rows[2]]
    maj = sum(t == "rear-end" for t in true_types)
    ce = np.array([math.hypot(0.5 - r["center_x"], 0.5 - r["center_y"]) for r in rows[2]])
    k2 = sum(r["type_correct"] for r in rows[2])
    decision = ("ADOPT narrow v2" if k2 > NARROW_V1_CORRECT else
                "keep narrow v1 (tie: the simpler version)" if k2 == NARROW_V1_CORRECT else "keep narrow v1")
    L += ["", "BASELINES on the same clips",
          f"  majority type 'rear-end': type accuracy {maj / n:.4f} ({maj}/{n})",
          f"  image-centre point (0.5, 0.5): point error mean {ce.mean():.4f}, median {np.median(ce):.4f}",
          "", f"DECISION (pre-registered: adopt v2 if its type accuracy > v1's {NARROW_V1_CORRECT}/{n}; a tie keeps v1): "
              f"v2 {k2}/{n} -> {decision}",
          f"narrow v1 re-run: {same}/{n} raw outputs identical to results/vlm_dev/narrow.csv"]
    text = "\n".join(L) + "\n"
    (NARROW2_DIR / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


# ---------------------------------------------------------------- frozen full scan on all 2211 synthetic clips

FULL_DIR = REPO_ROOT / "results" / "vlm_full_scan"
FROZEN_FILE = FULL_DIR / "FROZEN.txt"
FULL_CSV = FULL_DIR / "full_scan.csv"
FROZEN_VERSION = 3
POWER_CHECK_EVERY = 25
TAU_CURVE = REPO_ROOT / "results" / "diagnostics" / "tau_evidence" / "isotonic_curve.csv"
FULL_COLUMNS = ["rgb_path", "type", "weather", "day_night", "val_fold", "duration", "t_star", "t_hat", "dev_clip",
                "n_coarse", "coarse_raw", "coarse_value", "coarse_digit_string", "coarse_invalid", "coarse_edge", "t_c",
                "n_fine", "fine_window", "fine_raw", "fine_value", "fine_digit_string", "fine_invalid", "fine_edge",
                "t_fine", "gate", "fallback", "t_full", "error", "gpu_s_coarse", "gpu_s_fine", "wall_s", "gpu", "finished_utc"]


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def frozen_hashes():
    return {"vlm_zero_shot.py": sha256(Path(__file__)),
            "gpu_check/vlm_smoke_test.py": sha256(REPO_ROOT / "gpu_check" / "vlm_smoke_test.py"),
            "full_scan_prompt_v3_template": text_sha(FULL_PROMPT[FROZEN_VERSION]),
            "narrow_prompt_v1": text_sha(NARROW_PROMPT)}


def power():
    """(on mains, text) from Win32_Battery (BatteryStatus 2 = on AC mains)."""
    import subprocess
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "$b = Get-CimInstance Win32_Battery | Select-Object -First 1; "
                          "\"$($b.BatteryStatus) $($b.EstimatedChargeRemaining)\""],
                         capture_output=True, text=True).stdout.split()
    status, charge = (out + ["?", "?"])[:2]
    return status == "2", f"BatteryStatus {status} (2 = on AC mains), charge {charge}%"


def write_frozen():
    import subprocess
    if FROZEN_FILE.exists():
        fail(f"{FROZEN_FILE} exists; the protocol is frozen once")
    git = lambda *a: subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    dirty = git("status", "--porcelain", "--", "vlm_zero_shot.py", "gpu_check/vlm_smoke_test.py")
    if dirty:
        fail(f"commit the code first; uncommitted: {dirty}")
    h = frozen_hashes()
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    L = [f"FROZEN zero-shot VLM protocol; written {now()}; nothing below may change after the run starts (the run "
         "refuses to start or resume if a hash differs).",
         f"git commit {git('rev-parse', 'HEAD')} (branch {git('branch', '--show-current')})",
         *[f"sha256 {k}: {v}" for k, v in h.items()],
         f"model: {smoke.MODELS[MODEL_KEY]}, 4-bit NF4 (double quant, bf16 compute), greedy decoding, max_new_tokens "
         f"{MAX_NEW_TOKENS}; frames full frame, no crop, PIL bicubic",
         f"FULL SCAN = version 3 (selected by the pre-registered rule on the 32 development clips): {V3_FRAMES} "
         f"distinct frames per pass, each frame duplicated so every 2-frame token group holds one frame with its own "
         f"timestamp '<T.T seconds>'; frames resized to {PIXEL_BUDGET} pixels (448 x 256 area, aspect kept); coarse "
         f"pass evenly over the whole clip (0 to the last frame time) -> t_c; fine pass evenly over t_c +- "
         f"{FINE_HALF_S:g} s (shifted inside the clip) -> t_fine; answer {{\"time\": seconds}}; valid = JSON with a "
         "number or a string of digits (flagged) within [first - 0.1, last + 0.1] s; rule A: t < "
         f"{MIN_T_S} s invalid; coarse invalid -> t_full = duration / 2, no fine pass; fine invalid -> t_full = t_c; "
         "rule B gate: fine answer nearest the first or last of the 16 frame times -> t_full = t_c; else t_full = t_fine",
         f"full-scan prompt template (sha above): {FULL_PROMPT[FROZEN_VERSION]}",
         f"NARROW CALL = narrow v1 (kept by the pre-registered rule): {NARROW_FRAMES} frames evenly over t_hat +- "
         f"{NARROW_HALF_S:g} s (shifted inside the clip) at {PIXEL_BUDGET} pixels (448 x 256), frames paired natively; "
         f"answer collision_type, x, y; invalid -> {FALLBACK_TYPE}, {FALLBACK_POINT}. Not run in this full scan.",
         f"narrow prompt (sha above): {NARROW_PROMPT}",
         "run: all 2211 synthetic clips (metadata-synthetic.csv order), once, on mains power (checked at start and every "
         f"{POWER_CHECK_EVERY} clips; on battery the run stops and resumes later); results appended per clip to "
         "full_scan.csv (resumable: finished clips are skipped)",
         "p_full = share of the 2211 clips with |t_full - t*| <= 2 s, pooled, fallbacks at their fallback time; tau = "
         f"the smallest confidence c with p_narrow(c) >= p_full on {TAU_CURVE.relative_to(REPO_ROOT)} (computed once)",
         "FROZEN_JSON " + json.dumps({"hashes": h, "version": FROZEN_VERSION})]
    FROZEN_FILE.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def check_frozen():
    if not FROZEN_FILE.exists():
        fail(f"{FROZEN_FILE} missing")
    line = [x for x in FROZEN_FILE.read_text(encoding="utf-8").splitlines() if x.startswith("FROZEN_JSON ")]
    frozen = json.loads(line[0][len("FROZEN_JSON "):])
    now_h = frozen_hashes()
    bad = [k for k, v in frozen["hashes"].items() if now_h.get(k) != v]
    if bad:
        fail(f"differs from {FROZEN_FILE.name}: {bad}; the frozen protocol may not change")


def full_scan_one(vlm, clip, v=FROZEN_VERSION):
    """The frozen full scan (version v) of one clip; the same functions as the development run."""
    t0 = time.perf_counter()
    r = {"n_fine": 0, "fine_window": "", "fine_raw": "", "fine_value": "", "fine_digit_string": False,
         "fine_invalid": "", "fine_edge": False, "t_fine": "", "gate": False, "gpu_s_fine": 0.0}
    c_idx = to_indices(clip, np.linspace(0.0, last_time(clip), V3_FRAMES))
    ct = np.array(c_idx) / clip["fps"]
    frames = stack(decode(clip, c_idx), c_idx)
    text, s = vlm(clip, np.repeat(frames, 2, axis=0), [i for i in c_idx for _ in range(2)], full_prompt(v, clip, ct))
    t_c, val, digit, why, edge = parse_full(text, v, ct)
    r.update(n_coarse=len(c_idx), coarse_raw=text, coarse_value="" if val is None else val, coarse_digit_string=digit,
             coarse_invalid=why, coarse_edge=edge, t_c="" if t_c is None else t_c, gpu_s_coarse=s)
    if t_c is None:
        r["t_full"], r["fallback"] = clip["duration"] / 2, "coarse"
    else:
        fi = to_indices(clip, window_times(clip, t_c, FINE_HALF_S, V3_FRAMES))
        ft = np.array(fi) / clip["fps"]
        frames = stack(decode(clip, fi), fi)
        text, s = vlm(clip, np.repeat(frames, 2, axis=0), [i for i in fi for _ in range(2)], full_prompt(v, clip, ft))
        t_f, val, digit, why, edge = parse_full(text, v, ft)
        r.update(n_fine=len(fi), fine_window=f"{ft[0]:.2f}-{ft[-1]:.2f}", fine_raw=text,
                 fine_value="" if val is None else val, fine_digit_string=digit, fine_invalid=why, fine_edge=edge,
                 t_fine="" if t_f is None else t_f, gpu_s_fine=s)
        if t_f is None:
            r["t_full"], r["fallback"] = t_c, "fine"
        elif edge:
            r["t_full"], r["fallback"], r["gate"] = t_c, "", True
        else:
            r["t_full"], r["fallback"] = t_f, ""
    r["error"] = r["t_full"] - clip["t_star"]
    r["wall_s"] = time.perf_counter() - t0
    return r


def full_scan_run():
    check_frozen()
    ok, ptxt = power()
    events = FULL_DIR / "run_events.txt"

    def event(msg):
        with open(events, "a", encoding="utf-8") as f:
            f.write(f"{now()} {msg}\n")
        stamp(msg)
    if not ok:
        event(f"NOT STARTED: not on mains ({ptxt})")
        fail("not on mains power")
    clips = load_clips(None)
    dev = {norm(r["rgb_path"]) for r in read_csv(DEV_CSV)}
    done = {norm(r["rgb_path"]) for r in read_csv(FULL_CSV)} if FULL_CSV.exists() else set()
    todo = [c for c in clips if c["rgb_path"] not in done]
    event(f"START/RESUME: {len(done)} clips done, {len(todo)} to do; power {ptxt}")
    if not todo:
        return
    vlm = Vlm()
    new = not FULL_CSV.exists()
    with open(FULL_CSV, "a", newline="", encoding="utf-8") as f, \
            open(FULL_DIR / "raw_outputs.jsonl", "a", encoding="utf-8") as raw:
        w = csv.DictWriter(f, fieldnames=FULL_COLUMNS)
        if new:
            w.writeheader()
        for k, clip in enumerate(todo, 1):
            if k % POWER_CHECK_EVERY == 0:
                ok, ptxt = power()
                if not ok:
                    event(f"STOPPED (resumable): not on mains ({ptxt}) after {len(done) + k - 1} clips")
                    sys.exit(4)
            r = full_scan_one(vlm, clip)
            row = {**{key: clip[key] for key in ("rgb_path", "type", "weather", "day_night", "val_fold", "duration",
                                                  "t_star", "t_hat")},
                   "dev_clip": clip["rgb_path"] in dev, **r, "gpu": vlm.gpu, "finished_utc": now()}
            w.writerow(row)
            raw.write(json.dumps({"clip": clip["rgb_path"], "coarse": r["coarse_raw"], "fine": r["fine_raw"]}) + "\n")
            f.flush()
            raw.flush()
            os.fsync(f.fileno())
            if k % 50 == 0 or k == len(todo):
                stamp(f"{len(done) + k}/{len(clips)} clips done (last: {Path(clip['rgb_path']).stem}, "
                      f"{r['wall_s']:.1f} s wall, {r['gpu_s_coarse'] + r['gpu_s_fine']:.2f} GPU s)")
    ok, ptxt = power()
    event(f"FINISHED: {len(clips)} clips; power at end {ptxt}")


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return float("nan"), float("nan")
    p, z2 = k / n, z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return centre - half, centre + half


def full_scan_report():
    """p_full and the reference predictors on the 2211 clips; tau computed once with the frozen rule."""
    check_frozen()
    out = FULL_DIR / "report.txt"
    if out.exists():
        fail(f"{out} exists; the report and tau are computed once")
    rows = read_csv(FULL_CSV)
    clips = {c["rgb_path"]: c for c in load_clips(None)}
    if len(rows) != len(clips) or {norm(r["rgb_path"]) for r in rows} != set(clips):
        fail(f"full_scan.csv has {len(rows)} rows; the run is not complete")
    order = {p: i for i, p in enumerate(clips)}
    rows.sort(key=lambda r: order[norm(r["rgb_path"])])
    ts = np.array([float(r["t_star"]) for r in rows])
    d = np.array([float(r["duration"]) for r in rows])
    tf = np.array([float(r["t_full"]) for r in rows])
    tc_only = np.array([float(r["t_c"]) if r["t_c"] != "" else float(r["duration"]) / 2 for r in rows])
    within = np.abs(tf - ts) <= 2
    n = len(rows)
    L = [f"Frozen zero-shot full scan (version 3) on all {n} synthetic clips; report {now()}. Protocol: FROZEN.txt. "
         "Synthetic labels only; no real clip or real label opened.", ""]

    def block(name, mask):
        k, m = int(within[mask].sum()), int(mask.sum())
        lo, hi = wilson(k, m)
        return f"  {name:28s} n {m:5d}  p_full {k / m:.4f} ({k}/{m})   [Wilson 95% {lo:.4f}-{hi:.4f}, secondary]"
    dev = np.array([r["dev_clip"] == "True" for r in rows])
    L += ["P_FULL = share with |t_full - t*| <= 2 s (fallbacks at their fallback time)",
          block("all 2211 (pooled)", np.ones(n, bool)), block("without the 32 dev clips", ~dev)]
    for key, label in (("val_fold", "fold"), ("type", "type"), ("day_night", "day/night")):
        for val in sorted({r[key] for r in rows}):
            L.append(block(f"{label} {val}", np.array([r[key] == val for r in rows])))
    cz = sum(r["coarse_value"] != "" and float(r["coarse_value"]) == 0.0 for r in rows)
    L += ["", "FULL SCAN DETAILS (all 2211)",
          "  T@0.5 / T@1 / T@2: " + " / ".join(f"{temporal(tf, ts, s):.4f}" for s in (0.5, 1.0, 2.0)),
          f"  coarse-only share within 2 s (t_c; coarse fallbacks at duration / 2): {np.mean(np.abs(tc_only - ts) <= 2):.4f}",
          f"  coarse fallbacks {sum(r['fallback'] == 'coarse' for r in rows)}, fine fallbacks "
          f"{sum(r['fallback'] == 'fine' for r in rows)}, gated (rule B) {sum(r['gate'] == 'True' for r in rows)}, "
          f"digit strings {sum((r['coarse_digit_string'] == 'True') + (r['fine_digit_string'] == 'True') for r in rows)}",
          f"  coarse answers at 0.0: {cz}/{n} = {cz / n:.4f}; coarse invalid reasons: " + ", ".join(
              f"{why} {sum(r['coarse_invalid'] == why for r in rows)}" for why in sorted({r['coarse_invalid'] for r in rows} - {''})),
          f"  median |error| {np.median(np.abs(tf - ts)):.3f} s",
          f"  GPU s per clip (coarse + fine, synchronised generate): {np.mean([float(r['gpu_s_coarse']) + float(r['gpu_s_fine']) for r in rows]):.2f} "
          f"on {', '.join(sorted({r['gpu'] for r in rows}))}; wall s per clip (incl. CPU decoding) "
          f"{np.mean([float(r['wall_s']) for r in rows]):.2f}"]
    L += ["", "REFERENCE PREDICTORS on the same 2211 clips (within 2 s; T@0.5 / T@1 / T@2)"]
    refs = [("naive midpoint (duration / 2)", d / 2)]
    for s in (0, 1, 2):
        pr = {norm(p["rgb_path"]): float(p["t_hat"]) for p in
              read_csv(REPO_ROOT / "results" / "diagnostics" / "C_cctv_seeds012" / f"seed{s}_frozen" / "tta_clip_predictions_flip.csv")}
        refs.append((f"classifier C_cctv seed {s} (frozen, out-of-fold)", np.array([pr[norm(r['rgb_path'])] for r in rows])))
    vals = {}
    for name, t in refs:
        wv = float(np.mean(np.abs(t - ts) <= 2))
        vals[name] = wv
        L.append(f"  {name:48s} {wv:.4f}   " + " / ".join(f"{temporal(t, ts, s):.4f}" for s in (0.5, 1.0, 2.0)))
    cs = np.array([v for k, v in vals.items() if k.startswith("classifier")])
    L.append(f"  classifier C_cctv seeds 0-2 within 2 s: mean {cs.mean():.4f} +- {cs.std(ddof=1):.4f} (sample std, n 3)")
    devraw = {}
    for line in open(REPO_ROOT / "results" / "vlm_dev" / "v3" / "raw_outputs.jsonl", encoding="utf-8"):
        o = json.loads(line)
        if o.get("version") == FROZEN_VERSION:
            devraw[(norm(o["clip"]), o["pass"])] = o["output"]
    same = sum(devraw.get((norm(r["rgb_path"]), "coarse")) == r["coarse_raw"] and
               (r["fine_raw"] == "" or devraw.get((norm(r["rgb_path"]), "fine")) == r["fine_raw"]) for r in rows if r["dev_clip"] == "True")
    L.append(f"  consistency: {same}/{int(dev.sum())} dev clips give the same raw answers as the v3 development run")
    # tau, computed once with the frozen rule
    p_full = float(within.mean())
    curve = read_csv(TAU_CURVE)
    cgrid = np.array([float(c["confidence"]) for c in curve])
    pn = np.array([float(c["p_narrow_isotonic"]) for c in curve])
    ok_c = cgrid[pn >= p_full]
    L += ["", f"TAU (frozen rule: the smallest confidence c on {TAU_CURVE.relative_to(REPO_ROOT)} with p_narrow(c) >= "
              f"p_full = {p_full:.4f})"]
    if ok_c.size == 0:
        L.append("  no c satisfies the rule (p_full above the whole curve): no tau")
        tau = None
    else:
        tau = float(ok_c.min())
        L.append(f"  tau = {tau:.2f}  (p_narrow({tau:.2f}) = {pn[cgrid == tau][0]:.4f}; p_narrow at the grid point below: "
                 f"{pn[cgrid < tau][-1]:.4f})" if (cgrid < tau).any() else f"  tau = {tau:.2f} (the first grid point)")
        conf = {s: np.array([float(p["confidence"]) for p in read_csv(
            REPO_ROOT / "results" / "diagnostics" / "C_cctv_seeds012" / f"seed{s}_frozen" / "tta_clip_predictions_flip.csv")])
            for s in (0, 1, 2)}
        sh = [float(np.mean(conf[s] >= tau)) for s in (0, 1, 2)]
        pooled = np.concatenate(list(conf.values()))
        L.append(f"  synthetic clips with confidence >= tau (C_cctv frozen, out-of-fold): seeds 0 / 1 / 2 "
                 f"{' / '.join(f'{v:.4f}' for v in sh)}; mean {np.mean(sh):.4f}; pooled {np.mean(pooled >= tau):.4f} "
                 f"({int((pooled >= tau).sum())}/{pooled.size})")
    text = "\n".join(L) + "\n"
    out.write_text(text, encoding="utf-8")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--make-dev-list", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--clips", type=Path, default=DEV_CSV)
    ap.add_argument("--versions", type=int, nargs="+", choices=sorted(FULL_PROMPT), default=[1, 2])
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "vlm_dev")
    ap.add_argument("--inspect", action="store_true", help="bug checks of the model inputs (CPU, no model)")
    ap.add_argument("--previous-dir", type=Path, default=None,
                    help="with --run: an earlier run's folder (totals.csv, narrow.csv) for the selection and the narrow "
                         "re-run comparison")
    ap.add_argument("--narrow2", choices=["protocol", "inspect", "run"], default=None,
                    help="narrow v2 (addition to the plan): write PROTOCOL.txt, CPU check, or GPU run on --clips")
    ap.add_argument("--full-scan", choices=["freeze", "run", "report"], default=None,
                    help="frozen full scan on all 2211 synthetic clips: write FROZEN.txt, run (resumable), report + tau")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")
    if args.full_scan:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        {"freeze": write_frozen, "run": full_scan_run, "report": full_scan_report}[args.full_scan]()
    elif args.narrow2:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        {"protocol": narrow2_protocol, "inspect": narrow2_inspect,
         "run": lambda: narrow2_run(args.clips)}[args.narrow2]()
    elif args.inspect:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        rows = {v: {norm(r["rgb_path"]): r for r in read_csv(REPO_ROOT / "results" / "vlm_dev" / f"full_scan_v{v}.csv")}
                for v in (1, 2)}
        ts_clip = "synthetic_videos/videos/head-on/Town05_head-on_wet_24.mp4"
        inspect(args.out_dir / "inspect", ts_clip,
                [ts_clip, "synthetic_videos/videos/head-on/Town03_head-on_night_03.mp4"],
                {v: float(rows[v][ts_clip]["t_c"]) for v in (1, 2)})
    elif args.make_dev_list:
        make_dev_list()
    elif args.run:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        run(args.clips, args.versions, args.out_dir, args.previous_dir)
    else:
        fail("give --make-dev-list or --run")


if __name__ == "__main__":
    main()
