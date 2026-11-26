"""Narrow call, prompt version 2 (supervisor feedback of 2026-11-18, "Appendix: narrow prompt, version 2").

Synthetic only. Imports the frozen protocol code from vlm_zero_shot.py (not changed): frame sampling, decoding,
pixel budget, model loading helpers and the narrow-v1 fallback.

Contents
  PROMPT_V2        the appendix text, character for character (checked against the .docx by --check)
  settings         greedy decoding, MAX_NEW_TOKENS_V2 = 160
  parse_v2()       answer parser: JSON (a ``` fence allowed) with "vehicles", "motion", "collision_visible",
                   "collision_type", "bbox_2d". Valid if collision_type (trimmed and lower-cased; flagged
                   "type_normalised" when that changed it) is one of the 5 types and bbox_2d is 4 numbers (int or
                   float, not bool; a string of digits such as "120" or "120.5" is accepted and flagged
                   "bbox_digit_string", as in the full scan) in [0, 1000] with x1 < x2 and y1 < y2; a reversed pair
                   (x1 > x2 or y1 > y2) is swapped and flagged; an equal pair (zero width or height) is invalid.
                   (Digit strings and type normalisation added 2026-11-21, before any model answer was seen.)
                   vehicles / motion / collision_visible are stored as diagnostics; a missing one is recorded but
                   does not make the answer invalid. collision_visible never changes routing or the prediction.
  point            box centre ((x1 + x2) / 2000, (y1 + y2) / 2000) in the frame the model receives, corrected for
                   padding when the frames are letterboxed (PADDING; the check measures whether they are: they are
                   not, the processor resizes the whole frame, so the correction is the identity)
  fallback         invalid answer -> type FALLBACK_TYPE ("rear-end"), point FALLBACK_POINT (0.5, 0.5), as narrow v1;
                   counted by the caller from the "fallback" flag
  narrow_v2_input  16 frames evenly over t_hat +- 2 s (shifted inside the clip), 448 x 256 pixel area, each frame
                   duplicated so every 2-frame temporal token group holds one frame and gets that frame's own
                   timestamp "<T.T seconds>" (as full-scan v3), which makes "The text before each frame gives its
                   time in seconds" true

--check (CPU, no model): for 2 synthetic clips (one day, one night) through the real processor: token counts, the
  timestamps in the prompt text next to the true frame times, padding yes/no, and a grid of the 16 frames as the model
  receives them; also the prompt-vs-docx comparison and parser self-tests. Writes results/narrow_v2_check/.
Usage (from cctv-accident-ITS-system/):  .venv-vlm\\Scripts\\python.exe narrow_prompt_v2.py --check
"""

import argparse
import hashlib
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
import vlm_zero_shot as vz  # noqa: E402  (frozen protocol code: imported, not changed)

smoke = vz.smoke

PROMPT_V2 = """These 16 frames come from a fixed traffic camera, in time order, covering about 4 seconds. The text before each frame gives its time in seconds. A traffic collision is expected near the middle of these frames, but it may happen earlier or later, or may not be visible.

Work in this order:
1. Find the vehicle or vehicles involved in the collision and say where they are in the image.
2. Describe how they were moving just before contact: opposite directions toward each other, same direction one behind the other, side by side in parallel, paths crossing at an angle, or only one vehicle involved.
3. Choose the collision type:
   - head-on: two vehicles moving toward each other hit front to front.
   - rear-end: two vehicles moving in the same direction; the front of the rear vehicle hits the back of the one ahead.
   - sideswipe: two vehicles moving side by side in roughly parallel directions; their sides scrape along each other.
   - t-bone: the paths cross at an angle; the front of one vehicle hits the side of the other.
   - single: only one vehicle is involved; it hits an object, a barrier, the roadside, or rolls over.
   If the paths cross at an angle, prefer t-bone over sideswipe. If the vehicles move in the same direction and the front of one hits the back of the other, prefer rear-end over sideswipe.
4. Draw one box around the area where the vehicles first touch (for single: where the vehicle hits), in the frame where contact first happens.

Answer with one JSON object and nothing else, in this exact form:
{"vehicles": "<short description and position>", "motion": "<opposite | same direction | parallel | crossing | one vehicle>", "collision_visible": <true or false>, "collision_type": "<head-on | rear-end | sideswipe | single | t-bone>", "bbox_2d": [x1, y1, x2, y2]}

bbox_2d uses relative coordinates from 0 to 1000: x from the left edge, y from the top edge, [x1, y1] the top-left corner and [x2, y2] the bottom-right corner. Always give a collision_type and a bbox_2d, even if collision_visible is false."""

DO_SAMPLE = False  # greedy decoding
MAX_NEW_TOKENS_V2 = 160
N_FRAMES_V2 = 16
HALF_S_V2 = vz.NARROW_HALF_S  # 2.0 s
BUDGET_V2 = vz.PIXEL_BUDGET  # 448 x 256 area
TYPES = vz.TYPES
FALLBACK_TYPE, FALLBACK_POINT = vz.FALLBACK_TYPE, vz.FALLBACK_POINT
KEYS = ("vehicles", "motion", "collision_visible", "collision_type", "bbox_2d")
# Letterbox padding of the frames the model receives, (left, top, right, bottom) in model pixels. None = not padded
# (measured by --check: the processor resizes the whole frame to a multiple of 32, it does not pad).
PADDING = None

DOCX = REPO_ROOT.parent / "docs" / "Feedback on the Zero-Shot VLM Stage and Routing Threshold.docx"
APPENDIX_HEADING = "Appendix: narrow prompt, version 2"
CHECK_DIR = REPO_ROOT / "results" / "narrow_v2_check"


# ---------------------------------------------------------------- answer

def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)


DIGITS = re.compile(r"\d+(\.\d+)?")  # the full scan's digit-string rule ("12" or "12.5")


def as_num(v):
    """(number or None, digit-string flag): a number, or a string of digits (accepted and flagged, as the full scan)."""
    if is_num(v):
        return v, False
    if isinstance(v, str) and DIGITS.fullmatch(v.strip()):
        return float(v.strip()), True
    return None, False


def bbox_to_point(bbox, padding=PADDING, size=None):
    """Box centre as fractions of the image. bbox in 0..1000 of the frame the model receives; with padding
    (left, top, right, bottom) in pixels of that frame of size (W, H), the centre is mapped back to the content."""
    x1, y1, x2, y2 = bbox
    x, y = (x1 + x2) / 2000.0, (y1 + y2) / 2000.0
    if padding is None:
        return x, y
    left, top, right, bottom = padding
    w, h = size
    return (x * w - left) / (w - left - right), (y * h - top) / (h - top - bottom)


def parse_v2(text, padding=PADDING, size=None):
    """Parse one narrow-v2 answer. Returns a dict; "fallback" True means the answer was invalid and the narrow-v1
    fallback (FALLBACK_TYPE, FALLBACK_POINT) is used."""
    r = {"valid": False, "reason": "", "collision_type": FALLBACK_TYPE, "x": FALLBACK_POINT[0], "y": FALLBACK_POINT[1],
         "bbox_2d": None, "swapped_x": False, "swapped_y": False, "bbox_digit_string": False, "type_normalised": False,
         "vehicles": None, "motion": None, "collision_visible": None, "missing_keys": [], "fallback": True}
    obj = vz.parse_json(text)
    if obj is None:
        r["reason"] = "no JSON object"
        return r
    r["missing_keys"] = [k for k in KEYS if k not in obj]
    r["vehicles"], r["motion"] = obj.get("vehicles"), obj.get("motion")
    r["collision_visible"] = obj.get("collision_visible")  # diagnostic only
    typ_raw, box_raw = obj.get("collision_type"), obj.get("bbox_2d")
    typ = typ_raw.strip().lower() if isinstance(typ_raw, str) else typ_raw  # trimmed and lower-cased, flagged
    r["type_normalised"] = isinstance(typ_raw, str) and typ != typ_raw
    if typ not in TYPES:
        r["reason"] = f"collision_type not one of the 5 types: {typ_raw!r}"
        return r
    conv = [as_num(v) for v in box_raw] if isinstance(box_raw, list) else []
    if not (len(conv) == 4 and all(v is not None for v, _ in conv)):
        r["reason"] = f"bbox_2d is not 4 numbers: {box_raw!r}"
        return r
    box = [v for v, _ in conv]
    r["bbox_digit_string"] = any(d for _, d in conv)
    if not all(0 <= v <= 1000 for v in box):
        r["reason"] = f"bbox_2d outside 0-1000: {box_raw!r}"
        return r
    x1, y1, x2, y2 = box
    if x1 > x2:
        x1, x2, r["swapped_x"] = x2, x1, True
    if y1 > y2:
        y1, y2, r["swapped_y"] = y2, y1, True
    if x1 == x2 or y1 == y2:
        r["reason"] = f"bbox_2d has zero width or height: {box_raw!r}"
        return r
    x, y = bbox_to_point((x1, y1, x2, y2), padding, size)
    r.update(valid=True, collision_type=typ, x=x, y=y, bbox_2d=[x1, y1, x2, y2], fallback=False)
    return r


# ---------------------------------------------------------------- input

def narrow_v2_input(clip):
    """(distinct frame indices, model frames with each frame duplicated, duplicated indices, prompt)."""
    idx = vz.to_indices(clip, vz.window_times(clip, clip["t_hat"], HALF_S_V2, N_FRAMES_V2))
    frames = vz.stack(vz.decode(clip, idx, BUDGET_V2), idx)
    return idx, np.repeat(frames, 2, axis=0), [i for i in idx for _ in range(2)], PROMPT_V2


# ---------------------------------------------------------------- CPU check

def docx_appendix_text(path):
    """The paragraph after the appendix heading's instruction paragraph (style Code), w:br -> \\n, w:tab -> \\t."""
    w = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    import xml.etree.ElementTree as ET

    body = ET.fromstring(zipfile.ZipFile(path).read("word/document.xml")).find(f"{w}body")
    paras = []
    for p in body.findall(f"{w}p"):
        parts = []
        for node in p.iter():
            t = node.tag.replace(w, "")
            parts.append(node.text or "" if t == "t" else "\n" if t == "br" else "\t" if t == "tab" else "")
        st = p.find(f"{w}pPr/{w}pStyle")
        paras.append((st.get(f"{w}val") if st is not None else "", "".join(parts)))
    h = next(i for i, (_, t) in enumerate(paras) if t.strip() == APPENDIX_HEADING)
    code = [t for s, t in paras[h + 1:] if s == "Code"]
    if len(code) != 1:
        vz.fail(f"expected one Code paragraph after the appendix heading, found {len(code)}")
    return code[0]


def parser_selftest():
    """Hand-written answers: (text, expected valid, expected point or None, expected type or None, expected flags)."""
    F = ("swapped_x", "swapped_y", "bbox_digit_string", "type_normalised")
    cases = [
        ('{"vehicles": "a", "motion": "crossing", "collision_visible": true, "collision_type": "t-bone", '
         '"bbox_2d": [100, 200, 300, 400]}', True, (0.2, 0.3), "t-bone", ()),
        ('```json\n{"vehicles": "a", "motion": "opposite", "collision_visible": false, "collision_type": "head-on", '
         '"bbox_2d": [300, 400, 100, 200]}\n```', True, (0.2, 0.3), "head-on", ("swapped_x", "swapped_y")),
        ('{"collision_type": "rear-end", "bbox_2d": [0, 0, 1000, 1000]}', True, (0.5, 0.5), "rear-end", ()),
        # new case 1 (2026-11-21): collision_type trimmed and lower-cased, flagged
        ('{"collision_type": " T-Bone ", "bbox_2d": [100, 200, 300, 400]}', True, (0.2, 0.3), "t-bone",
         ("type_normalised",)),
        # new case 2 (2026-11-21): digit strings in bbox_2d accepted, flagged
        ('{"collision_type": "single", "bbox_2d": ["100", "200.5", 300, " 400 "]}', True, (0.2, 0.30025), "single",
         ("bbox_digit_string",)),
        ('{"collision_type": "Rear-end", "bbox_2d": [1, 2, 3, 4]}', True, (0.002, 0.003), "rear-end",
         ("type_normalised",)),
        ('{"collision_type": "single", "bbox_2d": ["1", 2, 3, 4]}', True, (0.002, 0.003), "single",
         ("bbox_digit_string",)),
        ('{"collision_type": "rearend", "bbox_2d": [1, 2, 3, 4]}', False, None, None, ()),
        ('{"collision_type": "single", "bbox_2d": ["-5", 2, 3, 4]}', False, None, None, ()),
        ('{"collision_type": "single", "bbox_2d": ["1e3", 2, 3, 4]}', False, None, None, ()),
        ('{"collision_type": "single", "bbox_2d": [1, 2, 3]}', False, None, None, ()),
        ('{"collision_type": "single", "bbox_2d": [1, 2, 3, 1001]}', False, None, None, ()),
        ('{"collision_type": "single", "bbox_2d": [1, 2, 3, "1001"]}', False, None, None, ("bbox_digit_string",)),
        ('{"collision_type": "single", "bbox_2d": [5, 2, 5, 4]}', False, None, None, ()),
        ('{"collision_type": "single", "bbox_2d": [true, 2, 3, 4]}', False, None, None, ()),
        ('{"collision_type": 3, "bbox_2d": [1, 2, 3, 4]}', False, None, None, ()),
        ("not json", False, None, None, ()),
    ]
    out, ok_all = [], True
    for text, valid, pt, typ, flags in cases:
        r = parse_v2(text)
        ok = r["valid"] == valid and (pt is None or np.allclose((r["x"], r["y"]), pt))
        ok = ok and (typ is None or r["collision_type"] == typ)
        ok = ok and all(bool(r[f]) == (f in flags) for f in F)
        ok = ok and (valid or (r["collision_type"], r["x"], r["y"]) == (FALLBACK_TYPE, *FALLBACK_POINT))
        ok_all &= ok
        out.append(f"  {'ok  ' if ok else 'FAIL'} valid {r['valid']!s:5s} point ({r['x']:.5f}, {r['y']:.5f}) type "
                   f"{r['collision_type']:9s} flags {[f for f in F if r[f]]} missing {r['missing_keys']} "
                   f"reason {r['reason']!r} <- {text[:75]!r}")
    return ok_all, out


def generate_v2(vlm, clip, frames, indices, prompt=PROMPT_V2, max_new_tokens=MAX_NEW_TOKENS_V2):
    """The narrow-v2 model call: vlm_zero_shot.Vlm.__call__ (same input building, bf16 pixels, greedy decoding,
    synchronised timing, peak memory of the call) with max_new_tokens = MAX_NEW_TOKENS_V2 instead of the frozen 64.
    Returns (text, generate seconds, number of new tokens)."""
    import time

    torch = vlm.torch
    inputs, _, _ = smoke.build_inputs(vlm.processor, frames, indices, clip["fps"], clip, prompt_text_in=prompt)
    n_in = inputs["input_ids"].shape[1]
    inputs = inputs.to(vlm.device)
    inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(torch.bfloat16)
    vlm.last_visual_tokens = int((inputs["input_ids"][0] == vlm.processor.video_token_id).sum())
    vlm.last_grid_thw = inputs["video_grid_thw"][0].tolist()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(vlm.device)
    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            out = vlm.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=DO_SAMPLE,
                                     temperature=None, top_p=None, top_k=None)
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return f"OOM: {str(e).splitlines()[0]}", time.perf_counter() - t0, 0
    torch.cuda.synchronize()
    gen_s = time.perf_counter() - t0
    vlm.last_peak_alloc_gb = torch.cuda.max_memory_allocated(vlm.device) / 2 ** 30
    vlm.last_peak_reserved_gb = torch.cuda.max_memory_reserved(vlm.device) / 2 ** 30
    new = out[0, n_in:]
    return vlm.processor.batch_decode(new[None], skip_special_tokens=True)[0], gen_s, int(new.numel())


def padding_check(frames_in, mv):
    """frames_in: the resized input frames (T, h, w, 3); mv: the frames as the model receives them (T, H, W, 3).
    Padded if a border band of the model frames is uniform in every frame, or the content does not match the whole
    input frame resized to the model size."""
    from PIL import Image

    t, H, W, _ = mv.shape
    diffs = [float(np.abs(np.asarray(Image.fromarray(f).resize((W, H), Image.BICUBIC), dtype=np.float32)
                          - m.astype(np.float32)).mean()) for f, m in zip(frames_in, mv)]

    def uniform_band(get):
        n = 0
        while n < 32 and all(get(m, n).std() < 1.0 for m in mv):
            n += 1
        return n

    bands = {"left": uniform_band(lambda m, n: m[:, n]), "right": uniform_band(lambda m, n: m[:, W - 1 - n]),
             "top": uniform_band(lambda m, n: m[n]), "bottom": uniform_band(lambda m, n: m[H - 1 - n])}
    return any(bands.values()), bands, diffs


def check():
    from transformers import AutoProcessor

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    L = [f"Narrow v2 CPU check (no model; processor {smoke.MODELS[vz.MODEL_KEY]}); {vz.now()}; synthetic clips only",
         f"narrow_prompt_v2.py sha256 {vz.sha256(Path(__file__))}; PROMPT_V2 sha256 {vz.text_sha(PROMPT_V2)}; "
         f"{len(PROMPT_V2)} characters, {PROMPT_V2.count(chr(10)) + 1} lines; greedy (do_sample {DO_SAMPLE}), "
         f"max_new_tokens {MAX_NEW_TOKENS_V2}"]
    frozen = vz.FROZEN_FILE.read_text(encoding="utf-8")
    L.append("frozen files unchanged (current sha256 found in FROZEN.txt): " + ", ".join(
        f"{k} {'yes' if v in frozen else 'NO'}" for k, v in vz.frozen_hashes().items()))

    # 1. prompt vs docx
    L.append(f"\n1. PROMPT_V2 vs the .docx appendix ({DOCX.name}, sha256 {vz.sha256(DOCX)})")
    doc = docx_appendix_text(DOCX)
    same = doc == PROMPT_V2
    L.append(f"  identical character for character: {same} (docx {len(doc)} characters, PROMPT_V2 {len(PROMPT_V2)})")
    if not same:
        k = next((i for i, (a, b) in enumerate(zip(doc, PROMPT_V2)) if a != b), min(len(doc), len(PROMPT_V2)))
        L.append(f"  first difference at character {k}: docx {doc[k:k + 40]!r} vs PROMPT_V2 {PROMPT_V2[k:k + 40]!r}")
    non_ascii = sorted({c for c in PROMPT_V2 if ord(c) > 127})
    L.append(f"  non-ASCII characters in PROMPT_V2: {non_ascii or 'none'}")

    # 2. parser self-test
    ok, lines = parser_selftest()
    L.append(f"\n2. PARSER SELF-TEST (hand-written strings, no model): all as expected: {ok}")
    L.extend(lines)

    # 3. inputs through the real processor
    processor = AutoProcessor.from_pretrained(smoke.MODELS[vz.MODEL_KEY])
    clips = {c["rgb_path"]: c for c in vz.load_clips(vz.DEV_CSV)}
    vp = processor.video_processor
    L.append(f"\n3. INPUTS THROUGH THE REAL PROCESSOR (video processor {type(vp).__name__}: patch {vp.patch_size}, merge "
             f"{vp.merge_size}, temporal patch {vp.temporal_patch_size}; do_pad attribute: "
             f"{getattr(vp, 'do_pad', 'none')})")
    for cp in vz.NARROW2_CLIPS:  # one day, one night (the narrow v2 inspection clips)
        c = clips[cp]
        idx, frames2, idx2, prompt = narrow_v2_input(c)
        inputs, _, _ = smoke.build_inputs(processor, frames2, idx2, c["fps"], c, prompt_text_in=prompt)
        ids = inputs["input_ids"][0]
        text = processor.tokenizer.decode(ids)
        stamps = [float(s) for s in re.findall(r"<(\d+(?:\.\d+)?) seconds>", text)]
        st = np.array(idx) / c["fps"]
        mv_all = vz.model_view(inputs, processor)
        mv = mv_all[::2]
        dup_same = bool(all(np.array_equal(mv_all[2 * k], mv_all[2 * k + 1]) for k in range(len(idx))))
        padded, bands, diffs = padding_check(frames2[::2], mv)
        grid = inputs["video_grid_thw"][0].tolist()
        L.append(f"\n  {Path(cp).stem} ({c['day_night']}, {c['weather']}): duration {c['duration']:.2f} s, fps "
                 f"{c['fps']:.4f}, t_hat {c['t_hat']:.2f} s (C_cctv seed 0, frozen, out-of-fold)")
        L.append(f"    frames {idx[0]}..{idx[-1]} ({st[0]:.2f}-{st[-1]:.2f} s), 16 distinct frames sent as 32 "
                 f"(each duplicated); input {frames2.shape[2]} x {frames2.shape[1]}, model receives {mv.shape[2]} x "
                 f"{mv.shape[1]} (W x H); grid_thw {grid}")
        L.append(f"    tokens: visual {int((ids == processor.video_token_id).sum())} ({grid[0]} groups x "
                 f"{grid[1] * grid[2] // vp.merge_size ** 2}), vision_start {int((ids == processor.vision_start_token_id).sum())}, "
                 f"input total {int(ids.numel())}; + up to {MAX_NEW_TOKENS_V2} new tokens")
        L.append(f"    both frames of every group identical in the model input: {dup_same}")
        L.append(f"    {'frame':>5s} {'index':>6s} {'true time s':>11s} {'prompt text':>14s} {'diff s':>7s}")
        for k in range(max(len(idx), len(stamps))):
            s = stamps[k] if k < len(stamps) else float("nan")
            tt = st[k] if k < len(st) else float("nan")
            L.append(f"    {k:5d} {idx[k] if k < len(idx) else '':>6} {tt:11.3f} {'<%.1f seconds>' % s:>14s} {s - tt:+7.3f}")
        d = np.abs(np.array(stamps[:len(st)]) - st)
        L.append(f"    {len(stamps)} timestamps for {len(idx)} frames (one per frame: {len(stamps) == len(idx)}); "
                 f"max |text - true time| {d.max():.3f} s (the processor writes one decimal); distinct texts "
                 f"{len(set(stamps))}")
        L.append(f"    padding: {'YES' if padded else 'no'} (uniform border bands in px {bands}); mean |model frame - "
                 f"whole input frame resized to {mv.shape[2]} x {mv.shape[1]}| per frame {min(diffs):.2f}-{max(diffs):.2f} "
                 f"(0-255 scale) -> the whole frame is resized, aspect {frames2.shape[2] / frames2.shape[1]:.3f} -> "
                 f"{mv.shape[2] / mv.shape[1]:.3f}; bbox 0-1000 maps to the full image on each axis")
        L.append(f"    PADDING constant in the code: {PADDING} -> {'consistent' if (PADDING is not None) == padded else 'MISMATCH'}")
        p = CHECK_DIR / f"narrow_v2_{Path(cp).stem}.png"
        vz.save_grid(mv, p)
        L.append(f"    grid: {p.name} (the 16 distinct frames as the model receives them, 4 per row, in time order); "
                 f"true point ({c['center_x']:.3f}, {c['center_y']:.3f}), t* {c['t_star']:.2f} s")
    L.append(f"\nprompt text (the part after the video, as the model receives it) ends with: "
             f"{text[text.rfind('<|vision_end|>') + 14:][:120]!r} ...")
    out = "\n".join(L) + "\n"
    (CHECK_DIR / "check.txt").write_text(out, encoding="utf-8")
    (CHECK_DIR / "prompt_v2.txt").write_text(PROMPT_V2, encoding="utf-8")
    print(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="CPU check through the real processor (no model)")
    ap.add_argument("--print-prompt", action="store_true")
    ap.add_argument("--selftest", type=Path, default=None, metavar="OUT_TXT",
                    help="parser self-test only (no processor, no model); writes OUT_TXT")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")
    if args.selftest:
        ok, lines = parser_selftest()
        text = "\n".join([f"narrow v2 parser self-test, {vz.now()}; narrow_prompt_v2.py sha256 "
                          f"{vz.sha256(Path(__file__))}; all as expected: {ok}", *lines]) + "\n"
        args.selftest.parent.mkdir(parents=True, exist_ok=True)
        args.selftest.write_text(text, encoding="utf-8")
        print(text)
        if not ok:
            sys.exit(1)
    elif args.check:
        check()
    elif args.print_prompt:
        print(PROMPT_V2)
    else:
        vz.fail("give --check or --print-prompt")


if __name__ == "__main__":
    main()
