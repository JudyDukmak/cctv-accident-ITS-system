"""Type model, option 1: zero-shot Qwen3-VL-8B (unsloth bnb 4-bit) on a crop around the QLoRA epoch-1 box. STAGE 1 only.

Pre-registration: results/type_vlm8b/PLAN.txt and PROMPT.txt (sha256 checked by every mode). SYNTHETIC ONLY: never
opens, lists or reads a real clip, real label or real metadata file. Changes no frozen file, checkpoint, split,
manifest or earlier result (vlm_zero_shot.py and gpu_check/vlm_smoke_test.py are imported, not changed).
Run with the .venv-vlm interpreter (no pandas / matplotlib there: csv and PIL only).

  --checks    (no model) a free disk; a2 Hub file sizes; b metadata columns / scene layout; c valid epoch-1 boxes;
              d contact sheet of 3 clips (16 crops each, true box in green)
  --download  download the model to the HF cache (timed), check the local file sizes
  --speed     (GPU, offline) memory and speed test on 5 clips; NO accuracy, NO comparison with the true types
  --report    stage1_report.txt
"""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "gpu_check"))
OUT = REPO / "results" / "type_vlm8b"
PLAN, PROMPT = OUT / "PLAN.txt", OUT / "PROMPT.txt"
PLAN_SHA = "d00eb03c32c0e2d200c3e81637c274840d8517a3620a9f360ecde0ae112deff6"
PROMPT_SHA = "a85361c1b94c6773d9f797de809ca7d90321f755150c723ef66bdee35c5aa552"
REPO_ID, REV = "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit", "10d842936325905934ea24e64325d774693f166e"
DATASET = REPO / "ACCIDENT" / "dataset"
META_SYN = DATASET / "metadata-synthetic.csv"
MANIFEST = REPO / "results" / "qlora_data" / "manifest.csv"
VAL_E1 = REPO / "results" / "qlora_train" / "val_epoch1.csv"
TYPES = ["head-on", "rear-end", "t-bone", "sideswipe", "single"]
RULE_LAYOUTS = {"highway", "tunnel", "grade-separated intersection"}
SIDE_FACTOR, SIDE_MIN, OUT_SIZE, MAX_NEW = 2.5, 256, 448, 64
CHECKS_JSON, SHEET, DL_JSON = OUT / "checks.json", OUT / "contact_sheet_crops.png", OUT / "download.json"
SPEED_JSON, ANSWERS, REPORT = OUT / "speed_test.json", OUT / "speed_test_answers.jsonl", OUT / "stage1_report.txt"


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def check_plan():
    for p, s in ((PLAN, PLAN_SHA), (PROMPT, PROMPT_SHA)):
        if hashlib.sha256(p.read_bytes()).hexdigest() != s:
            fail(f"{p.name} changed since pre-registration")


def refuse(*paths):
    for p in paths:
        if p.exists():
            fail(f"{p} exists; refusing to overwrite")


def rcsv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def norm(p):
    return p.replace("\\", "/")


def load_clips():
    """The 755 fold-1 window-A clips sorted by rgb_path: manifest row, metadata row, epoch-1 answer."""
    man = {norm(r["rgb_path"]): r for r in rcsv(MANIFEST) if r["fold"] == "1" and r["window"] == "A"}
    meta = {norm(r["rgb_path"]): r for r in rcsv(META_SYN)}
    e1 = {norm(r["rgb_path"]): r for r in rcsv(VAL_E1)}
    if len(man) != 755 or set(man) != set(e1) or not set(man) <= set(meta):
        fail("expected the same 755 fold-1 window-A clips in the manifest and val_epoch1.csv")
    if not all(p.startswith("synthetic_videos/") for p in man):
        fail("non-synthetic clip in the list")
    return [(p, man[p], meta[p], e1[p]) for p in sorted(man)]


def epoch1_box(e1, W, H):
    """Epoch-1 box in pixels (x1, y1, x2, y2) or None if there is no valid box."""
    try:
        b = json.loads(e1["bbox"])
        if e1["valid"] != "True" or len(b) != 4:
            return None
        x1, x2 = sorted((b[0] * W / 1000, b[2] * W / 1000))
        y1, y2 = sorted((b[1] * H / 1000, b[3] * H / 1000))
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None
    except (ValueError, TypeError, KeyError):
        return None


def crop_square(e1, W, H):
    """(left, top, side, info): square around the epoch-1 box centre; moved (not shrunk) to lie inside the image."""
    box = epoch1_box(e1, W, H)
    if box is None:
        cx, cy, side, src = W / 2, H / 2, SIDE_MIN, "image centre (no valid epoch-1 box)"
    else:
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side, src = max(SIDE_FACTOR * max(x2 - x1, y2 - y1), SIDE_MIN), "epoch-1 box"
    side = int(round(min(side, H, W)))
    l0, t0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    left, top = min(max(l0, 0), W - side), min(max(t0, 0), H - side)
    return left, top, side, {"centre_source": src, "box_px": box, "centre": (cx, cy), "side": side,
                             "shift": (left - l0, top - t0)}


def prepare(p, man, meta, e1):
    """16 window-A frames, cropped and resized: (frames uint8 (16, 448, 448, 3), indices, fps, clip, crop info)."""
    import vlm_zero_shot as vz
    from PIL import Image
    W, H = int(meta["width"]), int(meta["height"])
    idx = json.loads(man["frame_indices"])
    if len(idx) != 16:
        fail(f"{p}: {len(idx)} window-A frames")
    clip = {"video": str(DATASET / p), "rgb_path": p, "no_frames": int(meta["no_frames"]),
            "duration": float(meta["duration"])}
    got = vz.decode(clip, idx, budget=10 ** 12)
    left, top, side, info = crop_square(e1, W, H)
    out = []
    for i in idx:
        img = got[i]
        if img.size != (W, H):
            fail(f"{p}: decoded {img.size}, metadata {W}x{H}")
        out.append(np.asarray(img.crop((left, top, left + side, top + side)).resize((OUT_SIZE, OUT_SIZE), Image.BICUBIC)))
    info.update(left=left, top=top)
    return np.stack(out), idx, clip["no_frames"] / clip["duration"], clip, info


def prompt_text(scene_layout):
    return PROMPT.read_text(encoding="utf-8").rstrip("\n").replace("{scene_layout}", scene_layout)


def parse(text):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    m = re.search(r"\{.*?\}", t, flags=re.S)
    if not m:
        return None, "no JSON object"
    try:
        v = json.loads(m.group(0)).get("type")
    except (ValueError, AttributeError):
        return None, "JSON does not parse"
    if not isinstance(v, str) or v.strip().lower() not in TYPES:
        return None, f"type not allowed: {v!r}"
    return v.strip().lower(), ""


def apply_rule(t, scene_layout):
    return "rear-end" if t == "t-bone" and scene_layout in RULE_LAYOUTS else t


# ---------------------------------------------------------------- checks (no model)

def checks():
    from huggingface_hub import HfApi
    from PIL import Image, ImageDraw, ImageFont
    check_plan()
    refuse(CHECKS_JSON, SHEET)
    clips = load_clips()
    du = shutil.disk_usage(REPO)
    info = HfApi().model_info(REPO_ID, revision=REV, files_metadata=True)
    files = {s.rfilename: s.size for s in info.siblings}
    cols = list(rcsv(META_SYN)[0].keys())
    scene_cols = [c for c in cols if any(w in c.lower() for w in ("scene", "layout", "road", "junction", "intersection"))]
    n_bad = [p for p, man, meta, e1 in clips if epoch1_box(e1, int(meta["width"]), int(meta["height"])) is None]
    res = {"disk_free_gb": du.free / 1e9, "disk_total_gb": du.total / 1e9, "hub_revision": info.sha,
           "hub_files": files, "hub_total_bytes": sum(v or 0 for v in files.values()),
           "metadata_columns": cols, "scene_layout_columns": scene_cols,
           "scene_layout_value": "not given (no such column)" if not scene_cols else None,
           "n_clips": len(clips), "n_without_valid_epoch1_box": len(n_bad), "without_valid_box": n_bad}
    # contact sheet: 3 clips, default_rng(0), 16 crops each, true (metadata) box in green
    pick = sorted(np.random.default_rng(0).choice(len(clips), size=3, replace=False).tolist())
    cell, head = 224, 60
    sheet = Image.new("RGB", (3 * 4 * cell + 2 * 20, head + 4 * cell + 30), (20, 20, 20))
    dr = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default(15)
    res["contact_sheet"] = []
    for n, j in enumerate(pick):
        p, man, meta, e1 = clips[j]
        frames, idx, fps, clip, ci = prepare(p, man, meta, e1)
        W, H = int(meta["width"]), int(meta["height"])
        s = OUT_SIZE / ci["side"]
        tb = [(float(meta["x1"]) * W - ci["left"]) * s, (float(meta["y1"]) * H - ci["top"]) * s,
              (float(meta["x2"]) * W - ci["left"]) * s, (float(meta["y2"]) * H - ci["top"]) * s]
        x0 = n * (4 * cell + 20)
        dr.text((x0 + 4, 4), f"{Path(p).stem}: crop x {ci['left']}-{ci['left'] + ci['side']}, y {ci['top']}-"
                f"{ci['top'] + ci['side']} (side {ci['side']} px, shift {ci['shift']})", fill=(255, 255, 255), font=font)
        dr.text((x0 + 4, 24), f"frames {idx[0]}-{idx[-1]} ({idx[0] / fps:.2f}-{idx[-1] / fps:.2f} s); green = true box",
                fill=(255, 255, 255), font=font)
        for k in range(16):
            im = Image.fromarray(frames[k])
            ImageDraw.Draw(im).rectangle(tb, outline=(0, 230, 0), width=3)
            im = im.resize((cell, cell), Image.BICUBIC)
            ImageDraw.Draw(im).text((4, 4), f"{idx[k] / fps:.2f} s", fill=(255, 255, 0), font=font)
            sheet.paste(im, (x0 + (k % 4) * cell, head + (k // 4) * cell))
        res["contact_sheet"].append({"clip": Path(p).stem, "crop": {k: ci[k] for k in ("left", "top", "side", "shift", "centre_source")},
                                     "true_box_in_crop_448": [round(v, 1) for v in tb],
                                     "true_box_inside_crop": bool(tb[0] >= 0 and tb[1] >= 0 and tb[2] <= OUT_SIZE and tb[3] <= OUT_SIZE)})
    sheet.save(SHEET)
    # crop statistics over all 755 (no model; uses only the epoch-1 box and the metadata size)
    sides, shifted = [], 0
    for p, man, meta, e1 in clips:
        _, _, side, ci = crop_square(e1, int(meta["width"]), int(meta["height"]))
        sides.append(side)
        shifted += ci["shift"] != (0, 0)
    res["crop_side_px_quantiles"] = {q: float(np.quantile(sides, q)) for q in (0, 0.25, 0.5, 0.75, 1)}
    res["crop_at_min_256"] = int(sum(s == SIDE_MIN for s in sides))
    res["crop_capped_at_height"] = int(sum(s == 1080 for s in sides))
    res["crop_shifted_inside"] = int(shifted)
    CHECKS_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("hub_files", "without_valid_box")}, indent=1))


def download():
    from huggingface_hub import snapshot_download
    check_plan()
    refuse(DL_JSON)
    import vlm_zero_shot as vz
    on, ptxt = vz.power()
    if not on:
        fail("on battery")
    t0 = time.time()
    path = snapshot_download(REPO_ID, revision=REV)
    dt = time.time() - t0
    files = {f.name: f.stat().st_size for f in Path(path).iterdir() if f.is_file()}
    hub = json.loads(CHECKS_JSON.read_text())["hub_files"]
    missing = sorted(set(hub) - set(files))
    mismatch = sorted(k for k in hub if k in files and hub[k] is not None and files[k] != hub[k])
    res = {"path": path, "seconds": dt, "local_files": files, "local_total_bytes": sum(files.values()),
           "missing_vs_hub": missing, "size_mismatch_vs_hub": mismatch, "power": ptxt}
    DL_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    if missing or mismatch:
        fail("download incomplete or sizes differ from the Hub list")


# ---------------------------------------------------------------- memory and speed (GPU)

def speed():
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    import vlm_smoke_test as smoke
    import vlm_zero_shot as vz
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    check_plan()
    refuse(SPEED_JSON, ANSWERS)
    on, ptxt = vz.power()
    if not on:
        fail("on battery")
    clips = load_clips()
    pick = sorted(np.random.default_rng(1).choice(len(clips), size=5, replace=False).tolist())
    res = {"clips": [Path(clips[j][0]).stem for j in pick], "power_start": ptxt, "gpu": torch.cuda.get_device_name(0),
           "gpu_total_gb": torch.cuda.get_device_properties(0).total_memory / 2 ** 30, "per_clip": []}
    scene = "not given"
    t0 = time.time()
    try:
        proc = AutoProcessor.from_pretrained(REPO_ID, revision=REV)
        model = Qwen3VLForConditionalGeneration.from_pretrained(REPO_ID, revision=REV, dtype=torch.bfloat16,
                                                                device_map={"": 0}, attn_implementation="sdpa").eval()
    except torch.cuda.OutOfMemoryError as e:
        res.update(oom="at model load", error=str(e)[:500], peak_alloc_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
        SPEED_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
        fail("OUT OF MEMORY at model load; stopping (plan unchanged)")
    torch.cuda.synchronize()
    res["load_s"] = time.time() - t0
    res["mem_after_load_alloc_gb"] = torch.cuda.memory_allocated() / 2 ** 30
    res["mem_after_load_reserved_gb"] = torch.cuda.memory_reserved() / 2 ** 30
    res["linear4bit_modules"] = smoke.count_4bit(model)
    res["quantization_config_loaded"] = model.config.quantization_config.to_dict() if hasattr(
        model.config.quantization_config, "to_dict") else model.config.quantization_config
    print(f"loaded in {res['load_s']:.1f} s; allocated {res['mem_after_load_alloc_gb']:.2f} GB; 4-bit modules {res['linear4bit_modules']}")
    text_in = prompt_text(scene)
    with open(ANSWERS, "w", encoding="utf-8") as fa:
        for j in pick:
            p, man, meta, e1 = clips[j]
            torch.cuda.reset_peak_memory_stats()
            t1 = time.time()
            frames, idx, fps, clip, ci = prepare(p, man, meta, e1)
            prep = time.time() - t1
            inputs, _, _ = smoke.build_inputs(proc, frames, idx, fps, clip, prompt_text_in=text_in)
            x = {k: v.to("cuda") for k, v in inputs.items()}
            x["pixel_values_videos"] = x["pixel_values_videos"].to(torch.bfloat16)
            n_in = int(x["input_ids"].shape[1])
            try:
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                with torch.no_grad():
                    out = model.generate(**x, max_new_tokens=MAX_NEW, do_sample=False, use_cache=True,
                                         temperature=None, top_p=None, top_k=None)
                torch.cuda.synchronize()
                gen = time.perf_counter() - t2
            except torch.cuda.OutOfMemoryError as e:
                res["per_clip"].append({"clip": Path(p).stem, "oom": True, "input_tokens": n_in,
                                        "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "error": str(e)[:300]})
                res["oom"] = "at generate with 16 frames"
                SPEED_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
                fail("OUT OF MEMORY with 16 frames; stopping (plan unchanged)")
            raw = proc.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]
            t, why = parse(raw)
            row = {"clip": Path(p).stem, "prep_s": prep, "gen_s": gen, "input_tokens": n_in,
                   "new_tokens": int(out.shape[1] - n_in), "parse_ok": t is not None, "parse_reason": why,
                   "rule_applied": t is not None and apply_rule(t, scene) != t, "crop_source": ci["centre_source"],
                   "crop_side": ci["side"], "crop_shift": ci["shift"],
                   "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                   "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2 ** 30}
            res["per_clip"].append(row)
            fa.write(json.dumps({"clip": Path(p).stem, "raw_answer": raw, "parse_ok": t is not None}) + "\n")
            print(f"{row['clip']}: prep {prep:.2f} s, generate {gen:.2f} s, {row['input_tokens']} input tokens, "
                  f"{row['new_tokens']} new, parse ok {row['parse_ok']}, peak {row['peak_alloc_gb']:.2f} GB")
    on, ptxt = vz.power()
    res["power_end"] = ptxt
    SPEED_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- report

def report():
    check_plan()
    refuse(REPORT)
    C = json.loads(CHECKS_JSON.read_text())
    D = json.loads(DL_JSON.read_text()) if DL_JSON.is_file() else None
    S = json.loads(SPEED_JSON.read_text()) if SPEED_JSON.is_file() else None
    L = ["TYPE MODEL, OPTION 1 (ZERO-SHOT QWEN3-VL-8B ON A CROP): STAGE 1 REPORT (synthetic only; no real file; no frozen "
         "file, checkpoint, split, manifest or earlier result changed; nothing committed; no accuracy computed)",
         f"PLAN.txt sha256 {PLAN_SHA}; PROMPT.txt sha256 {PROMPT_SHA} (prompt used without its final newline; "
         "{scene_layout} replaced).", "",
         "STEP 1, CHECKS (no model)",
         f"  a. free disk {C['disk_free_gb']:.0f} GB of {C['disk_total_gb']:.0f} GB",
         f"  a2. model files on the Hub ({REPO_ID} @ {C['hub_revision']}): {len(C['hub_files'])} files, "
         f"{C['hub_total_bytes']:,} bytes ({C['hub_total_bytes'] / 2 ** 30:.2f} GiB)"
         + (f"; download time {D['seconds']:.0f} s ({D['local_total_bytes'] / 2 ** 30:.2f} GiB on disk; missing "
            f"{D['missing_vs_hub'] or 'none'}, size mismatches {D['size_mismatch_vs_hub'] or 'none'})" if D else "; not downloaded"),
         f"  b. metadata-synthetic.csv columns: {', '.join(C['metadata_columns'])}",
         f"     scene layout column: {C['scene_layout_columns'] or 'none'} -> scene_layout = \"not given\" for all 755 clips; "
         "the scene hint and the t-bone -> rear-end rule are untested on synthetic data (the rule cannot fire).",
         f"  c. valid QLoRA epoch-1 box: {C['n_clips'] - C['n_without_valid_epoch1_box']}/{C['n_clips']} "
         f"(without: {C['n_without_valid_epoch1_box']})",
         f"     crop side (px) over the 755: min {C['crop_side_px_quantiles']['0']:.0f}, quartiles "
         f"{C['crop_side_px_quantiles']['0.25']:.0f} / {C['crop_side_px_quantiles']['0.5']:.0f} / "
         f"{C['crop_side_px_quantiles']['0.75']:.0f}, max {C['crop_side_px_quantiles']['1']:.0f}; at the 256 minimum "
         f"{C['crop_at_min_256']}; capped at the image height {C['crop_capped_at_height']}; moved to fit inside the image "
         f"{C['crop_shifted_inside']}",
         f"  d. contact sheet: {SHEET.name} (3 clips, numpy default_rng(0); 16 crops each; true box in green)"]
    # extra, descriptive (added after seeing the contact sheet; boxes only, no type): where the true box falls
    full = part = out = 0
    for p, man, meta, e1 in load_clips():
        W, H = int(meta["width"]), int(meta["height"])
        left, top, side, _ = crop_square(e1, W, H)
        x1, y1, x2, y2 = (float(meta[k]) * s for k, s in (("x1", W), ("y1", H), ("x2", W), ("y2", H)))
        ix = max(0.0, min(x2, left + side) - max(x1, left)) * max(0.0, min(y2, top + side) - max(y1, top))
        a = (x2 - x1) * (y2 - y1)
        full += ix >= a - 1e-6
        out += ix <= 0
        part += 0 < ix < a - 1e-6
    L.insert(len(L) - 1, f"     extra (descriptive, added after seeing the contact sheet; boxes only): true (metadata) box "
             f"fully inside the crop {full}, partly {part}, outside {out} of 755")
    for c in C["contact_sheet"]:
        L.append(f"     {c['clip']}: crop left {c['crop']['left']}, top {c['crop']['top']}, side {c['crop']['side']}, "
                 f"shift {c['crop']['shift']}; true box inside the crop: {c['true_box_inside_crop']}")
    L += ["", "STEP 2, DOWNLOAD", f"  {'done: ' + D['path'] if D else 'not done'}"]
    # all download attempts, from their run logs (first line = command and start time; last error/exit lines)
    for n, name in enumerate(("run_log_download.txt", "run_log_download_retry1.txt", "run_log_download_retry2_http.txt"), 1):
        f = OUT / name
        if not f.is_file():
            continue
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        err = [s for s in lines if s.startswith("RuntimeError")]
        end = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(f.stat().st_mtime))
        L.append(f"  attempt {n} ({name}): {lines[0]}")
        L.append(f"    log last written {end}; {lines[-1].strip()}" + (f"; {err[-1][:200]}" if err else ""))
    if D:
        L.append(f"  The download time in a2 ({D['seconds']:.0f} s = {D['seconds'] / 3600:.2f} h, "
                 f"{D['local_total_bytes'] / D['seconds'] / 1e6:.2f} MB/s) is that of attempt 3, over plain HTTP "
                 "(HF_HUB_DISABLE_XET=1), after attempts 1 and 2 failed with Xet (CAS) transfer errors; files already "
                 "cached by the failed attempts were reused, so this time may slightly understate a fresh download.")
    L += ["", "STEP 3, MEMORY AND SPEED (5 clips, "
          "numpy default_rng(1); answers parsed only, not compared with the true types; no accuracy)"]
    slog = OUT / "run_log_speed.txt"
    if S is None and slog.is_file():
        # the run crashed before speed_test.json was written: report it from its log, unchanged
        lines = slog.read_text(encoding="utf-8", errors="replace").splitlines()
        keep = [s for s in lines if s.startswith(("command:", "loaded in", "FP4 quantization"))]
        L.append("  RAN AND CRASHED (not out of memory); speed_test.json was not written. From run_log_speed.txt:")
        L += ["    " + s.strip() for s in keep]
        L += ["    " + s.strip() for s in lines[-5:]]
        L += ["  Diagnosis (read-only inspection of the downloaded checkpoint, no model call): the checkpoint stores "
              "all 351 vision-tower tensors as plain bfloat16 with no 4-bit state (e.g. model.visual.blocks.0.attn.qkv.weight "
              "[3456, 1152] BF16), and its quantization_config lists 'visual' in llm_int8_skip_modules; but loading it as "
              "pre-registered (transformers 5.22.1, bitsandbytes 0.50.2, no BitsAndBytesConfig passed) built 116 vision "
              "modules as 4-bit layers ('4-bit modules' line above), so they received bfloat16 weights without a 4-bit "
              "quantisation state and the first forward pass through the vision tower failed the bitsandbytes assertion "
              "(module.weight.shape[1] == 1). The skip list was not applied to the vision tower.",
              "  No clip was processed: no generate time, no per-clip peak memory and no parsed answer exist. Only the "
              "model load is measured (time and memory after load, above). speed_test_answers.jsonl was created empty "
              "(0 bytes) before the crash and is left as it is.",
              "  RESULT: stopped. The plan is not changed; making the model run needs a change to the pre-registered load "
              "arguments, which is a decision for the user."]
    elif S is None:
        L.append("  not run")
    else:
        L.append(f"  GPU {S['gpu']} ({S['gpu_total_gb']:.2f} GB); power {S['power_start']}")
        if "load_s" in S:
            L.append(f"  model load {S['load_s']:.1f} s (not counted per clip); after load allocated "
                     f"{S['mem_after_load_alloc_gb']:.2f} GB, reserved {S['mem_after_load_reserved_gb']:.2f} GB; 4-bit linear "
                     f"modules {S['linear4bit_modules']}")
        for r in S["per_clip"]:
            if r.get("oom"):
                L.append(f"  {r['clip']}: OUT OF MEMORY at generate ({r['input_tokens']} input tokens; peak allocated "
                         f"{r['peak_alloc_gb']:.2f} GB)")
            else:
                L.append(f"  {r['clip']}: frames {r['prep_s']:.2f} s (CPU), generate {r['gen_s']:.2f} s (GPU), "
                         f"{r['input_tokens']} input tokens, {r['new_tokens']} new tokens, parses: {r['parse_ok']}"
                         f"{' (' + r['parse_reason'] + ')' if r['parse_reason'] else ''}; peak allocated {r['peak_alloc_gb']:.2f} GB, "
                         f"reserved {r['peak_reserved_gb']:.2f} GB")
        ok = [r for r in S["per_clip"] if not r.get("oom")]
        if S.get("oom"):
            L.append(f"  RESULT: OUT OF MEMORY ({S['oom']}); stopped as planned, plan unchanged.")
        elif ok:
            per = [r["prep_s"] + r["gen_s"] for r in ok]
            L += [f"  mean per clip: frames {np.mean([r['prep_s'] for r in ok]):.2f} s + generate "
                  f"{np.mean([r['gen_s'] for r in ok]):.2f} s = {np.mean(per):.2f} s; max peak allocated "
                  f"{max(r['peak_alloc_gb'] for r in ok):.2f} GB (reserved {max(r['peak_reserved_gb'] for r in ok):.2f} GB); "
                  f"answers that parse {sum(r['parse_ok'] for r in ok)}/{len(ok)}",
                  f"  estimate for the 755 clips at this rate: {755 * np.mean(per) / 3600:.2f} h plus one model load",
                  f"  power at end {S.get('power_end')}"]
    L += ["", "Raw answers of the 5 test clips: speed_test_answers.jsonl (no true types stored with them).",
          "Files: PLAN.txt, PROMPT.txt, checks.json, contact_sheet_crops.png, download.json, speed_test.json, "
          "speed_test_answers.jsonl, run_log_*.txt, stage1_report.txt."]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    modes = ("checks", "download", "speed", "report")
    for m in modes:
        g.add_argument(f"--{m}", action="store_true")
    a = ap.parse_args()
    {"checks": checks, "download": download, "speed": speed, "report": report}[next(m for m in modes if getattr(a, m))]()


if __name__ == "__main__":
    main()
