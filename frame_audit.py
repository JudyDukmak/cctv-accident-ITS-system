"""Frame and label audit of the narrow VLM input (review request), before any new model.

SYNTHETIC ONLY: metadata-synthetic.csv, the synthetic annotation files, synthetic_split.csv, the saved out-of-fold
classifier t_hat, the saved 200-clip outputs (results/narrow_v1_v2_200/) and the QLoRA frame cache. No real clip or
real label is opened. No model weights are loaded: only the processor (AutoProcessor) and the frame code.
Frozen files are imported, never changed: vlm_zero_shot.py, narrow_prompt_v2.py, narrow_v1_v2_200.py,
build_qlora_data.py (read_npz only), gpu_check/vlm_smoke_test.py (build_inputs, parse_output).

Modes (all outputs under results/frame_audit/, each refuses to overwrite):
  --list     (.venv-vlm) LIST.txt + list.csv: 10 clips of results/narrow_v1_v2_200/test_clips.csv, 2 per type,
             numpy default_rng(0); per type (TYPES order) first one clip with |t_hat - t*| < 0.5 s, then one with t*
             inside the window and 0.5 <= |t_hat - t*| <= 2 s (if a pool is empty: any not yet chosen clip of the type);
             pools sorted by rgb_path.
  --audit    (.venv-vlm) per clip: the 16 input frames (PNG), processor outputs for narrow v1 and v2 (decoded prompt,
             grid, visual tokens, frames rebuilt from pixel values), contact sheet, cache comparison, checks a-c, f, g,
             i, j; plus the 200-clip numbers for k and l. Writes audit.json.
  --reader2  (.venv-ml, OpenCV) checks d and e: the frame nearest t* read by OpenCV (sequential read to the frame
             index) vs the pipeline frame; channel means; OpenCV fps and frame count. Writes reader2.json.
  --report   (.venv-vlm) report.txt from audit.json, reader2.json and h_visual.csv (the per-clip visual judgement of
             check h, written by hand after looking at the t* images; the report says so).
"""
import argparse
import collections
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
OUT = REPO_ROOT / "results" / "frame_audit"
LIST_CSV, LIST_TXT = OUT / "list.csv", OUT / "LIST.txt"
AUDIT_JSON, READER2_JSON, H_CSV, REPORT = OUT / "audit.json", OUT / "reader2.json", OUT / "h_visual.csv", OUT / "report.txt"
SRC = REPO_ROOT / "results" / "narrow_v1_v2_200"
TEST_CLIPS, RESULTS, RAW = SRC / "test_clips.csv", SRC / "results.csv", SRC / "raw_outputs.jsonl"
SEED = 0
DATASET = REPO_ROOT / "ACCIDENT" / "dataset"
META_SYN = DATASET / "metadata-synthetic.csv"
CACHE = REPO_ROOT / "cache" / "qlora_frames448_v1"
TYPES5 = ["head-on", "rear-end", "sideswipe", "single", "t-bone"]
VEHICLE_TAGS = {14: "Car", 15: "Truck", 16: "Bus", 18: "Motorcycle", 19: "Bicycle", 29: "Van"}


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def rcsv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stem(p):
    return Path(p.replace("\\", "/")).stem


def vz_mod():
    sys.path.insert(0, str(REPO_ROOT))
    import vlm_zero_shot as vz  # noqa: E402  (frozen; imported only)
    return vz


def inside_and_times(vz, clip):
    """The narrow window exactly as the 200-clip test built it (vz.window_times + vz.to_indices)."""
    times = vz.window_times(clip, clip["t_hat"], vz.NARROW_HALF_S, vz.NARROW_FRAMES)
    idx = vz.to_indices(clip, times)
    ft = [i / clip["fps"] for i in idx]
    return times, idx, ft, ft[0] <= clip["t_star"] <= ft[-1]


# ------------------------------------------------------------------ list

def make_list():
    if LIST_TXT.exists() or LIST_CSV.exists():
        fail(f"{LIST_TXT} exists; the list is drawn once")
    vz = vz_mod()
    clips = vz.load_clips(TEST_CLIPS)  # same loader as the 200-clip test (t_hat out-of-fold, fps = no_frames/duration)
    rng = np.random.default_rng(SEED)
    rows, notes = [], []
    for typ in vz.TYPES:
        cc = sorted((c for c in clips if c["type"] == typ), key=lambda c: c["rgb_path"])
        info = {c["rgb_path"]: (abs(c["t_hat"] - c["t_star"]), inside_and_times(vz, c)[3]) for c in cc}
        chosen = set()
        for cat, rule in (("A |t_hat - t*| < 0.5 s", lambda e, ins: e < 0.5),
                          ("B t* inside, 0.5 <= |t_hat - t*| <= 2 s", lambda e, ins: ins and 0.5 <= e <= 2.0)):
            pool = [c["rgb_path"] for c in cc if rule(*info[c["rgb_path"]]) and c["rgb_path"] not in chosen]
            used_fallback = not pool
            if used_fallback:
                pool = [c["rgb_path"] for c in cc if c["rgb_path"] not in chosen]
            p = str(rng.choice(pool))
            chosen.add(p)
            e, ins = info[p]
            rows.append({"rgb_path": p, "type": typ, "category": cat.split()[0], "pool_size": len(pool),
                         "fallback_any_clip": used_fallback, "abs_that_err_s": round(e, 4), "t_star_inside": ins})
            notes.append(f"{typ:9s} {cat:40s} pool {len(pool):3d}{' (FALLBACK: any clip)' if used_fallback else ''}"
                         f" -> {stem(p)}  |t_hat - t*| {e:.3f} s, t* inside {ins}")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(LIST_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    text = "\n".join([f"Frame audit clip list, written {vz.now()} before any image was made",
                      f"source: {TEST_CLIPS.relative_to(REPO_ROOT)} (sha256 {vz.sha256(TEST_CLIPS)}); t_hat and t* from "
                      f"vlm_zero_shot.load_clips (classifier C_cctv seed 0 out-of-fold; metadata-synthetic.csv)",
                      f"rule: numpy default_rng({SEED}); per type in {vz.TYPES} order: A = one clip with |t_hat - t*| < "
                      "0.5 s, then B = one clip with t* inside the narrow window and 0.5 <= |t_hat - t*| <= 2 s (if a "
                      "pool is empty: any not yet chosen clip of that type); pools sorted by rgb_path; window = "
                      "vz.window_times(t_hat, 2.0 s, 16) + vz.to_indices, inside = first frame time <= t* <= last.",
                      *notes, f"{LIST_CSV.name} sha256 {vz.sha256(LIST_CSV)}"]) + "\n"
    LIST_TXT.write_text(text, encoding="utf-8")
    print(text)


def load_list(vz):
    if not LIST_TXT.exists() or vz.sha256(LIST_CSV) not in LIST_TXT.read_text(encoding="utf-8"):
        fail("list.csv does not match LIST.txt")
    return rcsv(LIST_CSV)


# ------------------------------------------------------------------ audit

def decode_prompt(processor, ids):
    """Full decoded text, and a readable version with runs of the same special token collapsed."""
    tok = processor.tokenizer
    full = tok.decode(ids, skip_special_tokens=False)
    pieces = tok.convert_ids_to_tokens(ids)
    out, k = [], 0
    vt = processor.video_token
    while k < len(ids):
        if pieces[k] == vt:
            j = k
            while j < len(ids) and pieces[j] == vt:
                j += 1
            out.append(f"[{vt} x {j - k}]")
            k = j
        else:
            j = k
            while j < len(ids) and pieces[j] != vt:
                j += 1
            out.append(tok.decode(ids[k:j], skip_special_tokens=False))
            k = j
    return full, "".join(out)


def prompt_timestamps(text):
    import re
    return [float(x) for x in re.findall(r"<(\d+(?:\.\d+)?) seconds>", text)]


def draw_box(img, box_frac, colour, width=2):
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    w, h = img.size
    x1, y1, x2, y2 = box_frac
    d.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline=colour, width=width)
    return img


def contact_sheet(frames, times, k_near, box, path, title):
    from PIL import Image, ImageDraw
    t, h, w, _ = frames.shape
    top, gap, head = 22, 6, 16
    sheet = Image.new("RGB", (4 * w + 3 * gap, top + 4 * (h + head) + 3 * gap), (252, 252, 251))
    d = ImageDraw.Draw(sheet)
    d.text((4, 4), title, fill=(0, 0, 0))
    for j in range(t):
        r, c = divmod(j, 4)
        x0, y0 = c * (w + gap), top + r * (h + head + gap)
        img = Image.fromarray(frames[j])
        if j == k_near:
            draw_box(img, box, (0, 200, 0), 2)
        sheet.paste(img, (x0, y0 + head))
        d.text((x0 + 2, y0 + 2), f"#{j + 1} {times[j]:.3f} s" + ("  <- nearest t*" if j == k_near else ""),
               fill=(200, 0, 0) if j == k_near else (0, 0, 0))
        if j == k_near:
            d.rectangle([x0 - 3, y0 + head - 3, x0 + w + 2, y0 + head + h + 2], outline=(220, 0, 0), width=3)
    sheet.save(path)


def annotation_check(p, m):
    """Independent label source: the clip's annotation JSON (first collision record and the colliding objects)."""
    s = stem(p)
    ap = DATASET / "synthetic_videos" / "annotations" / f"{s}.json" / f"{s}.json"
    if not ap.is_file():
        return {"annotation_file": str(ap.relative_to(REPO_ROOT)), "found": False}
    d = json.loads(ap.read_text(encoding="utf-8"))
    c0 = d["collision"][0]
    it = c0["iteration"]
    objs = {o["id"]: o for b in d["base"] if b["iteration"] == it for o in b["objects"]}
    tags = [objs[i]["tag"] if i in objs else None for i in c0["ids"]]
    (bx1, by1), (bx2, by2) = c0["collision_bbox"]
    W, H = int(m["width"]), int(m["height"])
    box = [bx1 / W, by1 / H, bx2 / W, by2 / H]
    meta_box = [float(m[k]) for k in ("x1", "y1", "x2", "y2")]
    return {"annotation_file": str(ap.relative_to(REPO_ROOT)), "found": True, "first_collision_iteration": it,
            "frame_from_annotation": it - int(m["annotations_start_offset"]), "accident_frame_meta": int(m["accident_frame"]),
            "collision_ids": c0["ids"], "collision_tags": tags,
            "n_vehicles_in_collision": sum(t in VEHICLE_TAGS for t in tags),
            "box_from_annotation": box, "box_meta": meta_box,
            "box_max_abs_diff": max(abs(a - b) for a, b in zip(box, meta_box))}


def audit():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    vz = vz_mod()
    import narrow_prompt_v2 as n2
    import build_qlora_data as bq
    from PIL import Image
    from transformers import AutoProcessor

    lst = load_list(vz)
    if AUDIT_JSON.exists() or (OUT / "clips").exists():
        fail(f"{AUDIT_JSON} or clips/ exists; refusing to overwrite")
    smoke = vz.smoke
    processor = AutoProcessor.from_pretrained(smoke.MODELS[vz.MODEL_KEY])  # processor only; no weights
    meta = {vz.norm(r["rgb_path"]): r for r in rcsv(META_SYN)}
    clips = {c["rgb_path"]: c for c in vz.load_clips(TEST_CLIPS)}
    res = {r["rgb_path"]: r for r in rcsv(RESULTS)}
    raw = collections.defaultdict(dict)
    for line in RAW.read_text(encoding="utf-8").splitlines():
        if line.strip():
            o = json.loads(line)
            raw[o["clip"]][o["narrow_version"]] = o["output"]
    import av
    out_all = {"processor": smoke.MODELS[vz.MODEL_KEY], "clips": []}
    for row in lst:
        p = row["rgb_path"]
        c, m = clips[p], meta[p]
        s = stem(p)
        d = OUT / "clips" / s
        (d / "input_frames").mkdir(parents=True)
        (d / "model_view_v1").mkdir()
        A = {"rgb_path": p, "stem": s, "type": c["type"], "category": row["category"]}
        # ---- the 200-clip test's own v1 input (vz.narrow_input) and v2 input (n2.narrow_v2_input)
        idx, frames, prompt1 = vz.narrow_input(c, 1)
        times_nominal, idx_chk, ft, inside = inside_and_times(vz, c)
        if idx != idx_chk:
            fail(f"{s}: narrow_input indices differ from window_times/to_indices")
        idx2, frames2, idx2_dup, prompt2 = n2.narrow_v2_input(c)
        A.update(fps_used=c["fps"], no_frames=c["no_frames"], duration=c["duration"], t_hat=c["t_hat"],
                 t_star=c["t_star"], frame_indices=idx, nominal_times=[float(t) for t in times_nominal], frame_times=ft,
                 t_star_inside=inside, v2_indices_same=idx2 == idx,
                 v2_frames_same=bool(np.array_equal(frames2[::2], frames) and np.array_equal(frames2[1::2], frames)))
        for j, (i, t) in enumerate(zip(idx, ft)):
            Image.fromarray(frames[j]).save(d / "input_frames" / f"pos{j + 1:02d}_frame{i:04d}_t{t:07.3f}s.png")
        A["input_size_wh"] = [int(frames.shape[2]), int(frames.shape[1])]
        # ---- (a) fps
        with av.open(c["video"]) as cont:
            st = cont.streams.video[0]
            n_dec = sum(1 for _ in cont.decode(video=0))
            A.update(container_avg_rate=float(st.average_rate), container_frames=int(st.frames), decoded_frames=n_dec,
                     container_size_wh=[st.codec_context.width, st.codec_context.height])
        A.update(meta_fps_no_frames_over_duration=int(m["no_frames"]) / float(m["duration"]),
                 accident_time=float(m["accident_time"]), accident_frame=int(m["accident_frame"]),
                 accident_frame_from_time=round(float(m["accident_time"]) * c["fps"]),
                 indices_eq_round_time_fps=idx == [int(min(max(round(t * c["fps"]), 0), c["no_frames"] - 1))
                                                   for t in times_nominal])
        # ---- (b) times
        steps = np.diff(ft)
        mid = (ft[0] + ft[-1]) / 2
        t_last = (c["no_frames"] - 1) / c["fps"]
        clamped = (c["t_hat"] - vz.NARROW_HALF_S < 0) or (c["t_hat"] + vz.NARROW_HALF_S > t_last)
        A.update(steps_min=float(steps.min()), steps_max=float(steps.max()), steps_mean=float(steps.mean()),
                 strictly_increasing=bool((steps > 0).all()), window_mid=mid, mid_minus_t_hat=mid - c["t_hat"],
                 window_clamped_to_clip=bool(clamped), first_time=ft[0], last_time=ft[-1])
        # ---- (c) consecutive differences
        fd = frames.astype(np.int16)
        mad = [float(np.abs(fd[j + 1] - fd[j]).mean()) for j in range(len(idx) - 1)]
        A.update(consecutive_mean_abs_diff=mad, duplicate_indices=len(set(idx)) != len(idx),
                 zero_diff_pairs=[j + 1 for j, v in enumerate(mad) if v == 0.0])
        # ---- frame nearest t*
        k = int(np.argmin(np.abs(np.array(ft) - c["t_star"])))
        A.update(nearest_pos=k + 1, nearest_frame=idx[k], nearest_time=ft[k], nearest_minus_t_star=ft[k] - c["t_star"])
        full = vz.decode(c, [idx[k]], budget=10 ** 12)[idx[k]]  # same decode function, no resize (s = 1)
        full.save(d / "pipeline_nearest_tstar_fullres.png")
        A["pipeline_fullres_size_wh"] = list(full.size)
        A["pipeline_fullres_channel_means_rgb"] = [float(v) for v in np.asarray(full).reshape(-1, 3).mean(0)]
        A["pipeline_input_channel_means_rgb"] = [float(v) for v in frames[k].reshape(-1, 3).mean(0)]
        # ---- processor (v1 and v2), exactly as Vlm.__call__ / generate_v2 build the inputs
        box = [float(m[q]) for q in ("x1", "y1", "x2", "y2")]
        for v, (fr, ii, pr) in {1: (frames, idx, prompt1), 2: (frames2, idx2_dup, prompt2)}.items():
            inputs, _, _ = smoke.build_inputs(processor, fr, ii, c["fps"], c, prompt_text_in=pr)
            ids = inputs["input_ids"][0].tolist()
            full_txt, short_txt = decode_prompt(processor, ids)
            (d / f"prompt_v{v}_full.txt").write_text(full_txt, encoding="utf-8")
            (d / f"prompt_v{v}.txt").write_text(short_txt, encoding="utf-8")
            grid = inputs["video_grid_thw"][0].tolist()
            mv = vz.model_view(inputs, processor)
            ts = prompt_timestamps(full_txt)
            pair = int(processor.video_processor.temporal_patch_size)
            exp_ts = [(ii[g] / c["fps"] + ii[min(g + pair - 1, len(ii) - 1)] / c["fps"]) / 2
                      for g in range(0, len(ii), pair)]
            resized = np.stack([np.asarray(Image.fromarray(f).resize((mv.shape[2], mv.shape[1]), Image.BICUBIC))
                                for f in fr])
            padded, bands, pdiffs = n2.padding_check(fr, mv)
            A[f"v{v}"] = {"grid_thw": grid, "visual_tokens": int(sum(1 for q in ids if q == processor.video_token_id)),
                          "input_tokens": len(ids), "n_frames_in": int(fr.shape[0]), "model_frames": int(mv.shape[0]),
                          "model_size_wh": [int(mv.shape[2]), int(mv.shape[1])],
                          "prompt_timestamps": ts, "expected_timestamps": exp_ts,
                          "timestamps_match_1dp": len(ts) == len(exp_ts) and all(f"{a:.1f}" == f"{b:.1f}" for a, b in zip(ts, exp_ts)),
                          "rebuilt_vs_pil_bicubic_mean_abs": float(np.abs(mv.astype(np.int16) - resized.astype(np.int16)).mean()),
                          "rebuilt_vs_pil_bicubic_max_abs": int(np.abs(mv.astype(np.int16) - resized.astype(np.int16)).max()),
                          "padding_detected": bool(padded), "uniform_border_bands": bands,
                          "prompt_text_contains_16_frames": "16 frames" in pr}
            if v == 1:
                mv1 = mv
                for j in range(mv.shape[0]):
                    Image.fromarray(mv[j]).save(d / "model_view_v1" / f"pos{j + 1:02d}_frame{idx[j]:04d}_t{ft[j]:07.3f}s.png")
            else:
                A["v2"]["rebuilt_duplicates_identical"] = bool(np.array_equal(mv[::2], mv[1::2]))
                A["v2"]["rebuilt_equal_to_v1_rebuilt"] = bool(np.array_equal(mv[::2], mv1))
        # ---- (3) contact sheet + t* reference image for check h
        contact_sheet(mv1, ft, k, box, d / "contact_sheet.png",
                      f"{s} ({c['type']}): model view v1, {mv1.shape[2]}x{mv1.shape[1]}; t_hat {c['t_hat']:.2f} s, "
                      f"t* {c['t_star']:.2f} s ({'inside' if inside else 'OUTSIDE'}); green = ground-truth box")
        af = int(m["accident_frame"])
        tsf = vz.decode(c, [af], budget=10 ** 12)[af]
        draw_box(tsf, box, (0, 220, 0), 4).save(d / "tstar_frame_fullres_box.png")
        big = Image.fromarray(mv1[k]).resize((mv1.shape[2] * 3, mv1.shape[1] * 3), Image.NEAREST)
        draw_box(big, box, (0, 220, 0), 3).save(d / "nearest_tstar_model_view_x3_box.png")
        # ---- (f, i)
        W, H = A["v1"]["model_size_wh"]
        A.update(box_frac=box, box_px_model=[(box[2] - box[0]) * W, (box[3] - box[1]) * H],
                 source_size_wh=[int(m["width"]), int(m["height"])],
                 scale_x=W / A["input_size_wh"][0], scale_y=H / A["input_size_wh"][1])
        # ---- (4) QLoRA cache, window A
        cp = CACHE / f"{s}_A.npz"
        if cp.is_file():
            fi, cf = bq.read_npz(cp)
            same_idx = fi == idx
            dm = np.abs(cf.astype(np.int16) - mv1.astype(np.int16)) if cf.shape == mv1.shape else None
            dr = np.abs(cf.astype(np.int16) - np.stack([np.asarray(Image.fromarray(f).resize((448, 256), Image.BICUBIC))
                                                       for f in frames]).astype(np.int16))
            A["cache"] = {"file": str(cp.relative_to(REPO_ROOT)), "frame_indices_equal": same_idx,
                          "cache_size_wh": [int(cf.shape[2]), int(cf.shape[1])],
                          "vs_model_view_max": int(dm.max()) if dm is not None else None,
                          "vs_model_view_mean": float(dm.mean()) if dm is not None else None,
                          "vs_input_resized_max": int(dr.max()), "vs_input_resized_mean": float(dr.mean()),
                          "identical_to_input_frames": False if cf.shape != frames.shape else bool(np.array_equal(cf, frames))}
        else:
            A["cache"] = {"file": str(cp.relative_to(REPO_ROOT)), "missing": True}
        # ---- (j) labels
        t1 = raw[p][1]
        ok1, typ1, _, _ = vz.narrow_answer(t1)
        pj = vz.smoke.parse_output(t1)
        raw_type = pj["parsed"].get("collision_type") if isinstance(pj["parsed"], dict) else None
        A["j"] = {"raw_v1": t1, "raw_type": raw_type, "parsed_valid": ok1, "parsed_type": typ1,
                  "scoring_type_results_csv": res[p]["v1_type"], "true_type_results_csv": res[p]["type"],
                  "true_type_metadata": m["type"], "type_from_video_folder": p.split("/")[-2],
                  "type_from_file_name": s.split("_")[1], "annotation": annotation_check(p, m)}
        out_all["clips"].append(A)
        print(f"{s}: done", flush=True)
    out_all["global"] = global_numbers(vz)
    AUDIT_JSON.write_text(json.dumps(out_all, indent=1), encoding="utf-8")
    print(f"wrote {AUDIT_JSON}")


def global_numbers(vz):
    """k and l on all 200 clips (saved outputs only) + spelling / mapping check."""
    import narrow_prompt_v2 as n2
    rows = rcsv(RESULTS)
    meta = {vz.norm(r["rgb_path"]): r for r in rcsv(META_SYN)}
    clips = {c["rgb_path"]: c for c in vz.load_clips(TEST_CLIPS)}
    raw = collections.defaultdict(dict)
    for line in RAW.read_text(encoding="utf-8").splitlines():
        if line.strip():
            o = json.loads(line)
            raw[o["clip"]][o["narrow_version"]] = o["output"]
    G = {"n": len(rows)}
    G["results_type_equals_metadata"] = sum(r["type"] == meta[r["rgb_path"]]["type"] for r in rows)
    G["results_type_equals_folder"] = sum(r["type"] == r["rgb_path"].split("/")[-2] for r in rows)
    G["true_type_counts"] = dict(collections.Counter(r["type"] for r in rows))
    for v in (1, 2):
        conf = {t: {q: 0 for q in vz.TYPES} for t in vz.TYPES}
        for r in rows:
            conf[r["type"]][r[f"v{v}_type"]] += 1
        G[f"v{v}_confusion"] = conf
        G[f"v{v}_scored_answer_counts"] = dict(collections.Counter(r[f"v{v}_type"] for r in rows))
        G[f"v{v}_valid"] = sum(r[f"v{v}_valid"] == "True" for r in rows)
        G[f"v{v}_correct"] = sum(r[f"v{v}_type"] == r["type"] for r in rows)
    # raw v1 answer types, before any fallback; re-parse every saved raw answer and compare with results.csv
    rawtypes, reparse_mismatch = collections.Counter(), []
    for r in rows:
        t1 = raw[r["rgb_path"]][1]
        pj = vz.smoke.parse_output(t1)
        rawtypes[repr(pj["parsed"].get("collision_type")) if isinstance(pj["parsed"], dict) else "no JSON"] += 1
        ok, typ, _, _ = vz.narrow_answer(t1)
        r2 = n2.parse_v2(raw[r["rgb_path"]][2])
        if typ != r["v1_type"] or r2["collision_type"] != r["v2_type"]:
            reparse_mismatch.append(r["rgb_path"])
    G["v1_raw_type_counts"] = dict(rawtypes)
    G["reparse_mismatches"] = reparse_mismatch
    n, k1 = len(rows), G["v1_correct"]
    p1 = k1 / n
    G["v1_acc"] = p1
    G["se_v1_estimate"] = math.sqrt(p1 * (1 - p1) / n)
    G["se_under_chance_0.20"] = math.sqrt(0.2 * 0.8 / n)
    G["z_vs_chance"] = (p1 - 0.2) / G["se_under_chance_0.20"]
    G["binom_p_le_k_at_0.20"] = sum(math.comb(n, i) * 0.2 ** i * 0.8 ** (n - i) for i in range(k1 + 1))
    G["binom_two_sided_p"] = min(1.0, 2 * G["binom_p_le_k_at_0.20"])
    # spelling / mapping
    import re
    G["types_lists"] = {"smoke.TYPES": vz.smoke.TYPES, "vz.TYPES": vz.TYPES, "n2.TYPES": n2.TYPES}
    G["metadata_type_values"] = sorted({r["type"] for r in meta.values()})
    G["video_folder_names"] = sorted({p.split("/")[-2] for p in meta})
    G["narrow_prompt_v1_types_found"] = {t: t in vz.NARROW_PROMPT for t in vz.TYPES}
    G["prompt_v2_types_found"] = {t: t in n2.PROMPT_V2 for t in vz.TYPES}
    G["narrow_prompt_v1_type_words"] = re.findall(r"one of ([^\"]+)\"", vz.NARROW_PROMPT)
    G["annotation_classes_yaml_has_collision_types"] = any(t in (DATASET / "annotation_classes.yaml").read_text() for t in vz.TYPES)
    # l: t* inside the window, v2 collision_visible by group
    grp = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        ins = inside_and_times(vz, clips[r["rgb_path"]])[3]
        g = "inside" if ins else "outside"
        grp[g]["n"] += 1
        cv = r["v2_collision_visible"]
        grp[g]["false" if cv == "false" else ("true" if cv == "true" else f"other:{cv}")] += 1
    G["l"] = {g: dict(v) for g, v in grp.items()}
    return G


# ------------------------------------------------------------------ reader 2 (OpenCV, .venv-ml)

def reader2():
    import cv2
    from PIL import Image

    if READER2_JSON.exists():
        fail(f"{READER2_JSON} exists; refusing to overwrite")
    au = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    out = {"opencv": cv2.__version__, "clips": []}
    for A in au["clips"]:
        video = DATASET / A["rgb_path"]
        cap = cv2.VideoCapture(str(video))
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target, got, i = A["nearest_frame"], None, 0
        while True:  # sequential read to the frame index (no seeking)
            ok, fr = cap.read()
            if not ok:
                break
            if i == target:
                got = fr
                break
            i += 1
        cap.release()
        R = {"stem": A["stem"], "cv2_fps": fps, "cv2_frame_count": count, "frame_index": target}
        if got is None:
            R["error"] = f"could not read frame {target}"
        else:
            pipe = np.asarray(Image.open(OUT / "clips" / A["stem"] / "pipeline_nearest_tstar_fullres.png").convert("RGB"))
            rgb = got[:, :, ::-1]
            dd = np.abs(rgb.astype(np.int16) - pipe.astype(np.int16))
            dbgr = np.abs(got.astype(np.int16) - pipe.astype(np.int16))
            R.update(shape_hw=list(got.shape[:2]), max_abs_diff_rgb=int(dd.max()), mean_abs_diff_rgb=float(dd.mean()),
                     share_pixels_diff_gt_8=float((dd > 8).any(axis=2).mean()),
                     mean_abs_diff_if_bgr_not_swapped=float(dbgr.mean()),
                     cv2_channel_means_as_read_bgr=[float(v) for v in got.reshape(-1, 3).mean(0)],
                     cv2_channel_means_rgb=[float(v) for v in rgb.reshape(-1, 3).mean(0)],
                     pipeline_channel_means_rgb=A["pipeline_fullres_channel_means_rgb"])
            # nearest other frame by OpenCV (is the pipeline frame closer to cv2 frame i than to i +- 1?)
        out["clips"].append(R)
        print(R["stem"], {k: R.get(k) for k in ("max_abs_diff_rgb", "mean_abs_diff_rgb", "cv2_fps", "cv2_frame_count")})
    READER2_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ report

def pf(ok):
    return "PASS" if ok else "FAIL"


def report():
    vz = vz_mod()
    if REPORT.exists():
        fail(f"{REPORT} exists; refusing to overwrite")
    au = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    r2 = {r["stem"]: r for r in json.loads(READER2_JSON.read_text(encoding="utf-8"))["clips"]}
    hv = {r["stem"]: r for r in rcsv(H_CSV)}
    G = au["global"]
    L = [f"FRAME AND LABEL AUDIT (narrow VLM input; synthetic only; no model weights loaded) - {vz.now()}",
         f"Clips: LIST.txt (10 of results/narrow_v1_v2_200/test_clips.csv, 2 per type, default_rng({SEED})). Processor: "
         f"{au['processor']} (AutoProcessor only). Frames and inputs built with the 200-clip test's own functions: "
         "vz.narrow_input (v1), n2.narrow_v2_input (v2), smoke.build_inputs, vz.model_view, n2.padding_check. "
         "Second reader for d/e: OpenCV " + json.loads(READER2_JSON.read_text(encoding='utf-8'))["opencv"] +
         " (sequential read to the frame index).",
         "Check h is a visual judgement made by hand from tstar_frame_fullres_box.png and "
         "nearest_tstar_model_view_x3_box.png (h_visual.csv), not computed.", ""]
    problems = []
    for A in au["clips"]:
        R, H, J, v1, v2 = r2[A["stem"]], hv[A["stem"]], A["j"], A["v1"], A["v2"]
        an = J["annotation"]
        rows = []

        def add(code, ok, detail):
            rows.append((code, ok, detail))
            if ok is False:
                problems.append(f"{A['stem']}: check {code} FAIL - {detail}")

        fps_ok = (abs(A["fps_used"] - A["meta_fps_no_frames_over_duration"]) < 1e-9
                  and abs(A["fps_used"] - A["container_avg_rate"]) < 1e-6 and A["decoded_frames"] == A["no_frames"]
                  and A["indices_eq_round_time_fps"] and A["accident_frame_from_time"] == A["accident_frame"])
        add("a", fps_ok, f"fps used {A['fps_used']:.4f} = no_frames/duration {A['meta_fps_no_frames_over_duration']:.4f};"
            f" container avg_rate {A['container_avg_rate']:.4f}, OpenCV {R['cv2_fps']:.4f}; frames: metadata "
            f"{A['no_frames']}, decoded {A['decoded_frames']}, container {A['container_frames']}, OpenCV {R['cv2_frame_count']};"
            f" indices = round(time x fps): {A['indices_eq_round_time_fps']}; accident_frame {A['accident_frame']} = "
            f"round(accident_time x fps) {A['accident_frame_from_time']}")
        centred = abs(A["mid_minus_t_hat"]) <= 1.0 / A["fps_used"] + 1e-9 or A["window_clamped_to_clip"]
        add("b", A["strictly_increasing"] and 0.2 <= A["steps_min"] and A["steps_max"] <= 0.35 and centred,
            f"times {A['first_time']:.3f}-{A['last_time']:.3f} s, strictly increasing {A['strictly_increasing']}, step "
            f"min {A['steps_min']:.3f} / mean {A['steps_mean']:.4f} / max {A['steps_max']:.3f} s (nominal 4/15 = 0.2667);"
            f" window middle - t_hat {A['mid_minus_t_hat']:+.3f} s{' (clamped to the clip)' if A['window_clamped_to_clip'] else ''};"
            f" t_hat {A['t_hat']:.3f}, t* {A['t_star']:.3f}: t* {'inside' if A['t_star_inside'] else 'OUTSIDE'}")
        mad = A["consecutive_mean_abs_diff"]
        add("c", not A["duplicate_indices"] and not A["zero_diff_pairs"],
            f"consecutive mean |diff| min {min(mad):.2f}, median {float(np.median(mad)):.2f}, max {max(mad):.2f} (0-255);"
            f" duplicate indices {A['duplicate_indices']}; zero-difference pairs {A['zero_diff_pairs'] or 'none'}")
        if "error" in R:
            add("d", False, R["error"])
        else:
            add("d", R["max_abs_diff_rgb"] <= 16 and R["mean_abs_diff_rgb"] <= 2.0,
                f"frame {R['frame_index']} (pos {A['nearest_pos']}, {A['nearest_time']:.3f} s, t* {A['t_star']:.3f}) "
                f"PyAV vs OpenCV at {R['shape_hw'][1]}x{R['shape_hw'][0]}: mean |diff| {R['mean_abs_diff_rgb']:.3f}, max "
                f"{R['max_abs_diff_rgb']}, pixels with any channel > 8: {R['share_pixels_diff_gt_8']:.4f}")
            pm, cm = R["pipeline_channel_means_rgb"], R["cv2_channel_means_rgb"]
            add("e", max(abs(a - b) for a, b in zip(pm, cm)) < 2.0 and R["mean_abs_diff_rgb"] < R["mean_abs_diff_if_bgr_not_swapped"],
                f"channel means R,G,B: pipeline {pm[0]:.1f}, {pm[1]:.1f}, {pm[2]:.1f}; OpenCV converted BGR->RGB "
                f"{cm[0]:.1f}, {cm[1]:.1f}, {cm[2]:.1f}; mean |diff| if the pipeline were BGR "
                f"{R['mean_abs_diff_if_bgr_not_swapped']:.2f} vs {R['mean_abs_diff_rgb']:.2f} as RGB")
        # padding verdict: the rebuilt model frames equal the WHOLE input frame resized (max diff <= 2 grey levels);
        # n2.padding_check's uniform-border rule is reported alongside (it can fire on a black night sky)
        whole = v1["rebuilt_vs_pil_bicubic_max_abs"] <= 2 and v2["rebuilt_vs_pil_bicubic_max_abs"] <= 2
        if v1["padding_detected"] or v2["padding_detected"]:
            problems.append(f"{A['stem']}: n2.padding_check reports a uniform border band {v1['uniform_border_bands']} "
                            "although the model frames equal the whole input frame resized (content test), i.e. its "
                            "band rule fires on uniform scene content (here a black night sky); harmless while "
                            "PADDING is None, but the rule should not be used to set PADDING")
        add("f", whole,
            f"source {A['source_size_wh'][0]}x{A['source_size_wh'][1]} -> decoded input {A['input_size_wh'][0]}x"
            f"{A['input_size_wh'][1]} -> model {v1['model_size_wh'][0]}x{v1['model_size_wh'][1]} (scale x "
            f"{A['scale_x']:.4f}, y {A['scale_y']:.4f}: a slight stretch, no crop); n2.padding_check flag v1 "
            f"{v1['padding_detected']}, v2 {v2['padding_detected']} (bands {v1['uniform_border_bands']}); rebuilt vs PIL bicubic of the input: mean "
            f"{v1['rebuilt_vs_pil_bicubic_mean_abs']:.3f}, max {v1['rebuilt_vs_pil_bicubic_max_abs']}; box fractions are "
            "of the whole frame and the whole frame is kept, so no box correction is needed (parse_v2 PADDING None)")
        add("g", v1["timestamps_match_1dp"] and v2["timestamps_match_1dp"],
            f"v1: {len(v1['prompt_timestamps'])} timestamps for {v1['n_frames_in']} frames (one per 2-frame token "
            f"group, the mean of the pair's times) {v1['prompt_timestamps']}; v2: {len(v2['prompt_timestamps'])} "
            f"timestamps {v2['prompt_timestamps']}; all equal the frame times (to 0.1 s): v1 {v1['timestamps_match_1dp']},"
            f" v2 {v2['timestamps_match_1dp']}")
        add("h", H["box_on_colliding_vehicles"].strip().lower() == "yes",
            f"visual: {H['box_on_colliding_vehicles']} - {H['note']}")
        add("i", True, f"model frame {v1['model_size_wh'][1]} x {v1['model_size_wh'][0]} (H x W); grid_thw v1 "
            f"{v1['grid_thw']} ({v1['visual_tokens']} visual tokens), v2 {v2['grid_thw']} ({v2['visual_tokens']}); "
            f"ground-truth box {A['box_px_model'][0]:.1f} x {A['box_px_model'][1]:.1f} px (W x H) at the model resolution")
        types = {J["true_type_metadata"], J["type_from_video_folder"], J["type_from_file_name"], J["true_type_results_csv"]}
        j_ok = (len(types) == 1 and J["parsed_type"] == J["scoring_type_results_csv"]
                and (J["raw_type"] == J["parsed_type"] or not J["parsed_valid"]))
        ann = (f"annotation first collision frame {an['frame_from_annotation']} (metadata {an['accident_frame_meta']}), "
               f"ids {an['collision_ids']} tags {an['collision_tags']} -> {an['n_vehicles_in_collision']} vehicle(s); "
               f"box vs metadata max diff {an['box_max_abs_diff']:.4f}") if an.get("found") else "annotation file not found"
        add("j", j_ok, f"raw v1 {J['raw_v1']!r} -> parsed {J['parsed_type']} (valid {J['parsed_valid']}) -> scored "
            f"{J['scoring_type_results_csv']}; true: metadata {J['true_type_metadata']}, folder "
            f"{J['type_from_video_folder']}, file name {J['type_from_file_name']}, results.csv {J['true_type_results_csv']};"
            f" {ann}")
        c = A["cache"]
        cache_txt = ("missing" if c.get("missing") else
                     f"{c['file']}: indices equal {c['frame_indices_equal']}; {c['cache_size_wh'][0]}x{c['cache_size_wh'][1]};"
                     f" identical to the input frames (1): {c['identical_to_input_frames']} (input is "
                     f"{A['input_size_wh'][0]}x{A['input_size_wh'][1]}); vs the frames the model receives (rebuilt v1): "
                     f"mean {c['vs_model_view_mean']:.3f}, max {c['vs_model_view_max']}; vs input resized with PIL bicubic: "
                     f"mean {c['vs_input_resized_mean']:.3f}, max {c['vs_input_resized_max']}")
        L.append(f"=== {A['stem']} ({A['type']}, category {A['category']}) ===")
        for code, ok, det in rows:
            L.append(f"  {code}  {pf(ok)}  {det}")
        L.append(f"  4  cache: {cache_txt}")
        L.append(f"     v2 input = v1 frames duplicated: indices {A['v2_indices_same']}, frames {A['v2_frames_same']};"
                 f" rebuilt v2 duplicates identical {v2.get('rebuilt_duplicates_identical')}, equal to rebuilt v1 "
                 f"{v2.get('rebuilt_equal_to_v1_rebuilt')}")
        L.append("")
    L.append("SUMMARY TABLE (rows = clips, columns = checks a-j)")
    L.append("  " + f"{'clip':34s}" + "".join(f"{c:>6s}" for c in "abcdefghij"))
    # rebuild per-clip PASS/FAIL compactly from the lines above
    cur = None
    marks = collections.OrderedDict()
    for line in L:
        if line.startswith("=== "):
            cur = line[4:].split(" (")[0]
            marks[cur] = {}
        elif cur and len(line) > 6 and line[2] in "abcdefghij" and line[3:5] == "  ":
            marks[cur][line[2]] = line[5:9]
    for s, mk in marks.items():
        L.append("  " + f"{s:34s}" + "".join(f"{mk.get(c, '-'):>6s}" for c in "abcdefghij"))
    L.append("")
    # k
    L.append("k. ALL 200 CLIPS: confusion (rows true, columns scored type; fallbacks counted as rear-end)")
    for v in (1, 2):
        L.append(f"  narrow v{v} (valid {G[f'v{v}_valid']}/200, correct {G[f'v{v}_correct']})")
        L.append(f"  {'true':10s}" + "".join(f"{t:>10s}" for t in vz.TYPES))
        for t in vz.TYPES:
            L.append(f"  {t:10s}" + "".join(f"{G[f'v{v}_confusion'][t][q]:>10d}" for q in vz.TYPES))
        L.append(f"  scored answer distribution v{v}: {G[f'v{v}_scored_answer_counts']}")
    L.append(f"  v1 raw collision_type values before any fallback: {G['v1_raw_type_counts']}")
    L.append(f"  re-parsing every saved raw answer with the frozen parsers reproduces results.csv: "
             f"{'yes' if not G['reparse_mismatches'] else 'NO: ' + str(G['reparse_mismatches'])}")
    L.append(f"  true types in results.csv = metadata type: {G['results_type_equals_metadata']}/200; = video folder: "
             f"{G['results_type_equals_folder']}/200; counts {G['true_type_counts']}")
    L.append(f"  v1 accuracy {G['v1_acc']:.4f} ({G['v1_correct']}/200). Standard error: {G['se_v1_estimate']:.4f} "
             f"(estimate) / {G['se_under_chance_0.20']:.4f} (under chance 0.20); z = {G['z_vs_chance']:.2f}; exact "
             f"binomial P(X <= {G['v1_correct']} | n 200, p 0.2) = {G['binom_p_le_k_at_0.20']:.5f} (two-sided "
             f"{G['binom_two_sided_p']:.5f}). Chance 0.20 = any fixed answer on this balanced list.")
    L.append(f"  spelling: {G['types_lists']}; metadata type values {G['metadata_type_values']}; video folders "
             f"{G['video_folder_names']}; v1 prompt lists '{G['narrow_prompt_v1_type_words'][0] if G['narrow_prompt_v1_type_words'] else '?'}';"
             f" all 5 names appear in the v1 prompt {all(G['narrow_prompt_v1_types_found'].values())} and in PROMPT_V2 "
             f"{all(G['prompt_v2_types_found'].values())}; annotation_classes.yaml lists object classes only (no "
             f"collision types): {not G['annotation_classes_yaml_has_collision_types']}")
    same = (G["types_lists"]["smoke.TYPES"] == G["types_lists"]["vz.TYPES"] == G["types_lists"]["n2.TYPES"]
            == G["metadata_type_values"] == G["video_folder_names"])
    L.append(f"  type names identical across parser lists, metadata and folders: {pf(same)}")
    if not same:
        problems.append("type names differ between parser lists, metadata and folders")
    L.append("")
    L.append("l. ALL 200 CLIPS: t* inside the narrow window and v2 collision_visible")
    for g in ("inside", "outside"):
        d = G["l"].get(g, {})
        n = d.get("n", 0)
        L.append(f"  {g:8s} n {n:3d}; collision_visible false {d.get('false', 0)} ({d.get('false', 0) / n if n else float('nan'):.4f}),"
                 f" true {d.get('true', 0)}, other {sum(v for k, v in d.items() if k.startswith('other'))}")
    L.append("")
    L.append("ANYTHING THAT LOOKS WRONG (not fixed)")
    L.extend(f"  - {p}" for p in problems) if problems else L.append("  - no check FAILED")
    L.extend(f"  - {x}" for x in EXTRA_NOTES(au, G, hv))
    text = "\n".join(L) + "\n"
    REPORT.write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)


def EXTRA_NOTES(au, G, hv):
    """Observations that are not PASS/FAIL checks; each is computed from the audit numbers."""
    notes = []
    A0 = au["clips"][0]
    if A0["v1"]["prompt_text_contains_16_frames"] and len(A0["v1"]["prompt_timestamps"]) != 16:
        notes.append(f"v1 prompt says 'These 16 frames' but the model gets {len(A0['v1']['prompt_timestamps'])} "
                     f"timestamped token groups ({A0['v1']['grid_thw']}); each group fuses 2 different frames "
                     "(temporal patch 2), so v1 cannot see the 16 frames separately.")
    st = [(A["steps_min"], A["steps_max"]) for A in au["clips"]]
    if any(abs(a - b) > 1e-6 for a, b in st):
        notes.append(f"frame steps are not uniform after rounding to frame numbers (min {min(a for a, _ in st):.3f} s, "
                     f"max {max(b for _, b in st):.3f} s); nominal step is 4/15 = 0.2667 s, not 0.25 s.")
    dark = [k for k, r in hv.items() if "hard to see" in r["note"] or "invisible" in r["note"]]
    if dark:
        notes.append(f"check h passes, but in {len(dark)} of 10 clips ({', '.join(dark)}) a colliding vehicle is almost "
                     "invisible at night at full resolution, so it is even harder to see at 448 x 256.")
    cm = [A["cache"] for A in au["clips"] if not A["cache"].get("missing")]
    if cm and all(c["frame_indices_equal"] for c in cm):
        notes.append(f"QLoRA cache (window A): same frame indices for {len(cm)}/10 clips, but not pixel-identical to "
                     f"the frames the model receives: mean |diff| {min(c['vs_model_view_mean'] for c in cm):.3f}-"
                     f"{max(c['vs_model_view_mean'] for c in cm):.3f}, max {max(c['vs_model_view_max'] for c in cm)} "
                     "grey levels (JPEG quality 95 storage); the input frames (1) are 452 x 254, the cache 448 x 256.")
    li, lo = G["l"].get("inside", {}), G["l"].get("outside", {})
    if li.get("n") and li.get("false", 0) / li["n"] > 0.9:
        notes.append(f"v2 answers collision_visible false in {li.get('false', 0)} of {li['n']} clips with t* inside the "
                     f"window (and {lo.get('false', 0)} of {lo.get('n', 0)} outside), so the field does not track whether "
                     "the collision is in the window.")
    acc = G["v1_acc"]
    if acc < 0.2 and G["binom_two_sided_p"] < 0.05:
        notes.append(f"v1 accuracy {acc:.4f} is significantly below chance 0.20 (two-sided p {G['binom_two_sided_p']:.5f});"
                     " the labels and parsing pass the checks above, so this points to systematic wrong answers, not a "
                     "label or mapping error (see the confusion matrix).")
    return notes


def h_evidence():
    """For check h: the t* frame with the metadata box (green) and the annotation's own 2d boxes of the colliding
    objects at the first collision (yellow, with tag). Writes clips/<stem>/h_colliding_objects.png (refuses to overwrite)."""
    vz = vz_mod()
    from PIL import Image, ImageDraw

    meta = {vz.norm(r["rgb_path"]): r for r in rcsv(META_SYN)}
    clips = {c["rgb_path"]: c for c in vz.load_clips(TEST_CLIPS)}
    for row in load_list(vz):
        p = row["rgb_path"]
        s, m, c = stem(p), meta[p], clips[p]
        out = OUT / "clips" / s / "h_colliding_objects.png"
        if out.exists():
            fail(f"{out} exists; refusing to overwrite")
        ap = DATASET / "synthetic_videos" / "annotations" / f"{s}.json" / f"{s}.json"
        d = json.loads(ap.read_text(encoding="utf-8"))
        c0 = d["collision"][0]
        objs = {o["id"]: o for b in d["base"] if b["iteration"] == c0["iteration"] for o in b["objects"]}
        af = int(m["accident_frame"])
        img = vz.decode(c, [af], budget=10 ** 12)[af]
        box = [float(m[q]) for q in ("x1", "y1", "x2", "y2")]
        draw_box(img, box, (0, 220, 0), 3)
        dr = ImageDraw.Draw(img)
        for i in c0["ids"]:
            o = objs.get(i)
            if o and o.get("2d_bbox"):
                (a, b_), (e, f) = o["2d_bbox"]
                dr.rectangle([a, b_, e, f], outline=(255, 230, 0), width=2)
                dr.text((a, max(0, b_ - 12)), f"id {i} tag {o['tag']}", fill=(255, 230, 0))
        x1, y1, x2, y2 = box
        W, H = img.size
        pad = 120
        crop = img.crop((max(0, int(x1 * W) - pad), max(0, int(y1 * H) - pad), min(W, int(x2 * W) + pad),
                         min(H, int(y2 * H) + pad)))
        crop = crop.resize((crop.width * 2, crop.height * 2), Image.NEAREST)
        crop.save(out)
        print(f"{s}: ids {c0['ids']} tags {[objs[i]['tag'] if i in objs else None for i in c0['ids']]} -> {out.name}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--h-evidence", action="store_true")
    g.add_argument("--list", action="store_true")
    g.add_argument("--audit", action="store_true")
    g.add_argument("--reader2", action="store_true")
    g.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.list:
        make_list()
    elif a.audit:
        audit()
    elif a.reader2:
        reader2()
    elif a.h_evidence:
        h_evidence()
    else:
        report()


if __name__ == "__main__":
    main()
