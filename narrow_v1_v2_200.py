"""Narrow call v1 (frozen) vs v2 (narrow_prompt_v2.py) on 200 new synthetic clips (supervisor feedback, point 1).

SYNTHETIC ONLY: metadata-synthetic.csv, synthetic_split.csv, the saved out-of-fold classifier t_hat (C_cctv seed 0,
frozen post-processing) and synthetic videos. No real clip or real label is opened.

--make-list (CPU): writes results/narrow_v1_v2_200/test_clips.csv once (refuses to overwrite) and LIST.txt with its
  sha256: 200 synthetic clips, none of the 32 design clips (vlm_dev_clips.csv), 40 per collision type; per type 20 day
  + 20 night (night = weather "night"), or, when a type has fewer than 20 night clips outside the design clips, all of
  its night clips and the rest day. Draw: numpy default_rng(TEST_SEED), each stratum's pool sorted by rgb_path, strata
  in TYPES order, day then night.
--run (GPU): writes PROTOCOL.txt before the first model call (refuses to start if the list's sha256 differs), then for
  every clip both narrow versions on the same window centre t_hat:
    v1 = vlm_zero_shot.narrow_input(clip, 1) + Vlm.__call__ + narrow_answer (frozen: 16 frames over t_hat +- 2 s,
         448 x 256 area, native pairing, NARROW_PROMPT, greedy, max_new_tokens 64)
    v2 = narrow_prompt_v2.narrow_v2_input + generate_v2 + parse_v2 (16 frames over t_hat +- 2 s, 448 x 256 area, each
         frame duplicated, PROMPT_V2, greedy, max_new_tokens 160)
  Appends one row per clip to results.csv and every raw answer to raw_outputs.jsonl (flushed), so the run resumes
  where it stopped; mains power checked at the start, every POWER_CHECK_EVERY clips and at the end (on battery: stop,
  exit 4, resumable).
--report (CPU): report.txt (see the report itself for the definitions).
Usage (from cctv-accident-ITS-system/):
    .venv-vlm\\Scripts\\python.exe narrow_v1_v2_200.py --make-list
    .venv-vlm\\Scripts\\python.exe -u narrow_v1_v2_200.py --run
    .venv-vlm\\Scripts\\python.exe narrow_v1_v2_200.py --report
"""

import argparse
import collections
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
import vlm_zero_shot as vz  # noqa: E402  (frozen protocol code: imported, not changed)
import narrow_prompt_v2 as n2  # noqa: E402

smoke = vz.smoke
OUT = REPO_ROOT / "results" / "narrow_v1_v2_200"
LIST_CSV, LIST_TXT = OUT / "test_clips.csv", OUT / "LIST.txt"
PROTOCOL, RESULTS, RAW, EVENTS, REPORT = (OUT / n for n in ("PROTOCOL.txt", "results.csv", "raw_outputs.jsonl",
                                                             "run_events.txt", "report.txt"))
TEST_SEED, PER_TYPE, NIGHT_PER_TYPE = 1, 40, 20
POWER_CHECK_EVERY = 25
SCORER_CONFIG = REPO_ROOT / "scorer_config.json"
TYPES = vz.TYPES
MAJORITY_TYPE = vz.FALLBACK_TYPE  # most frequent synthetic type overall (794 of 2211), as the narrow-v1 fallback
COLUMNS = ["rgb_path", "type", "day_night", "weather", "val_fold", "t_star", "t_hat", "center_x", "center_y",
           "v1_valid", "v1_reason", "v1_type", "v1_x", "v1_y", "v1_gpu_s", "v1_peak_alloc_gb", "v1_visual_tokens",
           "v2_valid", "v2_reason", "v2_type", "v2_x", "v2_y", "v2_bbox", "v2_swapped_x", "v2_swapped_y",
           "v2_bbox_digit_string", "v2_type_normalised", "v2_missing_keys", "v2_collision_visible", "v2_motion",
           "v2_vehicles", "v2_new_tokens", "v2_hit_max_tokens", "v2_gpu_s", "v2_peak_alloc_gb", "v2_visual_tokens",
           "wall_s", "gpu", "finished_utc"]


def day_night(m):
    return "night" if m["weather"] == "night" else "day"


# ---------------------------------------------------------------- list

def make_list():
    if LIST_CSV.exists():
        vz.fail(f"{LIST_CSV} exists; the test list is drawn once")
    meta = vz.synthetic_meta()
    split = {vz.norm(r["rgb_path"]): r for r in vz.read_csv(vz.SPLIT_CSV)}
    design = {vz.norm(r["rgb_path"]) for r in vz.read_csv(vz.DEV_CSV)}
    if len(design) != 32:
        vz.fail("expected 32 design clips")
    rng = np.random.default_rng(TEST_SEED)
    rows, notes = [], []
    for typ in TYPES:
        pools = {dn: sorted(p for p, m in meta.items() if m["type"] == typ and day_night(m) == dn and p not in design)
                 for dn in ("day", "night")}
        n_night = min(NIGHT_PER_TYPE, len(pools["night"]))
        n_day = PER_TYPE - n_night
        if n_day > len(pools["day"]):
            vz.fail(f"{typ}: not enough day clips")
        notes.append(f"{typ}: pool outside the design clips day {len(pools['day'])}, night {len(pools['night'])} -> "
                     f"drawn day {n_day}, night {n_night}")
        for dn, n in (("day", n_day), ("night", n_night)):
            for p in rng.choice(pools[dn], size=n, replace=False):
                rows.append((p, typ, dn, meta[p]["weather"], split[p]["val_fold"]))
    if len({r[0] for r in rows}) != 200 or any(r[0] in design for r in rows):
        vz.fail("list check failed")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(LIST_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rgb_path", "type", "day_night", "weather", "val_fold"])
        w.writerows(rows)
    sha = vz.sha256(LIST_CSV)
    text = "\n".join([f"Narrow v1 vs v2 test list, written {vz.now()} before any model call",
                      f"{LIST_CSV.name} sha256 {sha}; {len(rows)} synthetic clips; seed {TEST_SEED} "
                      f"(numpy default_rng); none of the 32 design clips ({vz.DEV_CSV.name} sha256 {vz.sha256(vz.DEV_CSV)})",
                      "rule: 40 per collision type; 20 day + 20 night, or all night clips and the rest day when a type "
                      "has fewer than 20 night clips; pools sorted by rgb_path; strata in TYPES order, day then night",
                      *notes,
                      "fold counts: " + ", ".join(f"{k} {v}" for k, v in sorted(collections.Counter(r[4] for r in rows).items()))]) + "\n"
    LIST_TXT.write_text(text, encoding="utf-8")
    print(text)


def load_list():
    if not LIST_TXT.exists() or vz.sha256(LIST_CSV) not in LIST_TXT.read_text(encoding="utf-8"):
        vz.fail(f"{LIST_CSV.name} sha256 does not match {LIST_TXT.name}")
    clips = vz.load_clips(LIST_CSV)
    if len(clips) != 200:
        vz.fail("expected 200 clips")
    return clips


# ---------------------------------------------------------------- run

def write_protocol():
    if PROTOCOL.exists():
        return
    L = [f"Narrow v1 vs v2 on 200 synthetic clips; protocol written {vz.now()} BEFORE the first model call.",
         f"clips: {LIST_CSV.name} sha256 {vz.sha256(LIST_CSV)} (see LIST.txt); synthetic only, no real clip or label",
         f"window centre (both versions): t_hat = {vz.CLASSIFIER_PRED.relative_to(REPO_ROOT)} (C_cctv seed 0, frozen "
         "post-processing, out-of-fold: checked per clip)",
         f"model: {smoke.MODELS[vz.MODEL_KEY]} {vz.PRECISION} (NF4, double quant, bf16 compute), loaded once",
         f"narrow v1 (frozen; vlm_zero_shot.py sha256 {vz.sha256(REPO_ROOT / 'vlm_zero_shot.py')}, narrow prompt v1 "
         f"sha256 {vz.text_sha(vz.NARROW_PROMPT)}): {vz.NARROW_FRAMES} frames over t_hat +- {vz.NARROW_HALF_S:g} s, "
         f"{vz.PIXEL_BUDGET} px area, native pairing, greedy, max_new_tokens {vz.MAX_NEW_TOKENS}; valid = "
         "vlm_smoke_test.parse_output values_ok; invalid -> fallback",
         f"narrow v2 (narrow_prompt_v2.py sha256 {vz.sha256(REPO_ROOT / 'narrow_prompt_v2.py')}, PROMPT_V2 sha256 "
         f"{vz.text_sha(n2.PROMPT_V2)}): {n2.N_FRAMES_V2} frames over t_hat +- {n2.HALF_S_V2:g} s, {n2.BUDGET_V2} px "
         f"area, each frame duplicated (own timestamp), greedy, max_new_tokens {n2.MAX_NEW_TOKENS_V2}; valid / flags / "
         "point = parse_v2 (digit strings and type normalisation accepted and flagged); invalid -> fallback",
         f"fallback (both): type {vz.FALLBACK_TYPE}, point {vz.FALLBACK_POINT}; counted",
         "collision_visible (v2) is stored as a diagnostic only; it changes no prediction",
         f"metrics: type accuracy (fallbacks included as their fallback type); location score S = mean exp(-(dx^2 / "
         f"2 sigma_x^2 + dy^2 / 2 sigma_y^2)) with sigma_x, sigma_y and spatial_k from {SCORER_CONFIG.name} "
         "(accs_scorer formula); point error = Euclidean distance in image fractions; baselines on the same clips: "
         f"majority type ({MAJORITY_TYPE}) and the image centre (0.5, 0.5)",
         "mains power checked at start, every 25 clips and at the end; nothing is chosen from this run here "
         "(report only)"]
    OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def v1_reason(text):
    p = smoke.parse_output(text)
    return "" if p["values_ok"] else ("keys or values not valid" if p["json_ok"] else "no JSON object")


def run():
    clips = load_list()
    write_protocol()
    done = {r["rgb_path"] for r in vz.read_csv(RESULTS)} if RESULTS.exists() else set()
    todo = [c for c in clips if c["rgb_path"] not in done]

    def event(msg):
        line = f"{vz.now()} {msg}"
        print(line, flush=True)
        with open(EVENTS, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    on, ptxt = vz.power()
    event(f"START/RESUME: {len(done)} clips done, {len(todo)} to do; power {ptxt}")
    if not on:
        event("STOPPED: on battery at start")
        sys.exit(4)
    if not todo:
        event("nothing to do")
        return
    vlm = vz.Vlm()
    new_file = not RESULTS.exists()
    with open(RESULTS, "a", newline="", encoding="utf-8") as fr, open(RAW, "a", encoding="utf-8") as fj:
        w = csv.DictWriter(fr, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        for i, c in enumerate(todo, 1):
            t0 = time.perf_counter()
            idx1, fr1, p1 = vz.narrow_input(c, 1)
            t1, s1 = vlm(c, fr1, idx1, p1)
            pk1, vt1 = getattr(vlm, "last_peak_alloc_gb", float("nan")), vlm.last_visual_tokens
            ok1, typ1, x1, y1 = vz.narrow_answer(t1)
            _, fr2, idx2, p2 = n2.narrow_v2_input(c)
            t2, s2, nt2 = n2.generate_v2(vlm, c, fr2, idx2, p2)
            pk2, vt2 = getattr(vlm, "last_peak_alloc_gb", float("nan")), vlm.last_visual_tokens
            r2 = n2.parse_v2(t2)
            row = {"rgb_path": c["rgb_path"], "type": c["type"], "day_night": "night" if c["weather"] == "night" else "day",
                   "weather": c["weather"], "val_fold": c["val_fold"], "t_star": c["t_star"], "t_hat": c["t_hat"],
                   "center_x": c["center_x"], "center_y": c["center_y"],
                   "v1_valid": ok1, "v1_reason": v1_reason(t1), "v1_type": typ1, "v1_x": x1, "v1_y": y1, "v1_gpu_s": s1,
                   "v1_peak_alloc_gb": pk1, "v1_visual_tokens": vt1,
                   "v2_valid": r2["valid"], "v2_reason": r2["reason"], "v2_type": r2["collision_type"], "v2_x": r2["x"],
                   "v2_y": r2["y"], "v2_bbox": json.dumps(r2["bbox_2d"]), "v2_swapped_x": r2["swapped_x"],
                   "v2_swapped_y": r2["swapped_y"], "v2_bbox_digit_string": r2["bbox_digit_string"],
                   "v2_type_normalised": r2["type_normalised"], "v2_missing_keys": "|".join(r2["missing_keys"]),
                   "v2_collision_visible": json.dumps(r2["collision_visible"]), "v2_motion": json.dumps(r2["motion"]),
                   "v2_vehicles": json.dumps(r2["vehicles"]), "v2_new_tokens": nt2,
                   "v2_hit_max_tokens": nt2 >= n2.MAX_NEW_TOKENS_V2, "v2_gpu_s": s2, "v2_peak_alloc_gb": pk2,
                   "v2_visual_tokens": vt2, "wall_s": time.perf_counter() - t0, "gpu": vlm.gpu, "finished_utc": vz.now()}
            for v, text in ((1, t1), (2, t2)):
                fj.write(json.dumps({"clip": c["rgb_path"], "narrow_version": v, "output": text}) + "\n")
            fj.flush()
            os.fsync(fj.fileno())
            w.writerow(row)
            fr.flush()
            os.fsync(fr.fileno())
            vz.stamp(f"{len(done) + i}/{len(clips)} {Path(c['rgb_path']).stem}: v1 {typ1} ({'ok' if ok1 else 'fallback'}, "
                     f"{s1:.1f} s) v2 {r2['collision_type']} ({'ok' if r2['valid'] else 'fallback'}, {s2:.1f} s, "
                     f"{nt2} tokens); true {c['type']}")
            if i % POWER_CHECK_EVERY == 0 and i < len(todo):
                on, ptxt = vz.power()
                event(f"power check after {len(done) + i} clips: {ptxt}")
                if not on:
                    event(f"STOPPED (resumable): on battery after {len(done) + i} clips")
                    sys.exit(4)
    on, ptxt = vz.power()
    event(f"FINISHED: {len(clips)} clips; power at end {ptxt}")


# ---------------------------------------------------------------- report

def truthy(v):
    return str(v) == "True"


def spatial_terms(x, y, xs, ys, cfg):
    sx, sy = cfg["sigma_x"] * cfg["spatial_k"], cfg["sigma_y"] * cfg["spatial_k"]
    x, y, xs, ys = (np.asarray(a, float) for a in (x, y, xs, ys))
    return np.exp(-((x - xs) ** 2 / (2 * sx ** 2) + (y - ys) ** 2 / (2 * sy ** 2))), np.hypot(x - xs, y - ys)


def block(rows, v, cfg):
    """Summary of version v on rows: n, valid, accuracy, S, mean / median point error."""
    if not rows:
        return {"n": 0}
    acc = np.mean([r[f"v{v}_type"] == r["type"] for r in rows])
    s, e = spatial_terms([float(r[f"v{v}_x"]) for r in rows], [float(r[f"v{v}_y"]) for r in rows],
                         [float(r["center_x"]) for r in rows], [float(r["center_y"]) for r in rows], cfg)
    valid = [r for r in rows if truthy(r[f"v{v}_valid"])]
    acc_valid = np.mean([r[f"v{v}_type"] == r["type"] for r in valid]) if valid else float("nan")
    return {"n": len(rows), "valid": len(valid), "fallback": len(rows) - len(valid),
            "correct": int(round(acc * len(rows))), "acc": acc, "acc_valid": acc_valid, "S": float(s.mean()),
            "err_mean": float(e.mean()), "err_median": float(np.median(e))}


def confusion(rows, v):
    head = "true \\ predicted"
    L = [f"  {head:16s} " + " ".join(f"{t:>9s}" for t in TYPES) + f" {'correct':>8s} {'of which fallbacks':>18s}"]
    for t in TYPES:
        rr = [r for r in rows if r["type"] == t]
        cnt = collections.Counter(r[f"v{v}_type"] for r in rr)
        fb = sum(not truthy(r[f"v{v}_valid"]) for r in rr)
        L.append(f"  {t:16s} " + " ".join(f"{cnt[p]:9d}" for p in TYPES) + f" {cnt[t]:5d}/{len(rr):<2d} {fb:18d}")
    return L


def report():
    if REPORT.exists():
        vz.fail(f"{REPORT} exists; refusing to overwrite")
    rows = vz.read_csv(RESULTS)
    if len(rows) != 200 or len({r["rgb_path"] for r in rows}) != 200:
        vz.fail(f"expected 200 distinct rows in {RESULTS.name}, found {len(rows)}")
    if {r["rgb_path"] for r in rows} != {r["rgb_path"] for r in load_list()}:
        vz.fail("results do not match the test list")
    cfg = json.loads(SCORER_CONFIG.read_text(encoding="utf-8"))
    ev = EVENTS.read_text(encoding="utf-8").strip().splitlines()
    L = [f"Narrow v1 (frozen) vs narrow v2 on 200 synthetic clips; report {vz.now()}. Synthetic labels only; no real "
         f"clip or real label opened. Protocol: PROTOCOL.txt; list: LIST.txt.",
         f"power: {ev[0]}", f"       {ev[-1]}",
         f"S: sigma_x {cfg['sigma_x']:.6f}, sigma_y {cfg['sigma_y']:.6f}, spatial_k {cfg['spatial_k']} ({SCORER_CONFIG.name}); "
         "point error = Euclidean distance in image fractions (width and height as 1). Type accuracy counts fallbacks "
         "as their fallback type (rear-end); 'accuracy on valid answers' leaves fallbacks out (secondary).", ""]
    b = {v: block(rows, v, cfg) for v in (1, 2)}
    maj = np.mean([r["type"] == MAJORITY_TYPE for r in rows])
    sc, ec = spatial_terms([0.5] * len(rows), [0.5] * len(rows), [float(r["center_x"]) for r in rows],
                           [float(r["center_y"]) for r in rows], cfg)
    L.append("1. SIDE BY SIDE (all 200 clips)")
    L.append(f"  {'':36s} {'narrow v1':>12s} {'narrow v2':>12s}")
    for key, name, fmt in (("valid", "valid answers", "{:d}"), ("fallback", "fallbacks", "{:d}"),
                           ("correct", "type correct", "{:d}"), ("acc", "type accuracy", "{:.4f}"),
                           ("acc_valid", "type accuracy on valid answers", "{:.4f}"), ("S", "location score S", "{:.4f}"),
                           ("err_mean", "mean point error", "{:.4f}"), ("err_median", "median point error", "{:.4f}")):
        L.append(f"  {name:36s} {fmt.format(b[1][key]):>12s} {fmt.format(b[2][key]):>12s}")
    flags = {"swapped x (x1 > x2)": "v2_swapped_x", "swapped y (y1 > y2)": "v2_swapped_y",
             "bbox digit strings": "v2_bbox_digit_string", "collision_type trimmed / lower-cased": "v2_type_normalised",
             f"answer hit max_new_tokens ({n2.MAX_NEW_TOKENS_V2})": "v2_hit_max_tokens"}
    any_flag = sum(any(truthy(r[c]) for c in flags.values()) for r in rows)
    L.append(f"  {'flagged answers (any flag)':36s} {'n/a':>12s} {any_flag:>12d}   (v1 parser has no flags)")
    for name, col in flags.items():
        L.append(f"    {name:34s} {'-':>12s} {sum(truthy(r[col]) for r in rows):>12d}")
    miss = collections.Counter(k for r in rows for k in r["v2_missing_keys"].split("|") if k)
    L.append(f"    {'missing keys (not invalid)':34s} {'-':>12s} {sum(bool(r['v2_missing_keys']) for r in rows):>12d}   {dict(miss)}")
    for v in (1, 2):
        reasons = collections.Counter(r[f"v{v}_reason"] for r in rows if not truthy(r[f"v{v}_valid"]))
        L.append(f"  v{v} invalid reasons: {dict(reasons) or 'none'}")
    for v in (1, 2):
        g = [float(r[f"v{v}_gpu_s"]) for r in rows]
        pk = [float(r[f"v{v}_peak_alloc_gb"]) for r in rows]
        vt = sorted({r[f"v{v}_visual_tokens"] for r in rows})
        L.append(f"  v{v}: GPU s per clip (synchronised generate) mean {np.mean(g):.2f}, median {np.median(g):.2f}, "
                 f"max {max(g):.2f}; peak allocated {max(pk):.2f} GB; visual tokens {vt}")
    nt = [int(r["v2_new_tokens"]) for r in rows]
    L.append(f"  v2 answer length: mean {np.mean(nt):.1f}, max {max(nt)} new tokens")
    L.append("")
    L.append("2. BASELINES (same 200 clips)")
    L.append(f"  majority type ({MAJORITY_TYPE}, the most frequent synthetic type): type accuracy {maj:.4f} "
             f"(the list is balanced, 40 per type, so any single type scores 0.2000)")
    L.append(f"  image centre (0.5, 0.5): S {sc.mean():.4f}, mean point error {ec.mean():.4f}, median {np.median(ec):.4f}")
    L.append("")
    for v in (1, 2):
        L.append(f"3.{v} CONFUSION MATRIX narrow v{v} (rows true type, columns predicted type; fallbacks counted as "
                 f"predicted {vz.FALLBACK_TYPE})")
        L.extend(confusion(rows, v))
        L.append("")
    both = collections.Counter((r["v1_type"] == r["type"], r["v2_type"] == r["type"]) for r in rows)
    L.append(f"  paired (secondary): both correct {both[(True, True)]}, only v1 {both[(True, False)]}, only v2 "
             f"{both[(False, True)]}, neither {both[(False, False)]}")
    L.append("")
    L.append("4. NARROW v2 collision_visible (diagnostic only; changes no prediction)")
    vis = collections.defaultdict(list)
    for r in rows:
        vis[r["v2_collision_visible"]].append(r)
    nf = len(vis.get("false", []))
    L.append(f"  collision_visible = false: {nf}/200 = {nf / 200:.4f}")
    for key in sorted(vis):
        bb = block(vis[key], 2, cfg)
        L.append(f"  {key:10s} n {bb['n']:3d}  type accuracy {bb['acc']:.4f} ({bb['correct']}/{bb['n']}), valid "
                 f"{bb['valid']}, S {bb['S']:.4f}")
    L.append("")
    L.append("5. NARROW v2 'motion' ANSWERS PER TRUE TYPE (trimmed, lower-cased; all answers, valid or not)")
    mot = collections.Counter()
    by = collections.defaultdict(collections.Counter)
    for r in rows:
        m = json.loads(r["v2_motion"])
        m = m.strip().lower() if isinstance(m, str) else repr(m)
        mot[m] += 1
        by[r["type"]][m] += 1
    names = [m for m, _ in mot.most_common()]
    w = max(12, *(len(m) for m in names))
    L.append(f"  {'true type':10s} " + " ".join(f"{m:>{w}s}" for m in names))
    for t in TYPES:
        L.append(f"  {t:10s} " + " ".join(f"{by[t][m]:>{w}d}" for m in names))
    L.append("")
    L.append("6. DAY / NIGHT")
    L.append(f"  {'':6s} {'version':8s} {'n':>4s} {'valid':>6s} {'accuracy':>9s} {'S':>7s} {'mean err':>9s}")
    for dn in ("day", "night"):
        rr = [r for r in rows if r["day_night"] == dn]
        for v in (1, 2):
            bb = block(rr, v, cfg)
            L.append(f"  {dn:6s} {'v' + str(v):8s} {bb['n']:4d} {bb['valid']:6d} {bb['acc']:9.4f} {bb['S']:7.4f} "
                     f"{bb['err_mean']:9.4f}")
        s0, e0 = spatial_terms([0.5] * len(rr), [0.5] * len(rr), [float(r["center_x"]) for r in rr],
                               [float(r["center_y"]) for r in rr], cfg)
        L.append(f"  {dn:6s} {'centre':8s} {len(rr):4d} {'':6s} {'':9s} {s0.mean():7.4f} {e0.mean():9.4f}")
    L.append("")
    L.append("7. PER TYPE (type accuracy and S)")
    L.append(f"  {'type':10s} {'n':>3s} {'v1 acc':>7s} {'v2 acc':>7s} {'v1 S':>7s} {'v2 S':>7s} {'centre S':>9s}")
    for t in TYPES:
        rr = [r for r in rows if r["type"] == t]
        b1, b2 = block(rr, 1, cfg), block(rr, 2, cfg)
        s0, _ = spatial_terms([0.5] * len(rr), [0.5] * len(rr), [float(r["center_x"]) for r in rr],
                              [float(r["center_y"]) for r in rr], cfg)
        L.append(f"  {t:10s} {len(rr):3d} {b1['acc']:7.4f} {b2['acc']:7.4f} {b1['S']:7.4f} {b2['S']:7.4f} {s0.mean():9.4f}")
    text = "\n".join(L) + "\n"
    REPORT.write_text(text, encoding="utf-8")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--make-list", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--report", action="store_true")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if args.make_list:
        make_list()
    elif args.run:
        run()
    else:
        report()


if __name__ == "__main__":
    main()
