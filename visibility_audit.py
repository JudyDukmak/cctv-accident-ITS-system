"""Low-visibility count of the colliding actors on all 2,211 synthetic clips (read only; no model, no torch, no GPU).

Run with the .venv-ml interpreter. SYNTHETIC ONLY: reads metadata-synthetic.csv, synthetic_split.csv, the synthetic
annotation JSON files and videos, results/qlora_data/manifest.csv, results/qlora_train/val_epoch1.csv and
val_epoch1_metrics.json, scorer_config.json. Never opens a real clip, real label or real metadata file. Changes no
label, manifest, split, result or frozen file, and removes or flags nothing in the training data: the flags below
live only in results/visibility_audit/. Writes nothing outside results/visibility_audit/ and never overwrites.

Signal (from the simulator code, see results/box_audit/report.txt): an object is in base[tick].objects only if
enough of its pixels are visible in the segmentation image, and its 2d_bbox covers only those pixels. A colliding
actor missing from base at a tick = "low visibility" at that tick (hidden, cut by the image edge, or very small /
far; the signal cannot tell which).

Tick rule (as in the box audit): tick = accident_frame + annotations_start_offset; the script stops if
collision[0].iteration != that tick for any clip. Other ticks: accident_frame + k + offset, k in -10, -5, +5, +10.
A tick that is not in the file (clip start/end) counts as "tick absent", never as "missing".

Flags, fixed before counting:
  LOW_AT_IMPACT     at least one colliding actor is missing at the collision tick
  LOW_AROUND_IMPACT at least one colliding actor (the same actor) is missing at the collision tick and at both the
                    -5 and +5 ticks (an absent tick does not count as missing)

Modes (in order):
  --count    per-clip table clips.csv (a-d + flags) and the count tables (overall, scenario, type, weather, fold)
  --pick     visual-check list: up to 8 LOW_AT_IMPACT clips and 8 unflagged clips, numpy default_rng(0)
             (flagged first, then unflagged, each rng.choice(sorted pool, k, replace=False)); writes check_list.csv
  --draw     the accident frame of each picked clip at full resolution: metadata box green, visible colliding
             actors' 2d_bbox blue; for an actor missing at the collision tick, its box at the nearest tick where it is
             present is drawn dashed orange and labelled (aid for the visual judgement only). Writes check/<stem>.jpg
             and check/<stem>_crop.jpg
  --explore  exploratory (decided before looking): fold-1 window A, saved epoch-1 answers; type accuracy and S for
             flagged vs unflagged windows, n per group. Recomputed overall accuracy and S must equal
             val_epoch1_metrics.json.
  --report   report.txt from the above and visual.csv (visual judgement, written by hand after looking)
"""
import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
OUT = REPO / "results" / "visibility_audit"
DATASET = REPO / "ACCIDENT" / "dataset"
META_SYN = DATASET / "metadata-synthetic.csv"
SPLIT = REPO / "synthetic_split.csv"
ANN_DIR = DATASET / "synthetic_videos" / "annotations"
MANIFEST = REPO / "results" / "qlora_data" / "manifest.csv"
VAL_E1 = REPO / "results" / "qlora_train" / "val_epoch1.csv"
VAL_E1_MET = REPO / "results" / "qlora_train" / "val_epoch1_metrics.json"
SCORER = REPO / "scorer_config.json"
CLIPS_CSV, COUNTS_CSV, COUNTS_JSON = OUT / "clips.csv", OUT / "counts.csv", OUT / "counts.json"
PICK_CSV, PICK_TXT, VISUAL_CSV = OUT / "check_list.csv", OUT / "CHECK_LIST.txt", OUT / "visual.csv"
EXPLORE_CSV, EXPLORE_JSON, REPORT = OUT / "explore_fold1_A.csv", OUT / "explore_fold1_A.json", OUT / "report.txt"
OFFS = (-10, -5, 0, 5, 10)
SEED, N_PICK = 0, 8
TAGS = {12: "pedestrian", 13: "rider", 14: "car", 15: "truck", 16: "bus", 17: "train", 18: "motorcycle",
        19: "bicycle", 20: "static", 21: "dynamic", 22: "other", 29: "van"}
FLAGS = ("LOW_AT_IMPACT", "LOW_AROUND_IMPACT")


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def rcsv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def wcsv(path, rows, cols):
    if path.exists():
        fail(f"{path} exists; refusing to overwrite")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def wtext(path, text):
    if path.exists():
        fail(f"{path} exists; refusing to overwrite")
    path.write_text(text, encoding="utf-8")


def norm(p):
    return p.replace("\\", "/")


def stem(p):
    return Path(norm(p)).stem


def load_meta():
    meta = {norm(r["rgb_path"]): r for r in rcsv(META_SYN)}
    split = {norm(r["rgb_path"]): r for r in rcsv(SPLIT)}
    if len(meta) != 2211 or set(meta) != set(split) or not all(p.startswith("synthetic_videos/") for p in meta):
        fail("expected the same 2211 synthetic clips in metadata-synthetic.csv and synthetic_split.csv")
    return meta, split


# ---------------------------------------------------------------- 1-3. count

def one_clip(args):
    """Per-clip record (runs in a worker). Returns (record, error or None)."""
    p, m, sp = args
    s = stem(p)
    ap = ANN_DIR / f"{s}.json" / f"{s}.json"
    if not ap.is_file():
        return None, f"{p}: annotation file not found ({ap})"
    try:
        d = json.loads(ap.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return None, f"{p}: unreadable annotation ({e})"
    if not d.get("collision"):
        return None, f"{p}: no collision record"
    c0 = d["collision"][0]
    W, H = int(m["width"]), int(m["height"])
    af, off = int(m["accident_frame"]), int(m["annotations_start_offset"])
    it = int(c0["iteration"])
    if it != af + off:
        return None, f"TICK RULE FAILS: {p}: collision[0].iteration {it} != accident_frame {af} + offset {off}"
    base = {int(b["iteration"]): {o["id"]: o for o in b["objects"]} for b in d["base"]}
    ids = list(c0["ids"])
    x1, y1, x2, y2 = (float(m[k]) for k in ("x1", "y1", "x2", "y2"))
    bw, bh = (x2 - x1) * W, (y2 - y1) * H
    barea = bw * bh
    rec = {"rgb_path": p, "stem": s, "scenario_id": sp["scenario_id"], "map": m["map"], "type": m["type"],
           "weather": m["weather"], "val_fold": sp["val_fold"], "accident_frame": af, "offset": off,
           "collision_tick": it, "n_actors": len(ids), "actor_ids": json.dumps(ids),
           "box_w_px": round(bw, 2), "box_h_px": round(bh, 2), "box_area_px": round(barea, 1)}
    missing = {}
    for k in OFFS:
        t = it + k
        lab = "0" if k == 0 else f"{k:+d}"
        if t not in base:
            rec[f"tick_absent_{lab}"] = True
            rec[f"n_missing_{lab}"] = ""
            missing[k] = None
        else:
            rec[f"tick_absent_{lab}"] = False
            miss = [i for i in ids if i not in base[t]]
            rec[f"n_missing_{lab}"] = len(miss)
            missing[k] = set(miss)
    rec["missing_ids_0"] = json.dumps(sorted(missing[0]))
    tags = {}
    for i in ids:
        for t in sorted(base, key=lambda t: abs(t - it)):
            if i in base[t]:
                tags[i] = TAGS.get(base[t][i]["tag"], base[t][i]["tag"])
                break
    rec["actor_tags"] = json.dumps([tags.get(i, "never in base") for i in ids])
    shares = {}
    for i in ids:
        o = base[it].get(i)
        if o is not None and o.get("2d_bbox"):
            (a, b), (c, e) = o["2d_bbox"]
            shares[i] = round(max(0, c - a) * max(0, e - b) / barea, 4) if barea > 0 else None
    rec["visible_area_share"] = json.dumps({str(k): v for k, v in shares.items()})
    vals = [v for v in shares.values() if v is not None]
    rec["min_visible_area_share"] = min(vals) if vals else ""
    rec["LOW_AT_IMPACT"] = bool(missing[0])
    around = set(missing[0])
    for k in (-5, 5):
        around &= missing[k] if missing[k] is not None else set()
    rec["LOW_AROUND_IMPACT"] = bool(around)
    return rec, None


COLS = ["rgb_path", "stem", "scenario_id", "map", "type", "weather", "val_fold", "accident_frame", "offset",
        "collision_tick", "n_actors", "actor_ids", "actor_tags", "n_missing_0", "missing_ids_0", "n_missing_-10",
        "n_missing_-5", "n_missing_+5", "n_missing_+10", "tick_absent_-10", "tick_absent_-5", "tick_absent_0",
        "tick_absent_+5", "tick_absent_+10", "visible_area_share", "min_visible_area_share", "box_w_px", "box_h_px",
        "box_area_px", "LOW_AT_IMPACT", "LOW_AROUND_IMPACT"]


def tables(recs):
    out = {}
    for by in ("all", "scenario_id", "type", "weather", "val_fold"):
        groups = {}
        for r in recs:
            groups.setdefault("all" if by == "all" else str(r[by]), []).append(r)
        out[by] = [{"by": by, "group": g, "n": len(rs),
                    **{f: sum(bool(r[f]) for r in rs) for f in FLAGS},
                    **{f"{f}_share": sum(bool(r[f]) for r in rs) / len(rs) for f in FLAGS}}
                   for g, rs in sorted(groups.items())]
    return out


def count():
    OUT.mkdir(parents=True, exist_ok=True)
    for p in (CLIPS_CSV, COUNTS_CSV, COUNTS_JSON):
        if p.exists():
            fail(f"{p} exists; refusing to overwrite")
    meta, split = load_meta()
    jobs = [(p, meta[p], split[p]) for p in sorted(meta)]
    recs, errs = [], []
    with ProcessPoolExecutor(max_workers=12) as ex:
        for k, (rec, err) in enumerate(ex.map(one_clip, jobs, chunksize=8)):
            if err:
                errs.append(err)
            else:
                recs.append(rec)
            if (k + 1) % 250 == 0:
                print(f"  {k + 1}/{len(jobs)} clips read")
    if errs:
        for e in errs[:20]:
            print(e)
        fail(f"{len(errs)} clips failed (tick rule or unreadable annotation); nothing written")
    wcsv(CLIPS_CSV, recs, COLS)
    T = tables(recs)
    rows = [r for v in T.values() for r in v]
    wcsv(COUNTS_CSV, rows, list(rows[0]))
    n = len(recs)
    absent = {k: sum(r[f"tick_absent_{k}"] for r in recs) for k in ("-10", "-5", "0", "+5", "+10")}
    miss = {k: sum((r[f"n_missing_{k}"] or 0) for r in recs if r[f"n_missing_{k}"] != "") for k in absent}
    clips_miss = {k: sum(bool(r[f"n_missing_{k}"]) for r in recs if r[f"n_missing_{k}"] != "") for k in absent}
    na = [r["n_actors"] for r in recs]
    shares = [v for r in recs for v in json.loads(r["visible_area_share"]).values() if v is not None]
    summ = {"n_clips": n, "tick_rule_holds": n, "n_actors_distribution": {str(k): na.count(k) for k in sorted(set(na))},
            "ticks_absent": absent, "actors_missing_by_tick": miss, "clips_with_missing_by_tick": clips_miss,
            "visible_area_share_quantiles": {q: float(np.quantile(shares, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "box_area_px_quantiles": {q: float(np.quantile([r["box_area_px"] for r in recs], q))
                                      for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "box_area_px_median_by_flag": {f: {str(v): float(np.median([r["box_area_px"] for r in recs if r[f] == v]))
                                               for v in (True, False) if any(r[f] == v for r in recs)} for f in FLAGS},
            "tables": T}
    wtext(COUNTS_JSON, json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "tables"}, indent=1))
    for by, rs in T.items():
        print(f"by {by}:")
        for r in rs:
            print(f"  {r['group']:20s} n {r['n']:5d}  LOW_AT_IMPACT {r['LOW_AT_IMPACT']:4d} ({r['LOW_AT_IMPACT_share']:.3f})"
                  f"  LOW_AROUND_IMPACT {r['LOW_AROUND_IMPACT']:4d} ({r['LOW_AROUND_IMPACT_share']:.3f})")


def load_clips():
    if not CLIPS_CSV.is_file():
        fail("run --count first")
    recs = rcsv(CLIPS_CSV)
    for r in recs:
        for f in FLAGS:
            r[f] = r[f] == "True"
    return recs


# ---------------------------------------------------------------- 4. visual check

def pick():
    if PICK_CSV.exists() or PICK_TXT.exists():
        fail("check list exists; it is drawn once")
    recs = load_clips()
    flagged = sorted(r["rgb_path"] for r in recs if r["LOW_AT_IMPACT"])
    unflagged = sorted(r["rgb_path"] for r in recs if not r["LOW_AT_IMPACT"])
    rng = np.random.default_rng(SEED)
    by = {r["rgb_path"]: r for r in recs}
    rows = []
    for grp, pool in (("LOW_AT_IMPACT", flagged), ("unflagged", unflagged)):
        for p in rng.choice(pool, size=min(N_PICK, len(pool)), replace=False):
            r = by[p]
            rows.append({"group": grp, "pool_size": len(pool), "rgb_path": p, "stem": r["stem"],
                         "scenario_id": r["scenario_id"], "weather": r["weather"], "val_fold": r["val_fold"],
                         "LOW_AROUND_IMPACT": r["LOW_AROUND_IMPACT"], "missing_ids_0": r["missing_ids_0"],
                         "actor_ids": r["actor_ids"], "actor_tags": r["actor_tags"]})
    wcsv(PICK_CSV, rows, list(rows[0]))
    lines = [f"Visual-check list: up to {N_PICK} LOW_AT_IMPACT clips then up to {N_PICK} unflagged clips, numpy "
             f"default_rng({SEED}), one generator, rng.choice(sorted pool, k, replace=False). Written before drawing.", ""]
    lines += [f"{r['group']:14s} (pool {r['pool_size']:4d})  {r['stem']:32s} {r['weather']:7s} fold {r['val_fold']}  "
              f"missing at impact {r['missing_ids_0']}  LOW_AROUND_IMPACT {r['LOW_AROUND_IMPACT']}" for r in rows]
    wtext(PICK_TXT, "\n".join(lines) + "\n")
    print("\n".join(lines))


def dashed_rect(dr, box, color, width=3, dash=14):
    x1, y1, x2, y2 = box
    for (a, b), (c, e) in (((x1, y1), (x2, y1)), ((x1, y2), (x2, y2)), ((x1, y1), (x1, y2)), ((x2, y1), (x2, y2))):
        L = max(abs(c - a), abs(e - b))
        n = max(1, int(L // dash))
        for j in range(0, n, 2):
            t0, t1 = j / n, min(1, (j + 1) / n)
            dr.line([a + (c - a) * t0, b + (e - b) * t0, a + (c - a) * t1, b + (e - b) * t1], fill=color, width=width)


def draw():
    import vlm_zero_shot as vz
    from PIL import Image, ImageDraw, ImageFont

    def font(sz):
        try:
            return ImageFont.truetype("arial.ttf", sz)
        except OSError:
            return ImageFont.load_default(sz)

    if not PICK_CSV.is_file():
        fail("run --pick first")
    if (OUT / "check").exists():
        fail(f"{OUT / 'check'} exists; refusing to overwrite")
    (OUT / "check").mkdir()
    meta, _ = load_meta()
    for row in rcsv(PICK_CSV):
        p, s = row["rgb_path"], row["stem"]
        m = meta[p]
        W, H = int(m["width"]), int(m["height"])
        af, off = int(m["accident_frame"]), int(m["annotations_start_offset"])
        d = json.loads((ANN_DIR / f"{s}.json" / f"{s}.json").read_text(encoding="utf-8"))
        c0 = d["collision"][0]
        it = int(c0["iteration"])
        if it != af + off:
            fail(f"{p}: tick rule fails")
        base = {int(b["iteration"]): {o["id"]: o for o in b["objects"]} for b in d["base"]}
        img = vz.decode({"video": str(DATASET / p), "rgb_path": p}, [af], budget=10 ** 12)[af]
        if img.size != (W, H):
            fail(f"{p}: decoded size {img.size} != metadata {W}x{H}")
        dr = ImageDraw.Draw(img)
        fs = font(26)
        boxes = []
        for i in c0["ids"]:
            o = base[it].get(i)
            if o and o.get("2d_bbox"):
                (a, b), (c, e) = o["2d_bbox"]
                dr.rectangle([a, b, c, e], outline=(40, 150, 255), width=3)
                lab, col = f"id {i} {TAGS.get(o['tag'], o['tag'])}", (40, 150, 255)
                boxes.append([a, b, c, e])
            else:
                near = next(((t - it, base[t][i]) for t in sorted(base, key=lambda t: abs(t - it)) if i in base[t]), None)
                if near is None:
                    lab, col, a, e = f"id {i}: never in base", (255, 150, 0), 10, 40
                else:
                    (a, b), (c, e) = near[1]["2d_bbox"]
                    dashed_rect(dr, [a, b, c, e], (255, 150, 0))
                    boxes.append([a, b, c, e])
                    lab, col = (f"id {i} {TAGS.get(near[1]['tag'], near[1]['tag'])}: MISSING at impact; dashed = its "
                                f"box at tick {near[0]:+d} ({near[0] / (int(m['no_frames']) / float(m['duration'])):+.2f} s)"), (255, 150, 0)
            tb = dr.textbbox((0, 0), lab, font=fs)
            ty = e + 4 if e + 4 + tb[3] < H else max(0, b - tb[3] - 6)
            dr.rectangle([a, ty, a + tb[2] + 8, ty + tb[3] + 6], fill=(0, 0, 0))
            dr.text((a + 4, ty + 2), lab, fill=col, font=fs)
        mb = [float(m["x1"]) * W, float(m["y1"]) * H, float(m["x2"]) * W, float(m["y2"]) * H]
        dr.rectangle(mb, outline=(0, 230, 0), width=4)
        boxes.append([int(v) for v in mb])
        dr.rectangle([0, 0, W, 44], fill=(0, 0, 0))
        dr.text((10, 6), f"{s}  [{row['group']}]  accident_frame {af} (t = {float(m['accident_time']):.2f} s)  "
                         f"green = metadata box, blue = visible colliding actors at the collision tick",
                fill=(255, 255, 255), font=font(30))
        img.save(OUT / "check" / f"{s}.jpg", quality=92)
        x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w = min(W, max(480, (x2 - x1) * 2.2, (y2 - y1) * 2.2 * 16 / 9)); h = w * 9 / 16
        l = min(max(cx - w / 2, 0), W - w); t = min(max(cy - h / 2, 0), H - h)
        img.crop((int(l), int(t), int(l + w), int(t + h))).resize((1280, 720)).save(
            OUT / "check" / f"{s}_crop.jpg", quality=92)
        print(f"{s}: drawn ({row['group']}, missing at impact {row['missing_ids_0']})")


# ---------------------------------------------------------------- 5. exploratory

def explore():
    import pandas as pd

    for p in (EXPLORE_CSV, EXPLORE_JSON):
        if p.exists():
            fail(f"{p} exists; refusing to overwrite")
    cfg = json.loads(SCORER.read_text())
    sx, sy = cfg["sigma_x"] * cfg["spatial_k"], cfg["sigma_y"] * cfg["spatial_k"]
    man = pd.read_csv(MANIFEST, dtype={"fold": str})
    man = man[(man["fold"] == "1") & (man["window"] == "A")]
    d = pd.read_csv(VAL_E1)
    if len(d) != len(man) or set(d["rgb_path"]) != set(man["rgb_path"]):
        fail("val_epoch1.csv windows differ from manifest fold 1 window A")
    d["correct"] = d["pred_type"] == d["type"]
    d["s"] = np.exp(-((d["x"] - d["center_x"]) ** 2 / (2 * sx ** 2) + (d["y"] - d["center_y"]) ** 2 / (2 * sy ** 2)))
    saved = json.loads(VAL_E1_MET.read_text())
    for k, v in (("type_accuracy", d["correct"].mean()), ("S", d["s"].mean())):
        if abs(v - saved[k]) > 1e-9:
            fail(f"recomputed {k} {v} != saved {saved[k]}")
    cl = pd.DataFrame(load_clips())[["rgb_path", *FLAGS]]
    d = d.merge(cl, on="rgb_path", how="left", validate="one_to_one")
    if d[list(FLAGS)].isna().any().any():
        fail("some fold-1 windows have no visibility record")
    d = d.merge(man[["rgb_path", "t_star_inside"]].rename(columns={"t_star_inside": "t_star_inside_man"}),
                on="rgb_path", validate="one_to_one")
    rows = []
    for f in FLAGS:
        for v in (True, False):
            g = d[d[f] == v]
            rows.append({"flag": f, "group": "flagged" if v else "unflagged", "n": len(g),
                         "type_accuracy": g["correct"].mean() if len(g) else None,
                         "S": g["s"].mean() if len(g) else None,
                         "share_t_star_inside_window": (g["t_star_inside_man"].astype(str) == "True").mean() if len(g) else None,
                         "majority_type_rear_end_share": (g["type"] == saved["majority_type"]).mean() if len(g) else None})
    out = pd.DataFrame(rows)
    out.to_csv(EXPLORE_CSV, index=False)
    res = {"n_windows": len(d), "overall_type_accuracy": float(d["correct"].mean()), "overall_S": float(d["s"].mean()),
           "matches_val_epoch1_metrics": True, "groups": rows,
           "flagged_types": {f: d[d[f]]["type"].value_counts().to_dict() for f in FLAGS}}
    wtext(EXPLORE_JSON, json.dumps(res, indent=1, default=float))
    print(out.to_string(index=False))
    print(json.dumps(res["flagged_types"]))


# ---------------------------------------------------------------- 6. report

def report():
    if REPORT.exists():
        fail(f"{REPORT} exists; refusing to overwrite")
    for p in (COUNTS_JSON, PICK_CSV, VISUAL_CSV, EXPLORE_JSON):
        if not p.is_file():
            fail(f"{p} missing")
    S = json.loads(COUNTS_JSON.read_text(encoding="utf-8"))
    E = json.loads(EXPLORE_JSON.read_text(encoding="utf-8"))
    vis = {r["stem"]: r for r in rcsv(VISUAL_CSV)}
    picks = rcsv(PICK_CSV)
    if {r["stem"] for r in picks} != set(vis):
        fail("visual.csv must have one row per picked clip")
    n = S["n_clips"]
    L = ["LOW-VISIBILITY COUNT, ALL SYNTHETIC CLIPS (read only; synthetic only; no real clip, label or metadata file "
         "opened; no label, manifest, split, result or frozen file changed; nothing removed or flagged in the training "
         "data; no model, no GPU)", "",
         "Signal: a colliding actor (collision[0].ids) missing from base[tick].objects = LOW VISIBILITY at that tick. The "
         "simulator drops an object when too few of its pixels are visible in the segmentation image; causes can be "
         "an occluder, the image edge, or the object being very small or far. It is called low visibility, not "
         "occlusion.",
         "Note (expectation, not verified here): CARLA's segmentation camera labels classes, not lit pixels, so "
         "darkness alone should not drop an object; a vehicle can be in base and still be hard to see at night "
         "(seen in the box audit).",
         f"Tick rule: tick = accident_frame + annotations_start_offset == collision[0].iteration held for {n}/{n} "
         "clips (the script stops on any failure). Ticks -10/-5/+5/+10 = accident_frame -10/-5/+5/+10 frames "
         "(-0.5/-0.25/+0.25/+0.5 s at 20 fps). A tick not in the file counts as absent, never as missing.",
         "Flags (fixed before counting): LOW_AT_IMPACT = >= 1 colliding actor missing at the collision tick; "
         "LOW_AROUND_IMPACT = >= 1 colliding actor (the same one) missing at the collision tick and at both the -5 and "
         "+5 ticks.", "",
         f"1. PER-CLIP RECORD (clips.csv, {n} rows)",
         f"  colliding actors per clip: {S['n_actors_distribution']}",
         f"  ticks absent from the file: {S['ticks_absent']}",
         f"  clips with >= 1 colliding actor missing, by tick: {S['clips_with_missing_by_tick']}",
         f"  colliding actors missing, by tick: {S['actors_missing_by_tick']}",
         "  c. visible-box area / metadata-box area, actors present at the collision tick, quantiles 5/25/50/75/95%: "
         + ", ".join(f"{v:.3f}" for v in S["visible_area_share_quantiles"].values()),
         "  d. metadata box area in pixels, quantiles 5/25/50/75/95%: "
         + ", ".join(f"{v:.0f}" for v in S["box_area_px_quantiles"].values()),
         "  median metadata box area by flag (px): " + "; ".join(
             f"{f}: flagged {v.get('True', float('nan')):.0f}, unflagged {v.get('False', float('nan')):.0f}"
             for f, v in S["box_area_px_median_by_flag"].items()), "",
         "2-3. COUNTS (n, flagged count, share)"]
    for by, rs in S["tables"].items():
        L.append(f"  by {by}:")
        L.append(f"    {'group':20s} {'n':>5s}  {'LOW_AT_IMPACT':>20s}  {'LOW_AROUND_IMPACT':>20s}")
        for r in rs:
            L.append(f"    {r['group']:20s} {r['n']:5d}  {r['LOW_AT_IMPACT']:9d} ({r['LOW_AT_IMPACT_share']:.3f})   "
                     f"{r['LOW_AROUND_IMPACT']:9d} ({r['LOW_AROUND_IMPACT_share']:.3f})")
    L += ["", "4. VISUAL CHECK OF THE SIGNAL (CHECK_LIST.txt written before drawing; images in check/; every entry "
          "below is my VISUAL JUDGEMENT, not a computed check)",
          "  In the images: green = metadata box, blue = colliding actors present at the collision tick; an actor "
          "missing at impact is drawn dashed orange at its nearest tick where it is present (aid only)."]
    for r in picks:
        v = vis[r["stem"]]
        L.append(f"  [{r['group']:13s}] {r['stem']:32s} judgement: {v['judgement']:14s} {v['note']}")
    jud = [vis[r["stem"]]["judgement"] for r in picks if r["group"] == "LOW_AT_IMPACT"]
    L.append(f"  flagged clips by judgement: " + ", ".join(f"{k} {jud.count(k)}" for k in sorted(set(jud))))
    jud = [vis[r["stem"]]["judgement"] for r in picks if r["group"] != "LOW_AT_IMPACT"]
    L.append(f"  unflagged clips by judgement: " + ", ".join(f"{k} {jud.count(k)}" for k in sorted(set(jud))))
    L += ["", "5. EXPLORATORY (decided before looking): fold-1 window A, saved epoch-1 answers (no model run)",
          f"  windows {E['n_windows']}; recomputed overall type accuracy {E['overall_type_accuracy']:.4f}, S "
          f"{E['overall_S']:.4f} (equal to val_epoch1_metrics.json)",
          f"  {'flag':18s} {'group':10s} {'n':>4s} {'type acc':>9s} {'S':>7s} {'t* inside':>10s} {'rear-end share':>15s}"]
    for g in E["groups"]:
        fm = lambda x: f"{x:.4f}" if x is not None else "  n/a"  # noqa: E731
        L.append(f"  {g['flag']:18s} {g['group']:10s} {g['n']:4d} {fm(g['type_accuracy']):>9s} {fm(g['S']):>7s} "
                 f"{fm(g['share_t_star_inside_window']):>10s} {fm(g['majority_type_rear_end_share']):>15s}")
    L.append(f"  types of the flagged windows: {E['flagged_types']}")
    L += ["  Caution: exploratory, one fold, one checkpoint, no test of significance; groups differ in type mix and in "
          "the share of windows that contain t*, both of which affect accuracy and S on their own.", "",
          "Nothing was fixed or excluded. Files: clips.csv (per-clip record, all 2211 clips), counts.csv, counts.json, "
          "CHECK_LIST.txt, check_list.csv, check/*.jpg, visual.csv, explore_fold1_A.csv/json, run_log_*.txt, report.txt."]
    wtext(REPORT, "\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for a in ("--count", "--pick", "--draw", "--explore", "--report"):
        g.add_argument(a, action="store_true")
    a = ap.parse_args()
    {"count": count, "pick": pick, "draw": draw, "explore": explore, "report": report}[
        next(k for k in ("count", "pick", "draw", "explore", "report") if getattr(a, k))]()


if __name__ == "__main__":
    main()
