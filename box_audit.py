"""Ground-truth box audit on synthetic clips (no model, no torch, no GPU; run with the .venv-ml interpreter).

SYNTHETIC ONLY: reads metadata-synthetic.csv, synthetic_split.csv (scenario_id), the synthetic annotation JSON files
and videos, results/qlora_data/manifest.csv, the cached fold-1 window-A frames and
results/qlora_train/figures/examples.csv. Never opens a real clip, real label or real metadata file. Changes no
label, manifest or frozen file (vlm_zero_shot.py and build_qlora_data.py are imported, not changed).
Everything is written to results/box_audit/ and nothing there is ever overwritten.

Modes (run in this order):
  --list    42 clips: 3 per scenario (14 scenario_ids of synthetic_split.csv, sorted), numpy default_rng(0), one
            generator in scenario order, rng.choice(sorted pool, 3, replace=False). Writes list.csv and LIST.txt.
  --audit   per clip: decodes accident_frame - 10, accident_frame, accident_frame + 10 at full resolution (no
            resize), draws the metadata box (green), the colliding actors' 2D boxes of the matching annotation tick
            (blue, with ids) and the frame number and time; checks a-e. Writes clips/<stem>.jpg (three frames side
            by side), grids/<scenario>.jpg (per scenario: the accident frame + crops around the boxes), checks.csv,
            actors.csv, checks.json.
  --fig6    figure 6 redrawn: the same 8 windows (examples.csv); for each, the cached frame the figure used next to
            the frame at t* (accident_frame, built with the cache's own steps), same boxes. Writes fig6_tstar.png
            and fig6_tstar.csv.
  --report  report.txt from checks.json, actors.csv, fig6_tstar.csv and visual.csv (the visual judgement, written
            by hand after looking at the images; every entry there is a visual judgement, not a computed check).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
OUT = REPO / "results" / "box_audit"
DATASET = REPO / "ACCIDENT" / "dataset"
META_SYN = DATASET / "metadata-synthetic.csv"
SPLIT = REPO / "synthetic_split.csv"
ANN_DIR = DATASET / "synthetic_videos" / "annotations"
MANIFEST = REPO / "results" / "qlora_data" / "manifest.csv"
EXAMPLES = REPO / "results" / "qlora_train" / "figures" / "examples.csv"
VAL_EPOCH1 = REPO / "results" / "qlora_train" / "val_epoch1.csv"
SEED, PER_SCENARIO, N_SCENARIOS, OFFSETS = 0, 3, 14, (-10, 0, 10)
LIST_CSV, LIST_TXT = OUT / "list.csv", OUT / "LIST.txt"
CHECKS_CSV, ACTORS_CSV, CHECKS_JSON = OUT / "checks.csv", OUT / "actors.csv", OUT / "checks.json"
FIG6_PNG, FIG6_CSV, VISUAL_CSV, REPORT = OUT / "fig6_tstar.png", OUT / "fig6_tstar.csv", OUT / "visual.csv", OUT / "report.txt"
TAGS = {12: "pedestrian", 13: "rider", 14: "car", 15: "truck", 16: "bus", 17: "train", 18: "motorcycle",
        19: "bicycle", 20: "static", 21: "dynamic", 22: "other", 29: "van"}
GREEN, BLUE, RED = (0, 230, 0), (40, 150, 255), (255, 40, 40)
VIS_WORDS = ("visib", "occlu", "trunc", "hidden")


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


def synthetic_meta():
    rows = {norm(r["rgb_path"]): r for r in rcsv(META_SYN)}
    if len(rows) != 2211 or not all(p.startswith("synthetic_videos/") for p in rows):
        fail(f"{META_SYN.name}: expected 2211 synthetic clips")
    return rows


def font(size):
    from PIL import ImageFont
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default(size)


# ---------------------------------------------------------------- 1. list

def make_list():
    OUT.mkdir(parents=True, exist_ok=True)
    if LIST_CSV.exists() or LIST_TXT.exists():
        fail("list exists; the clips are drawn once")
    meta, split = synthetic_meta(), {norm(r["rgb_path"]): r for r in rcsv(SPLIT)}
    if set(split) != set(meta):
        fail(f"{SPLIT.name} clips differ from {META_SYN.name}")
    scen = sorted({r["scenario_id"] for r in split.values()})
    if len(scen) != N_SCENARIOS:
        fail(f"expected {N_SCENARIOS} scenarios, found {len(scen)}")
    rng = np.random.default_rng(SEED)
    rows = []
    for s in scen:
        pool = sorted(p for p, r in split.items() if r["scenario_id"] == s)
        for p in rng.choice(pool, size=PER_SCENARIO, replace=False):
            m = meta[p]
            if f"{m['map']}_{m['type']}" != s:
                fail(f"{p}: scenario {s} != metadata map_type")
            rows.append({"rgb_path": p, "stem": stem(p), "scenario_id": s, "pool_size": len(pool), "type": m["type"],
                         "weather": m["weather"], "val_fold": split[p]["val_fold"],
                         "accident_frame": m["accident_frame"], "accident_time": m["accident_time"]})
    wcsv(LIST_CSV, rows, list(rows[0]))
    lines = [f"Box audit clip list: {len(rows)} synthetic clips, {PER_SCENARIO} per scenario ({len(scen)} scenario_ids "
             f"of {SPLIT.name}, sorted), numpy default_rng({SEED}), one generator in scenario order, "
             f"rng.choice(sorted pool, {PER_SCENARIO}, replace=False). Written before any image was made.", ""]
    for r in rows:
        lines.append(f"{r['scenario_id']:18s} (pool {r['pool_size']:3d})  {r['stem']:32s} weather {r['weather']:7s} "
                     f"fold {r['val_fold']}  accident_frame {r['accident_frame']}")
    wtext(LIST_TXT, "\n".join(lines) + "\n")
    print("\n".join(lines))


def load_list():
    if not LIST_CSV.is_file():
        fail("run --list first")
    return rcsv(LIST_CSV)


# ---------------------------------------------------------------- 2-3. audit

def rect_inter(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0, w) * max(0, h)


def area(a):
    return max(0, a[2] - a[0]) * max(0, a[3] - a[1])


def union_iou(meta_px, boxes, W, H):
    """IoU of the metadata box with the pixel union of the actor boxes (integer pixel rectangles [x1,x2) x [y1,y2))."""
    if not boxes:
        return None
    m = np.zeros((H, W), bool)
    for b in boxes:
        m[max(0, b[1]):min(H, b[3]), max(0, b[0]):min(W, b[2])] = True
    g = np.zeros((H, W), bool)
    g[meta_px[1]:meta_px[3], meta_px[0]:meta_px[2]] = True
    u = (m | g).sum()
    return float((m & g).sum() / u) if u else None


def find_vis_keys(obj, prefix=""):
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(w in k.lower() for w in VIS_WORDS):
                out.add(prefix + k)
            out |= find_vis_keys(v, prefix + k + ".")
    elif isinstance(obj, list):
        for v in obj[:50]:
            out |= find_vis_keys(v, prefix)
    return out


def annotation(s):
    ap = ANN_DIR / f"{s}.json" / f"{s}.json"
    if not ap.is_file():
        return None, ap
    return json.loads(ap.read_text(encoding="utf-8")), ap


def check_clip(row, m):
    s = row["stem"]
    W, H = int(m["width"]), int(m["height"])
    nf, dur = int(m["no_frames"]), float(m["duration"])
    fps, af, off = nf / dur, int(m["accident_frame"]), int(m["annotations_start_offset"])
    d, ap = annotation(s)
    rec = {"stem": s, "scenario_id": row["scenario_id"], "weather": row["weather"], "val_fold": row["val_fold"],
           "W": W, "H": H, "fps": fps, "accident_frame": af, "accident_time": float(m["accident_time"]),
           "annotations_start_offset": off, "annotation_file": str(ap.relative_to(REPO))}
    if d is None:
        rec.update({"annotation_found": False})
        return rec, [], None
    rec["annotation_found"] = True
    col = d["collision"]
    c0 = col[0]
    it = int(c0["iteration"])
    base = {int(b["iteration"]): {o["id"]: o for o in b["objects"]} for b in d["base"]}
    base_its = [int(b["iteration"]) for b in d["base"]]
    # a. metadata box == first collision record's box
    (bx1, by1), (bx2, by2) = c0["collision_bbox"]
    meta_n = [float(m[k]) for k in ("x1", "y1", "x2", "y2")]
    meta_px_f = [meta_n[0] * W, meta_n[1] * H, meta_n[2] * W, meta_n[3] * H]
    ann_px = [bx1, by1, bx2, by2]
    diff_px = max(abs(a - b) for a, b in zip(meta_px_f, ann_px))
    rec.update({"n_collision_records": len(col), "first_record_is_earliest": it == min(int(c["iteration"]) for c in col),
                "collision_ids": c0["ids"], "ann_box_px": ann_px, "meta_box_px": [round(v, 3) for v in meta_px_f],
                "a_box_max_diff_px": diff_px, "a_pass": diff_px <= 0.5})
    # b. metadata box inside the image
    inside = 0 <= meta_n[0] < meta_n[2] <= 1 and 0 <= meta_n[1] < meta_n[3] <= 1
    edges = [n for n, hit in (("left", bx1 <= 0), ("top", by1 <= 0), ("right", bx2 >= W - 1),
                              ("bottom", by2 >= H - 1)) if hit]
    rec.update({"b_pass": inside, "b_touches_edge": ",".join(edges)})
    # c. colliding actors' boxes vs the metadata box, at the collision tick
    meta_px = [int(round(v)) for v in meta_px_f]
    actors, boxes = [], []
    for i in c0["ids"]:
        o = base.get(it, {}).get(i)
        bb = o.get("2d_bbox") if o else None
        a = {"stem": s, "id": i, "tag": TAGS.get(o["tag"], o["tag"]) if o else "",
             "in_base_at_collision_tick": o is not None, "has_2d_bbox": bool(bb)}
        for k, fr in zip(("prev", "acc", "next"), OFFSETS):
            a[f"in_base_at_frame_{k}"] = i in base.get(it + fr, {})
        if bb:
            b = [int(bb[0][0]), int(bb[0][1]), int(bb[1][0]), int(bb[1][1])]
            a.update({"box_px": b, "box_area_px": area(b),
                      "share_inside_meta": rect_inter(b, meta_px) / area(b) if area(b) else None})
            boxes.append(b)
        actors.append(a)
    present = [a for a in actors if a["has_2d_bbox"]]
    ub = ([min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
          if boxes else None)
    rec.update({"n_actors": len(actors), "n_actors_with_box": len(present),
                "c_min_share_inside": min((a["share_inside_meta"] for a in present), default=None),
                "c_iou_union": union_iou(meta_px, boxes, W, H),
                "c_iou_enclosing_box": (rect_inter(meta_px, ub) / (area(meta_px) + area(ub) - rect_inter(meta_px, ub)))
                if ub else None, "actor_enclosing_box_px": ub,
                "c_pass": len(present) == len(actors) and all((a["share_inside_meta"] or 0) > 0 for a in present)})
    # d. frame offset: sim_tick = accident_frame + annotations_start_offset
    rec.update({"first_collision_tick": it, "d_frame_from_annotation": it - off,
                "d_pass": it - off == af,
                "base_first_tick": base_its[0], "base_ticks_contiguous": base_its == list(range(base_its[0], base_its[0] + len(base_its))),
                "base_n_ticks": len(base_its), "no_frames": nf,
                "base_first_tick_eq_offset": base_its[0] == off,
                "accident_time_x_fps": float(m["accident_time"]) * fps})
    # e. visibility / occlusion fields anywhere in the file
    vk = find_vis_keys({"collision": col[:5], "sensor": d.get("sensor"),
                        "base_objects": [o for b in d["base"][:: max(1, len(d["base"]) // 20)] for o in b["objects"]]})
    obj_keys = sorted({k for b in d["base"] for o in b["objects"] for k in o})
    rec.update({"e_vis_fields": sorted(vk), "object_keys": obj_keys, "top_keys": sorted(d),
                "collision_keys": sorted({k for c in col for k in c})})
    return rec, actors, base


def draw_panel(img, meta_n, it_objs, ids, label, W, H):
    from PIL import Image, ImageDraw
    f_big, f_small = font(34), font(26)
    dr = ImageDraw.Draw(img)
    for i in ids:
        o = it_objs.get(i) if it_objs else None
        if o and o.get("2d_bbox"):
            (a, b), (c, e) = o["2d_bbox"]
            dr.rectangle([a, b, c, e], outline=BLUE, width=3)
            t = f"id {i} {TAGS.get(o['tag'], o['tag'])}"
            tb = dr.textbbox((0, 0), t, font=f_small)
            ty = e + 4 if e + 4 + tb[3] < H else max(0, b - tb[3] - 6)
            dr.rectangle([a, ty, a + tb[2] + 8, ty + tb[3] + 6], fill=(0, 0, 0))
            dr.text((a + 4, ty + 2), t, fill=BLUE, font=f_small)
    x1, y1, x2, y2 = meta_n[0] * W, meta_n[1] * H, meta_n[2] * W, meta_n[3] * H
    dr.rectangle([x1, y1, x2, y2], outline=GREEN, width=4)
    band = Image.new("RGB", (W, 56), (0, 0, 0))
    ImageDraw.Draw(band).text((12, 8), label, fill=(255, 255, 255), font=f_big)
    out = Image.new("RGB", (W, H + 56))
    out.paste(band, (0, 0))
    out.paste(img, (0, 56))
    return out


def crop_box(boxes_px, W, H, aspect=16 / 9, pad=0.6, min_w=320):
    x1 = min(b[0] for b in boxes_px); y1 = min(b[1] for b in boxes_px)
    x2 = max(b[2] for b in boxes_px); y2 = max(b[3] for b in boxes_px)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w = max((x2 - x1) * (1 + 2 * pad), (y2 - y1) * (1 + 2 * pad) * aspect, min_w)
    w = min(w, W); h = min(w / aspect, H); w = h * aspect
    l = min(max(cx - w / 2, 0), W - w); t = min(max(cy - h / 2, 0), H - h)
    return int(l), int(t), int(l + w), int(t + h)


def audit():
    import vlm_zero_shot as vz
    from PIL import Image, ImageDraw

    rows = load_list()
    for p in (CHECKS_CSV, ACTORS_CSV, CHECKS_JSON, OUT / "clips", OUT / "grids"):
        if p.exists():
            fail(f"{p} exists; refusing to overwrite")
    (OUT / "clips").mkdir()
    (OUT / "grids").mkdir()
    meta = synthetic_meta()
    recs, all_actors, grid_rows = [], [], {}
    for row in rows:
        p, s = row["rgb_path"], row["stem"]
        m = meta[p]
        rec, actors, base = check_clip(row, m)
        W, H, af, off = rec["W"], rec["H"], rec["accident_frame"], rec["annotations_start_offset"]
        nf = int(m["no_frames"])
        want = [af + k for k in OFFSETS]
        idx = [min(max(i, 0), nf - 1) for i in want]
        rec["frames_drawn"] = idx
        rec["frames_clamped"] = idx != want
        got = vz.decode({"video": str(DATASET / p), "rgb_path": p}, idx, budget=10 ** 12)
        sizes = {got[i].size for i in idx}
        rec["decoded_size"] = sorted(sizes)
        rec["decoded_size_eq_meta"] = sizes == {(W, H)}
        meta_n = [float(m[k]) for k in ("x1", "y1", "x2", "y2")]
        ids = rec.get("collision_ids", [])
        panels, crops_src = [], []
        for k, i in zip(OFFSETS, idx):
            tick = i + off
            objs = base.get(tick, {}) if base else {}
            rel = "accident_frame" if k == 0 else f"accident_frame {'+' if k > 0 else '-'} {abs(k)}"
            lab = f"frame {i} ({rel})   t = {i / rec['fps']:.2f} s   annotation tick {tick}"
            panels.append(draw_panel(got[i].copy(), meta_n, objs, ids, lab, W, H))
            crops_src.append(panels[-1])
        sheet = Image.new("RGB", (3 * W, H + 56 + 64), (0, 0, 0))
        ImageDraw.Draw(sheet).text(
            (12, 12), f"{s}   scenario {row['scenario_id']}   weather {row['weather']}   fold {row['val_fold']}   "
                      f"accident_frame {af} (accident_time {float(m['accident_time']):.2f} s, fps {rec['fps']:g})   "
                      f"green = metadata box, blue = colliding actors' 2D boxes (ids {ids}) at each frame's tick",
            fill=(255, 255, 255), font=font(36))
        for j, pnl in enumerate(panels):
            sheet.paste(pnl, (j * W, 64))
        out = OUT / "clips" / f"{s}.jpg"
        sheet.save(out, quality=90)
        # grid row: accident frame (whole) + crops around the boxes at the three frames
        bx = [[int(round(v)) for v in (meta_n[0] * W, meta_n[1] * H, meta_n[2] * W, meta_n[3] * H)]]
        for i in idx:
            for o in (base.get(i + off, {}) if base else {}).values():
                if o["id"] in ids and o.get("2d_bbox"):
                    (a, b), (c, e) = o["2d_bbox"]
                    bx.append([a, b, c, e])
        l, t, r, b = crop_box(bx, W, H)
        cells = [panels[1].crop((0, 56, W, H + 56)).resize((640, 360))]
        for pnl in panels:
            cells.append(pnl.crop((l, t + 56, r, b + 56)).resize((640, 360)))
        grid_rows.setdefault(row["scenario_id"], []).append((s, row, rec, cells, (l, t, r, b)))
        rec["grid_crop_px"] = [l, t, r, b]
        recs.append(rec)
        all_actors.extend(actors)
        print(f"{s}: a {rec.get('a_pass')} b {rec.get('b_pass')} c {rec.get('c_pass')} d {rec.get('d_pass')} "
              f"min share {rec.get('c_min_share_inside')} IoU {rec.get('c_iou_union')} -> {out.name}")
    for sid, gr in grid_rows.items():
        cw, chh, head = 640, 360, 34
        g = Image.new("RGB", (4 * cw + 30, 50 + len(gr) * (chh + head) + 10), (20, 20, 20))
        dg = ImageDraw.Draw(g)
        dg.text((10, 10), f"{sid}: col 1 = accident_frame (whole frame); cols 2-4 = crop around the boxes at "
                          f"accident_frame -10 / 0 / +10. Green = metadata box, blue = colliding actors.",
                fill=(255, 255, 255), font=font(22))
        for r_i, (s, row, rec, cells, crop) in enumerate(gr):
            y = 50 + r_i * (chh + head)
            dg.text((10, y + 4), f"{s} ({row['weather']}, fold {row['val_fold']}), accident_frame "
                                 f"{rec['accident_frame']}, crop x {crop[0]}-{crop[2]}, y {crop[1]}-{crop[3]}",
                    fill=(255, 255, 255), font=font(22))
            for c_i, cell in enumerate(cells):
                g.paste(cell, (10 + c_i * (cw + 3), y + head))
        g.save(OUT / "grids" / f"{sid}.jpg", quality=90)
    cols = ["stem", "scenario_id", "weather", "val_fold", "accident_frame", "annotations_start_offset",
            "annotation_found", "n_collision_records", "first_record_is_earliest", "collision_ids", "meta_box_px",
            "ann_box_px", "a_box_max_diff_px", "a_pass", "b_pass", "b_touches_edge", "n_actors", "n_actors_with_box",
            "c_min_share_inside", "c_iou_union", "c_iou_enclosing_box", "actor_enclosing_box_px", "c_pass",
            "first_collision_tick", "d_frame_from_annotation", "d_pass", "base_first_tick_eq_offset",
            "base_ticks_contiguous", "base_n_ticks", "no_frames", "accident_time_x_fps", "e_vis_fields",
            "object_keys", "frames_drawn", "frames_clamped", "decoded_size", "decoded_size_eq_meta"]
    wcsv(CHECKS_CSV, [{c: r.get(c) for c in cols} for r in recs], cols)
    acols = ["stem", "id", "tag", "in_base_at_collision_tick", "has_2d_bbox", "box_px", "box_area_px",
             "share_inside_meta", "in_base_at_frame_prev", "in_base_at_frame_acc", "in_base_at_frame_next"]
    wcsv(ACTORS_CSV, [{c: a.get(c) for c in acols} for a in all_actors], acols)
    wtext(CHECKS_JSON, json.dumps(recs, indent=1, default=str))
    print(f"wrote {len(recs)} clip images, {len(grid_rows)} grids, checks.csv, actors.csv, checks.json")


# ---------------------------------------------------------------- 4. figure 6 with the t* frame

def fig6():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import build_qlora_data as bq
    import vlm_zero_shot as vz
    from PIL import Image
    import io
    import pandas as pd

    for p in (FIG6_PNG, FIG6_CSV):
        if p.exists():
            fail(f"{p} exists; refusing to overwrite")
    ex = pd.read_csv(EXAMPLES)
    if len(ex) != 8:
        fail("examples.csv must hold the 8 figure-6 windows")
    man = pd.read_csv(MANIFEST, dtype={"fold": str})
    man = man[(man["fold"] == "1") & (man["window"] == "A")].set_index("rgb_path")
    e1 = pd.read_csv(VAL_EPOCH1).set_index("rgb_path")
    meta = synthetic_meta()
    fig, axes = plt.subplots(4, 4, figsize=(17, 12.5), constrained_layout=True)
    info = []
    for n, r in ex.iterrows():
        p = r["rgb_path"]
        mm, mt = man.loc[p], meta[p]
        fi, frames = bq.read_npz(REPO / mm["frames_path"])
        if fi != json.loads(mm["frame_indices"]):
            fail(f"{p}: cached frame indices differ from the manifest")
        fps = float(mm["fps"])
        times = np.array(fi) / fps
        k = int(np.argmin(np.abs(times - float(mm["t_star"]))))
        if k + 1 != int(r["frame_shown"]):
            fail(f"{p}: frame shown {k + 1} != examples.csv {r['frame_shown']}")
        af = int(mt["accident_frame"])
        # t* frame with the cache's own steps: vz.decode (452 x 254) -> bq.model_frames (448 x 256) -> JPEG q95
        clip = {"video": str(DATASET / p), "rgb_path": p}
        got = vz.decode(clip, sorted({fi[k], af}))
        rebuilt = bq.model_frames(got, [fi[k], af])
        jpg = [np.asarray(Image.open(io.BytesIO(bq.jpeg_bytes(x).tobytes())).convert("RGB")) for x in rebuilt]
        diff_shown = int(np.abs(jpg[0].astype(int) - frames[k].astype(int)).max())
        tstar_img = jpg[1]
        tb = [mm["x1_1000"], mm["y1_1000"], mm["x2_1000"], mm["y2_1000"]]
        pb = json.loads(e1.loc[p, "bbox"])
        pb = [min(pb[0], pb[2]), min(pb[1], pb[3]), max(pb[0], pb[2]), max(pb[1], pb[3])]
        row_i, col_i = n // 2, (n % 2) * 2
        for j, (img, title) in enumerate((
                (frames[k], f"figure 6 frame: {k + 1}/16 = video frame {fi[k]}, t = {times[k]:.2f} s"),
                (tstar_img, f"t* frame: video frame {af} (accident_frame), t = {af / fps:.2f} s"))):
            ax = axes[row_i, col_i + j]
            h, w = img.shape[:2]
            ax.imshow(img)
            for b, c in ((tb, "#00c000"), (pb, "#ff2020")):
                ax.add_patch(Rectangle((b[0] / 1000 * w, b[1] / 1000 * h), (b[2] - b[0]) / 1000 * w,
                                       (b[3] - b[1]) / 1000 * h, fill=False, ec=c, lw=1.8))
            ax.set_xticks([]); ax.set_yticks([])
            head = (f"{r['category']}: {Path(p).stem}\n" if j == 0 else f"true {r['true_type']} / epoch-1 "
                    f"{r['pred_type_epoch1']}; window {mm['win_start_s']:.2f}-{mm['win_end_s']:.2f} s\n")
            ax.set_title(head + title, fontsize=7.5)
        info.append({"category": r["category"], "rgb_path": p, "frame_shown_k": k + 1, "frame_shown_index": fi[k],
                     "frame_shown_time_s": times[k], "accident_frame": af, "accident_time": float(mt["accident_time"]),
                     "tstar_frame_time_s": af / fps, "gap_s": af / fps - times[k],
                     "t_star_inside": mm["t_star_inside"], "win_start_s": mm["win_start_s"],
                     "win_end_s": mm["win_end_s"], "rebuilt_shown_frame_max_pixel_diff_vs_cache": diff_shown})
    fig.suptitle("Figure 6 windows: left = the cached frame figure 6 used (nearest t* inside window A), right = the "
                 "frame at t* (accident_frame), same steps as the cache.\nGreen = true (metadata) box, red = epoch-1 "
                 "box; both drawn on both panels.", fontsize=10)
    fig.savefig(FIG6_PNG, dpi=150)
    plt.close(fig)
    wcsv(FIG6_CSV, info, list(info[0]))
    for i in info:
        print(i)


# ---------------------------------------------------------------- 5. report

def report():
    if REPORT.exists():
        fail(f"{REPORT} exists; refusing to overwrite")
    for p in (CHECKS_JSON, ACTORS_CSV, FIG6_CSV, VISUAL_CSV):
        if not p.is_file():
            fail(f"{p} missing")
    recs = json.loads(CHECKS_JSON.read_text(encoding="utf-8"))
    actors = rcsv(ACTORS_CSV)
    f6 = rcsv(FIG6_CSV)
    vis = {r["stem"]: r for r in rcsv(VISUAL_CSV)}
    if set(vis) != {r["stem"] for r in recs}:
        fail("visual.csv must have one row per audited clip")
    L = ["GROUND-TRUTH BOX AUDIT (synthetic only; no real clip, label or metadata file was opened; no label, manifest "
         "or frozen file was changed)", "",
         f"Clips: {len(recs)} = 3 per scenario x 14 scenarios (synthetic_split.csv scenario_id), numpy default_rng(0); "
         f"list in LIST.txt (written before any image).",
         "Sources: metadata-synthetic.csv (box x1,y1,x2,y2, accident_frame, annotations_start_offset), the clip's "
         "annotation JSON (collision[0] = first collision record; base[tick].objects = per-tick objects with 2d_bbox).",
         "Annotation code (ACCIDENT/generation/.../core): collision_bbox = enclosing box of the colliding actors' "
         "projected 3D boxes, clipped to the image (collisions.py, get_circumferential_bbox); base 2d_bbox = projected "
         "box shrunk to the visible segmentation pixels, and objects with too few visible pixels are dropped "
         "(bbox_segmentation.py, filter_bboxes_by_segmentation).", "",
         "CHECKS", "  a: metadata box == collision[0].collision_bbox (pixels; pass if max |diff| <= 0.5 px)",
         "  b: metadata box inside the image (0 <= x1 < x2 <= 1, same for y); 'edge' = box touches the image border",
         "  c: each colliding actor (collision[0].ids) has a 2D box at the collision tick that overlaps the metadata box"
         " (min share = smallest share of an actor box inside the metadata box; IoU = metadata box vs the pixel union "
         "of the actor boxes)",
         "  d: collision[0].iteration - annotations_start_offset == accident_frame", ""]
    hdr = (f"{'clip':32s} {'a':>2s} {'a_px':>5s} {'b':>2s} {'edge':12s} {'c':>2s} {'act':>4s} {'minshr':>6s} "
           f"{'IoU_u':>6s} {'d':>2s} {'tick':>5s}")
    L.append(hdr)
    yn = lambda v: "Y" if v else "N"  # noqa: E731
    for r in recs:
        L.append(f"{r['stem']:32s} {yn(r.get('a_pass')):>2s} {r.get('a_box_max_diff_px', float('nan')):5.2f} "
                 f"{yn(r.get('b_pass')):>2s} {r.get('b_touches_edge') or '-':12s} {yn(r.get('c_pass')):>2s} "
                 f"{r.get('n_actors_with_box')}/{r.get('n_actors')}  "
                 f"{(r['c_min_share_inside'] if r.get('c_min_share_inside') is not None else float('nan')):6.3f} "
                 f"{(r['c_iou_union'] if r.get('c_iou_union') is not None else float('nan')):6.3f} "
                 f"{yn(r.get('d_pass')):>2s} {r.get('first_collision_tick'):>5}")
    n = len(recs)
    L += ["", "TOTALS", *(f"  {k}: pass {sum(bool(r.get(k + '_pass')) for r in recs)}/{n}" for k in "abcd"),
          f"  b: boxes touching the image border: {sum(bool(r.get('b_touches_edge')) for r in recs)}/{n}",
          f"  c: colliding actors {len(actors)}; with a 2D box at the collision tick "
          f"{sum(a['has_2d_bbox'] == 'True' for a in actors)}; share inside the metadata box = 1.000 for "
          f"{sum(a['share_inside_meta'] not in ('', 'None') and float(a['share_inside_meta']) >= 0.9995 for a in actors)}"
          f"; IoU (union) median {np.median([r['c_iou_union'] for r in recs if r.get('c_iou_union') is not None]):.3f},"
          f" min {min(r['c_iou_union'] for r in recs if r.get('c_iou_union') is not None):.3f}",
          f"  d: base first tick == annotations_start_offset {sum(bool(r.get('base_first_tick_eq_offset')) for r in recs)}"
          f"/{n}; base ticks contiguous {sum(bool(r.get('base_ticks_contiguous')) for r in recs)}/{n}; number of base "
          f"ticks == no_frames {sum(r.get('base_n_ticks') == r.get('no_frames') for r in recs)}/{n}; accident_time x fps "
          f"== accident_frame {sum(abs(r['accident_time_x_fps'] - r['accident_frame']) < 1e-6 for r in recs)}/{n}; "
          f"collision[0] is the earliest record {sum(bool(r.get('first_record_is_earliest')) for r in recs)}/{n}",
          f"  frames: decoded size == metadata size {sum(bool(r.get('decoded_size_eq_meta')) for r in recs)}/{n}; "
          f"frames clamped at the clip ends {sum(bool(r.get('frames_clamped')) for r in recs)}/{n}",
          "", "e. VISIBILITY / OCCLUSION FIELDS"]
    vk = sorted({k for r in recs for k in r.get("e_vis_fields", [])})
    ok = sorted({tuple(r.get("object_keys", [])) for r in recs})
    L += [f"  fields whose name contains {list(VIS_WORDS)}: {vk if vk else 'none in any of the 42 files'}",
          f"  object keys in base: {[list(k) for k in ok]}; collision record keys: "
          f"{sorted({tuple(r.get('collision_keys', [])) for r in recs})}; top-level keys: "
          f"{sorted({tuple(r.get('top_keys', [])) for r in recs})}",
          "  Indirect signal only (from the annotation code, not a field): an object is in base at a tick only if "
          "enough of its pixels are visible in the segmentation image, and its 2d_bbox covers its visible pixels.",
          f"  colliding actors missing from base: at the collision tick "
          f"{sum(a['in_base_at_collision_tick'] != 'True' for a in actors)}/{len(actors)}, at accident_frame - 10 "
          f"{sum(a['in_base_at_frame_prev'] != 'True' for a in actors)}/{len(actors)}, at accident_frame + 10 "
          f"{sum(a['in_base_at_frame_next'] != 'True' for a in actors)}/{len(actors)}"]
    miss = [a for a in actors if "False" in (a["in_base_at_collision_tick"], a["in_base_at_frame_prev"],
                                             a["in_base_at_frame_next"])]
    for a in miss:
        L.append(f"    {a['stem']} id {a['id']} {a['tag']}: in base at -10 {a['in_base_at_frame_prev']}, "
                 f"at the collision tick {a['in_base_at_collision_tick']}, at +10 {a['in_base_at_frame_next']}")
    L += ["", "PER-ACTOR DETAIL (share of the actor's 2D box inside the metadata box; tag)"]
    for a in actors:
        sh = a["share_inside_meta"]
        L.append(f"  {a['stem']:32s} id {a['id']:>6s} {a['tag']:10s} box {a['box_px'] or '-':24s} share "
                 f"{float(sh):.3f}" if sh not in ("", "None") else f"  {a['stem']:32s} id {a['id']:>6s} "
                 f"{a['tag']:10s} no 2D box at the collision tick")
    L += ["", "CLIPS WHERE A CHECK FAILS OR THE BOXES LOOK OFF"]
    flagged = 0
    for r in recs:
        why = [f"check {k} fails" for k in "abcd" if not r.get(f"{k}_pass")]
        v = vis[r["stem"]]
        if v["looks_off"].strip().lower() == "yes":
            why.append(f"visual judgement: {v['note']}")
        if why:
            flagged += 1
            L.append(f"  {r['stem']}: " + "; ".join(why))
    if not flagged:
        L.append("  none")
    L += ["", "VISUAL JUDGEMENT (all entries in this section are my visual judgement from clips/*.jpg and grids/*.jpg, "
          "not computed checks)"]
    for r in recs:
        v = vis[r["stem"]]
        L.append(f"  {r['stem']:32s} looks off: {v['looks_off']:3s}  {v['note']}")
    L += ["", "FIGURE 6 WITH THE t* FRAME (fig6_tstar.png)",
          "  Left panel of each pair = the cached frame figure 6 used (nearest t* inside window A); right = the frame at "
          "accident_frame built with the cache's own steps (vz.decode -> model_frames -> JPEG q95). Same boxes on both.",
          "  The figure-6 frame rebuilt from the video with the same steps vs the cache: max pixel diff "
          f"{max(int(r['rebuilt_shown_frame_max_pixel_diff_vs_cache']) for r in f6)} over the 8 windows."]
    for r in f6:
        L.append(f"  {r['category']:14s} {Path(r['rgb_path']).stem:30s} figure frame {int(r['frame_shown_index']):4d} "
                 f"at {float(r['frame_shown_time_s']):6.2f} s; t* frame {int(r['accident_frame']):4d} at "
                 f"{float(r['tstar_frame_time_s']):6.2f} s; gap {float(r['gap_s']):+6.2f} s; t* inside window "
                 f"{r['t_star_inside']}")
    if VISUAL_CSV.is_file() and (OUT / "visual_fig6.txt").is_file():
        L += ["", "  Visual judgement on fig6_tstar.png:"] + [
            "    " + x for x in (OUT / "visual_fig6.txt").read_text(encoding="utf-8").strip().splitlines()]
    L += ["", "Nothing was fixed. Files: LIST.txt, list.csv, checks.csv, checks.json, actors.csv, clips/*.jpg (42), "
          "grids/*.jpg (14), fig6_tstar.png, fig6_tstar.csv, visual.csv, visual_fig6.txt, run_log_*.txt, report.txt."]
    wtext(REPORT, "\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for a in ("--list", "--audit", "--fig6", "--report"):
        g.add_argument(a, action="store_true")
    a = ap.parse_args()
    if a.list:
        make_list()
    elif a.audit:
        audit()
    elif a.fig6:
        fig6()
    else:
        report()


if __name__ == "__main__":
    main()
