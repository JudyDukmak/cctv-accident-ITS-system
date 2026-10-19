"""Check whether synthetic clips share CARLA simulation runs (same crash, other camera/weather).

READ-ONLY. Reads only ``metadata-synthetic.csv`` and the synthetic annotation files
listed in its ``annotations_path`` column. Never writes to the dataset and never
touches real videos, real metadata or the frozen real test set. Outputs go to
``synthetic_check/`` next to this script.

PART 1 (metadata, all rows): sim_tick = accident_frame + annotations_start_offset,
summarised per (map, type, weather) and per (map, type).

PART 2 (annotations, fixed sample, seed 42): first "collision" entry per clip ->
collision actor ids, iteration, crash location; tests T1 / T2 / T3.

Notes on the annotation files, checked against the data and the generator code:
  * On disk, ``annotations/<stem>.json.gz`` is a FOLDER ``annotations/<stem>.json/``
    holding one plain JSON file ``<stem>.json`` (decompressed). The loader below
    accepts that layout, a flat ``<stem>.json`` and the original ``.json.gz``.
  * For actors (vehicles, pedestrians) ``base[].objects[].location`` is
    ``actor.bounding_box.location`` (bbox_segmentation.py:302), i.e. the box offset
    in the actor's OWN frame (about (0, 0, 0.8)), NOT a world position. It is saved
    as ``actor_bbox_offset`` for transparency but not used for distances.
  * The crash world location is therefore estimated by back-projecting the
    bottom-centre of the first collision_bbox through the camera (sensor pose from
    the json, fov 90 and 1920x1080 as in every scenario YAML) onto the local ground
    plane. Ground height = median bottom of static world objects (whose location IS
    in world coordinates). The camera model is validated on those static objects.

Usage:
    py check_synthetic_runs.py [--dataset-path PATH] [--out synthetic_check]
"""

import argparse
import gzip
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATASET = Path(
    r"C:\Users\ASUS\Documents\Uni\Senior 2\cctv-accident-ITS-system\ACCIDENT\dataset"
)
SEED = 42
MAX_SAMPLE = 120
TICK_TOL = 2
CAMERA_FOV = 90.0  # every scenario YAML in generation/.../scenarios uses 90
STATIC_ID_MIN = 2**32  # environment objects have huge ids; actors have small ids
GROUND_RADIUS = 40.0  # m, static objects used for local ground height
T1_FORCED = [("Town03", "t-bone", "rain"), ("Town05", "sideswipe", "clear"), ("Town06", "head-on", "clear")]
N_GROUPS = 6
N_PER_GROUP = 6
T3_EXTRA_PER_MAPTYPE = 3


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------- #
# PART 1 - metadata only
# --------------------------------------------------------------------------- #
def tick_stats(s):
    med = s.median()
    return pd.Series(
        {
            "n_clips": len(s),
            "n_distinct_sim_tick": s.nunique(),
            "sim_tick_min": s.min(),
            "sim_tick_max": s.max(),
            "sim_tick_std": s.std(ddof=0),
            "share_within_2_of_median": float(((s - med).abs() <= TICK_TOL).mean()),
        }
    )


def part1(df, out_dir):
    section("PART 1 - sim_tick = accident_frame + annotations_start_offset (all rows)")
    df["sim_tick"] = df["accident_frame"] + df["annotations_start_offset"]
    by_mtw = df.groupby(["map", "type", "weather"])["sim_tick"].apply(tick_stats).unstack().reset_index()
    by_mt = df.groupby(["map", "type"])["sim_tick"].apply(tick_stats).unstack().reset_index()
    for t in (by_mtw, by_mt):
        for c in ["n_clips", "n_distinct_sim_tick", "sim_tick_min", "sim_tick_max"]:
            t[c] = t[c].astype(int)

    out = pd.concat(
        [by_mtw.assign(level="map_type_weather"), by_mt.assign(level="map_type", weather="ALL")],
        ignore_index=True,
    )
    cols = ["level", "map", "type", "weather"] + [c for c in by_mtw.columns if c not in ("map", "type", "weather")]
    out[cols].to_csv(out_dir / "sim_tick_summary.csv", index=False, float_format="%.4f")

    fmt = {"sim_tick_std": "{:.2f}".format, "share_within_2_of_median": "{:.2f}".format}
    print(f"\nPer (map, type, weather): {len(by_mtw)} groups (std is population std)")
    print(by_mtw.to_string(index=False, formatters=fmt))
    print(f"\nPer (map, type), all weathers: {len(by_mt)} groups")
    print(by_mt.to_string(index=False, formatters=fmt))
    print(f"\nsaved {out_dir / 'sim_tick_summary.csv'}")
    return df


# --------------------------------------------------------------------------- #
# Annotation loading and camera geometry
# --------------------------------------------------------------------------- #
def resolve_annotation(dataset_root, ann_path):
    """Return the file holding the JSON, trying the documented and on-disk layouts."""
    p = dataset_root / ann_path.replace("\\", "/")
    name = p.name[:-3] if p.name.endswith(".gz") else p.name  # <stem>.json
    candidates = [p, p.with_name(name) / name, p.with_name(name)]
    for c in candidates:
        if c.is_file():
            return c
    return None


def load_annotation(path):
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("ascii"))


def carla_matrix(loc, rot):
    """4x4 transform, identical to ClientSideBoundingBoxes.get_matrix."""
    cy, sy = np.cos(np.radians(rot["yaw"])), np.sin(np.radians(rot["yaw"]))
    cr, sr = np.cos(np.radians(rot["roll"])), np.sin(np.radians(rot["roll"]))
    cp, sp = np.cos(np.radians(rot["pitch"])), np.sin(np.radians(rot["pitch"]))
    m = np.identity(4)
    m[:3, 3] = [loc["x"], loc["y"], loc["z"]]
    m[0, :3] = [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr]
    m[1, :3] = [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr]
    m[2, :3] = [sp, -cp * sr, cp * cr]
    return m


class Camera:
    def __init__(self, sensor, width, height, fov=CAMERA_FOV):
        self.m = carla_matrix(sensor["location"], sensor["rotation"])
        self.m_inv = np.linalg.inv(self.m)
        self.f = width / (2.0 * np.tan(np.radians(fov) / 2.0))
        self.cx, self.cy = width / 2.0, height / 2.0
        self.pos = self.m[:3, 3]

    def project(self, xyz):
        """World point -> (u, v) pixel, or None if behind the camera."""
        s = self.m_inv @ np.append(xyz, 1.0)  # sensor frame: x fwd, y right, z up
        if s[0] <= 0:
            return None
        return self.f * s[1] / s[0] + self.cx, self.f * -s[2] / s[0] + self.cy

    def ray_to_ground(self, u, v, ground_z):
        d_sensor = np.array([1.0, (u - self.cx) / self.f, -(v - self.cy) / self.f])
        d = self.m[:3, :3] @ d_sensor
        if d[2] >= -1e-6:
            return None  # ray does not hit the ground in front of the camera
        t = (ground_z - self.pos[2]) / d[2]
        return self.pos + t * d


def static_objects(frame):
    return [o for o in frame["objects"] if o["id"] >= STATIC_ID_MIN]


def ground_height(statics, near_xy=None):
    pts = [(o["location"]["x"], o["location"]["y"], o["location"]["z"] - o["extent"]["z"]) for o in statics]
    if near_xy is not None:
        pts = [p for p in pts if np.hypot(p[0] - near_xy[0], p[1] - near_xy[1]) <= GROUND_RADIUS]
    if len(pts) < 3:
        return None
    return float(np.median([p[2] for p in pts]))


def analyse_clip(row, ann):
    """Extract first-collision facts. Returns (record dict, (n_static_checked, n_inside))."""
    rec = {}
    coll = ann.get("collision") or []
    if not coll:
        rec["error"] = "empty collision list"
        return rec, (0, 0)
    c0 = coll[0]
    ids = tuple(sorted(c0["ids"]))
    it = int(c0["iteration"])
    rec.update(collision_ids=" ".join(map(str, ids)), collision_iteration=it, n_collision_entries=len(coll))
    rec["iteration_eq_sim_tick"] = bool(it == row["sim_tick"])
    rec["iteration_minus_sim_tick"] = it - int(row["sim_tick"])

    sensor = ann["sensor"]
    for k in "xyz":
        rec[f"sensor_{k}"] = sensor["location"][k]
    for k in ("pitch", "yaw", "roll"):
        rec[f"sensor_{k}"] = sensor["rotation"][k]

    frame = next((fr for fr in ann["base"] if fr["iteration"] == it), None)
    if frame is None:
        rec["error"] = f"iteration {it} not in base"
        return rec, (0, 0)
    actors = {o["id"]: o for o in frame["objects"] if o["id"] in ids}
    rec["actor_bbox_offset"] = json.dumps(
        {i: [round(actors[i]["location"][k], 3) for k in "xyz"] for i in ids if i in actors}
    )
    rec["n_colliding_actors_in_base"] = len(actors)

    cam = Camera(sensor, int(row["width"]), int(row["height"]))
    statics = static_objects(frame)

    # Camera-model validation: project static objects' world centres into the image.
    checked = inside = 0
    for o in statics:
        uv = cam.project(np.array([o["location"][k] for k in "xyz"]))
        (x1, y1), (x2, y2) = o["2d_bbox"]
        if uv is None:
            continue
        checked += 1
        inside += int(x1 - 5 <= uv[0] <= x2 + 5 and y1 - 5 <= uv[1] <= y2 + 5)

    (bx1, by1), (bx2, by2) = c0["collision_bbox"]
    u, v = (bx1 + bx2) / 2.0, float(by2)
    rec["collision_bbox_clipped"] = bool(by2 >= int(row["height"]) - 1 or bx1 <= 0 or bx2 >= int(row["width"]) - 1)
    gz = ground_height(statics)
    if gz is None:
        gz, src = 0.0, "default0"
    else:
        src = "statics_all"
    p = cam.ray_to_ground(u, v, gz)
    if p is not None:  # refine ground height with static objects near the first estimate
        gz2 = ground_height(statics, near_xy=p[:2])
        p2 = cam.ray_to_ground(u, v, gz2) if gz2 is not None else None
        if p2 is not None:
            p, gz, src = p2, gz2, "statics_near"
    rec["ground_z"], rec["ground_z_source"] = gz, src
    if p is None:
        rec["error"] = "collision ray does not hit ground"
    else:
        rec["crash_x"], rec["crash_y"], rec["crash_z"] = (float(x) for x in p)
        rec["crash_dist_to_camera_m"] = float(np.linalg.norm(p - cam.pos))
    return rec, (checked, inside)


# --------------------------------------------------------------------------- #
# PART 2 - sampling, loading, tests
# --------------------------------------------------------------------------- #
def build_sample(df):
    rng = random.Random(SEED)
    picks = {}  # row index -> set of test tags

    def add(idx, tag):
        if idx in picks or len(picks) < MAX_SAMPLE:
            picks.setdefault(idx, set()).add(tag)

    # T1: same (map, type, weather), different cameras.
    groups = sorted(df.groupby(["map", "type", "weather"]).groups)
    forced = [g for g in T1_FORCED if g in groups]
    for g in T1_FORCED:
        if g not in groups:
            print(f"  note: forced T1 group {g} not in metadata")
    others = [g for g in groups if g not in forced]
    t1_groups = forced + rng.sample(others, max(0, N_GROUPS - len(forced)))
    for g in t1_groups:
        sub = df[(df["map"] == g[0]) & (df["type"] == g[1]) & (df["weather"] == g[2])]
        for idx in rng.sample(sorted(sub.index), min(N_PER_GROUP, len(sub))):
            add(idx, "T1")

    # T2: same (map, type, camera), >= 3 weathers.
    nw = df.groupby(["map", "type", "camera_position"])["weather"].nunique()
    combos = sorted(nw[nw >= 3].index)
    t2_combos = rng.sample(combos, min(N_GROUPS, len(combos)))
    for g in t2_combos:
        sub = df[(df["map"] == g[0]) & (df["type"] == g[1]) & (df["camera_position"] == g[2])]
        for idx in sorted(sub.index)[:N_PER_GROUP]:
            add(idx, "T2")

    # T3: a few extra clips per (map, type) so every combination has varied weather + camera.
    for mt in sorted(df.groupby(["map", "type"]).groups):
        sub = df[(df["map"] == mt[0]) & (df["type"] == mt[1])]
        free = [i for i in sorted(sub.index) if i not in picks]
        for idx in rng.sample(free, min(T3_EXTRA_PER_MAPTYPE, len(free))):
            add(idx, "T3")
    return picks, t1_groups, t2_combos


def max_pairwise(points):
    pts = [p for p in points if not np.any(np.isnan(p))]
    if len(pts) < 2:
        return float("nan")
    return max(float(np.linalg.norm(a - b)) for a, b in itertools.combinations(pts, 2))


def group_report(res, label, keys):
    xyz = res[["crash_x", "crash_y", "crash_z"]].to_numpy(dtype=float)
    sxyz = res[["sensor_x", "sensor_y", "sensor_z"]].to_numpy(dtype=float)
    print(
        f"  {label:<38} n={len(res):<2} "
        f"distinct_id_sets={res['collision_ids'].nunique():<2} "
        f"distinct_iterations={res['collision_iteration'].nunique():<2} "
        f"max_crash_dist_m={max_pairwise(xyz):7.1f}  "
        f"max_sensor_dist_m={max_pairwise(sxyz):7.1f}"
    )
    for _, r in res.iterrows():
        print(
            f"      {r['clip']:<32} ids=({r['collision_ids']}) it={r['collision_iteration']} "
            f"crash=({r['crash_x']:.1f},{r['crash_y']:.1f})"
        )


def part2(df, dataset_root, out_dir):
    section("PART 2 - annotations on a fixed sample (seed 42)")
    picks, t1_groups, t2_combos = build_sample(df)
    print(f"Sampled clips: {len(picks)} (cap {MAX_SAMPLE})")

    records, n_checked, n_inside = [], 0, 0
    for idx in sorted(picks):
        row = df.loc[idx]
        clip = Path(row["rgb_path"].replace("\\", "/")).stem
        base = {
            "clip": clip, "map": row["map"], "type": row["type"], "weather": row["weather"],
            "camera_position": int(row["camera_position"]), "sim_tick": int(row["sim_tick"]),
            "tests": "+".join(sorted(picks[idx])),
        }
        path = resolve_annotation(dataset_root, row["annotations_path"])
        if path is None:
            print(f"  MISSING annotation: {row['annotations_path']}")
            records.append({**base, "error": "missing annotation"})
            continue
        try:
            ann = load_annotation(path)
            rec, (c, i) = analyse_clip(row, ann)
        except (OSError, ValueError, KeyError, TypeError, EOFError) as e:
            print(f"  FAILED to parse {path}: {type(e).__name__}: {e}")
            records.append({**base, "error": f"parse: {type(e).__name__}"})
            continue
        n_checked += c
        n_inside += i
        if "error" in rec:
            print(f"  {clip}: {rec['error']}")
        records.append({**base, "annotation_file": str(path.relative_to(dataset_root)), **rec})

    res = pd.DataFrame(records)
    for c in ["crash_x", "crash_y", "crash_z", "collision_iteration", "collision_ids", "error"]:
        if c not in res.columns:
            res[c] = np.nan
    res.to_csv(out_dir / "run_check.csv", index=False, float_format="%.4f")
    print(f"saved {len(res)} rows to {out_dir / 'run_check.csv'}")

    ok = res[res["collision_iteration"].notna()].copy()
    print(f"\nClips with a parsed first collision: {len(ok)} / {len(res)}")
    if ok.empty:
        print("Nothing to test.")
        return

    # Sanity checks.
    mism = ok[~ok["iteration_eq_sim_tick"].astype(bool)]
    print(f"first-collision iteration != sim_tick: {len(mism)} / {len(ok)}")
    if len(mism):
        print(f"  (iteration - sim_tick) values: {mism['iteration_minus_sim_tick'].value_counts().to_dict()}")
    if len(mism) > 0.1 * len(ok):
        print("  WARNING: mismatches are common - accident_frame/offset may not mark the first collision tick.")
    rate = n_inside / n_checked if n_checked else float("nan")
    print(
        f"camera-model check: {n_inside} / {n_checked} static-object world centres project inside "
        f"their own 2d_bbox ({rate:.1%}). High = crash locations below are trustworthy."
    )
    if n_checked and rate < 0.8:
        print("  WARNING: camera model does not reproduce the 2d boxes well; treat crash distances with care.")
    print(f"collision_bbox clipped at an image border: {int(ok['collision_bbox_clipped'].sum())} / {len(ok)}")
    print(f"ground_z source: {ok['ground_z_source'].value_counts().to_dict()}")

    section("T1 - same (map, type, weather), different camera_position")
    for g in t1_groups:
        sub = ok[(ok["map"] == g[0]) & (ok["type"] == g[1]) & (ok["weather"] == g[2]) & ok["tests"].str.contains("T1")]
        if len(sub):
            group_report(sub, "_".join(g), g)

    section("T2 - same (map, type, camera_position), different weathers")
    for g in t2_combos:
        sub = ok[(ok["map"] == g[0]) & (ok["type"] == g[1]) & (ok["camera_position"] == g[2]) & ok["tests"].str.contains("T2")]
        if len(sub):
            group_report(sub, f"{g[0]}_{g[1]}_cam{int(g[2]):02d}", g)

    section("T3 - same (map, type), different weather AND camera (all sampled clips)")
    all_d, all_same_ids = [], []
    for (mp, tp), sub in ok.groupby(["map", "type"]):
        d, same_ids = [], 0
        for (_, a), (_, b) in itertools.combinations(sub.iterrows(), 2):
            if a["weather"] == b["weather"] or a["camera_position"] == b["camera_position"]:
                continue
            pa = np.array([a["crash_x"], a["crash_y"], a["crash_z"]], dtype=float)
            pb = np.array([b["crash_x"], b["crash_y"], b["crash_z"]], dtype=float)
            if np.any(np.isnan(pa)) or np.any(np.isnan(pb)):
                continue
            d.append(float(np.linalg.norm(pa - pb)))
            same_ids += int(a["collision_ids"] == b["collision_ids"])
        if d:
            print(
                f"  {mp}_{tp:<10} clips={len(sub):<3} pairs={len(d):<4} "
                f"dist_m min={min(d):6.1f} median={np.median(d):6.1f} max={max(d):6.1f}  "
                f"pairs_with_identical_ids={same_ids}"
            )
            all_d += d
            all_same_ids.append(same_ids)
        else:
            print(f"  {mp}_{tp:<10} clips={len(sub):<3} no valid pairs")
    if all_d:
        print(
            f"\n  ALL pairs={len(all_d)}  dist_m min={min(all_d):.1f} median={np.median(all_d):.1f} "
            f"max={max(all_d):.1f}  pairs_with_identical_ids={sum(all_same_ids)}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "synthetic_check")
    args = ap.parse_args()

    dataset_root = args.dataset_path.resolve()
    if (dataset_root / "dataset" / "metadata-synthetic.csv").is_file():
        dataset_root = dataset_root / "dataset"
    meta_csv = dataset_root / "metadata-synthetic.csv"
    if not meta_csv.is_file():
        sys.exit(f"metadata-synthetic.csv not found in {dataset_root}")
    out_dir = args.out.resolve()
    if out_dir == dataset_root or dataset_root in out_dir.parents:
        sys.exit(f"refusing to write inside the dataset: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset root : {dataset_root}")
    print(f"Metadata     : {meta_csv}")
    print(f"Output dir   : {out_dir}")

    df = pd.read_csv(meta_csv)
    need = {"rgb_path", "annotations_path", "type", "map", "weather", "camera_position",
            "accident_frame", "annotations_start_offset", "width", "height"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"metadata is missing columns: {sorted(missing)}")
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        df = part1(df, out_dir)
        part2(df, dataset_root, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
