"""Inspect whether the trailing {number} in synthetic clip names is an event id.

Synthetic clip names look like ``Town05_head-on_clear_32``, i.e.
``{map}_{type}_{weather}_{number}``. This script tests (it does not assume)
what ``{number}`` means, by checking:
  - how many videos share one event_key, and whether their accident_time agrees;
  - how {number} values spread over (map, type, weather) and over weathers;
  - visual montages of supposedly-same / supposedly-different events.

READ-ONLY. It reads only ``metadata-synthetic.csv`` and the videos under
``synthetic_videos/videos``. It never writes to the dataset and never touches
real videos, real metadata or the frozen real test set. All outputs go to
``synthetic_check/`` next to this script.

Usage:
    py inspect_synthetic_events.py [--dataset-path PATH] [--out synthetic_check]
"""

import argparse
import random
import re
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULT_DATASET = Path(
    r"C:\Users\ASUS\Documents\Uni\Senior 2\cctv-accident-ITS-system\ACCIDENT\dataset"
)
SEED = 42
MAX_FAIL_FRACTION = 0.01
STD_THRESHOLD = 0.5  # seconds
N_TILES = 6

# {map}_{type}_{weather}_{number}; type may contain hyphens (head-on, t-bone, rear-end).
EVENT_RE = re.compile(
    r"^(?P<map>Town[0-9]+[A-Za-z]*)_(?P<type>[a-z]+(?:-[a-z]+)*)_(?P<weather>[a-z]+)_(?P<number>[0-9]+)$"
)


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def list_dir(path, limit=10):
    try:
        entries = sorted(path.iterdir(), key=lambda p: p.name)
    except OSError as e:
        return [], f"cannot list {path}: {e}"
    return entries, None


# --------------------------------------------------------------------------- #
# Step 1 - inspect
# --------------------------------------------------------------------------- #
def step1_inspect(df, dataset_root, videos_root):
    section("STEP 1 - INSPECT")
    print(f"Rows: {len(df)}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print("\nFirst 5 rows:")
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(df.head(5).to_string())

    print("\n10 sample rgb_path values (seed 42) and whether the file exists:")
    sample = df["rgb_path"].sample(n=min(10, len(df)), random_state=SEED)
    for p in sample:
        full = dataset_root / p.replace("\\", "/")
        print(f"  [{'OK     ' if full.is_file() else 'MISSING'}] {p}")

    df["video_file"] = [dataset_root / p.replace("\\", "/") for p in df["rgb_path"]]
    exists = df["video_file"].map(lambda f: f.is_file())
    print(f"\nAll rows: {int(exists.sum())} / {len(df)} rgb_path files exist on disk.")

    print(f"\nFolder layout of {videos_root} (first 2 levels, max 10 entries each):")
    level1, err = list_dir(videos_root)
    if err:
        print("  " + err)
    else:
        print(f"  {len(level1)} entries at level 1")
        for e in level1[:10]:
            if e.is_dir():
                level2, err2 = list_dir(e)
                if err2:
                    print(f"  {e.name}/  ({err2})")
                    continue
                print(f"  {e.name}/  ({len(level2)} entries)")
                for f in level2[:10]:
                    print(f"      {f.name}{'/' if f.is_dir() else ''}")
                if len(level2) > 10:
                    print(f"      ... ({len(level2) - 10} more)")
            else:
                print(f"  {e.name}")

    # Derive event_key from the file stem.
    stems = df["rgb_path"].map(lambda p: Path(p.replace("\\", "/")).stem)
    matches = stems.map(EVENT_RE.match)
    failed = stems[matches.isna()]
    print(f"\nRegex: {EVENT_RE.pattern}")
    print(f"Paths NOT matching: {len(failed)} / {len(df)}")
    for s in failed.head(10):
        print(f"  no match: {s}")
    if len(failed) > MAX_FAIL_FRACTION * len(df):
        print(
            f"\nSTOP: {len(failed) / len(df):.1%} of paths do not match the "
            f"expected '{{map}}_{{type}}_{{weather}}_{{number}}' pattern "
            f"(limit {MAX_FAIL_FRACTION:.0%}). Not guessing - fix the regex first."
        )
        sys.exit(1)

    df = df[matches.notna()].copy()
    m = matches[matches.notna()]
    df["event_key"] = stems[matches.notna()]
    df["n_map"] = m.map(lambda x: x["map"])
    df["n_type"] = m.map(lambda x: x["type"])
    df["n_weather"] = m.map(lambda x: x["weather"])
    df["n_number"] = m.map(lambda x: int(x["number"]))
    df["exists"] = exists[df.index]

    # Cross-check name fields against the metadata columns (do not assume they agree).
    print("\nName fields vs metadata columns (rows that disagree):")
    for name_col, meta_col in [("n_map", "map"), ("n_type", "type"), ("n_weather", "weather")]:
        if meta_col in df.columns:
            bad = int((df[name_col] != df[meta_col].astype(str)).sum())
            print(f"  {name_col} != {meta_col}: {bad}")
    if "camera_position" in df.columns:
        same = int((df["n_number"] == df["camera_position"]).sum())
        print(
            f"  {{number}} == camera_position: {same} / {len(df)} rows "
            "(if ~all, {number} is the camera id, not an event id)"
        )
    return df


# --------------------------------------------------------------------------- #
# Step 2 - per-event table
# --------------------------------------------------------------------------- #
def step2_table(df, out_dir):
    section("STEP 2 - EVENT TABLE")
    g = df.groupby("event_key")["accident_time"]
    # std with ddof=0, so a single-video key has std 0 (not NaN).
    summary = pd.DataFrame(
        {
            "n_videos": g.size(),
            "t_min": g.min(),
            "t_max": g.max(),
            "t_std": g.std(ddof=0),
        }
    ).reset_index()
    out_csv = out_dir / "event_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(summary.sort_values("n_videos", ascending=False).head(15).to_string(index=False))
    print(f"(saved {len(summary)} rows to {out_csv})")

    n = summary["n_videos"]
    print(f"\nDistinct event_keys: {len(summary)}  (rows: {len(df)})")
    print(f"Videos per event_key: min {n.min()} / median {n.median():g} / max {n.max()}")
    multi = summary[n >= 2]
    print(
        f"event_keys with accident_time std > {STD_THRESHOLD} s: "
        f"{int((summary['t_std'] > STD_THRESHOLD).sum())} of {len(summary)} "
        f"(among keys with >=2 videos: "
        f"{int((multi['t_std'] > STD_THRESHOLD).sum())} of {len(multi)})"
    )

    per_mtw = df.groupby(["n_map", "n_type", "n_weather"])["n_number"].nunique()
    print("\nDistinct {number} values per (map, type, weather):")
    print(
        f"  groups: {len(per_mtw)}, min {per_mtw.min()} / median {per_mtw.median():g} "
        f"/ max {per_mtw.max()}"
    )
    print(per_mtw.rename("n_numbers").reset_index().to_string(index=False))

    per_mtn = df.groupby(["n_map", "n_type", "n_number"])["n_weather"].nunique()
    print("\nIn how many weathers does the same (map, type, {number}) appear:")
    print(f"  groups: {len(per_mtn)}")
    print(per_mtn.value_counts().sort_index().rename_axis("n_weathers").rename("n_groups").to_string())
    return summary, per_mtw, per_mtn


# --------------------------------------------------------------------------- #
# Step 3 - montages
# --------------------------------------------------------------------------- #
def read_frame_at(video_file, t, meta_fps):
    """Return (RGB frame, frame_index, fps_used) at time t, or (None, reason, None)."""
    cap = cv2.VideoCapture(str(video_file))
    try:
        if not cap.isOpened():
            return None, "cannot open", None
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps != fps:
            fps = meta_fps
        if not fps or fps <= 0:
            return None, "no fps", None
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = int(round(t * fps))
        if n_frames > 0:
            idx = min(max(idx, 0), n_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None, f"cannot read frame {idx}", None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), idx, fps
    finally:
        cap.release()


def meta_fps_of(row):
    try:
        if row["duration"] > 0:
            return float(row["no_frames"]) / float(row["duration"])
    except (KeyError, TypeError, ValueError):
        pass
    return None


def build_montage(rows, title, out_png):
    """rows: list of metadata rows (Series). Saves a 2x3 (or smaller) grid."""
    n = len(rows)
    cols = min(3, n)
    nrows = (n + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(6 * cols, 3.8 * nrows), squeeze=False)
    print(f"  files used for {out_png.name}:")
    for ax, (_, row) in zip(axes.flat, rows):
        name = Path(row["rgb_path"].replace("\\", "/")).name
        frame, idx, fps = read_frame_at(row["video_file"], row["accident_time"], meta_fps_of(row))
        if frame is None:
            ax.text(0.5, 0.5, f"unreadable:\n{idx}", ha="center", va="center")
            print(f"    {name}  t={row['accident_time']:.2f}s  -> {idx}")
        else:
            ax.imshow(frame)
            print(f"    {name}  t={row['accident_time']:.2f}s  frame={idx}  fps={fps:g}")
        ax.set_title(f"{name}\naccident_time={row['accident_time']:.2f}s", fontsize=9)
        ax.axis("off")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)
    print(f"  saved {out_png}")


def step3_montages(df, summary, per_mtw, per_mtn, out_dir):
    section("STEP 3 - MONTAGES (seed 42)")
    rng = random.Random(SEED)
    ok = df[df["exists"]]

    # A) one event_key with >= 6 videos -> 6 different cameras of the same event.
    print("\nA) same event_key, 6 cameras")
    cand = sorted(summary.loc[summary["n_videos"] >= N_TILES, "event_key"])
    if not cand:
        print(
            f"  cannot build: no event_key has >= {N_TILES} videos "
            f"(max is {summary['n_videos'].max()}). Each event_key is (nearly) one video."
        )
    else:
        key = rng.choice(cand)
        sub = ok[ok["event_key"] == key]
        if "camera_position" in sub.columns:
            sub = sub.drop_duplicates("camera_position")
        if len(sub) < N_TILES:
            print(f"  cannot build: {key} has only {len(sub)} readable distinct-camera videos")
        else:
            rows = list(sub.sample(n=N_TILES, random_state=SEED).iterrows())
            build_montage(rows, f"A) event_key {key}", out_dir / "montage_A_same_event_key.png")

    # B) one (map, type, number) in >= 3 weathers -> one video per weather.
    print("\nB) same (map, type, {number}) across weathers")
    cand = sorted(per_mtn[per_mtn >= 3].index.tolist())
    if not cand:
        print(f"  cannot build: no (map, type, number) appears in >= 3 weathers (max {per_mtn.max()})")
    else:
        mp, tp, num = rng.choice(cand)
        sub = ok[(ok["n_map"] == mp) & (ok["n_type"] == tp) & (ok["n_number"] == num)]
        # Prefer the camera that appears in the most weathers, so tiles share a camera.
        pref_cam = None
        if "camera_position" in sub.columns and len(sub):
            pref_cam = sub.groupby("camera_position")["n_weather"].nunique().idxmax()
        rows = []
        for w in sorted(sub["n_weather"].unique())[:N_TILES]:
            sw = sub[sub["n_weather"] == w]
            if pref_cam is not None and (sw["camera_position"] == pref_cam).any():
                sw = sw[sw["camera_position"] == pref_cam]
            rows.append(next(sw.sample(n=1, random_state=SEED).iterrows()))
        if len(rows) < 3:
            print(f"  cannot build: only {len(rows)} weathers have a readable video")
        else:
            build_montage(
                rows,
                f"B) {mp} {tp} number={num} across weathers",
                out_dir / "montage_B_same_number_across_weathers.png",
            )

    # C) one (map, type, weather) with >= 6 distinct numbers -> one video per number.
    print("\nC) same (map, type, weather), 6 different {number} values")
    cand = sorted(per_mtw[per_mtw >= N_TILES].index.tolist())
    if not cand:
        print(f"  cannot build: no (map, type, weather) has >= {N_TILES} distinct numbers")
    else:
        mp, tp, w = rng.choice(cand)
        sub = ok[(ok["n_map"] == mp) & (ok["n_type"] == tp) & (ok["n_weather"] == w)]
        nums = sorted(sub["n_number"].unique())
        if len(nums) < N_TILES:
            print(f"  cannot build: only {len(nums)} readable numbers in {mp} {tp} {w}")
        else:
            chosen = sorted(rng.sample(nums, N_TILES))
            rows = [
                next(sub[sub["n_number"] == k].sample(n=1, random_state=SEED).iterrows())
                for k in chosen
            ]
            build_montage(
                rows,
                f"C) {mp} {tp} {w}, different numbers",
                out_dir / "montage_C_different_numbers.png",
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
    videos_root = dataset_root / "synthetic_videos" / "videos"
    if not meta_csv.is_file():
        sys.exit(f"metadata-synthetic.csv not found in {dataset_root}")
    if not videos_root.is_dir():
        sys.exit(f"synthetic videos folder not found: {videos_root}")

    out_dir = args.out.resolve()
    if out_dir == dataset_root or dataset_root in out_dir.parents:
        sys.exit(f"refusing to write inside the dataset: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset root : {dataset_root}")
    print(f"Metadata     : {meta_csv}")
    print(f"Videos root  : {videos_root}")
    print(f"Output dir   : {out_dir}")

    df = pd.read_csv(meta_csv)
    df = step1_inspect(df, dataset_root, videos_root)
    summary, per_mtw, per_mtn = step2_table(df, out_dir)
    step3_montages(df, summary, per_mtw, per_mtn, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
