"""Freeze the real-clip TEST subset (Week 2, Proposal A section 6 and 8).

Rule (written BEFORE drawing, never changed afterwards):
  1. Keep every clip of metadata-real.csv (optionally only those whose video file exists).
  2. From each source video pick ONE clip at random (seeded)  -> one clip per source.
  3. From those, draw N clips at random (seeded), 300 <= N <= 500.
  4. Save the rows to real_test_set.csv + a manifest, mark them read-only.

Run it ONCE. Do not try other seeds to get a "nicer" subset: that would be tuning on the test data.

Usage:
  python make_test_set.py metadata-real.csv --root "path\\to\\folder\\with\\videos" --n 400 --seed 42
"""
import argparse, datetime, hashlib, json, os, platform, stat, sys
import numpy as np
import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="path to metadata-real.csv (a FILE)")
    ap.add_argument("--root", default=None, help="folder containing the .mp4 files (searched recursively)")
    ap.add_argument("--n", type=int, default=400, help="subset size, 300..500")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="real_test_set.csv")
    ap.add_argument("--force", action="store_true", help="overwrite an existing frozen file (do not use)")
    a = ap.parse_args()

    if not 300 <= a.n <= 500:
        sys.exit("N must be between 300 and 500 (Proposal A section 6).")
    if os.path.exists(a.out) and not a.force:
        sys.exit(f"{a.out} already exists. The test set is frozen; refusing to overwrite.")
    if not os.path.isfile(a.csv):
        sys.exit(f"{a.csv} is not a file. Pass the path of metadata-real.csv.")

    df = pd.read_csv(a.csv)
    for c in ("path", "type"):
        if c not in df.columns:
            sys.exit(f"column '{c}' not found in {a.csv}")
    total_rows = len(df)

    # source video = file name without the trailing _NN.mp4 clip number
    norm = df.path.str.replace("\\", "/", regex=False)
    df["source_video"] = norm.str.replace(r"_\d+\.mp4$", "", regex=True).str.split("/").str[-1]
    df = df.sort_values("path").reset_index(drop=True)   # order of the file must not matter

    if a.root:
        names = {f for _, _, files in os.walk(a.root) for f in files}
        keep = df.path.map(lambda p: os.path.basename(p.replace("\\", "/")) in names)
        print(f"videos found on disk for {int(keep.sum())} of {total_rows} rows")
        df = df[keep].reset_index(drop=True)
    else:
        print("WARNING: --root not given, video files were not checked")

    rs = np.random.RandomState(a.seed)          # legacy RandomState: its stream is frozen across numpy versions
    groups = df.groupby("source_video").indices  # {source: array of row positions}
    picked = [idx[rs.randint(len(idx))] for _, idx in sorted(groups.items())]
    picked = np.array(picked)
    if len(picked) < a.n:
        sys.exit(f"only {len(picked)} distinct sources available, cannot draw {a.n}")
    chosen = picked[rs.permutation(len(picked))[: a.n]]
    sel = df.iloc[chosen].sort_values("path").reset_index(drop=True)

    assert sel.source_video.is_unique and len(sel) == a.n and sel.path.is_unique

    # how representative is the subset? (informational)
    print(f"\nsources available: {len(picked)} | clips drawn: {len(sel)} | seed: {a.seed}")
    print(f"worst-case 95% margin of error for an accuracy measured on {a.n} clips: "
          f"+/-{1.96 * (0.25 / a.n) ** 0.5:.3f}")
    for col in ("type", "quality", "day_time", "weather", "scene_layout"):
        if col in df.columns:
            full = df[col].value_counts(normalize=True) * 100
            sub = sel[col].value_counts(normalize=True) * 100
            t = pd.concat([full.rename("full %"), sub.rename("subset %")], axis=1).fillna(0).round(1)
            t["diff"] = (t["subset %"] - t["full %"]).round(1)
            print(f"\n{col}\n{t.to_string()}")

    if a.force and os.path.exists(a.out):
        os.chmod(a.out, stat.S_IWRITE)
    sel.to_csv(a.out, index=False)
    manifest = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "rule": "one random clip per source video, then N random sources; seeded numpy RandomState",
        "seed": a.seed, "n": a.n,
        "source_csv": os.path.basename(a.csv), "source_csv_sha256": sha256(a.csv),
        "source_csv_rows": total_rows, "rows_after_file_check": len(df),
        "distinct_sources_available": int(len(picked)),
        "output": os.path.basename(a.out), "output_sha256": sha256(a.out),
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
    }
    mpath = os.path.splitext(a.out)[0] + "_manifest.json"
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    for p in (a.out, mpath):
        os.chmod(p, stat.S_IREAD)               # read-only
    print(f"\nsaved {a.out} and {mpath} (read-only). sha256 of the test set: {manifest['output_sha256']}")


if __name__ == "__main__":
    main()