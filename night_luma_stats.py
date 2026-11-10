"""Night guard statistics for Ablation 1 (cctv augmentation): window mean luma by weather, before and after the
cctv profile, without and with the night guard. SYNTHETIC DATA ONLY (frame cache); no model, no GPU.

Sample: about N_PER_WEATHER clips per weather (synthetic_split.csv weather column), drawn with a fixed seed from the
clips with split_dev == "train" (the fold-0 training clips); every window of each sampled clip.
Luma: Y = 0.299 R + 0.587 G + 0.114 B, mean over the 16 frames and all pixels of a window (0-255).
Augmented: the real pipeline (WindowDataset.load, profile cctv, config cctv_aug), one generator per window seeded with
(seed, window number), no horizontal flip (it does not change the mean luma). The same generator seed is used with
and without the night guard, so both see the same draws.
dark_luma (proposal) = unaugmented night median of this sample (--dark-luma overrides it).

Usage: .venv-ml\\Scripts\\python.exe night_luma_stats.py [--config configs\\videomae_v2_C.json] [--per-weather 60]
       [--seed 0] [--workers 8] [--out runs\\aug_preview\\C_v2_cctv]
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
import train_classifier as tc  # noqa: E402  (data and augmentation code only; no model, no CUDA)

WEATHERS = ("clear", "sunset", "wet", "rain", "night")
PCTS = (1, 5, 25, 50)


def sample_windows(per_weather: int, seed: int, model_fps: int) -> pd.DataFrame:
    data, _, _ = tc.load_labels(model_fps)
    split = pd.read_csv(tc.SPLIT_CSV)
    pool = split[split["split_dev"] == "train"]
    rng = np.random.default_rng(seed)
    clips = []
    for w in WEATHERS:
        c = sorted(pool.loc[pool["weather"] == w, "rgb_path"])
        clips += [(p, w) for p in rng.choice(c, size=min(per_weather, len(c)), replace=False)]
    sel = pd.DataFrame(clips, columns=["rgb_path", "weather"])
    rows = data.merge(sel, on="rgb_path", how="inner").sort_values(["weather", "rgb_path", "window_idx"])
    if rows["rgb_path"].nunique() != len(sel):
        tc.fail("some sampled clips have no windows")
    return rows.reset_index(drop=True).assign(win_no=lambda d: np.arange(len(d)))


def _work(args: tuple) -> list[dict]:
    """One chunk of clips: unaugmented, cctv (no guard) and cctv (guard) luma of every window."""
    rows, cfg, seed, dark_luma = args
    c = cfg["cctv_aug"]
    ds = tc.WindowDataset(rows["stem"], rows["k0"], rows["label_B"], None, [0, 0, 0], [1, 1, 1],
                          cfg["frames"], cfg["model_fps"], cfg["lru_size"], "cctv", c)
    out = []
    for i, r in enumerate(rows.itertuples(index=False)):
        x, _ = ds.load(i, False, "none", None)
        rec = {"win_no": r.win_no, "luma_orig": tc.window_luma(x)}
        for tag, dl in (("old", None), ("new", dark_luma)):
            ds.dark_luma = dl
            xa, applied = ds.load(i, False, "cctv", np.random.default_rng([seed, int(r.win_no)]))
            col = [a for a in applied if a.startswith("colour")]
            rec[f"luma_{tag}"] = tc.window_luma(xa)
            rec[f"colour_{tag}"] = bool(col)
            rec[f"b_{tag}"] = float(col[0].split()[1][1:]) if col else np.nan  # 2 decimals, from the label
            rec[f"dark_{tag}"] = bool(col) and " dark" in col[0]
            rec[f"applied_{tag}"] = ", ".join(applied)
        # cctv_v2 (dark-window rules with the same dark_luma), same per-window seed
        ds.dark_luma = dark_luma
        xa, applied = ds.load(i, False, "cctv_v2", np.random.default_rng([seed, int(r.win_no)]))
        rec["luma_v2"] = tc.window_luma(xa)
        rec["dark_v2"] = applied[0].endswith(" dark")
        rec["n_degr_v2"] = len([a for a in applied[1:] if not a.startswith("hflip")])
        rec["applied_v2"] = ", ".join(applied)
        out.append(rec)
    return out


def pct_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby("weather")[col]
    t = pd.DataFrame({f"p{p}": g.quantile(p / 100) for p in PCTS})
    t.insert(0, "windows", g.size())
    t.insert(0, "clips", df.groupby("weather")["rgb_path"].nunique())
    t.loc["all"] = [df["rgb_path"].nunique(), len(df)] + [df[col].quantile(p / 100) for p in PCTS]
    return t.reindex(list(WEATHERS) + ["all"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "videomae_v2_C.json")
    ap.add_argument("--per-weather", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dark-luma", type=float, default=None, help="default: unaugmented night median")
    ap.add_argument("--out", type=Path, default=tc.AUG_PREVIEW_DIR / "C_v2_cctv")
    ap.add_argument("--compare-csv", type=Path, default=None,
                    help="night_luma_windows.csv of an earlier run: luma columns must be identical")
    args = ap.parse_args()
    if (args.out / "night_luma_windows.csv").exists():
        tc.fail(f"{args.out} already has night_luma_windows.csv; choose another --out")
    cfg, cfg_hash = tc.load_config(args.config.resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    print(f"config {args.config} (sha256 {cfg_hash[:12]}...); cctv_aug {json.dumps(cfg['cctv_aug'])}")

    rows = sample_windows(args.per_weather, args.seed, cfg["model_fps"])
    clip_list = rows.groupby("rgb_path").size()
    print(f"sample: {rows['rgb_path'].nunique()} clips (split_dev train, {args.per_weather} per weather, seed "
          f"{args.seed}), {len(rows)} windows")

    # unaugmented luma first (cheap): needed to propose dark_luma
    chunks = [g for _, g in rows.groupby("rgb_path", sort=True)]
    with ProcessPoolExecutor(args.workers) as ex:
        orig = pd.DataFrame([rec for part in ex.map(_unaug, [(g, cfg) for g in chunks]) for rec in part])
    rows = rows.merge(orig, on="win_no")
    t0 = pct_table(rows, "luma_orig")
    night_p1 = float(t0.loc["night", "p1"])
    dark_luma = round(float(t0.loc["night", "p50"]), 1) if args.dark_luma is None else args.dark_luma
    print("\nTable 1. Window mean luma (0-255), unaugmented")
    print(t0.round(2).to_string())

    # candidate thresholds: share of windows marked dark (unaugmented luma < threshold)
    night = rows["weather"] == "night"
    cand = []
    for name, v in (("night p25", t0.loc["night", "p25"]), ("night p50 (proposed)", t0.loc["night", "p50"]),
                    ("night p75", rows.loc[night, "luma_orig"].quantile(0.75))):
        v = round(float(v), 1)
        cand.append({"threshold": name, "dark_luma": v,
                     "night windows dark": f"{(rows.loc[night, 'luma_orig'] < v).mean():.3f}",
                     "non-night windows dark": f"{(rows.loc[~night, 'luma_orig'] < v).mean():.3f}",
                     **{w: f"{(rows.loc[rows['weather'] == w, 'luma_orig'] < v).mean():.3f}" for w in WEATHERS}})
    cand = pd.DataFrame(cand)
    print("\nTable 2. Share of windows marked dark (unaugmented luma < dark_luma), candidate thresholds")
    print(cand.to_string(index=False))
    print(f"\ndark_luma = {dark_luma} (PROVISIONAL: unaugmented night median of this sample; to be confirmed)")

    # augmented, without and with the guard (same per-window generator seeds)
    with ProcessPoolExecutor(args.workers) as ex:
        aug = pd.DataFrame([rec for part in ex.map(_work, [(g, cfg, args.seed, dark_luma) for g in chunks])
                            for rec in part]).drop(columns=["luma_orig"])
    rows = rows.merge(aug, on="win_no")
    t1 = pct_table(rows, "luma_old")
    t2 = pct_table(rows, "luma_new")
    t3 = pct_table(rows, "luma_v2")
    for t, tag in ((t1, "old"), (t2, "new"), (t3, "v2")):
        below = rows.groupby("weather")[f"luma_{tag}"].apply(lambda s: int((s < night_p1).sum()))
        t["below_night_p1"] = list(below.reindex(WEATHERS)) + [int((rows[f"luma_{tag}"] < night_p1).sum())]
        t["share_below"] = t["below_night_p1"] / t["windows"]
    t0["below_night_p1"] = list(rows.groupby("weather")["luma_orig"].apply(
        lambda s: int((s < night_p1).sum())).reindex(WEATHERS)) + [int((rows["luma_orig"] < night_p1).sum())]
    print(f"\nunaugmented night p1 = {night_p1:.2f}")
    print("\nTable 3. Window mean luma after the cctv profile, old rule (brightness x[0.5, 1.3] for every window)")
    print(t1.round(3).to_string())
    print(f"\nTable 4. Window mean luma after the cctv profile, new rule (night guard, dark_luma {dark_luma}: windows "
          f"darker than that draw brightness x[1.0, 1.3])")
    print(t2.round(3).to_string())

    # guard firing (the rule's own luma: after frame-rate drop, blur and downscale, before the colour shift)
    g = rows.groupby("weather")
    fire = pd.DataFrame({"windows": g.size(), "colour_shifted": g["colour_new"].sum(), "guard_fired": g["dark_new"].sum()})
    fire.loc["all"] = fire.sum()
    fire = fire.reindex(list(WEATHERS) + ["all"])
    fire["fired_share_of_colour_shifted"] = fire["guard_fired"] / fire["colour_shifted"]
    fire["fired_share_of_windows"] = fire["guard_fired"] / fire["windows"]
    print("\nTable 5. Night guard firing (new rule), by weather")
    print(fire.round(3).to_string())
    same = (rows["applied_old"].str.replace(r"colour b\S+", "colour", regex=True)
            == rows["applied_new"].str.replace(r"colour b\S+", "colour", regex=True).str.replace(" dark", ""))
    nd = ~rows["dark_new"]
    print(f"\ncheck: same augmentation draws except brightness in {int(same.sum())}/{len(rows)} windows; "
          f"windows where the guard did not fire have identical output luma: "
          f"{int(np.isclose(rows.loc[nd, 'luma_old'], rows.loc[nd, 'luma_new']).sum())}/{int(nd.sum())}")

    print(f"\nTable 6. Window mean luma after cctv_v2 (dark windows < {dark_luma}: at most 1 degradation, brightness "
          "x[1.0, 1.3], contrast x[1.0, 1.2])")
    print(t3.round(3).to_string())
    nd = rows.groupby("weather")["n_degr_v2"].value_counts().unstack(fill_value=0).reindex(WEATHERS)
    nd.loc["all"] = nd.sum()
    nd["dark windows"] = list(rows.groupby("weather")["dark_v2"].sum().reindex(WEATHERS)) + [int(rows["dark_v2"].sum())]
    print("\nTable 7. cctv_v2 degradations per window (excluding hflip), by weather")
    print(nd.to_string())
    side = pd.DataFrame({f"{lab} {p}": t[p] for lab, t in (("unaug", t0), ("cctv", t1), ("cctv+guard", t2),
                                                           ("cctv_v2", t3)) for p in ("p1", "p5", "p25", "p50")})
    for lab, t in (("unaug", t0), ("cctv", t1), ("cctv+guard", t2), ("cctv_v2", t3)):
        side[f"below p1: {lab}"] = t["below_night_p1"]
    print(f"\nTable 8. Side by side: luma percentiles and windows below the unaugmented night p1 ({night_p1:.2f})")
    print(side.round(2).to_string())
    if args.compare_csv is not None and args.compare_csv.is_file():
        prev = pd.read_csv(args.compare_csv)
        m = rows.merge(prev, on="win_no", suffixes=("", "_prev"))
        same = {c: bool(np.array_equal(m[c].to_numpy(), m[f"{c}_prev"].to_numpy()))
                for c in ("luma_orig", "luma_old", "luma_new") if f"{c}_prev" in m}
        print(f"\nregression check vs {args.compare_csv}: {len(m)}/{len(prev)} windows matched; identical columns {same}")

    rows.drop(columns=["applied_old", "applied_new", "applied_v2"]).to_csv(args.out / "night_luma_windows.csv",
                                                                            index=False)
    with open(args.out / "night_luma_tables.txt", "w", encoding="utf-8") as fh:
        fh.write(f"config {args.config} (sha256 {cfg_hash}); sample {rows['rgb_path'].nunique()} clips "
                 f"(split_dev train, {args.per_weather} per weather, seed {args.seed}), {len(rows)} windows\n")
        for title, t in (("Table 1 unaugmented", t0), ("Table 2 candidate thresholds", cand),
                         ("Table 3 cctv old rule", t1), (f"Table 4 cctv new rule (dark_luma {dark_luma})", t2),
                         ("Table 5 guard firing", fire), ("Table 6 cctv_v2", t3), ("Table 7 cctv_v2 degradations", nd),
                         ("Table 8 side by side", side)):
            fh.write(f"\n{title}\n{t.round(3).to_string()}\n")
        fh.write(f"\nunaugmented night p1 {night_p1:.4f}; dark_luma {dark_luma} PROVISIONAL\n")
    json.dump({"dark_luma_provisional": dark_luma, "night_p1": night_p1, "night_p50": float(t0.loc["night", "p50"]),
               "n_clips": int(rows["rgb_path"].nunique()), "n_windows": len(rows), "seed": args.seed,
               "clips": sorted(clip_list.index)},
              open(args.out / "night_luma_summary.json", "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {args.out / 'night_luma_windows.csv'}, night_luma_tables.txt, night_luma_summary.json")


def _unaug(args: tuple) -> list[dict]:
    rows, cfg = args
    ds = tc.WindowDataset(rows["stem"], rows["k0"], rows["label_B"], None, [0, 0, 0], [1, 1, 1],
                          cfg["frames"], cfg["model_fps"], cfg["lru_size"])
    return [{"win_no": r.win_no, "luma_orig": tc.window_luma(ds.load(i, False, "none", None)[0])}
            for i, r in enumerate(rows.itertuples(index=False))]


if __name__ == "__main__":
    main()
