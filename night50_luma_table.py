"""Luma table and degradation shares for cctv_v2 with the config's night_guard dark_luma, next to the earlier tables.
SYNTHETIC ONLY; no model, no GPU. Same 300-clip sample and per-window generator seeds (seed, window number) as
night_luma_stats.py, whose night_luma_windows.csv (--previous-csv) supplies the earlier columns: unaugmented, cctv,
cctv + night guard (17.9) and cctv_v2 with dark_luma 17.9.

Usage: .venv-ml\\Scripts\\python.exe night50_luma_table.py --config configs\\videomae_v2_C_cctv.json
       --previous-csv runs\\aug_preview\\C_v2_cctv_v2\\night_luma_windows.csv --out runs\\aug_preview\\C_v2_cctv_v2_night50
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
import train_classifier as tc  # noqa: E402
from night_luma_stats import WEATHERS, pct_table, sample_windows  # noqa: E402


def _work(args: tuple) -> list[dict]:
    rows, cfg, seed, dark_luma = args
    ds = tc.WindowDataset(rows["stem"], rows["k0"], rows["label_B"], None, [0, 0, 0], [1, 1, 1], cfg["frames"],
                          cfg["model_fps"], cfg["lru_size"], "cctv_v2", None, dark_luma)
    out = []
    for i, r in enumerate(rows.itertuples(index=False)):
        x, applied = ds.load(i, False, "cctv_v2", np.random.default_rng([seed, int(r.win_no)]))
        out.append({"win_no": r.win_no, "luma_orig_now": tc.window_luma(ds.load(i, False, "none", None)[0]),
                    "luma_v2n": tc.window_luma(x), "prot_v2n": applied[0].endswith(" dark"),
                    "n_degr_v2n": len([a for a in applied[1:] if not a.startswith("hflip")])})
    return out


def degr_shares(df: pd.DataFrame, col: str) -> pd.DataFrame:
    grp = np.where(df["weather"] == "night", "night", "day")
    t = pd.crosstab(grp, df[col], normalize="index").round(3)
    t["windows"] = pd.Series(grp).value_counts()
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "videomae_v2_C_cctv.json")
    ap.add_argument("--previous-csv", type=Path, required=True)
    ap.add_argument("--per-weather", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg, cfg_hash = tc.load_config(args.config.resolve())
    dl = tc.dark_luma_of(cfg)
    if cfg["augmentation"] != "cctv_v2" or dl is None:
        tc.fail("needs a cctv_v2 config with night_guard")
    out_csv = args.out / "night50_luma_windows.csv"
    if out_csv.exists():
        tc.fail(f"{out_csv} exists; not overwritten")
    pd.set_option("display.width", 250)
    rows = sample_windows(args.per_weather, args.seed, cfg["model_fps"])
    prev = pd.read_csv(args.previous_csv)
    chunks = [g for _, g in rows.groupby("rgb_path", sort=True)]
    with ProcessPoolExecutor(args.workers) as ex:
        new = pd.DataFrame([r for part in ex.map(_work, [(g, cfg, args.seed, dl) for g in chunks]) for r in part])
    d = prev.merge(new, on="win_no", how="inner")
    print(f"config {args.config.name} (sha256 {cfg_hash[:12]}), dark_luma {dl:g}; {d['rgb_path'].nunique()} clips, "
          f"{len(d)} windows; previous {args.previous_csv}")
    print(f"check: unaugmented luma identical to the previous run: {np.array_equal(d['luma_orig'], d['luma_orig_now'])}; "
          f"protected == (unaugmented luma < {dl:g}): {bool((d['prot_v2n'] == (d['luma_orig'] < dl)).all())}")
    keep = d["luma_orig"] >= dl
    print(f"check: windows not protected under {dl:g} (and not dark under 17.9) have the same cctv_v2 output luma as "
          f"before: {int(np.isclose(d.loc[keep, 'luma_v2'], d.loc[keep, 'luma_v2n'], rtol=0, atol=1e-9).sum())}/"
          f"{int(keep.sum())}")
    night_p1 = float(d.loc[d["weather"] == "night", "luma_orig"].quantile(0.01))
    cols = (("unaug", "luma_orig"), ("cctv", "luma_old"), ("cctv+guard17.9", "luma_new"),
            ("cctv_v2 dl17.9", "luma_v2"), (f"cctv_v2 dl{dl:g}", "luma_v2n"))
    tabs = {lab: pct_table(d, c) for lab, c in cols}
    side = pd.DataFrame({f"{lab} {p}": t[p] for lab, t in tabs.items() for p in ("p1", "p5", "p25", "p50")})
    for lab, c in cols:
        side[f"<p1 {lab}"] = list(d.groupby("weather")[c].apply(lambda s: int((s < night_p1).sum())).reindex(WEATHERS)) \
            + [int((d[c] < night_p1).sum())]
    print(f"\nLuma percentiles by weather and windows below the unaugmented night p1 ({night_p1:.2f})")
    print(side.round(2).to_string())
    print(f"\nNew rule alone (cctv_v2, dark_luma {dl:g})")
    t = tabs[f"cctv_v2 dl{dl:g}"].copy()
    t["below_night_p1"] = side[f"<p1 cctv_v2 dl{dl:g}"]
    print(t.round(2).to_string())
    print(f"\nprotected windows (luma < {dl:g}) by weather: " + ", ".join(
        f"{w} {int(d.loc[d['weather'] == w, 'prot_v2n'].sum())}/{int((d['weather'] == w).sum())}" for w in WEATHERS))
    print("\nShare of windows with 0 / 1 / 2 degradations (excluding hflip), night vs day")
    print(f"cctv_v2, dark_luma 17.9 (previous):\n{degr_shares(d, 'n_degr_v2').to_string()}")
    print(f"cctv_v2, dark_luma {dl:g} (new):\n{degr_shares(d, 'n_degr_v2n').to_string()}")
    args.out.mkdir(parents=True, exist_ok=True)
    d[["win_no", "rgb_path", "weather", "window_idx", "luma_orig", "luma_v2", "luma_v2n", "prot_v2n", "n_degr_v2",
       "n_degr_v2n"]].to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
