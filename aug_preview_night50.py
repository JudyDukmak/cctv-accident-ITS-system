"""Preview grids for cctv_v2 with the night protection at the config's dark_luma (night50). SYNTHETIC ONLY; no model,
no GPU, no training. Night/day clips from split_dev == "train" (fold-0 training clips), window centred on t*, frames
0/5/10/15 at full resolution, 4 windows per PNG (helpers from aug_preview_v2.py).

  a_night_real_NN.png : 12 night windows with their REAL random draws, half with luma < dark_luma (protected), half >=.
  b_protected_worst_01.png : Town03_head-on_night_45, Town05_rear-end_night_233 and 2 more night windows with luma 30-50:
                        worst protected case = 1 degradation, downscale 0.7 (the protected minimum).
  c_welllit_worst_01.png : 4 night windows with luma 50-70 (not protected): downscale 0.5 + brightness 0.7 (other colour
                        parameters from real draws).
  d_day_real_01.png   : 4 day windows with their real random draws (rule unchanged for day).

Usage: .venv-ml\\Scripts\\python.exe aug_preview_night50.py --config configs\\videomae_v2_C_cctv.json [--seed 0]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
import train_classifier as tc  # noqa: E402
from aug_preview_v2 import band, centre_window, save_pages, strip  # noqa: E402

NAMED_B = ("Town03_head-on_night_45", "Town05_rear-end_night_233")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "videomae_v2_C_cctv.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=tc.AUG_PREVIEW_DIR / "C_v2_cctv_v2_night50")
    args = ap.parse_args()
    cfg, cfg_hash = tc.load_config(args.config.resolve())
    dl = tc.dark_luma_of(cfg)
    if cfg["augmentation"] != "cctv_v2" or dl is None:
        tc.fail("needs a cctv_v2 config with night_guard")
    v = tc.CCTV_V2
    data, manifest, _ = tc.load_labels(cfg["model_fps"])
    split = pd.read_csv(tc.SPLIT_CSV)
    pool = split[split["split_dev"] == "train"]
    weather = pool.set_index("rgb_path")["weather"]
    rows = data[data["rgb_path"].isin(pool["rgb_path"])].reset_index(drop=True)
    ds = tc.WindowDataset(rows["stem"], rows["k0"], rows["label_B"], None, [0, 0, 0], [1, 1, 1], cfg["frames"],
                          cfg["model_fps"], cfg["lru_size"], "cctv_v2", None, dl)
    idx = {p: centre_window(rows, p) for p in pool["rgb_path"]}
    stem_to_path = {tc.stem_of(p): p for p in pool["rgb_path"]}
    night = sorted(pool.loc[pool["weather"] == "night", "rgb_path"])
    luma = {p: tc.window_luma(ds.load(idx[p], False, "none", None)[0]) for p in night}
    lum = pd.Series(luma)
    print(f"night clips (split_dev train): {len(night)}; t*-centred window luma: < {dl:g}: {int((lum < dl).sum())}, "
          f"30-{dl:g}: {int(((lum >= 30) & (lum < dl)).sum())}, {dl:g}-70: {int(((lum >= dl) & (lum < 70)).sum())}, "
          f">= 70: {int((lum >= 70).sum())}")
    rng = np.random.default_rng(args.seed)

    def take(cands: list[str], n: int, used: set) -> list[str]:
        c = sorted(set(cands) - used)
        if len(c) < n:
            print(f"WARNING: only {len(c)} candidates for {n} windows; using all")
        return list(rng.choice(c, size=min(n, len(c)), replace=False))

    used: set = set()
    named = [stem_to_path[s] for s in NAMED_B]
    used |= set(named)
    b = named + take(list(lum[(lum >= 30) & (lum < dl)].index), 2, used)
    used |= set(b)
    c = take(list(lum[(lum >= dl) & (lum < 70)].index), 4, used)
    used |= set(c)
    a = take(list(lum[lum < dl].index), 6, used) + take(list(lum[lum >= dl].index), 6, used)
    d = take(list(pool.loc[pool["weather"] != "night", "rgb_path"]), 4, used)
    tc.check_cache(rows.iloc[[idx[p] for p in a + b + c + d]], rows.iloc[:0], manifest, partial=True)
    args.out.mkdir(parents=True, exist_ok=True)
    head = (f"cctv_v2 night protection; config {args.config.name} (sha256 {cfg_hash[:12]}); dark_luma {dl:g}; seed "
            f"{args.seed}; frames {tc.PREVIEW_FRAMES}")
    txt, written = [head], []

    def window_luma(p: str) -> float:
        return luma.get(p) or tc.window_luma(ds.load(idx[p], False, "none", None)[0])

    def block(i: int, p: str, aug: np.ndarray, applied: list[str], tag: str) -> list[np.ndarray]:
        r = rows.iloc[idx[p]]
        lp = window_luma(p)
        name = (f"{i}: {r['stem']} ({weather[p]}) t_start {r['t_start']:.1f} s, t* {r['t_star']:.2f} s, window luma "
                f"{lp:.1f} ({'PROTECTED' if lp < dl else 'not protected'})")
        par = "applied: " + ", ".join(applied)
        txt.append(f"{name}; {par}")
        orig, _ = ds.load(idx[p], False, "none", None)
        return [strip(orig, f"{i} original"), strip(aug, f"{i} {tag}"), band([name, par])]

    def real(i: int, p: str, tag: str) -> list[np.ndarray]:
        flip = bool(np.random.default_rng([args.seed, 3, i]).random() < cfg["hflip_p"])
        aug, applied = ds.load(idx[p], flip, "cctv_v2", np.random.default_rng([args.seed, 1, i]))
        return block(i, p, aug, applied, tag)

    def forced(i: int, p: str, ops: list[str], down: float, b_low: float | None, tag: str) -> list[np.ndarray]:
        def make(_rng, dark):
            g = np.random.default_rng([args.seed, 4, i])
            plan = {"drawn": ops, "kept": ops, "dark": dark, "downscale": down}
            if "colour" in ops:
                ct = float(g.uniform(*v["contrast_dark" if dark else "contrast"]))
                plan["colour"] = (b_low, ct, float(g.uniform(*v["saturation"])), float(g.uniform(-v["hue"], v["hue"])),
                                  bool(g.random() < v["grayscale_p"]))
            return plan
        aug, applied = ds.load_v2(idx[p], False, None, make)
        return block(i, p, aug, applied, tag)

    txt.append("a) night windows, real draws")
    written += save_pages([real(i, p, "cctv_v2") for i, p in enumerate(a)],
                          [head, f"a) night windows with their real draws (0-5: luma < {dl:g}, protected; 6-11: >= "
                                 f"{dl:g})"], "a_night_real", args.out)
    txt.append(f"b) protected worst case: downscale {v['downscale_dark'][0]} only")
    written += save_pages([forced(i, p, ["downscale"], v["downscale_dark"][0], None, "protected worst")
                           for i, p in enumerate(b)],
                          [head, f"b) protected night (luma < {dl:g}) worst case: 1 degradation, downscale "
                                 f"{v['downscale_dark'][0]}"], "b_protected_worst", args.out)
    txt.append(f"c) well-lit night worst case: downscale {v['downscale'][0]} + brightness {v['brightness'][0]}")
    written += save_pages([forced(i, p, ["downscale", "colour"], v["downscale"][0], v["brightness"][0],
                                  "well-lit worst") for i, p in enumerate(c)],
                          [head, f"c) well-lit night (luma >= {dl:g}) worst case: downscale {v['downscale'][0]} + "
                                 f"brightness {v['brightness'][0]}"], "c_welllit_worst", args.out)
    txt.append("d) day windows, real draws")
    written += save_pages([real(100 + i, p, "cctv_v2") for i, p in enumerate(d)],
                          [head, "d) day windows with their real draws (rule unchanged)"], "d_day_real", args.out)
    (args.out / "previews.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")
    print("\n".join(str(p) for p in written))
    print("\n".join(txt))


if __name__ == "__main__":
    main()
