"""Preview grids for Ablation 1 (cctv augmentation): windows centred on t*, frames 0/5/10/15. SYNTHETIC ONLY; no model,
no GPU, no training. Clips are drawn with a fixed seed from split_dev == "train" (the fold-0 training clips); the window
of a clip is the one whose t_centre is closest to t*.

  a) cctv_random_n12.png: N random windows, original vs the config's cctv profile (night guard as configured), random
     horizontal flip p hflip_p; the applied parameters are printed under each window.
  b) night_lowest_brightness_n12.png: N night windows, original / old rule (brightness from [0.5, 1.3]) / new rule (night
     guard: windows darker than dark_luma draw from [1.0, 1.3]); colour shift forced on and the brightness draw set to
     the lowest value of its range (u = 0); every other draw comes from the same seeded generator in both rows, so the
     two augmented rows differ only in the brightness factor. No horizontal flip.

Usage: .venv-ml\\Scripts\\python.exe aug_preview_grids.py --config configs\\videomae_v2_C_cctv.json [--n 12] [--seed 0]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
import train_classifier as tc  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX


class LowestBrightness:
    """Generator wrapper for WindowDataset.load(..., "cctv"): random() call number `colour_call` (the colour gate) returns
    0.0 (on) and the brightness draw returns the low end of its range; everything else passes through."""

    def __init__(self, g: np.random.Generator, colour_call: int, brightness: list[float]) -> None:
        self.g, self.colour_call, self.n = g, colour_call, 0
        self.lo_full, self.hi = brightness

    def random(self):
        self.n += 1
        v = self.g.random()  # consumed either way, so both rules see the same later draws
        return 0.0 if self.n == self.colour_call else v

    def uniform(self, lo, hi):
        v = self.g.uniform(lo, hi)
        return float(lo) if hi == self.hi and lo in (tc.NIGHT_GUARD_MIN_BRIGHTNESS, self.lo_full) else v

    def integers(self, lo, hi):
        return self.g.integers(lo, hi)

    def choice(self, a):
        return self.g.choice(a)


def strip(x: np.ndarray, label: str) -> np.ndarray:
    s = np.concatenate([x[k] for k in tc.PREVIEW_FRAMES], axis=1).copy()
    cv2.putText(s, label, (6, 18), FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(s, label, (6, 18), FONT, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    return s


def text_band(lines: list[str], width: int) -> np.ndarray:
    band = np.full((10 + 18 * len(lines), width, 3), 255, np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(band, line, (6, 20 + 18 * i), FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return band


def centre_windows(train_rows: pd.DataFrame, clips: list[str]) -> list[int]:
    out = []
    for p in clips:
        g = train_rows[train_rows["rgb_path"] == p]
        out.append(int((g["t_centre"] - g["t_star"]).abs().idxmin()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "videomae_v2_C_cctv.json")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=tc.AUG_PREVIEW_DIR / "C_v2_cctv")
    args = ap.parse_args()
    cfg, cfg_hash = tc.load_config(args.config.resolve())
    if cfg["augmentation"] != "cctv" or tc.dark_luma_of(cfg) is None:
        tc.fail("needs a cctv config with night_guard")
    c, dark_luma = cfg["cctv_aug"], tc.dark_luma_of(cfg)
    data, manifest, _ = tc.load_labels(cfg["model_fps"])
    split = pd.read_csv(tc.SPLIT_CSV)
    pool = split[split["split_dev"] == "train"]
    weather = pool.set_index("rgb_path")["weather"]
    rows = data[data["rgb_path"].isin(pool["rgb_path"])].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    clips_a = list(rng.choice(sorted(pool["rgb_path"]), size=args.n, replace=False))
    clips_b = list(rng.choice(sorted(pool.loc[pool["weather"] == "night", "rgb_path"]), size=args.n, replace=False))
    idx_a, idx_b = centre_windows(rows, clips_a), centre_windows(rows, clips_b)
    tc.check_cache(rows.iloc[idx_a + idx_b], rows.iloc[:0], manifest, partial=True)
    ds = tc.WindowDataset(rows["stem"], rows["k0"], rows["label_B"], None, [0, 0, 0], [1, 1, 1], cfg["frames"],
                          cfg["model_fps"], cfg["lru_size"], "cctv", c, dark_luma)
    args.out.mkdir(parents=True, exist_ok=True)
    W = 224 * len(tc.PREVIEW_FRAMES)
    head = (f"config {args.config.name} (sha256 {cfg_hash[:12]}); seed {args.seed}; frames {tc.PREVIEW_FRAMES}; "
            f"night_guard dark_luma {dark_luma:g} PROVISIONAL")

    # a) random windows
    tiles, txt = [text_band([head, "a) original vs cctv-augmented; windows centred on t*"], W)], [head]
    for i, r in enumerate(idx_a):
        row = rows.iloc[r]
        g = np.random.default_rng([args.seed, 1, i])
        flip = bool(g.random() < cfg["hflip_p"])
        orig, _ = ds.load(r, False, "none", None)
        aug, applied = ds.load(r, flip, "cctv", g)
        name = (f"{i}: {row['stem']} ({weather[row['rgb_path']]}) t_start {row['t_start']:.1f} s, t* {row['t_star']:.2f} s"
                f", luma {tc.window_luma(orig):.1f}")
        par = "applied: " + (", ".join(applied) if applied else "none")
        tiles += [strip(orig, f"{i} original"), strip(aug, f"{i} cctv"), text_band([name, par], W)]
        txt.append(f"{name}; {par}")
    out_a = args.out / f"cctv_random_n{args.n}.png"
    cv2.imwrite(str(out_a), cv2.cvtColor(np.concatenate(tiles), cv2.COLOR_RGB2BGR))
    out_a.with_suffix(".txt").write_text("\n".join(txt) + "\n", encoding="utf-8")

    # b) night windows, lowest brightness draw, old vs new rule
    lo_full, hi = c["brightness"]
    tiles = [text_band([head, f"b) night windows: original / old rule b from [{lo_full}, {hi}] / new rule (dark: b from "
                              f"[{tc.NIGHT_GUARD_MIN_BRIGHTNESS:g}, {hi}]); colour forced on, lowest brightness draw"], W)]
    txt = [head]
    for i, r in enumerate(idx_b):
        row = rows.iloc[r]
        orig, _ = ds.load(r, False, "none", None)
        outs = {}
        for tag, dl in (("old", None), ("new", dark_luma)):
            ds.dark_luma = dl
            # random() calls in load + cctv_after_load: 1 fps gate, 2 blur gate, 3 downscale gate, 4 colour gate
            w = LowestBrightness(np.random.default_rng([args.seed, 2, i]), 4, c["brightness"])
            outs[tag] = ds.load(r, False, "cctv", w)
        ds.dark_luma = dark_luma
        lum = tc.window_luma(orig)
        name = (f"{i}: {row['stem']} t* {row['t_star']:.2f} s; luma original {lum:.1f} ("
                f"{'dark' if lum < dark_luma else 'not dark'} vs {dark_luma:g}), old {tc.window_luma(outs['old'][0]):.1f}"
                f", new {tc.window_luma(outs['new'][0]):.1f}")
        tiles += [strip(orig, f"{i} original"), strip(outs["old"][0], f"{i} old rule"),
                  strip(outs["new"][0], f"{i} new rule (guard)"),
                  text_band([name, "old: " + ", ".join(outs["old"][1]), "new: " + ", ".join(outs["new"][1])], W)]
        txt.append(f"{name}; old: {', '.join(outs['old'][1])}; new: {', '.join(outs['new'][1])}")
    out_b = args.out / f"night_lowest_brightness_n{args.n}.png"
    cv2.imwrite(str(out_b), cv2.cvtColor(np.concatenate(tiles), cv2.COLOR_RGB2BGR))
    out_b.with_suffix(".txt").write_text("\n".join(txt) + "\n", encoding="utf-8")
    print(f"wrote {out_a}\nwrote {out_b}")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
