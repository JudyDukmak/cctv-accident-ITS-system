"""Preview grids for the cctv_v2 profile (Ablation 1). SYNTHETIC ONLY; no model, no GPU, no training.
Clips are drawn with a fixed seed from split_dev == "train" (the fold-0 training clips); each window is the one whose
t_centre is closest to t*. Frames 0/5/10/15; full resolution (224 px per frame); one PNG per 4 windows.

  a_real_NN.png   : 8 night + 16 other-weather windows (4 per weather) with their REAL random draws (cctv_v2 plan from a
                    seeded generator per window, flip p hflip_p from a separate generator): original vs augmented.
  b_worst_NN.png  : worst case that the rules allow with 2 degradations: downscale 0.5 + colour shift at the lowest
                    brightness allowed (0.7); other colour parameters from real draws. 6 night windows that are NOT dark
                    (dark windows cannot get 2 degradations) and 6 day windows.
  b_dark_01.png   : 4 dark night windows with their worst allowed case: 1 degradation, downscale 0.5 (brightness of a
                    dark window can only rise, >= 1.0).

Usage: .venv-ml\\Scripts\\python.exe aug_preview_v2.py --config configs\\videomae_v2_C_cctv.json [--seed 0]
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
PER_PNG = 4
W = 224 * len(tc.PREVIEW_FRAMES)


def strip(x: np.ndarray, label: str) -> np.ndarray:
    s = np.concatenate([x[k] for k in tc.PREVIEW_FRAMES], axis=1).copy()
    for i, k in enumerate(tc.PREVIEW_FRAMES):
        cv2.line(s, (224 * i, 0), (224 * i, 223), (255, 255, 255), 1)
        cv2.putText(s, f"f{k}", (224 * i + 190, 18), FONT, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(s, f"f{k}", (224 * i + 190, 18), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(s, label, (6, 18), FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(s, label, (6, 18), FONT, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    return s


def band(lines: list[str]) -> np.ndarray:
    b = np.full((12 + 18 * len(lines), W, 3), 255, np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(b, line, (6, 20 + 18 * i), FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return b


def centre_window(rows: pd.DataFrame, path: str) -> int:
    g = rows[rows["rgb_path"] == path]
    return int((g["t_centre"] - g["t_star"]).abs().idxmin())


def save_pages(blocks: list[list[np.ndarray]], header: list[str], stem: str, out: Path) -> list[Path]:
    paths = []
    for p in range(0, len(blocks), PER_PNG):
        tiles = [band(header + [f"windows {p}-{min(p + PER_PNG, len(blocks)) - 1}"])]
        for b in blocks[p:p + PER_PNG]:
            tiles += b + [np.full((8, W, 3), 128, np.uint8)]
        path = out / f"{stem}_{p // PER_PNG + 1:02d}.png"
        cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(tiles), cv2.COLOR_RGB2BGR))
        paths.append(path)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "videomae_v2_C_cctv.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=tc.AUG_PREVIEW_DIR / "C_v2_cctv_v2")
    args = ap.parse_args()
    cfg, cfg_hash = tc.load_config(args.config.resolve())
    if cfg["augmentation"] != "cctv_v2":
        tc.fail("needs a cctv_v2 config")
    dl = tc.dark_luma_of(cfg)
    v = tc.CCTV_V2
    data, manifest, _ = tc.load_labels(cfg["model_fps"])
    split = pd.read_csv(tc.SPLIT_CSV)
    pool = split[split["split_dev"] == "train"]
    weather = pool.set_index("rgb_path")["weather"]
    rows = data[data["rgb_path"].isin(pool["rgb_path"])].reset_index(drop=True)
    ds = tc.WindowDataset(rows["stem"], rows["k0"], rows["label_B"], None, [0, 0, 0], [1, 1, 1], cfg["frames"],
                          cfg["model_fps"], cfg["lru_size"], "cctv_v2", None, dl)
    rng = np.random.default_rng(args.seed)

    def pick(w: str | None, n: int, exclude: set) -> list[str]:
        c = sorted(set(pool.loc[pool["weather"] == w, "rgb_path"] if w else
                       pool.loc[pool["weather"] != "night", "rgb_path"]) - exclude)
        return list(rng.choice(c, size=n, replace=False))

    used: set = set()
    clips_a = pick("night", 8, used)
    for w in ("clear", "sunset", "wet", "rain"):
        clips_a += pick(w, 4, used)
    used |= set(clips_a)
    idx = {p: centre_window(rows, p) for p in pool["rgb_path"]}
    luma = {}

    def luma_of(p: str) -> float:
        if p not in luma:
            luma[p] = tc.window_luma(ds.load(idx[p], False, "none", None)[0])
        return luma[p]

    # worst-case candidates: night windows not dark, day windows, dark night windows
    night_all = sorted(set(pool.loc[pool["weather"] == "night", "rgb_path"]) - used)
    order = rng.permutation(len(night_all))
    night_bright, night_dark = [], []
    for i in order:
        p = night_all[i]
        (night_dark if luma_of(p) < dl else night_bright).append(p)
        if len(night_bright) >= 6 and len(night_dark) >= 4:
            break
    clips_b, clips_dark = night_bright[:6] + pick(None, 6, used | set(night_all)), night_dark[:4]
    tc.check_cache(rows.iloc[[idx[p] for p in clips_a + clips_b + clips_dark]], rows.iloc[:0], manifest, partial=True)
    args.out.mkdir(parents=True, exist_ok=True)
    head = (f"cctv_v2; config {args.config.name} (sha256 {cfg_hash[:12]}); seed {args.seed}; frames {tc.PREVIEW_FRAMES}; "
            f"dark_luma {dl:g}")
    txt, written = [head], []

    def describe(i: int, p: str, applied: list[str]) -> list[str]:
        r = rows.iloc[idx[p]]
        name = (f"{i}: {r['stem']} ({weather[p]}) t_start {r['t_start']:.1f} s, t* {r['t_star']:.2f} s, "
                f"window luma {luma_of(p):.1f}{' DARK' if luma_of(p) < dl else ''}")
        par = "applied: " + ", ".join(applied)
        txt.append(f"{name}; {par}")
        return [name, par]

    # a) real draws
    blocks = []
    txt.append("a) real random draws")
    for i, p in enumerate(clips_a):
        flip = bool(np.random.default_rng([args.seed, 3, i]).random() < cfg["hflip_p"])
        orig, _ = ds.load(idx[p], False, "none", None)
        aug, applied = ds.load(idx[p], flip, "cctv_v2", np.random.default_rng([args.seed, 1, i]))
        blocks.append([strip(orig, f"{i} original"), strip(aug, f"{i} cctv_v2"), band(describe(i, p, applied))])
    written += save_pages(blocks, [head, "a) original vs cctv_v2 with the real random draws; windows centred on t*"],
                          "a_real", args.out)

    # b) worst case with 2 degradations
    def worst_plan(b_low: float, ops: list[str], seed_i: int):
        def make(_rng, dark):
            g = np.random.default_rng([args.seed, 4, seed_i])
            plan = {"drawn": ops, "kept": ops, "dark": dark, "downscale": 0.5}
            if "colour" in ops:
                ct = float(g.uniform(*v["contrast_dark" if dark else "contrast"]))
                plan["colour"] = (b_low, ct, float(g.uniform(*v["saturation"])), float(g.uniform(-v["hue"], v["hue"])),
                                  bool(g.random() < v["grayscale_p"]))
            return plan
        return make

    blocks = []
    txt.append("b) worst case: downscale 0.5 + colour at the lowest brightness allowed (0.7)")
    for i, p in enumerate(clips_b):
        orig, _ = ds.load(idx[p], False, "none", None)
        aug, applied = ds.load_v2(idx[p], False, None, worst_plan(v["brightness"][0], ["downscale", "colour"], i))
        blocks.append([strip(orig, f"{i} original"), strip(aug, f"{i} worst case"), band(describe(i, p, applied))])
    written += save_pages(blocks, [head, "b) worst case with 2 degradations: downscale 0.5 + colour shift, brightness "
                                         f"{v['brightness'][0]} (6 night windows not dark, 6 day windows)"],
                          "b_worst", args.out)
    blocks = []
    txt.append("b_dark) dark night windows: worst allowed case = 1 degradation, downscale 0.5")
    for i, p in enumerate(clips_dark):
        orig, _ = ds.load(idx[p], False, "none", None)
        aug, applied = ds.load_v2(idx[p], False, None, worst_plan(1.0, ["downscale"], 100 + i))
        blocks.append([strip(orig, f"{i} original"), strip(aug, f"{i} worst (dark)"), band(describe(i, p, applied))])
    written += save_pages(blocks, [head, f"b_dark) dark night windows (luma < {dl:g}): at most 1 degradation; worst "
                                         "allowed = downscale 0.5 (brightness >= 1.0)"], "b_dark", args.out)
    (args.out / "previews.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")
    print("\n".join(str(p) for p in written))
    print("\n".join(txt))


if __name__ == "__main__":
    main()
