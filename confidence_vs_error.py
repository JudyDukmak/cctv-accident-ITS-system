"""Clip confidence vs temporal error (SYNTHETIC ONLY; reads the saved validation clip predictions, as
compare_diagnostics.py; no model, no GPU, writes nothing in runs/). Per item: the fold's clips split into 4
confidence quartiles (pd.qcut; Q1 = lowest), with n, within_2s, median |t_hat - t_star| and T@1 per quartile,
and the Spearman correlation between confidence and |t_hat - t_star| over all clips.
confidence = peak smoothed score - max smoothed score more than confidence_exclude windows from the peak
(train_classifier.clip_predictions).
Output: results/diagnostics/confidence_vs_error_<name>.txt

Usage:
    .venv-ml\\Scripts\\python.exe confidence_vs_error.py --fold 0 --name B_C \\
        "B_v2_seed0 (best)=B_v2_seed0:best" "C_v2_seed0 (best)=C_v2_seed0:best"
Items as in compare_diagnostics.py: LABEL=RUN:WHICH.
Several folds: --folds 0,1,2 gives the blocks per fold and a pooled block (the folds' clips together, quartiles over
the pooled clips); --out-dir writes confidence_vs_error_<name>.txt there instead of results/diagnostics/. Description
only: no confidence threshold is chosen.

--tau-evidence (evidence for the routing threshold tau; no items, no --name; nothing is chosen):
    .venv-ml\\Scripts\\python.exe confidence_vs_error.py --tau-evidence --out-dir results\\diagnostics\\tau_evidence
  Reads results/diagnostics/C_cctv_seeds012/seed{0,1,2}_frozen/tta_clip_predictions_flip.csv (frozen post-processing;
  each must hold the 2211 synthetic clips once) and synthetic_window_labels.csv (label_B = 1 windows per clip).
  Writes to --out-dir: tau_cumulative.csv (confidence >= c, c = 0.9 ... 0.0), tau_bins.csv ([0.9, 1.0], [0.8, 0.9),
  ..., [0.0, 0.1)), both pooled (6633 predictions) and per seed, with n, coverage, n and share within 2 s (+ 95%
  Wilson interval), T@1 and the label_B = 1 windows of those clips; isotonic_curve.csv (p_narrow(c) = P(|t_hat - t*|
  <= 2 s | confidence c), increasing isotonic regression by pool-adjacent-violators on the pooled predictions, ties
  in confidence pooled; evaluated on c = 0.00 ... 0.96 by linear interpolation between the fitted values at the
  observed confidences, constant outside them); isotonic_fit_points.csv; tau_evidence.txt.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
import train_classifier as tc  # noqa: E402
from accs_scorer import temporal  # noqa: E402
from compare_diagnostics import DIAG_ROOT, load  # noqa: E402


def block(label: str, pred: pd.DataFrame, clips: pd.DataFrame) -> list[str]:
    p = pred[["rgb_path", "t_hat", "confidence"]].merge(clips, on="rgb_path", validate="one_to_one")
    if len(p) != len(clips) or p["confidence"].isna().any():
        tc.fail(f"{label}: predictions do not cover the fold's validation clips or lack confidence")
    p["abs_err"] = (p["t_hat"] - p["t_star"]).abs()
    p["quartile"] = pd.qcut(p["confidence"], 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
    g = p.groupby("quartile", observed=True)
    out = pd.DataFrame({"n": g.size(), "conf_min": g["confidence"].min(), "conf_max": g["confidence"].max(),
                        "within_2s": g["abs_err"].apply(lambda e: (e <= 2).mean()),
                        "median_abs_err_s": g["abs_err"].median(),
                        "T@1": g.apply(lambda d: temporal(d["t_hat"], d["t_star"], 1.0), include_groups=False)})
    rho = p["confidence"].rank().corr(p["abs_err"].rank())  # Spearman = Pearson on average ranks (no scipy here)
    return [f"=== {label} ({len(p)} clips) ===", out.reset_index().to_string(index=False, float_format=lambda v: f"{v:.4f}"),
            f"Spearman rho(confidence, |t_hat - t_star|) = {rho:.4f}", ""]


TAU_SEEDS = (0, 1, 2)
TAU_PRED = REPO_ROOT / "results" / "diagnostics" / "C_cctv_seeds012" / "seed{s}_frozen" / "tta_clip_predictions_flip.csv"
WINDOW_LABELS = REPO_ROOT / "synthetic_window_labels.csv"
N_SYNTHETIC_CLIPS = 2211
WILSON_Z = 1.959963984540054  # two-sided 95%


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval of a binomial share k / n (nan for n = 0)."""
    if n == 0:
        return float("nan"), float("nan")
    p, z2 = k / n, WILSON_Z ** 2
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = WILSON_Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return centre - half, centre + half


def isotonic_increasing(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted least-squares increasing fit (pool adjacent violators). Equal x values are pooled first (their mean,
    weight = count). Returns the unique sorted x, the fitted value at each and the count at each."""
    ux, inv = np.unique(x, return_inverse=True)
    w = np.bincount(inv).astype(float)
    m = np.bincount(inv, weights=y) / w
    blocks: list[list[float]] = []  # [weighted sum, weight, number of unique x in the block]
    for mi, wi in zip(m, w):
        blocks.append([mi * wi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, ww, c = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += ww
            blocks[-1][2] += c
    fit = np.concatenate([np.full(int(c), s / ww) for s, ww, c in blocks])
    return ux, fit, w


def tau_rows(p: pd.DataFrame, scope: str, total: int, masks: list[tuple[str, pd.Series]]) -> list[dict]:
    rows = []
    for name, mk in masks:
        g = p[mk]
        k = int(g["within_2s"].sum())
        lo, hi = wilson(k, len(g))
        rows.append({"scope": scope, "confidence": name, "n_clips": len(g), "coverage": len(g) / total,
                     "n_within_2s": k, "share_within_2s": k / len(g) if len(g) else float("nan"),
                     "wilson95_lo": lo, "wilson95_hi": hi,
                     "T@1": float(temporal(g["t_hat"], g["t_star"], 1.0)) if len(g) else float("nan"),
                     "label_B1_windows": int(g["label_B1_windows"].sum())})
    return rows


def tau_evidence(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lab = pd.read_csv(WINDOW_LABELS, usecols=["rgb_path", "label_B"])
    # label_B is 1 / 0 / -1 (-1 = not used by scheme B); count only the windows with label_B == 1
    pos = lab.assign(b1=lab["label_B"] == 1).groupby("rgb_path")["b1"].sum().rename("label_B1_windows")
    parts, checks = [], []
    for s in TAU_SEEDS:
        f = Path(str(TAU_PRED).format(s=s))
        p = pd.read_csv(f)
        if len(p) != N_SYNTHETIC_CLIPS or p["rgb_path"].nunique() != N_SYNTHETIC_CLIPS:
            tc.fail(f"{f}: {len(p)} rows, {p['rgb_path'].nunique()} distinct clips; expected {N_SYNTHETIC_CLIPS}")
        if p[["t_hat", "t_star", "confidence"]].isna().any().any():
            tc.fail(f"{f}: missing t_hat, t_star or confidence")
        missing = sorted(set(p["rgb_path"]) - set(pos.index))
        if missing:
            tc.fail(f"{f}: {len(missing)} clips not in {WINDOW_LABELS.name}, e.g. {missing[0]}")
        checks.append(f"seed {s}: {f.relative_to(REPO_ROOT)}: {len(p)} rows = {N_SYNTHETIC_CLIPS} distinct clips; "
                      f"confidence {p['confidence'].min():.4f} .. {p['confidence'].max():.4f}")
        parts.append(p.assign(seed=s))
    p = pd.concat(parts, ignore_index=True)
    if (p["confidence"] < 0).any() or (p["confidence"] > 1).any():
        tc.fail("confidence outside [0, 1]")
    p = p.merge(pos, left_on="rgb_path", right_index=True, how="left", validate="many_to_one")
    p["abs_err"] = (p["t_hat"] - p["t_star"]).abs()
    p["within_2s"] = p["abs_err"] <= 2.0

    thresholds = [round(0.9 - 0.1 * i, 1) for i in range(10)]
    bins = [(round(0.1 * i, 1), round(0.1 * (i + 1), 1)) for i in range(9, -1, -1)]
    scopes = [("pooled (seeds 0-2)", p)] + [(f"seed {s}", p[p["seed"] == s]) for s in TAU_SEEDS]
    cum, binned = [], []
    for scope, q in scopes:
        c = q["confidence"]
        cum += tau_rows(q, scope, len(q), [(f">= {t:.1f}", c >= t) for t in thresholds])
        binned += tau_rows(q, scope, len(q), [(f"[{lo:.1f}, {hi:.1f}]" if hi == 1.0 else f"[{lo:.1f}, {hi:.1f})",
                                               (c >= lo) & ((c <= hi) if hi == 1.0 else (c < hi))) for lo, hi in bins])
    cum, binned = pd.DataFrame(cum), pd.DataFrame(binned)
    for scope, q in scopes:
        if binned.loc[binned["scope"] == scope, "n_clips"].sum() != len(q):
            tc.fail(f"bins of {scope} do not cover all predictions exactly once")
    ux, fit, w = isotonic_increasing(p["confidence"].to_numpy(float), p["within_2s"].to_numpy(float))
    grid = np.round(np.arange(0, 0.9601, 0.01), 2)
    curve = pd.DataFrame({"confidence": grid, "p_narrow_isotonic": np.interp(grid, ux, fit)})
    pd.DataFrame({"confidence": ux, "n": w.astype(int), "fitted": fit}).to_csv(out_dir / "isotonic_fit_points.csv",
                                                                            index=False)
    cum.to_csv(out_dir / "tau_cumulative.csv", index=False)
    binned.to_csv(out_dir / "tau_bins.csv", index=False)
    curve.to_csv(out_dir / "isotonic_curve.csv", index=False)

    def show(t: pd.DataFrame, scope: str) -> str:
        return t[t["scope"] == scope].drop(columns="scope").to_string(index=False, float_format=lambda v: f"{v:.4f}")

    def brief(t: pd.DataFrame) -> str:
        s = t[t["scope"] != "pooled (seeds 0-2)"].pivot(index="confidence", columns="scope",
                                                         values=["coverage", "share_within_2s"])
        return s.reindex(t["confidence"].drop_duplicates()).to_string(float_format=lambda v: f"{v:.4f}")

    at = curve.set_index("confidence")["p_narrow_isotonic"]
    text = "\n".join(
        ["Evidence for the routing threshold tau (SYNTHETIC ONLY, CPU; no real data; nothing chosen). C_cctv seeds 0-2,",
         "best checks, frozen post-processing (smoothing 3 + parabola + flip TTA), pooled out-of-fold.",
         "within 2 s = |t_hat - t*| <= 2 s; coverage = share of the scope's predictions (pooled 6633, seed 2211);",
         "label_B1_windows = label_B = 1 windows of those clips in synthetic_window_labels.csv.", ""] + checks + [
         f"confidence of the pooled predictions: min {p['confidence'].min():.4f}, median {p['confidence'].median():.4f}, "
         f"max {p['confidence'].max():.4f}; {len(ux)} distinct values", "",
         "1. cumulative (confidence >= c), pooled", show(cum, "pooled (seeds 0-2)"), "",
         "   per seed: coverage and share within 2 s", brief(cum), "",
         "2. bins, pooled", show(binned, "pooled (seeds 0-2)"), "",
         "   per seed: coverage and share within 2 s", brief(binned), "",
         "3. isotonic p_narrow(c) (increasing, pooled 6633; isotonic_curve.csv, grid 0.00-0.96)",
         "   " + "; ".join(f"c {c:.1f}: {at[round(c, 2)]:.4f}" for c in (0.1, 0.2, 0.3, 0.4, 0.5)),
         f"   fitted range {fit.min():.4f} .. {fit.max():.4f} over {len(set(np.round(fit, 12)))} constant pieces", "",
         "Seeds are pooled as three predictions per clip, so the 6633 predictions are not independent; the Wilson",
         "intervals treat them as independent and are therefore too narrow. No tau is chosen here (RUNS.md).", ""])
    (out_dir / "tau_evidence.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("items", nargs="*", help="LABEL=RUN:WHICH")
    ap.add_argument("--fold", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--folds", default=None, help="comma-separated folds: per-fold blocks and a pooled block")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--tau-evidence", action="store_true", help="tau evidence tables (see module doc)")
    args = ap.parse_args()
    if args.tau_evidence:
        if args.items or args.name:
            tc.fail("--tau-evidence takes no items and no --name")
        if args.out_dir is None:
            tc.fail("--tau-evidence needs --out-dir")
        tau_evidence(args.out_dir)
        return
    if not args.items or not args.name:
        tc.fail("items and --name are required (unless --tau-evidence)")

    data, _, _ = tc.load_labels(8)
    folds = [int(f) for f in args.folds.split(",")] if args.folds else [args.fold]
    rep, pooled = [], {}
    for k in folds:
        _, val_rows = tc.split_fold(data, k, smoke=False)
        clips = val_rows.groupby("rgb_path", sort=True).agg(t_star=("t_star", "first")).reset_index()
        rep += [f"fold {k}, synthetic validation: {len(clips)} clips; confidence quartiles by pd.qcut", ""]
        for item in args.items:
            label, spec = item.split("=", 1)
            run, which = spec.split(":", 1)
            pred = load(run, which, k)[1]
            rep += block(label, pred, clips)
            pooled.setdefault(label, []).append((pred, clips))
    if len(folds) > 1:
        rep += [f"pooled folds {folds} (quartiles over the pooled clips; confidence scales of different folds' models "
                "are mixed)", ""]
        for label, parts in pooled.items():
            rep += block(f"{label}, pooled", pd.concat([p for p, _ in parts], ignore_index=True),
                         pd.concat([c for _, c in parts], ignore_index=True))
    text = "\n".join(rep)
    out = (args.out_dir or DIAG_ROOT) / f"confidence_vs_error_{args.name}.txt"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
