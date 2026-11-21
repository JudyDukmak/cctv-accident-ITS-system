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

--routing-t1 (added 2026-11-21; replaces the within-2-s tau rule on the supervisor's request, synthetic only, before
  any real cascade run; nothing is fitted on real data):
    .venv-ml\\Scripts\\python.exe confidence_vs_error.py --routing-t1 --out-dir results\\routing_T1
  Same C_cctv predictions as --tau-evidence joined with the frozen full-scan v3 results (results/vlm_full_scan/
  full_scan.csv, t_full with its fallbacks) and the clip midpoint (duration / 2). Per-clip T@1 = exp(-(t - t*)^2 / 2).
  1. bins [0, 0.01), [0.01, 0.02), [0.02, 0.05), [0.05, 0.1), [0.1, 0.2), [0.2, 1] and cumulative confidence < c
     (c = 0.01, 0.02, 0.05, 0.1), per seed and pooled: n and mean T@1 of the classifier t_hat, t_full and the midpoint.
  2. fallback (fixed rule): on confidence < 0.05, pooled, the higher mean T@1 of full scan and midpoint (tie: midpoint).
  3. tau: increasing isotonic fits (the function above) of the classifier's per-clip T@1 and of the chosen fallback's
     per-clip T@1 against the classifier confidence, pooled 6633, on the grid c = 0.00 ... 0.96 as --tau-evidence;
     tau = the smallest grid c with classifier fit >= fallback fit. Writes fallback_by_confidence.csv,
     isotonic_curves_T1.csv, isotonic_fit_points_T1.csv, isotonic_curves_T1.png, report.txt (refuses to overwrite).
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


FULL_SCAN_CSV = REPO_ROOT / "results" / "vlm_full_scan" / "full_scan.csv"
RT_BINS = [(0.0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 1.0)]
RT_CUM = (0.01, 0.02, 0.05, 0.1)
RT_CHOICE_BELOW = 0.05
PREDICTORS = {"classifier": "T1_classifier", "full_scan": "T1_full_scan", "midpoint": "T1_midpoint"}


def t1(t_hat: pd.Series, t_star: pd.Series) -> np.ndarray:
    """Per-clip T@1 = exp(-(t_hat - t*)^2 / 2) (accs_scorer eq. 1 with sigma_t = 1 s)."""
    return np.exp(-((t_hat.to_numpy(float) - t_star.to_numpy(float)) ** 2) / 2.0)


def routing_t1(out_dir: Path) -> None:
    """--routing-t1: fallback comparison by confidence bin, the fallback choice and tau on T@1 (see module doc)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "report.txt").exists():
        tc.fail(f"{out_dir / 'report.txt'} exists; refusing to overwrite")
    fs = pd.read_csv(FULL_SCAN_CSV, usecols=["rgb_path", "duration", "t_star", "t_full"])
    if len(fs) != N_SYNTHETIC_CLIPS or fs["rgb_path"].nunique() != N_SYNTHETIC_CLIPS or fs.isna().any().any():
        tc.fail(f"{FULL_SCAN_CSV}: expected {N_SYNTHETIC_CLIPS} complete rows")
    parts, checks = [], []
    for s in TAU_SEEDS:
        f = Path(str(TAU_PRED).format(s=s))
        p = pd.read_csv(f, usecols=["rgb_path", "t_hat", "confidence", "t_star", "fold"])
        if len(p) != N_SYNTHETIC_CLIPS or p["rgb_path"].nunique() != N_SYNTHETIC_CLIPS or p.isna().any().any():
            tc.fail(f"{f}: expected {N_SYNTHETIC_CLIPS} complete rows")
        m = p.merge(fs, on="rgb_path", suffixes=("", "_fs"), validate="one_to_one")
        if len(m) != N_SYNTHETIC_CLIPS or not np.allclose(m["t_star"], m["t_star_fs"], atol=1e-9, rtol=0):
            tc.fail(f"seed {s}: predictions and full scan do not cover the same clips with the same t*")
        checks.append(f"seed {s}: {f.relative_to(REPO_ROOT)} ({len(p)} clips, out-of-fold, frozen post-processing) "
                      f"joined with {FULL_SCAN_CSV.relative_to(REPO_ROOT)}: {len(m)} clips, t* identical")
        parts.append(m.drop(columns="t_star_fs").assign(seed=s))
    p = pd.concat(parts, ignore_index=True)
    p["T1_classifier"] = t1(p["t_hat"], p["t_star"])
    p["T1_full_scan"] = t1(p["t_full"], p["t_star"])
    p["T1_midpoint"] = t1(p["duration"] / 2.0, p["t_star"])
    scopes = [("pooled (seeds 0-2)", p)] + [(f"seed {s}", p[p["seed"] == s]) for s in TAU_SEEDS]

    def rows_for(q, scope, masks, kind):
        out = []
        for name, mk in masks:
            g = q[mk]
            out.append({"scope": scope, "kind": kind, "confidence": name, "n": len(g), "share": len(g) / len(q),
                        **{f"T@1 {k}": float(g[c].mean()) if len(g) else float("nan") for k, c in PREDICTORS.items()}})
        return out

    rows = []
    for scope, q in scopes:
        c = q["confidence"]
        bmask = [(f"[{lo:g}, {hi:g}]" if hi == 1.0 else f"[{lo:g}, {hi:g})",
                  (c >= lo) & ((c <= hi) if hi == 1.0 else (c < hi))) for lo, hi in RT_BINS]
        if sum(int(m.sum()) for _, m in bmask) != len(q):
            tc.fail(f"bins of {scope} do not cover all predictions exactly once")
        rows += rows_for(q, scope, bmask, "bin")
        rows += rows_for(q, scope, [(f"< {t:g}", c < t) for t in RT_CUM], "cumulative")
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "fallback_by_confidence.csv", index=False)

    low = p[p["confidence"] < RT_CHOICE_BELOW]
    fs_t1, mid_t1 = float(low["T1_full_scan"].mean()), float(low["T1_midpoint"].mean())
    choice = "full_scan" if fs_t1 > mid_t1 else "midpoint"  # a tie keeps the midpoint (simpler, no model call)

    x = p["confidence"].to_numpy(float)
    ux, fit_c, w = isotonic_increasing(x, p["T1_classifier"].to_numpy(float))
    _, fit_f, _ = isotonic_increasing(x, p[PREDICTORS[choice]].to_numpy(float))
    grid = np.round(np.arange(0, 0.9601, 0.01), 2)
    curve = pd.DataFrame({"confidence": grid, "classifier_T1_isotonic": np.interp(grid, ux, fit_c),
                          f"fallback_{choice}_T1_isotonic": np.interp(grid, ux, fit_f)})
    curve["classifier_minus_fallback"] = curve["classifier_T1_isotonic"] - curve[f"fallback_{choice}_T1_isotonic"]
    curve.to_csv(out_dir / "isotonic_curves_T1.csv", index=False)
    pd.DataFrame({"confidence": ux, "n": w.astype(int), "classifier_T1_fitted": fit_c,
                  f"fallback_{choice}_T1_fitted": fit_f}).to_csv(out_dir / "isotonic_fit_points_T1.csv", index=False)
    ok = curve["classifier_minus_fallback"] >= 0
    tau = float(curve.loc[ok, "confidence"].iloc[0]) if ok.any() else float("nan")
    below_after = curve[(curve["confidence"] > tau) & ~ok] if ok.any() else curve.iloc[0:0]
    at_tau = curve[curve["confidence"] == tau].iloc[0] if ok.any() else None
    prev = curve[curve["confidence"] == round(tau - 0.01, 2)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    pooled_bins = table[(table["scope"] == "pooled (seeds 0-2)") & (table["kind"] == "bin")]
    for ax, xmax in zip(axes, (0.96, 0.2)):
        ax.plot(curve["confidence"], curve["classifier_T1_isotonic"], label="classifier t_hat (isotonic)", color="C0")
        ax.plot(curve["confidence"], curve[f"fallback_{choice}_T1_isotonic"], label=f"{choice} (isotonic)", color="C1")
        ax.axvline(tau, color="k", ls="--", lw=1, label=f"tau = {tau:.2f}")
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1)
        ax.set_xlabel("classifier confidence")
        ax.set_ylabel("T@1 (sigma 1 s)")
        ax.grid(alpha=0.3)
    mids = [(lo + hi) / 2 for lo, hi in RT_BINS]
    for k, mk in (("classifier", "o"), ("full_scan", "s"), ("midpoint", "^")):
        axes[1].scatter(mids[:5], pooled_bins[f"T@1 {k}"].to_numpy()[:5], marker=mk, s=25,
                        label=f"{k}, bin mean (pooled)")
    axes[0].set_title("Isotonic T@1 vs confidence, pooled 6633 synthetic OOF predictions")
    axes[1].set_title("Zoom 0-0.2 with bin means (bins below 0.2)")
    axes[1].legend(fontsize=7, loc="lower right")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "isotonic_curves_T1.png", dpi=130)
    plt.close(fig)

    share = {scope: float((q["confidence"] >= tau).mean()) for scope, q in scopes}
    n_at = {scope: int((q["confidence"] >= tau).sum()) for scope, q in scopes}

    def show(scope, kind):
        t = table[(table["scope"] == scope) & (table["kind"] == kind)].drop(columns=["scope", "kind"])
        return t.to_string(index=False, float_format=lambda v: f"{v:.4f}")

    L = ["Routing on T@1 (SYNTHETIC ONLY, CPU; no model run, no real data). Replaces the within-2-s tau rule (supervisor "
         "feedback 2026-11-18), decided on synthetic data before any real cascade run.",
         "C_cctv seeds 0-2, frozen post-processing, out-of-fold; full scan = frozen v3 (results/vlm_full_scan); midpoint "
         "= duration / 2. T@1 per clip = exp(-(t - t*)^2 / 2); tables give the mean over the clips of a cell.", ""] + checks + [
         "", "1. FALLBACK COMPARISON BY CONFIDENCE BIN (n, share of the scope's predictions, mean T@1 of each predictor "
         "on the same clips)"]
    for scope, _ in scopes:
        L += [f"  {scope}: bins", "    " + show(scope, "bin").replace("\n", "\n    "),
              f"  {scope}: cumulative (confidence < c)", "    " + show(scope, "cumulative").replace("\n", "\n    "), ""]
    L += [f"2. FALLBACK CHOICE (fixed rule: confidence < {RT_CHOICE_BELOW:g}, pooled over seeds; higher mean T@1; a tie "
          "keeps the midpoint)",
          f"  n {len(low)} predictions ({low['rgb_path'].nunique()} distinct clips): full scan T@1 {fs_t1:.4f}, "
          f"midpoint T@1 {mid_t1:.4f} (classifier on the same: {low['T1_classifier'].mean():.4f})",
          f"  CHOICE: {choice}", "",
          "3. TAU ON T@1 (isotonic, increasing, pool adjacent violators, ties in confidence pooled, on the 6633 pooled "
          "predictions; the fallback's T@1 fitted against the same classifier confidence with the same method; both "
          "evaluated on c = 0.00 ... 0.96 by linear interpolation, constant outside the observed confidences, as the "
          "frozen tau evidence)",
          f"  tau = smallest c on the grid with classifier fitted T@1 >= {choice} fitted T@1"]
    if at_tau is not None:
        L.append(f"  tau = {tau:.2f}: classifier {at_tau['classifier_T1_isotonic']:.4f} vs {choice} "
                 f"{at_tau[f'fallback_{choice}_T1_isotonic']:.4f}" + (
                     f"; at c = {tau - 0.01:.2f}: classifier {prev['classifier_T1_isotonic'].iloc[0]:.4f} vs {choice} "
                     f"{prev[f'fallback_{choice}_T1_isotonic'].iloc[0]:.4f}" if len(prev) else " (first grid point)"))
        L.append(f"  grid points above tau where the classifier is below the fallback again: "
                 f"{', '.join(f'{v:.2f}' for v in below_after['confidence']) or 'none'}")
    else:
        L.append("  no grid point satisfies the rule: tau undefined")
    L.append(f"  fitted ranges: classifier {fit_c.min():.4f} .. {fit_c.max():.4f}; {choice} {fit_f.min():.4f} .. "
             f"{fit_f.max():.4f}")
    L.append("  predictions with confidence >= tau (routed to the classifier): " + "; ".join(
        f"{scope} {n_at[scope]}/{len(q)} = {share[scope]:.4f}" for scope, q in scopes))
    L += ["", "Seeds are pooled as three predictions per clip, so the 6633 predictions are not independent. The fallback "
          "curve is constrained to increase with confidence (same method as the classifier's curve).",
          "Files: fallback_by_confidence.csv, isotonic_curves_T1.csv, isotonic_fit_points_T1.csv, isotonic_curves_T1.png",
          f"TAU_T1 {tau:.2f} FALLBACK {choice}"]
    text = "\n".join(L) + "\n"
    (out_dir / "report.txt").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("items", nargs="*", help="LABEL=RUN:WHICH")
    ap.add_argument("--fold", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--folds", default=None, help="comma-separated folds: per-fold blocks and a pooled block")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--tau-evidence", action="store_true", help="tau evidence tables (see module doc)")
    ap.add_argument("--routing-t1", action="store_true", help="fallback comparison and tau on T@1 (see module doc)")
    args = ap.parse_args()
    if args.routing_t1:
        if args.items or args.name or args.out_dir is None:
            tc.fail("--routing-t1 takes no items and no --name, and needs --out-dir")
        routing_t1(args.out_dir)
        return
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
