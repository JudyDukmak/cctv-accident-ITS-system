"""Report figures F3-F8 from existing outputs (SYNTHETIC ONLY; no model, no GPU, no training, no real data; reads
results/diagnostics/ and runs/ read-only; writes only to --out, default results/figures/).

F1 and F2 (training curves) are made by overfit_vs_lr.py --report-figures. This separate script exists because F3-F8
need the post-processed clip predictions and window scores of several runs and seeds (aggregate_runs.py --tta-study
outputs), which no existing script plots; aggregate_runs.py is the official summary tool and stays free of plotting.
Window scores and the peak rule are read / recomputed with aggregate_runs.window_scores and peak_t_hat (the same
code as the reported numbers).

Inputs (frozen post-processing (b) = smoothing 3 + parabola + flip TTA; (a) default = smoothing 3, no interpolation,
no TTA, saved predictions):
  C_cctv seeds 0-2: results/diagnostics/C_cctv_seeds012/seed<s>_frozen/ (tta_table.csv, tta_clip_predictions_flip.csv),
                    independent_postproc_check.csv
  C_v2 seed 0:      results/diagnostics/C_v2_seed0_frozen/frozen/tta_table.csv
  post-processing:  results/diagnostics/postproc_C_cctv_seed0/postproc_variants.csv
  window scores:    results/diagnostics/postproc_C_cctv_seed0/tta/fold_<k>/val_window_scores_{orig,hflip}.csv
  previews:         runs/aug_preview/C_v2_cctv_v2_night50/{d_day_real_01,a_night_real_01}.png
  tau evidence:     results/diagnostics/tau_evidence/{tau_bins.csv,isotonic_curve.csv} (confidence_vs_error.py
                    --tau-evidence)
Outputs: F3_ablation_bars.png, F4_signed_error.png, F5_confidence_quartiles.png, F6_example_curves.png (+ .csv of the
chosen clips), F7_postprocessing.png, F8_augmentation_examples.png (+ the two source pages copied to F8_sources/),
F9_confidence_evidence.png; PNG at 300 dpi.

Usage (training environment, CPU):
    .venv-ml\\Scripts\\python.exe report_figures.py [--out results\\figures] [--only 9]
"""

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from accs_scorer import temporal  # noqa: E402
from aggregate_runs import peak_t_hat, window_scores  # noqa: E402
from overfit_vs_lr import C_CCTV_SEED_COLOURS, C_V2_COLOUR, GRID, INK, MUTED, style_axis  # noqa: E402

DIAG = REPO_ROOT / "results" / "diagnostics"
SEEDS_DIR = DIAG / "C_cctv_seeds012"
C_V2_FROZEN = DIAG / "C_v2_seed0_frozen" / "frozen" / "tta_table.csv"
POSTPROC = DIAG / "postproc_C_cctv_seed0"
PREVIEW = REPO_ROOT / "runs" / "aug_preview" / "C_v2_cctv_v2_night50"
TAU_DIR = DIAG / "tau_evidence"  # confidence_vs_error.py --tau-evidence
C_CCTV_MEAN = "#2a78d6"
T_HAT_COLOUR = "#6250d6"  # prediction marker in F6 (not a configuration colour)
SEEDS = (0, 1, 2)
SCOPES = ["fold 0", "fold 1", "fold 2", "pooled"]
FROZEN_ARM, DEFAULT_ARM = "flip TTA", "saved best CSV (training GPU), no TTA"
DPI = 300


def fail(msg: str) -> None:
    sys.exit(f"ERROR: {msg}")


def need(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        fail("missing input(s): " + ", ".join(missing))


def arm_table(path: Path, arm: str) -> pd.DataFrame:
    t = pd.read_csv(path)
    t = t[t["arm"] == arm].set_index("scope")
    if list(t.index) != SCOPES:
        fail(f"{path}: arm {arm!r} scopes {list(t.index)}")
    return t


def frozen_preds() -> pd.DataFrame:
    out = []
    for s in SEEDS:
        p = pd.read_csv(SEEDS_DIR / f"seed{s}_frozen" / "tta_clip_predictions_flip.csv").assign(seed=s)
        if len(p) != 2211 or p["rgb_path"].duplicated().any():
            fail(f"seed {s}: frozen predictions do not cover 2211 distinct clips")
        out.append(p)
    p = pd.concat(out, ignore_index=True)
    p["err"] = p["t_hat"] - p["t_star"]
    return p


def f3(out: Path) -> None:
    need(C_V2_FROZEN, *[SEEDS_DIR / f"seed{s}_frozen" / "tta_table.csv" for s in SEEDS])
    v2 = arm_table(C_V2_FROZEN, FROZEN_ARM)
    cc = {s: arm_table(SEEDS_DIR / f"seed{s}_frozen" / "tta_table.csv", FROZEN_ARM) for s in SEEDS}
    panels = [("T@1", "T@1 (0-1)"), ("T@0.5", "T@0.5 (0-1)"), ("within_2s", "share within 2 s (0-1)")]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    x = np.arange(len(SCOPES))
    for a, (m, ylab) in zip(ax, panels):
        vals = np.array([[cc[s].loc[sc, m] for sc in SCOPES] for s in SEEDS])
        mean, sd = vals.mean(axis=0), vals.std(axis=0, ddof=1)
        a.bar(x + 0.12, mean, width=0.36, color=C_CCTV_MEAN, alpha=0.35, edgecolor="none",
              label="C_cctv mean of seeds 0-2")
        a.errorbar(x + 0.12, mean, yerr=sd, fmt="none", ecolor=INK, elinewidth=1, capsize=3,
                   label="+- sample std (ddof 1, n 3)")
        for i, s in enumerate(SEEDS):
            a.plot(x + 0.12 + (i - 1) * 0.09, vals[i], "o", ms=4.5, color=C_CCTV_SEED_COLOURS[s],
                   markeredgecolor="white", markeredgewidth=0.5, label=f"C_cctv seed {s}")
        a.plot(x - 0.22, [v2.loc[sc, m] for sc in SCOPES], "D", ms=6, color=C_V2_COLOUR,
               markeredgecolor="white", markeredgewidth=0.6, label="C_v2 seed 0")
        a.text(x[-1] + 0.12, mean[-1] + sd[-1] + 0.02, f"{mean[-1]:.3f}", ha="center", fontsize=7, color=INK)
        a.text(x[-1] - 0.22, v2.loc["pooled", m] + 0.02, f"{v2.loc['pooled', m]:.3f}", ha="center", fontsize=7,
               color=INK)
        a.set_xticks(x, SCOPES)
        a.set_ylim(0, 1)
        a.set_title(ylab, fontsize=9, color=INK, loc="left")
        style_axis(a)
    ax[0].set_ylabel("score (synthetic validation)", fontsize=8, color=INK)
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=6, fontsize=7.5, frameon=False)
    fig.text(0.5, 0.01, "frozen post-processing (smoothing 3 + parabola + flip TTA); pooled = out-of-fold over 2211 "
             "clips", ha="center", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(out / "F3_ablation_bars.png", dpi=DPI)
    plt.close(fig)


def f4(out: Path, p: pd.DataFrame) -> None:
    lo, hi = -10.0, 15.0
    bins = np.arange(lo, hi + 0.5, 0.5)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for k, a in enumerate(ax):
        e = p.loc[p["fold"] == k, "err"].to_numpy()
        n_lo, n_hi = int((e < lo).sum()), int((e >= hi).sum())
        a.hist(np.clip(e, lo, hi - 1e-9), bins=bins, color=C_CCTV_MEAN, edgecolor="white", linewidth=0.4)
        a.axvspan(-2, 2, color=GRID, alpha=0.5, zorder=0, lw=0)
        a.axvline(0, color=INK, lw=0.8)
        a.set_title(f"fold {k}: n {len(e)} ({len(e) // 3} clips x 3 seeds); within 2 s {np.mean(np.abs(e) <= 2):.3f}",
                    fontsize=8, color=INK, loc="left")
        a.text(0.99, 0.95, f"clipped into edge bins:\n{n_lo} below {lo:g} s, {n_hi} at/above +{hi:g} s",
               transform=a.transAxes, ha="right", va="top", fontsize=6.5, color=MUTED)
        a.set_xlabel("signed error t_hat - t* (s; + = late)", fontsize=8, color=INK)
        style_axis(a)
    ax[0].set_ylabel("predictions per 0.5 s bin", fontsize=8, color=INK)
    fig.text(0.5, 0.005, "C_cctv seeds 0-2, frozen post-processing; shaded band = |error| <= 2 s", ha="center",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "F4_signed_error.png", dpi=DPI)
    plt.close(fig)


def f5(out: Path, p: pd.DataFrame) -> None:
    p = p.copy()
    p["q"], edges = pd.qcut(p["confidence"], 4, labels=False, retbins=True)
    rows = []
    for q in range(4):
        g = p[p["q"] == q]
        rows.append({"q": q, "n": len(g), "within": float((g["err"].abs() <= 2).mean()),
                     "T1": float(temporal(g["t_hat"], g["t_star"], 1.0)),
                     **{f"within s{s}": float((g.loc[g["seed"] == s, "err"].abs() <= 2).mean()) for s in SEEDS},
                     **{f"T1 s{s}": float(temporal(g.loc[g["seed"] == s, "t_hat"], g.loc[g["seed"] == s, "t_star"], 1.0))
                        for s in SEEDS}})
    r = pd.DataFrame(rows)
    labels = [f"Q{q + 1}\n{edges[q]:.2f}-{edges[q + 1]:.2f}\nn {r['n'][q]}" for q in range(4)]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    x = np.arange(4)
    for a, (key, title) in zip(ax, [("within", "share within 2 s (0-1)"), ("T1", "T@1 (0-1)")]):
        a.bar(x, r[key], width=0.55, color=C_CCTV_MEAN, alpha=0.35, edgecolor="none", label="seeds pooled")
        for i, s in enumerate(SEEDS):
            a.plot(x + (i - 1) * 0.12, r[f"{key} s{s}"], "o", ms=4.5, color=C_CCTV_SEED_COLOURS[s],
                   markeredgecolor="white", markeredgewidth=0.5, label=f"seed {s}")
        for xi, v in zip(x, r[key]):
            a.text(xi, v + 0.03, f"{v:.3f}", ha="center", fontsize=7, color=INK)
        a.set_xticks(x, labels, fontsize=7)
        a.set_ylim(0, 1.05)
        a.set_title(title, fontsize=9, color=INK, loc="left")
        a.set_xlabel("confidence quartile (peak - max outside +-2 windows; Q1 lowest)", fontsize=8, color=INK)
        style_axis(a)
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, fontsize=7.5, frameon=False)
    fig.text(0.5, 0.005, "C_cctv seeds 0-2, frozen post-processing; quartile edges over the 6633 pooled predictions; "
             "no threshold chosen", ha="center", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(out / "F5_confidence_quartiles.png", dpi=DPI)
    plt.close(fig)
    r.to_csv(out / "F5_confidence_quartiles.csv", index=False)


def f6(out: Path, p: pd.DataFrame) -> None:
    """Seed 0, frozen post-processing. Per category, the clip whose |error| is closest to the category median |error|
    (ties: rgb_path order): correct |err| <= 0.5 s, late err > +2 s, early err < -2 s."""
    s0 = p[p["seed"] == 0].sort_values("rgb_path")
    cats = {"correct (|error| <= 0.5 s)": s0[s0["err"].abs() <= 0.5],
            "late peak (error > +2 s)": s0[s0["err"] > 2], "early peak (error < -2 s)": s0[s0["err"] < -2]}
    chosen = []
    for name, g in cats.items():
        d = (g["err"].abs() - g["err"].abs().median()).abs()
        chosen.append(g.loc[d.sort_values(kind="stable").index[0]].to_dict() | {"category": name, "n_in_category": len(g)})
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    rows = []
    for a, c in zip(ax, chosen):
        k = int(c["fold"])
        win = window_scores("C_cctv_seed0", k, POSTPROC / "tta", "flip")
        w = win[win["rgb_path"] == c["rgb_path"]].sort_values("t_centre")
        sm = np.convolve(w["score"].to_numpy(np.float32).astype(float), np.ones(3) / 3, mode="same")
        chk = peak_t_hat(w, 3, "parabola", 2)  # exclude = confidence_exclude of the runs (2 windows)
        if not np.isclose(chk["t_hat"].iloc[0], c["t_hat"]):
            fail(f"{c['rgb_path']}: recomputed t_hat {chk['t_hat'].iloc[0]} != {c['t_hat']}")
        a.plot(w["t_centre"], w["score"], "-", color=C_CCTV_SEED_COLOURS[2], lw=1, label="window score (flip TTA)")
        a.plot(w["t_centre"], sm, "-", color=C_CCTV_SEED_COLOURS[0], lw=2, label="smoothed (width 3)")
        a.axvline(c["t_star"], color=INK, lw=1.2, ls="--", label="t* (label)")
        a.axvline(c["t_hat"], color=T_HAT_COLOUR, lw=1.4, label="t_hat (parabola peak)")
        clip = Path(c["rgb_path"]).stem
        a.set_title(f"{c['category']}\n{clip} (fold {k}); error {c['err']:+.2f} s", fontsize=8, color=INK, loc="left")
        a.set_xlabel("window centre time (s)", fontsize=8, color=INK)
        a.set_ylim(0, 1.02)
        style_axis(a)
        rows.append({k2: c[k2] for k2 in ("category", "n_in_category", "rgb_path", "fold", "t_star", "t_hat", "err",
                                           "confidence")})
    ax[0].set_ylabel("accident score (sigmoid, 0-1)", fontsize=8, color=INK)
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, fontsize=7.5, frameon=False)
    fig.text(0.5, 0.005, "C_cctv seed 0, best checks, frozen post-processing; clip per category = |error| closest to "
             "the category median", ha="center", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(out / "F6_example_curves.png", dpi=DPI)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "F6_example_clips.csv", index=False)


def f7(out: Path) -> None:
    need(POSTPROC / "postproc_variants.csv", SEEDS_DIR / "independent_postproc_check.csv")
    v = pd.read_csv(POSTPROC / "postproc_variants.csv")
    v = v[v["run"] == "C_cctv_seed0"].reset_index(drop=True)
    fz = arm_table(SEEDS_DIR / "seed0_frozen" / "tta_table.csv", FROZEN_ARM).loc["pooled"]
    labels = [f"width {int(r.smooth)}, {r.interp}" + (" (default)" if r.default else "") for r in v.itertuples()]
    labels.append("width 3, parabola + flip TTA (frozen)")
    ind = pd.read_csv(SEEDS_DIR / "independent_postproc_check.csv")
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.2))
    y = np.arange(len(labels))[::-1]
    for a, (col, fcol, title) in zip(ax[0], [("pooled T@1", "T@1", "T@1 (0-1)"), ("pooled T@0.5", "T@0.5", "T@0.5 (0-1)")]):
        vals = list(v[col]) + [fz[fcol]]
        a.hlines(y, min(vals) - 0.01, vals, color=GRID, lw=1)
        colours = [C_CCTV_SEED_COLOURS[1]] * len(v) + [C_CCTV_SEED_COLOURS[0]]
        a.scatter(vals, y, s=36, c=colours, edgecolors="white", linewidths=0.5, zorder=3)
        for yi, val in zip(y, vals):
            a.text(val + 0.002, yi, f"{val:.4f}", va="center", fontsize=7, color=INK)
        a.set_yticks(y, labels if a is ax[0, 0] else [""] * len(labels), fontsize=7)
        a.set_xlim(min(vals) - 0.01, max(vals) + 0.02)
        a.set_title(f"seed 0, pooled out-of-fold {title}", fontsize=8.5, color=INK, loc="left")
        a.set_xlabel(f"{title.split(' ')[0]} (dots not anchored at 0: differences are small)", fontsize=7.5, color=INK)
        style_axis(a)
    x = np.arange(len(SCOPES))
    for a, m in zip(ax[1], ["T@1", "T@0.5"]):
        for i, s in enumerate((1, 2)):
            d = ind[ind["seed"] == s].set_index("scope").loc[SCOPES, f"d{m} (b)-(a)"]
            a.bar(x + (i - 0.5) * 0.36, d, width=0.34, color=C_CCTV_SEED_COLOURS[s],
                  label=f"seed {s}" if m == "T@1" else None)
        a.axhline(0, color=INK, lw=0.8)
        a.set_xticks(x, SCOPES)
        a.set_ylabel(f"gain in {m}, (b) - (a)", fontsize=8, color=INK)
        a.set_title(f"seeds 1, 2: {m} gain, frozen (b) - default (a)", fontsize=8.5, color=INK, loc="left")
        a.set_ylim(min(0.0, ind[f"d{m} (b)-(a)"].min() * 1.3), ind[f"d{m} (b)-(a)"].max() * 1.3)  # legend headroom
        style_axis(a)
    ax[1, 0].legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
    fig.text(0.5, 0.005, "top: chosen on C_cctv seed 0 (variants on the saved scores; frozen = flip TTA scored on the "
             "RTX 5060); bottom: independent check on seeds 1 and 2 (not used for the choice)", ha="center",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "F7_postprocessing.png", dpi=DPI)
    plt.close(fig)


def f8(out: Path) -> None:
    """Copies two preview pages and stacks one window block of each (pixels copied, nothing regenerated): day
    window 102 (block 2 of d_day_real_01.png) and protected night window 0 (block 0 of a_night_real_01.png)."""
    day, night = PREVIEW / "d_day_real_01.png", PREVIEW / "a_night_real_01.png"
    need(day, night)
    src = out / "F8_sources"
    src.mkdir(exist_ok=True)
    for f in (day, night):
        shutil.copy2(f, src / f.name)

    def blocks(path: Path) -> tuple[np.ndarray, list[tuple[int, int]]]:
        a = np.asarray(Image.open(path).convert("RGB"))
        f = a.astype(float)
        sep = np.where((f.std(axis=(1, 2)) < 2) & (np.abs(f.mean(axis=(1, 2)) - 128) < 1))[0]
        starts = [0] + [int(r) + 1 for r in sep if r + 1 not in sep]
        ends = [int(r) for r in sep if r - 1 not in sep]
        if len(ends) != 4:
            fail(f"{path}: expected 4 window blocks, found {len(ends)}")
        return a, list(zip(starts, ends))

    a_day, b_day = blocks(day)
    a_night, b_night = blocks(night)
    # block 0 starts with the page header (white rows with text); the window starts at the first row after the
    # last all-white row of the header (searched in the first 100 rows)
    top = a_night[:100].astype(float).min(axis=(1, 2)) > 240
    header = int(np.where(top)[0].max()) + 1
    d0, d1 = b_day[2]
    n0, n1 = b_night[0]
    gap = np.full((12, a_day.shape[1], 3), 255, np.uint8)
    img = np.concatenate([a_day[d0:d1], gap, a_night[n0 + header:n1]], axis=0)
    Image.fromarray(img).save(out / "F8_augmentation_examples.png", dpi=(DPI, DPI))


def f9(out: Path, p: pd.DataFrame) -> None:
    """Confidence evidence for tau: histogram of confidence (top); binned share within 2 s with 95% Wilson intervals
    and the isotonic p_narrow curve (bottom). Tables from confidence_vs_error.py --tau-evidence."""
    bins_csv, curve_csv = TAU_DIR / "tau_bins.csv", TAU_DIR / "isotonic_curve.csv"
    need(bins_csv, curve_csv)
    b = pd.read_csv(bins_csv)
    b = b[b["scope"] == "pooled (seeds 0-2)"].copy()
    if int(b["n_clips"].sum()) != len(p):
        fail(f"{bins_csv}: pooled bins hold {int(b['n_clips'].sum())} predictions, predictions {len(p)}")
    b["lo"] = b["confidence"].str.extract(r"\[(\d\.\d)")[0].astype(float)
    curve = pd.read_csv(curve_csv)
    fig, ax = plt.subplots(2, 1, figsize=(7.5, 6.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.5]})
    ax[0].hist(p["confidence"], bins=np.arange(0, 1.0001, 0.02), color=C_CCTV_MEAN, edgecolor="white", linewidth=0.4)
    ax[0].set_ylabel("predictions per 0.02 bin", fontsize=8, color=INK)
    ax[0].set_title(f"confidence of the {len(p)} pooled predictions (C_cctv seeds 0-2, 2211 clips each)", fontsize=8.5,
                    color=INK, loc="left")
    x = b["lo"] + 0.05
    ax[1].errorbar(x, b["share_within_2s"], yerr=[b["share_within_2s"] - b["wilson95_lo"],
                                                  b["wilson95_hi"] - b["share_within_2s"]],
                   fmt="o", ms=5, color=C_CCTV_SEED_COLOURS[0], ecolor=C_CCTV_SEED_COLOURS[0], elinewidth=1, capsize=3,
                   label="share within 2 s per 0.1 bin, 95% Wilson interval")
    for xi, yi, n in zip(x, b["share_within_2s"], b["n_clips"]):
        ax[1].text(xi, 0.03, f"n {n}", ha="center", fontsize=6.5, color=MUTED)
    ax[1].plot(curve["confidence"], curve["p_narrow_isotonic"], "-", color=T_HAT_COLOUR, lw=1.8,
               label="isotonic p_narrow(c) (increasing fit, pooled)")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_ylabel("P(|t_hat - t*| <= 2 s)", fontsize=8, color=INK)
    ax[1].set_xlabel("confidence c (peak - max smoothed score outside +-2 windows)", fontsize=8, color=INK)
    ax[1].set_xlim(0, 1)
    ax[1].legend(fontsize=7, frameon=False, loc="upper left")
    for a in ax:
        style_axis(a)
    fig.text(0.5, 0.005, "frozen post-processing; seeds pooled (3 predictions per clip, not independent: intervals are "
             "too narrow); no threshold chosen", ha="center", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "F9_confidence_evidence.png", dpi=DPI)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "figures")
    ap.add_argument("--only", default=None, help="comma-separated figure numbers to make, e.g. 9 (default: 3-9)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8})
    p = frozen_preds()
    if args.only is not None:
        figs = {3: lambda: f3(args.out), 4: lambda: f4(args.out, p), 5: lambda: f5(args.out, p),
                6: lambda: f6(args.out, p), 7: lambda: f7(args.out), 8: lambda: f8(args.out), 9: lambda: f9(args.out, p)}
        for k in (int(v) for v in args.only.split(",")):
            figs[k]()
        print(f"wrote F{args.only} to {args.out}")
        return
    f3(args.out)
    f4(args.out, p)
    f5(args.out, p)
    f6(args.out, p)
    f7(args.out)
    f8(args.out)
    f9(args.out, p)
    print(f"wrote F3-F9 to {args.out}")


if __name__ == "__main__":
    main()
