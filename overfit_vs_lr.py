"""Overfitting vs learning rate too high: per-check curves of existing runs (CPU only; reads metrics.csv and
config.json of each run; no model, no torch, writes nothing in runs/).

Per check: update, epoch_fraction, val T@0.5/1/2, val_loss, train_loss (mean training loss since the previous
check, as logged), head lr (the peak lr group) as logged and as recomputed from config.json (linear warmup over
warmup_updates, then cosine to 0 at total_updates; same formula as train_classifier.make_lr_lambda).
Per run: best check, lr there / peak lr, linear fit of val T@1 vs epoch_fraction from the best check on (slope,
residual std), sign changes of consecutive T@1 differences, largest single drop and rise.
Outputs: results/diagnostics/overfit_vs_lr/ (one CSV and one PNG per run, summary.txt).

Usage (training environment, CPU):
    .venv-ml\\Scripts\\python.exe overfit_vs_lr.py
    .venv-ml\\Scripts\\python.exe overfit_vs_lr.py --report-figures results\\figures
        report figures only (nothing else written): F1_learning_curves.png (val T@1, val loss, train loss vs
        epoch_fraction per fold, C_v2_seed0 and C_cctv seeds 0-2, best checks marked) and F2_C_v2_vs_C_cos3.png
        (val T@1 and lr / peak lr per fold, C_v2_seed0 and C_cos3_seed0); 300 dpi.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
RUNS = {"B_v2_seed0 fold 0": REPO_ROOT / "runs" / "B_v2_seed0" / "fold_0",
        "C_v2_seed0 fold 0": REPO_ROOT / "runs" / "C_v2_seed0" / "fold_0",
        "C_v2_seed0 fold 2": REPO_ROOT / "runs" / "C_v2_seed0" / "fold_2"}
OUT_DIR = REPO_ROOT / "results" / "diagnostics" / "overfit_vs_lr"


def lr_factor(step: int, warmup: int, total: int) -> float:
    """Copy of train_classifier.make_lr_lambda: multiplier of the peak lr used at optimizer update `step` (0-based)."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def sign_changes(d: np.ndarray) -> int:
    s = np.sign(d)
    s = s[s != 0]
    return int((s[1:] != s[:-1]).sum())


def per_check(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    m = pd.read_csv(run_dir / "metrics.csv")
    info = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    eff = info["effective"]
    peak = eff["scaled_lr"]
    # lr_head = lr of the last update before the check, i.e. update index global_step - 1
    recomputed = [peak * lr_factor(int(g) - 1, eff["warmup_updates"], eff["total_updates"]) for g in m["global_step"]]
    has_lr = "lr_head" in m and m["lr_head"].notna().all()
    df = pd.DataFrame({
        "check_idx": m["check_idx"] if "check_idx" in m else m["epoch"],
        "update": m["global_step"],
        "epoch_fraction": m["epoch_fraction"] if "epoch_fraction" in m else m["global_step"] / eff["updates_per_epoch"],
        "val_T_0.5": m["val_T_0.5"], "val_T_1": m["val_T_1"], "val_T_2": m["val_T_2"],
        "val_loss": m["val_loss"], "train_loss_since_prev_check": m["train_loss"],
        "lr_head": m["lr_head"] if has_lr else recomputed,
        "lr_source": "logged" if has_lr else "recomputed",
        "lr_head_recomputed": recomputed,
        "lr_backbone_top_block": m["lr_backbone"] if "lr_backbone" in m else np.nan,
    })
    df["lr_frac_of_peak"] = df["lr_head"] / peak
    return df, info


def summarise(df: pd.DataFrame, info: dict) -> dict:
    t = df["val_T_1"].to_numpy()
    x = df["epoch_fraction"].to_numpy()
    b = int(np.argmax(t))  # first occurrence of the max
    d = np.diff(t)
    xs, ts = x[b:], t[b:]
    if len(xs) >= 2:
        slope, icpt = np.polyfit(xs, ts, 1)
        resid_std = float(np.std(ts - (slope * xs + icpt))) if len(xs) >= 3 else float("nan")
    else:
        slope = resid_std = float("nan")
    eff = info["effective"]
    i_drop, i_rise = int(np.argmin(d)), int(np.argmax(d))
    return {
        "label scheme": eff.get("label_scheme", info["config"].get("label_scheme", "B")),
        "checks": len(df),
        "updates/epoch, warmup, total updates": f"{eff['updates_per_epoch']}, {eff['warmup_updates']}, "
                                                f"{eff['total_updates']}",
        "peak lr (head)": f"{eff['scaled_lr']:.4g}",
        "warmup ends at epoch_fraction": f"{eff['warmup_updates'] / eff['updates_per_epoch']:.3f}",
        "lr source": df["lr_source"].iloc[0],
        "max |logged - recomputed| lr / peak": f"{(df['lr_head'] - df['lr_head_recomputed']).abs().max() / eff['scaled_lr']:.2e}",
        "best check (max val T@1)": int(df["check_idx"].iloc[b]),
        "best val T@1": f"{t[b]:.4f}",
        "best: update / epoch_fraction": f"{int(df['update'].iloc[b])} / {x[b]:.3f}",
        "best: lr / peak lr": f"{df['lr_frac_of_peak'].iloc[b]:.3f}",
        "last check: epoch_fraction, lr / peak lr": f"{x[-1]:.3f}, {df['lr_frac_of_peak'].iloc[-1]:.3f}",
        "from best on: n points in fit": len(xs),
        "from best on: slope T@1 per epoch": f"{slope:+.4f}",
        "from best on: residual std": f"{resid_std:.4f}",
        "sign changes of dT@1, whole run": f"{sign_changes(d)} of {max(0, len(d) - 1)} possible",
        "sign changes of dT@1, from best on": f"{sign_changes(d[b:])} of {max(0, len(d[b:]) - 1)} possible",
        "largest drop (check a -> b)": f"{d[i_drop]:+.4f} ({int(df['check_idx'].iloc[i_drop])} -> "
                                       f"{int(df['check_idx'].iloc[i_drop + 1])})",
        "largest rise (check a -> b)": f"{d[i_rise]:+.4f} ({int(df['check_idx'].iloc[i_rise])} -> "
                                       f"{int(df['check_idx'].iloc[i_rise + 1])})",
        "train loss: first / at best / last check": " / ".join(
            f"{df['train_loss_since_prev_check'].iloc[i]:.4f}" for i in (0, b, -1)),
        "val loss: first / at best / min / last check": " / ".join(
            f"{v:.4f}" for v in (df["val_loss"].iloc[0], df["val_loss"].iloc[b], df["val_loss"].min(),
                                 df["val_loss"].iloc[-1])),
    }


def plot(df: pd.DataFrame, title: str, path: Path, warmup_ef: float) -> None:
    x = df["epoch_fraction"]
    b = int(np.argmax(df["val_T_1"].to_numpy()))
    fig, ax = plt.subplots(4, 1, figsize=(8, 9), dpi=110, sharex=True)
    ax[0].plot(x, df["val_T_1"], "o-", color="tab:blue")
    ax[0].plot(x.iloc[b], df["val_T_1"].iloc[b], "*", color="tab:red", ms=16,
               label=f"best: check {int(df['check_idx'].iloc[b])}, T@1 {df['val_T_1'].iloc[b]:.4f}")
    ax[0].set_ylabel("val T@1")
    ax[0].legend(fontsize=8)
    ax[1].plot(x, df["train_loss_since_prev_check"], "o-", color="tab:green")
    ax[1].set_ylabel("train loss\n(mean since prev. check)")
    ax[2].plot(x, df["val_loss"], "o-", color="tab:orange")
    ax[2].set_ylabel("val loss\n(BCE, label_B 0/1)")
    ax[3].plot(x, df["lr_head"], "o-", color="tab:purple", label=f"head lr ({df['lr_source'].iloc[0]})")
    ax[3].set_ylabel("lr")
    ax[3].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax[3].legend(fontsize=8)
    for a in ax:
        a.axvline(x.iloc[b], color="tab:red", lw=1, ls=":")
        a.axvline(warmup_ef, color="0.5", lw=1, ls="--")
        a.grid(alpha=0.3)
    ax[3].set_xlabel("epoch_fraction (dashed grey: end of warmup; dotted red: best check)")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# Report figure colours (one hue per configuration; seeds of C_cctv as steps of the blue ramp, seed 0 darkest)
C_V2_COLOUR = "#eb6834"
C_COS3_COLOUR = "#1baf7a"
C_CCTV_SEED_COLOURS = {0: "#1c5cab", 1: "#3987e5", 2: "#86b6ef"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def style_axis(a) -> None:
    a.grid(color=GRID, lw=0.6)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_color("#c3c2b7")
    a.tick_params(colors=MUTED, labelsize=7)


def report_figures(out_dir: Path) -> None:
    """F1 and F2 of the report (see module doc); reads metrics.csv / config.json only."""
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = REPO_ROOT / "runs"
    f1 = [("C_v2 seed 0", "C_v2_seed0", C_V2_COLOUR)] + \
         [(f"C_cctv seed {s}", f"C_cctv_seed{s}", C_CCTV_SEED_COLOURS[s]) for s in (0, 1, 2)]
    cols = [("val_T_1", "val T@1 (0-1)"), ("val_loss", "val loss (BCE)"),
            ("train_loss_since_prev_check", "train loss (mean since\nprevious check)")]
    fig, ax = plt.subplots(3, 3, figsize=(10, 8), sharex="row")
    for k in (0, 1, 2):
        for label, run, colour in f1:
            df, _ = per_check(runs_dir / run / f"fold_{k}")
            b = int(np.argmax(df["val_T_1"].to_numpy()))
            for j, (c, ylab) in enumerate(cols):
                a = ax[k, j]
                a.plot(df["epoch_fraction"], df[c], "-", color=colour, lw=1.5, marker="o", ms=2.5,
                       label=label if (k, j) == (0, 0) else None)
                a.plot(df["epoch_fraction"].iloc[b], df[c].iloc[b], marker="*", ms=10, color=colour,
                       markeredgecolor="white", markeredgewidth=0.6, zorder=5)
                a.set_ylabel(ylab, fontsize=8, color=INK)
        for j in range(3):
            style_axis(ax[k, j])
            ax[k, j].set_title(f"fold {k}", fontsize=8, color=MUTED, loc="left")
    for j in range(3):
        ax[2, j].set_xlabel("training progress (epochs)", fontsize=8, color=INK)
    handles, labels = ax[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], ls="", marker="*", ms=10, color=MUTED))
    labels.append("best check (max val T@1)")
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "F1_learning_curves.png", dpi=300)
    plt.close(fig)

    f2 = [("C_v2 seed 0 (cosine to 0 over 10 epochs)", "C_v2_seed0", C_V2_COLOUR, 14),
          ("C_cos3 seed 0 (cosine to 0 over 3 epochs)", "C_cos3_seed0", C_COS3_COLOUR, 9)]
    fig, ax = plt.subplots(3, 2, figsize=(8.5, 7.5), sharex=True)
    for k in (0, 1, 2):
        for label, run, colour, star in f2:  # C_v2 star larger: both runs can share a best check
            df, _ = per_check(runs_dir / run / f"fold_{k}")
            b = int(np.argmax(df["val_T_1"].to_numpy()))
            ax[k, 0].plot(df["epoch_fraction"], df["val_T_1"], "-", color=colour, lw=1.5, marker="o", ms=2.5,
                          label=label if k == 0 else None)
            ax[k, 0].plot(df["epoch_fraction"].iloc[b], df["val_T_1"].iloc[b], marker="*", ms=star, color=colour,
                          markeredgecolor="white", markeredgewidth=0.6, zorder=5)
            ax[k, 1].plot(df["epoch_fraction"], df["lr_frac_of_peak"], "-", color=colour, lw=1.5, marker="o", ms=2.5)
        ax[k, 0].set_ylabel("val T@1 (0-1)", fontsize=8, color=INK)
        ax[k, 1].set_ylabel("head lr / peak lr", fontsize=8, color=INK)
        for j in range(2):
            style_axis(ax[k, j])
            ax[k, j].set_title(f"fold {k}", fontsize=8, color=MUTED, loc="left")
    for j in range(2):
        ax[2, j].set_xlabel("training progress (epochs)", fontsize=8, color=INK)
    handles, labels = ax[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], ls="", marker="*", ms=10, color=MUTED))
    labels.append("best check")
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=False,
               title="both: linear warmup over 0.5 epoch", title_fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "F2_C_v2_vs_C_cos3.png", dpi=300)
    plt.close(fig)
    print(f"wrote {out_dir / 'F1_learning_curves.png'} and {out_dir / 'F2_C_v2_vs_C_cos3.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report-figures", type=Path, default=None, metavar="DIR",
                    help="write only the report figures F1 and F2 to DIR")
    args = ap.parse_args()
    if args.report_figures is not None:
        report_figures(args.report_figures)
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries, pngs = {}, []
    for label, run_dir in RUNS.items():
        m = pd.read_csv(run_dir / "metrics.csv")
        print(f"=== {run_dir.relative_to(REPO_ROOT)}\\metrics.csv: {len(m)} rows, {len(m.columns)} columns")
        print(m.head(3).to_string())
        print()
        df, info = per_check(run_dir)
        slug = label.replace(" ", "_")
        df.to_csv(OUT_DIR / f"{slug}.csv", index=False)
        eff = info["effective"]
        png = OUT_DIR / f"{slug}.png"
        plot(df, f"{label} ({Path(info['config_file']).name}, seed {info['seed']})", png,
             eff["warmup_updates"] / eff["updates_per_epoch"])
        pngs.append(png)
        summaries[label] = summarise(df, info)

    c0 = json.loads((RUNS["C_v2_seed0 fold 0"] / "config.json").read_text(encoding="utf-8"))["effective"]
    table = pd.DataFrame(summaries)
    text = "\n".join([
        "Overfitting vs learning rate too high: existing validation checks (synthetic validation; no training)",
        "Runs: " + "; ".join(f"{k} = {v.relative_to(REPO_ROOT)}" for k, v in RUNS.items()),
        "(runs\\B_v2_seed0 has no metrics.csv of its own; the B_v2 run is fold_0.)",
        "",
        table.to_string(),
        "",
        "Notes:",
        "- Training T (T@sigma on training clips) is not logged; training loss (mean over the windows of the",
        "  updates since the previous check) is used as the proxy for fit to the training data.",
        "- B and C training losses are not on the same scale (B: hard label_B 0/1 targets; C: soft weight_C_text",
        "  targets over all windows, different pos_weight). val_loss is BCE on label_B 0/1 validation windows for",
        "  both schemes.",
        "- Per-check checkpoints do not exist: each run keeps only best.pt (weights of the best check), last.pt",
        "  (end of the last completed epoch) and step_latest.pt (the latest check / step). Intermediate checks",
        "  cannot be re-scored.",
        "- lr = head (peak lr group) lr used at the last update before the check, as logged in metrics.csv",
        "  (lr_head); it was also recomputed from config.json (warmup then cosine to 0) and the maximum difference",
        "  is listed above. Lower layers use layer decay 0.7 per layer below the head.",
        "- Fit from the best check on includes the best check itself; residual std = population std (ddof 0) of",
        "  the residuals. Sign changes ignore zero differences. Each check is one value on one validation fold,",
        "  one seed: check-to-check noise is not separated from trends.",
        "",
        f"C_v2_seed0 fold 0 schedule (config.json 'effective'): updates_per_epoch {c0['updates_per_epoch']}, "
        f"warmup_updates {c0['warmup_updates']}, total_updates {c0['total_updates']} "
        f"(max_epochs {c0['max_epochs']}, warmup_fraction 0.05).",
        "",
    ])
    (OUT_DIR / "summary.txt").write_text(text, encoding="utf-8")
    print(text)
    for p in pngs:
        print(f"wrote {p}")
    print(f"wrote {OUT_DIR / 'summary.txt'}")


if __name__ == "__main__":
    main()
