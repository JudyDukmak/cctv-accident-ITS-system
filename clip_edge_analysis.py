"""Clip-edge description of a run's best-check out-of-fold predictions (SYNTHETIC ONLY; saved CSVs only: runs/<run>/
fold_{0,1,2}/val_*_best.csv, labels, metadata-synthetic.csv; no model, no GPU, writes nothing in runs/).
Description only: no new rule or threshold is fitted or applied.

1. Clips whose t_hat (or t*) lies in the first 2 s (t < 2) or the last 2 s (duration - t <= 2) of the clip: n,
   share, T@1, within_2s; per fold and pooled. duration = metadata-synthetic.csv.
2. Windows outside t* +- 2 s (|t_centre - t*| > 2): mean smoothed score (the run's smooth_window, as in training)
   by time to clip end (duration - t_centre) and by time from clip start (t_centre), bins 0-1, 1-2, 2-5, > 5 s.
3. Clips with |t_hat - t*| > 5 s: how many have t_hat in the last 2 s, by fold and by collision type.
4. The 10 worst of those (largest |error|), with existing score-curve plots in results/diagnostics/<run>_fold<k>/.
Outputs: results/diagnostics/<run>_3folds/clip_edge.txt and clip_edge_*.csv

Usage (training environment, CPU):
    .venv-ml\\Scripts\\python.exe clip_edge_analysis.py --run C_v2_seed0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
import train_classifier as tc  # noqa: E402  (labels and split only; no model, no CUDA)
from accs_scorer import temporal  # noqa: E402
from analyse_predictions import with_context  # noqa: E402
from compare_diagnostics import DIAG_ROOT, METADATA_SYN  # noqa: E402
from cv_fold_summary import FOLDS, fold_data  # noqa: E402

EDGE = 2.0  # s, the first / last 2 s of a clip (as specified)
BINS = [0, 1, 2, 5, np.inf]
BIN_LABELS = ["0-1 s", "1-2 s", "2-5 s", "> 5 s"]


def table(df: pd.DataFrame, index: bool = True) -> str:
    return df.to_string(index=index, float_format=lambda v: f"{v:.4f}")


def group_stats(p: pd.DataFrame, mask: pd.Series) -> dict:
    sub = p[mask]
    err = sub["t_hat"] - sub["t_star"]
    return {"n": int(mask.sum()), "share": float(mask.mean()),
            "T@1": float(temporal(sub["t_hat"], sub["t_star"], 1.0)) if len(sub) else np.nan,
            "within_2s": float((err.abs() <= 2).mean()) if len(sub) else np.nan}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="C_v2_seed0")
    args = ap.parse_args()
    out_dir = DIAG_ROOT / f"{args.run}_3folds"
    out_dir.mkdir(parents=True, exist_ok=True)

    data, _, _ = tc.load_labels(8)
    meta = pd.read_csv(METADATA_SYN, usecols=["rgb_path", "duration"])
    preds, wins = [], []
    for k in FOLDS:
        fd = fold_data(args.run, k, data, meta)
        win = pd.read_csv(tc.RUNS_DIR / args.run / f"fold_{k}" / "val_window_scores_best.csv")
        w = with_context(win, fd["val_rows"], fd["k"]).merge(meta, on="rgb_path", how="left", validate="many_to_one")
        preds.append(fd["pred"].assign(fold=k))
        wins.append(w.assign(fold=k))
    p = pd.concat(preds, ignore_index=True)
    w = pd.concat(wins, ignore_index=True)
    if p["rgb_path"].duplicated().any() or w[["rgb_path", "window_idx"]].duplicated().any():
        tc.fail("a clip or window appears in more than one fold")
    p["err"] = p["t_hat"] - p["t_star"]
    p["stem"] = p["rgb_path"].map(tc.stem_of)
    p["t_hat first 2 s"] = p["t_hat"] < EDGE
    p["t_hat last 2 s"] = p["duration"] - p["t_hat"] <= EDGE
    p["t* first 2 s"] = p["t_star"] < EDGE
    p["t* last 2 s"] = p["duration"] - p["t_star"] <= EDGE

    # coverage of the window grid near the clip ends (context for the 'last 2 s' rows)
    ends = w.groupby("rgb_path").agg(first_centre=("t_centre", "min"), last_centre=("t_centre", "max"),
                                     duration=("duration", "first"))
    gap_end = ends["duration"] - ends["last_centre"]
    grid = (f"window grid: first window centre per clip median {ends['first_centre'].median():.3f} s "
            f"(min {ends['first_centre'].min():.3f}); duration - last window centre median {gap_end.median():.3f} s "
            f"(p5 {gap_end.quantile(0.05):.3f}, p95 {gap_end.quantile(0.95):.3f}, min {gap_end.min():.3f}, "
            f"max {gap_end.max():.3f}); t_hat is always a window centre")

    # 1.
    rows = {}
    for scope, sub in [("pooled", p)] + [(f"fold {k}", p[p["fold"] == k]) for k in FOLDS]:
        for lab, m in (("t_hat in first 2 s", sub["t_hat first 2 s"]), ("t_hat in last 2 s", sub["t_hat last 2 s"]),
                       ("t_hat elsewhere", ~(sub["t_hat first 2 s"] | sub["t_hat last 2 s"])),
                       ("t* in first 2 s", sub["t* first 2 s"]), ("t* in last 2 s", sub["t* last 2 s"]),
                       ("t* elsewhere", ~(sub["t* first 2 s"] | sub["t* last 2 s"])), ("all clips", sub["fold"] >= 0)):
            rows[(scope, lab)] = group_stats(sub, m)
    t1 = pd.DataFrame(rows).T
    t1.index.names = ["scope", "clips"]
    t1["n"] = t1["n"].astype(int)
    both = pd.crosstab(p["t_hat last 2 s"], p["t* last 2 s"]).rename_axis(index="t_hat in last 2 s",
                                                                           columns="t* in last 2 s")

    # 2.
    out = w[(w["t_centre"] - w["t_star"]).abs() > 2].copy()
    out["to end"] = pd.cut(out["duration"] - out["t_centre"], BINS, right=False, labels=BIN_LABELS)
    out["from start"] = pd.cut(out["t_centre"], BINS, right=False, labels=BIN_LABELS)
    t2 = {}
    for col in ("to end", "from start"):
        g = out.groupby(["fold", col], observed=False)["smooth"]
        per = pd.DataFrame({"n": g.size(), "mean": g.mean()}).unstack("fold")
        gp = out.groupby(col, observed=False)["smooth"]
        per[("n", "pooled")], per[("mean", "pooled")] = gp.size(), gp.mean()
        t2[col] = per
    # 3.
    big = p[p["err"].abs() > 5]
    t3 = {}
    for by in ("fold", "ctype"):
        g = big.groupby(by)
        t3[by] = pd.DataFrame({"clips |err| > 5 s": g.size(), "of which t_hat in last 2 s": g["t_hat last 2 s"].sum(),
                               "of which t_hat in first 2 s": g["t_hat first 2 s"].sum(),
                               "of which late (err > 5 s)": g["err"].apply(lambda e: int((e > 5).sum()))})
        t3[by].loc["all"] = t3[by].sum()
        t3[by]["share in last 2 s"] = t3[by]["of which t_hat in last 2 s"] / t3[by]["clips |err| > 5 s"]
    # 4.
    worst = big[big["t_hat last 2 s"]].assign(abs_err=lambda d: d["err"].abs()).sort_values(
        ["abs_err", "rgb_path"], ascending=[False, True]).head(10)
    plots = []
    for _, r in worst.iterrows():
        found = sorted(q.name for q in (DIAG_ROOT / f"{args.run}_fold{r['fold']}").glob(f"curve_*_{r['stem']}.png"))
        plots.append(", ".join(found) or "-")
    worst = worst.assign(plots=plots)[["fold", "stem", "ctype", "duration", "t_star", "t_hat", "err", "plots"]]

    t1.to_csv(out_dir / "clip_edge_tail_shares.csv")
    for col, df in t2.items():
        df.to_csv(out_dir / f"clip_edge_score_by_time_{col.replace(' ', '_')}.csv")
    for by, df in t3.items():
        df.to_csv(out_dir / f"clip_edge_big_errors_by_{by}.csv")
    worst.to_csv(out_dir / "clip_edge_worst10.csv", index=False)

    lines = [f"Clip-edge description: run {args.run}, best check per fold, out-of-fold, {len(p)} clips, {len(w)} windows "
             "(saved CSVs only; description, no rule or threshold fitted)", grid, "",
             "=== 1. clips with t_hat / t* in the first 2 s (t < 2) or the last 2 s (duration - t <= 2) ===",
             table(t1), "", "pooled cross-table, clips:", both.to_string(), "",
             "=== 2. mean smoothed window score, windows outside t* +- 2 s (|t_centre - t*| > 2) ===",
             "by time to clip end (duration - t_centre):", table(t2["to end"]), "",
             "by time from clip start (t_centre):", table(t2["from start"]), "",
             "=== 3. clips with |t_hat - t*| > 5 s ===", "by fold:", table(t3["fold"]), "",
             "by collision type:", table(t3["ctype"]), "",
             "=== 4. the 10 worst clips with |error| > 5 s and t_hat in the last 2 s ===",
             table(worst, index=False), ""]
    text = "\n".join(lines)
    (out_dir / "clip_edge.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out_dir / 'clip_edge.txt'}")


if __name__ == "__main__":
    main()
