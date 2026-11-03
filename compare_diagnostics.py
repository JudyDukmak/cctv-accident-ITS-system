"""One comparison table across saved validation predictions (SYNTHETIC ONLY; reads runs/<run>/fold_<k>/
val_*_best.csv and the last-check CSVs written by analyse_predictions.py; no model, no GPU, writes nothing
in runs/). Columns: T@0.5/1/2, within_2s, signed error mean/median, within_2s per collision type and for
Town04 rear-end (all, night, clear), mean window score of windows starting more than 5 s after t*
(aftermath) and of windows ending more than 1 s before t* (pre-impact, = label_B 0).
Output: results/diagnostics/comparison_<name>.txt

Usage:
    .venv-ml\\Scripts\\python.exe compare_diagnostics.py --fold 0 --name std_vs_B_v2 \\
        "std_seed0 (best)=std_seed0:best" "B_v2_seed0 (best)=B_v2_seed0:best" "B_v2_seed0 (check 7)=B_v2_seed0:check7"
Each item is LABEL=RUN:WHICH, WHICH = best (val_*_best.csv in the run) or epochN / checkN (CSVs in
results/diagnostics/<run>_fold<k>/).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
import train_classifier as tc  # noqa: E402
from accs_scorer import temporal  # noqa: E402

METADATA_SYN = REPO_ROOT / "ACCIDENT" / "dataset" / "metadata-synthetic.csv"
DIAG_ROOT = REPO_ROOT / "results" / "diagnostics"


def load(run: str, which: str, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if which == "best":
        d, w, p = tc.RUNS_DIR / run / f"fold_{fold}", "val_window_scores_best.csv", "val_clip_predictions_best.csv"
    else:
        d, w, p = DIAG_ROOT / f"{run}_fold{fold}", f"val_window_scores_{which}.csv", f"val_clip_predictions_{which}.csv"
    if not (d / w).is_file() or not (d / p).is_file():
        tc.fail(f"missing {d / w} or {d / p}")
    return pd.read_csv(d / w), pd.read_csv(d / p)


def column(win: pd.DataFrame, pred: pd.DataFrame, val_rows: pd.DataFrame, clips: pd.DataFrame) -> dict:
    p = pred[["rgb_path", "t_hat"]].merge(clips, on="rgb_path", validate="one_to_one")
    w = win[["rgb_path", "window_idx", "score"]].merge(val_rows[["rgb_path", "window_idx", "t_start", "t_end", "t_star"]],
                                                       on=["rgb_path", "window_idx"], validate="one_to_one")
    if len(p) != len(clips) or len(w) != len(val_rows):
        tc.fail("predictions do not cover the fold's validation clips / windows")
    err = p["t_hat"] - p["t_star"]
    w2 = lambda m: float((err[m].abs() <= 2).mean())  # noqa: E731
    out = {f"val T@{s:g}": float(temporal(p["t_hat"], p["t_star"], s)) for s in (0.5, 1.0, 2.0)}
    out |= {"within_2s": w2(slice(None)), "signed err mean (s)": float(err.mean()),
            "signed err median (s)": float(err.median())}
    for t in tc.TYPES:
        m = p["ctype"] == t
        out[f"within_2s {t} (n={int(m.sum())})"] = w2(m)
    r = p["scenario_id"] == "Town04_rear-end"
    for lab, m in (("all", r), ("night", r & (p["weather"] == "night")), ("clear", r & (p["weather"] == "clear"))):
        out[f"within_2s Town04 rear-end {lab} (n={int(m.sum())})"] = w2(m)
    aft = w["t_start"] - w["t_star"] > 5
    pre = w["t_end"] < w["t_star"] - 1
    out[f"mean score aftermath, start > t*+5 s (n={int(aft.sum())})"] = float(w.loc[aft, "score"].mean())
    out[f"mean score pre-impact, end < t*-1 s (n={int(pre.sum())})"] = float(w.loc[pre, "score"].mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("items", nargs="+", help="LABEL=RUN:WHICH")
    ap.add_argument("--fold", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    data, _, _ = tc.load_labels(8)
    _, val_rows = tc.split_fold(data, args.fold, smoke=False)
    meta = pd.read_csv(METADATA_SYN, usecols=["rgb_path", "weather"])
    clips = (val_rows.groupby("rgb_path", sort=True)
             .agg(t_star=("t_star", "first"), scenario_id=("scenario_id", "first"), ctype=("ctype", "first"))
             .reset_index().merge(meta, on="rgb_path", validate="one_to_one"))
    cols = {}
    for item in args.items:
        label, spec = item.split("=", 1)
        run, which = spec.split(":", 1)
        cols[label] = column(*load(run, which, args.fold), val_rows, clips)
    table = pd.DataFrame(cols)
    text = (f"fold {args.fold}, synthetic validation: {len(clips)} clips, {len(val_rows)} windows\n"
            + table.to_string(float_format=lambda v: f"{v:.4f}") + "\n")
    out = DIAG_ROOT / f"comparison_{args.name}.txt"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
