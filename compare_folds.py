"""One table across saved validation predictions of DIFFERENT folds (SYNTHETIC ONLY; reads the same CSVs as
compare_diagnostics.py; no model, no GPU, writes nothing in runs/). Rows: T@0.5/1/2, within_2s, signed error
median, within_2s per collision type ("n/a" where the fold has none), mean window score near t* (label_B 1,
windows containing t*), aftermath (start > t*+5 s) and pre-impact (end < t*-1 s), and the late-peak share
(compare_diagnostics.late_peak_share). Counts differ per fold, so each cell carries its own n.
Output: results/diagnostics/comparison_<name>.txt

Usage:
    .venv-ml\\Scripts\\python.exe compare_folds.py --name C_v2_fold0_vs_fold2 \\
        "C_v2 fold 0 (best)=C_v2_seed0:best:0" "C_v2 fold 2 (best)=C_v2_seed0:best:2"
Each item is LABEL=RUN:WHICH:FOLD, WHICH as in compare_diagnostics.py.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
import train_classifier as tc  # noqa: E402
from accs_scorer import temporal  # noqa: E402
from compare_diagnostics import DIAG_ROOT, late_peak_share, load, smooth_window  # noqa: E402


def cell(v: float, n: int | None = None) -> str:
    return f"{v:.4f}" + ("" if n is None else f" (n={n})")


def column(run: str, which: str, fold: int, data: pd.DataFrame) -> dict:
    _, val_rows = tc.split_fold(data, fold, smoke=False)
    clips = (val_rows.groupby("rgb_path", sort=True)
             .agg(t_star=("t_star", "first"), ctype=("ctype", "first")).reset_index())
    win, pred = load(run, which, fold)
    p = pred[["rgb_path", "t_hat"]].merge(clips, on="rgb_path", validate="one_to_one")
    w = win[["rgb_path", "window_idx", "score"]].merge(
        val_rows[["rgb_path", "window_idx", "t_start", "t_end", "t_centre", "t_star", "label_B"]],
        on=["rgb_path", "window_idx"], validate="one_to_one")
    if len(p) != len(clips) or len(w) != len(val_rows):
        tc.fail(f"{run}:{which} fold {fold}: predictions do not cover the fold's validation clips / windows")
    err = p["t_hat"] - p["t_star"]
    out = {"clips / windows": f"{len(p)} / {len(w)}"}
    out |= {f"val T@{s:g}": cell(float(temporal(p["t_hat"], p["t_star"], s))) for s in (0.5, 1.0, 2.0)}
    out |= {"within_2s": cell(float((err.abs() <= 2).mean())), "signed err median (s)": cell(float(err.median()))}
    for t in tc.TYPES:
        m = p["ctype"] == t
        out[f"within_2s {t}"] = cell(float((err[m].abs() <= 2).mean()), int(m.sum())) if m.any() else "n/a"
    for lab, m in (("near t*, label_B 1", w["label_B"] == 1),
                   ("aftermath, start > t*+5 s", w["t_start"] - w["t_star"] > 5),
                   ("pre-impact, end < t*-1 s", w["t_end"] < w["t_star"] - 1)):
        out[f"mean score {lab}"] = cell(float(w.loc[m, "score"].mean()), int(m.sum()))
    share, n = late_peak_share(w, smooth_window(run, fold))
    out["share late peak > peak near t*"] = cell(share, n)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("items", nargs="+", help="LABEL=RUN:WHICH:FOLD")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    data, _, _ = tc.load_labels(8)
    cols = {}
    for item in args.items:
        label, spec = item.split("=", 1)
        run, which, fold = spec.split(":", 2)
        cols[label] = column(run, which, int(fold), data)
    text = "synthetic validation, each column on its own fold\n" + pd.DataFrame(cols).to_string() + "\n"
    out = DIAG_ROOT / f"comparison_{args.name}.txt"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
