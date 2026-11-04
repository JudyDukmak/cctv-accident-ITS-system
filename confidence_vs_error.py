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
"""

import argparse
import sys
from pathlib import Path

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("items", nargs="+", help="LABEL=RUN:WHICH")
    ap.add_argument("--fold", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    data, _, _ = tc.load_labels(8)
    _, val_rows = tc.split_fold(data, args.fold, smoke=False)
    clips = val_rows.groupby("rgb_path", sort=True).agg(t_star=("t_star", "first")).reset_index()
    rep = [f"fold {args.fold}, synthetic validation: {len(clips)} clips; confidence quartiles by pd.qcut", ""]
    for item in args.items:
        label, spec = item.split("=", 1)
        run, which = spec.split(":", 1)
        rep += block(label, load(run, which, args.fold)[1], clips)
    text = "\n".join(rep)
    out = DIAG_ROOT / f"confidence_vs_error_{args.name}.txt"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
