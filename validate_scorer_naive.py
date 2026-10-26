"""Validate accs_scorer.py by reproducing the paper's naive baseline on all real clips.

Reads metadata-real.csv READ-ONLY (no real_test_set.csv, no videos, no randomness).
Every number is computed with accs_scorer.py.

Naive baseline (paper Sec. 6; ACCIDENT/baselines/heuristic/naive.py):
    t_hat = duration / 2, x_hat = y_hat = 0.5, type_hat = most frequent type ("single").

Outputs:
    real_check/naive_validation.txt   the printout
    scorer_config.json                fixed scorer settings (written only if every check is OK)

Usage:
    py validate_scorer_naive.py [--dataset-path PATH] [--overwrite]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

import accs_scorer as acs

DEFAULT_DATASET = Path(
    r"C:\Users\ASUS\Documents\Uni\Senior 2\cctv-accident-ITS-system\ACCIDENT\dataset"
)
REPO_ROOT = Path(__file__).resolve().parent
OUT_TXT = REPO_ROOT / "real_check" / "naive_validation.txt"
OUT_CONFIG = REPO_ROOT / "scorer_config.json"
EXPECTED_N = 2027
TOL = 0.005
PAPER = {"T@0.5": 0.107, "T@1": 0.190, "T@2": 0.295, "S": 0.250, "C": 0.335, "ACCS": 0.245}
NAIVE_TYPE = "single"  # naive.py line 96


class Report:
    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(str(text))


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing scorer_config.json whose content differs")
    args = ap.parse_args()

    dataset_root = args.dataset_path.resolve()
    if (dataset_root / "dataset" / "metadata-real.csv").is_file():
        dataset_root = dataset_root / "dataset"
    meta_csv = dataset_root / "metadata-real.csv"
    if not meta_csv.is_file():
        fail(f"metadata-real.csv not found in {dataset_root}")
    raw = meta_csv.read_bytes()
    meta_sha = hashlib.sha256(raw).hexdigest()
    gt = pd.read_csv(meta_csv)
    need = ["path", "accident_time", "center_x", "center_y", "type", "duration", "x1", "y1", "x2", "y2"]
    missing = [c for c in need if c not in gt.columns]
    if missing:
        fail(f"metadata-real.csv is missing columns {missing}")
    if len(gt) != EXPECTED_N:
        fail(f"expected {EXPECTED_N} real clips, found {len(gt)}")

    r = Report()
    r("Scorer validation on the naive baseline (all real clips)")
    r(f"metadata: {meta_csv}")
    r(f"metadata sha256: {meta_sha}")
    r(f"clips: {len(gt)}")

    # 1. Spatial sigmas over all real clips.
    sigma_x, sigma_y = acs.spatial_sigmas(gt["x1"], gt["y1"], gt["x2"], gt["y2"])
    r("")
    r(f"1. sigma_x = {sigma_x:.6f}, sigma_y = {sigma_y:.6f}  (mean box width / height, all {len(gt)} clips)")

    # 2. Naive predictions.
    counts = gt["type"].value_counts()
    mode, share = counts.index[0], counts.iloc[0] / len(gt)
    r(f"2. most frequent type: {mode} ({counts.iloc[0]} / {len(gt)} = {share:.4f})")
    r("   type counts: " + ", ".join(f"{t} {n}" for t, n in counts.items()))
    if mode != NAIVE_TYPE:
        fail(f"most frequent type is '{mode}', but naive.py uses '{NAIVE_TYPE}'")
    if (counts == counts.iloc[0]).sum() > 1:
        fail(f"tie for the most frequent type: {counts[counts == counts.iloc[0]].index.tolist()}")
    pred = pd.DataFrame({
        "path": gt["path"],
        "t_hat": gt["duration"] / 2,
        "x_hat": 0.5,
        "y_hat": 0.5,
        "type_hat": mode,
    })

    # 3. Score and compare with the paper.
    res = acs.score_all(gt[["path", "accident_time", "center_x", "center_y", "type"]], pred, sigma_x, sigma_y)
    ours = {"T@0.5": res["T_sigma_0.5"], "T@1": res["T_sigma_1"], "T@2": res["T_sigma_2"],
            "S": res["S"], "C": res["C"], "ACCS": res["ACCS"]}
    r("")
    r(f"3. score_all on N = {res['N']} clips (ACCS uses sigma_t = {res['ACCS_sigma_t']}, spatial k = {res['spatial_k']})")
    r(f"   {'component':<10} {'ours':>8} {'paper':>8} {'|diff|':>8}  status")
    n_bad = 0
    for k, p in PAPER.items():
        d = abs(ours[k] - p)
        ok = d <= TOL
        n_bad += not ok
        r(f"   {k:<10} {ours[k]:8.4f} {p:8.3f} {d:8.4f}  {'OK' if ok else 'MISMATCH'}")
    r(f"   tolerance |diff| <= {TOL}")
    r("   per-class type accuracy (no paper value):")
    for t, v in res["C_per_class"].items():
        r(f"     {t:<10} {'n/a' if v is None else f'{v:.4f}'}")
    r(f"   macro type accuracy (no paper value): {res['C_macro']:.4f}")

    # 4. Scorer config (only when the scorer reproduces the paper).
    r("")
    if n_bad:
        r(f"4. {n_bad} MISMATCH line(s): scorer_config.json NOT written")
    else:
        config = {
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "sigma_scope": "mean over all 2027 real clips",
            "spatial_k": acs.SPATIAL_K,
            "accs_sigma_t": acs.ACCS_SIGMA_T,
            "sigmas_t": list(acs.SIGMAS_T),
            "source": "paper arXiv 2604.09819 Sec. 5, footnote 4; validated on naive baseline",
            "metadata_real_sha256": meta_sha,
        }
        text = json.dumps(config, indent=2) + "\n"
        if OUT_CONFIG.exists() and OUT_CONFIG.read_text(encoding="ascii") != text and not args.overwrite:
            fail(f"{OUT_CONFIG} exists with different content; rerun with --overwrite to replace it")
        if OUT_CONFIG.exists() and OUT_CONFIG.read_text(encoding="ascii") == text:
            r(f"4. {OUT_CONFIG.name} already exists with identical content (unchanged)")
        else:
            OUT_CONFIG.write_text(text, encoding="ascii", newline="\n")
            r(f"4. wrote {OUT_CONFIG}")

    OUT_TXT.parent.mkdir(exist_ok=True)
    OUT_TXT.write_text("\n".join(r.lines) + "\n", encoding="ascii", newline="\n")
    print(f"saved {OUT_TXT}")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
