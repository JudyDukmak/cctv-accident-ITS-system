"""Expected routing on real clips (cost estimate only; tau frozen at 0.01 on synthetic data, RUNS.md).

Reads ONLY the columns rgb_path, confidence and fallback of the saved real-test clip predictions of C_cctv seeds
0, 1, 2 (results/real_test/predictions/C_cctv_seed{s}/clip_predictions.csv), and ONLY the path column of
real_test_set.csv to know which clips belong to the frozen 400. No t_star, time, type or box label is read.
Counts clips with confidence < TAU (routed to the full scan). Clips without a prediction row or without a window
(fallback) count as confidence 0, i.e. below tau.
"""
import csv
import sys
from pathlib import Path

TAU = 0.01
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PRED = REPO / "results" / "real_test" / "predictions"
TEST_CSV = REPO / "real_test_set.csv"
SEEDS = (0, 1, 2)


def name(p: str) -> str:
    return p.replace("\\", "/").rsplit("/", 1)[-1]


def main() -> None:
    with open(TEST_CSV, newline="") as f:
        frozen = {name(r["path"]) for r in csv.DictReader(f)}  # only the path column is used
    out = [f"Expected routing on real clips: confidence < tau = {TAU} -> full scan. Only rgb_path / confidence /"
           f" fallback of the predictions and the path column of real_test_set.csv were read.",
           f"frozen 400 file names read: {len(frozen)}"]
    counts = {}
    for s in SEEDS:
        path = PRED / f"C_cctv_seed{s}" / "clip_predictions.csv"
        conf, fb = {}, {}
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                k = name(r["rgb_path"])
                conf[k] = float(r["confidence"]) if r["confidence"] not in ("", "nan") else 0.0
                fb[k] = r["fallback"]
        n_fb = sum(1 for v in fb.values() if v not in ("", "0", "False", "false"))
        fb_conf = sorted({conf[k] for k, v in fb.items() if v not in ("", "0", "False", "false")})
        missing400 = frozen - set(conf)
        sets = {"all": list(conf), "frozen400": [k for k in conf if k in frozen]}
        line = [f"seed {s}: {path.relative_to(REPO)} rows {len(conf)}; fallback rows {n_fb} (confidence {fb_conf});"
                f" frozen-400 clips missing from predictions {len(missing400)}"]
        for key, clips in sets.items():
            below = sum(1 for k in clips if conf[k] < TAU) + (len(missing400) if key == "frozen400" else 0)
            n = len(clips) + (len(missing400) if key == "frozen400" else 0)
            counts[(s, key)] = (below, n)
            line.append(f"  {key:10s} n {n:5d}  confidence < {TAU}: {below:5d}  share {below / n:.4f}")
        out.extend(line)
    for key in ("all", "frozen400"):
        b = [counts[(s, key)][0] for s in SEEDS]
        n = counts[(SEEDS[0], key)][1]
        mean = sum(b) / len(b)
        out.append(f"mean over seeds {key:10s} n {n:5d}  below tau {mean:.2f}  share {mean / n:.4f}"
                   f"  (per seed {b[0]} / {b[1]} / {b[2]})")
    text = "\n".join(out)
    print(text)
    (HERE / "report.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
