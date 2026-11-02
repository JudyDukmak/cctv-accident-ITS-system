"""Summarise classifier runs on the synthetic cross-validation (SYNTHETIC ONLY: reads runs/ and
synthetic_split.csv; never opens real videos, metadata-real.csv or real_test_set.csv).

For each run name, runs/<name>/fold_{0,1,2}/ must hold config.json, metrics.csv and
val_clip_predictions_best.csv (written by train_classifier.py).
  Per fold:  the best epoch (last metrics row with best == True): val T@0.5/1/2 and val_within_2s.
  Pooled:    the best-epoch clip predictions of the three folds must cover each of the 2,211 synthetic
             clips in synthetic_split.csv exactly once (stops otherwise); T@0.5/1/2 (accs_scorer.temporal)
             and within_2s over all clips.
  Seeds:     runs with the same config hash and different seeds: mean +- sample std (ddof=1) of the
             pooled values, with n stated.
Writes results/classifier_summary.csv and results/classifier_summary.md.

Usage:
    .venv-ml\\Scripts\\python.exe aggregate_runs.py RUN_NAME [RUN_NAME ...] [--overwrite]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from accs_scorer import temporal  # noqa: E402

RUNS_DIR = REPO_ROOT / "runs"
SPLIT_CSV = REPO_ROOT / "synthetic_split.csv"
MANIFEST = REPO_ROOT / "split_manifest.json"
OUT_CSV = REPO_ROOT / "results" / "classifier_summary.csv"
OUT_MD = REPO_ROOT / "results" / "classifier_summary.md"
FOLDS = (0, 1, 2)
SIGMAS_T = (0.5, 1.0, 2.0)
METRICS = ("T_0.5", "T_1", "T_2", "within_2s")


def fail(msg: str) -> None:
    sys.exit(f"ERROR: {msg}")


def all_clips() -> list[str]:
    """All synthetic clips of the split file, after checking its hash against split_manifest.json."""
    manifest = json.loads(MANIFEST.read_text(encoding="ascii"))
    if hashlib.sha256(SPLIT_CSV.read_bytes()).hexdigest() != manifest.get("split_sha256"):
        fail("synthetic_split.csv SHA-256 differs from split_sha256 in split_manifest.json")
    clips = pd.read_csv(SPLIT_CSV)["rgb_path"]
    if clips.duplicated().any():
        fail("synthetic_split.csv has duplicated rgb_path values")
    return sorted(clips)


def scores(pred: pd.DataFrame) -> dict[str, float]:
    out = {f"T_{s:g}": float(temporal(pred["t_hat"], pred["t_star"], s)) for s in SIGMAS_T}
    out["within_2s"] = float(((pred["t_hat"] - pred["t_star"]).abs() <= 2.0).mean())
    return out


def summarise_run(name: str, clips: list[str]) -> tuple[dict, list[dict], dict]:
    """Returns (run info, per-fold rows, pooled row)."""
    infos, fold_rows, preds = [], [], []
    for k in FOLDS:
        d = RUNS_DIR / name / f"fold_{k}"
        for f in ("config.json", "metrics.csv", "val_clip_predictions_best.csv"):
            if not (d / f).is_file():
                fail(f"{name}: missing {d / f}")
        info = json.loads((d / "config.json").read_text(encoding="utf-8"))
        if info.get("smoke_test"):
            fail(f"{name}: fold {k} is a smoke-test run; smoke tests are not aggregated")
        if info.get("fold") != k:
            fail(f"{name}: {d / 'config.json'} says fold {info.get('fold')}")
        infos.append(info)
        m = pd.read_csv(d / "metrics.csv")
        best = m[m["best"].astype(str) == "True"]
        if best.empty:
            fail(f"{name}: fold {k} has no best epoch in metrics.csv")
        b = best.iloc[-1]
        fold_rows.append({"run": name, "level": f"fold {k}", "best_epoch": int(b["epoch"]),
                          "n_clips": None, "T_0.5": b["val_T_0.5"], "T_1": b["val_T_1"], "T_2": b["val_T_2"],
                          "within_2s": b["val_within_2s"] if "val_within_2s" in b else np.nan})
        p = pd.read_csv(d / "val_clip_predictions_best.csv")
        fold_rows[-1]["n_clips"] = len(p)
        preds.append(p.assign(fold=k))
    for key in ("config_hash", "seed", "data_hashes"):
        if any(i.get(key) != infos[0].get(key) for i in infos):
            fail(f"{name}: {key} differs between folds")
    pooled = pd.concat(preds, ignore_index=True)
    dup = pooled["rgb_path"][pooled["rgb_path"].duplicated()]
    if len(dup):
        fail(f"{name}: {dup.nunique()} clips predicted in more than one fold, e.g. {dup.iloc[0]}")
    missing = sorted(set(clips) - set(pooled["rgb_path"]))
    extra = sorted(set(pooled["rgb_path"]) - set(clips))
    if missing or extra:
        fail(f"{name}: {len(missing)} clips missing (e.g. {missing[:2]}), {len(extra)} unknown (e.g. {extra[:2]})")
    pooled_row = {"run": name, "level": "pooled out-of-fold", "best_epoch": None, "n_clips": len(pooled),
                  **scores(pooled)}
    run = {"run": name, "config_file": Path(infos[0].get("config_file", "")).name,
           "config_hash": infos[0]["config_hash"], "seed": infos[0]["seed"]}
    return run, fold_rows, pooled_row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="run names under runs/")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if (OUT_CSV.exists() or OUT_MD.exists()) and not args.overwrite:
        fail(f"{OUT_CSV.name} / {OUT_MD.name} already exist; use --overwrite")
    if len(set(args.runs)) != len(args.runs):
        fail("a run name is given twice")

    clips = all_clips()
    runs, rows = [], []
    for name in args.runs:
        run, fold_rows, pooled_row = summarise_run(name, clips)
        runs.append({**run, **{k: pooled_row[k] for k in METRICS}})
        for r in fold_rows + [pooled_row]:
            rows.append({**{k: run[k] for k in ("config_file", "config_hash", "seed")}, **r})

    seed_rows = []
    for h, g in pd.DataFrame(runs).groupby("config_hash", sort=False):
        if g["seed"].nunique() < 2:
            continue
        if g["seed"].duplicated().any():
            fail(f"config {h[:12]}: the same seed appears in more than one run")
        row = {"config_file": g["config_file"].iloc[0], "config_hash": h, "seed": "", "run": " + ".join(g["run"]),
               "level": f"seeds mean (n={len(g)})", "best_epoch": None, "n_clips": None}
        sd = dict(row, level=f"seeds sample std (ddof=1, n={len(g)})")
        for k in METRICS:
            row[k], sd[k] = float(g[k].mean()), float(g[k].std(ddof=1))
        seed_rows += [row, sd]
    table = pd.DataFrame(rows + seed_rows)
    table[["best_epoch", "n_clips"]] = table[["best_epoch", "n_clips"]].astype("Int64")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)

    md = ["# Classifier summary (synthetic cross-validation)", "",
          f"Best epoch per fold = last epoch marked best (val T@1). Pooled = best-epoch predictions of the three "
          f"folds, {len(clips)} clips, each exactly once. Validation only; no real data.", ""]
    show = table.copy()
    for k in METRICS:
        show[k] = show[k].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    show["config_hash"] = show["config_hash"].str[:12]
    cols = ["run", "config_file", "config_hash", "seed", "level", "best_epoch", "n_clips", *METRICS]
    md.append("| " + " | ".join(cols) + " |")
    md.append("|" + "---|" * len(cols))
    for _, r in show[cols].iterrows():
        md.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    if seed_rows:
        md += ["", "## Across seeds", ""]
        for i in range(0, len(seed_rows), 2):
            m, s = seed_rows[i], seed_rows[i + 1]
            md.append(f"- {m['config_file']} ({m['config_hash'][:12]}), {m['level'].split('(')[1][:-1]}: "
                      + "; ".join(f"{k} {m[k]:.4f} +- {s[k]:.4f}" for k in METRICS))
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(table.to_string(index=False))
    print(f"wrote {OUT_CSV} and {OUT_MD}")


if __name__ == "__main__":
    main()
