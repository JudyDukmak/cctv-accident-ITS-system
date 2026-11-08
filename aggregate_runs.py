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
  Folds of a run must agree in seed, window-labels and split hashes. Two documented deviations are possible,
  each only on request (fold_consistency): config differences limited to micro_batch / grad_accum with equal
  micro_batch x grad_accum (--allow-config-diff), and a frame cache rebuilt on another machine whose manifest
  differs only in run metadata (--run-cache-manifest, cache_manifest_check.py). They are listed under
  "Documented deviations".
Writes results/classifier_summary.csv and results/classifier_summary.md.

Usage:
    .venv-ml\\Scripts\\python.exe aggregate_runs.py RUN_NAME [RUN_NAME ...] [--overwrite]
        [--allow-config-diff micro_batch,grad_accum] [--run-cache-manifest RUN:FOLD=PATH ...]
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
from cache_manifest_check import check_cache_manifest  # noqa: E402

RUNS_DIR = REPO_ROOT / "runs"
SPLIT_CSV = REPO_ROOT / "synthetic_split.csv"
MANIFEST = REPO_ROOT / "split_manifest.json"
CACHE_MANIFEST = REPO_ROOT / "cache" / "frames224_v1" / "cache_manifest.json"
BATCH_KEYS = ("micro_batch", "grad_accum")  # the only config keys --allow-config-diff accepts
ENV_FIELDS = ("gpu", "python", "torch", "transformers", "cuda", "git_commit")  # recorded, not required equal
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


def fold_consistency(name: str, infos: list[dict], allow: set[str], run_manifests: dict[int, Path]) -> list[str]:
    """Check that the folds of a run (config.json contents, fold order) are comparable; returns the documented
    deviations (empty when config, seed and data hashes are equal, as before). Seed, window labels and split
    hashes must be equal. Config: equal, or (opt-in) differing only in keys of `allow` (a subset of BATCH_KEYS)
    with equal micro_batch x grad_accum. Cache manifest: equal, or each fold whose hash differs from the current
    cache manifest passes cache_manifest_check with its own manifest file (run_manifests[fold])."""
    ref = infos[0]
    if any(i.get("seed") != ref.get("seed") for i in infos):
        fail(f"{name}: seed differs between folds")
    if any(i["data_hashes"].keys() != ref["data_hashes"].keys() for i in infos):
        fail(f"{name}: data_hashes keys differ between folds")
    for h in ("window_labels_sha256", "split_sha256"):
        if any(i["data_hashes"][h] != ref["data_hashes"][h] for i in infos):
            fail(f"{name}: {h} differs between folds")
    dev = []
    if any(i["config_hash"] != ref["config_hash"] for i in infos):
        missing = object()
        keys = sorted({k for i in infos for k in set(i["config"]) | set(ref["config"])
                       if i["config"].get(k, missing) != ref["config"].get(k, missing)})
        if not keys:
            fail(f"{name}: config_hash differs between folds but the stored configs are equal")
        if set(keys) - allow:
            fail(f"{name}: config differs between folds in {keys}; only {sorted(allow) or 'nothing'} allowed "
                 "(--allow-config-diff accepts " + ",".join(BATCH_KEYS) + ")")
        eff = {i["fold"]: i["config"]["micro_batch"] * i["config"]["grad_accum"] for i in infos}
        if len(set(eff.values())) > 1:
            fail(f"{name}: micro_batch x grad_accum differs between folds: {eff}")
        for i in infos[1:]:
            if i["config_hash"] != ref["config_hash"]:
                dev.append(f"fold {i['fold']}: config {Path(i['config_file']).name} (hash {i['config_hash'][:12]}...) "
                           f"differs from fold {ref['fold']} ({Path(ref['config_file']).name}, "
                           f"{ref['config_hash'][:12]}...) only in "
                           + ", ".join(f"{k} {ref['config'][k]} -> {i['config'][k]}" for k in keys)
                           + f"; effective batch micro_batch x grad_accum = {eff[i['fold']]} in every fold")
    cache = {i["fold"]: i["data_hashes"]["cache_manifest_sha256"] for i in infos}
    if len(set(cache.values())) > 1:
        windows = json.loads(MANIFEST.read_text(encoding="ascii")).get("windows_sha256")
        if json.loads(CACHE_MANIFEST.read_text(encoding="ascii")).get("windows_sha256") != windows:
            fail(f"{name}: current cache manifest windows_sha256 differs from split_manifest.json")
        for k, h in cache.items():
            try:
                note = check_cache_manifest(h, CACHE_MANIFEST, run_manifests.get(k))
            except ValueError as e:
                fail(f"{name} fold {k}: {e}")
            if note:
                dev.append(f"fold {k}: {note}")
    for f in ENV_FIELDS:
        vals = [str(i.get(f)) for i in infos]
        if len(set(vals)) > 1:
            dev.append(f"environment {f} (recorded, not checked): "
                       + ", ".join(f"fold {i['fold']} {v[:7] if f == 'git_commit' else v}" for i, v in zip(infos, vals)))
    return dev


def scores(pred: pd.DataFrame) -> dict[str, float]:
    out = {f"T_{s:g}": float(temporal(pred["t_hat"], pred["t_star"], s)) for s in SIGMAS_T}
    out["within_2s"] = float(((pred["t_hat"] - pred["t_star"]).abs() <= 2.0).mean())
    return out


def summarise_run(name: str, clips: list[str], allow: set[str] = frozenset(),
                  run_manifests: dict[int, Path] | None = None) -> tuple[dict, list[dict], dict, list[str]]:
    """Returns (run info, per-fold rows, pooled row, documented deviations)."""
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
        fold_rows.append({"run": name, "level": f"fold {k}", "config_file": Path(info.get("config_file", "")).name,
                          "config_hash": info["config_hash"], "best_epoch": int(b["epoch"]),
                          "best_check": int(b["check_idx"]) if "check_idx" in b and not pd.isna(b["check_idx"]) else None,
                          "n_clips": None, "T_0.5": b["val_T_0.5"], "T_1": b["val_T_1"], "T_2": b["val_T_2"],
                          "within_2s": b["val_within_2s"] if "val_within_2s" in b else np.nan})
        p = pd.read_csv(d / "val_clip_predictions_best.csv")
        fold_rows[-1]["n_clips"] = len(p)
        preds.append(p.assign(fold=k))
    deviations = fold_consistency(name, infos, set(allow), run_manifests or {})
    pooled = pd.concat(preds, ignore_index=True)
    dup = pooled["rgb_path"][pooled["rgb_path"].duplicated()]
    if len(dup):
        fail(f"{name}: {dup.nunique()} clips predicted in more than one fold, e.g. {dup.iloc[0]}")
    missing = sorted(set(clips) - set(pooled["rgb_path"]))
    extra = sorted(set(pooled["rgb_path"]) - set(clips))
    if missing or extra:
        fail(f"{name}: {len(missing)} clips missing (e.g. {missing[:2]}), {len(extra)} unknown (e.g. {extra[:2]})")
    pooled_row = {"run": name, "level": "pooled out-of-fold", "best_epoch": None, "best_check": None,
                  "n_clips": len(pooled),
                  **scores(pooled)}
    run = {"run": name, "config_file": Path(infos[0].get("config_file", "")).name,
           "config_hash": infos[0]["config_hash"], "seed": infos[0]["seed"]}
    return run, fold_rows, pooled_row, deviations


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="run names under runs/")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow-config-diff", default="", metavar="KEYS",
                    help=f"comma-separated config keys allowed to differ between folds (only {','.join(BATCH_KEYS)}, "
                         "and only with equal micro_batch x grad_accum); listed as documented deviations")
    ap.add_argument("--run-cache-manifest", action="append", default=[], metavar="RUN:FOLD=PATH",
                    help="the cache_manifest.json a fold was trained with, if its cache was built on another machine")
    args = ap.parse_args()
    if (OUT_CSV.exists() or OUT_MD.exists()) and not args.overwrite:
        fail(f"{OUT_CSV.name} / {OUT_MD.name} already exist; use --overwrite")
    if len(set(args.runs)) != len(args.runs):
        fail("a run name is given twice")
    allow = {k for k in args.allow_config_diff.split(",") if k}
    if allow - set(BATCH_KEYS):
        fail(f"--allow-config-diff accepts only {','.join(BATCH_KEYS)} (got {sorted(allow)})")
    manifests: dict[str, dict[int, Path]] = {}
    for item in args.run_cache_manifest:
        try:
            spec, path = item.split("=", 1)
            run_name, fold = spec.rsplit(":", 1)
            manifests.setdefault(run_name, {})[int(fold)] = Path(path)
        except ValueError:
            fail(f"--run-cache-manifest {item!r}: expected RUN:FOLD=PATH")
    if set(manifests) - set(args.runs):
        fail(f"--run-cache-manifest names runs not aggregated: {sorted(set(manifests) - set(args.runs))}")

    clips = all_clips()
    runs, rows, deviations = [], [], {}
    for name in args.runs:
        run, fold_rows, pooled_row, deviations[name] = summarise_run(name, clips, allow, manifests.get(name))
        runs.append({**run, **{k: pooled_row[k] for k in METRICS}})
        for r in fold_rows + [pooled_row]:  # fold rows carry their own config; the pooled row the run's (fold 0)
            rows.append({**{k: run[k] for k in ("config_file", "config_hash", "seed")}, **r})

    seed_rows = []
    for h, g in pd.DataFrame(runs).groupby("config_hash", sort=False):
        if g["seed"].nunique() < 2:
            continue
        if g["seed"].duplicated().any():
            fail(f"config {h[:12]}: the same seed appears in more than one run")
        row = {"config_file": g["config_file"].iloc[0], "config_hash": h, "seed": "", "run": " + ".join(g["run"]),
               "level": f"seeds mean (n={len(g)})", "best_epoch": None, "best_check": None, "n_clips": None}
        sd = dict(row, level=f"seeds sample std (ddof=1, n={len(g)})")
        for k in METRICS:
            row[k], sd[k] = float(g[k].mean()), float(g[k].std(ddof=1))
        seed_rows += [row, sd]
    table = pd.DataFrame(rows + seed_rows)
    table[["best_epoch", "best_check", "n_clips"]] = table[["best_epoch", "best_check", "n_clips"]].astype("Int64")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)

    md = ["# Classifier summary (synthetic cross-validation)", "",
          f"Best epoch per fold = last epoch marked best (val T@1). Pooled = best-epoch predictions of the three "
          f"folds, {len(clips)} clips, each exactly once. Validation only; no real data.", ""]
    show = table.copy()
    for k in METRICS:
        show[k] = show[k].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    show["config_hash"] = show["config_hash"].str[:12]
    cols = ["run", "config_file", "config_hash", "seed", "level", "best_epoch", "best_check", "n_clips", *METRICS]
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
    dev_lines = [f"- {name}: {d}" for name, ds in deviations.items() for d in ds]
    if dev_lines:
        md += ["", "## Documented deviations", ""] + dev_lines
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(table.to_string(index=False))
    if dev_lines:
        print("\nDocumented deviations:\n" + "\n".join(dev_lines))
    print(f"wrote {OUT_CSV} and {OUT_MD}")


if __name__ == "__main__":
    main()
