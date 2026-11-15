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

Post-processing of the window scores (peak_t_hat): --smooth W (centred moving average of width W, zero-padded edges
as np.convolve mode "same", = train_classifier.clip_predictions) and --interp none | parabola (sub-window peak:
delta = 0.5 (s-1 - s+1) / (s-1 - 2 s0 + s+1), clipped to [-0.5, 0.5] windows, 0 at clip edges or if the denominator
>= 0; t_hat = t0 + delta * 0.5 s). Defaults (no --smooth, --interp none) use the saved best-check clip predictions,
so existing results do not change; otherwise t_hat is recomputed from val_window_scores_best.csv.
--postproc-grid W,W,...: no summary files; for the first run, pooled out-of-fold metrics of every (W, interp) variant,
the reproduction check of (smooth_window, none) against the saved predictions, the confidence / |error| Spearman
rho, and the selection rule (primary pooled T@1; keep the run's default unless the best variant beats it by
>= --postproc-min-gain); further runs are reference tables only. Output: --postproc-out (report.txt, CSV).

--window-ensemble (real-test path; reads no label): one run, --scores-dir DIR from analyse_predictions.py
--score-windows (DIR/fold_<k>/window_scores_{orig,hflip}.csv) for --ensemble-folds; per fold model the flip-TTA score
of a window = mean of its orig and hflip sigmoid scores (as --tta flip); the ensemble score = mean over the fold models
(one model: that model's TTA score, so the frozen synthetic path is reproduced); then peak_t_hat with --smooth and
--interp (frozen: 3, parabola) and the run's confidence_exclude. Writes --postproc-out/clip_predictions.csv (rgb_path,
t_hat, confidence, delta, n_windows), ensemble_window_scores.csv and ensemble_report.txt. Code check on synthetic
only: --check-against PRED_CSV (saved frozen predictions with t_star) and --check-scores-dir DIR (the window scores
those predictions came from) compare t_hat, T@1, confidence and window scores.

Real test, run once (results/real_test/; the only modes that read real labels, and only in --real-eval):
  --real-lock      writes RUN_ONCE.txt (refuses if it exists): date, git HEAD and modified tracked files, sha256 of the
                   code and real inputs (checked again before every scoring run and evaluation), config hashes and best
                   checks of every fold model, frozen post-processing, the planned stages C1 (C_v2_seed0, C_cctv_seed0)
                   and C2 (C_cctv_seed1, C_cctv_seed2).
  --real-eval C1|C2  once per stage (refuses a second time; C2 only after C1): reads accident_time and the frozen-400
                   paths, checks the naive midpoint reproduces 0.1068 / 0.1896 / 0.2952 on all 2,027 clips, builds the
                   frozen predictions (ensemble_predictions over folds 0, 1, 2; clips without a window: t_hat = duration
                   / 2, confidence 0), and writes stage_<s>/ tables: C1 the ablation table; C2 per-run, final (C_cctv
                   mean +- std over seeds next to C_v2 seed 0 and the naive midpoint), transfer gap, confidence tables;
                   both the GPU seconds and the secondary single-fold-model table. Scopes: all 2027, frozen 400, clips
                   with >= 1 window, early impact (t* < 2.875 s).
  --dry-run        (with --real-eval and another --real-root) FAKE t* = 0.37 x duration; no label is read.
  --real-analysis  (added to the plan 2026-11-15, after both stages; no model run, nothing chosen, RUN_ONCE.txt only
                   read) analysis of the saved predictions in <real root>/predictions/ (sha256 checked against the
                   lock's EVENT lines; stage tables reproduced first). Writes <real root>/analysis/ once: paired
                   bootstrap of C_v2_seed0 - C_cctv_seed0 and C_v2_seed0 - C_cctv mean over seeds 0-2 (source videos =
                   clip stem without _NN resampled, BOOT_N resamples, seed BOOT_SEED, percentile 95% interval) on all
                   2027 and the frozen 400; per-fold synthetic validation T@1 next to the real T@1 of that fold model
                   alone; duration and gain over the naive midpoint for t* < 2.875 s and the rest; confidence tables
                   synthetic vs real per C_cctv seed (each clip once per table) and their mean over seeds, Wilson 95%
                   interval as a secondary last column of the per-seed tables only.

Usage:
    .venv-ml\\Scripts\\python.exe aggregate_runs.py RUN_NAME [RUN_NAME ...] [--overwrite]
        [--allow-config-diff micro_batch,grad_accum] [--run-cache-manifest RUN:FOLD=PATH ...]
        [--smooth W] [--interp none|parabola]
    .venv-ml\\Scripts\\python.exe aggregate_runs.py C_cctv_seed0 C_v2_seed0 --postproc-grid 1,3,5 \\
        --postproc-out results\\diagnostics\\postproc_C_cctv_seed0 --allow-config-diff micro_batch,grad_accum \\
        --run-cache-manifest C_cctv_seed0:1=PATH --run-cache-manifest C_v2_seed0:1=PATH
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from accs_scorer import temporal, temporal_per_clip  # noqa: E402
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


INTERPS = ("none", "parabola")
WINDOW_STRIDE_S = 0.5  # window centres are 0.5 s apart (window labels); checked in peak_t_hat


def peak_t_hat(win: pd.DataFrame, smooth: int, interp: str, exclude: int) -> pd.DataFrame:
    """Per clip t_hat and confidence from window scores (columns rgb_path, t_centre, score). smooth = width of the
    centred moving average (np.convolve mode "same", as train_classifier.clip_predictions); interp "none" = t_centre
    of the argmax (first maximum), "parabola" = sub-window vertex of the smoothed scores around it (see module doc);
    confidence = peak - max smoothed score more than `exclude` windows from the peak. Scores are cast to float32
    first: training wrote float32 scores, so this recovers its exact values from the CSV."""
    if interp not in INTERPS:
        fail(f"interp must be one of {INTERPS}")
    kernel = np.ones(smooth) / smooth
    rows = []
    for path, g in win.groupby("rgb_path", sort=True):
        g = g.sort_values("t_centre")
        t = g["t_centre"].to_numpy(float)
        if len(t) > 1 and not np.allclose(np.diff(t), WINDOW_STRIDE_S):
            fail(f"{path}: window centres are not {WINDOW_STRIDE_S} s apart")
        s = np.convolve(g["score"].to_numpy(np.float32).astype(float), kernel, mode="same")
        p = int(np.argmax(s))
        idx = np.arange(len(s))
        outside = s[np.abs(idx - p) > exclude]
        delta = 0.0
        if interp == "parabola" and 0 < p < len(s) - 1:
            den = s[p - 1] - 2.0 * s[p] + s[p + 1]
            if den < 0:
                delta = float(np.clip(0.5 * (s[p - 1] - s[p + 1]) / den, -0.5, 0.5))
        rows.append({"rgb_path": path, "t_hat": float(t[p] + delta * WINDOW_STRIDE_S),
                     "confidence": float(s[p] - outside.max()) if len(outside) else 0.0, "delta": delta})
    return pd.DataFrame(rows)


TTAS = ("none", "flip")


def window_scores(name: str, fold: int, scores_dir: Path | None, tta: str) -> pd.DataFrame:
    """Best-check window scores of a fold: the run's val_window_scores_best.csv (scores_dir None), or the scores
    written by analyse_predictions.py --score-best in scores_dir/fold_<k>/ (orig; tta "flip" = mean of the orig and
    hflip sigmoid scores per window)."""
    if scores_dir is None:
        if tta != "none":
            fail("--tta flip needs --scores-dir (analyse_predictions.py --score-best --hflip)")
        return pd.read_csv(RUNS_DIR / name / f"fold_{fold}" / "val_window_scores_best.csv")
    d = scores_dir / f"fold_{fold}"
    orig = pd.read_csv(d / "val_window_scores_orig.csv")
    if tta == "none":
        return orig
    flip = pd.read_csv(d / "val_window_scores_hflip.csv")
    m = orig.merge(flip[["rgb_path", "window_idx", "score"]], on=["rgb_path", "window_idx"], suffixes=("", "_flip"),
                   validate="one_to_one")
    if len(m) != len(orig) or len(m) != len(flip):
        fail(f"{d}: orig and hflip window scores cover different windows")
    m["score"] = (m["score"].astype(np.float32).astype(float) + m["score_flip"].astype(np.float32).astype(float)) / 2
    return m.drop(columns=["score_flip"])


def recomputed_preds(name: str, smooth: int | None, interp: str, scores_dir: Path | None = None,
                     tta: str = "none") -> list[pd.DataFrame]:
    """Per fold: clip predictions recomputed from the window scores (window_scores; t_star from the saved
    predictions); smooth None = the fold's config smooth_window."""
    out = []
    for k in FOLDS:
        d = RUNS_DIR / name / f"fold_{k}"
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))["config"]
        win = window_scores(name, k, scores_dir, tta)
        saved = pd.read_csv(d / "val_clip_predictions_best.csv")
        p = peak_t_hat(win, smooth or cfg["smooth_window"], interp, cfg["confidence_exclude"])
        p = p.merge(saved[["rgb_path", "t_star"]], on="rgb_path", validate="one_to_one")
        if len(p) != len(saved):
            fail(f"{name} fold {k}: window scores and clip predictions cover different clips")
        out.append(p.assign(fold=k))
    return out


def scores(pred: pd.DataFrame) -> dict[str, float]:
    out = {f"T_{s:g}": float(temporal(pred["t_hat"], pred["t_star"], s)) for s in SIGMAS_T}
    out["within_2s"] = float(((pred["t_hat"] - pred["t_star"]).abs() <= 2.0).mean())
    return out


def summarise_run(name: str, clips: list[str], allow: set[str] = frozenset(),
                  run_manifests: dict[int, Path] | None = None, smooth: int | None = None,
                  interp: str = "none", scores_dir: Path | None = None,
                  tta: str = "none") -> tuple[dict, list[dict], dict, list[str]]:
    """Returns (run info, per-fold rows, pooled row, documented deviations). Default post-processing (smooth None,
    interp none): the saved best-check predictions and metrics.csv values; otherwise per-fold and pooled values are
    recomputed from the best-check window scores with peak_t_hat."""
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
    if smooth is not None or interp != "none" or scores_dir is not None or tta != "none":
        preds = recomputed_preds(name, smooth, interp, scores_dir, tta)
        for r, p in zip(fold_rows, preds):
            r.update(scores(p))
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


def variant_metrics(preds: list[pd.DataFrame]) -> dict:
    p = pd.concat(preds, ignore_index=True)
    err = p["t_hat"] - p["t_star"]
    out = {f"pooled T@{s:g}": float(temporal(p["t_hat"], p["t_star"], s)) for s in SIGMAS_T}
    out |= {f"fold {int(f['fold'].iloc[0])} T@1": float(temporal(f["t_hat"], f["t_star"], 1.0)) for f in preds}
    out |= {"within_2s": float((err.abs() <= 2).mean()), "median |err| s": float(err.abs().median()),
            "rho(conf, |err|)": float(p["confidence"].rank().corr(err.abs().rank())),
            "share interpolated": float((p["delta"] != 0).mean()), "n_clips": len(p)}
    return out


def postproc_grid(args: argparse.Namespace, clips: list[str], allow: set[str],
                  manifests: dict[str, dict[int, Path]]) -> None:
    widths = [int(w) for w in args.postproc_grid.split(",")]
    out_dir = args.postproc_out
    if out_dir is None:
        fail("--postproc-grid needs --postproc-out")
    out_dir.mkdir(parents=True, exist_ok=True)
    sel = args.runs[0]
    rule = (f"Selection rule (fixed before the results are printed): on {sel}, pooled out-of-fold T@1 over all clips "
            f"is the primary metric; the variant with the highest pooled T@1 is chosen only if it beats the current "
            f"default (smoothing = the run's smooth_window, no interpolation) by at least {args.postproc_min_gain}; "
            "otherwise the default is kept. T@0.5 is reported separately and does not enter the choice. Reference "
            "runs are not used for the choice.")
    rep = [f"Post-processing of the best-check window scores, synthetic pooled out-of-fold (no training, no GPU, "
           f"no real data); runs: {', '.join(args.runs)}; widths {widths} x interp {list(INTERPS)}", "", rule, ""]
    print(rep[-2], flush=True)
    tables = {}
    for name in args.runs:
        run, fold_rows, pooled_row, _ = summarise_run(name, clips, allow, manifests.get(name))
        infos = [json.loads((RUNS_DIR / name / f"fold_{k}" / "config.json").read_text(encoding="utf-8")) for k in FOLDS]
        base_w = {i["config"]["smooth_window"] for i in infos}
        if len(base_w) != 1:
            fail(f"{name}: smooth_window differs between folds")
        base_w = base_w.pop()
        rows, check = [], []
        for w in widths:
            for interp in INTERPS:
                preds = recomputed_preds(name, w, interp)
                rows.append({"run": name, "smooth": w, "interp": interp,
                             "default": w == base_w and interp == "none", **variant_metrics(preds)})
                if w == base_w and interp == "none":
                    for k, p in zip(FOLDS, preds):
                        saved = pd.read_csv(RUNS_DIR / name / f"fold_{k}" / "val_clip_predictions_best.csv")
                        m = p.merge(saved[["rgb_path", "t_hat", "confidence"]], on="rgb_path", suffixes=("", "_saved"))
                        check.append(f"fold {k}: t_hat equal to the saved best-check predictions in "
                                     f"{int((m['t_hat'] == m['t_hat_saved']).sum())}/{len(m)} clips; confidence max "
                                     f"|diff| {float((m['confidence'] - m['confidence_saved']).abs().max()):.1e}")
                    v = rows[-1]
                    check.append("pooled T@0.5 / 1 / 2 recomputed " + " / ".join(f"{v[f'pooled T@{s:g}']:.4f}"
                                                                                for s in SIGMAS_T)
                                 + " vs aggregate (saved predictions) " + " / ".join(f"{pooled_row[f'T_{s:g}']:.4f}"
                                                                                     for s in SIGMAS_T)
                                 + "; max |diff| " + f"{max(abs(v[f'pooled T@{s:g}'] - pooled_row[f'T_{s:g}']) for s in SIGMAS_T):.1e}")
        t = pd.DataFrame(rows)
        tables[name] = t
        role = "SELECTION" if name == sel else "reference only (not used for the choice)"
        rep += [f"=== {name} ({role}) ===", "reproduction check of the default (smooth "
                f"{base_w}, none):"] + [f"  {c}" for c in check] + ["",
                t.drop(columns=["run"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"), ""]
        if name == sel:
            base = t[t["default"]].iloc[0]
            best = t.sort_values(["pooled T@1", "smooth"], ascending=[False, True]).iloc[0]
            gain = float(best["pooled T@1"] - base["pooled T@1"])
            keep = gain < args.postproc_min_gain
            choice = base if keep else best
            rep += [f"best pooled T@1: smooth {int(best['smooth'])}, {best['interp']} = {best['pooled T@1']:.4f}; default "
                    f"smooth {int(base['smooth'])}, none = {base['pooled T@1']:.4f}; gain {gain:+.4f} "
                    f"{'<' if keep else '>='} {args.postproc_min_gain}",
                    f"CHOICE: smooth {int(choice['smooth'])}, interp {choice['interp']} "
                    f"({'default kept' if keep else 'new variant'}); pooled T@0.5 / 1 / 2 = {choice['pooled T@0.5']:.4f} "
                    f"/ {choice['pooled T@1']:.4f} / {choice['pooled T@2']:.4f}", ""]
    pd.concat(tables.values(), ignore_index=True).to_csv(out_dir / "postproc_variants.csv", index=False)
    text = "\n".join(rep) + "\n"
    (out_dir / "report.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out_dir / 'report.txt'} and postproc_variants.csv")


def tta_study(args: argparse.Namespace, clips: list[str], allow: set[str],
              manifests: dict[str, dict[int, Path]]) -> None:
    """No TTA vs flip TTA for the first run, both arms from --scores-dir (same GPU), post-processing --smooth /
    --interp; rule: adopt flip only if pooled T@1 improves by >= --tta-min-gain."""
    name = args.runs[0]
    if args.scores_dir is None or args.postproc_out is None or args.smooth is None:
        fail("--tta-study needs --scores-dir, --postproc-out and an explicit --smooth (and --interp)")
    out_dir = args.postproc_out
    out_dir.mkdir(parents=True, exist_ok=True)
    rule = (f"Rule (fixed before the results are printed): with post-processing smooth {args.smooth} + interp "
            f"{args.interp}, flip TTA is adopted only if pooled out-of-fold T@1 of {name} over all clips improves by "
            f">= {args.tta_min_gain} over no TTA (both arms scored on the same GPU); T@0.5 and T@2 are reported but "
            "do not decide.")
    print(rule, flush=True)
    summarise_run(name, clips, allow, manifests.get(name))  # fold consistency and coverage checks
    split = pd.read_csv(SPLIT_CSV, usecols=["rgb_path", "weather", "type"])
    arms = {"no TTA": recomputed_preds(name, args.smooth, args.interp, args.scores_dir, "none"),
            "flip TTA": recomputed_preds(name, args.smooth, args.interp, args.scores_dir, "flip"),
            "saved best CSV (training GPU), no TTA": recomputed_preds(name, args.smooth, args.interp)}
    rows, pooled = [], {}
    for arm, preds in arms.items():
        for scope, p in [(f"fold {int(f['fold'].iloc[0])}", f) for f in preds] + \
                        [("pooled", pd.concat(preds, ignore_index=True))]:
            err = p["t_hat"] - p["t_star"]
            r = {"arm": arm, "scope": scope, "n_clips": len(p),
                 **{f"T@{s:g}": float(temporal(p["t_hat"], p["t_star"], s)) for s in SIGMAS_T},
                 "within_2s": float((err.abs() <= 2).mean()), "median |err| s": float(err.abs().median()),
                 "rho(conf, |err|)": float(p["confidence"].rank().corr(err.abs().rank()))}
            rows.append(r)
            if scope == "pooled":
                pooled[arm] = p.merge(split, on="rgb_path", validate="one_to_one")
    t = pd.DataFrame(rows)
    if any(len(p) != len(clips) for p in pooled.values()):
        fail("an arm does not cover all clips")
    grp = []
    for by in ("weather", "type"):
        for key in sorted(pooled["no TTA"][by].unique()):
            a, b = (pooled[x][pooled[x][by] == key] for x in ("no TTA", "flip TTA"))
            ta, tb = float(temporal(a["t_hat"], a["t_star"], 1.0)), float(temporal(b["t_hat"], b["t_star"], 1.0))
            grp.append({"by": by, "group": key, "n": len(a), "T@1 no TTA": ta, "T@1 flip TTA": tb, "dT@1": tb - ta})
    g = pd.DataFrame(grp)
    p0, p1 = (t[(t["arm"] == x) & (t["scope"] == "pooled")].iloc[0] for x in ("no TTA", "flip TTA"))
    gain = float(p1["T@1"] - p0["T@1"])
    adopt = gain >= args.tta_min_gain
    decision = (f"DECISION: {'adopt' if adopt else 'do not adopt'} flip TTA: pooled T@1 {p0['T@1']:.4f} -> "
                f"{p1['T@1']:.4f} ({gain:+.4f} {'>=' if adopt else '<'} {args.tta_min_gain}); T@0.5 {p0['T@0.5']:.4f} -> "
                f"{p1['T@0.5']:.4f}, T@2 {p0['T@2']:.4f} -> {p1['T@2']:.4f}")
    reports = sorted(args.scores_dir.glob("fold_*/scoring_report.txt"))
    text = "\n".join([f"Flip test-time augmentation, {name}, synthetic pooled out-of-fold, post-processing smooth "
                      f"{args.smooth} + {args.interp}; scores from {args.scores_dir}", "", rule, "",
                      "scoring reports (reproduction check and classifier time):"]
                     + [ln for r in reports for ln in ["  " + x for x in r.read_text(encoding="utf-8").splitlines()]]
                     + ["", "no TTA vs flip TTA (and the saved training-GPU scores for reference):",
                        t.to_string(index=False, float_format=lambda v: f"{v:.4f}"), "",
                        "T@1 by weather and collision type (pooled):",
                        g.to_string(index=False, float_format=lambda v: f"{v:.4f}"), "", decision]) + "\n"
    (out_dir / "tta_report.txt").write_text(text, encoding="utf-8")
    t.to_csv(out_dir / "tta_table.csv", index=False)
    g.to_csv(out_dir / "tta_by_group.csv", index=False)
    for arm in ("no TTA", "flip TTA"):
        pooled[arm].to_csv(out_dir / f"tta_clip_predictions_{arm.split()[0].lower()}.csv", index=False)
    print(text)
    print(f"wrote {out_dir / 'tta_report.txt'}")


ENSEMBLE_KEYS = ["rgb_path", "window_idx", "t_centre"]


def ensemble_predictions(name: str, scores_dir: Path, folds: list[int], smooth: int, interp: str
                         ) -> tuple[pd.DataFrame, pd.DataFrame, int, list[str]]:
    """Flip TTA per fold model, mean over the fold models, peak_t_hat (see module doc). Returns (clip predictions,
    ensemble window scores, confidence_exclude, scoring-report lines)."""
    excl = {json.loads((RUNS_DIR / name / f"fold_{k}" / "config.json").read_text(encoding="utf-8"))["config"]
            ["confidence_exclude"] for k in folds}
    if len(excl) != 1:
        fail(f"{name}: confidence_exclude differs between folds")
    exclude = excl.pop()
    keys, cols, lines = None, {}, []
    for k in folds:
        d = scores_dir / f"fold_{k}"
        o = pd.read_csv(d / "window_scores_orig.csv")
        f = pd.read_csv(d / "window_scores_hflip.csv")
        if not (o[ENSEMBLE_KEYS].equals(f[ENSEMBLE_KEYS])):
            fail(f"{d}: orig and hflip window scores cover different windows")
        if keys is None:
            keys = o[ENSEMBLE_KEYS]
        elif not o[ENSEMBLE_KEYS].equals(keys):
            fail(f"{d}: windows differ from fold {folds[0]}")
        # flip TTA of one model, exactly as window_scores(..., "flip")
        cols[f"score_fold{k}"] = (o["score"].astype(np.float32).astype(float)
                                  + f["score"].astype(np.float32).astype(float)) / 2
        rep = d / "scoring_report.txt"
        lines += ["  " + x for x in rep.read_text(encoding="utf-8").splitlines()] if rep.is_file() else []
    ens = keys.assign(**cols)
    ens["score"] = ens[list(cols)].mean(axis=1) if len(cols) > 1 else ens[next(iter(cols))]
    pred = peak_t_hat(ens[["rgb_path", "t_centre", "score"]], smooth, interp, exclude)
    pred = pred.merge(ens.groupby("rgb_path").size().rename("n_windows").reset_index(), on="rgb_path",
                      validate="one_to_one")
    return pred, ens, exclude, lines


def window_ensemble(args: argparse.Namespace) -> None:
    """--window-ensemble: see module doc. No label is read (except t_star of the synthetic --check-against file)."""
    if len(args.runs) != 1:
        fail("--window-ensemble takes exactly one run")
    name = args.runs[0]
    if args.scores_dir is None or args.postproc_out is None or args.smooth is None or args.tta != "flip":
        fail("--window-ensemble needs --scores-dir, --postproc-out, --smooth, --interp and --tta flip")
    folds = [int(f) for f in args.ensemble_folds.split(",")]
    if not folds or set(folds) - set(FOLDS) or len(set(folds)) != len(folds):
        fail(f"--ensemble-folds {args.ensemble_folds!r}: distinct values from 0, 1, 2")
    pred, ens, exclude, lines = ensemble_predictions(name, args.scores_dir, folds, args.smooth, args.interp)
    out = args.postproc_out
    out.mkdir(parents=True, exist_ok=True)
    head = [f"Window ensemble: run {name}, fold models {folds}; scores {args.scores_dir}; flip TTA per model (mean of "
            f"orig and hflip), mean over models, smoothing {args.smooth} + {args.interp}, confidence_exclude {exclude}; "
            f"{len(ens)} windows, {len(pred)} clips", "scoring reports:"] + lines
    checks = []
    if args.check_scores_dir is not None:
        for k in folds:
            for nm, ref_nm in (("orig", "val_window_scores_orig.csv"), ("hflip", "val_window_scores_hflip.csv")):
                mine = pd.read_csv(args.scores_dir / f"fold_{k}" / f"window_scores_{nm}.csv")
                ref = pd.read_csv(args.check_scores_dir / f"fold_{k}" / ref_nm)
                m = mine.merge(ref[["rgb_path", "window_idx", "t_centre", "score"]], on=["rgb_path", "window_idx"],
                               suffixes=("", "_ref"), validate="one_to_one")
                if len(m) != len(mine) or len(m) != len(ref):
                    fail(f"fold {k} {nm}: windows differ from {args.check_scores_dir}")
                checks.append(f"fold {k} {nm} window scores vs {args.check_scores_dir}: {len(m)} windows, t_centre max "
                              f"|diff| {float((m['t_centre'] - m['t_centre_ref']).abs().max()):.1e}; score max |diff| "
                              f"{float((m['score'] - m['score_ref']).abs().max()):.2e}, "
                              f"{int((m['score'].astype(np.float32) != m['score_ref'].astype(np.float32)).sum())} differ")
    if args.check_against is not None:
        ref = pd.read_csv(args.check_against)
        ref = ref[ref["rgb_path"].isin(set(pred["rgb_path"]))]
        m = pred.merge(ref[["rgb_path", "t_hat", "confidence", "t_star"]], on="rgb_path", suffixes=("", "_ref"),
                       validate="one_to_one")
        if len(m) != len(pred):
            fail(f"{args.check_against} does not cover all {len(pred)} clips")
        t_new, t_ref = (float(temporal(m[c], m["t_star"], 1.0)) for c in ("t_hat", "t_hat_ref"))
        checks.append(f"clip predictions vs {args.check_against}: t_hat equal in {int((m['t_hat'] == m['t_hat_ref']).sum())}"
                      f"/{len(m)} clips (max |diff| {float((m['t_hat'] - m['t_hat_ref']).abs().max()):.1e} s); T@1 "
                      f"{t_new:.4f} vs {t_ref:.4f} (diff {t_new - t_ref:+.4f}); confidence max |diff| "
                      f"{float((m['confidence'] - m['confidence_ref']).abs().max()):.1e}")
    text = "\n".join(head + ([""] + ["CHECKS:"] + [f"  {c}" for c in checks] if checks else [])) + "\n"
    pred.to_csv(out / "clip_predictions.csv", index=False)
    ens.to_csv(out / "ensemble_window_scores.csv", index=False)
    (out / "ensemble_report.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out / 'clip_predictions.csv'}, ensemble_window_scores.csv, ensemble_report.txt")


# --------------------------------------------------------------------------- #
# Real test, run once (--real-lock, --real-eval STAGE)
# --------------------------------------------------------------------------- #
REAL_DIR = REPO_ROOT / "results" / "real_test"
LOCK_NAME = "RUN_ONCE.txt"
STAGES = {"C1": ("C_v2_seed0", "C_cctv_seed0"), "C2": ("C_cctv_seed1", "C_cctv_seed2")}
C_CCTV = ("C_cctv_seed0", "C_cctv_seed1", "C_cctv_seed2")
REAL_META = REPO_ROOT / "ACCIDENT" / "dataset" / "metadata-real.csv"
REAL_TEST_CSV = REPO_ROOT / "real_test_set.csv"
REAL_TEST_MANIFEST = REPO_ROOT / "real_test_set_manifest.json"
REAL_CACHE_MANIFEST = REPO_ROOT / "cache" / "frames224_real_v1" / "cache_manifest.json"
REAL_WINDOWS = REPO_ROOT / "cache" / "frames224_real_v1" / "real_windows.csv"
LOCKED_FILES = ("aggregate_runs.py", "analyse_predictions.py", "build_frame_cache.py", "train_classifier.py",
                "accs_scorer.py", "make_windows.py", "cache_manifest_check.py")
FROZEN = {"tta": "flip (mean of orig and hflip sigmoid scores per fold model)", "fold_models": [0, 1, 2],
          "ensemble": "mean of the fold models' TTA window scores", "smooth": 3, "interp": "parabola",
          "fallback": "clips without a window: t_hat = duration / 2, confidence 0 (decision 2026-11-15)"}
EARLY_T = 2.875  # early-impact clips: t* < 2.875 s
NAIVE_PAPER = {"T_0.5": 0.1068, "T_1": 0.1896, "T_2": 0.2952}  # results/real_check/naive_validation.txt
DIAG = REPO_ROOT / "results" / "diagnostics"
SYN_TTA = {"C_v2_seed0": DIAG / "C_v2_seed0_frozen" / "frozen" / "tta_table.csv",
           **{f"C_cctv_seed{s}": DIAG / "C_cctv_seeds012" / f"seed{s}_frozen" / "tta_table.csv" for s in (0, 1, 2)}}
TAU_DIR = DIAG / "tau_evidence"
WILSON_Z = 1.959963984540054


def sha_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson interval; the same formula as confidence_vs_error.wilson (copied so this script stays torch-free)."""
    if n == 0:
        return float("nan"), float("nan")
    p, z2 = k / n, WILSON_Z ** 2
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = WILSON_Z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return float(centre - half), float(centre + half)


def utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_lock(root: Path) -> tuple[dict, list[str]]:
    p = root / LOCK_NAME
    if not p.is_file():
        fail(f"{p} does not exist; create it with --real-lock first")
    lines = p.read_text(encoding="utf-8").splitlines()
    js = [ln for ln in lines if ln.startswith("LOCK_JSON ")]
    if len(js) != 1:
        fail(f"{p}: expected one LOCK_JSON line")
    return json.loads(js[0][len("LOCK_JSON "):]), [ln for ln in lines if ln.startswith("EVENT ")]


def append_event(root: Path, text: str) -> None:
    with open(root / LOCK_NAME, "a", encoding="utf-8") as fh:
        fh.write(f"EVENT {utc_now()} {text}\n")


def check_locked_inputs(lock: dict) -> None:
    """Code and real inputs must be exactly as when the lock was written (nothing changes between stages)."""
    changed = [f for f, h in lock["file_sha256"].items() if sha_of(REPO_ROOT / f) != h]
    if changed:
        fail(f"files changed since {LOCK_NAME} was written: {changed}")
    for key, path in (("real_windows_sha256", REAL_WINDOWS), ("real_cache_manifest_sha256", REAL_CACHE_MANIFEST),
                      ("metadata_real_sha256", REAL_META), ("real_test_set_sha256", REAL_TEST_CSV)):
        if sha_of(path) != lock[key]:
            fail(f"{path.name} changed since {LOCK_NAME} was written")


def stage_of(run: str) -> str:
    for s, runs in STAGES.items():
        if run in runs:
            return s
    fail(f"{run} is in no planned stage {STAGES}")


def stage_done(events: list[str], stage: str) -> bool:
    return any(f" STAGE {stage} EVALUATED" in e for e in events)


def guard_real_scoring(run: str, root: Path) -> None:
    """Called by analyse_predictions.py --score-windows real before any GPU work."""
    lock, events = read_lock(root)
    check_locked_inputs(lock)
    s = stage_of(run)
    if stage_done(events, s):
        fail(f"stage {s} was already evaluated; refusing to score {run} again")
    if s == "C2" and not stage_done(events, "C1"):
        fail("stage C2 starts only after stage C1 has been evaluated")
    append_event(root, f"SCORING {run} (stage {s}) started")


def real_lock(root: Path) -> None:
    """--real-lock: write RUN_ONCE.txt (refuses if it exists)."""
    import subprocess
    p = root / LOCK_NAME
    if p.exists():
        fail(f"{p} exists; the real test runs once")
    man = json.loads(REAL_CACHE_MANIFEST.read_text(encoding="ascii"))
    if man.get("complete") is not True or man.get("real_windows_sha256") != sha_of(REAL_WINDOWS):
        fail("real frame cache incomplete or real_windows.csv differs from its manifest")
    tm = json.loads(REAL_TEST_MANIFEST.read_text(encoding="utf-8"))
    if sha_of(REAL_META) != tm["source_csv_sha256"] or sha_of(REAL_TEST_CSV) != tm["output_sha256"]:
        fail("metadata-real.csv or real_test_set.csv differs from real_test_set_manifest.json")
    git = lambda *a: subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    head = git("rev-parse", "HEAD")
    dirty = [ln for ln in git("status", "--porcelain", "--untracked-files=no").splitlines() if ln.strip()]
    runs = {}
    for stage, names in STAGES.items():
        for name in names:
            for k in FOLDS:
                d = RUNS_DIR / name / f"fold_{k}"
                info = json.loads((d / "config.json").read_text(encoding="utf-8"))
                m = pd.read_csv(d / "metrics.csv")
                b = m[m["best"].astype(str) == "True"].iloc[-1]
                runs[f"{name}/fold_{k}"] = {"stage": stage, "config_file": Path(info["config_file"]).name,
                                            "config_hash": info["config_hash"], "run_hash": info["run_hash"],
                                            "seed": info["seed"], "best_check": int(b["check_idx"]),
                                            "best_global_step": int(b["global_step"]),
                                            "best_pt_sha256": sha_of(d / "checkpoints" / "best.pt"),
                                            "trained_gpu": info.get("gpu")}
    lock = {"created": utc_now(), "git_head": head, "git_tracked_modified_vs_head": dirty,
            "file_sha256": {f: sha_of(REPO_ROOT / f) for f in LOCKED_FILES},
            "real_windows_sha256": sha_of(REAL_WINDOWS), "real_cache_manifest_sha256": sha_of(REAL_CACHE_MANIFEST),
            "metadata_real_sha256": sha_of(REAL_META), "real_test_set_sha256": sha_of(REAL_TEST_CSV),
            "frozen_post_processing": FROZEN, "stages": STAGES, "runs": runs}
    text = [f"RUN ONCE: real test of the window classifier (no VLM, no tau). Created {lock['created']}.",
            f"git HEAD {head}; tracked files modified vs HEAD: {', '.join(dirty) if dirty else 'none'}",
            "code / input sha256 (checked before every scoring run and evaluation; any change stops the run):"]
    text += [f"  {f} {h}" for f, h in lock["file_sha256"].items()]
    text += [f"  real_windows.csv {lock['real_windows_sha256']}", f"  metadata-real.csv {lock['metadata_real_sha256']}",
             f"  real_test_set.csv {lock['real_test_set_sha256']}",
             "frozen post-processing: " + "; ".join(f"{k} {v}" for k, v in FROZEN.items()),
             "planned stages (nothing may change between them; no setting depends on stage C1):",
             f"  C1: {', '.join(STAGES['C1'])} on all 2,027 real clips -> ablation table; then C2 without waiting",
             f"  C2: {', '.join(STAGES['C2'])} -> C_cctv mean +- std over seeds 0-2 next to C_v2 seed 0",
             "runs (config hash, best check):"]
    text += [f"  {r}: {v['config_file']} ({v['config_hash'][:8]}), seed {v['seed']}, best check {v['best_check']} "
             f"(update {v['best_global_step']})" for r, v in runs.items()]
    text += ["Each stage is evaluated at most once (EVENT ... STAGE <s> EVALUATED below); scoring files of a stage may "
             "be resumed after an interruption only before that stage is evaluated (no label is read while scoring).",
             "LOCK_JSON " + json.dumps(lock, sort_keys=True)]
    root.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(text) + "\n", encoding="utf-8")
    append_event(root, "LOCK CREATED")
    print("\n".join(text[:-1]))
    print(f"wrote {p}")


def real_clips(dry: bool) -> pd.DataFrame:
    """The 2,027 real clips with t_star (accident_time), duration and the frozen-400 flag. dry: NO label is read;
    t_star is FAKE (0.37 * duration) to exercise the code."""
    tm = json.loads(REAL_TEST_MANIFEST.read_text(encoding="utf-8"))
    if sha_of(REAL_META) != tm["source_csv_sha256"] or sha_of(REAL_TEST_CSV) != tm["output_sha256"]:
        fail("metadata-real.csv or real_test_set.csv differs from real_test_set_manifest.json")
    m = pd.read_csv(REAL_META, usecols=["path", "duration"] + ([] if dry else ["accident_time"]))
    m = m.rename(columns={"path": "rgb_path", "accident_time": "t_star"})
    if dry:
        m["t_star"] = 0.37 * m["duration"]
    test = set(pd.read_csv(REAL_TEST_CSV, usecols=["path"])["path"])
    m["in_test400"] = m["rgb_path"].isin(test)
    if len(m) != 2027 or int(m["in_test400"].sum()) != 400 or m["t_star"].isna().any():
        fail("expected 2027 real clips, 400 in the frozen subset, no missing accident_time")
    m["early"] = m["t_star"] < EARLY_T
    m["has_window"] = m["rgb_path"].isin(set(pd.read_csv(REAL_WINDOWS, usecols=["rgb_path"])["rgb_path"]))
    return m


def with_fallback(pred: pd.DataFrame, clips: pd.DataFrame) -> pd.DataFrame:
    """Clip predictions on all 2,027 clips; clips without a window get t_hat = duration / 2, confidence 0."""
    extra = set(pred["rgb_path"]) - set(clips["rgb_path"])
    if extra:
        fail(f"{len(extra)} predicted clips are not real clips, e.g. {sorted(extra)[:2]}")
    p = clips.merge(pred[["rgb_path", "t_hat", "confidence", "delta", "n_windows"]], on="rgb_path", how="left",
                    validate="one_to_one")
    fb = p["t_hat"].isna()
    p.loc[fb, "t_hat"] = p.loc[fb, "duration"] / 2
    p.loc[fb, ["confidence", "delta", "n_windows"]] = 0.0
    p["fallback"] = fb
    if (p["fallback"] == p["has_window"]).any():
        fail("clips predicted from windows differ from the clips with windows in real_windows.csv")
    return p


REAL_SCOPES = (("all 2027", lambda p: p["t_star"].notna()), ("frozen 400", lambda p: p["in_test400"]),
               ("all, >= 1 window", lambda p: p["has_window"]),
               ("400, >= 1 window", lambda p: p["in_test400"] & p["has_window"]),
               ("all, early t* < 2.875 s", lambda p: p["early"]),
               ("400, early t* < 2.875 s", lambda p: p["in_test400"] & p["early"]),
               ("all, t* >= 2.875 s", lambda p: ~p["early"]))


def real_metrics(p: pd.DataFrame, predictor: str) -> list[dict]:
    rows = []
    for scope, sel in REAL_SCOPES:
        q = p[sel(p)]
        rows.append({"predictor": predictor, "scope": scope, "n_clips": len(q),
                     "n_fallback": int(q["fallback"].sum()) if "fallback" in q else 0,
                     **({k: float("nan") for k in METRICS} if q.empty else scores(q))})
    return rows


def fmt(t: pd.DataFrame) -> str:
    return t.to_string(index=False, float_format=lambda v: f"{v:.4f}")


def wide(t: pd.DataFrame, scopes: tuple[str, ...]) -> str:
    """One block per scope: predictors x T@0.5 / T@1 / T@2 / within 2 s."""
    out = []
    for s in scopes:
        q = t[t["scope"] == s]
        out += [f"[{s}] n = {int(q['n_clips'].iloc[0])} clips (fallback {int(q['n_fallback'].max())})",
                fmt(q.drop(columns=["scope", "n_clips", "n_fallback"])), ""]
    return "\n".join(out)


def mean_sd(vals: pd.Series) -> str:
    return f"{vals.mean():.4f} +- {vals.std(ddof=1):.4f}"


def run_predictions(run: str, root: Path, clips: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, pd.DataFrame], list[str]]:
    """Frozen ensemble (folds 0, 1, 2) and the single-fold-model predictions (secondary) of a run, with the fallback;
    writes predictions/<run>/ (no label in these files)."""
    sd = root / "scores" / run
    keys = pd.read_csv(REAL_WINDOWS, usecols=ENSEMBLE_KEYS).sort_values(["rgb_path", "window_idx"]).reset_index(drop=True)
    pred, ens, exclude, lines = ensemble_predictions(run, sd, list(FOLDS), FROZEN["smooth"], FROZEN["interp"])
    if not ens[ENSEMBLE_KEYS].reset_index(drop=True).equals(keys):
        fail(f"{run}: scored windows differ from real_windows.csv")
    od = root / "predictions" / run
    od.mkdir(parents=True, exist_ok=True)
    full = with_fallback(pred, clips)
    cols = ["rgb_path", "t_hat", "confidence", "delta", "n_windows", "fallback"]
    full[cols].to_csv(od / "clip_predictions.csv", index=False)
    ens.to_csv(od / "ensemble_window_scores.csv", index=False)
    singles = {}
    for k in FOLDS:
        pk, _, _, _ = ensemble_predictions(run, sd, [k], FROZEN["smooth"], FROZEN["interp"])
        singles[k] = with_fallback(pk, clips)
        singles[k][cols].to_csv(od / f"clip_predictions_fold{k}_model_only.csv", index=False)
    return full, singles, lines


def gpu_seconds(run: str, root: Path) -> dict:
    import re
    tot, gpu, passes = 0.0, set(), 0
    for k in FOLDS:
        txt = (root / "scores" / run / f"fold_{k}" / "scoring_report.txt").read_text(encoding="utf-8")
        gpu |= set(re.findall(r"GPU (.+?); \d+ windows", txt))
        for t in re.findall(r"^(?:orig|hflip): classifier time ([\d.]+) s", txt, flags=re.M):
            tot += float(t)
            passes += 1
    if passes != 6:
        fail(f"{run}: expected 6 timed passes (3 fold models x orig/hflip), found {passes}")
    return {"run": run, "gpu": "; ".join(sorted(gpu)), "passes": passes, "gpu_s_total": tot}


def conf_tables(p: pd.DataFrame, scope: str) -> tuple[list[dict], list[dict]]:
    thresholds = [round(0.9 - 0.1 * i, 1) for i in range(10)]
    bins = [(round(0.1 * i, 1), round(0.1 * (i + 1), 1)) for i in range(9, -1, -1)]
    c, n = p["confidence"], len(p)

    def rows(masks):
        out = []
        for name, mk in masks:
            g = p[mk]
            k = int(((g["t_hat"] - g["t_star"]).abs() <= 2).sum())
            lo, hi = wilson(k, len(g))
            out.append({"scope": scope, "confidence": name, "n_clips": len(g), "coverage": len(g) / n,
                        "n_within_2s": k, "share_within_2s": k / len(g) if len(g) else float("nan"),
                        "wilson95_lo": lo, "wilson95_hi": hi,
                        "T@1": float(temporal(g["t_hat"], g["t_star"], 1.0)) if len(g) else float("nan")})
        return out
    cum = rows([(f">= {t:.1f}", c >= t) for t in thresholds])
    binned = rows([(f"[{lo:.1f}, {hi:.1f}]" if hi == 1.0 else f"[{lo:.1f}, {hi:.1f})",
                    (c >= lo) & ((c <= hi) if hi == 1.0 else (c < hi))) for lo, hi in bins])
    if sum(r["n_clips"] for r in binned) != n:
        fail(f"confidence bins of {scope} do not cover every prediction once")
    return cum, binned


def real_eval(stage: str, root: Path, dry: bool) -> None:
    """--real-eval STAGE: see module doc. Stage C1 evaluates C_v2_seed0 and C_cctv_seed0; stage C2 C_cctv_seed1/2 and
    the final tables (reading the stage-C1 predictions as saved)."""
    lock, events = read_lock(root)
    check_locked_inputs(lock)
    if stage_done(events, stage):
        fail(f"stage {stage} was already evaluated; the real test runs once")
    if stage == "C2" and not stage_done(events, "C1"):
        fail("stage C2 is evaluated only after stage C1")
    for run in STAGES[stage]:
        for k in FOLDS:
            for nm in ("orig", "hflip"):
                if not (root / "scores" / run / f"fold_{k}" / f"window_scores_{nm}.csv").is_file():
                    fail(f"{run} fold {k}: {nm} window scores missing; scoring not finished")
    out = root / f"stage_{stage}"
    out.mkdir(parents=True, exist_ok=True)
    tag = "DRY RUN WITH FAKE t* (0.37 x duration) AND WHATEVER SCORES ARE IN THE FOLDER: NOT A RESULT. " if dry else ""
    append_event(root, f"{tag}STAGE {stage} EVALUATION STARTED (labels read now)")
    clips = real_clips(dry)
    rep = [f"{tag}Real test, stage {stage} ({', '.join(STAGES[stage])}); {utc_now()}; lock {root / LOCK_NAME} "
           f"(git HEAD {lock['git_head'][:7]}); frozen post-processing: "
           + "; ".join(f"{k} {v}" for k, v in FROZEN.items()),
           f"clips {len(clips)} (frozen 400: {int(clips['in_test400'].sum())}); early-impact (t* < {EARLY_T} s): "
           f"{int(clips['early'].sum())} (in the 400: {int((clips['early'] & clips['in_test400']).sum())}); metrics: "
           f"accs_scorer.temporal T@0.5/1/2, within 2 s = |t_hat - t*| <= 2 s", ""]

    # naive midpoint first: the scorer and labels must reproduce the paper's naive baseline on all 2,027 clips
    naive = clips.assign(t_hat=clips["duration"] / 2, confidence=0.0, fallback=False)
    nrows = real_metrics(naive, "naive midpoint (duration / 2)")
    n_all = nrows[0]
    ok = all(round(n_all[k], 4) == v for k, v in NAIVE_PAPER.items())
    rep.append("naive midpoint on all 2027: " + " / ".join(f"{n_all[k]:.4f}" for k in NAIVE_PAPER)
               + f" (T@0.5 / T@1 / T@2); expected 0.1068 / 0.1896 / 0.2952: {'REPRODUCED' if ok else 'NOT REPRODUCED'}")
    if not ok and not dry:
        fail("the naive baseline is not reproduced; stopping before any classifier metric")

    preds, singles, gpu_rows = {}, {}, []
    for run in STAGES[stage]:
        full, sing, lines = run_predictions(run, root, clips)
        preds[run], singles[run] = full, sing
        gpu_rows.append(gpu_seconds(run, root))
        rep += [f"{run}: {int((~full['fallback']).sum())} clips predicted from windows, {int(full['fallback'].sum())} "
                f"fallback (no window)"]
    rows = list(nrows)
    for run, p in preds.items():
        rows += real_metrics(p, run)
    sec = []
    for run, sing in singles.items():
        for k, p in sing.items():
            sec += [r | {"run": run, "fold_model": k} for r in real_metrics(p, f"{run} fold-{k} model only")
                    if r["scope"] in ("all 2027", "frozen 400")]
    sec = pd.DataFrame(sec)
    sec.to_csv(out / "secondary_single_fold_models.csv", index=False)

    if stage == "C1":
        t = pd.DataFrame(rows)
        d = t[t["predictor"] == "C_cctv_seed0"].set_index("scope")[list(METRICS)] - \
            t[t["predictor"] == "C_v2_seed0"].set_index("scope")[list(METRICS)]
        diff = d.reset_index().assign(predictor="C_cctv_seed0 - C_v2_seed0", n_clips=t[t["predictor"] == "C_v2_seed0"]
                                      ["n_clips"].to_numpy(), n_fallback=0)
        t = pd.concat([t, diff[t.columns]], ignore_index=True)
        t.to_csv(out / "ablation_table.csv", index=False)
        rep += ["", "ABLATION 1 ON REAL (like-for-like: same clips, same frozen post-processing, same ensemble of 3 "
                "fold models; the ensemble has no synthetic validation)", wide(t, tuple(s for s, _ in REAL_SCOPES))]
    else:
        prev = {}
        for run in STAGES["C1"]:
            f = root / "predictions" / run / "clip_predictions.csv"
            h = [e for e in events if f"PREDICTIONS {run} sha256 " in e]
            if not h or sha_of(f) != h[-1].split()[-1]:
                fail(f"{f} differs from the stage-C1 record")
            prev[run] = with_fallback(pd.read_csv(f).query("not fallback")[["rgb_path", "t_hat", "confidence", "delta",
                                                                             "n_windows"]], clips)
        allp = {**prev, **preds}
        rows = list(nrows) + [r for run in ("C_v2_seed0",) + C_CCTV for r in real_metrics(allp[run], run)]
        t = pd.DataFrame(rows)
        t.to_csv(out / "per_run_table.csv", index=False)
        fin = []
        for scope, _ in REAL_SCOPES:
            q = t[t["scope"] == scope].set_index("predictor")
            fin.append({"scope": scope, "n_clips": int(q["n_clips"].iloc[0]),
                        **{f"C_cctv seeds 0-2 {m}": mean_sd(q.loc[list(C_CCTV), m]) for m in METRICS},
                        **{f"C_v2 seed 0 {m}": f"{q.loc['C_v2_seed0', m]:.4f}" for m in METRICS},
                        **{f"naive {m}": f"{q.loc['naive midpoint (duration / 2)', m]:.4f}" for m in METRICS}})
        fin = pd.DataFrame(fin)
        fin.to_csv(out / "final_table.csv", index=False)
        # transfer gap: synthetic pooled out-of-fold (frozen post-processing; one model per clip) vs real
        gap = []
        for run in ("C_v2_seed0",) + C_CCTV:
            s = pd.read_csv(SYN_TTA[run])
            s = s[(s["arm"] == "flip TTA") & (s["scope"] == "pooled")].iloc[0]
            for m, sm in zip(METRICS, ("T@0.5", "T@1", "T@2", "within_2s")):
                r2027 = float(t[(t["predictor"] == run) & (t["scope"] == "all 2027")][m].iloc[0])
                r400 = float(t[(t["predictor"] == run) & (t["scope"] == "frozen 400")][m].iloc[0])
                gap.append({"run": run, "metric": m, "synthetic_pooled_oof": float(s[sm]), "real_2027": r2027,
                            "gap_2027": r2027 - float(s[sm]), "real_400": r400, "gap_400": r400 - float(s[sm])})
        gap = pd.DataFrame(gap)
        gm = gap[gap["run"].isin(C_CCTV)].groupby("metric", sort=False).agg(
            **{c: (c, lambda v: f"{v.mean():.4f} +- {v.std(ddof=1):.4f}") for c in
               ("synthetic_pooled_oof", "real_2027", "gap_2027", "real_400", "gap_400")}).reset_index()
        gap.to_csv(out / "transfer_gap.csv", index=False)
        # confidence on real (C_cctv seeds 0-2), as the synthetic tau tables; no tau chosen
        pooled = pd.concat([allp[r].assign(seed=i) for i, r in enumerate(C_CCTV)], ignore_index=True)
        cum, binned = [], []
        for scope, q in [("pooled (seeds 0-2)", pooled)] + [(f"seed {i}", pooled[pooled["seed"] == i]) for i in range(3)]:
            a, b = conf_tables(q, scope)
            cum += a
            binned += b
        cum, binned = pd.DataFrame(cum), pd.DataFrame(binned)
        cum.to_csv(out / "confidence_cumulative_real.csv", index=False)
        binned.to_csv(out / "confidence_bins_real.csv", index=False)
        syn_c = pd.read_csv(TAU_DIR / "tau_cumulative.csv").query("scope == 'pooled (seeds 0-2)'")
        syn_b = pd.read_csv(TAU_DIR / "tau_bins.csv").query("scope == 'pooled (seeds 0-2)'")
        cmp_c = syn_c[["confidence", "coverage", "share_within_2s"]].merge(
            cum[cum["scope"] == "pooled (seeds 0-2)"][["confidence", "coverage", "share_within_2s"]], on="confidence",
            suffixes=(" synthetic", " real"))
        cmp_b = syn_b[["confidence", "coverage", "share_within_2s"]].merge(
            binned[binned["scope"] == "pooled (seeds 0-2)"][["confidence", "coverage", "share_within_2s"]],
            on="confidence", suffixes=(" synthetic", " real"))
        syn_conf = pd.concat([pd.read_csv(DIAG / "C_cctv_seeds012" / f"seed{s}_frozen" / "tta_clip_predictions_flip.csv")
                              ["confidence"] for s in (0, 1, 2)])
        qs = [0.1, 0.25, 0.5, 0.75, 0.9]
        quant = pd.DataFrame({"quantile": qs, "synthetic (6633)": syn_conf.quantile(qs).to_numpy(),
                              f"real ({len(pooled)})": pooled["confidence"].quantile(qs).to_numpy()})

        def brief(tb):
            s = tb[tb["scope"] != "pooled (seeds 0-2)"].pivot(index="confidence", columns="scope",
                                                               values=["coverage", "share_within_2s"])
            return s.reindex(tb["confidence"].drop_duplicates()).to_string(float_format=lambda v: f"{v:.4f}")
        rep += ["", "1. PER RUN (frozen post-processing, ensemble of the 3 fold models)",
                wide(t, tuple(s for s, _ in REAL_SCOPES)),
                "2. FINAL: C_cctv mean +- sample std over seeds 0-2 (ddof 1, n 3) next to C_v2 seed 0 and the naive "
                "midpoint", fmt(fin), "",
                "3. TRANSFER GAP: synthetic pooled out-of-fold (2211 clips, frozen post-processing, one out-of-fold model "
                "per clip) vs real (ensemble of 3 fold models); gap = real - synthetic", fmt(gap), "",
                "   C_cctv seeds 0-2, mean +- std:", gm.to_string(index=False), "",
                f"4. CONFIDENCE ON REAL (C_cctv seeds 0-2 on all 2027 clips, {len(pooled)} predictions pooled; fallback "
                "clips have confidence 0; no tau chosen; real has no window labels, so no label_B column)",
                "   cumulative, pooled", fmt(cum[cum["scope"] == "pooled (seeds 0-2)"].drop(columns="scope")),
                "   per seed (coverage, share within 2 s)", brief(cum), "",
                "   bins, pooled", fmt(binned[binned["scope"] == "pooled (seeds 0-2)"].drop(columns="scope")),
                "   per seed", brief(binned), "",
                "   synthetic (results/diagnostics/tau_evidence) vs real, pooled, cumulative:", fmt(cmp_c),
                "   bins:", fmt(cmp_b), "   confidence quantiles:", fmt(quant), ""]
        gpu_rows = [gpu_seconds(r, root) for r in ("C_v2_seed0",) + C_CCTV]
    g = pd.DataFrame(gpu_rows)
    n_win_clips = int((~next(iter(preds.values()))["fallback"]).sum())
    n_windows = len(pd.read_csv(REAL_WINDOWS, usecols=["window_idx"]))
    g["gpu_s_per_clip (clips with windows)"] = g["gpu_s_total"] / n_win_clips
    g["gpu_s_per_clip (all 2027)"] = g["gpu_s_total"] / len(clips)
    g["ms_per_window_pass"] = 1000 * g["gpu_s_total"] / (n_windows * g["passes"])
    g.to_csv(out / "gpu_seconds.csv", index=False)
    rep += ["GPU SECONDS (classifier only: forward incl. host-to-device copy, synchronised; 3 fold models x orig + hflip "
            f"= 6 passes over {n_windows} windows of {n_win_clips} clips; data loading overlapped, not counted)", fmt(g), "",
            "SECONDARY (nothing chosen from it): each fold model alone (flip TTA, same post-processing and fallback)",
            fmt(sec.drop(columns=["run", "fold_model"])), ""]
    text = "\n".join(rep) + "\n"
    (out / "report.txt").write_text(text, encoding="utf-8")
    print(text)
    for run in STAGES[stage]:
        append_event(root, f"{tag}PREDICTIONS {run} sha256 {sha_of(root / 'predictions' / run / 'clip_predictions.csv')}")
    append_event(root, f"{tag}STAGE {stage} EVALUATED; outputs {out}")
    print(f"wrote {out}")


# --------------------------------------------------------------------------- #
# Real test: analysis of the saved predictions (added to the plan 2026-11-15; no model run, nothing chosen)
# --------------------------------------------------------------------------- #
BOOT_N, BOOT_SEED = 10_000, 0  # fixed; not an option, so the bootstrap is not rerun with other seeds
ANALYSIS_RUNS = ("C_v2_seed0",) + C_CCTV
WILSON_NOTE = ("Secondary, added to help read bins with few clips; not part of the evaluation protocol and not used by "
               "the tau rule.")


def source_video(rgb_path: str) -> str:
    """Source video of a real clip: the file stem without the trailing _NN."""
    return re.sub(r"_\d+$", "", Path(str(rgb_path).replace("\\", "/")).stem)


def clip_terms(p: pd.DataFrame) -> pd.DataFrame:
    """Per-clip terms whose mean over clips is each metric (T@sigma: accs_scorer.temporal_per_clip; within 2 s: 0/1)."""
    t = pd.DataFrame({f"T_{s:g}": temporal_per_clip(p["t_hat"], p["t_star"], s) for s in SIGMAS_T}, index=p.index)
    t["within_2s"] = ((p["t_hat"] - p["t_star"]).abs() <= 2.0).astype(float)
    return t


def saved_real(f: Path, clips: pd.DataFrame) -> pd.DataFrame:
    """A saved real prediction file joined to the clips (labels), rows in the order and index of `clips`."""
    p = pd.read_csv(f)
    m = clips.merge(p[["rgb_path", "t_hat", "confidence", "fallback"]], on="rgb_path", how="inner",
                    validate="one_to_one")
    if len(p) != len(clips) or len(m) != len(clips) or not (m["rgb_path"].to_numpy() == clips["rgb_path"].to_numpy()).all():
        fail(f"{f}: expected each of the {len(clips)} real clips once")
    if (m["fallback"].astype(bool) == m["has_window"]).any():
        fail(f"{f}: fallback flags differ from the clips without a window")
    return m.set_index(clips.index)


def check_close(label: str, got: float, saved: float, tol: float = 1e-9) -> None:
    if (np.isnan(got) and np.isnan(saved)) or abs(got - saved) <= tol:
        return
    fail(f"{label}: recomputed {got!r} differs from the saved {saved!r}")


def real_analysis(root: Path) -> None:
    """--real-analysis: see module doc."""
    out = root / "analysis"
    if (out / "report.txt").exists():
        fail(f"{out / 'report.txt'} exists; the analysis is written once")
    _, events = read_lock(root)  # read only
    if not all(stage_done(events, s) for s in STAGES):
        fail("both real-test stages must be evaluated before this analysis")
    clips = real_clips(dry=False)
    clips["source"] = clips["rgb_path"].map(source_video)
    two = REAL_SCOPES[:2]  # all 2027, frozen 400

    # saved predictions, checked against the lock and against the saved stage tables
    preds = {}
    for run in ANALYSIS_RUNS:
        f = root / "predictions" / run / "clip_predictions.csv"
        h = [e for e in events if f"PREDICTIONS {run} sha256 " in e and "DRY RUN" not in e]
        if len(h) != 1 or sha_of(f) != h[0].split()[-1]:
            fail(f"{f} differs from its sha256 in {LOCK_NAME}")
        preds[run] = saved_real(f, clips)
    singles = {(run, k): saved_real(root / "predictions" / run / f"clip_predictions_fold{k}_model_only.csv", clips)
               for run in ANALYSIS_RUNS for k in FOLDS}
    prt = pd.read_csv(root / "stage_C2" / "per_run_table.csv")
    sec = pd.concat([pd.read_csv(root / f"stage_{s}" / "secondary_single_fold_models.csv") for s in STAGES])
    for scope, sel in two:
        for run, p in preds.items():
            got = scores(p[sel(p)])
            r = prt[(prt["predictor"] == run) & (prt["scope"] == scope)].iloc[0]
            for m in METRICS:
                check_close(f"{run} {scope} {m}", got[m], float(r[m]))
        for (run, k), p in singles.items():
            got = scores(p[sel(p)])
            r = sec[(sec["run"] == run) & (sec["fold_model"] == k) & (sec["scope"] == scope)].iloc[0]
            for m in METRICS:
                check_close(f"{run} fold {k} model {scope} {m}", got[m], float(r[m]))
    rep = [f"Real test: analysis of the saved predictions (ADDITION TO THE PLAN, 2026-11-15; after stages C1 and C2; no "
           f"model run; nothing chosen or changed; {LOCK_NAME} only read). {utc_now()}",
           f"inputs: {root / 'predictions'} (clip_predictions.csv sha256 = the lock's EVENT lines; per-fold-model files), "
           f"stage_C1/C2 tables (all {len(ANALYSIS_RUNS)} runs' metrics on all 2027 / frozen 400 and the single-fold-model "
           "metrics recomputed and equal to the saved tables), synthetic frozen predictions "
           "(results/diagnostics/.../tta_clip_predictions_flip.csv), metadata-real.csv (accident_time, duration)", ""]

    # 1. paired bootstrap over source videos
    rng = np.random.default_rng(BOOT_SEED)
    terms = {run: clip_terms(p) for run, p in preds.items()}
    cctv_mean = sum(terms[r] for r in C_CCTV) / len(C_CCTV)
    comps = (("C_v2_seed0 - C_cctv_seed0", terms["C_v2_seed0"] - terms["C_cctv_seed0"]),
             ("C_v2_seed0 - C_cctv mean over seeds 0-2", terms["C_v2_seed0"] - cctv_mean))
    boot, src = [], []
    for scope, sel in two:
        mask = sel(clips).to_numpy()
        codes, uniq = pd.factorize(clips.loc[mask, "source"])
        g = len(uniq)
        n_g = np.bincount(codes, minlength=g).astype(float)
        src.append({"scope": scope, "n_clips": int(mask.sum()), "n_source_videos": g,
                    "clips_per_source_median": float(np.median(n_g)), "clips_per_source_max": int(n_g.max())})
        w = rng.multinomial(g, np.full(g, 1.0 / g), size=BOOT_N).astype(float)  # resample counts per source video
        den = w @ n_g
        for name, d in comps:
            for m in METRICS:
                v = d.loc[mask, m].to_numpy()
                stat = (w @ np.bincount(codes, weights=v, minlength=g)) / den
                lo, hi = np.percentile(stat, [2.5, 97.5])
                boot.append({"scope": scope, "n_clips": int(mask.sum()), "n_source_videos": g, "comparison": name,
                             "metric": m, "difference": float(v.mean()), "ci95_lo": float(lo), "ci95_hi": float(hi)})
    boot, src = pd.DataFrame(boot), pd.DataFrame(src)

    # 2. per fold: synthetic validation T@1 of fold model k next to its real T@1 alone
    pf, syn_pred = [], {}
    for run in ANALYSIS_RUNS:
        syn = pd.read_csv(SYN_TTA[run].parent / "tta_clip_predictions_flip.csv")
        if len(syn) != 2211 or syn["rgb_path"].duplicated().any() or set(syn["fold"]) != set(FOLDS):
            fail(f"{run}: synthetic frozen predictions must hold the 2211 clips once, folds 0-2")
        syn_pred[run] = syn
        tt = pd.read_csv(SYN_TTA[run])
        tt = tt[tt["arm"] == "flip TTA"].set_index("scope")
        for k in list(FOLDS) + ["pooled"]:
            q = syn if k == "pooled" else syn[syn["fold"] == k]
            sv = temporal(q["t_hat"], q["t_star"], 1.0)
            check_close(f"{run} synthetic {k} T@1", sv, float(tt.loc["pooled" if k == "pooled" else f"fold {k}", "T@1"]))
            real = preds[run] if k == "pooled" else singles[(run, k)]
            pf.append({"run": run, "fold_model": "pooled / 3-model ensemble" if k == "pooled" else f"fold {k}",
                       "synthetic_val_n_clips": len(q), "synthetic_val_T@1": sv,
                       **{f"real_T@1 {scope}": scores(real[sel(real)])["T_1"] for scope, sel in two}})
    pf = pd.DataFrame(pf)

    # 3. early impact: durations and gain over the naive midpoint per group
    ei = []
    for scope, sel in two:
        base = clips[sel(clips)]
        for grp, gm in ((f"t* < {EARLY_T} s", base["early"]), (f"t* >= {EARLY_T} s", ~base["early"])):
            q = base[gm]
            naive = scores(q.assign(t_hat=q["duration"] / 2))
            row = {"scope": scope, "group": grp, "n_clips": len(q), "duration_mean": q["duration"].mean(),
                   "duration_median": q["duration"].median(), "duration_min": q["duration"].min(),
                   "duration_max": q["duration"].max()}
            row |= {f"naive {m}": naive[m] for m in METRICS}
            v2 = scores(preds["C_v2_seed0"].loc[q.index])
            row |= {f"C_v2_seed0 gain {m}": v2[m] - naive[m] for m in METRICS}
            for m in METRICS:
                gains = pd.Series([scores(preds[r].loc[q.index])[m] - naive[m] for r in C_CCTV])
                row[f"C_cctv gain {m} mean"], row[f"C_cctv gain {m} sd"] = gains.mean(), gains.std(ddof=1)
            ei.append(row)
    ei = pd.DataFrame(ei)

    # 4. confidence, synthetic vs real, per C_cctv seed (each clip once per table), then the mean over seeds
    conf = []
    syn_saved = {"cumulative": pd.read_csv(TAU_DIR / "tau_cumulative.csv"), "bins": pd.read_csv(TAU_DIR / "tau_bins.csv")}
    real_saved = {"cumulative": pd.read_csv(root / "stage_C2" / "confidence_cumulative_real.csv"),
                  "bins": pd.read_csv(root / "stage_C2" / "confidence_bins_real.csv")}
    for i, run in enumerate(C_CCTV):
        for domain, p, saved in (("synthetic", syn_pred[run], syn_saved), ("real", preds[run], real_saved)):
            cum, binned = conf_tables(p, f"seed {i}")
            for kind, rows in (("cumulative", cum), ("bins", binned)):
                sv = saved[kind][saved[kind]["scope"] == f"seed {i}"].set_index("confidence")
                for r in rows:
                    for c in ("n_clips", "coverage", "share_within_2s", "T@1"):
                        check_close(f"{domain} seed {i} {kind} {r['confidence']} {c}", float(r[c]),
                                    float(sv.loc[r["confidence"], c]))
                    conf.append({"table": kind, "domain": domain, "seed": i, "confidence": r["confidence"],
                                 "n_clips": r["n_clips"], "coverage": r["coverage"],
                                 "share_within_2s": r["share_within_2s"], "T@1": r["T@1"],
                                 "wilson95_lo (secondary)": r["wilson95_lo"], "wilson95_hi (secondary)": r["wilson95_hi"]})
    conf = pd.DataFrame(conf)
    cmean = conf.groupby(["table", "domain", "confidence"], sort=False).agg(
        n_clips=("n_clips", "mean"), coverage=("coverage", "mean"), share_within_2s=("share_within_2s", "mean"),
        **{"T@1": ("T@1", "mean")}, seeds_with_clips=("n_clips", lambda v: int((v > 0).sum()))).reset_index()

    def side(tb: pd.DataFrame, with_wilson: bool) -> str:
        s = tb[tb["domain"] == "synthetic"].set_index("confidence")
        r = tb[tb["domain"] == "real"].set_index("confidence").reindex(s.index)
        t = pd.DataFrame({"syn n": s["n_clips"], "syn coverage": s["coverage"], "syn within_2s": s["share_within_2s"],
                          "syn T@1": s["T@1"], "real n": r["n_clips"], "real coverage": r["coverage"],
                          "real within_2s": r["share_within_2s"], "real T@1": r["T@1"]})
        if with_wilson:
            w = lambda x: "-" if x["n_clips"] == 0 else f"[{x['wilson95_lo (secondary)']:.4f}, {x['wilson95_hi (secondary)']:.4f}]"  # noqa: E731
            t["syn Wilson95 (secondary)"] = s.apply(w, axis=1)
            t["real Wilson95 (secondary)"] = r.apply(w, axis=1)
        else:
            t["seeds with clips syn/real"] = s["seeds_with_clips"].astype(str) + "/" + r["seeds_with_clips"].astype(str)
        return t.reset_index().to_string(index=False, float_format=lambda v: f"{v:.4f}")

    # report
    rep += [f"1. PAIRED BOOTSTRAP OF THE ABLATION ON REAL (difference = C_v2_seed0 minus C_cctv; positive = C_v2 higher; "
            f"source videos resampled with replacement, {BOOT_N} resamples, numpy default_rng seed {BOOT_SEED}, the same "
            "resamples for both comparisons and all metrics of a scope; metric of a resample = mean over its clips; "
            "percentile 95% interval; C_cctv mean = per-clip mean of the three seeds' terms)",
            "   source videos (clip stem without _NN):", fmt(src)]
    for scope, _ in two:
        b = boot[boot["scope"] == scope]
        rep += [f"   [{scope}] n = {int(b['n_clips'].iloc[0])} clips, {int(b['n_source_videos'].iloc[0])} source videos",
                fmt(b.drop(columns=["scope", "n_clips", "n_source_videos"]))]
    rep += ["", "2. PER FOLD MODEL: synthetic validation T@1 of fold model k on its own validation fold (frozen "
            "post-processing, saved predictions) next to the real T@1 of that fold model alone (flip TTA, same "
            "post-processing, fallback); last row per run: synthetic pooled out-of-fold and the real 3-model ensemble",
            fmt(pf), "",
            f"3. EARLY IMPACT (t* < {EARLY_T} s) vs THE REST: clip duration (s) and gain over the naive midpoint "
            "(classifier - naive; C_cctv mean +- sample std over seeds 0-2)"]
    for scope, _ in two:
        q = ei[ei["scope"] == scope]
        rep += [f"   [{scope}] duration and naive midpoint",
                fmt(q[["group", "n_clips", "duration_mean", "duration_median", "duration_min", "duration_max"]
                      + [f"naive {m}" for m in METRICS]]),
                f"   [{scope}] gain over the naive midpoint",
                pd.DataFrame({"group": q["group"], **{f"C_v2_seed0 {m}": q[f"C_v2_seed0 gain {m}"].map(lambda v: f"{v:+.4f}")
                                                     for m in METRICS},
                              **{f"C_cctv {m}": [f"{a:+.4f} +- {s:.4f}" for a, s in
                                                 zip(q[f"C_cctv gain {m} mean"], q[f"C_cctv gain {m} sd"])]
                                 for m in METRICS}}).to_string(index=False)]
    rep += ["", "4. CONFIDENCE, SYNTHETIC vs REAL, PER C_cctv SEED (each clip once per table: synthetic 2211 out-of-fold "
            "predictions, real 2027 clips incl. 3 fallback clips with confidence 0; no tau chosen)"]
    for kind in ("bins", "cumulative"):
        for i in range(3):
            rep += [f"   {kind}, seed {i}", side(conf[(conf["table"] == kind) & (conf["seed"] == i)], True),
                    f"   Wilson column: {WILSON_NOTE}", ""]
        rep += [f"   {kind}, mean over seeds 0-2 (n = mean count; share within 2 s and T@1 averaged over the seeds "
                "with clips in the row; no interval: a mean of three dependent shares is not a binomial count)",
                side(cmean[cmean["table"] == kind], False), ""]
    out.mkdir(parents=True, exist_ok=True)
    boot.to_csv(out / "bootstrap_ablation.csv", index=False)
    src.to_csv(out / "source_videos.csv", index=False)
    pf.to_csv(out / "per_fold_table.csv", index=False)
    ei.to_csv(out / "early_impact.csv", index=False)
    conf.to_csv(out / "confidence_per_seed.csv", index=False)
    cmean.to_csv(out / "confidence_mean_over_seeds.csv", index=False)
    text = "\n".join(rep) + "\n"
    (out / "report.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="*", help="run names under runs/ (none with --real-lock / --real-eval)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow-config-diff", default="", metavar="KEYS",
                    help=f"comma-separated config keys allowed to differ between folds (only {','.join(BATCH_KEYS)}, "
                         "and only with equal micro_batch x grad_accum); listed as documented deviations")
    ap.add_argument("--run-cache-manifest", action="append", default=[], metavar="RUN:FOLD=PATH",
                    help="the cache_manifest.json a fold was trained with, if its cache was built on another machine")
    ap.add_argument("--smooth", type=int, default=None,
                    help="smoothing width for t_hat (default: the run's smooth_window and its saved predictions)")
    ap.add_argument("--interp", choices=INTERPS, default="none", help="peak interpolation (default none)")
    ap.add_argument("--postproc-grid", default=None, metavar="W,W,...",
                    help="post-processing study of the first run (see module doc); writes no summary files")
    ap.add_argument("--postproc-out", type=Path, default=None)
    ap.add_argument("--postproc-min-gain", type=float, default=0.002)
    ap.add_argument("--scores-dir", type=Path, default=None,
                    help="window scores from analyse_predictions.py --score-best (DIR/fold_<k>/val_window_scores_*.csv) "
                         "instead of the runs' val_window_scores_best.csv")
    ap.add_argument("--tta", choices=TTAS, default="none", help="flip = mean of orig and hflip scores (needs --scores-dir)")
    ap.add_argument("--tta-study", action="store_true", help="no TTA vs flip TTA for the first run; no summary files")
    ap.add_argument("--tta-min-gain", type=float, default=0.002)
    ap.add_argument("--window-ensemble", action="store_true",
                    help="real-test path: TTA + fold-model ensemble + post-processing (see module doc); no summary files")
    ap.add_argument("--ensemble-folds", default="0,1,2")
    ap.add_argument("--check-against", type=Path, default=None, help="with --window-ensemble, synthetic check only")
    ap.add_argument("--check-scores-dir", type=Path, default=None, help="with --window-ensemble, synthetic check only")
    ap.add_argument("--real-lock", action="store_true", help="write RUN_ONCE.txt for the real test (see module doc)")
    ap.add_argument("--real-eval", choices=tuple(STAGES), default=None, help="evaluate one real-test stage, once")
    ap.add_argument("--real-root", type=Path, default=REAL_DIR)
    ap.add_argument("--dry-run", action="store_true",
                    help="with --real-eval: FAKE t* = 0.37 x duration, no label read (code test in another --real-root)")
    ap.add_argument("--real-analysis", action="store_true",
                    help="analysis of the saved real-test predictions (see module doc); writes <real root>/analysis/")
    args = ap.parse_args()
    if args.real_analysis:
        if args.runs or args.real_lock or args.real_eval or args.dry_run:
            fail("--real-analysis takes no run names and no other real-test mode")
        real_analysis(args.real_root)
        return
    if args.real_lock or args.real_eval:
        if args.runs:
            fail("--real-lock / --real-eval take no run names")
        if args.dry_run and args.real_root.resolve() == REAL_DIR.resolve():
            fail("--dry-run must use another --real-root")
        if args.real_lock:
            real_lock(args.real_root)
        else:
            real_eval(args.real_eval, args.real_root, args.dry_run)
        return
    if args.dry_run:
        fail("--dry-run only applies with --real-eval")
    if not args.runs:
        fail("give at least one run name")
    if args.window_ensemble:
        window_ensemble(args)
        return
    if args.check_against or args.check_scores_dir:
        fail("--check-against / --check-scores-dir only apply with --window-ensemble")
    if (args.postproc_grid is None and not args.tta_study and (OUT_CSV.exists() or OUT_MD.exists())
            and not args.overwrite):
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
    if args.postproc_grid is not None:
        postproc_grid(args, clips, allow, manifests)
        return
    if args.tta_study:
        tta_study(args, clips, allow, manifests)
        return
    runs, rows, deviations = [], [], {}
    for name in args.runs:
        run, fold_rows, pooled_row, deviations[name] = summarise_run(name, clips, allow, manifests.get(name),
                                                                     args.smooth, args.interp, args.scores_dir, args.tta)
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
    if args.smooth is not None or args.interp != "none" or args.scores_dir is not None or args.tta != "none":
        md[-1:-1] = [f"Post-processing: t_hat recomputed from the best-check window scores "
                     f"({args.scores_dir or 'runs/<run>/fold_<k>/val_window_scores_best.csv'}, TTA {args.tta}) with "
                     f"smoothing {args.smooth or 'smooth_window'} and interpolation {args.interp} (not the saved "
                     "predictions).", ""]
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
