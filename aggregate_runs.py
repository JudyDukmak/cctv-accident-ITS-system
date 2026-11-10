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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="run names under runs/")
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
    args = ap.parse_args()
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
