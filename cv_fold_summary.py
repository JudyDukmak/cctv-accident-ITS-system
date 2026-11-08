"""Integrity check and 3-fold summary of one run name (SYNTHETIC ONLY; reads runs/<run>/fold_{0,1,2}/ config.json,
train_log.txt, metrics.csv and val_*_best.csv, synthetic_split.csv and metadata-synthetic.csv; no model, no GPU,
writes nothing in runs/). Fold consistency uses aggregate_runs.fold_consistency with the same options
(--allow-config-diff, --run-cache-manifest); allowed differences are listed under "Documented deviations". Unlike
aggregate_runs.py it does NOT stop when that check fails: it labels the summary PROVISIONAL instead.

1. Integrity: config / data hashes, git, GPU, library versions, seed, scheme, schedule per fold; differing values
   flagged; config keys that differ listed; documented deviations (incl. --deviation-note text and the last-check
   T@1 rescored from the saved CSVs vs metrics.csv).
1b. Window score by phase (analyse_predictions.phase_of) at the best and the last check per fold, ratios
   near / aftermath and pre / near of the phase means, late-peak and early > 2 s shares; mean +- sample std.
   The last-check CSVs come from analyse_predictions.py (results/diagnostics/<run>_fold<k>/).
2. Per fold at the best check: epoch_fraction, T@0.5/1/2, within_2s, median error, early > 2 s share, late-peak
   share (compare_diagnostics.late_peak_share), naive t_hat = duration / 2 T@1 on the same clips; mean +- sample
   std (ddof 1, n = 3) over folds. Fold-0-only reference rows for other runs (--fold0-refs).
3. within_2s per collision type, per fold and pooled.
4. Confidence vs error (confidence_vs_error.block), per fold and pooled.
5. Pooled out-of-fold: best-check predictions of the 3 folds must cover each clip of synthetic_split.csv exactly
   once; T@0.5/1/2, within_2s and the naive reference on those clips.
Outputs: results/diagnostics/<run>_3folds/ (summary.txt + CSVs).

Usage (training environment, CPU):
    .venv-ml\\Scripts\\python.exe cv_fold_summary.py --run C_v2_seed0 --fold0-refs B_v2_seed0 std_seed0 \\
        [--allow-config-diff micro_batch,grad_accum] [--run-cache-manifest FOLD=PATH] [--deviation-note TEXT ...]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
import train_classifier as tc  # noqa: E402  (labels, split and constants only; no model, no CUDA)
from accs_scorer import temporal  # noqa: E402
from aggregate_runs import BATCH_KEYS, all_clips, fold_consistency  # noqa: E402
from analyse_predictions import phase_of, with_context  # noqa: E402
from compare_diagnostics import DIAG_ROOT, METADATA_SYN, late_peak_share, load, smooth_window  # noqa: E402
from confidence_vs_error import block as confidence_block  # noqa: E402

FOLDS = (0, 1, 2)
MUST_MATCH = ("config_hash", "window_labels_sha256", "split_sha256", "cache_manifest_sha256", "seed", "label_scheme")
EXPECTED = ("git_commit", "git_dirty", "gpu", "updates_per_epoch", "warmup_updates", "total_updates",
            "n_train_windows", "n_val_clips", "n_val_windows", "output_dir")


def table(df: pd.DataFrame, index: bool = True) -> str:
    return df.to_string(index=index, float_format=lambda v: f"{v:.4f}")


def flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out |= flat(v, f"{prefix}{k}.")
        else:
            out[f"{prefix}{k}"] = json.dumps(v)
    return out


# --------------------------------------------------------------------------- #
# 1. Integrity
# --------------------------------------------------------------------------- #
def integrity(run: str) -> tuple[pd.DataFrame, list[str], bool]:
    rows, cfgs = {}, {}
    for k in FOLDS:
        d = tc.RUNS_DIR / run / f"fold_{k}"
        info = json.loads((d / "config.json").read_text(encoding="utf-8"))
        log = (d / "train_log.txt").read_text(encoding="utf-8")
        head = re.search(r"GPU (.+?); torch (\S+); transformers (\S+); git (\w+)( \(dirty\))?", log)
        if info.get("fold") != k or info.get("smoke_test"):
            tc.fail(f"{d}: not a full fold-{k} run")
        eff = info["effective"]
        rows[f"fold {k}"] = {
            "files": ", ".join(sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file())),
            "config_file": Path(info["config_file"]).name, "config_hash": info["config_hash"],
            **info["data_hashes"],
            "git_commit": info["git_commit"][:7], "git_dirty": info["git_dirty"], "gpu": info["gpu"],
            "torch": info["torch"], "transformers": info["transformers"], "python": info["python"],
            "cuda": info["cuda"], "seed": info["seed"], "label_scheme": eff["label_scheme"],
            "updates_per_epoch": eff["updates_per_epoch"], "warmup_updates": eff["warmup_updates"],
            "total_updates": eff["total_updates"], "n_train_windows": info["n_train_windows"],
            "n_val_clips": info["n_val_clips"], "n_val_windows": info["n_val_windows"],
            "output_dir": re.search(r"output (\S.*)", log).group(1).strip(),
            "train_log header agrees with config.json": (
                head.group(1) == info["gpu"] and head.group(2) == info["torch"]
                and head.group(3) == info["transformers"] and head.group(4) == info["git_commit"]
                and bool(head.group(5)) == bool(info["git_dirty"])),
        }
        cfgs[k] = flat(info["config"])
    t = pd.DataFrame(rows)

    def flag(r: pd.Series) -> str:
        if r.name == "files" or r.nunique() == 1:
            return ""
        if r.name in MUST_MATCH:
            return "DIFFERS (must match)"
        return "differs (expected)" if r.name in EXPECTED else "DIFFERS"
    t["flag"] = t.apply(flag, axis=1)
    keys = sorted(set().union(*cfgs.values()))
    cfg_diff = [f"  {key}: " + ", ".join(f"fold {k} {cfgs[k].get(key, '<absent>')}" for k in FOLDS)
                for key in keys if len({cfgs[k].get(key, "<absent>") for k in FOLDS}) > 1]
    ok = not t["flag"].str.startswith("DIFFERS").any()
    return t, cfg_diff or ["  none"], ok


# --------------------------------------------------------------------------- #
# 2-5. Metrics
# --------------------------------------------------------------------------- #
def fold_data(run: str, fold: int, data: pd.DataFrame, meta: pd.DataFrame) -> dict:
    _, val_rows = tc.split_fold(data, fold, smoke=False)
    clips = (val_rows.groupby("rgb_path", sort=True)
             .agg(t_star=("t_star", "first"), ctype=("ctype", "first")).reset_index()
             .merge(meta, on="rgb_path", how="left", validate="one_to_one"))
    if clips["duration"].isna().any():
        tc.fail(f"fold {fold}: clips missing from {METADATA_SYN.name}")
    win, pred = load(run, "best", fold)
    p = pred[["rgb_path", "t_hat", "confidence"]].merge(clips, on="rgb_path", validate="one_to_one")
    w = win[["rgb_path", "window_idx", "score"]].merge(
        val_rows[["rgb_path", "window_idx", "t_centre", "t_star"]], on=["rgb_path", "window_idx"],
        validate="one_to_one")
    if len(p) != len(clips) or len(pred) != len(clips) or len(w) != len(val_rows):
        tc.fail(f"{run} fold {fold}: best predictions do not cover the fold's validation clips / windows")
    m = pd.read_csv(tc.RUNS_DIR / run / f"fold_{fold}" / "metrics.csv")
    best = m[m["best"].astype(str) == "True"].iloc[-1]
    return {"clips": clips, "pred": p, "win": w, "best": best, "last": m.iloc[-1], "val_rows": val_rows,
            "k": smooth_window(run, fold)}


PHASES = {"pre (label_B 0)": "pre", "gap (label_B -1, ends < 1 s before t*)": "gap", "near (label_B 1)": "near",
          "aftermath: starts 0-2 s after t*": "aft 0-2 s", "aftermath: starts 2-5 s after t*": "aft 2-5 s",
          "aftermath: starts > 5 s after t*": "aft > 5 s"}


def phase_row(run: str, fold: int, which: str, fd: dict) -> dict:
    """Mean window score by phase (analyse_predictions.phase_of), the two ratios, late-peak and early shares, and
    T@1 recomputed from the CSVs, for the best check ("best") or a saved last check ("checkN")."""
    win, pred = load(run, which, fold)
    w = with_context(win, fd["val_rows"], fd["k"])
    ph = phase_of(w).map(PHASES)
    means = w.groupby(ph)["score"].mean()
    aftermath = float(w.loc[ph.str.startswith("aft"), "score"].mean())
    p = pred[["rgb_path", "t_hat", "t_star"]].merge(fd["clips"][["rgb_path"]], on="rgb_path", validate="one_to_one")
    if len(p) != len(fd["clips"]):
        tc.fail(f"{run} fold {fold} {which}: clip predictions do not cover the fold's validation clips")
    share, _ = late_peak_share(w, fd["k"])
    return {**{v: float(means[v]) for v in PHASES.values()}, "aftermath all": aftermath,
            "near / aftermath": float(means["near"]) / aftermath, "pre / near": float(means["pre"] / means["near"]),
            "late-peak share": share, "early > 2 s": float(((p["t_hat"] - p["t_star"]) < -2).mean()),
            "T@1 (from CSV)": float(temporal(p["t_hat"], p["t_star"], 1.0))}


def metric_row(fd: dict) -> dict:
    p, b = fd["pred"], fd["best"]
    err = p["t_hat"] - p["t_star"]
    share, n_lp = late_peak_share(fd["win"], fd["k"])
    t1 = float(temporal(p["t_hat"], p["t_star"], 1.0))
    if abs(t1 - float(b["val_T_1"])) > 1e-6:
        tc.fail(f"T@1 from val_clip_predictions_best.csv {t1:.6f} != metrics.csv {float(b['val_T_1']):.6f}")
    return {"n_clips": len(p),
            "best check": int(b["check_idx"]) if "check_idx" in b and not pd.isna(b["check_idx"]) else int(b["epoch"]),
            "epoch_fraction": float(b["epoch_fraction"]) if "epoch_fraction" in b and not pd.isna(b["epoch_fraction"])
            else float(b["epoch"] + 1),
            **{f"T@{s:g}": float(temporal(p["t_hat"], p["t_star"], s)) for s in (0.5, 1.0, 2.0)},
            "within_2s": float((err.abs() <= 2).mean()), "median err (s)": float(err.median()),
            "early > 2 s": float((err < -2).mean()), f"late-peak share": share, "late-peak n": n_lp,
            "naive T@1": float(temporal(p["duration"] / 2, p["t_star"], 1.0))}


def per_type(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = {}
    for label, p in preds.items():
        ok = (p["t_hat"] - p["t_star"]).abs() <= 2
        out[label] = {t: (f"{ok[p['ctype'] == t].mean():.4f} (n={int((p['ctype'] == t).sum())})"
                          if (p["ctype"] == t).any() else "n/a (n=0)") for t in tc.TYPES}
        out[label]["all"] = f"{ok.mean():.4f} (n={len(p)})"
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="C_v2_seed0")
    ap.add_argument("--fold0-refs", nargs="*", default=[], help="other runs shown as fold-0-only rows")
    ap.add_argument("--allow-config-diff", default="", metavar="KEYS", help="as in aggregate_runs.py")
    ap.add_argument("--run-cache-manifest", action="append", default=[], metavar="FOLD=PATH",
                    help="the cache_manifest.json a fold was trained with (as in aggregate_runs.py)")
    ap.add_argument("--deviation-note", action="append", default=[], metavar="TEXT",
                    help="externally measured evidence for a documented deviation, copied into the summary")
    args = ap.parse_args()
    out_dir = DIAG_ROOT / f"{args.run}_3folds"
    out_dir.mkdir(parents=True, exist_ok=True)

    integ, cfg_diff, ok = integrity(args.run)
    infos = [json.loads((tc.RUNS_DIR / args.run / f"fold_{k}" / "config.json").read_text(encoding="utf-8"))
             for k in FOLDS]
    allow = {k for k in args.allow_config_diff.split(",") if k}
    manifests = {int(f): Path(p) for f, p in (item.split("=", 1) for item in args.run_cache_manifest)}
    try:
        if allow - set(BATCH_KEYS):
            tc.fail(f"--allow-config-diff accepts only {','.join(BATCH_KEYS)}")
        deviations = fold_consistency(args.run, infos, allow, manifests)
        status = ""
        if not ok:  # every flagged difference passed fold_consistency
            integ["flag"] = integ["flag"].str.replace(r"^DIFFERS.*", "documented deviation", regex=True)
    except SystemExit as e:
        deviations = []
        status = f"PROVISIONAL: {e} (aggregate_runs.py refuses this run with these options)"
    integ.to_csv(out_dir / "integrity.csv")

    data, _, _ = tc.load_labels(8)
    meta = pd.read_csv(METADATA_SYN, usecols=["rgb_path", "duration"])
    fds = {k: fold_data(args.run, k, data, meta) for k in FOLDS}

    # 1b. phases at the best and the last check
    ph_rows, rescore = {}, []
    for k, fd in fds.items():
        b, last = int(fd["best"]["check_idx"]), int(fd["last"]["check_idx"])
        ph_rows[(f"fold {k}", f"best (check {b})")] = phase_row(args.run, k, "best", fd)
        ph_rows[(f"fold {k}", f"last (check {last})")] = phase_row(args.run, k, f"check{last}", fd)
        t1 = ph_rows[(f"fold {k}", f"last (check {last})")]["T@1 (from CSV)"]
        rescore.append(f"fold {k} last check {last}: T@1 rescored from the saved CSV {t1:.4f} vs metrics.csv "
                       f"{float(fd['last']['val_T_1']):.4f} (diff {t1 - float(fd['last']['val_T_1']):+.4f})")
    phases = pd.DataFrame(ph_rows).T
    phases.index.names = ["fold", "check"]
    ph_agg = {}
    for which in ("best", "last"):
        sub = phases[phases.index.get_level_values("check").str.startswith(which)].astype(float)
        ph_agg[(which, "mean (n=3)")] = sub.mean()
        ph_agg[(which, "sample std (ddof=1, n=3)")] = sub.std(ddof=1)
    ph_agg = pd.DataFrame(ph_agg).T
    phases.to_csv(out_dir / "phases_best_last.csv")
    ph_agg.to_csv(out_dir / "phases_best_last_mean_std.csv")

    # 2. per fold + mean +- std
    per_fold = pd.DataFrame({f"fold {k}": metric_row(fd) for k, fd in fds.items()}).T
    num = per_fold.drop(columns=["n_clips", "best check", "late-peak n"]).astype(float)
    agg = pd.DataFrame({"mean (n=3)": num.mean(), "sample std (ddof=1, n=3)": num.std(ddof=1)}).T
    refs = {}
    for r in args.fold0_refs:
        refs[f"{r} fold 0 ONLY"] = metric_row(fold_data(r, 0, data, meta))
    refs_df = pd.DataFrame(refs).T
    per_fold.to_csv(out_dir / "per_fold.csv")
    agg.to_csv(out_dir / "per_fold_mean_std.csv")
    refs_df.to_csv(out_dir / "fold0_only_references.csv")

    # 3. per type
    pooled = pd.concat([fd["pred"].assign(fold=k) for k, fd in fds.items()], ignore_index=True)
    types = per_type({**{f"fold {k}": fd["pred"] for k, fd in fds.items()}, "pooled (3 folds)": pooled})
    types.to_csv(out_dir / "within_2s_by_type.csv")

    # 4. confidence
    conf = []
    for k, fd in fds.items():
        conf += confidence_block(f"fold {k} (best check)", fd["pred"], fd["clips"][["rgb_path", "t_star"]])
    conf += confidence_block("pooled over 3 folds (each clip scored by its own fold's model)", pooled,
                             pooled[["rgb_path", "t_star"]])

    # 5. pooled out-of-fold
    clips_all = all_clips()
    dup = pooled["rgb_path"].duplicated()
    missing = sorted(set(clips_all) - set(pooled["rgb_path"]))
    extra = sorted(set(pooled["rgb_path"]) - set(clips_all))
    if dup.any() or missing or extra:
        tc.fail(f"pooled: {int(dup.sum())} duplicates, {len(missing)} missing, {len(extra)} unknown clips")
    err = pooled["t_hat"] - pooled["t_star"]
    oof = pd.DataFrame([
        {"predictor": f"{args.run} best checks, pooled out-of-fold", "n_clips": len(pooled),
         **{f"T@{s:g}": float(temporal(pooled["t_hat"], pooled["t_star"], s)) for s in (0.5, 1.0, 2.0)},
         "within_2s": float((err.abs() <= 2).mean())},
        {"predictor": "naive duration / 2 (same clips)", "n_clips": len(pooled),
         **{f"T@{s:g}": float(temporal(pooled["duration"] / 2, pooled["t_star"], s)) for s in (0.5, 1.0, 2.0)},
         "within_2s": float(((pooled["duration"] / 2 - pooled["t_star"]).abs() <= 2).mean())}])
    oof.to_csv(out_dir / "pooled_out_of_fold.csv", index=False)
    pooled.to_csv(out_dir / "pooled_out_of_fold_predictions.csv", index=False)

    show = integ.drop(index="files").copy()
    for c in ("config_hash", "window_labels_sha256", "split_sha256", "cache_manifest_sha256"):
        show.loc[c, [f"fold {k}" for k in FOLDS]] = show.loc[c, [f"fold {k}" for k in FOLDS]].str[:12] + "..."
    show = show.drop(index="output_dir")
    lines = [f"3-fold summary: run {args.run} (synthetic validation; best check per fold; no training, no GPU)"]
    lines += [status] if status else []
    lines += ["", "=== 1. integrity ===", "files per fold:"]
    lines += [f"  fold {k}: {integ.loc['files', f'fold {k}']}" for k in FOLDS]
    lines += [table(show), "", "config keys that differ between folds:"] + cfg_diff
    lines += ["output dirs (train_log): " + "; ".join(f"fold {k} {integ.loc['output_dir', f'fold {k}']}" for k in FOLDS)]
    lines += ["", "Documented deviations (checked by aggregate_runs.fold_consistency unless marked as notes):"]
    lines += [f"  - {d}" for d in deviations] or ["  none"]
    lines += [f"  - note: {n}" for n in args.deviation_note]
    lines += [f"  - last-check rescore: {r}" for r in rescore]
    lines += ["", "=== 1b. window score by phase at the best and the last check ===",
              "phases as analyse_predictions.phase_of (aftermath by window start relative to t*); aftermath all = "
              "every window starting after t*; ratios of phase means; T@1 (from CSV) = recomputed from the saved "
              "clip predictions", table(phases), "", table(ph_agg)]
    lines += ["", "=== 2. per fold (best check) and mean +- sample std over folds ===",
              table(per_fold), "", table(agg), "",
              "fold-0-only references (NOT 3-fold results; same 770 fold-0 clips):",
              table(refs_df) if len(refs_df) else "  none", "",
              "=== 3. within_2s by collision type (n clips per cell) ===", types.to_string(), "",
              "=== 4. confidence vs error ==="] + conf
    lines += ["=== 5. pooled out-of-fold ===",
              f"clips: {len(pooled)} rows, {pooled['rgb_path'].nunique()} unique, 0 duplicates, all "
              f"{len(clips_all)} clips of synthetic_split.csv covered (hash-checked against split_manifest.json)",
              table(oof, index=False), "",
              "Notes: early > 2 s = share of clips with t_hat - t_star < -2 s. Late-peak share: clips whose max "
              "smoothed score with window centre > t*+2 s exceeds the max within t* +-1 s, among clips with both "
              "(n = late-peak n). Pooled confidence quartiles mix three models' confidence scales."]
    text = "\n".join(lines) + "\n"
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
