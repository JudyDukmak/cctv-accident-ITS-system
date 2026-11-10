"""Diagnose a fold's validation predictions at the best epoch and at the last epoch (SYNTHETIC ONLY:
reads runs/<run>/fold_<k>/, synthetic_window_labels.csv and metadata-synthetic.csv; never opens real
videos, metadata-real.csv or real_test_set.csv). Read-only on data and runs; no training.
For runs that validate every N updates, "epoch" below means "check" (labels use check_idx).

1. Scores all validation windows of the fold with the checkpoint saved at the last metrics row's update
   (last.pt or step_latest.pt; eval, bf16, the run's val batch), with the training script's smoothing
   and peak rule (train_classifier.validate).
2. Naive reference t_hat = duration / 2 on the same clips.
3. For the best epoch (val_*_best.csv) and the last epoch: signed-error distribution, results by
   collision type / scenario / t_star tercile, mean window score by phase, late-peak share.
4. Best vs last epoch per clip; T@1 by scenario at both.
5. Score-curve PNGs for the clips with the largest and the smallest error at the last epoch.
Outputs go to results/diagnostics/<run>_fold<k>/ (report.txt, CSVs, PNGs); nothing in runs/ is written.

Usage (training environment, uses the GPU for step 1):
    .venv-ml\\Scripts\\python.exe analyse_predictions.py [--run std_seed0] [--fold 0] [--rescore]
        [--run-cache-manifest PATH]  (only for a run whose frame cache was built on another machine; see
                                      cache_manifest_check.py)

Best-checkpoint scoring for test-time augmentation (--score-best --out-dir DIR [--hflip]): no report; scores every
validation window of the fold with checkpoints/best.pt (eval, bf16, the run's val batch and workers, as
train_classifier.validate) with the original frames and, with --hflip, with all 16 frames horizontally flipped
(WindowDataset flip code 2*row+1; same normalisation, nothing else changes). Writes DIR/fold_<k>/
val_window_scores_orig.csv (and _hflip.csv), the reproduction check of the original-frame scores against the run's
saved best-check predictions, and the GPU time of the classifier (forward incl. host-to-device copy, synchronised)
per pass. Defaults (no --score-best) leave the behaviour above unchanged.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
import train_classifier as tc  # noqa: E402
from accs_scorer import temporal  # noqa: E402
from cache_manifest_check import check_cache_manifest  # noqa: E402

METADATA_SYN = REPO_ROOT / "ACCIDENT" / "dataset" / "metadata-synthetic.csv"
OUT_ROOT = REPO_ROOT / "results" / "diagnostics"
SIGMAS_T = (0.5, 1.0, 2.0)
HIST_EDGES = np.arange(-10, 16, 1.0)  # 1 s bins from -10 to +15
N_WORST, N_BEST = 6, 3


def fail(msg: str) -> None:
    sys.exit(f"ERROR: {msg}")


def table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    return df.to_string(index=False, float_format=lambda v: floatfmt.format(v))


# --------------------------------------------------------------------------- #
# Inputs and checks
# --------------------------------------------------------------------------- #
def load_run(run_dir: Path, fold: int, run_cache_manifest: Path | None = None
             ) -> tuple[dict, pd.DataFrame, pd.DataFrame, str | None]:
    """Run info, validation rows of the fold (with scenario, type, t_start), metrics and the cache-deviation note
    (None if the cache manifest hash matches); labels and split hashes must match exactly, the cache manifest
    through cache_manifest_check."""
    info = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if info.get("fold") != fold or info.get("smoke_test"):
        fail(f"{run_dir}: not a full fold-{fold} run")
    cfg = info["config"]
    data, manifest, data_hashes = tc.load_labels(cfg["model_fps"])
    _, val_rows = tc.split_fold(data, fold, smoke=False)
    data_hashes["cache_manifest_sha256"] = tc.check_cache(val_rows.iloc[:0], val_rows, manifest, partial=False)
    recorded = info["data_hashes"]
    if recorded.keys() != data_hashes.keys() or any(recorded[k] != v for k, v in data_hashes.items()
                                                    if k != "cache_manifest_sha256"):
        fail("current labels / split hashes differ from the run's")
    try:
        cache_note = check_cache_manifest(recorded["cache_manifest_sha256"], tc.CACHE_MANIFEST, run_cache_manifest)
    except ValueError as e:
        fail(str(e))
    if cache_note:
        print(cache_note)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    return info, val_rows, metrics, cache_note


def score_last(run_dir: Path, info: dict, val_rows: pd.DataFrame, win_csv: Path, pred_csv: Path,
               global_step: int) -> tuple[pd.DataFrame, pd.DataFrame, dict, str]:
    """Score the validation windows with the checkpoint (last.pt or step_latest.pt) saved at `global_step`,
    the update of the last metrics row; returns (windows, clip predictions, ckpt state, checkpoint name)."""
    import torch
    from transformers import VideoMAEImageProcessor
    cfg = info["config"]
    ck, name, seen = None, None, []
    for n in ("last.pt", "step_latest.pt"):
        p = run_dir / "checkpoints" / n
        if p.is_file():
            c = torch.load(p, map_location="cpu", weights_only=False)
            seen.append(f"{n} at update {c['state']['global_step']}")
            if c["state"]["global_step"] == global_step:
                ck, name = c, n
                break
            del c
    if ck is None:
        fail(f"no checkpoint at update {global_step} (the last metrics row); found: {', '.join(seen) or 'none'}")
    if ck.get("run_hash") != info["run_hash"]:
        fail(f"{name} run_hash differs from config.json")
    if not torch.cuda.is_available():
        fail("CUDA is not available")
    device = torch.device("cuda")
    proc = VideoMAEImageProcessor.from_pretrained(cfg["model_id"])
    ds = tc.WindowDataset(val_rows["stem"], val_rows["k0"], val_rows["label_B"].clip(lower=0), None,
                          list(proc.image_mean), list(proc.image_std), cfg["frames"], cfg["model_fps"], cfg["lru_size"])
    sampler = tc.ListSampler()
    loader = tc.make_loader(ds, cfg["val_batch"], cfg["num_workers"], sampler)
    model = tc.build_model(cfg, device, None if not info["drop_path_applied"] else tc.drop_path_field())
    model.load_state_dict(ck["model"])
    state = ck["state"]
    del ck
    val_loss, T, win_df, pred, ooms = tc.validate(model, loader, sampler, val_rows, cfg, device, print)
    print(f"scored {len(win_df)} windows with {name} (state epoch {state['epoch']}, global step "
          f"{state['global_step']}); val loss {val_loss:.4f}; T@1 {T[1.0]:.4f}; OOM fallbacks {ooms}")
    win_df[["rgb_path", "window_idx", "t_centre", "label_B", "score"]].to_csv(win_csv, index=False)
    pred.to_csv(pred_csv, index=False)
    return win_df[["rgb_path", "window_idx", "t_centre", "label_B", "score"]], pred, state, name


def score_best(run_dir: Path, info: dict, val_rows: pd.DataFrame, metrics: pd.DataFrame, fold: int,
               out_dir: Path, hflip: bool) -> None:
    """Score all validation windows with best.pt, original frames and (hflip) flipped frames; see module doc."""
    import time
    import torch
    from transformers import VideoMAEImageProcessor
    cfg = info["config"]
    ck = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    best_row = metrics[metrics["best"].astype(str) == "True"].iloc[-1]
    if ck.get("run_hash") != info["run_hash"]:
        fail("best.pt run_hash differs from config.json")
    if int(ck["global_step"]) != int(best_row["global_step"]):
        fail(f"best.pt is from update {ck['global_step']}, the best metrics row from {int(best_row['global_step'])}")
    if not torch.cuda.is_available():
        fail("CUDA is not available")
    device = torch.device("cuda")
    proc = VideoMAEImageProcessor.from_pretrained(cfg["model_id"])
    ds = tc.WindowDataset(val_rows["stem"], val_rows["k0"], val_rows["label_B"].clip(lower=0), None,
                          list(proc.image_mean), list(proc.image_std), cfg["frames"], cfg["model_fps"], cfg["lru_size"])
    sampler = tc.ListSampler()
    loader = tc.make_loader(ds, cfg["val_batch"], cfg["num_workers"], sampler)
    model = tc.build_model(cfg, device, None if not info["drop_path_applied"] else tc.drop_path_field())
    model.load_state_dict(ck["model"])
    model.eval()
    fd = out_dir / f"fold_{fold}"
    fd.mkdir(parents=True, exist_ok=True)
    n_clips = val_rows["rgb_path"].nunique()
    lines = [f"run {run_dir.parent.name} fold {fold}: best.pt (check {int(ck['check_idx'])}, update {int(ck['global_step'])}, "
             f"val T@1 logged {float(ck['val_T_1']):.4f}); GPU {torch.cuda.get_device_name(0)}; {len(val_rows)} windows, "
             f"{n_clips} clips; val batch {cfg['val_batch']}, workers {cfg['num_workers']}"]
    win_cols = ["rgb_path", "window_idx", "t_centre", "label_B"]
    for name, flip in (("orig", False), ("hflip", True)) if hflip else (("orig", False),):
        sampler.set(range(int(flip), 2 * len(val_rows), 2))  # dataset codes 2*row + flip, rows in order
        logits = np.empty(len(val_rows), dtype=np.float32)
        gpu_s, t_wall = 0.0, time.perf_counter()
        with torch.no_grad():
            for x, _, _, rows, _ in loader:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                logits[rows.numpy()] = tc.val_logits(model, x, device, len(x))  # returns on CPU: synchronised
                gpu_s += time.perf_counter() - t0
        wall = time.perf_counter() - t_wall
        scores = 1.0 / (1.0 + np.exp(-logits))  # float32, as train_classifier.validate
        win = val_rows[win_cols].assign(score=scores)
        win.to_csv(fd / f"val_window_scores_{name}.csv", index=False)
        lines.append(f"{name}: classifier time {gpu_s:.1f} s = {1000 * gpu_s / n_clips:.1f} ms per clip, "
                     f"{1000 * gpu_s / len(val_rows):.2f} ms per window; wall time incl. data loading {wall:.1f} s "
                     f"({len(val_rows) / wall:.1f} windows/s)")
        if name == "orig":
            saved_w = pd.read_csv(run_dir / "val_window_scores_best.csv")
            mw = win.merge(saved_w[["rgb_path", "window_idx", "score"]], on=["rgb_path", "window_idx"],
                           suffixes=("", "_saved"), validate="one_to_one")
            _, pred = tc.clip_predictions(val_rows, scores, cfg["smooth_window"], cfg["confidence_exclude"])
            saved = pd.read_csv(run_dir / "val_clip_predictions_best.csv")
            mp = pred.merge(saved[["rgb_path", "t_hat"]], on="rgb_path", suffixes=("", "_saved"), validate="one_to_one")
            t_new = float(temporal(mp["t_hat"], mp["t_star"], 1.0))
            t_old = float(temporal(mp["t_hat_saved"], mp["t_star"], 1.0))
            lines.append(f"reproduction (orig, default rule smooth {cfg['smooth_window']}, no interpolation): window "
                         f"scores max |diff| vs saved {float((mw['score'] - mw['score_saved']).abs().max()):.2e} "
                         f"({int((mw['score'] != mw['score_saved'].astype(np.float32)).sum())} of {len(mw)} differ); "
                         f"t_hat equal in {int((mp['t_hat'] == mp['t_hat_saved']).sum())}/{len(mp)} clips; T@1 "
                         f"{t_new:.4f} vs saved {t_old:.4f} (diff {t_new - t_old:+.4f}); logged {float(best_row['val_T_1']):.4f}")
    (fd / "scoring_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #
def with_context(win: pd.DataFrame, val_rows: pd.DataFrame, smooth_window: int) -> pd.DataFrame:
    """Window scores + t_start, t_star, scenario, type, and the smoothed score used for the peak."""
    keep = ["rgb_path", "window_idx", "t_start", "t_star", "scenario_id", "ctype"]
    w = win.merge(val_rows[keep], on=["rgb_path", "window_idx"], how="left", validate="one_to_one")
    if w["t_star"].isna().any() or len(w) != len(val_rows):
        fail("window scores do not match the fold's validation windows")
    w = w.sort_values(["rgb_path", "t_centre"]).reset_index(drop=True)
    kernel = np.ones(smooth_window) / smooth_window
    w["smooth"] = np.concatenate([np.convolve(g["score"].to_numpy(float), kernel, mode="same")
                                  for _, g in w.groupby("rgb_path", sort=True)])
    return w


def summary_row(name: str, t_hat: pd.Series, t_star: pd.Series) -> dict:
    err = t_hat - t_star
    return {"predictor": name, "n_clips": len(err),
            **{f"T@{s:g}": float(temporal(t_hat, t_star, s)) for s in SIGMAS_T},
            "within_2s": float((err.abs() <= 2).mean()), "median_err_s": float(err.median()),
            "mean_err_s": float(err.mean())}


def error_section(pred: pd.DataFrame) -> list[str]:
    err = pred["err"]
    clipped = err.clip(HIST_EDGES[0], HIST_EDGES[-1] - 1e-9)
    counts, _ = np.histogram(clipped, bins=HIST_EDGES)
    hist = pd.DataFrame({"bin_s": [f"[{a:+.0f}, {b:+.0f})" for a, b in zip(HIST_EDGES[:-1], HIST_EDGES[1:])],
                         "clips": counts, "share": counts / len(err)})
    q = err.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    shares = pd.DataFrame([{"early > 2 s": (err < -2).mean(), "within +-2 s": (err.abs() <= 2).mean(),
                            "late 2-5 s": ((err > 2) & (err <= 5)).mean(), "late > 5 s": (err > 5).mean()}])
    return ["signed error t_hat - t_star (s), 1 s bins; values below -10 / at or above +15 are counted in the "
            f"first / last bin ({int((err < HIST_EDGES[0]).sum())} / {int((err >= HIST_EDGES[-1]).sum())} clips)",
            table(hist, "{:.3f}"), "",
            "percentiles: " + ", ".join(f"p{int(k * 100)} {v:+.3f}" for k, v in q.items()), "",
            "shares:", table(shares, "{:.3f}")]


def group_section(pred: pd.DataFrame, by: str) -> str:
    g = pred.groupby(by, observed=True)
    out = pd.DataFrame({"n": g.size(), "within_2s": g["err"].apply(lambda e: (e.abs() <= 2).mean()),
                        "median_err_s": g["err"].median(),
                        "T@1": g.apply(lambda d: temporal(d["t_hat"], d["t_star"], 1.0), include_groups=False)})
    return table(out.reset_index(), "{:.3f}")


def phase_of(w: pd.DataFrame) -> pd.Series:
    after = w["t_start"] - w["t_star"]
    phase = pd.Series("gap (label_B -1, ends < 1 s before t*)", index=w.index)
    phase[w["label_B"] == 0] = "pre (label_B 0)"
    phase[w["label_B"] == 1] = "near (label_B 1)"
    phase[(w["label_B"] == -1) & (after > 0) & (after <= 2)] = "aftermath: starts 0-2 s after t*"
    phase[(w["label_B"] == -1) & (after > 2) & (after <= 5)] = "aftermath: starts 2-5 s after t*"
    phase[(w["label_B"] == -1) & (after > 5)] = "aftermath: starts > 5 s after t*"
    # label_B -1 starting at or before t* must end before t* (otherwise it would be label 1)
    bad = (w["label_B"] == -1) & (after <= 0) & (w["t_start"] + 1.875 >= w["t_star"])
    if bad.any():
        fail(f"{int(bad.sum())} label_B -1 windows fit no phase")
    return phase


def phase_section(w: pd.DataFrame) -> str:
    order = ["pre (label_B 0)", "gap (label_B -1, ends < 1 s before t*)", "near (label_B 1)",
             "aftermath: starts 0-2 s after t*", "aftermath: starts 2-5 s after t*", "aftermath: starts > 5 s after t*"]
    g = w.assign(phase=phase_of(w)).groupby("phase")
    out = pd.DataFrame({"windows": g.size(), "mean_score": g["score"].mean(), "median_score": g["score"].median(),
                        "share_score>0.5": g["score"].apply(lambda s: (s > 0.5).mean())}).reindex(order)
    return table(out.reset_index(), "{:.3f}")


def late_peak_section(w: pd.DataFrame) -> str:
    rows = []
    for path, g in w.groupby("rgb_path", sort=True):
        d = g["t_centre"] - g["t_star"]
        near, late = g["smooth"][d.abs() <= 1], g["smooth"][d > 2]
        rows.append({"rgb_path": path, "has_both": len(near) > 0 and len(late) > 0,
                     "late_higher": len(near) > 0 and len(late) > 0 and late.max() > near.max()})
    r = pd.DataFrame(rows)
    n = int(r["has_both"].sum())
    return (f"clips with windows both within t* +-1 s and after t* + 2 s (by window centre): {n} of {len(r)}\n"
            f"max smoothed score after t* + 2 s > max within t* +- 1 s: {int(r['late_higher'].sum())} clips = "
            f"{r['late_higher'].sum() / max(1, n):.3f} of those {n}")


def epoch_block(label: str, w: pd.DataFrame, pred: pd.DataFrame) -> list[str]:
    out = [f"=== 3. {label} ===", "", "3a. signed error"] + error_section(pred)
    for name, col in (("collision type", "ctype"), ("scenario_id", "scenario_id"), ("t_star tercile", "t_star_tercile")):
        out += ["", f"3b. by {name}", group_section(pred, col)]
    out += ["", "3c. mean window score by phase (aftermath phases by window start relative to t*)", phase_section(w),
            "", "3d. late peak", late_peak_section(w), ""]
    return out


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_clip(path: Path, title: str, curves: list[tuple[str, pd.DataFrame, str, float]], t_star: float) -> None:
    """Window scores (dots) and 3-window smoothed scores (lines) per epoch; t* solid, each t_hat dashed."""
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=110)
    ax.axvspan(t_star - 1, t_star + 1, color="0.92", label="t* +- 1 s")
    ax.axvline(t_star, color="black", lw=2, label=f"t* = {t_star:.2f} s")
    for name, g, colour, t_hat in curves:
        ax.plot(g["t_centre"], g["smooth"], color=colour, lw=2, label=f"{name}: smoothed score")
        ax.scatter(g["t_centre"], g["score"], color=colour, s=10, alpha=0.6)
        ax.axvline(t_hat, color=colour, lw=1.5, ls="--", label=f"{name}: t_hat = {t_hat:.2f} s")
    ax.set_xlabel("window centre (s)")
    ax.set_ylabel("score")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="std_seed0")
    ap.add_argument("--fold", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--rescore", action="store_true", help="score again even if the last-epoch CSVs exist")
    ap.add_argument("--run-cache-manifest", type=Path, default=None,
                    help="the run's own cache_manifest.json, if its frame cache was built on another machine")
    ap.add_argument("--score-best", action="store_true", help="score best.pt for TTA (see module doc); no report")
    ap.add_argument("--hflip", action="store_true", help="with --score-best: also score horizontally flipped frames")
    ap.add_argument("--out-dir", type=Path, default=None, help="with --score-best: output folder (under results/)")
    args = ap.parse_args()

    run_dir = tc.RUNS_DIR / args.run / f"fold_{args.fold}"
    out_dir = OUT_ROOT / f"{args.run}_fold{args.fold}"
    info, val_rows, metrics, cache_note = load_run(run_dir, args.fold, args.run_cache_manifest)
    if args.score_best:
        if args.out_dir is None:
            fail("--score-best needs --out-dir")
        if cache_note:
            print(cache_note)
        score_best(run_dir, info, val_rows, metrics, args.fold, args.out_dir, args.hflip)
        return
    if args.hflip:
        fail("--hflip only applies with --score-best")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = info["config"]

    # labels from metrics.csv: epochs for once-per-epoch validation (v1), checks for validate_every_updates
    unit, key = ("check", "check_idx") if cfg.get("validate_every_updates") else ("epoch", "epoch")
    best_row = metrics[metrics["best"].astype(str) == "True"].iloc[-1]
    last_row = metrics.iloc[-1]
    best_epoch, last_epoch = int(best_row[key]), int(last_row[key])
    lab_b, lab_l = f"{unit} {best_epoch} (best)", f"{unit} {last_epoch} (last)"

    # 1. last-check scores
    win_csv = out_dir / f"val_window_scores_{unit}{last_epoch}.csv"
    pred_csv = out_dir / f"val_clip_predictions_{unit}{last_epoch}.csv"
    if args.rescore or not (win_csv.is_file() and pred_csv.is_file()):
        win_l, pred_l, state, ck_name = score_last(run_dir, info, val_rows, win_csv, pred_csv,
                                                   int(last_row["global_step"]))
        scored_note = f"{ck_name} (global step {state['global_step']}), scored in this run"
    else:
        win_l, pred_l = pd.read_csv(win_csv), pd.read_csv(pred_csv)
        scored_note = f"reused {win_csv.name} / {pred_csv.name} from an earlier run of this script"

    # best-epoch scores from the run
    win_b = pd.read_csv(run_dir / "val_window_scores_best.csv")
    pred_b = pd.read_csv(run_dir / "val_clip_predictions_best.csv")

    # consistency: recomputed peaks and T match what training logged
    checks = []
    for label, win, pred, ep in ((lab_b, win_b, pred_b, best_epoch), (lab_l, win_l, pred_l, last_epoch)):
        w = win.merge(val_rows[["rgb_path", "window_idx", "t_star"]], on=["rgb_path", "window_idx"])
        _, p2 = tc.clip_predictions(w, w["score"].to_numpy(), cfg["smooth_window"], cfg["confidence_exclude"])
        p2 = p2.set_index("rgb_path")
        differ = p2.index[p2["t_hat"] != pred.set_index("rgb_path").loc[p2.index, "t_hat"]]
        gaps = []  # top-2 smoothed-score gap of clips whose peak differs: ~0 means a tie broken by CSV rounding
        for c in differ:
            s = np.sort(w[w["rgb_path"] == c].sort_values("t_centre")["score"].pipe(
                lambda x: np.convolve(x.to_numpy(float), np.ones(cfg["smooth_window"]) / cfg["smooth_window"],
                                      mode="same")))
            gaps.append(s[-1] - s[-2])
        logged = float(metrics.loc[metrics[key] == ep, "val_T_1"].iloc[0])
        t1 = float(temporal(pred["t_hat"], pred["t_star"], 1.0))
        peak_note = ("peaks recomputed from the window-score CSV match the clip predictions" if not len(differ) else
                     f"peaks recomputed from the window-score CSV differ in {len(differ)} clips, all ties "
                     f"(max top-2 smoothed gap {max(gaps):.1e}) broken differently after CSV rounding"
                     if max(gaps) < 1e-9 else f"peaks DIFFER in {len(differ)} clips (max top-2 gap {max(gaps):.2e})")
        checks.append(f"{label}: {len(win)} windows, {len(pred)} clips; {peak_note}; tables use the stored clip "
                      f"predictions; T@1 {t1:.4f} vs metrics.csv {logged:.4f} (diff {t1 - logged:+.4f})")

    # clip-level context
    meta = pd.read_csv(METADATA_SYN, usecols=["rgb_path", "duration", "type"])
    clips = val_rows.groupby("rgb_path", sort=True).agg(t_star=("t_star", "first"), scenario_id=("scenario_id", "first"),
                                                        ctype=("ctype", "first")).reset_index()
    clips = clips.merge(meta, on="rgb_path", how="left", validate="one_to_one")
    if clips["duration"].isna().any():
        fail(f"{int(clips['duration'].isna().sum())} validation clips missing from {METADATA_SYN.name}")
    if (clips["type"] != clips["ctype"]).any():
        fail("metadata type differs from the rgb_path folder for some clips")
    terc = pd.qcut(clips["t_star"], 3)
    clips["t_star_tercile"] = [f"T{i + 1} t* {iv.left:.2f}-{iv.right:.2f} s" for i, iv in
                               zip(terc.cat.codes, terc)]

    def ctx(pred: pd.DataFrame) -> pd.DataFrame:
        p = pred[["rgb_path", "t_hat"]].merge(clips, on="rgb_path", validate="one_to_one")
        if len(p) != len(clips):
            fail("clip predictions do not cover the fold's validation clips")
        return p.assign(err=p["t_hat"] - p["t_star"])

    pb, pl = ctx(pred_b), ctx(pred_l)
    wb = with_context(win_b, val_rows, cfg["smooth_window"])
    wl = with_context(win_l, val_rows, cfg["smooth_window"])

    rep = [f"Prediction diagnostics: run {args.run}, fold {args.fold} (synthetic validation, "
           f"{len(clips)} clips, {len(val_rows)} windows)",
           f"config {Path(info['config_file']).name} (sha256 {info['config_hash'][:12]}...), seed {info['seed']}; "
           f"best {unit} {best_epoch} = val_*_best.csv from the run; last {unit} {last_epoch} (update "
           f"{int(last_row['global_step'])}) = {scored_note}", "", "checks:"] + [f"  {c}" for c in checks]
    rep += ([f"  {cache_note}"] if cache_note else []) + [""]

    # 2. reference
    naive = clips["duration"] / 2
    ref = pd.DataFrame([summary_row(lab_b, pb["t_hat"], pb["t_star"]), summary_row(lab_l, pl["t_hat"], pl["t_star"]),
                        summary_row("naive duration / 2", naive, clips["t_star"])])
    rep += ["=== 2. overall, with the naive reference t_hat = duration / 2 ===", table(ref), ""]

    # 3.
    rep += epoch_block(lab_b, wb, pb) + epoch_block(lab_l, wl, pl)

    # 4. best vs last per clip
    m = pb[["rgb_path", "t_hat", "t_star", "scenario_id"]].merge(pl[["rgb_path", "t_hat"]], on="rgb_path",
                                                                    suffixes=("_b", "_l"))
    d = m["t_hat_l"] - m["t_hat_b"]
    moves = pd.DataFrame([{"later": (d > 0).mean(), "earlier": (d < 0).mean(), "same": (d == 0).mean(),
                           "median shift of moved clips (s)": d[d != 0].median()}])
    by_scen = m.groupby("scenario_id").apply(lambda g: pd.Series({
        "n": len(g), f"T@1 {lab_b}": temporal(g["t_hat_b"], g["t_star"], 1.0),
        f"T@1 {lab_l}": temporal(g["t_hat_l"], g["t_star"], 1.0)}), include_groups=False).reset_index()
    by_scen["change"] = by_scen[f"T@1 {lab_l}"] - by_scen[f"T@1 {lab_b}"]
    rep += [f"=== 4. {lab_b} vs {lab_l}, per clip ===", "share of clips whose t_hat moved:", table(moves, "{:.3f}"), "",
            "T@1 by scenario:", table(by_scen, "{:.3f}"), ""]

    # 5. plots
    pl_sorted = pl.assign(abs_err=pl["err"].abs()).sort_values(["abs_err", "rgb_path"])
    worst = pl_sorted.tail(N_WORST).iloc[::-1]
    best = pl_sorted.head(N_BEST)
    rep += [f"=== 5. score-curve plots (blue: {lab_b}, red: {lab_l}) ==="]
    for kind, sel in (("largest error", worst), ("error near 0", best)):
        for _, r in sel.iterrows():
            stem = tc.stem_of(r["rgb_path"])
            eb = float(pb.set_index("rgb_path").loc[r["rgb_path"], "err"])
            title = (f"{stem}  ({r['ctype']})  t* {r['t_star']:.2f} s;  error {lab_b} {eb:+.2f} s, "
                     f"{lab_l} {r['err']:+.2f} s")
            png = out_dir / f"curve_{'worst' if kind == 'largest error' else 'best'}_{stem}.png"
            plot_clip(png, title, [(lab_b, wb[wb["rgb_path"] == r["rgb_path"]], "tab:blue",
                                    float(pb.set_index("rgb_path").loc[r["rgb_path"], "t_hat"])),
                                   (lab_l, wl[wl["rgb_path"] == r["rgb_path"]], "tab:red", float(r["t_hat"]))],
                      float(r["t_star"]))
            rep.append(f"  {kind}: {png.name}  (t* {r['t_star']:.2f}, err {lab_b} {eb:+.2f}, {lab_l} {r['err']:+.2f})")

    text = "\n".join(rep) + "\n"
    (out_dir / "report.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out_dir / 'report.txt'}")


if __name__ == "__main__":
    main()
