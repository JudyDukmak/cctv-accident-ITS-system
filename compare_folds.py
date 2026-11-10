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

Paired mode (--paired RUN_A RUN_B --folds 0,2): two runs on the SAME validation clips of each fold (e.g. an ablation
vs its baseline), descriptive only, nothing is chosen. Sections: integrity (stored configs differ only in listed
keys, equal data hashes, cctv_v2 profile = train_classifier.CCTV_V2, git commit / dirty flag); per fold and pooled
over the folds (clip-weighted): best check, T@0.5/1/2, within_2s, median error, early / late > 2 s; per weather,
collision type, scenario and scenario x weather (T@1, within_2s, differences B - A); share of clips whose |error|
improved / got worse by more than 1 s; paired bootstrap of the T@1 difference (clips resampled within each fold,
fixed seed; clips of a scenario are not independent, so the interval is optimistic) plus a scenario-cluster bootstrap
for reference; phase table (analyse_predictions.phase_of: mean window score pre / gap / near / aftermath bins,
near / aftermath ratio, late-peak share) at the best and the last check (last-check CSVs written by
analyse_predictions.py); the cctv_v2 protected-window counts parsed from train_log.txt.
Output: results/diagnostics/<name>/report.txt and CSVs.
    .venv-ml\\Scripts\\python.exe compare_folds.py --paired C_v2_seed0 C_cctv_seed0 --folds 0,2 \\
        --name C_cctv_vs_C_v2_folds02 --expect-diff augmentation,night_guard
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
import train_classifier as tc  # noqa: E402
from accs_scorer import temporal, temporal_per_clip  # noqa: E402
from analyse_predictions import phase_of  # noqa: E402
from compare_diagnostics import DIAG_ROOT, late_peak_share, load, smooth_window  # noqa: E402

WEATHERS = ("clear", "sunset", "wet", "rain", "night")
PHASES = ["pre (label_B 0)", "gap (label_B -1, ends < 1 s before t*)", "near (label_B 1)",
          "aftermath: starts 0-2 s after t*", "aftermath: starts 2-5 s after t*", "aftermath: starts > 5 s after t*"]
GUARD_RE = re.compile(r"cctv_v2 since the last check: (\d+) of (\d+) training windows protected .*"
                      r"by weather \(protected/windows\): (.*)$")


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


# --------------------------------------------------------------------------- #
# Paired mode: run B vs run A on the same validation clips
# --------------------------------------------------------------------------- #
def fmt(df: pd.DataFrame, index: bool = False) -> str:
    return df.to_string(index=index, float_format=lambda v: f"{v:.4f}")


def run_info(run: str, fold: int) -> dict:
    return json.loads((tc.RUNS_DIR / run / f"fold_{fold}" / "config.json").read_text(encoding="utf-8"))


def flat(d: dict, p: str = "") -> dict:
    out = {}
    for k, v in d.items():
        out.update(flat(v, f"{p}{k}.") if isinstance(v, dict) else {f"{p}{k}": v})
    return out


def integrity(run_a: str, run_b: str, folds: list[int], expect: set[str]) -> tuple[list[str], bool]:
    lines, ok = [], True
    missing = "<absent>"
    for k in folds:
        ia, ib = run_info(run_a, k), run_info(run_b, k)
        fa, fb = flat(ia["config"]), flat(ib["config"])
        diff = {key: (fa.get(key, missing), fb.get(key, missing)) for key in sorted(set(fa) | set(fb))
                if fa.get(key, missing) != fb.get(key, missing)}
        top = {key.split(".")[0] for key in diff}
        same_data = {h: ia["data_hashes"].get(h) == ib["data_hashes"].get(h)
                     for h in sorted(set(ia["data_hashes"]) | set(ib["data_hashes"]))}
        prof = ib.get("cctv_v2_profile")
        prof_ok = None if prof is None else prof == tc.CCTV_V2
        f_ok = (top == expect and all(same_data.values()) and prof_ok is not False and ia["fold"] == ib["fold"] == k
                and not ia.get("smoke_test") and not ib.get("smoke_test") and ia["seed"] == ib["seed"])
        ok &= f_ok
        lines += [f"fold {k}: {'OK' if f_ok else 'FAILED'}",
                  f"  {run_a}: {Path(ia['config_file']).name} hash {ia['config_hash'][:12]}..., seed {ia['seed']}, "
                  f"git {str(ia.get('git_commit'))[:7]}{' (dirty)' if ia.get('git_dirty') else ''}, {ia.get('gpu')}",
                  f"  {run_b}: {Path(ib['config_file']).name} hash {ib['config_hash'][:12]}..., seed {ib['seed']}, "
                  f"git {str(ib.get('git_commit'))[:7]}{' (dirty)' if ib.get('git_dirty') else ''}, {ib.get('gpu')}",
                  "  stored config differences (A -> B): " + "; ".join(f"{key}: {a} -> {b}" for key, (a, b) in diff.items())
                  + f"  [expected top-level keys {sorted(expect)}; got {sorted(top)}]",
                  "  data hashes equal: " + ", ".join(f"{h} {'yes' if v else 'NO'}" for h, v in same_data.items()),
                  "  cctv_v2_profile in config.json == train_classifier.CCTV_V2: "
                  + ("not recorded (profile not cctv_v2)" if prof_ok is None else "yes" if prof_ok else "NO")]
    return lines, ok


def check_rows(run: str, fold: int) -> tuple[pd.Series, pd.Series]:
    m = pd.read_csv(tc.RUNS_DIR / run / f"fold_{fold}" / "metrics.csv")
    return m[m["best"].astype(str) == "True"].iloc[-1], m.iloc[-1]


def paired_inputs(run: str, which: str, fold: int, val_rows: pd.DataFrame, clips: pd.DataFrame
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    win, pred = load(run, which, fold)
    p = pred[["rgb_path", "t_hat"]].merge(clips, on="rgb_path", validate="one_to_one")
    w = win[["rgb_path", "window_idx", "score"]].merge(
        val_rows[["rgb_path", "window_idx", "t_start", "t_end", "t_centre", "t_star", "label_B"]],
        on=["rgb_path", "window_idx"], validate="one_to_one")
    if len(p) != len(clips) or len(w) != len(val_rows):
        tc.fail(f"{run}:{which} fold {fold}: predictions do not cover the fold's validation clips / windows")
    return p.assign(err=p["t_hat"] - p["t_star"], fold=fold), w


def summ(p: pd.DataFrame) -> dict:
    e = p["err"]
    return {"n_clips": len(p), **{f"T@{s:g}": float(temporal(p["t_hat"], p["t_star"], s)) for s in (0.5, 1.0, 2.0)},
            "within_2s": float((e.abs() <= 2).mean()), "median_err_s": float(e.median()),
            "early>2s": float((e < -2).mean()), "late>2s": float((e > 2).mean())}


def group_cmp(m: pd.DataFrame, by: list[str], la: str, lb: str) -> pd.DataFrame:
    rows = []
    for key, g in m.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        ta = float(temporal(g["t_hat_A"], g["t_star"], 1.0))
        tb = float(temporal(g["t_hat_B"], g["t_star"], 1.0))
        wa, wb = float((g["err_A"].abs() <= 2).mean()), float((g["err_B"].abs() <= 2).mean())
        rows.append({**dict(zip(by, key)), "n": len(g), f"T@1 {la}": ta, f"T@1 {lb}": tb, "dT@1": tb - ta,
                     f"within_2s {la}": wa, f"within_2s {lb}": wb, "dwithin_2s": wb - wa})
    return pd.DataFrame(rows)


def moves(m: pd.DataFrame) -> dict:
    d = m["err_B"].abs() - m["err_A"].abs()
    return {"n": len(m), "improved > 1 s": float((d < -1).mean()), "worse > 1 s": float((d > 1).mean()),
            "within +-1 s": float((d.abs() <= 1).mean()), "t_hat identical": float((m["t_hat_A"] == m["t_hat_B"]).mean())}


def bootstrap(m: pd.DataFrame, n_boot: int, seed: int) -> list[str]:
    diff = temporal_per_clip(m["t_hat_B"], m["t_star"], 1.0) - temporal_per_clip(m["t_hat_A"], m["t_star"], 1.0)
    m = m.assign(d=np.asarray(diff, float))
    obs = float(m["d"].mean())
    rng = np.random.default_rng(seed)
    by_fold = [g["d"].to_numpy() for _, g in m.groupby("fold")]
    clip_bs = np.empty(n_boot)
    for b in range(n_boot):
        s = sum(float(x[rng.integers(0, len(x), len(x))].sum()) for x in by_fold)
        clip_bs[b] = s / len(m)
    # scenario clusters, resampled within each fold
    rng2 = np.random.default_rng(seed + 1)
    clusters = [[(g["d"].sum(), len(g)) for _, g in f.groupby("scenario_id")] for _, f in m.groupby("fold")]
    clu_bs = np.empty(n_boot)
    for b in range(n_boot):
        tot = cnt = 0.0
        for c in clusters:
            for i in rng2.integers(0, len(c), len(c)):
                tot, cnt = tot + c[i][0], cnt + c[i][1]
        clu_bs[b] = tot / cnt
    q = lambda a: (np.percentile(a, 2.5), np.percentile(a, 97.5))  # noqa: E731
    (l1, h1), (l2, h2) = q(clip_bs), q(clu_bs)
    nclu = sum(len(c) for c in clusters)
    return [f"observed mean per-clip T@1 difference (B - A) = {obs:+.4f} over {len(m)} clips "
            f"(= pooled T@1 B - pooled T@1 A)",
            f"clip bootstrap ({n_boot} resamples, seed {seed}, clips resampled within each fold): 95% percentile "
            f"interval [{l1:+.4f}, {h1:+.4f}]; share of resamples <= 0: {(clip_bs <= 0).mean():.3f}",
            "  clips within a scenario (same map, collision type, camera positions) are not independent, so this "
            "interval is optimistic (too narrow).",
            f"for reference, scenario-cluster bootstrap ({nclu} scenario clusters resampled within each fold, seed "
            f"{seed + 1}): [{l2:+.4f}, {h2:+.4f}]; share <= 0: {(clu_bs <= 0).mean():.3f} (few clusters: rough)"]


def phase_rows(label: str, w: pd.DataFrame, k: int) -> dict:
    ph = phase_of(w)
    means = w.groupby(ph)["score"].mean().reindex(PHASES)
    aft = w.loc[ph.str.startswith("aftermath"), "score"].mean()
    share, n = late_peak_share(w, k)
    short = ("pre", "gap", "near", "aft 0-2 s", "aft 2-5 s", "aft > 5 s")
    return {"run / check": label, **{s: float(means[p]) for s, p in zip(short, PHASES)},
            "near / aftermath (all)": float(means[PHASES[2]] / aft),
            "near / aftermath > 5 s": float(means[PHASES[2]] / means[PHASES[5]]),
            "late-peak share": share, "late-peak n": n}


def guard_counts(run: str, fold: int) -> dict:
    log = (tc.RUNS_DIR / run / f"fold_{fold}" / "train_log.txt").read_text(encoding="utf-8")
    night = [0, 0]
    other = [0, 0]
    n_lines = 0
    for line in log.splitlines():
        mt = GUARD_RE.search(line)
        if not mt:
            continue
        n_lines += 1
        for part in mt.group(3).split(", "):
            w, frac = part.split(" ")
            a, b = map(int, frac.split("/"))
            tgt = night if w == "night" else other
            tgt[0], tgt[1] = tgt[0] + a, tgt[1] + b
    return {"fold": fold, "check lines": n_lines, "resumed": "resumed from" in log,
            "night protected / loaded": f"{night[0]}/{night[1]}", "night share": night[0] / max(1, night[1]),
            "non-night protected / loaded": f"{other[0]}/{other[1]}", "non-night share": other[0] / max(1, other[1]),
            "all share": (night[0] + other[0]) / max(1, night[1] + other[1])}


def paired(args: argparse.Namespace) -> None:
    run_a, run_b = args.paired
    la, lb = args.labels.split(",") if args.labels else (run_a, run_b)
    folds = [int(f) for f in args.folds.split(",")]
    out_dir = DIAG_ROOT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    data, _, _ = tc.load_labels(8)
    weather = pd.read_csv(tc.SPLIT_CSV, usecols=["rgb_path", "weather"]).set_index("rgb_path")["weather"]
    rep = [f"Paired comparison on the same synthetic validation clips: A = {run_a} ({la}), B = {run_b} ({lb}); "
           f"folds {folds}; differences are B - A. Descriptive only; nothing is chosen.", ""]
    ilines, ok = integrity(run_a, run_b, folds, set(args.expect_diff.split(",")) if args.expect_diff else set())
    rep += ["=== 1. integrity ==="] + ilines + [""]
    if not ok:
        print("\n".join(rep))
        tc.fail("integrity check failed")

    pairs, wins, fold_rows, ph_rows = [], {}, [], []
    for k in folds:
        _, val_rows = tc.split_fold(data, k, smoke=False)
        clips = (val_rows.groupby("rgb_path", sort=True)
                 .agg(t_star=("t_star", "first"), scenario_id=("scenario_id", "first"), ctype=("ctype", "first"))
                 .reset_index())
        clips["weather"] = clips["rgb_path"].map(weather)
        per = {}
        for tag, run, lab in (("A", run_a, la), ("B", run_b, lb)):
            best, last = check_rows(run, k)
            for kind, row, which in (("best", best, "best"), ("last", last, f"check{int(last['check_idx'])}")):
                p, w = paired_inputs(run, which, k, val_rows, clips)
                per[(tag, kind)] = p
                wins[(tag, kind, k)] = w
                ph_rows.append({"fold": k, **phase_rows(f"{lab} {kind} (check {int(row['check_idx'])})", w,
                                                        smooth_window(run, k))})
            fold_rows.append({"fold": k, "run": lab, "best check": int(best["check_idx"]),
                              "epoch": float(best["epoch_fraction"]), "update": int(best["global_step"]),
                              "last check": int(last["check_idx"]), **summ(per[(tag, "best")])})
        m = per[("A", "best")].merge(per[("B", "best")][["rgb_path", "t_hat", "err"]], on="rgb_path",
                                     suffixes=("_A", "_B"), validate="one_to_one")
        if len(m) != len(clips):
            tc.fail(f"fold {k}: the two runs do not cover the same clips")
        pairs.append(m)
    m = pd.concat(pairs, ignore_index=True)
    ft = pd.DataFrame(fold_rows)
    pooled = []
    for tag, lab in (("A", la), ("B", lb)):
        pa = m.rename(columns={f"t_hat_{tag}": "t_hat", f"err_{tag}": "err"})
        pooled.append({"fold": "+".join(map(str, folds)), "run": lab, "best check": "per fold", **summ(pa)})
    ft = pd.concat([ft, pd.DataFrame(pooled)], ignore_index=True)
    met = ["T@0.5", "T@1", "T@2", "within_2s", "median_err_s", "early>2s", "late>2s"]
    diff_rows = []
    for f in [str(k) for k in folds] + ["+".join(map(str, folds))]:
        g = ft[ft["fold"].astype(str) == f]
        diff_rows.append({"fold": f, **{c: float(g[c].iloc[1] - g[c].iloc[0]) for c in met}})
    rep += ["=== 4a. per fold and pooled (best check of each run; pooled = clip-weighted over the folds) ===", fmt(ft),
            "", "differences B - A:", fmt(pd.DataFrame(diff_rows)), ""]

    gw = pd.concat([group_cmp(m[m["fold"] == k], ["weather"], la, lb).assign(fold=str(k)) for k in folds]
                   + [group_cmp(m, ["weather"], la, lb).assign(fold="pooled")], ignore_index=True)
    gw["weather"] = pd.Categorical(gw["weather"], WEATHERS)
    gw = gw.sort_values(["fold", "weather"])[["fold"] + [c for c in gw.columns if c != "fold"]]
    gt = group_cmp(m, ["ctype"], la, lb)
    gs = group_cmp(m, ["fold", "scenario_id"], la, lb)
    gsw = group_cmp(m, ["fold", "scenario_id", "weather"], la, lb)
    hl = gsw[gsw["scenario_id"].eq("Town04_rear-end") & gsw["weather"].isin(["night", "clear"])]
    rep += ["=== 4b. per weather (best check) ===", fmt(gw), "",
            "=== 4c. per collision type (pooled) ===", fmt(gt), "",
            "per scenario (scenarios belong to one validation fold):", fmt(gs), "",
            "highlight Town04_rear-end night / clear:" + ("" if len(hl) else " not in these folds"),
            fmt(hl) if len(hl) else "", ""]
    mv = pd.DataFrame([{"fold": str(k), **moves(m[m["fold"] == k])} for k in folds]
                      + [{"fold": "pooled", **moves(m)}]
                      + [{"fold": f"pooled {w}", **moves(m[m["weather"] == w])} for w in WEATHERS])
    rep += ["=== 4d. clips whose |t_hat - t*| changed by more than 1 s (B vs A, best checks) ===", fmt(mv), ""]
    rep += ["=== 4e. paired bootstrap of the T@1 difference, folds pooled ==="] + bootstrap(m, args.bootstrap,
                                                                                          args.seed) + [""]
    pt = pd.DataFrame(ph_rows)
    pooled_ph = []
    for tag, run, lab in (("A", run_a, la), ("B", run_b, lb)):
        for kind in ("best", "last"):
            w = pd.concat([wins[(tag, kind, k)] for k in folds], ignore_index=True)
            r = phase_rows(f"{lab} {kind}", w, smooth_window(run, folds[0]))
            pooled_ph.append({"fold": "+".join(map(str, folds)), **r})
    pt = pd.concat([pt, pd.DataFrame(pooled_ph)], ignore_index=True)
    rep += ["=== 5. mean window score by phase, near / aftermath ratio and late-peak share (best and last check) ===",
            "phases as analyse_predictions.phase_of; aftermath bins by window start relative to t*; pooled = windows of "
            "all listed folds together (window-weighted); late-peak share = compare_diagnostics.late_peak_share; last "
            "checks are rescored by analyse_predictions.py on this machine", fmt(pt), ""]
    gc = pd.DataFrame([guard_counts(run_b, k) for k in folds])
    rep += [f"=== 6. {lb}: protected training windows from train_log.txt (counts of windows loaded, summed over all "
            "validation-check intervals; a window is counted each time it is loaded) ===", fmt(gc), ""]

    text = "\n".join(rep) + "\n"
    (out_dir / "report.txt").write_text(text, encoding="utf-8")
    m.to_csv(out_dir / "paired_clips_best.csv", index=False)
    for nm, df in (("per_fold", ft), ("per_fold_diff", pd.DataFrame(diff_rows)), ("per_weather", gw),
                   ("per_type", gt), ("per_scenario", gs), ("per_scenario_weather", gsw), ("moves", mv),
                   ("phases", pt), ("guard_counts", gc)):
        df.to_csv(out_dir / f"{nm}.csv", index=False)
    print(text)
    print(f"wrote {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("items", nargs="*", help="LABEL=RUN:WHICH:FOLD")
    ap.add_argument("--name", required=True)
    ap.add_argument("--paired", nargs=2, metavar=("RUN_A", "RUN_B"), default=None)
    ap.add_argument("--labels", default=None, help="paired: 'LABEL_A,LABEL_B' (default the run names)")
    ap.add_argument("--folds", default="0,1,2", help="paired: comma-separated folds")
    ap.add_argument("--expect-diff", default="", help="paired: top-level config keys allowed (and expected) to differ")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.paired:
        paired(args)
        return
    if not args.items:
        ap.error("give LABEL=RUN:WHICH:FOLD items or --paired RUN_A RUN_B")

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
