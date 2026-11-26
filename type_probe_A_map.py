"""Type probe A, second attempt: leave-one-map-out (synthetic only; CPU only; no new features).

Pre-registration: results/type_probe_A_map/PLAN.txt (sha256 checked by every mode). Reads only the first attempt's
saved features (cache/type_probe_A/features_fold<k>.npz), its windows.csv and metrics.json, and the QLoRA epoch-1
answers; changes none of them. Never opens a real clip, real label or real metadata file. Run with .venv-ml.

  --probe   6 rounds (one per map) x encoder; inner leave-one-map-out CV over C; predictions, metrics, baselines,
            scenario bootstrap.
  --report  report.txt and confusion.png.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import type_probe_A as A  # noqa: E402  (fit_lr / train_predict / metrics of the first attempt, imported not copied)

OUT = REPO / "results" / "type_probe_A_map"
PLAN = OUT / "PLAN.txt"
PLAN_SHA = "15d5e5314338d1b719c29625c31b31d9e1526e41830b719a66cbd716e607d8a9"
FIRST = REPO / "results" / "type_probe_A"
MAPS = ["Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"]
TYPES, CS = A.TYPES, A.CS
N_BOOT, BOOT_SEED = 2000, 0
INNER_CSV, PRED_CSV, METRICS_JSON, REPORT = OUT / "inner_cv.csv", OUT / "predictions_oof.csv", OUT / "metrics.json", OUT / "report.txt"


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def check_plan():
    if hashlib.sha256(PLAN.read_bytes()).hexdigest() != PLAN_SHA:
        fail("PLAN.txt changed since pre-registration")


def refuse(*paths):
    for p in paths:
        if p.exists():
            fail(f"{p} exists; refusing to overwrite")


def load():
    w = pd.read_csv(FIRST / "windows.csv")
    w["map"] = w["scenario_id"].str.split("_").str[0]
    if len(w) != 2211 or sorted(w["map"].unique()) != MAPS:
        fail("expected 2211 clips on 6 maps")
    w["y"] = w["type"].map({t: i for i, t in enumerate(TYPES)})
    feats = {}
    for k in A.FOLDS:
        z = np.load(A.CACHE / f"features_fold{k}.npz", allow_pickle=True)
        key = {(r, int(i)): j for j, (r, i) in enumerate(zip(z["rgb_path"], z["window_idx"]))}
        F = z["feat"].astype(np.float64).mean(1)  # mean of original and flipped features
        feats[k] = (F[[key[(r, i)] for r, i in zip(w["rgb_path"], w["w_tstar"])]],
                    F[[key[(r, i)] for r, i in zip(w["rgb_path"], w["w_that"])]])
    return w, feats


def macro_of(y, p):
    present = np.unique(y)
    return float(np.mean([(p[y == t] == t).mean() for t in present]))


def probe():
    check_plan()
    refuse(INNER_CSV, PRED_CSV, METRICS_JSON)
    w, feats = load()
    maps = w["map"].to_numpy()
    inner_rows, preds = [], []
    for M in MAPS:
        te_map = maps == M
        tr = ~te_map
        # majority type of the training maps (clip counts; ties -> alphabetical first)
        vc = w.loc[tr, "type"].value_counts()
        maj = sorted(vc[vc == vc.max()].index)[0]
        for k in sorted(w.loc[te_map, "fold"].unique()):
            te = te_map & (w["fold"] == k).to_numpy()
            fs, fh = feats[k]
            y_tr, m_tr = w.loc[tr, "y"].to_numpy(), maps[tr]
            Xs_tr, Xh_tr = fs[tr], fh[tr]
            res_c = {}
            for C in CS:
                p_in = np.empty(tr.sum(), int)
                gmax = 0.0
                for m in sorted(set(m_tr)):
                    a, h = m_tr != m, m_tr == m
                    p_in[h], g = A.train_predict(np.vstack([Xs_tr[a], Xh_tr[a]]), np.concatenate([y_tr[a], y_tr[a]]),
                                                 Xh_tr[h], C)
                    gmax = max(gmax, g)
                mi = A.metrics(y_tr, p_in)
                res_c[C] = mi["macro_accuracy"]
                inner_rows.append({"test_map": M, "encoder_fold": int(k), "C": C, "inner_macro_accuracy": mi["macro_accuracy"],
                                   "inner_accuracy": mi["accuracy"], "max_rel_grad": gmax})
                print(f"{M} encoder {k} C {C:g}: inner macro {mi['macro_accuracy']:.4f} acc {mi['accuracy']:.4f} (rel grad {gmax:.1e})")
            best = max(res_c.values())
            C_best = min(C for C in CS if res_c[C] == best)
            p_te, g = A.train_predict(np.vstack([fs[tr], fh[tr]]), np.concatenate([y_tr, y_tr]), fh[te], C_best)
            preds.append(w.loc[te, ["rgb_path", "map", "scenario_id", "fold", "type"]].assign(
                encoder_fold=int(k), C=C_best, outer_rel_grad=g, pred_type=[TYPES[i] for i in p_te], majority_type=maj))
            print(f"{M} encoder {k}: chosen C {C_best:g}; {int(te.sum())} test clips")
    P = pd.concat(preds).sort_values("rgb_path").reset_index(drop=True)
    if len(P) != 2211 or P["rgb_path"].duplicated().any():
        fail("predictions do not cover the 2211 clips exactly once")
    tix = {t: i for i, t in enumerate(TYPES)}
    y, p, pm = (P[c].map(tix).to_numpy() for c in ("type", "pred_type", "majority_type"))
    rounds = {}
    for M in MAPS:
        s = (P["map"] == M).to_numpy()
        rounds[M] = {"probe": A.metrics(y[s], p[s]), "majority": A.metrics(y[s], pm[s]),
                     "majority_type": P.loc[s, "majority_type"].iloc[0],
                     "encoders": {str(k): {"n": int(((P["map"] == M) & (P["encoder_fold"] == k)).sum()),
                                           "C": float(P.loc[(P["map"] == M) & (P["encoder_fold"] == k), "C"].iloc[0])}
                                  for k in sorted(P.loc[s, "encoder_fold"].unique())}}
    pooled, pooled_maj = A.metrics(y, p), A.metrics(y, pm)
    # fold-1 clips: QLoRA epoch 1 vs the probe on the same clips
    q = pd.read_csv(A.QLORA_E1)
    f1 = P[P["fold"] == 1]
    if set(q["rgb_path"]) != set(f1["rgb_path"]):
        fail("QLoRA epoch-1 clips differ from the fold-1 clips")
    qm = json.loads(A.QLORA_E1_MET.read_text())
    mq = A.metrics(q["type"].map(tix).to_numpy(), q["pred_type"].map(tix).to_numpy())
    if abs(mq["accuracy"] - qm["type_accuracy"]) > 1e-9:
        fail("recomputed QLoRA epoch-1 accuracy differs from its metrics file")
    fold1 = {"probe": A.metrics(f1["type"].map(tix).to_numpy(), f1["pred_type"].map(tix).to_numpy()),
             "majority_of_training_maps": A.metrics(f1["type"].map(tix).to_numpy(), f1["majority_type"].map(tix).to_numpy()),
             "qlora_epoch1": mq}
    # scenario bootstrap of the pooled macro accuracy (fixed predictions)
    scen = P["scenario_id"].to_numpy()
    groups = {s: np.flatnonzero(scen == s) for s in sorted(set(scen))}
    names = list(groups)
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([groups[names[i]] for i in rng.integers(0, len(names), len(names))])
        boot[b] = macro_of(y[idx], p[idx])
    lo, hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    success = bool(pooled["macro_accuracy"] > 0.20 and lo > 0.20)
    res = {"rounds": rounds, "pooled": pooled, "pooled_majority": pooled_maj, "random_guess_macro": 0.20,
           "fold1": fold1, "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "unit": "scenario", "n_scenarios": len(names),
                                         "ci95": [lo, hi], "mean": float(boot.mean())}, "success": success}
    pd.DataFrame(inner_rows).to_csv(INNER_CSV, index=False)
    P.to_csv(PRED_CSV, index=False)
    for name, cm in [("pooled", pooled["confusion"])] + [(f"map_{M}", rounds[M]["probe"]["confusion"]) for M in MAPS]:
        c = np.array(cm, float)
        idx, cols = [f"true {t}" for t in TYPES], [f"pred {t}" for t in TYPES]
        pd.DataFrame(c.astype(int), index=idx, columns=cols).to_csv(OUT / f"confusion_{name}_counts.csv")
        rn = np.divide(c, c.sum(1, keepdims=True), out=np.full_like(c, np.nan), where=c.sum(1, keepdims=True) > 0)
        pd.DataFrame(rn, index=idx, columns=cols).to_csv(OUT / f"confusion_{name}_rownorm.csv")
    METRICS_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"pooled macro {pooled['macro_accuracy']:.4f} acc {pooled['accuracy']:.4f}; bootstrap 95% [{lo:.4f}, {hi:.4f}]; "
          f"SUCCESS {success}")


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    check_plan()
    refuse(REPORT, OUT / "confusion.png")
    M_ = json.loads(METRICS_JSON.read_text())
    F_ = json.loads((FIRST / "metrics.json").read_text())
    inner = pd.read_csv(INNER_CSV)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    c = np.array(M_["pooled"]["confusion"], float)
    for ax, (title, v, norm) in zip(axes, [("pooled leave-one-map-out, counts", c, False),
                                          ("pooled leave-one-map-out, row-normalised", c / c.sum(1, keepdims=True), True)]):
        ax.imshow(v, cmap="Blues", vmin=0, vmax=1 if norm else v.max())
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{v[i, j]:.2f}" if norm else f"{int(v[i, j])}", ha="center", va="center",
                        color="white" if v[i, j] > (0.6 if norm else 0.6 * v.max()) else "black", fontsize=9)
        ax.set_xticks(range(5), TYPES, rotation=30)
        ax.set_yticks(range(5), TYPES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(title)
    fig.suptitle("Type probe A, second attempt (leave one map out), C_cctv seed 0 frozen features, t_hat window")
    fig.savefig(OUT / "confusion.png", dpi=150)
    plt.close(fig)
    f = lambda x: f"{x:.4f}"  # noqa: E731
    b = M_["bootstrap"]
    L = ["TYPE PROBE A, SECOND ATTEMPT: LEAVE ONE MAP OUT (synthetic only; CPU only; no new features; the first "
         "attempt's files, checkpoints, split and manifest unchanged; nothing committed)",
         f"PLAN.txt sha256 {PLAN_SHA} (checked by every mode). Decided after seeing the first attempt's result.", "",
         "1. INNER CV (leave one training map out; pooled inner macro accuracy per C) AND CHOSEN C"]
    for (m, k), g in inner.groupby(["test_map", "encoder_fold"], sort=False):
        L.append(f"  test {m:9s} encoder fold {k}: " + ", ".join(f"C {r.C:g} {r.inner_macro_accuracy:.4f}" for r in g.itertuples())
                 + f"  -> C {M_['rounds'][m]['encoders'][str(k)]['C']:g} ({M_['rounds'][m]['encoders'][str(k)]['n']} test clips)")
    L += ["", "2. PER ROUND (test map) AND POOLED; macro accuracy over the types present",
          f"  {'set':9s} {'n':>5s} {'types':>5s} {'probe macro':>11s} {'probe acc':>9s} {'majority (type)':>26s} {'maj macro':>9s} {'maj acc':>8s}  probe recall per type"]
    for m in MAPS:
        r = M_["rounds"][m]
        L.append(f"  {m:9s} {r['probe']['n']:5d} {r['probe']['n_types_in_macro']:5d} {f(r['probe']['macro_accuracy']):>11s} "
                 f"{f(r['probe']['accuracy']):>9s} {r['majority_type']:>26s} {f(r['majority']['macro_accuracy']):>9s} "
                 f"{f(r['majority']['accuracy']):>8s}  " + ", ".join(f"{t} {v:.3f}" for t, v in r["probe"]["recall"].items()))
    p, pm = M_["pooled"], M_["pooled_majority"]
    L.append(f"  {'pooled':9s} {p['n']:5d} {p['n_types_in_macro']:5d} {f(p['macro_accuracy']):>11s} {f(p['accuracy']):>9s} "
             f"{'(per-round majority)':>26s} {f(pm['macro_accuracy']):>9s} {f(pm['accuracy']):>8s}  "
             + ", ".join(f"{t} {v:.3f}" for t, v in p["recall"].items()))
    L += [f"  random guess: macro accuracy 0.20 (expected value)",
          "", "  pooled confusion (rows true, columns predicted; " + ", ".join(TYPES) + "):"]
    L += ["    " + f"{t:10s} " + " ".join(f"{v:5d}" for v in row) for t, row in zip(TYPES, p["confusion"])]
    L += ["  CSV: confusion_pooled_{counts,rownorm}.csv and confusion_map_<map>_{counts,rownorm}.csv; PNG: confusion.png",
          "", "3. FOLD-1 CLIPS (755): same clips, three predictors", f"  {'model':40s} {'accuracy':>9s} {'macro acc':>10s}"]
    for name, key in (("majority of the training maps (per round)", "majority_of_training_maps"),
                      ("QLoRA epoch 1 (window A)", "qlora_epoch1"), ("probe, leave one map out", "probe")):
        m = M_["fold1"][key]
        L.append(f"  {name:40s} {m['accuracy']:9.4f} {m['macro_accuracy']:10.4f}")
    L += ["", f"4. SUCCESS RULE: pooled macro accuracy {f(p['macro_accuracy'])}; 95% bootstrap interval "
          f"[{f(b['ci95'][0])}, {f(b['ci95'][1])}] ({b['n']} draws, seed {b['seed']}, resampling the {b['n_scenarios']} "
          f"scenarios, fixed predictions). Rule: > 0.20 and the interval excludes 0.20 -> "
          f"{'MET' if M_['success'] else 'NOT MET'}",
          ("  Met: reported only; other seeds are not run now (as planned)." if M_["success"] else
           "  Not met."), "",
          "5. FIRST ATTEMPT (results/type_probe_A/, scenario folds), side by side",
          f"  {'':26s} {'first: macro':>12s} {'first: acc':>10s}   {'second: macro':>13s} {'second: acc':>11s}",
          f"  {'pooled (2211)':26s} {F_['pooled']['macro_accuracy']:12.4f} {F_['pooled']['accuracy']:10.4f}   "
          f"{p['macro_accuracy']:13.4f} {p['accuracy']:11.4f}",
          f"  {'fold-1 clips (755)':26s} {F_['folds']['1']['macro_accuracy']:12.4f} {F_['folds']['1']['accuracy']:10.4f}   "
          f"{M_['fold1']['probe']['macro_accuracy']:13.4f} {M_['fold1']['probe']['accuracy']:11.4f}",
          f"  first attempt per fold (macro / acc): " + "; ".join(
              f"fold {k} {F_['folds'][k]['macro_accuracy']:.4f} / {F_['folds'][k]['accuracy']:.4f}" for k in ("0", "1", "2")),
          f"  first attempt pooled recall: " + ", ".join(f"{t} {v:.3f}" for t, v in F_["pooled"]["recall"].items()),
          f"  second attempt pooled recall: " + ", ".join(f"{t} {v:.3f}" for t, v in p["recall"].items()), "",
          "6. LIMITS",
          "  The encoder of each test clip is fold model k = its val_fold (seed 0). For Town04, Town05 and Town10HD it saw "
          "no scenario of the test map; for Town03, Town06 and Town07 it was trained on another scenario of the test map "
          "(accident detection, not type). The probe's training rows come from clips the encoder was trained on. One "
          "encoder seed; 6 maps and 14 scenarios, so the interval is wide; types are tied to maps (each map has 2-3 of "
          "the 5 types), so a round can only show 2-3 types.", "",
          "Files: PLAN.txt, inner_cv.csv, predictions_oof.csv, metrics.json, confusion_*.csv, confusion.png, run_log_*.txt, "
          "report.txt."]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--probe", action="store_true")
    g.add_argument("--report", action="store_true")
    a = ap.parse_args()
    probe() if a.probe else report()


if __name__ == "__main__":
    main()
