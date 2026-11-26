"""Type probe, option A: frozen window-Classifier features + a linear probe (synthetic only).

Pre-registration: results/type_probe_A/PLAN.txt (written before any feature; its sha256 is checked by every mode).
SYNTHETIC ONLY: never opens, lists or reads a real clip, real label or real metadata file. Changes no Classifier
checkpoint, frozen file, split or manifest (all are only read). Run with the .venv-ml interpreter.

Modes (in order; each refuses to overwrite its outputs):
  --safety   (GPU) per fold model k = 0, 1, 2: 5 fold-k clips (numpy default_rng(0) over the sorted fold-k clips),
             every window, original and flipped frames: hooked head output == model logit exactly, and the saved head
             applied to the extracted feature (bf16 autocast, as the model) -> sigmoid within 0.005 of the saved
             window score on this GPU. Writes safety.json; any failure stops the script.
  --extract  (GPU) after a passing safety check: windows.csv (per clip: the t* window and the frozen out-of-fold t_hat
             window), then, per fold model k, the features (fc_norm(mean last hidden state), 768) of every chosen
             window, original and flipped, to cache/type_probe_A/features_fold<k>.npz; the fold-k windows' logits are
             compared with the saved scores (extract_check.json). Mains power is checked before each fold model.
  --probe    (CPU) per validation fold k: leave-one-scenario-out inner CV over C in {0.01, 0.1, 1, 10} on the training
             folds, outer fit, out-of-fold predictions; metrics, confusion matrices, comparison on fold 1.
  --report   report.txt and the confusion PNG.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("HF_HUB_OFFLINE", "1")
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
OUT = REPO / "results" / "type_probe_A"
CACHE = REPO / "cache" / "type_probe_A"
PLAN = OUT / "PLAN.txt"
PLAN_SHA = "abc245b68809ec1f2af2215e4f9eae1a02451cc0659f9e2dc0e503ec4e6fc156"
RUN = "C_cctv_seed0"
FOLDS = (0, 1, 2)
TTHAT = REPO / "results" / "diagnostics" / "C_cctv_seeds012" / "seed0_frozen" / "tta_clip_predictions_flip.csv"
SAVED_SCORES = REPO / "results" / "diagnostics" / "postproc_C_cctv_seed0" / "tta"
QLORA_E1 = REPO / "results" / "qlora_train" / "val_epoch1.csv"
QLORA_E1_MET = REPO / "results" / "qlora_train" / "val_epoch1_metrics.json"
TYPES = ["head-on", "rear-end", "sideswipe", "single", "t-bone"]
CS = (0.01, 0.1, 1.0, 10.0)
SAFETY_CLIPS, SAFETY_TOL = 5, 0.005
SAFETY_JSON, EXTRACT_JSON, WINDOWS_CSV = OUT / "safety.json", OUT / "extract_check.json", OUT / "windows.csv"
INNER_CSV, PRED_CSV, METRICS_JSON = OUT / "inner_cv.csv", OUT / "predictions_oof.csv", OUT / "metrics.json"
COMP_CSV, REPORT = OUT / "comparison_fold1.csv", OUT / "report.txt"


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def check_plan():
    if hashlib.sha256(PLAN.read_bytes()).hexdigest() != PLAN_SHA:
        fail("PLAN.txt changed since pre-registration")


def refuse(*paths):
    for p in paths:
        if p.exists():
            fail(f"{p} exists; refusing to overwrite")


# ---------------------------------------------------------------- windows

def window_table():
    """Per clip: fold, scenario, type, t*, frozen OOF t_hat, and the nearest-centre Classifier window for each."""
    import train_classifier as tc
    data, _, _ = tc.load_labels(8)
    th = pd.read_csv(TTHAT)
    if len(th) != 2211 or th["rgb_path"].duplicated().any():
        fail(f"{TTHAT.name}: expected 2211 clips")
    rows = []
    thd = th.set_index("rgb_path")
    for p, g in data.groupby("rgb_path", sort=True):
        g = g.sort_values("t_centre")
        if p not in thd.index:
            fail(f"{p}: no frozen t_hat")
        r = thd.loc[p]
        if int(r["fold"]) != int(g["val_fold"].iloc[0]):
            fail(f"{p}: t_hat fold {r['fold']} != val_fold {g['val_fold'].iloc[0]} (not out-of-fold)")
        tc_ = g["t_centre"].to_numpy(float)
        ts, that = float(g["t_star"].iloc[0]), float(r["t_hat"])
        i_s, i_h = int(np.argmin(np.abs(tc_ - ts))), int(np.argmin(np.abs(tc_ - that)))  # ties -> earlier window
        gs, gh = g.iloc[i_s], g.iloc[i_h]
        rows.append({"rgb_path": p, "stem": gs["stem"], "fold": int(gs["val_fold"]), "scenario_id": gs["scenario_id"],
                     "type": gs["ctype"], "t_star": ts, "t_hat": that,
                     "w_tstar": int(gs["window_idx"]), "k0_tstar": int(gs["k0"]), "tc_tstar": float(gs["t_centre"]),
                     "w_that": int(gh["window_idx"]), "k0_that": int(gh["k0"]), "tc_that": float(gh["t_centre"])})
    w = pd.DataFrame(rows)
    if len(w) != 2211 or set(w["type"]) != set(TYPES):
        fail("window table: expected 2211 clips and 5 types")
    return w


# ---------------------------------------------------------------- GPU

class Extractor:
    """Fold model k (best.pt, read only) and a hook on its classification head."""

    def __init__(self, k):
        import torch
        import analyse_predictions as ap
        import train_classifier as tc
        from transformers import VideoMAEImageProcessor
        self.torch, self.tc = torch, tc
        info, ck, best = ap.load_best(RUN, k)
        cfg = info["config"]
        self.cfg, self.info, self.ck = cfg, info, ck
        self.device = torch.device("cuda")
        proc = VideoMAEImageProcessor.from_pretrained(cfg["model_id"])
        self.mean, self.std = list(proc.image_mean), list(proc.image_std)
        if not (np.allclose(self.mean, info["normalisation"]["mean"]) and np.allclose(self.std, info["normalisation"]["std"])):
            fail("processor mean/std differ from the run's normalisation")
        self.model = tc.build_model(cfg, self.device, None if not info["drop_path_applied"] else tc.drop_path_field())
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        if self.model.fc_norm is None:
            fail("model has no fc_norm (not mean pooling); PLAN feature definition does not apply")
        self._cap = []
        self.model.classifier.register_forward_hook(lambda m, i, o: self._cap.append((i[0].detach(), o.detach())))
        self.gpu_s = 0.0

    def run(self, stems, k0s):
        """features (n, 2, 768) float32 [orig, flip], hooked logits (n, 2), model logits (n, 2), head(feature) (n, 2)."""
        torch, tc = self.torch, self.tc
        ds = tc.WindowDataset(pd.Series(stems), pd.Series(k0s), pd.Series(np.zeros(len(stems))), None, self.mean,
                              self.std, self.cfg["frames"], self.cfg["model_fps"], self.cfg["lru_size"])
        sampler = tc.ListSampler()
        loader = tc.make_loader(ds, self.cfg["val_batch"], self.cfg["num_workers"], sampler)
        n = len(stems)
        feat = np.empty((n, 2, 768), np.float32)
        hook_l, model_l, head_l = (np.empty((n, 2), np.float32) for _ in range(3))
        for flip in (0, 1):
            sampler.set(range(flip, 2 * n, 2))
            with torch.no_grad():
                for x, _, _, rows, _ in loader:
                    self._cap.clear()
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = self.model(pixel_values=x.to(self.device, non_blocking=True)).logits.squeeze(-1)
                    if len(self._cap) != 1:
                        fail("head hook did not fire exactly once")
                    f, hl = self._cap[0]
                    with torch.autocast("cuda", dtype=torch.bfloat16):  # the saved head applied to the feature
                        re = self.model.classifier(f).squeeze(-1)
                    torch.cuda.synchronize()
                    self.gpu_s += time.perf_counter() - t0
                    r = rows.numpy()
                    feat[r, flip] = f.float().cpu().numpy()
                    hook_l[r, flip] = hl.squeeze(-1).float().cpu().numpy()
                    model_l[r, flip] = out.float().cpu().numpy()
                    head_l[r, flip] = re.float().cpu().numpy()
                    self._cap.clear()
        return feat, hook_l, model_l, head_l


def saved_scores(k):
    d = {}
    for i, name in enumerate(("orig", "hflip")):
        s = pd.read_csv(SAVED_SCORES / f"fold_{k}" / f"val_window_scores_{name}.csv")
        d[i] = s.set_index(["rgb_path", "window_idx"])["score"]
    return d


def compare(rgb, widx, head_l, model_l, hook_l, sv):
    sig = lambda z: 1.0 / (1.0 + np.exp(-z.astype(np.float64)))  # noqa: E731
    diffs = []
    for flip in (0, 1):
        ref = sv[flip].reindex(list(zip(rgb, widx))).to_numpy(float)
        if np.isnan(ref).any():
            fail("a window has no saved score")
        diffs.append(np.abs(sig(head_l[:, flip]) - ref))
    d = np.concatenate(diffs)
    return {"n_window_passes": int(d.size), "max_abs_score_diff_head_on_feature": float(d.max()),
            "mean_abs_score_diff": float(d.mean()),
            "hook_equals_model_logit": bool(np.array_equal(hook_l, model_l)),
            "head_on_feature_equals_model_logit": bool(np.array_equal(head_l, model_l)),
            "max_abs_logit_diff_head_vs_model": float(np.abs(head_l - model_l).max())}


def safety():
    import torch
    check_plan()
    refuse(SAFETY_JSON)
    if not torch.cuda.is_available():
        fail("CUDA not available")
    import train_classifier as tc
    import vlm_zero_shot as vz
    on, ptxt = vz.power()
    print(f"power {ptxt}")
    if not on:
        fail("on battery; not starting")
    data, _, _ = tc.load_labels(8)
    res = {"gpu": torch.cuda.get_device_name(0), "tolerance": SAFETY_TOL, "power": ptxt, "folds": {}}
    ok = True
    for k in FOLDS:
        clips = sorted(data.loc[data["val_fold"] == k, "rgb_path"].unique())
        pick = list(np.random.default_rng(0).choice(clips, size=SAFETY_CLIPS, replace=False))
        w = data[data["rgb_path"].isin(pick)].sort_values(["rgb_path", "window_idx"])
        ex = Extractor(k)
        feat, hook_l, model_l, head_l = ex.run(w["stem"].tolist(), w["k0"].tolist())
        c = compare(w["rgb_path"].to_numpy(), w["window_idx"].to_numpy(), head_l, model_l, hook_l, saved_scores(k))
        c.update({"clips": pick, "n_windows": len(w), "feature_dim": int(feat.shape[-1]),
                  "best_check": int(ex.ck["check_idx"]), "best_update": int(ex.ck["global_step"]), "gpu_s": ex.gpu_s})
        c["pass"] = c["hook_equals_model_logit"] and c["max_abs_score_diff_head_on_feature"] <= SAFETY_TOL
        ok &= c["pass"]
        res["folds"][str(k)] = c
        print(f"fold model {k}: {json.dumps({x: c[x] for x in c if x != 'clips'})}")
        del ex
        torch.cuda.empty_cache()
    res["pass"] = ok
    OUT.mkdir(exist_ok=True)
    SAFETY_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    if not ok:
        fail("SAFETY CHECK FAILED; stopping (no features extracted)")
    print("SAFETY CHECK PASS")


def extract():
    import torch
    import vlm_zero_shot as vz
    check_plan()
    if not SAFETY_JSON.is_file() or not json.loads(SAFETY_JSON.read_text())["pass"]:
        fail("run --safety first (it must pass)")
    refuse(WINDOWS_CSV, EXTRACT_JSON)
    CACHE.mkdir(parents=True, exist_ok=True)
    w = window_table()
    w.to_csv(WINDOWS_CSV, index=False)
    print(f"windows.csv: {len(w)} clips; t* window == t_hat window in {int((w['w_tstar'] == w['w_that']).sum())}")
    uw = pd.concat([w[["rgb_path", "stem", "fold", "w_tstar", "k0_tstar"]].set_axis(["rgb_path", "stem", "fold", "window_idx", "k0"], axis=1),
                    w[["rgb_path", "stem", "fold", "w_that", "k0_that"]].set_axis(["rgb_path", "stem", "fold", "window_idx", "k0"], axis=1)])
    uw = uw.drop_duplicates(["rgb_path", "window_idx"]).sort_values(["rgb_path", "window_idx"]).reset_index(drop=True)
    res = {"n_unique_windows": len(uw), "folds": {}}
    for k in FOLDS:
        f_out = CACHE / f"features_fold{k}.npz"
        refuse(f_out)
        on, ptxt = vz.power()
        print(f"fold model {k}: power {ptxt}")
        if not on:
            fail("on battery; stopping before fold model " + str(k))
        ex = Extractor(k)
        t0 = time.perf_counter()
        feat, hook_l, model_l, head_l = ex.run(uw["stem"].tolist(), uw["k0"].tolist())
        wall = time.perf_counter() - t0
        np.savez(f_out, rgb_path=uw["rgb_path"].to_numpy(), window_idx=uw["window_idx"].to_numpy(), feat=feat,
                 logit=model_l)
        m = (uw["fold"] == k).to_numpy()
        c = compare(uw.loc[m, "rgb_path"].to_numpy(), uw.loc[m, "window_idx"].to_numpy(), head_l[m], model_l[m],
                    hook_l[m], saved_scores(k))
        c.update({"gpu_s": ex.gpu_s, "wall_s": wall, "n_windows": len(uw), "n_fold_k_windows": int(m.sum()),
                  "file": str(f_out.relative_to(REPO))})
        res["folds"][str(k)] = c
        print(f"fold model {k}: {json.dumps(c)}")
        del ex
        torch.cuda.empty_cache()
    on, ptxt = vz.power()
    res["power_at_end"] = ptxt
    EXTRACT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- probe (CPU, float64)

def fit_lr(X, y, K, C):
    """Multinomial LR: min C * sum_i w_i CE_i + 0.5 ||W||^2 (intercepts unpenalised), balanced w, L-BFGS float64."""
    import torch
    torch.set_num_threads(8)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y).long()
    cnt = np.bincount(y, minlength=K)
    wc = len(y) / (K * cnt)
    wt = torch.from_numpy(wc[y])
    W = torch.zeros(X.shape[1], K, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(K, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=2000, tolerance_grad=1e-10, tolerance_change=1e-14,
                            history_size=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = C * (wt * torch.nn.functional.cross_entropy(Xt @ W + b, yt, reduction="none")).sum() + 0.5 * (W * W).sum()
        loss.backward()
        return loss

    scale = max(1.0, C * len(y))  # gradient relative to the size of the summed loss
    for _ in range(10):
        opt.step(closure)
        closure()
        rel = max(float(W.grad.abs().max()), float(b.grad.abs().max())) / scale
        if rel <= 1e-8:
            break
    return W.detach().numpy(), b.detach().numpy(), rel


def standardise(Xtr, *Xs):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    return [(Xtr - mu) / sd] + [(x - mu) / sd for x in Xs]


def train_predict(Xtr, ytr, Xte, C):
    present = sorted(set(ytr.tolist()))
    remap = {c: i for i, c in enumerate(present)}
    Xtr_s, Xte_s = standardise(Xtr, Xte)
    W, b, g = fit_lr(Xtr_s, np.array([remap[v] for v in ytr]), len(present), C)
    return np.array(present)[np.argmax(Xte_s @ W + b, 1)], g


def metrics(y, p):
    present = sorted(set(y.tolist()))
    rec = {TYPES[t]: float((p[y == t] == t).mean()) for t in present}
    cm = np.array([[int(((y == t) & (p == q)).sum()) for q in range(5)] for t in range(5)])
    return {"n": int(len(y)), "accuracy": float((y == p).mean()), "macro_accuracy": float(np.mean(list(rec.values()))),
            "n_types_in_macro": len(present), "recall": rec, "confusion": cm.tolist()}


def probe():
    check_plan()
    refuse(INNER_CSV, PRED_CSV, METRICS_JSON, COMP_CSV)
    w = pd.read_csv(WINDOWS_CSV)
    tix = {t: i for i, t in enumerate(TYPES)}
    w["y"] = w["type"].map(tix)
    inner_rows, preds, fold_info = [], [], {}
    for k in FOLDS:
        z = np.load(CACHE / f"features_fold{k}.npz", allow_pickle=True)
        key = {(r, int(i)): j for j, (r, i) in enumerate(zip(z["rgb_path"], z["window_idx"]))}
        F = z["feat"].astype(np.float64).mean(1)  # mean of original and flipped features
        fs = F[[key[(r, i)] for r, i in zip(w["rgb_path"], w["w_tstar"])]]
        fh = F[[key[(r, i)] for r, i in zip(w["rgb_path"], w["w_that"])]]
        tr, te = (w["fold"] != k).to_numpy(), (w["fold"] == k).to_numpy()
        # inner leave-one-scenario-out CV on the training folds
        scen = w.loc[tr, "scenario_id"].to_numpy()
        Xs_tr, Xh_tr, y_tr = fs[tr], fh[tr], w.loc[tr, "y"].to_numpy()
        res_c = {}
        for C in CS:
            p_in = np.empty(tr.sum(), int)
            gmax = 0.0
            for s in sorted(set(scen)):
                a, h = scen != s, scen == s
                p_in[h], g = train_predict(np.vstack([Xs_tr[a], Xh_tr[a]]), np.concatenate([y_tr[a], y_tr[a]]), Xh_tr[h], C)
                gmax = max(gmax, g)
            m = metrics(y_tr, p_in)
            res_c[C] = m["macro_accuracy"]
            inner_rows.append({"fold": k, "C": C, "inner_macro_accuracy": m["macro_accuracy"],
                               "inner_accuracy": m["accuracy"], "n_scenarios": len(set(scen)),
                               "max_rel_grad": gmax})
            print(f"fold {k} C {C:g}: inner macro {m['macro_accuracy']:.4f} acc {m['accuracy']:.4f} (rel grad {gmax:.1e})")
        best = max(res_c.values())
        C_best = min(C for C in CS if res_c[C] == best)  # ties -> smaller C (stronger shrinkage)
        p_te, g = train_predict(np.vstack([fs[tr], fh[tr]]), np.concatenate([y_tr, y_tr]), fh[te], C_best)
        y_te = w.loc[te, "y"].to_numpy()
        fold_info[str(k)] = {"C": C_best, "outer_rel_grad": g, **metrics(y_te, p_te)}
        print(f"fold {k}: chosen C {C_best:g}; test macro {fold_info[str(k)]['macro_accuracy']:.4f} "
              f"acc {fold_info[str(k)]['accuracy']:.4f}")
        preds.append(w.loc[te, ["rgb_path", "fold", "scenario_id", "type"]].assign(
            pred_type=[TYPES[i] for i in p_te], C=C_best))
    P = pd.concat(preds).sort_values("rgb_path")
    if len(P) != 2211:
        fail("pooled predictions do not cover the 2211 clips")
    pooled = metrics(P["type"].map(tix).to_numpy(), P["pred_type"].map(tix).to_numpy())
    # fold-1 comparison: majority baseline and QLoRA epoch 1, recomputed from the saved files
    q = pd.read_csv(QLORA_E1)
    qm = json.loads(QLORA_E1_MET.read_text())
    f1 = P[P["fold"] == 1]
    if set(q["rgb_path"]) != set(f1["rgb_path"]):
        fail("QLoRA epoch-1 clips differ from the fold-1 clips")
    mq = metrics(q["type"].map(tix).to_numpy(), q["pred_type"].map(tix).to_numpy())
    if abs(mq["accuracy"] - qm["type_accuracy"]) > 1e-9:
        fail("recomputed QLoRA epoch-1 accuracy differs from its metrics file")
    maj = qm["majority_type"]
    mm = metrics(f1["type"].map(tix).to_numpy(), np.full(len(f1), tix[maj]))
    if abs(mm["accuracy"] - qm["majority_accuracy"]) > 1e-9:
        fail("recomputed majority accuracy differs from the QLoRA metrics file")
    mp = fold_info["1"]
    comp = pd.DataFrame([{"model": f"majority ({maj})", "n": mm["n"], "accuracy": mm["accuracy"], "macro_accuracy": mm["macro_accuracy"]},
                         {"model": "QLoRA epoch 1 (window A)", "n": mq["n"], "accuracy": mq["accuracy"], "macro_accuracy": mq["macro_accuracy"]},
                         {"model": "probe A (t_hat window)", "n": mp["n"], "accuracy": mp["accuracy"], "macro_accuracy": mp["macro_accuracy"]}])
    success = bool(mp["accuracy"] > max(mm["accuracy"], mq["accuracy"]) and
                   mp["macro_accuracy"] > max(mm["macro_accuracy"], mq["macro_accuracy"]))
    pd.DataFrame(inner_rows).to_csv(INNER_CSV, index=False)
    P.to_csv(PRED_CSV, index=False)
    comp.to_csv(COMP_CSV, index=False)
    for name, cm in [("pooled", pooled["confusion"])] + [(f"fold{k}", fold_info[str(k)]["confusion"]) for k in FOLDS]:
        c = np.array(cm, float)
        pd.DataFrame(c.astype(int), index=[f"true {t}" for t in TYPES], columns=[f"pred {t}" for t in TYPES]).to_csv(OUT / f"confusion_{name}_counts.csv")
        rn = np.divide(c, c.sum(1, keepdims=True), out=np.full_like(c, np.nan), where=c.sum(1, keepdims=True) > 0)
        pd.DataFrame(rn, index=[f"true {t}" for t in TYPES], columns=[f"pred {t}" for t in TYPES]).to_csv(OUT / f"confusion_{name}_rownorm.csv")
    res = {"folds": fold_info, "pooled": pooled, "fold1_majority": mm, "fold1_qlora_epoch1": mq,
           "success": success}
    METRICS_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(comp.to_string(index=False))
    print(f"pooled: macro {pooled['macro_accuracy']:.4f} acc {pooled['accuracy']:.4f}; SUCCESS RULE: {success}")


# ---------------------------------------------------------------- report

def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    check_plan()
    refuse(REPORT, OUT / "confusion.png")
    S = json.loads(SAFETY_JSON.read_text())
    E = json.loads(EXTRACT_JSON.read_text())
    M = json.loads(METRICS_JSON.read_text())
    inner = pd.read_csv(INNER_CSV)
    comp = pd.read_csv(COMP_CSV)
    w = pd.read_csv(WINDOWS_CSV)
    # figure: pooled counts, pooled row-normalised, fold-1 row-normalised
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    for ax, (title, cm, norm) in zip(axes, [("pooled out-of-fold, counts", M["pooled"]["confusion"], False),
                                            ("pooled out-of-fold, row-normalised", M["pooled"]["confusion"], True),
                                            ("fold 1, row-normalised", M["folds"]["1"]["confusion"], True)]):
        c = np.array(cm, float)
        v = np.divide(c, c.sum(1, keepdims=True), out=np.zeros_like(c), where=c.sum(1, keepdims=True) > 0) if norm else c
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
    fig.suptitle("Type probe A (C_cctv seed 0 frozen features + logistic regression), t_hat window")
    fig.savefig(OUT / "confusion.png", dpi=150)
    plt.close(fig)
    f = lambda x: f"{x:.4f}"  # noqa: E731
    L = ["TYPE PROBE, OPTION A: REPORT (synthetic only; no real clip, label or metadata file opened; no checkpoint, "
         "frozen file, split or manifest changed; nothing committed)",
         f"Pre-registration: PLAN.txt sha256 {PLAN_SHA} (checked by every mode).", "",
         "1. SAFETY CHECK (5 fold-k clips per fold model, every window, original + flipped; saved head on the extracted "
         f"feature vs the saved window score on this GPU; tolerance {S['tolerance']})"]
    for k, c in S["folds"].items():
        L.append(f"  fold model {k}: {c['n_windows']} windows ({c['n_window_passes']} passes); hooked head output == model "
                 f"logit: {c['hook_equals_model_logit']}; head on feature == model logit: {c['head_on_feature_equals_model_logit']}; "
                 f"max |score diff| vs saved {c['max_abs_score_diff_head_on_feature']:.2e} (mean {c['mean_abs_score_diff']:.2e}); "
                 f"PASS {c['pass']}")
    L.append(f"  verdict: {'PASS' if S['pass'] else 'FAIL'} (GPU {S['gpu']})")
    L += ["", "   Same comparison on every extracted window of fold k (fold model k, saved scores exist only for its own "
          "validation windows):"]
    for k, c in E["folds"].items():
        L.append(f"  fold model {k}: {c['n_fold_k_windows']} fold-{k} windows; max |score diff| "
                 f"{c['max_abs_score_diff_head_on_feature']:.2e}; hook == model {c['hook_equals_model_logit']}")
    L += ["", "2. WINDOWS AND FEATURES",
          f"  clips 2211; t* window == t_hat window in {int((w['w_tstar'] == w['w_that']).sum())} clips; unique windows "
          f"{E['n_unique_windows']}, each scored by all 3 fold models, original + flipped",
          "  GPU time (forward passes, synchronised): " + ", ".join(
              f"fold model {k} {c['gpu_s']:.1f} s ({c['gpu_s'] / (2 * c['n_windows']) * 1000:.1f} ms per pass)"
              for k, c in E["folds"].items())
          + f"; total {sum(c['gpu_s'] for c in E['folds'].values()):.1f} s; safety check "
          f"{sum(c['gpu_s'] for c in S['folds'].values()):.1f} s; power at end {E['power_at_end']}", "",
          "3. INNER CV (leave one scenario out on the training folds; pooled inner macro accuracy per C)"]
    for k in FOLDS:
        g = inner[inner["fold"] == k]
        L.append(f"  fold {k}: " + ", ".join(f"C {r.C:g}: {r.inner_macro_accuracy:.4f}" for r in g.itertuples())
                 + f"  -> chosen C {M['folds'][str(k)]['C']:g} ({int(g['n_scenarios'].iloc[0])} scenarios)")
    L += ["", "4. OUT-OF-FOLD RESULTS (t_hat window)",
          f"  {'set':8s} {'n':>5s} {'macro acc':>10s} {'accuracy':>9s}  per-type recall"]
    for name, m in [(f"fold {k}", M["folds"][str(k)]) for k in FOLDS] + [("pooled", M["pooled"])]:
        L.append(f"  {name:8s} {m['n']:5d} {f(m['macro_accuracy']):>10s} {f(m['accuracy']):>9s}  "
                 + ", ".join(f"{t} {v:.3f}" for t, v in m["recall"].items())
                 + (f"  (macro over {m['n_types_in_macro']} types)" if m["n_types_in_macro"] != 5 else ""))
    L += ["  confusion matrices: confusion_{pooled,fold0,fold1,fold2}_{counts,rownorm}.csv, confusion.png", "",
          "  pooled confusion (rows true, columns predicted; " + ", ".join(TYPES) + "):"]
    L += ["    " + f"{t:10s} " + " ".join(f"{v:5d}" for v in row) for t, row in zip(TYPES, M["pooled"]["confusion"])]
    # post-hoc, descriptive (not in the plan): do the predictions follow the map rather than the type?
    P = pd.read_csv(PRED_CSV).merge(w[["rgb_path"]], on="rgb_path")
    P["map"] = P["scenario_id"].str.split("_").str[0]
    w["map"] = w["scenario_id"].str.split("_").str[0]
    L += ["", "4b. POST-HOC, DESCRIPTIVE (added after seeing the results; not part of the plan): map vs type",
          "  Each scenario is one map + one type. For each fold-k test clip: is its map present in the training folds, "
          "and is the predicted type one of the types that map has in the training folds (a 'same-map training type')?"]
    for k in FOLDS:
        trm = w[w["fold"] != k].groupby("map")["type"].agg(lambda s: set(s)).to_dict()
        pk = P[P["fold"] == k]
        has = pk["map"].isin(trm)
        same = pk.apply(lambda r: r["pred_type"] in trm.get(r["map"], set()), axis=1)
        own = pk.apply(lambda r: r["type"] in trm.get(r["map"], set()), axis=1)
        L.append(f"  fold {k}: {len(pk)} test clips; map seen in training {int(has.sum())}; of those, prediction = a "
                 f"same-map training type {int(same[has].sum())} ({same[has].mean() if has.any() else float('nan'):.3f}); "
                 f"true type = a same-map training type {int(own[has].sum())}; accuracy where the map was seen "
                 f"{(pk.loc[has, 'type'] == pk.loc[has, 'pred_type']).mean() if has.any() else float('nan'):.4f}, where not "
                 f"{(pk.loc[~has, 'type'] == pk.loc[~has, 'pred_type']).mean() if (~has).any() else float('nan'):.4f} (n {int((~has).sum())})")
    L.append("  The inner leave-one-scenario-out macro accuracies (section 3) are far below chance (0.2) in every fold; "
             "this is consistent with the features carrying the map/scene more than the collision type (interpretation, "
             "not tested).")
    L += ["", "5. FOLD 1 COMPARISON (755 clips)", f"  {'model':28s} {'accuracy':>9s} {'macro acc':>10s}"]
    for r in comp.itertuples():
        L.append(f"  {r.model:28s} {r.accuracy:9.4f} {r.macro_accuracy:10.4f}")
    L += [f"  SUCCESS RULE (probe beats both, in accuracy AND macro accuracy, on fold 1): "
          f"{'MET' if M['success'] else 'NOT MET'}",
          "  Note: QLoRA saw a 4 s window A (16 frames, 448 x 256); the probe sees the Classifier's 2 s window nearest "
          "t_hat (16 frames, 224 x 224). Both are centred on the same frozen out-of-fold t_hat.", "",
          ("Next step in the plan (NOT done now): repeat with Classifier seeds 1 and 2, mean and SD." if M["success"]
           else "The plan's seed 1 / 2 repeat applies only on success; not run."),
          "Files: PLAN.txt, safety.json, windows.csv, extract_check.json, inner_cv.csv, predictions_oof.csv, metrics.json, "
          "comparison_fold1.csv, confusion_*.csv, confusion.png, run_log_*.txt, report.txt; features in "
          "cache/type_probe_A/ (gitignored)."]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for a in ("--safety", "--extract", "--probe", "--report"):
        g.add_argument(a, action="store_true")
    a = ap.parse_args()
    {"safety": safety, "extract": extract, "probe": probe, "report": report}[
        next(k for k in ("safety", "extract", "probe", "report") if getattr(a, k))]()


if __name__ == "__main__":
    main()
