"""Type model: DINOv2-base linear probe (benchmark paper arXiv 2604.09819 Sec. 6.3 style), every setting fixed in advance.

Pre-registration: results/type_dinov2/PLAN.txt (sha256 checked by every mode). SYNTHETIC ONLY: never opens, lists or
reads a real clip, real label or real metadata file. Changes no frozen file, checkpoint, split, manifest or earlier
result (vlm_zero_shot.py and type_probe_A.py are imported, not changed). Run with .venv-ml; mains power required.

  --frames    decode the t* frame and the t_hat frame of every synthetic clip at full resolution (vz.decode), resize
              to 224 x 224 (bicubic), keep them in cache/type_dinov2/frames224.npy; contact sheet of 6 clips
  --features  (GPU) safety check (5 frames encoded twice, bit-identical), then the CLS features of all frames
  --checks    synthetic checks (a) scenario folds, (b) leave one map out; information only
  --final     head on the t* frames of all 2211 clips -> models/type_dinov2/
  --report    report.txt, confusion matrices (CSV, PNG)
"""
import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
OUT = REPO / "results" / "type_dinov2"
CACHE = REPO / "cache" / "type_dinov2"
MODEL_DIR = REPO / "models" / "type_dinov2"
PLAN = OUT / "PLAN.txt"
PLAN_SHA = "e7a46dcc977a7d6e47ae52be2a4ed451ffa73ce679b1d5a8fd5cde70fb095a76"
MODEL_ID, MODEL_REV = "facebook/dinov2-base", "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
SIZE, BATCH, C_FIXED = 224, 32, 1.0
TYPES = ["head-on", "rear-end", "sideswipe", "single", "t-bone"]
MAPS = ["Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"]
FRAMES_NPY, INDEX_CSV, FEAT_NPZ = CACHE / "frames224.npy", CACHE / "frames_index.csv", CACHE / "features.npz"
SHEET, SAFETY_JSON, FEAT_JSON = OUT / "contact_sheet.png", OUT / "safety.json", OUT / "features.json"
CHECKS_JSON, PRED_CSV, FINAL_JSON, REPORT = OUT / "checks.json", OUT / "predictions_checks.csv", OUT / "final.json", OUT / "report.txt"
HEAD_NPZ, HEAD_JSON = MODEL_DIR / "type_head_dinov2.npz", MODEL_DIR / "type_head_dinov2.json"


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def check_plan():
    if hashlib.sha256(PLAN.read_bytes()).hexdigest() != PLAN_SHA:
        fail("PLAN.txt changed since pre-registration")


def refuse(*paths):
    for p in paths:
        if p.exists():
            fail(f"{p} exists; refusing to overwrite")


def mains():
    import vlm_zero_shot as vz
    on, txt = vz.power()
    print(f"power {txt}")
    if not on:
        fail("on battery; not starting")
    return txt


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- 1. frames

def decode_one(c):
    import vlm_zero_shot as vz
    from PIL import Image
    i_s, i_h = vz.to_indices(c, [c["t_star"], c["t_hat"]])
    got = vz.decode(c, sorted({i_s, i_h}), budget=10 ** 12)  # full frame, no resize (the frame audit's decoder)
    out = []
    for i in (i_s, i_h):
        img = got[i]
        if img.size != (c["width"], c["height"]):
            raise RuntimeError(f"{c['rgb_path']}: decoded {img.size}, metadata {c['width']}x{c['height']}")
        out.append(np.asarray(img.resize((SIZE, SIZE), Image.BICUBIC)))
    return c["rgb_path"], int(i_s), int(i_h), np.stack(out)


def frames():
    import vlm_zero_shot as vz
    check_plan()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    refuse(FRAMES_NPY, INDEX_CSV, SHEET)
    mains()
    clips = sorted(vz.load_clips(), key=lambda c: c["rgb_path"])
    if len(clips) != 2211:
        fail("expected 2211 synthetic clips")
    arr = np.lib.format.open_memmap(CACHE / "frames224.tmp.npy", mode="w+", dtype=np.uint8, shape=(len(clips), 2, SIZE, SIZE, 3))
    pos = {c["rgb_path"]: j for j, c in enumerate(clips)}
    rows = [None] * len(clips)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as ex:
        for n, (p, i_s, i_h, x) in enumerate(ex.map(decode_one, clips, chunksize=4)):
            j = pos[p]
            arr[j] = x
            c = clips[j]
            rows[j] = {"row": j, "rgb_path": p, "stem": Path(p).stem, "type": c["type"], "fold": c["val_fold"],
                       "map": Path(p).stem.split("_")[0], "scenario_id": "_".join(Path(p).stem.split("_")[:2]),
                       "fps": c["fps"], "t_star": c["t_star"], "t_hat": c["t_hat"], "frame_tstar": i_s, "frame_that": i_h}
            if (n + 1) % 250 == 0:
                print(f"  {n + 1}/{len(clips)} clips decoded ({time.time() - t0:.0f} s)")
    arr.flush()
    del arr
    os.replace(CACHE / "frames224.tmp.npy", FRAMES_NPY)
    idx = pd.DataFrame(rows)
    idx.to_csv(INDEX_CSV, index=False)
    print(f"frames: {len(idx)} clips x 2 frames at {SIZE}x{SIZE}; t* frame == t_hat frame in "
          f"{int((idx['frame_tstar'] == idx['frame_that']).sum())} clips; {time.time() - t0:.0f} s")
    # contact sheet
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    F = np.load(FRAMES_NPY, mmap_mode="r")
    pick = sorted(np.random.default_rng(0).choice(len(idx), size=6, replace=False).tolist())
    fig, axes = plt.subplots(3, 4, figsize=(12, 9.6), constrained_layout=True)
    for n, j in enumerate(pick):
        r = idx.iloc[j]
        for m, (lab, fr, t) in enumerate((("t*", r["frame_tstar"], r["t_star"]), ("t_hat", r["frame_that"], r["t_hat"]))):
            ax = axes[n // 2, (n % 2) * 2 + m]
            ax.imshow(F[j, m], interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{r['stem']} ({r['type']})\n{lab} frame {fr}, t = {fr / r['fps']:.2f} s ({lab} = {t:.2f} s)" if m == 0
                         else f"\n{lab} frame {fr}, t = {fr / r['fps']:.2f} s ({lab} = {t:.2f} s)", fontsize=8)
    fig.suptitle("DINOv2 input check: whole 1920x1080 frame resized to 224x224 (bicubic), before normalisation; "
                 "6 clips, numpy default_rng(0)", fontsize=10)
    fig.savefig(SHEET, dpi=110)
    plt.close(fig)
    print(f"contact sheet: {SHEET.name} ({', '.join(idx.iloc[j]['stem'] for j in pick)})")


# ---------------------------------------------------------------- 2. features

def load_encoder():
    import torch
    from transformers import AutoImageProcessor, Dinov2Model
    proc = AutoImageProcessor.from_pretrained(MODEL_ID, revision=MODEL_REV)
    model = Dinov2Model.from_pretrained(MODEL_ID, revision=MODEL_REV, dtype=torch.float32).to("cuda").eval()
    if proc.resample != 3:
        fail("processor resample is not bicubic")
    mean = torch.tensor(proc.image_mean, dtype=torch.float32, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(proc.image_std, dtype=torch.float32, device="cuda").view(1, 3, 1, 1)
    return model, mean, std, proc


def encode(model, mean, std, x_uint8):
    """x (n, 224, 224, 3) uint8 -> CLS after the final layer norm (n, 768) float32; returns (features, gpu seconds)."""
    import torch
    out, gpu = [], 0.0
    with torch.no_grad():
        for s in range(0, len(x_uint8), BATCH):
            x = torch.from_numpy(np.ascontiguousarray(x_uint8[s:s + BATCH])).to("cuda").permute(0, 3, 1, 2).float().div_(255.0)
            x = (x - mean) / std
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            o = model(pixel_values=x)
            f = o.pooler_output
            torch.cuda.synchronize()
            gpu += time.perf_counter() - t0
            if not torch.equal(f, o.last_hidden_state[:, 0]):
                fail("pooler_output is not the CLS token of the (layer-normed) last hidden state")
            out.append(f.float().cpu().numpy())
    return np.concatenate(out), gpu


def features():
    import torch
    check_plan()
    refuse(SAFETY_JSON, FEAT_NPZ, FEAT_JSON)
    ptxt = mains()
    if not torch.cuda.is_available():
        fail("CUDA not available")
    F = np.load(FRAMES_NPY, mmap_mode="r")
    idx = pd.read_csv(INDEX_CSV)
    model, mean, std, proc = load_encoder()
    flat = F.reshape(-1, SIZE, SIZE, 3)
    pick = sorted(np.random.default_rng(0).choice(len(flat), size=5, replace=False).tolist())
    x5 = np.stack([flat[i] for i in pick])
    a, g1 = encode(model, mean, std, x5)
    b, g2 = encode(model, mean, std, x5)
    same = bool(np.array_equal(a, b))
    saf = {"frames": [{"flat_index": int(i), "clip": idx.iloc[i // 2]["stem"], "which": ("t*", "t_hat")[i % 2]} for i in pick],
           "identical": same, "max_abs_diff": float(np.abs(a - b).max()), "gpu_s": g1 + g2,
           "gpu": torch.cuda.get_device_name(0), "dtype": "float32", "power": ptxt}
    SAFETY_JSON.write_text(json.dumps(saf, indent=1), encoding="utf-8")
    print(f"safety: 5 frames encoded twice; identical {same}; max |diff| {saf['max_abs_diff']:.3e}")
    if not same:
        fail("SAFETY CHECK FAILED: features differ between two passes; stopping")
    feats, gpu = np.empty((len(idx), 2, 768), np.float32), 0.0
    t0 = time.time()
    for m in (0, 1):
        f, g = encode(model, mean, std, F[:, m])
        feats[:, m] = f
        gpu += g
    wall = time.time() - t0
    np.savez(FEAT_NPZ, rgb_path=idx["rgb_path"].to_numpy().astype(str), feat=feats)
    # information: the safety frames again, now inside batches of 32
    again = feats.reshape(-1, 768)[pick]
    info = {"n_clips": len(idx), "shape": list(feats.shape), "columns": "[:, 0] t* frame, [:, 1] t_hat frame",
            "gpu_s": gpu, "wall_s": wall, "ms_per_frame_gpu": 1000 * gpu / (2 * len(idx)), "batch": BATCH,
            "safety_frames_vs_full_run_max_abs_diff": float(np.abs(again - a).max()),
            "file": str(FEAT_NPZ.relative_to(REPO)), "sha256": sha256(FEAT_NPZ), "model": f"{MODEL_ID}@{MODEL_REV}",
            "power_at_end": mains()}
    FEAT_JSON.write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info, indent=1))


# ---------------------------------------------------------------- 3. synthetic checks

def load_feats():
    idx = pd.read_csv(INDEX_CSV)
    z = np.load(FEAT_NPZ)
    if list(z["rgb_path"]) != idx["rgb_path"].tolist():
        fail("features and frame index are not in the same order")
    idx["y"] = idx["type"].map({t: i for i, t in enumerate(TYPES)})
    return idx, z["feat"].astype(np.float64)


def majority(types):
    vc = pd.Series(types).value_counts()
    return sorted(vc[vc == vc.max()].index)[0]


def checks():
    import type_probe_A as A
    check_plan()
    refuse(CHECKS_JSON, PRED_CSV)
    idx, X = load_feats()
    y = idx["y"].to_numpy()
    rows, res = [], {}
    for split, groups in (("scenario_folds", [("fold", k) for k in (0, 1, 2)]), ("leave_one_map_out", [("map", m) for m in MAPS])):
        parts = {}
        for col, g in groups:
            te = (idx[col] == g).to_numpy()
            tr = ~te
            p, rel = A.train_predict(X[tr, 0], y[tr], X[te, 1], C_FIXED)  # train t* frames, test t_hat frames
            maj = majority(idx.loc[tr, "type"])
            pm = np.full(te.sum(), TYPES.index(maj))
            parts[str(g)] = {"probe": A.metrics(y[te], p), "majority": A.metrics(y[te], pm), "majority_type": maj,
                             "n_train": int(tr.sum()), "train_counts": idx.loc[tr, "type"].value_counts().to_dict(),
                             "rel_grad": rel}
            rows.append(idx.loc[te, ["rgb_path", "fold", "map", "scenario_id", "type"]].assign(
                split=split, test_group=str(g), pred_type=[TYPES[i] for i in p], majority_type=maj))
            print(f"{split} {g}: macro {parts[str(g)]['probe']['macro_accuracy']:.4f} acc {parts[str(g)]['probe']['accuracy']:.4f} "
                  f"(majority {maj}: {parts[str(g)]['majority']['macro_accuracy']:.4f} / {parts[str(g)]['majority']['accuracy']:.4f})")
        P = pd.concat(rows[-len(groups):])
        tix = {t: i for i, t in enumerate(TYPES)}
        res[split] = {"parts": parts, "pooled": A.metrics(P["type"].map(tix).to_numpy(), P["pred_type"].map(tix).to_numpy()),
                      "pooled_majority": A.metrics(P["type"].map(tix).to_numpy(), P["majority_type"].map(tix).to_numpy())}
        print(f"{split} pooled: macro {res[split]['pooled']['macro_accuracy']:.4f} acc {res[split]['pooled']['accuracy']:.4f}")
    # fold-1 clips: QLoRA epoch 1
    q = pd.read_csv(A.QLORA_E1)
    qm = json.loads(A.QLORA_E1_MET.read_text())
    tix = {t: i for i, t in enumerate(TYPES)}
    mq = A.metrics(q["type"].map(tix).to_numpy(), q["pred_type"].map(tix).to_numpy())
    if abs(mq["accuracy"] - qm["type_accuracy"]) > 1e-9:
        fail("recomputed QLoRA epoch-1 accuracy differs from its metrics file")
    if set(q["rgb_path"]) != set(idx.loc[idx["fold"] == 1, "rgb_path"]):
        fail("QLoRA epoch-1 clips differ from the fold-1 clips")
    res["qlora_epoch1_fold1"] = mq
    pd.concat(rows).to_csv(PRED_CSV, index=False)
    CHECKS_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- 4. final model

def final():
    import type_probe_A as A
    check_plan()
    refuse(FINAL_JSON, HEAD_NPZ, HEAD_JSON)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    idx, X = load_feats()
    Xtr, ytr = X[:, 0], idx["y"].to_numpy()  # t* frames of all 2211 clips
    present = sorted(set(ytr.tolist()))
    if present != list(range(5)):
        fail("not all 5 types present")
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    W, b, rel = A.fit_lr((Xtr - mu) / sd, ytr, 5, C_FIXED)
    train_acc = float((np.argmax(((Xtr - mu) / sd) @ W + b, 1) == ytr).mean())
    counts = idx["type"].value_counts().reindex(TYPES).astype(int).to_dict()
    np.savez(HEAD_NPZ, W=W, b=b, scaler_mean=mu, scaler_sd=sd, classes=np.array(TYPES))
    meta = {"model": "multinomial logistic regression on standardised DINOv2 CLS features",
            "encoder": f"{MODEL_ID}@{MODEL_REV}", "feature": "pooler_output = CLS after the final layer norm (768)",
            "image": "whole frame resized to 224x224, bicubic, /255, ImageNet mean/std", "C": C_FIXED,
            "objective": "C * sum_i w_i CE_i + 0.5 ||W||^2, intercepts unpenalised, balanced class weights",
            "classes": TYPES, "train_rows": "t* frame of each of the 2211 synthetic clips", "train_counts": counts,
            "class_weights": {t: 2211 / (5 * counts[t]) for t in TYPES}, "train_accuracy_in_sample": train_acc,
            "rel_grad_at_end": rel, "plan_sha256": PLAN_SHA, "features_sha256": sha256(FEAT_NPZ),
            "use": "x = CLS feature (768); z = (x - scaler_mean) / scaler_sd; type = classes[argmax(z @ W + b)]"}
    HEAD_JSON.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    info = {"head_npz": str(HEAD_NPZ.relative_to(REPO)), "head_npz_bytes": HEAD_NPZ.stat().st_size,
            "head_npz_sha256": sha256(HEAD_NPZ), "head_json": str(HEAD_JSON.relative_to(REPO)),
            "head_json_bytes": HEAD_JSON.stat().st_size, "train_counts": counts,
            "train_accuracy_in_sample": train_acc, "rel_grad_at_end": rel}
    FINAL_JSON.write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info, indent=1))


# ---------------------------------------------------------------- 5. report

def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    check_plan()
    refuse(REPORT, OUT / "confusion.png")
    S, Fi, R, Z = (json.loads(p.read_text()) for p in (SAFETY_JSON, FEAT_JSON, CHECKS_JSON, FINAL_JSON))
    PA = json.loads((REPO / "results" / "type_probe_A" / "metrics.json").read_text())
    PM = json.loads((REPO / "results" / "type_probe_A_map" / "metrics.json").read_text())
    idx = pd.read_csv(INDEX_CSV)
    lab = {"scenario_folds": "scenario folds", "leave_one_map_out": "leave one map out"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 10.5), constrained_layout=True)
    for r_i, split in enumerate(("scenario_folds", "leave_one_map_out")):
        c = np.array(R[split]["pooled"]["confusion"], float)
        cols, rws = [f"pred {t}" for t in TYPES], [f"true {t}" for t in TYPES]
        pd.DataFrame(c.astype(int), index=rws, columns=cols).to_csv(OUT / f"confusion_{split}_pooled_counts.csv")
        pd.DataFrame(c / c.sum(1, keepdims=True), index=rws, columns=cols).to_csv(OUT / f"confusion_{split}_pooled_rownorm.csv")
        for g, part in R[split]["parts"].items():
            cg = np.array(part["probe"]["confusion"], float)
            pd.DataFrame(cg.astype(int), index=rws, columns=cols).to_csv(OUT / f"confusion_{split}_{g}_counts.csv")
            rn = np.divide(cg, cg.sum(1, keepdims=True), out=np.full_like(cg, np.nan), where=cg.sum(1, keepdims=True) > 0)
            pd.DataFrame(rn, index=rws, columns=cols).to_csv(OUT / f"confusion_{split}_{g}_rownorm.csv")
        for c_i, norm in enumerate((False, True)):
            ax = axes[r_i, c_i]
            v = c / c.sum(1, keepdims=True) if norm else c
            ax.imshow(v, cmap="Blues", vmin=0, vmax=1 if norm else v.max())
            for i in range(5):
                for j in range(5):
                    ax.text(j, i, f"{v[i, j]:.2f}" if norm else f"{int(v[i, j])}", ha="center", va="center", fontsize=9,
                            color="white" if v[i, j] > (0.6 if norm else 0.6 * v.max()) else "black")
            ax.set_xticks(range(5), TYPES, rotation=30)
            ax.set_yticks(range(5), TYPES)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title(f"{lab[split]}, pooled, {'row-normalised' if norm else 'counts'}")
    fig.suptitle("DINOv2-base linear probe (C = 1, train t* frames, test t_hat frames), synthetic checks")
    fig.savefig(OUT / "confusion.png", dpi=140)
    plt.close(fig)
    f4 = lambda x: f"{x:.4f}"  # noqa: E731
    L = ["TYPE MODEL: DINOv2 LINEAR PROBE: REPORT (synthetic only; no real clip, label or metadata file opened; no frozen "
         "file, checkpoint, split, manifest or earlier result changed; nothing committed)",
         f"PLAN.txt sha256 {PLAN_SHA} (checked by every mode). Every setting fixed in advance; no search.", "",
         "1. INPUT",
         f"  {len(idx)} clips; frames decoded with vlm_zero_shot.decode at 1920x1080, resized to 224x224 (bicubic); t* frame "
         f"== t_hat frame in {int((idx['frame_tstar'] == idx['frame_that']).sum())} clips. Contact sheet: contact_sheet.png.",
         "", "2. SAFETY CHECK",
         f"  5 frames ({', '.join(f['clip'] + ' ' + f['which'] for f in S['frames'])}) encoded twice in separate passes "
         f"(float32): identical {S['identical']}, max |diff| {S['max_abs_diff']:.1e} -> {'PASS' if S['identical'] else 'FAIL'}",
         f"  (information) the same 5 frames inside the full run's batches of {Fi['batch']}: max |diff| "
         f"{Fi['safety_frames_vs_full_run_max_abs_diff']:.2e} vs the safety pass", "",
         "3. FEATURES AND GPU TIME",
         f"  {Fi['file']} shape {Fi['shape']} ({Fi['columns']}); sha256 {Fi['sha256'][:16]}...",
         f"  GPU (synchronised forward) {Fi['gpu_s']:.1f} s for {2 * Fi['n_clips']} frames ({Fi['ms_per_frame_gpu']:.2f} ms per "
         f"frame), wall {Fi['wall_s']:.1f} s; safety check {S['gpu_s']:.2f} s; GPU {S['gpu']}; power {S['power']} / at end "
         f"{Fi['power_at_end']}", "",
         "4. SYNTHETIC CHECKS (information only; train on t* frames, test on t_hat frames; C = 1 fixed)",
         "  Both splits are biased for collision type, as found earlier: each scenario is one map + one type, the scenario "
         "folds put the same map in training and test with a different type (probe A: accuracy 0.005 on fold-0 clips "
         "whose map was in training), and leaving one map out still leaves each test map with only 2-3 types and the "
         "training maps' own map-type pairing (probe A map: pooled macro accuracy below chance). Neither split is a clean "
         "estimate of type recognition on unseen scenes.", ""]
    for split, earlier_key in (("scenario_folds", "folds"), ("leave_one_map_out", "rounds")):
        L.append(f"  ({'a' if split == 'scenario_folds' else 'b'}) {lab[split]}")
        L.append(f"    {'test set':10s} {'n':>5s} {'DINOv2 macro':>12s} {'DINOv2 acc':>10s} {'majority (type)':>24s} "
                 f"{'maj macro':>9s} {'maj acc':>8s} {'probe A macro':>13s} {'probe A acc':>11s}")
        for g, part in R[split]["parts"].items():
            e = PA["folds"][g] if split == "scenario_folds" else PM["rounds"][g]["probe"]
            name = f"fold {g}" if split == "scenario_folds" else g
            L.append(f"    {name:10s} {part['probe']['n']:5d} {f4(part['probe']['macro_accuracy']):>12s} "
                     f"{f4(part['probe']['accuracy']):>10s} {part['majority_type']:>24s} {f4(part['majority']['macro_accuracy']):>9s} "
                     f"{f4(part['majority']['accuracy']):>8s} {f4(e['macro_accuracy']):>13s} {f4(e['accuracy']):>11s}")
        e = PA["pooled"] if split == "scenario_folds" else PM["pooled"]
        p, pm = R[split]["pooled"], R[split]["pooled_majority"]
        L.append(f"    {'pooled':10s} {p['n']:5d} {f4(p['macro_accuracy']):>12s} {f4(p['accuracy']):>10s} {'(per part)':>24s} "
                 f"{f4(pm['macro_accuracy']):>9s} {f4(pm['accuracy']):>8s} {f4(e['macro_accuracy']):>13s} {f4(e['accuracy']):>11s}")
        L.append("    DINOv2 per-type recall, pooled: " + ", ".join(f"{t} {v:.3f}" for t, v in p["recall"].items()))
        for g, part in R[split]["parts"].items():
            L.append(f"      {('fold ' + g) if split == 'scenario_folds' else g:9s} recall: "
                     + ", ".join(f"{t} {v:.3f}" for t, v in part["probe"]["recall"].items()))
        L.append("    pooled confusion (rows true, columns predicted; " + ", ".join(TYPES) + "):")
        L += ["      " + f"{t:10s} " + " ".join(f"{v:5d}" for v in row) for t, row in zip(TYPES, p["confusion"])]
        L.append("")
    q, d1 = R["qlora_epoch1_fold1"], R["scenario_folds"]["parts"]["1"]
    L += ["  Fold-1 clips (755), same clips:", f"    {'model':32s} {'accuracy':>9s} {'macro acc':>10s}",
          f"    {'majority (' + d1['majority_type'] + ')':32s} {d1['majority']['accuracy']:9.4f} {d1['majority']['macro_accuracy']:10.4f}",
          f"    {'QLoRA epoch 1 (window A)':32s} {q['accuracy']:9.4f} {q['macro_accuracy']:10.4f}",
          f"    {'probe A (scenario folds)':32s} {PA['folds']['1']['accuracy']:9.4f} {PA['folds']['1']['macro_accuracy']:10.4f}",
          f"    {'probe A map (leave one map out)':32s} {PM['fold1']['probe']['accuracy']:9.4f} {PM['fold1']['probe']['macro_accuracy']:10.4f}",
          f"    {'DINOv2 (scenario folds)':32s} {d1['probe']['accuracy']:9.4f} {d1['probe']['macro_accuracy']:10.4f}",
          "  Confusion matrices: confusion_<split>_pooled_{counts,rownorm}.csv, confusion_<split>_<fold or map>_{counts,"
          "rownorm}.csv, confusion.png", "",
          "5. FINAL MODEL (t* frames of all 2211 clips, C = 1, balanced weights)",
          f"  {Z['head_npz']} ({Z['head_npz_bytes']} bytes, sha256 {Z['head_npz_sha256'][:16]}...): W (768 x 5), b, scaler mean "
          f"and SD, class order; settings in {Z['head_json']} ({Z['head_json_bytes']} bytes)",
          "  trained on: " + ", ".join(f"{t} {n}" for t, n in Z["train_counts"].items())
          + f"; in-sample accuracy {Z['train_accuracy_in_sample']:.4f} (training data, not an estimate)",
          "  Note: models/ is not in .gitignore, so the folder shows as untracked; nothing was committed.", "",
          "Files: PLAN.txt, contact_sheet.png, safety.json, features.json, checks.json, predictions_checks.csv, final.json, "
          "confusion_*.csv, confusion.png, run_log_*.txt, report.txt; cache/type_dinov2/ (frames224.npy, frames_index.csv, "
          "features.npz; gitignored); models/type_dinov2/."]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    modes = ("frames", "features", "checks", "final", "report")
    for m in modes:
        g.add_argument(f"--{m}", action="store_true")
    a = ap.parse_args()
    {"frames": frames, "features": features, "checks": checks, "final": final, "report": report}[
        next(m for m in modes if getattr(a, m))]()


if __name__ == "__main__":
    main()
