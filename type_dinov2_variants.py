"""DINOv2 probe, two variants (information only): V1 = the model's own processor (resize 256 + centre crop 224),
V2 = whole frame squashed to 518 x 518. Everything else as results/type_dinov2/PLAN.txt.

Pre-registration: results/type_dinov2_variants/PLAN.txt (sha256 checked by every mode). Synthetic only; no real file.
Imports type_dinov2 (load_encoder, encode, majority, mains, sha256), type_probe_A (train_predict, fit_lr, metrics) and
feature_view (silhouette); changes none of them and nothing in results/type_dinov2/. Run with .venv-ml.

  --frames    decode the t* / t_hat frames again (same indices as cache/type_dinov2/frames_index.csv); V1 processor
              pixel values and V2 518 x 518 frames to cache/type_dinov2_variants/; contact sheets of the same 6 clips
  --features  (GPU) safety check per variant, features, GPU time, peak GPU memory of V2
  --checks    scenario folds, leave one map out, fold-1 table, in-sample accuracy, silhouette (incl. the 224 squash)
  --report    report.txt
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
import type_dinov2 as td  # noqa: E402

OUT = REPO / "results" / "type_dinov2_variants"
CACHE = REPO / "cache" / "type_dinov2_variants"
PLAN = OUT / "PLAN.txt"
PLAN_SHA = "1bec873718bee9ef5f491676116e12d987befc40509958b47e37b73acbf291a9"
DATASET = REPO / "ACCIDENT" / "dataset"
V2_SIZE = 518
V1_NPY, V2_NPY, FEAT_NPZ = CACHE / "v1_pixel_values.npy", CACHE / "v2_frames518.npy", CACHE / "features.npz"
FEAT_JSON, CHECKS_JSON, PRED_CSV, REPORT = OUT / "features.json", OUT / "checks.json", OUT / "predictions_checks.csv", OUT / "report.txt"
TYPES, MAPS = td.TYPES, td.MAPS
_PROC = None


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def check_plan():
    if hashlib.sha256(PLAN.read_bytes()).hexdigest() != PLAN_SHA:
        fail("PLAN.txt changed since pre-registration")
    if hashlib.sha256(td.PLAN.read_bytes()).hexdigest() != td.PLAN_SHA:
        fail("results/type_dinov2/PLAN.txt changed")


def refuse(*paths):
    for p in paths:
        if p.exists():
            fail(f"{p} exists; refusing to overwrite")


def index():
    idx = pd.read_csv(td.INDEX_CSV)
    if len(idx) != 2211:
        fail("expected 2211 clips in cache/type_dinov2/frames_index.csv")
    return idx


def six(idx):
    """The same 6 clips as results/type_dinov2/contact_sheet.png (same rule, same row order)."""
    return sorted(np.random.default_rng(0).choice(len(idx), size=6, replace=False).tolist())


# ---------------------------------------------------------------- frames

def decode_one(args):
    global _PROC
    import vlm_zero_shot as vz
    from PIL import Image
    from transformers import AutoImageProcessor
    j, p, i_s, i_h, W, H = args
    if _PROC is None:
        _PROC = AutoImageProcessor.from_pretrained(td.MODEL_ID, revision=td.MODEL_REV)
    got = vz.decode({"video": str(DATASET / p), "rgb_path": p}, sorted({i_s, i_h}), budget=10 ** 12)
    v1, v2 = [], []
    for i in (i_s, i_h):
        img = got[i]
        if img.size != (W, H):
            raise RuntimeError(f"{p}: decoded {img.size}, expected {W}x{H}")
        v1.append(_PROC(images=img, return_tensors="np")["pixel_values"][0])
        v2.append(np.asarray(img.resize((V2_SIZE, V2_SIZE), Image.BICUBIC)))
    return j, np.stack(v1).astype(np.float32), np.stack(v2)


def frames():
    import vlm_zero_shot as vz
    check_plan()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    refuse(V1_NPY, V2_NPY, OUT / "contact_sheet_v1.png", OUT / "contact_sheet_v2.png")
    td.mains()
    idx = index()
    meta = {c["rgb_path"]: c for c in vz.load_clips()}
    jobs = [(j, r.rgb_path, int(r.frame_tstar), int(r.frame_that), meta[r.rgb_path]["width"], meta[r.rgb_path]["height"])
            for j, r in enumerate(idx.itertuples())]
    a1 = np.lib.format.open_memmap(CACHE / "v1.tmp.npy", mode="w+", dtype=np.float32, shape=(len(idx), 2, 3, 224, 224))
    a2 = np.lib.format.open_memmap(CACHE / "v2.tmp.npy", mode="w+", dtype=np.uint8, shape=(len(idx), 2, V2_SIZE, V2_SIZE, 3))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as ex:
        for n, (j, x1, x2) in enumerate(ex.map(decode_one, jobs, chunksize=4)):
            a1[j], a2[j] = x1, x2
            if (n + 1) % 250 == 0:
                print(f"  {n + 1}/{len(jobs)} clips ({time.time() - t0:.0f} s)")
    a1.flush(); a2.flush()
    del a1, a2
    os.replace(CACHE / "v1.tmp.npy", V1_NPY)
    os.replace(CACHE / "v2.tmp.npy", V2_NPY)
    print(f"frames: {len(idx)} clips x 2; V1 {V1_NPY.name}, V2 {V2_NPY.name}; {time.time() - t0:.0f} s")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(td.MODEL_ID, revision=td.MODEL_REV)
    mean, std = np.array(proc.image_mean).reshape(3, 1, 1), np.array(proc.image_std).reshape(3, 1, 1)
    A1, A2 = np.load(V1_NPY, mmap_mode="r"), np.load(V2_NPY, mmap_mode="r")
    pick = six(idx)
    for name, show, desc in (("v1", lambda j, m: np.clip((A1[j, m] * std + mean) * 255, 0, 255).astype(np.uint8).transpose(1, 2, 0),
                              "V1: the model's processor (resize shortest edge 256, centre crop 224), de-normalised for display"),
                             ("v2", lambda j, m: A2[j, m], "V2: whole frame squashed to 518x518 (bicubic)")):
        fig, axes = plt.subplots(3, 4, figsize=(12, 9.6), constrained_layout=True)
        for n, j in enumerate(pick):
            r = idx.iloc[j]
            for m, (lab, fr, t) in enumerate((("t*", r["frame_tstar"], r["t_star"]), ("t_hat", r["frame_that"], r["t_hat"]))):
                ax = axes[n // 2, (n % 2) * 2 + m]
                ax.imshow(show(j, m), interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title((f"{r['stem']} ({r['type']})\n" if m == 0 else "\n")
                             + f"{lab} frame {fr}, t = {fr / r['fps']:.2f} s ({lab} = {t:.2f} s)", fontsize=8)
        fig.suptitle(f"{desc}; same 6 clips as results/type_dinov2/contact_sheet.png", fontsize=10)
        fig.savefig(OUT / f"contact_sheet_{name}.png", dpi=110)
        plt.close(fig)
    print(f"contact sheets: {', '.join(idx.iloc[j]['stem'] for j in pick)}")


# ---------------------------------------------------------------- features

def encode_float(model, x):
    """V1: processor pixel values (n, 3, 224, 224) float32, already normalised -> CLS (n, 768); (features, gpu s)."""
    import torch
    out, gpu = [], 0.0
    with torch.no_grad():
        for s in range(0, len(x), td.BATCH):
            xb = torch.from_numpy(np.ascontiguousarray(x[s:s + td.BATCH])).to("cuda")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            o = model(pixel_values=xb)
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
    refuse(FEAT_NPZ, FEAT_JSON)
    ptxt = td.mains()
    idx = index()
    A1, A2 = np.load(V1_NPY, mmap_mode="r"), np.load(V2_NPY, mmap_mode="r")
    model, mean, std, _ = td.load_encoder()
    run = {"v1": lambda x: encode_float(model, x), "v2": lambda x: td.encode(model, mean, std, x)}
    info = {"gpu": torch.cuda.get_device_name(0), "power_start": ptxt, "variants": {}}
    feats = {}
    for name, A in (("v1", A1), ("v2", A2)):
        flat = A.reshape(-1, *A.shape[2:])
        pick = sorted(np.random.default_rng(0).choice(len(flat), size=5, replace=False).tolist())
        x5 = np.stack([flat[i] for i in pick])
        a, g1 = run[name](x5)
        b, g2 = run[name](x5)
        same = bool(np.array_equal(a, b))
        print(f"{name} safety: 5 frames twice; identical {same}; max |diff| {np.abs(a - b).max():.3e}")
        if not same:
            info["variants"][name] = {"safety_identical": False}
            FEAT_JSON.write_text(json.dumps(info, indent=1), encoding="utf-8")
            fail(f"SAFETY CHECK FAILED for {name}; stopping")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        f = np.empty((len(idx), 2, 768), np.float32)
        gpu, batch, t0 = 0.0, td.BATCH, time.time()
        try:
            for m in (0, 1):
                f[:, m], g = run[name](A[:, m])
                gpu += g
        except torch.cuda.OutOfMemoryError:
            if name != "v2":
                raise
            torch.cuda.empty_cache()
            td.BATCH = batch = 8  # the plan's fallback for V2
            torch.cuda.reset_peak_memory_stats()
            gpu, t0 = 0.0, time.time()
            print("v2: out of memory at batch 32; batch 8")
            for m in (0, 1):
                f[:, m], g = run[name](A[:, m])
                gpu += g
        feats[name] = f
        again = f.reshape(-1, 768)[pick]
        info["variants"][name] = {
            "safety_frames": [{"clip": idx.iloc[i // 2]["stem"], "which": ("t*", "t_hat")[i % 2]} for i in pick],
            "safety_identical": same, "safety_max_abs_diff": float(np.abs(a - b).max()), "safety_gpu_s": g1 + g2,
            "safety_vs_full_run_max_abs_diff": float(np.abs(again - a).max()), "batch": batch, "gpu_s": gpu,
            "wall_s": time.time() - t0, "ms_per_frame_gpu": 1000 * gpu / (2 * len(idx)),
            "peak_gpu_mem_allocated_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
            "peak_gpu_mem_reserved_gb": torch.cuda.max_memory_reserved() / 2 ** 30}
        print(f"{name}: {json.dumps(info['variants'][name])}")
        td.BATCH = 32
    np.savez(FEAT_NPZ, rgb_path=idx["rgb_path"].to_numpy().astype(str), v1=feats["v1"], v2=feats["v2"])
    info["file"], info["sha256"], info["power_end"] = str(FEAT_NPZ.relative_to(REPO)), td.sha256(FEAT_NPZ), td.mains()
    FEAT_JSON.write_text(json.dumps(info, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- checks

def sil_scores(Xh, idx):
    from feature_view import silhouette
    sd = Xh.std(0)
    sd[sd == 0] = 1.0
    Xs = (Xh - Xh.mean(0)) / sd
    G = Xs @ Xs.T
    sq = np.diag(G)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * G, 0.0))
    np.fill_diagonal(D, 0.0)
    rng = np.random.default_rng(0)
    out = {}
    for by in ("map", "type"):
        lab = idx[by].to_numpy()
        out[by] = silhouette(D, lab)
        out[f"{by}_shuffled"] = silhouette(D, rng.permutation(lab))
    return out


def run_checks(X, idx, name):
    import type_probe_A as A
    y = idx["type"].map({t: i for i, t in enumerate(TYPES)}).to_numpy()
    tix = {t: i for i, t in enumerate(TYPES)}
    res, rows = {}, []
    for split, groups in (("scenario_folds", [("fold", k) for k in (0, 1, 2)]), ("leave_one_map_out", [("map", m) for m in MAPS])):
        parts, sp_rows = {}, []
        for col, g in groups:
            te = (idx[col] == g).to_numpy()
            tr = ~te
            p, _ = A.train_predict(X[tr, 0], y[tr], X[te, 1], td.C_FIXED)
            maj = td.majority(idx.loc[tr, "type"])
            parts[str(g)] = {"probe": A.metrics(y[te], p), "majority": A.metrics(y[te], np.full(te.sum(), tix[maj])),
                             "majority_type": maj}
            sp_rows.append(idx.loc[te, ["rgb_path", "fold", "map", "type"]].assign(
                variant=name, split=split, test_group=str(g), pred_type=[TYPES[i] for i in p], majority_type=maj))
        P = pd.concat(sp_rows)
        rows += sp_rows
        res[split] = {"parts": parts, "pooled": A.metrics(P["type"].map(tix).to_numpy(), P["pred_type"].map(tix).to_numpy()),
                      "pooled_majority": A.metrics(P["type"].map(tix).to_numpy(), P["majority_type"].map(tix).to_numpy())}
        print(f"{name} {split}: pooled macro {res[split]['pooled']['macro_accuracy']:.4f} acc {res[split]['pooled']['accuracy']:.4f}")
    Xtr = X[:, 0]
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    W, b, _ = A.fit_lr((Xtr - mu) / sd, y, 5, td.C_FIXED)
    res["in_sample_accuracy"] = float((np.argmax(((Xtr - mu) / sd) @ W + b, 1) == y).mean())
    res["silhouette_that"] = sil_scores(X[:, 1], idx)
    print(f"{name}: in-sample {res['in_sample_accuracy']:.4f}; silhouette {json.dumps(res['silhouette_that'])}")
    return res, rows


def checks():
    import type_probe_A as A
    check_plan()
    refuse(CHECKS_JSON, PRED_CSV)
    idx = index()
    z = np.load(FEAT_NPZ)
    if list(z["rgb_path"]) != idx["rgb_path"].tolist():
        fail("variant features not in the frame-index order")
    res, rows = {}, []
    for name in ("v1", "v2"):
        res[name], r = run_checks(z[name].astype(np.float64), idx, name)
        rows += r
    # the 224 squash: checks and in-sample accuracy as saved in results/type_dinov2 (read only); silhouette computed now
    z0 = np.load(td.FEAT_NPZ)
    if list(z0["rgb_path"]) != idx["rgb_path"].tolist():
        fail("224 squash features not in the frame-index order")
    res["squash224"] = {"saved_checks": json.loads(td.CHECKS_JSON.read_text()),
                        "in_sample_accuracy": json.loads(td.FINAL_JSON.read_text())["train_accuracy_in_sample"],
                        "silhouette_that": sil_scores(z0["feat"].astype(np.float64)[:, 1], idx)}
    print(f"squash224 silhouette {json.dumps(res['squash224']['silhouette_that'])}")
    q = pd.read_csv(A.QLORA_E1)
    tix = {t: i for i, t in enumerate(TYPES)}
    res["qlora_epoch1_fold1"] = A.metrics(q["type"].map(tix).to_numpy(), q["pred_type"].map(tix).to_numpy())
    pd.concat(rows).to_csv(PRED_CSV, index=False)
    CHECKS_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- report

def report():
    check_plan()
    refuse(REPORT)
    Fj, R = json.loads(FEAT_JSON.read_text()), json.loads(CHECKS_JSON.read_text())
    PA = json.loads((REPO / "results" / "type_probe_A" / "metrics.json").read_text())
    PM = json.loads((REPO / "results" / "type_probe_A_map" / "metrics.json").read_text())
    s224 = R["squash224"]["saved_checks"]
    rows = [("224 squash (results/type_dinov2)", s224, R["squash224"]), ("V1 paper default (256 + crop 224)", R["v1"], R["v1"]),
            ("V2 518 squash", R["v2"], R["v2"])]
    f4 = lambda x: f"{x:.4f}"  # noqa: E731
    L = ["DINOv2 PROBE, TWO VARIANTS (information only; synthetic only; no real file; type_dinov2.py and "
         "results/type_dinov2/ unchanged; nothing committed)",
         f"PLAN.txt sha256 {PLAN_SHA}. Decided in advance: if DINOv2 is later run on real clips as a reference, V1 is used.",
         "", "MAIN TABLE (train on t* frames, test on t_hat frames, C = 1; macro = mean recall over the types present; "
         "silhouette of the standardised t_hat features of all 2211 clips, shuffled-label reference in brackets)",
         f"  {'variant':36s} {'folds macro':>11s} {'folds acc':>9s} {'LOMO macro':>10s} {'LOMO acc':>8s} {'in-sample':>9s} "
         f"{'silh. map':>18s} {'silh. type':>18s}"]
    for name, c, s in rows:
        sl = s["silhouette_that"]
        L.append(f"  {name:36s} {f4(c['scenario_folds']['pooled']['macro_accuracy']):>11s} "
                 f"{f4(c['scenario_folds']['pooled']['accuracy']):>9s} {f4(c['leave_one_map_out']['pooled']['macro_accuracy']):>10s} "
                 f"{f4(c['leave_one_map_out']['pooled']['accuracy']):>8s} {f4(s['in_sample_accuracy']):>9s} "
                 f"{f4(sl['map']) + ' (' + f4(sl['map_shuffled']) + ')':>18s} {f4(sl['type']) + ' (' + f4(sl['type_shuffled']) + ')':>18s}")
    L += [f"  majority baseline (per part): folds macro {f4(s224['scenario_folds']['pooled_majority']['macro_accuracy'])} acc "
          f"{f4(s224['scenario_folds']['pooled_majority']['accuracy'])}; LOMO macro "
          f"{f4(s224['leave_one_map_out']['pooled_majority']['macro_accuracy'])} acc "
          f"{f4(s224['leave_one_map_out']['pooled_majority']['accuracy'])}; random guess macro 0.20",
          "  Both splits are biased for collision type (results/type_probe_A, results/type_probe_A_map); the numbers are "
          "information only and change no setting.", "", "PER PART (macro / accuracy)"]
    for split, keys in (("scenario_folds", ["0", "1", "2"]), ("leave_one_map_out", MAPS)):
        L.append(f"  {split}: " + "; ".join(
            f"{k}: " + " | ".join(f"{c[split]['parts'][k]['probe']['macro_accuracy']:.3f}/{c[split]['parts'][k]['probe']['accuracy']:.3f}"
                                  for _, c, _ in rows) for k in keys) + "   (224 | V1 | V2)")
    L += ["", "PER-TYPE RECALL, POOLED"]
    for name, c, _ in rows:
        for split in ("scenario_folds", "leave_one_map_out"):
            L.append(f"  {name:36s} {split:18s} " + ", ".join(f"{t} {v:.3f}" for t, v in c[split]["pooled"]["recall"].items()))
    q = R["qlora_epoch1_fold1"]
    d = s224["scenario_folds"]["parts"]["1"]
    L += ["", "FOLD-1 CLIPS (755)", f"  {'model':36s} {'accuracy':>9s} {'macro acc':>10s}",
          f"  {'majority (' + d['majority_type'] + ')':36s} {d['majority']['accuracy']:9.4f} {d['majority']['macro_accuracy']:10.4f}",
          f"  {'QLoRA epoch 1 (window A)':36s} {q['accuracy']:9.4f} {q['macro_accuracy']:10.4f}",
          f"  {'probe A (scenario folds)':36s} {PA['folds']['1']['accuracy']:9.4f} {PA['folds']['1']['macro_accuracy']:10.4f}",
          f"  {'probe A map (leave one map out)':36s} {PM['fold1']['probe']['accuracy']:9.4f} {PM['fold1']['probe']['macro_accuracy']:10.4f}"]
    for name, c, _ in rows:
        p = c["scenario_folds"]["parts"]["1"]["probe"]
        L.append(f"  {name + ', scenario folds':36s} {p['accuracy']:9.4f} {p['macro_accuracy']:10.4f}")
    L += ["", "SAFETY, GPU TIME, MEMORY"]
    for name in ("v1", "v2"):
        v = Fj["variants"][name]
        L.append(f"  {name}: safety 5 frames twice identical {v['safety_identical']} (max |diff| {v['safety_max_abs_diff']:.1e}); "
                 f"same frames in the full run max |diff| {v['safety_vs_full_run_max_abs_diff']:.2e}; batch {v['batch']}; GPU "
                 f"{v['gpu_s']:.1f} s for 4422 frames ({v['ms_per_frame_gpu']:.2f} ms per frame); peak GPU memory allocated "
                 f"{v['peak_gpu_mem_allocated_gb']:.2f} GB (reserved {v['peak_gpu_mem_reserved_gb']:.2f} GB)")
    L += [f"  GPU {Fj['gpu']}; power at start {Fj['power_start']}, at end {Fj['power_end']}",
          f"  features: {Fj['file']} (sha256 {Fj['sha256'][:16]}...)", "",
          "Contact sheets: contact_sheet_v1.png (processor output, de-normalised), contact_sheet_v2.png (518 x 518); "
          "same 6 clips as results/type_dinov2/contact_sheet.png.",
          "Files: PLAN.txt, contact_sheet_v1.png, contact_sheet_v2.png, features.json, checks.json, predictions_checks.csv, "
          "run_log_*.txt, report.txt; cache/type_dinov2_variants/ (gitignored)."]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    modes = ("frames", "features", "checks", "report")
    for m in modes:
        g.add_argument(f"--{m}", action="store_true")
    a = ap.parse_args()
    {"frames": frames, "features": features, "checks": checks, "report": report}[next(m for m in modes if getattr(a, m))]()


if __name__ == "__main__":
    main()
