"""Measured cost per clip of the pipeline parts on this GPU (supervisor feedback, point 4). Synthetic clips only.

Two synthetic clips (the stage-test clips: about median length and the longest synthetic clip); each part is run once
as a warm-up and then REPEATS times; the report gives the median of the repeats and the range. Models are loaded once
before the timing (load time and weight memory are reported apart). Mains power is checked at the start and the end.

--classifier (run with .venv-ml): the frozen classifier C_cctv seed 0 as used on the real test: the clip's 8-fps frame
  grid is decoded and cached exactly as build_frame_cache.py does (PyAV sequential decode, 224 x 224 INTER_AREA, JPEG
  q95; into a temporary folder, never the project caches), its windows by build_frame_cache.windows_table, then the
  3 fold models x 2 passes (original + horizontally flipped frames) score every window with train_classifier.val_logits
  (bf16 autocast, one batch per val_batch of the fold's config; data loader with 0 workers for a single clip), the
  flip TTA mean per model, the mean over the 3 models, and aggregate_runs.peak_t_hat (frozen: smoothing 3, parabola).
  All three fold models stay loaded on the GPU (as a deployed system would keep them).
--vlm (run with .venv-vlm): Qwen3-VL-4B NF4 loaded once; narrow v1 (frozen), narrow v2 (narrow_prompt_v2.py) and the
  frozen full scan v3 (coarse + fine pass) on the same clips, window centre t_hat of the saved out-of-fold predictions.
--table: cost_table.txt from the two JSON files.
Numbers: GPU s = synchronised model time only (classifier forward passes incl. host-to-device copy; VLM generate);
wall s = the whole per-clip path including video decoding and preprocessing on the CPU (classifier: frame cache +
6 passes incl. JPEG decoding and normalisation + post-processing; VLM: frame decoding + processor + generate + parsing);
peak memory = torch.cuda.max_memory_allocated / max_memory_reserved of the process during the part (weights included).
Usage (from cctv-accident-ITS-system/):
    .venv-ml\\Scripts\\python.exe cost_per_clip.py --classifier
    .venv-vlm\\Scripts\\python.exe cost_per_clip.py --vlm
    .venv-vlm\\Scripts\\python.exe cost_per_clip.py --table
"""

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
import vlm_zero_shot as vz  # noqa: E402  (frozen protocol code: imported, not changed)

OUT = REPO_ROOT / "results" / "narrow_v1_v2_200" / "cost"
CLIPS = ("synthetic_videos/videos/head-on/Town06_head-on_clear_01.mp4",
         "synthetic_videos/videos/rear-end/Town04_rear-end_night_32.mp4")  # vlm_smoke_test.STAGE_CLIPS
REPEATS = 3
CLASSIFIER_RUN, FOLDS = "C_cctv_seed0", (0, 1, 2)
GB = 2 ** 30


def summary(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "values": values}


def write(name, res):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.json"
    if p.exists():
        vz.fail(f"{p} exists; refusing to overwrite")
    p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "clips"}, indent=1))
    print(f"wrote {p}")


# ---------------------------------------------------------------- classifier (.venv-ml)

def classifier():
    import pandas as pd
    import torch
    from transformers import VideoMAEImageProcessor

    import aggregate_runs as ar
    import analyse_predictions as apr
    import build_frame_cache as bfc
    import train_classifier as tc

    on, ptxt = vz.power()
    res = {"part": "classifier", "power_start": ptxt, "run": CLASSIFIER_RUN, "folds": list(FOLDS),
           "post_processing": ar.FROZEN, "repeats": REPEATS, "loader_workers": 0}
    if not on:
        vz.fail(f"not on mains: {ptxt}")
    device = torch.device("cuda")
    res["gpu"] = torch.cuda.get_device_name(0)
    t0 = time.perf_counter()
    models, cfgs, excl = {}, {}, set()
    for k in FOLDS:
        info, ck, _ = apr.load_best(CLASSIFIER_RUN, k)
        cfg = info["config"]
        m = tc.build_model(cfg, device, None if not info["drop_path_applied"] else tc.drop_path_field())
        m.load_state_dict(ck["model"])
        m.eval()
        models[k], cfgs[k] = m, cfg
        excl.add(cfg["confidence_exclude"])
        del ck
    torch.cuda.synchronize()
    res["load_s_3_models"] = time.perf_counter() - t0
    res["weights_alloc_gb"] = torch.cuda.memory_allocated(device) / GB
    if len(excl) != 1:
        vz.fail("confidence_exclude differs between folds")
    exclude = excl.pop()
    proc = VideoMAEImageProcessor.from_pretrained(cfgs[0]["model_id"])
    meta = pd.read_csv(apr.METADATA_SYN)
    meta["rgb_path"] = meta["rgb_path"].str.replace("\\", "/", regex=False)
    tmp = Path(tempfile.mkdtemp(prefix="cost_cache_"))
    res["clips"] = []
    try:
        for cp in CLIPS:
            row = meta[meta["rgb_path"] == cp]
            if len(row) != 1 or not cp.startswith("synthetic_videos/"):
                vz.fail(f"{cp}: not one synthetic clip")
            nf, dur = int(row["no_frames"].iloc[0]), float(row["duration"].iloc[0])
            reps = []
            for r in range(REPEATS + 1):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
                t_clip = time.perf_counter()
                win = bfc.windows_table(row[["rgb_path", "no_frames", "duration"]])
                stem = win["stem"].iloc[0]
                _, grid = bfc.build_grid(win, nf / dur, nf, stem)
                (tmp / f"{stem}.npz").unlink(missing_ok=True)
                st = bfc.process_clip(vz.DATASET / cp, stem, grid, tmp)
                if st[1] != "written":
                    vz.fail(f"{cp}: frame cache {st}")
                cache_s = time.perf_counter() - t_clip
                gpu_s, pass_wall, cols = 0.0, 0.0, {}
                for k in FOLDS:
                    cfg = cfgs[k]
                    ds = apr.CacheDirWindowDataset(tmp, win["stem"], win["k0"], np.zeros(len(win), np.float32), None,
                                                   list(proc.image_mean), list(proc.image_std), cfg["frames"],
                                                   cfg["model_fps"], cfg["lru_size"])
                    sampler = tc.ListSampler()
                    loader = tc.make_loader(ds, cfg["val_batch"], 0, sampler)
                    sc = {}
                    for name, flip in (("orig", False), ("hflip", True)):
                        sampler.set(range(int(flip), 2 * len(win), 2))
                        logits = np.empty(len(win), dtype=np.float32)
                        tw = time.perf_counter()
                        with torch.no_grad():
                            for x, _, _, rows, _ in loader:
                                torch.cuda.synchronize()
                                tg = time.perf_counter()
                                logits[rows.numpy()] = tc.val_logits(models[k], x, device, len(x))
                                gpu_s += time.perf_counter() - tg
                        pass_wall += time.perf_counter() - tw
                        sc[name] = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
                    cols[k] = (sc["orig"].astype(float) + sc["hflip"].astype(float)) / 2
                tp = time.perf_counter()
                ens = win[["rgb_path", "t_centre"]].assign(score=np.mean([cols[k] for k in FOLDS], axis=0))
                pred = ar.peak_t_hat(ens, ar.FROZEN["smooth"], ar.FROZEN["interp"], exclude)
                post_s = time.perf_counter() - tp
                torch.cuda.synchronize()
                wall = time.perf_counter() - t_clip
                reps.append({"cache_s": cache_s, "gpu_s": gpu_s, "passes_wall_s": pass_wall, "post_s": post_s,
                             "wall_s": wall, "peak_alloc_gb": torch.cuda.max_memory_allocated(device) / GB,
                             "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / GB,
                             "t_hat": float(pred["t_hat"].iloc[0]), "confidence": float(pred["confidence"].iloc[0])})
                vz.stamp(f"classifier {Path(cp).stem} {'warm-up' if r == 0 else f'repeat {r}'}: {len(win)} windows, "
                         f"cache {cache_s:.2f} s, GPU {gpu_s:.2f} s, wall {wall:.2f} s")
            timed = reps[1:]
            res["clips"].append({"rgb_path": cp, "duration": dur, "no_frames": nf, "windows": len(win),
                                 "passes": len(FOLDS) * 2, "val_batch": [cfgs[k]["val_batch"] for k in FOLDS],
                                 "warmup": reps[0], **{key: summary([x[key] for x in timed]) for key in timed[0]}})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    res["power_end"] = vz.power()[1]
    write("classifier", res)


# ---------------------------------------------------------------- VLM (.venv-vlm)

class Recorder:
    """Wraps vz.Vlm so the frozen full_scan_one can be timed per call; records each call's peak memory."""

    def __init__(self, vlm):
        self.vlm, self.peaks = vlm, []

    def __call__(self, *args):
        text, s = self.vlm(*args)
        self.peaks.append((self.vlm.last_peak_alloc_gb, self.vlm.last_peak_reserved_gb))
        return text, s


def vlm_part():
    import torch

    import narrow_prompt_v2 as n2

    on, ptxt = vz.power()
    res = {"part": "vlm", "power_start": ptxt, "model": vz.smoke.MODELS[vz.MODEL_KEY], "precision": vz.PRECISION,
           "repeats": REPEATS}
    if not on:
        vz.fail(f"not on mains: {ptxt}")
    vlm = vz.Vlm()
    res["gpu"], res["load_s"] = vlm.gpu, vlm.load_s
    res["weights_alloc_gb"] = torch.cuda.memory_allocated(vlm.device) / GB
    clips = {c["rgb_path"]: c for c in vz.load_clips(vz.DEV_CSV)}
    res["clips"] = []
    for cp in CLIPS:
        c = clips[cp]
        out = {"rgb_path": cp, "duration": c["duration"], "t_hat": c["t_hat"]}
        for part in ("narrow_v1", "narrow_v2", "full_scan_v3"):
            reps = []
            for r in range(REPEATS + 1):
                t0 = time.perf_counter()
                if part == "narrow_v1":
                    idx, fr, p = vz.narrow_input(c, 1)
                    text, s = vlm(c, fr, idx, p)
                    vz.narrow_answer(text)
                    rec = {"gpu_s": s, "peak_alloc_gb": vlm.last_peak_alloc_gb,
                           "peak_reserved_gb": vlm.last_peak_reserved_gb, "answer": text}
                elif part == "narrow_v2":
                    _, fr, idx, p = n2.narrow_v2_input(c)
                    text, s, nt = n2.generate_v2(vlm, c, fr, idx, p)
                    n2.parse_v2(text)
                    rec = {"gpu_s": s, "peak_alloc_gb": vlm.last_peak_alloc_gb,
                           "peak_reserved_gb": vlm.last_peak_reserved_gb, "new_tokens": nt, "answer": text}
                else:
                    rw = Recorder(vlm)
                    row = vz.full_scan_one(rw, c)
                    rec = {"gpu_s": row["gpu_s_coarse"] + row["gpu_s_fine"], "passes": 1 + (row["n_fine"] > 0),
                           "peak_alloc_gb": max(a for a, _ in rw.peaks), "peak_reserved_gb": max(b for _, b in rw.peaks),
                           "answer": f"t_c {row['t_c']} t_full {row['t_full']}"}
                rec["wall_s"] = time.perf_counter() - t0
                reps.append(rec)
                vz.stamp(f"{part} {Path(cp).stem} {'warm-up' if r == 0 else f'repeat {r}'}: GPU {rec['gpu_s']:.2f} s, "
                         f"wall {rec['wall_s']:.2f} s, peak {rec['peak_alloc_gb']:.2f} GB")
            timed = reps[1:]
            out[part] = {"warmup": reps[0], "same_answer_all_runs": len({x["answer"] for x in reps}) == 1,
                         **{k: summary([x[k] for x in timed]) for k in ("gpu_s", "wall_s", "peak_alloc_gb",
                                                                         "peak_reserved_gb")}}
            if part == "narrow_v2":
                out[part]["new_tokens"] = [x["new_tokens"] for x in reps]
            if part == "full_scan_v3":
                out[part]["passes"] = [x["passes"] for x in reps]
        res["clips"].append(out)
    res["power_end"] = vz.power()[1]
    write("vlm", res)


# ---------------------------------------------------------------- table

def table():
    cl = json.loads((OUT / "classifier.json").read_text(encoding="utf-8"))
    vl = json.loads((OUT / "vlm.json").read_text(encoding="utf-8"))
    if cl["gpu"] != vl["gpu"]:
        vz.fail("different GPUs")
    L = [f"Measured cost per clip on {cl['gpu']}, on mains (classifier: {cl['power_start']} -> {cl['power_end']}; "
         f"VLM: {vl['power_start']} -> {vl['power_end']}). Median of {REPEATS} repeats after one warm-up "
         "[min-max]. Synthetic clips only.", ""]

    def cell(d, key, f="{:.2f}"):
        return f"{f.format(d[key]['median'])} [{f.format(d[key]['min'])}-{f.format(d[key]['max'])}]"

    for i, cp in enumerate(CLIPS):
        c, v = cl["clips"][i], vl["clips"][i]
        L.append(f"{Path(cp).stem}: duration {c['duration']:.2f} s, {c['windows']} classifier windows")
        L.append(f"  {'part':44s} {'GPU s':>18s} {'wall s':>18s} {'peak alloc GB':>18s} {'peak reserved GB':>18s}")
        L.append(f"  {'classifier (3 fold models x 2 passes)':44s} {cell(c, 'gpu_s'):>18s} {cell(c, 'wall_s'):>18s} "
                 f"{cell(c, 'peak_alloc_gb'):>18s} {cell(c, 'peak_reserved_gb'):>18s}")
        L.append(f"    of which frame cache (CPU) {cell(c, 'cache_s')} s; 6 passes incl. data loading "
                 f"{cell(c, 'passes_wall_s')} s; post-processing {cell(c, 'post_s', '{:.4f}')} s")
        for part, name in (("narrow_v1", "narrow v1 (16 frames, 896 visual tokens)"),
                           ("narrow_v2", "narrow v2 (16 frames duplicated, 1792 tokens)"),
                           ("full_scan_v3", "full scan v3 (coarse + fine pass)")):
            d = v[part]
            L.append(f"  {name:44s} {cell(d, 'gpu_s'):>18s} {cell(d, 'wall_s'):>18s} {cell(d, 'peak_alloc_gb'):>18s} "
                     f"{cell(d, 'peak_reserved_gb'):>18s}")
        L.append(f"    narrow v2 new tokens per run {v['narrow_v2']['new_tokens']}; full scan passes "
                 f"{v['full_scan_v3']['passes']}; same answer in every run: v1 {v['narrow_v1']['same_answer_all_runs']}, "
                 f"v2 {v['narrow_v2']['same_answer_all_runs']}, full scan {v['full_scan_v3']['same_answer_all_runs']}")
        L.append("")
    L += [f"one-off (not per clip): classifier 3 models load {cl['load_s_3_models']:.1f} s, weights "
          f"{cl['weights_alloc_gb']:.2f} GB allocated; VLM load {vl['load_s']:.1f} s, weights {vl['weights_alloc_gb']:.2f} GB",
          "Definitions: GPU s = synchronised model time only (classifier: forward passes incl. host-to-device copy; "
          "VLM: generate). Wall s = the whole per-clip path incl. CPU video decoding and preprocessing (classifier: "
          "frame cache build + 6 passes incl. JPEG decoding / normalisation + post-processing; VLM: frame decoding + "
          "processor + generate + parsing). Peak = torch max allocated / reserved in the process during the part, "
          "weights included (classifier: all 3 fold models resident; VLM: the 4-bit model). Classifier data loader "
          "with 0 workers (one clip); the real-test scoring used 4 workers over 2,027 clips."]
    text = "\n".join(L) + "\n"
    (OUT / "cost_table.txt").write_text(text, encoding="utf-8")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--classifier", action="store_true")
    g.add_argument("--vlm", action="store_true")
    g.add_argument("--table", action="store_true")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if args.classifier:
        classifier()
    elif args.vlm:
        vlm_part()
    else:
        table()


if __name__ == "__main__":
    main()
