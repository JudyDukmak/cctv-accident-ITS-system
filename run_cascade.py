"""Cascade runner for the single real run: Classifier t_hat -> window A -> QLoRA epoch-1 answer (type, box).

For each clip and each Classifier seed it
  a. takes t_hat from that seed's Classifier prediction CSV;
  b. builds window A exactly as the QLoRA frame cache was built: vz.window_times(t_hat, bq.HALF_S, bq.N_FRAMES) +
     vz.to_indices (the manifest rule), then build_qlora_data.cache_clip (vz.decode, model_frames 452x254 -> 448x256
     PIL bicubic, JPEG quality 95, the cache's .npz format) into this run's own frame folder, read back with
     build_qlora_data.read_npz (all imported; bq.CACHE / bq.CLIP_INFO are pointed at this run in memory only);
  c. runs the checkpoint named in results/qlora_train/checkpoint_choice.txt (epoch 1) on the base model of
     train_qlora.load (NF4), with train_qlora.example (same prompt, processor settings, frame indices and fps) and the
     epoch-1 validation's generate call (greedy, max_new_tokens 64, use_cache) and decoding;
  d. parses the answer with narrow_prompt_v2.parse_v2 and writes one row per clip and seed.
The type source is a setting (--type-source; now only "qlora"): x, y always come from the QLoRA answer, the type from
the chosen source, so a separate type model can be added later in type_from_source() only.

Data modes:
  --data synthetic  every path must be a synthetic clip; clip info from vz.load_clips (metadata-synthetic.csv).
  --data real       refuses to start unless results/real_cascade/APPROVED.txt exists (this script never creates it).
                    Clip info from metadata-real.csv, keeping ONLY path, no_frames, duration, height, width (the
                    columns build_frame_cache.py --real reads; fps = no_frames / duration, the Classifier's rule).
Nothing real is read before the APPROVED.txt check.

Modes (from cctv-accident-ITS-system/):
  .venv-vlm\\Scripts\\python.exe run_cascade.py --make-dry-list
  .venv-vlm\\Scripts\\python.exe -u run_cascade.py --run --data synthetic --clips results\\cascade_dry_run\\dry_list.csv
        --pred 0=results\\diagnostics\\C_cctv_seeds012\\seed0_frozen\\tta_clip_predictions_flip.csv
        --out results\\cascade_dry_run --frames-dir cache\\cascade_dry_run_v1
  .venv-vlm\\Scripts\\python.exe run_cascade.py --compare          (dry run vs results/qlora_train val_epoch1 + cache)
  .venv-ml\\Scripts\\python.exe run_cascade.py --score --data synthetic --out results\\cascade_dry_run
  .venv-ml\\Scripts\\python.exe run_cascade.py --report
Mains power is checked at the start, every POWER_CHECK_EVERY windows and at the end (on battery: stop, exit 4; the
run resumes from predictions.csv).
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
DATASET = REPO_ROOT / "ACCIDENT" / "dataset"
META_SYN = DATASET / "metadata-synthetic.csv"
META_REAL = DATASET / "metadata-real.csv"
REAL_META_KEEP = ("path", "no_frames", "duration", "height", "width")  # no label column is kept
APPROVED = REPO_ROOT / "results" / "real_cascade" / "APPROVED.txt"
DRY = REPO_ROOT / "results" / "cascade_dry_run"
DRY_LIST_CSV, DRY_LIST_TXT = DRY / "dry_list.csv", DRY / "LIST.txt"
MANIFEST = REPO_ROOT / "results" / "qlora_data" / "manifest.csv"
QT = REPO_ROOT / "results" / "qlora_train"
CHOICE = QT / "checkpoint_choice.txt"
SCORER_CONFIG = REPO_ROOT / "scorer_config.json"
SEED0_PRED = REPO_ROOT / "results" / "diagnostics" / "C_cctv_seeds012" / "seed0_frozen" / "tta_clip_predictions_flip.csv"
DRY_SEED, DRY_N = 0, 100
TYPE_SOURCES = ("qlora",)
POWER_CHECK_EVERY = 25
COLUMNS = ["clip", "seed", "t_hat", "x", "y", "type", "type_source", "raw_answer", "valid", "reason", "bbox",
           "qlora_type", "status", "frame_first", "frame_last", "win_start_s", "win_end_s", "frames_file",
           "prep_s", "gpu_s", "new_tokens", "finished_utc"]


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rcsv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def norm(p):
    return p.replace("\\", "/")


def stem(p):
    return Path(norm(p)).stem


def chosen_checkpoint():
    m = re.search(r"CHOSEN: epoch (\d+) -> (.+?) \(", CHOICE.read_text(encoding="utf-8"))
    if not m:
        fail(f"no CHOSEN line in {CHOICE}")
    p = Path(m.group(2).strip())
    if not (p / "adapter_model.safetensors").is_file():
        fail(f"{p}: adapter_model.safetensors not found")
    return int(m.group(1)), p


# ------------------------------------------------------------------ dry-run list

def make_dry_list():
    if DRY_LIST_CSV.exists() or DRY_LIST_TXT.exists():
        fail(f"{DRY_LIST_TXT} exists; the dry-run list is drawn once")
    pool = sorted((r for r in rcsv(MANIFEST) if r["fold"] == "1" and r["window"] == "A"), key=lambda r: r["rgb_path"])
    if len(pool) != 755:
        fail(f"expected 755 fold-1 window-A rows, found {len(pool)}")
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(pool), size=DRY_N, replace=False).tolist())
    rows = [pool[i] for i in pick]
    DRY.mkdir(parents=True, exist_ok=True)
    with open(DRY_LIST_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rgb_path", "manifest_centre_s", "manifest_frame_indices", "frames_path"])
        w.writerows([(r["rgb_path"], r["centre_s"], r["frame_indices"], r["frames_path"]) for r in rows])
    txt = "\n".join([
        f"Cascade dry-run list, written {now()} before any run",
        f"{DRY_N} of the 755 fold-1 window-A rows of {MANIFEST.relative_to(REPO_ROOT)} (sha256 {sha256(MANIFEST)}): "
        "rows sorted by rgb_path, positions drawn with numpy default_rng(0).choice(755, 100, replace=False), kept in "
        "rgb_path order. t_hat: Classifier seed 0 (the source of window A), "
        f"{SEED0_PRED.relative_to(REPO_ROOT)} (sha256 {sha256(SEED0_PRED)}).",
        f"types: {dict(sorted(collections.Counter(r['type'] for r in rows).items()))}",
        f"{DRY_LIST_CSV.name} sha256 {sha256(DRY_LIST_CSV)}"]) + "\n"
    DRY_LIST_TXT.write_text(txt, encoding="utf-8")
    print(txt)


# ------------------------------------------------------------------ clip info

def clip_info_synthetic(vz, paths):
    clips = {c["rgb_path"]: c for c in vz.load_clips()}  # all 2211 synthetic clips (fps = no_frames / duration)
    out = {}
    for p in paths:
        if not norm(p).startswith("synthetic_videos/") or norm(p) not in clips:
            fail(f"{p}: not a synthetic clip (--data synthetic)")
        c = dict(clips[norm(p)])
        c.pop("t_hat", None)  # t_hat comes from the --pred CSVs only
        out[norm(p)] = c
    return out


def clip_info_real(paths):
    """Only after APPROVED.txt. Keeps path, no_frames, duration, height, width of metadata-real.csv; drops the rest."""
    keep = {}
    with open(META_REAL, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            keep[norm(r["path"])] = {k: r[k] for k in REAL_META_KEEP}  # label columns are not kept
    out = {}
    for p in paths:
        m = keep.get(norm(p))
        if m is None:
            fail(f"{p}: not in {META_REAL.name}")
        nf, dur = int(m["no_frames"]), float(m["duration"])
        out[norm(p)] = {"rgb_path": norm(p), "video": str(DATASET / norm(p)), "no_frames": nf, "duration": dur,
                        "fps": nf / dur, "width": int(m["width"]), "height": int(m["height"])}
    return out


# ------------------------------------------------------------------ type source

def type_from_source(source, qlora_answer, clip, seed):
    """The collision type of one window. Only "qlora" exists now: the QLoRA answer's (parsed) type."""
    if source == "qlora":
        return qlora_answer["collision_type"]
    fail(f"unknown type source {source!r}")


# ------------------------------------------------------------------ run

def run(args):
    if args.data == "real" and not APPROVED.is_file():  # first statement: nothing real is read before this
        fail(f"real mode needs {APPROVED} (created by the user, never by this script); not found, nothing was read")
    if args.type_source not in TYPE_SOURCES:
        fail(f"--type-source must be one of {TYPE_SOURCES}")
    sys.path.insert(0, str(REPO_ROOT))
    import vlm_zero_shot as vz  # frozen; imported only
    import narrow_prompt_v2 as n2
    import build_qlora_data as bq
    import train_qlora as tq

    out, frames_dir = Path(args.out).resolve(), Path(args.frames_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    pred_csv, raw_jsonl, info_txt = out / "predictions.csv", out / "raw_answers.jsonl", out / "RUN_INFO.txt"
    clips_list = [norm(r["rgb_path"]) for r in rcsv(args.clips)]
    if args.limit:
        clips_list = clips_list[:args.limit]
    seeds = {}
    for spec in args.pred:
        s, p = spec.split("=", 1)
        rows = {norm(r[args.path_col]): r for r in rcsv(p)}
        seeds[int(s)] = (Path(p), rows)
    info = clip_info_synthetic(vz, clips_list) if args.data == "synthetic" else clip_info_real(clips_list)
    t_hat = {}
    for s, (p, rows) in seeds.items():
        for c in clips_list:
            if c not in rows:
                fail(f"{c}: no row in {p}")
            v = float(rows[c][args.t_col])
            if not np.isfinite(v):
                fail(f"{c}: t_hat not finite in {p}")
            t_hat[(c, s)] = v
    tq.check_config()  # the frozen QLoRA decisions (prompt etc.) still match CONFIG.txt
    epoch, ckpt = chosen_checkpoint()
    done = {(r["clip"], int(r["seed"])) for r in rcsv(pred_csv)} if pred_csv.exists() else set()
    todo = [(c, s) for c in clips_list for s in sorted(seeds) if (c, s) not in done]

    def event(msg):
        line = f"{now()} {msg}"
        print(line, flush=True)
        with open(out / "run_events.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    on, ptxt = vz.power()
    event(f"START/RESUME: {len(done)} windows done, {len(todo)} to do; power {ptxt}")
    if not on:
        event("STOPPED: on battery at start")
        sys.exit(4)
    if not info_txt.exists():
        info_txt.write_text("\n".join([
            f"Cascade run started {now()}; data {args.data}; type source {args.type_source}",
            f"command: {' '.join(sys.argv)}",
            f"clips: {args.clips} ({len(clips_list)} clips, sha256 {sha256(args.clips)})",
            *[f"seed {s}: {p} (sha256 {sha256(p)}), path column {args.path_col}, t_hat column {args.t_col}"
              for s, (p, _) in sorted(seeds.items())],
            f"checkpoint: epoch {epoch} {ckpt} (adapter sha256 {sha256(ckpt / 'adapter_model.safetensors')}), from "
            f"{CHOICE.relative_to(REPO_ROOT)}",
            f"prompt sha256 {vz.text_sha(tq.PROMPT)} (train_qlora.PROMPT; CONFIG.txt checked); greedy, max_new_tokens "
            f"{tq.MAX_NEW_TOKENS}; parser narrow_prompt_v2.parse_v2",
            f"window: vz.window_times(t_hat, {bq.HALF_S} s, {bq.N_FRAMES}) + vz.to_indices; frames: "
            f"build_qlora_data.cache_clip into {frames_dir} (JPEG quality {bq.JPEG_Q}, {bq.MODEL_W}x{bq.MODEL_H})",
            f"sha256 run_cascade.py {sha256(Path(__file__))}"]) + "\n", encoding="utf-8")
    if not todo:
        event("nothing to do")
        return

    import torch
    from peft import PeftModel

    proc, base = tq.load(lora=False)  # processor + NF4 base model, exactly as the training script loads it
    model = PeftModel.from_pretrained(base, str(ckpt), is_trainable=False)
    model.eval()
    device = next(model.parameters()).device
    event(f"model loaded: base {vz.smoke.MODELS[vz.MODEL_KEY]} ({vz.PRECISION}) + adapter {ckpt.name}")
    bq.CACHE = frames_dir  # in memory only: cache_clip writes this run's windows here, never into the QLoRA cache
    new = not pred_csv.exists()
    by_clip = {}
    for c, s in todo:
        by_clip.setdefault(c, []).append(s)
    n_done = 0
    with open(pred_csv, "a", newline="", encoding="utf-8") as fp, open(raw_jsonl, "a", encoding="utf-8") as fj:
        w = csv.DictWriter(fp, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        for c, ss in by_clip.items():
            ci = info[c]
            bq.CLIP_INFO[c] = ci
            wins = {s: vz.to_indices(ci, vz.window_times(ci, t_hat[(c, s)], bq.HALF_S, bq.N_FRAMES)) for s in ss}
            t0 = time.perf_counter()
            err = None
            try:
                bq.cache_clip(c, [(f"s{s}", wins[s]) for s in ss])
            except (SystemExit, RuntimeError) as e:  # vz.decode / model_frames refuse: recorded, no answer invented
                err = str(e)
            prep = (time.perf_counter() - t0) / len(ss)
            for s in ss:
                idx = wins[s]
                ff = frames_dir / f"{stem(c)}_s{s}.npz"
                row = {"clip": c, "seed": s, "t_hat": t_hat[(c, s)], "type_source": args.type_source,
                       "frame_first": idx[0], "frame_last": idx[-1], "win_start_s": idx[0] / ci["fps"],
                       "win_end_s": idx[-1] / ci["fps"], "frames_file": str(ff), "prep_s": prep, "finished_utc": now()}
                if err is not None:
                    row.update(status=f"ERROR frames: {err}", valid=False, raw_answer="", x="", y="", type="")
                else:
                    r = {"frames_path": str(ff), "idx": idx, "fps": ci["fps"], "no_frames": ci["no_frames"],
                         "duration": ci["duration"],
                         "box": [0, 0, 1000, 1000], "type": "rear-end", "visible": False}  # target placeholders: unused
                    inputs, _, _ = tq.example(proc, r, flip=False, with_target=False)
                    n_in = inputs["input_ids"].shape[1]
                    x = tq.to_device(inputs, device)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    with torch.inference_mode():
                        o = model.generate(**x, max_new_tokens=tq.MAX_NEW_TOKENS, do_sample=False, use_cache=True,
                                           temperature=None, top_p=None, top_k=None)
                    torch.cuda.synchronize()
                    gpu_s = time.perf_counter() - t1
                    text = proc.batch_decode(o[0, n_in:][None], skip_special_tokens=True)[0]
                    p = n2.parse_v2(text)
                    row.update(status="ok", raw_answer=text, valid=p["valid"], reason=p["reason"],
                               bbox=json.dumps(p["bbox_2d"]), qlora_type=p["collision_type"], x=p["x"], y=p["y"],
                               type=type_from_source(args.type_source, p, c, s), gpu_s=gpu_s,
                               new_tokens=int(o.shape[1] - n_in))
                    fj.write(json.dumps({"clip": c, "seed": s, "output": text}) + "\n")
                w.writerow(row)
                fp.flush()
                fj.flush()
                n_done += 1
                if n_done % POWER_CHECK_EVERY == 0 and n_done < len(todo):
                    on, ptxt = vz.power()
                    event(f"{len(done) + n_done}/{len(done) + len(todo)} windows; power {ptxt}")
                    if not on:
                        event(f"STOPPED (resumable): on battery after {len(done) + n_done} windows")
                        sys.exit(4)
    on, ptxt = vz.power()
    event(f"FINISHED: {len(done) + n_done} windows; power at end {ptxt}")


# ------------------------------------------------------------------ dry-run comparison

def compare():
    sys.path.insert(0, str(REPO_ROOT))
    import build_qlora_data as bq

    out_json = DRY / "compare.json"
    if out_json.exists():
        fail(f"{out_json} exists; refusing to overwrite")
    lst = rcsv(DRY_LIST_CSV)
    pred = {r["clip"]: r for r in rcsv(DRY / "predictions.csv") if int(r["seed"]) == DRY_SEED}
    val = {r["rgb_path"]: r for r in rcsv(QT / "val_epoch1.csv")}
    raw_saved = {json.loads(x)["clip"]: json.loads(x)["output"]
                 for x in (QT / "val_epoch1_raw.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
    cfg = json.loads(SCORER_CONFIG.read_text(encoding="utf-8"))
    sx, sy = cfg["sigma_x"] * cfg["spatial_k"], cfg["sigma_y"] * cfg["spatial_k"]
    if len(pred) != len(lst):
        fail(f"predictions has {len(pred)} seed-{DRY_SEED} rows, list has {len(lst)}")
    F, A = [], []
    for L in lst:
        c = L["rgb_path"]
        p, v = pred[c], val[c]
        fi_r, fr_r = bq.read_npz(Path(p["frames_file"]))
        fi_c, fr_c = bq.read_npz(REPO_ROOT / L["frames_path"])
        with np.load(p["frames_file"]) as zr, np.load(REPO_ROOT / L["frames_path"]) as zc:
            bytes_equal = bool(np.array_equal(zr["data"], zc["data"]) and np.array_equal(zr["offsets"], zc["offsets"]))
        d = np.abs(fr_r.astype(np.int16) - fr_c.astype(np.int16))
        F.append({"clip": c, "t_hat_equal_manifest_centre": abs(float(p["t_hat"]) - float(L["manifest_centre_s"])) < 1e-9,
                  "indices_equal": fi_r == fi_c == json.loads(L["manifest_frame_indices"]),
                  "jpeg_bytes_equal": bytes_equal, "max_abs": int(d.max()), "mean_abs": float(d.mean())})
        A.append({"clip": c, "type_true": v["type"], "cx": float(v["center_x"]), "cy": float(v["center_y"]),
                  "raw_equal": p["raw_answer"] == raw_saved[c], "runner_type": p["type"], "saved_type": v["pred_type"],
                  "runner_xy": (float(p["x"]), float(p["y"])), "saved_xy": (float(v["x"]), float(v["y"])),
                  "runner_bbox": p["bbox"], "saved_bbox": v["bbox"], "runner_raw": p["raw_answer"], "saved_raw": raw_saved[c],
                  "runner_valid": p["valid"], "saved_valid": v["valid"]})

    def acc_s(key_type, key_xy):
        acc = float(np.mean([a[key_type] == a["type_true"] for a in A]))
        s = float(np.mean([np.exp(-((a[key_xy][0] - a["cx"]) ** 2 / (2 * sx ** 2) + (a[key_xy][1] - a["cy"]) ** 2 / (2 * sy ** 2)))
                           for a in A]))
        return acc, s

    ar, sr = acc_s("runner_type", "runner_xy")
    asv, ssv = acc_s("saved_type", "saved_xy")
    diffs = [a for a in A if a["runner_type"] != a["saved_type"] or a["runner_bbox"] != a["saved_bbox"]]
    res = {"n": len(A),
           "frames": {"all_indices_equal": all(f["indices_equal"] for f in F),
                      "all_t_hat_equal_manifest_centre": all(f["t_hat_equal_manifest_centre"] for f in F),
                      "all_jpeg_bytes_equal": all(f["jpeg_bytes_equal"] for f in F),
                      "max_abs_over_all": max(f["max_abs"] for f in F), "mean_abs_over_all": float(np.mean([f["mean_abs"] for f in F]))},
           "answers": {"raw_identical": sum(a["raw_equal"] for a in A),
                       "type_or_box_differs": [{k: a[k] for k in ("clip", "type_true", "runner_type", "saved_type",
                                                                  "runner_bbox", "saved_bbox", "runner_raw", "saved_raw")}
                                               for a in diffs]},
           "metrics": {"runner": {"type_accuracy": ar, "S": sr}, "saved_epoch1": {"type_accuracy": asv, "S": ssv},
                       "abs_diff": {"type_accuracy": abs(ar - asv), "S": abs(sr - ssv)}}}
    res["PASS"] = bool(res["frames"]["all_indices_equal"] and res["frames"]["max_abs_over_all"] == 0
                       and res["metrics"]["abs_diff"]["type_accuracy"] <= 0.005 and res["metrics"]["abs_diff"]["S"] <= 0.005)
    out_json.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "answers"}, indent=1))
    print("raw identical", res["answers"]["raw_identical"], "/", res["n"], "; type or box differs:",
          len(res["answers"]["type_or_box_differs"]))


# ------------------------------------------------------------------ scoring (official scorer)

def score(args):
    """accs_scorer.score_all per seed: T from the Classifier t_hat, S and C from the runner; then mean over seeds."""
    if args.data == "real" and not APPROVED.is_file():
        fail(f"real scoring needs {APPROVED}; not found, nothing was read")
    import pandas as pd
    sys.path.insert(0, str(REPO_ROOT))
    import accs_scorer as acs

    out = Path(args.out).resolve()
    out_json = out / "score.json"
    if out_json.exists():
        fail(f"{out_json} exists; refusing to overwrite")
    pred = pd.read_csv(out / "predictions.csv")
    if (pred["status"] != "ok").any():
        fail(f"{int((pred['status'] != 'ok').sum())} windows have no answer (status not ok); the scorer needs one per clip")
    if args.data == "synthetic":
        gt = pd.read_csv(META_SYN, usecols=["rgb_path", "accident_time", "center_x", "center_y", "type"]).rename(
            columns={"rgb_path": "path"})
        gt = gt[gt["path"].map(norm).isin(set(pred["clip"]))].reset_index(drop=True)
    else:  # real: the frozen test set's labels, read only here and only with APPROVED.txt
        gt = pd.read_csv(REPO_ROOT / "real_test_set.csv")
        missing = [c for c in acs.GT_COLUMNS if c not in gt.columns]
        if missing:
            fail(f"real_test_set.csv lacks {missing}; the real scoring path must be decided")
        gt = gt[acs.GT_COLUMNS]
    cfg = json.loads(SCORER_CONFIG.read_text(encoding="utf-8"))
    res = {"sigma_x": cfg["sigma_x"], "sigma_y": cfg["sigma_y"], "per_seed": {}}
    for s, d in pred.groupby("seed"):
        pdf = pd.DataFrame({"path": d["clip"], "t_hat": d["t_hat"], "x_hat": d["x"], "y_hat": d["y"], "type_hat": d["type"]})
        r = acs.score_all(gt, pdf, cfg["sigma_x"], cfg["sigma_y"])
        res["per_seed"][str(int(s))] = r
    keys = ["T_sigma_0.5", "T_sigma_1", "T_sigma_2", "S", "C", "C_macro", "ACCS"]
    res["mean_over_seeds"] = {k: float(np.mean([r[k] for r in res["per_seed"].values()])) for k in keys}
    res["std_over_seeds"] = {k: float(np.std([r[k] for r in res["per_seed"].values()], ddof=1))
                             if len(res["per_seed"]) > 1 else None for k in keys}
    if args.data == "synthetic":  # the same 100 clips scored with the saved epoch-1 answers (reference)
        val = pd.read_csv(QT / "val_epoch1.csv")
        d0 = pred[pred["seed"] == sorted(pred["seed"].unique())[0]]
        v = val.set_index("rgb_path").loc[d0["clip"]]
        pdf = pd.DataFrame({"path": d0["clip"].values, "t_hat": d0["t_hat"].values, "x_hat": v["x"].values,
                            "y_hat": v["y"].values, "type_hat": v["pred_type"].values})
        res["saved_epoch1_same_clips_same_T"] = acs.score_all(gt, pdf, cfg["sigma_x"], cfg["sigma_y"])
    out_json.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))


# ------------------------------------------------------------------ report

def report():
    out = DRY / "report.txt"
    if out.exists():
        fail(f"{out} exists; refusing to overwrite")
    C = json.loads((DRY / "compare.json").read_text(encoding="utf-8"))
    S = json.loads((DRY / "score.json").read_text(encoding="utf-8"))
    P = rcsv(DRY / "predictions.csv")
    ev = (DRY / "run_events.txt").read_text(encoding="utf-8").strip().splitlines()
    gpu = np.array([float(r["gpu_s"]) for r in P])
    prep = np.array([float(r["prep_s"]) for r in P])
    load_line = next((x for x in ev if "model loaded" in x), "")
    start = datetime.fromisoformat(ev[0].split(" ")[0]) if ev else None
    loaded = datetime.fromisoformat(load_line.split(" ")[0]) if load_line else None
    load_s = (loaded - start).total_seconds() if start and loaded else float("nan")
    n_real = 400 * 3
    vm = json.loads((QT / "val_epoch1_metrics.json").read_text(encoding="utf-8"))
    fold_share = {r["fold"]: float(r["share_inside"]) for r in rcsv(QT / "inside_outside" / "5_share_inside_by_fold.csv")}
    est = n_real * (gpu.mean() + prep.mean())
    L = [f"CASCADE DRY RUN (synthetic only; no real clip, real label or real metadata file opened) - report {now()}",
         "", "1. RUNNER", "  run_cascade.py; RUN_INFO.txt:"]
    L += ["    " + x for x in (DRY / "RUN_INFO.txt").read_text(encoding="utf-8").strip().splitlines()]
    L += [f"  power: {ev[0]}", f"         {ev[-1]}",
          f"  real mode refusal test: see run_log_real_refusal.txt",
          "", "2. DRY RUN vs THE SAVED EPOCH-1 VALIDATION (same 100 fold-1 window-A windows; LIST.txt)",
          f"  frames: frame indices equal to the manifest for all windows {C['frames']['all_indices_equal']}; t_hat equal "
          f"to the manifest window-A centre {C['frames']['all_t_hat_equal_manifest_centre']}; JPEG bytes identical to the "
          f"cache {C['frames']['all_jpeg_bytes_equal']}; decoded pixel difference vs the cache: max "
          f"{C['frames']['max_abs_over_all']}, mean {C['frames']['mean_abs_over_all']:.4f}",
          f"  answers: raw answer identical {C['answers']['raw_identical']}/{C['n']}; windows whose type or box differs: "
          f"{len(C['answers']['type_or_box_differs'])}"]
    for d in C["answers"]["type_or_box_differs"]:
        L.append(f"    {stem(d['clip'])} (true {d['type_true']}): runner {d['runner_type']} {d['runner_bbox']} | saved "
                 f"{d['saved_type']} {d['saved_bbox']}")
    m = C["metrics"]
    L += [f"  metrics on these {C['n']}: runner type accuracy {m['runner']['type_accuracy']:.4f}, S {m['runner']['S']:.4f}; "
          f"saved epoch-1 type accuracy {m['saved_epoch1']['type_accuracy']:.4f}, S {m['saved_epoch1']['S']:.4f}; "
          f"|difference| {m['abs_diff']['type_accuracy']:.4f} and {m['abs_diff']['S']:.4f} (tolerance 0.005)",
          f"  VERDICT: {'PASS' if C['PASS'] else 'FAIL'} (frames identical and both metrics within 0.005)",
          "", "3. OFFICIAL SCORER (accs_scorer.score_all; T from the Classifier t_hat, S and C from the runner; sigma_x "
          f"{S['sigma_x']:.6f}, sigma_y {S['sigma_y']:.6f} from scorer_config.json; ACCS at sigma_t 1 s)"]
    for s, r in S["per_seed"].items():
        L.append(f"  seed {s}: N {r['N']}; T@0.5 {r['T_sigma_0.5']:.4f}, T@1 {r['T_sigma_1']:.4f}, T@2 "
                 f"{r['T_sigma_2']:.4f}; S {r['S']:.4f}; C {r['C']:.4f} (macro {r['C_macro']:.4f}); ACCS {r['ACCS']:.4f}")
    mo = S["mean_over_seeds"]
    L.append(f"  mean over seeds ({len(S['per_seed'])} seed(s)): T@1 {mo['T_sigma_1']:.4f}, S {mo['S']:.4f}, C {mo['C']:.4f}, "
             f"ACCS {mo['ACCS']:.4f}")
    if "saved_epoch1_same_clips_same_T" in S:
        r = S["saved_epoch1_same_clips_same_T"]
        L.append(f"  reference, saved epoch-1 answers on the same clips with the same T: S {r['S']:.4f}, C {r['C']:.4f}, "
                 f"ACCS {r['ACCS']:.4f}")
    L += ["", "4. TIME",
          f"  generate (GPU) per window: mean {gpu.mean():.2f} s, median {np.median(gpu):.2f} s, max {gpu.max():.2f} s "
          f"(n {len(gpu)}); frame building (decode + resize + JPEG, CPU) per window: mean {prep.mean():.2f} s, max "
          f"{prep.max():.2f} s; model load {load_s:.0f} s",
          f"  estimate for 400 clips x 3 seeds = {n_real} windows: {n_real} x ({gpu.mean():.2f} + {prep.mean():.2f}) s = "
          f"{est / 3600:.2f} h, plus the model load. In the real run each clip is decoded once for its 3 windows, so "
          "the frame-building share can be smaller; but vz.decode reads from the start of the clip to the last frame "
          "needed, so its cost grows with t_hat and with the real clips' length and resolution, which this synthetic "
          "dry run does not measure.",
          "", "5. STILL UNDECIDED",
          f"  - type source: only 'qlora' exists (epoch-1 type accuracy {vm['type_accuracy']:.4f} on fold 1 window A, "
          f"below the majority-type baseline {vm['majority_accuracy']:.4f}); a separate type model would be added in "
          "type_from_source() (--type-source).",
          f"  - window: window A = t_hat +- 2 s, 16 frames; t* lies outside it for {1 - fold_share['all']:.4f} of all "
          f"synthetic window-A windows (fold 1: {1 - fold_share['1']:.4f}), where S drops (results/qlora_train/"
          "inside_outside/report.txt). Whether to keep, widen or move it is open.",
          "  - real clips whose decoded size is not 452 x 254 (aspect other than 16:9): build_qlora_data.model_frames "
          "refuses them; the runner records 'ERROR frames' and no answer, and the scorer refuses to score. A rule is "
          "needed before the real run (the QLoRA model saw only 448 x 256 frames).",
          "  - real files that end before metadata no_frames (decision 2026-11-15): a window reaching past the last "
          "decodable frame makes vz.decode refuse; same handling as above, rule needed.",
          "  - the real Classifier prediction files (columns, fallback rows, non-finite t_hat) were not opened here; the "
          "runner checks the given path and t_hat columns at start and refuses a missing clip or a non-finite t_hat.",
          "  - real scoring reads real_test_set.csv (only with APPROVED.txt) and needs the columns path, accident_time, "
          "center_x, center_y, type; not checked here (no real file opened).",
          "  - how the 3 seeds are reported (mean and SD over seeds, as for the Classifier) is implemented; confirm."]
    text = "\n".join(L) + "\n"
    out.write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for m in ("--make-dry-list", "--run", "--compare", "--score", "--report"):
        g.add_argument(m, action="store_true")
    ap.add_argument("--data", choices=("synthetic", "real"))
    ap.add_argument("--clips")
    ap.add_argument("--pred", action="append", default=[], help="SEED=CSV (repeat per seed)")
    ap.add_argument("--path-col", default="rgb_path")
    ap.add_argument("--t-col", default="t_hat")
    ap.add_argument("--out")
    ap.add_argument("--frames-dir")
    ap.add_argument("--type-source", default="qlora")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.make_dry_list:
        make_dry_list()
    elif a.run:
        if a.data == "real" and not APPROVED.is_file():
            fail(f"real mode needs {APPROVED} (created by the user, never by this script); not found, nothing was read")
        if not (a.data and a.clips and a.pred and a.out and a.frames_dir):
            fail("--run needs --data, --clips, --pred, --out and --frames-dir")
        run(a)
    elif a.compare:
        compare()
    elif a.score:
        if a.data == "real" and not APPROVED.is_file():
            fail(f"real scoring needs {APPROVED}; not found, nothing was read")
        if not (a.data and a.out):
            fail("--score needs --data and --out")
        score(a)
    else:
        report()


if __name__ == "__main__":
    main()
