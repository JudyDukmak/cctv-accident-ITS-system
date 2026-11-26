"""QLoRA fine-tuning of the narrow call (type and collision box) on synthetic windows. SYNTHETIC ONLY.

A new script because no VLM training script exists. It reuses the cached frames and labels of build_qlora_data.py
(results/qlora_data/manifest.csv, cache/qlora_frames448_v1/), the prompt / loss-mask builder and the model loader of
gpu_check/vlm_smoke_test.py, the answer parser of narrow_prompt_v2.py and the LoRA target set of
vlm_qlora_memory_test.py. The frozen files are imported, not changed. No real clip or real label is opened.

Modes (from cctv-accident-ITS-system/, all with .venv-vlm\\Scripts\\python.exe -u):
  --write-config   results/qlora_train/CONFIG.txt (date, git commit, file hashes, the decisions; written once,
                   before any training step; every GPU mode refuses to run if the decisions differ from it)
  --checks         CPU unit checks: flip of frames + box (grid), targets parse with narrow_prompt_v2.parse_v2, loss
                   mask covers only the answer
  --baseline       the untrained model with the same prompt on fold 1, window A
  --smoke --max-steps N   the training loop for N steps into results/qlora_train/smoke/ (weights runs/qlora_train_smoke/)
  --train          the full training (2 epochs; resumable; validation after each epoch; checkpoint choice)
Watch the full training:  Get-Content results\\qlora_train\\run.log -Wait -Tail 20
"""

import argparse
import collections
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
import vlm_zero_shot as vz  # noqa: E402  (frozen protocol code: imported, not changed)
import narrow_prompt_v2 as n2  # noqa: E402
import build_qlora_data as bq  # noqa: E402

smoke = vz.smoke
OUT = REPO_ROOT / "results" / "qlora_train"
WEIGHTS = REPO_ROOT / "runs" / "qlora_train"
CONFIG_TXT = OUT / "CONFIG.txt"
SCORER_CONFIG = REPO_ROOT / "scorer_config.json"
LORA_TARGET = (r".*language_model.*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
               r"|mlp\.(gate_proj|up_proj|down_proj))")  # = vlm_qlora_memory_test.LORA_TARGET
GB = 2 ** 30

# ---------------------------------------------------------------- prompt (hybrid)
PART1 = ("These 16 frames come from a fixed traffic camera, in time order, covering about 4 seconds. The text before "
         "the frames gives their times in seconds. A traffic collision is expected near the middle of these frames, "
         "but it may happen earlier or later, or may not be visible.")
_a = n2.PROMPT_V2.index("   - head-on:")
_b = n2.PROMPT_V2.index("prefer rear-end over sideswipe.") + len("prefer rear-end over sideswipe.")
DEFINITIONS = n2.PROMPT_V2[_a:_b]  # the five type definitions and the two tie-break rules, verbatim
ANSWER = ('Answer with one JSON object and nothing else, in this exact form: {"collision_visible": <true or false>, '
          '"collision_type": "<head-on | rear-end | sideswipe | single | t-bone>", "bbox_2d": [x1, y1, x2, y2]}. '
          "bbox_2d uses relative coordinates from 0 to 1000: x from the left edge, y from the top edge, [x1, y1] the "
          "top-left corner and [x2, y2] the bottom-right corner. Always give a collision_type and a bbox_2d, even if "
          "collision_visible is false.")
PROMPT = PART1 + "\n\n" + DEFINITIONS + "\n\n" + ANSWER

# ---------------------------------------------------------------- decisions (written to CONFIG.txt, then fixed)
CFG = {
    "input_layout": "v1: the 16 cached frames of the window (manifest frame_indices, 448 x 256 JPEG cache), native "
                    "frame pairs (2 frames per temporal token group: 8 groups, 896 visual tokens), the processor's "
                    "own timestamps from the frame indices and fps",
    "prompt_sha256": vz.text_sha(PROMPT),
    "prompt_assembly": "PART1 + blank line + PROMPT_V2 text from '   - head-on:' to 'prefer rear-end over sideswipe.' "
                       "(verbatim, incl. its 3-space indentation) + blank line + ANSWER",
    "target": '{"collision_visible": t* inside the window (first to last frame time), "collision_type": type, '
              '"bbox_2d": [x1, y1, x2, y2] = round-half-up(1000 x metadata box)} as json.dumps (keys in this order)',
    "loss": "cross-entropy on the answer tokens only: the target JSON and <|im_end|>; prompt, video and the newline "
            "after <|im_end|> masked (-100)",
    "train": "folds 0 and 2 of synthetic_split.csv, windows A and B (manifest)",
    "validation": "fold 1, window A (no flip)",
    "single_oversampling": "single training windows repeated until their count equals the t-bone training windows: "
                           "whole copies, then the remainder drawn without replacement with default_rng(seed)",
    "augmentation": "horizontal flip with p 0.5 per training example per epoch (all 16 frames; box x1' = 1000 - x2, "
                    "x2' = 1000 - x1); nothing else",
    "model": "Qwen/Qwen3-VL-4B-Instruct, 4-bit NF4 (double quant, bf16 compute), sdpa attention",
    "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "bias": "none", "target_modules": LORA_TARGET},
    "gradient_checkpointing": "on (non-reentrant), use_cache off in training",
    "batch": 1,
    "optimizer": "torch.optim.AdamW, lr 2e-4, torch defaults otherwise (betas 0.9 / 0.999, eps 1e-8, weight_decay 0.01)",
    "schedule": "cosine with 3% linear warmup (transformers get_cosine_schedule_with_warmup), stepped every example, "
                "over 2 epochs",
    "lr": 2e-4, "warmup_frac": 0.03, "epochs": 2, "seed": 0,
    "gradient_clipping": "none (not in the decisions)",
    "epoch_order": "default_rng(seed * 1000 + epoch).permutation, flips default_rng(seed * 1000 + epoch) after it",
    "validation_decoding": "greedy, max_new_tokens 64; answer parsed by narrow_prompt_v2.parse_v2 (point = box centre; "
                           "invalid -> rear-end, (0.5, 0.5), counted)",
    "validation_metrics": "type accuracy, per-type confusion, S (scorer_config.json sigmas), mean point error, valid "
                          "answers; baselines on the same windows: majority training type and the image centre",
    "checkpoint_choice": "the epoch with the higher mean of type accuracy and S on fold 1 window A",
    "save_every_steps": 500,
    "power": "mains checked at start, at every checkpoint (every 500 steps and epoch end) and at the end; on battery: "
             "checkpoint saved, run paused (exit 4), resumable",
}
SEED = CFG["seed"]
MAX_NEW_TOKENS = 64
TRAIN_FOLDS, VAL_FOLD = ("0", "2"), "1"


def now():
    return vz.now()


# ---------------------------------------------------------------- data

def round_half_up(v):
    return int(math.floor(v + 0.5))


def load_rows():
    """Manifest rows with the integer box (0-1000) from the raw metadata, no_frames and the split role."""
    meta = vz.synthetic_meta()
    rows = vz.read_csv(bq.MANIFEST)
    if len(rows) != 4422:
        vz.fail("expected 4422 manifest rows")
    for r in rows:
        m = meta[r["rgb_path"]]
        r["box"] = [round_half_up(1000 * float(m[k])) for k in ("x1", "y1", "x2", "y2")]
        r["no_frames"] = int(m["no_frames"])
        r["idx"] = json.loads(r["frame_indices"])
        r["visible"] = r["t_star_inside"] == "True"
        if not (r["box"][0] < r["box"][2] and r["box"][1] < r["box"][3]):
            vz.fail(f"{r['rgb_path']}: integer box empty")
    return rows


def splits(rows):
    train = [r for r in rows if r["fold"] in TRAIN_FOLDS]
    val = [r for r in rows if r["fold"] == VAL_FOLD and r["window"] == "A"]
    single = [r for r in train if r["type"] == "single"]
    n_tbone = sum(r["type"] == "t-bone" for r in train)
    rng = np.random.default_rng(SEED)
    k, rem = divmod(n_tbone, len(single))
    extra = [single[i] for i in rng.choice(len(single), size=rem, replace=False)] if rem else []
    train_list = [r for r in train if r["type"] != "single"] + single * k + extra
    return train, train_list, val


def epoch_plan(train_list, epoch):
    rng = np.random.default_rng(SEED * 1000 + epoch)
    order = rng.permutation(len(train_list))
    flips = rng.random(len(train_list)) < 0.5
    return [(train_list[i], bool(flips[k])) for k, i in enumerate(order)]


def flip_box(box):
    x1, y1, x2, y2 = box
    return [1000 - x2, y1, 1000 - x1, y2]


def target_text(r, box):
    return json.dumps({"collision_visible": r["visible"], "collision_type": r["type"], "bbox_2d": box})


def example(processor, r, flip=False, with_target=True):
    """(inputs, labels or None, target) for one window; frames from the cache, flipped if asked."""
    fi, frames = bq.read_npz(REPO_ROOT / r["frames_path"])
    if fi != r["idx"]:
        vz.fail(f"{r['frames_path']}: cached frame indices differ from the manifest")
    box = r["box"]
    if flip:
        frames, box = np.ascontiguousarray(frames[:, :, ::-1]), flip_box(box)
    target = target_text(r, box)
    clip = {"no_frames": r["no_frames"], "duration": float(r["duration"])}
    inputs, labels, _ = smoke.build_inputs(processor, frames, fi, float(r["fps"]), clip,
                                           target if with_target else None, prompt_text_in=PROMPT)
    if with_target:
        ids = inputs["input_ids"][0]
        if processor.tokenizer.decode(ids[-1:]) != "\n" or processor.tokenizer.decode(ids[-2:-1]) != "<|im_end|>":
            vz.fail("the answer does not end with <|im_end|> and a newline")
        labels[0, -1] = -100  # the newline after <|im_end|> is not part of the answer
    return inputs, labels, target


# ---------------------------------------------------------------- config

def git_head():
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True).stdout
        return h, bool(dirty.strip())
    except OSError:
        return "unknown", True


FILES = ["train_qlora.py", "build_qlora_data.py", "narrow_prompt_v2.py", "vlm_zero_shot.py", "gpu_check/vlm_smoke_test.py",
         "results/qlora_data/manifest.csv", "synthetic_split.csv", "scorer_config.json"]


def write_config():
    if CONFIG_TXT.exists():
        vz.fail(f"{CONFIG_TXT} exists; the decisions are written once")
    head, dirty = git_head()
    rows = load_rows()
    train, train_list, val = splits(rows)
    tc = collections.Counter(r["type"] for r in train_list)
    L = [f"QLoRA training configuration; written {now()} BEFORE any training step; the decisions below do not change "
         "after this (every GPU mode compares them with CONFIG_JSON and refuses to run if they differ).",
         f"git commit {head} (branch vlm-zero-shot); working tree {'has uncommitted changes: the files below are hashed' if dirty else 'clean'}",
         *[f"sha256 {f}: {vz.sha256(REPO_ROOT / f)}" for f in FILES],
         f"data: {len(train)} training windows (folds {', '.join(TRAIN_FOLDS)}, windows A and B) -> {len(train_list)} per "
         f"epoch after single oversampling; by type {dict(sorted(tc.items()))}; validation {len(val)} windows (fold "
         f"{VAL_FOLD}, window A)",
         f"steps: {len(train_list)} per epoch, {CFG['epochs'] * len(train_list)} in all, warmup "
         f"{round(CFG['warmup_frac'] * CFG['epochs'] * len(train_list))}",
         "", "PROMPT (full text between the lines):", "-" * 40, PROMPT, "-" * 40, ""]
    L += [f"{k}: {v}" for k, v in CFG.items()]
    L += ["", "CONFIG_JSON " + json.dumps(CFG)]
    OUT.mkdir(parents=True, exist_ok=True)
    CONFIG_TXT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def check_config():
    if not CONFIG_TXT.exists():
        vz.fail("write CONFIG.txt first (--write-config)")
    line = next(x for x in CONFIG_TXT.read_text(encoding="utf-8").splitlines() if x.startswith("CONFIG_JSON "))
    if json.loads(line[len("CONFIG_JSON "):]) != json.loads(json.dumps(CFG)):
        vz.fail("the decisions in the code differ from CONFIG.txt")


# ---------------------------------------------------------------- unit checks (CPU)

def checks():
    from PIL import Image, ImageDraw
    from transformers import AutoProcessor

    out = OUT / "checks"
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    train, train_list, val = splits(rows)
    L = [f"Unit checks (CPU, no model); {now()}", ""]
    # 1. flip
    L.append("1. FLIP (3 training windows B, frame nearest t*): box drawn before (left) and after the flip (right)")
    picks = [next(r for r in train if r["window"] == "B" and r["type"] == t) for t in ("head-on", "sideswipe", "t-bone")]
    tiles, ok_all = [], True
    for r in picks:
        fi, fr = bq.read_npz(REPO_ROOT / r["frames_path"])
        k = int(np.argmin(np.abs(np.array(fi) / float(r["fps"]) - float(r["t_star"]))))
        flipped = np.ascontiguousarray(fr[:, :, ::-1])
        fb = flip_box(r["box"])
        px = lambda b: [round(b[0] * 448 / 1000), round(b[1] * 256 / 1000), round(b[2] * 448 / 1000), round(b[3] * 256 / 1000)]
        a, b = px(r["box"]), px(fb)
        crop_a = fr[k, a[1]:a[3], a[0]:a[2]]
        crop_b = flipped[k, b[1]:b[3], b[0]:b[2]]
        same = crop_a.shape == crop_b.shape and np.array_equal(crop_a[:, ::-1], crop_b)
        twice = flip_box(fb) == r["box"]
        ok_all &= same and twice
        for img_arr, bb, label in ((fr[k], r["box"], "original"), (flipped[k], fb, "flipped")):
            img = Image.fromarray(img_arr).convert("RGB")
            d = ImageDraw.Draw(img)
            p = px(bb)
            d.rectangle(p, outline=(255, 0, 0), width=2)
            d.rectangle([0, 0, 448, 14], fill=(0, 0, 0))
            d.text((3, 1), f"{r['stem']} {label} box {bb}", fill=(255, 255, 255))
            tiles.append(np.asarray(img))
        L.append(f"  {r['stem']} ({r['type']}): box {r['box']} -> flipped {fb}; box pixels of the flipped frame = "
                 f"mirrored box pixels of the original: {same}; flip twice = original: {twice}")
    vz.save_grid(np.stack(tiles), out / "flip_check.png", cols=2)
    L.append(f"  grid: checks/flip_check.png; all as expected: {ok_all}")
    # 2. targets parse
    L.append("\n2. TARGETS PARSE WITH narrow_prompt_v2.parse_v2 (all training windows, unflipped and flipped; all "
             "validation windows)")
    bad, n = [], 0
    for r in train + val:
        for flip in (False, True):
            box = flip_box(r["box"]) if flip else r["box"]
            t = target_text(r, box)
            p = n2.parse_v2(t)
            n += 1
            ok = (p["valid"] and p["collision_type"] == r["type"] and p["collision_visible"] == r["visible"]
                  and abs(p["x"] - (box[0] + box[2]) / 2000) < 1e-12 and abs(p["y"] - (box[1] + box[3]) / 2000) < 1e-12
                  and not any(p[f] for f in ("swapped_x", "swapped_y", "bbox_digit_string", "type_normalised")))
            if not ok:
                bad.append((r["rgb_path"], flip, t))
    L.append(f"  {n} targets: {n - len(bad)} valid with the right type, visibility, point and no flag; problems: "
             f"{bad[:3] or 'none'}")
    err = [max(abs((r["box"][0] + r["box"][2]) / 2000 - float(r["center_x"])),
               abs((r["box"][1] + r["box"][3]) / 2000 - float(r["center_y"]))) for r in val]
    L.append(f"  integer-box centre vs the label point (validation windows): max |difference| {max(err):.4f}")
    L.append(f"  example target: {target_text(picks[0], picks[0]['box'])}")
    # 3. loss mask
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    proc = AutoProcessor.from_pretrained(smoke.MODELS[vz.MODEL_KEY])
    L.append("\n3. LOSS MASK (3 examples, one flipped)")
    for r, flip in ((picks[0], False), (picks[1], True), (val[0], False)):
        inputs, labels, target = example(proc, r, flip)
        ids = inputs["input_ids"][0]
        keep = labels[0] != -100
        ans = proc.tokenizer.decode(ids[keep])
        vis = ids == proc.video_token_id
        ok = ans == target + "<|im_end|>" and not bool((keep & vis).any()) and bool(keep[-1] == False)  # noqa: E712
        first = int(keep.nonzero()[0])
        L.append(f"  {r['stem']} window {r['window']} flip {flip}: {int(ids.numel())} tokens, {int(vis.sum())} visual "
                 f"(grid_thw {inputs['video_grid_thw'][0].tolist()}), {int(keep.sum())} in the loss, from position "
                 f"{first}; answer tokens decode to {ans!r}; only the answer: {ok}")
        L.append(f"    token before the answer: {proc.tokenizer.decode(ids[first - 3:first])!r}")
    L.append(f"\nPROMPT ({len(PROMPT)} characters):\n{'-' * 40}\n{PROMPT}\n{'-' * 40}")
    text = "\n".join(L) + "\n"
    (out / "checks.txt").write_text(text, encoding="utf-8")
    print(text)


# ---------------------------------------------------------------- model

def load(lora):
    import torch
    from transformers import AutoProcessor

    if not torch.cuda.is_available():
        vz.fail("CUDA is not available")
    torch.manual_seed(SEED)
    proc = AutoProcessor.from_pretrained(smoke.MODELS[vz.MODEL_KEY])
    model = smoke.load_model({"model": vz.MODEL_KEY, "precision": vz.PRECISION})
    if lora:
        from peft import LoraConfig, get_peft_model

        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        c = CFG["lora"]
        model = get_peft_model(model, LoraConfig(r=c["r"], lora_alpha=c["alpha"], lora_dropout=c["dropout"],
                                                 bias=c["bias"], target_modules=c["target_modules"]))
    return proc, model


def to_device(inputs, device):
    import torch

    x = {k: v.to(device) for k, v in inputs.items()}
    x["pixel_values_videos"] = x["pixel_values_videos"].to(torch.bfloat16)
    return x


# ---------------------------------------------------------------- validation

VAL_COLUMNS = ["rgb_path", "stem", "type", "day_night", "t_star_inside", "center_x", "center_y", "valid", "reason",
               "pred_type", "x", "y", "bbox", "collision_visible", "new_tokens", "gpu_s"]


def validate(model, proc, val, out_csv, raw_jsonl, log):
    import torch

    done = {r["rgb_path"] for r in vz.read_csv(out_csv)} if out_csv.exists() else set()
    model.eval()
    device = next(model.parameters()).device
    new = not out_csv.exists()
    with open(out_csv, "a", newline="", encoding="utf-8") as f, open(raw_jsonl, "a", encoding="utf-8") as fj:
        w = csv.DictWriter(f, fieldnames=VAL_COLUMNS)
        if new:
            w.writeheader()
        for i, r in enumerate(val, 1):
            if r["rgb_path"] in done:
                continue
            inputs, _, _ = example(proc, r, flip=False, with_target=False)
            n_in = inputs["input_ids"].shape[1]
            x = to_device(inputs, device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(**x, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True,
                                     temperature=None, top_p=None, top_k=None)
            torch.cuda.synchronize()
            s = time.perf_counter() - t0
            text = proc.batch_decode(out[0, n_in:][None], skip_special_tokens=True)[0]
            p = n2.parse_v2(text)
            w.writerow({"rgb_path": r["rgb_path"], "stem": r["stem"], "type": r["type"], "day_night": r["day_night"],
                        "t_star_inside": r["visible"], "center_x": r["center_x"], "center_y": r["center_y"],
                        "valid": p["valid"], "reason": p["reason"], "pred_type": p["collision_type"], "x": p["x"],
                        "y": p["y"], "bbox": json.dumps(p["bbox_2d"]), "collision_visible": json.dumps(p["collision_visible"]),
                        "new_tokens": int(out.shape[1] - n_in), "gpu_s": s})
            fj.write(json.dumps({"clip": r["rgb_path"], "output": text}) + "\n")
            f.flush()
            fj.flush()
            if i % 100 == 0:
                log(f"validation {i}/{len(val)}")
    model.train()


def metrics(rows, majority, cfg):
    sx, sy = cfg["sigma_x"] * cfg["spatial_k"], cfg["sigma_y"] * cfg["spatial_k"]

    def s_err(xs, ys, gx, gy):
        xs, ys, gx, gy = (np.asarray(a, float) for a in (xs, ys, gx, gy))
        return (np.exp(-((xs - gx) ** 2 / (2 * sx ** 2) + (ys - gy) ** 2 / (2 * sy ** 2))),
                np.hypot(xs - gx, ys - gy))

    gx = [float(r["center_x"]) for r in rows]
    gy = [float(r["center_y"]) for r in rows]
    S, E = s_err([float(r["x"]) for r in rows], [float(r["y"]) for r in rows], gx, gy)
    S0, E0 = s_err([0.5] * len(rows), [0.5] * len(rows), gx, gy)
    acc = float(np.mean([r["pred_type"] == r["type"] for r in rows]))
    vis = collections.Counter(r["collision_visible"] for r in rows)
    agree = np.mean([json.loads(r["collision_visible"]) == (r["t_star_inside"] == "True") for r in rows])
    m = {"n": len(rows), "valid": sum(r["valid"] == "True" for r in rows), "type_accuracy": acc, "S": float(S.mean()),
         "mean_point_error": float(E.mean()), "median_point_error": float(np.median(E)),
         "score_mean_acc_S": (acc + float(S.mean())) / 2,
         "majority_type": majority, "majority_accuracy": float(np.mean([r["type"] == majority for r in rows])),
         "centre_S": float(S0.mean()), "centre_mean_point_error": float(E0.mean()),
         "collision_visible_counts": dict(vis), "collision_visible_agrees_with_t_star_inside": float(agree),
         "gpu_s_per_window": float(np.mean([float(r["gpu_s"]) for r in rows]))}
    conf = {t: collections.Counter(r["pred_type"] for r in rows if r["type"] == t) for t in vz.TYPES}
    m["confusion"] = {t: {p: conf[t][p] for p in vz.TYPES} for t in vz.TYPES}
    m["per_day_night"] = {}
    for dn in ("day", "night"):
        rr = [r for r in rows if r["day_night"] == dn]
        s, _ = s_err([float(r["x"]) for r in rr], [float(r["y"]) for r in rr], [float(r["center_x"]) for r in rr],
                     [float(r["center_y"]) for r in rr])
        m["per_day_night"][dn] = {"n": len(rr), "type_accuracy": float(np.mean([r["pred_type"] == r["type"] for r in rr])),
                                  "S": float(s.mean())}
    return m


def metrics_text(title, m):
    L = [title,
         f"  windows {m['n']}, valid answers {m['valid']}, fallbacks {m['n'] - m['valid']}",
         f"  type accuracy {m['type_accuracy']:.4f}; S {m['S']:.4f}; mean point error {m['mean_point_error']:.4f} "
         f"(median {m['median_point_error']:.4f}); mean of accuracy and S {m['score_mean_acc_S']:.4f}",
         f"  baselines on the same windows: majority type ({m['majority_type']}) accuracy {m['majority_accuracy']:.4f}; "
         f"image centre S {m['centre_S']:.4f}, mean point error {m['centre_mean_point_error']:.4f}",
         f"  collision_visible answers {m['collision_visible_counts']}; agrees with t* inside the window "
         f"{m['collision_visible_agrees_with_t_star_inside']:.4f} (diagnostic)",
         "  per day / night: " + "; ".join(f"{k} n {v['n']} acc {v['type_accuracy']:.4f} S {v['S']:.4f}"
                                           for k, v in m["per_day_night"].items()),
         f"  GPU s per window (generate) {m['gpu_s_per_window']:.2f}",
         "  confusion (rows true, columns predicted; fallbacks counted as rear-end):",
         "    " + f"{'':10s} " + " ".join(f"{t:>9s}" for t in vz.TYPES) + "   correct"]
    for t in vz.TYPES:
        row = m["confusion"][t]
        L.append("    " + f"{t:10s} " + " ".join(f"{row[p]:9d}" for p in vz.TYPES) + f"   {row[t]}/{sum(row.values())}")
    return "\n".join(L)


def run_validation(model, proc, val, d, tag, majority, log):
    d.mkdir(parents=True, exist_ok=True)
    csv_p, raw = d / f"val_{tag}.csv", d / f"val_{tag}_raw.jsonl"
    validate(model, proc, val, csv_p, raw, log)
    rows = vz.read_csv(csv_p)
    if len(rows) != len(val):
        vz.fail(f"{csv_p}: {len(rows)} rows, expected {len(val)}")
    m = metrics(rows, majority, json.loads(SCORER_CONFIG.read_text(encoding="utf-8")))
    (d / f"val_{tag}_metrics.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    return m


def majority_type(train):
    return collections.Counter(r["type"] for r in train if r["window"] == "A").most_common(1)[0][0]


# ---------------------------------------------------------------- baseline

def baseline():
    import torch

    check_config()
    d = OUT / "baseline"
    if (d / "report.txt").exists():
        vz.fail(f"{d / 'report.txt'} exists")
    d.mkdir(parents=True, exist_ok=True)
    on, ptxt = vz.power()
    if not on:
        vz.fail(f"not on mains: {ptxt}")
    rows = load_rows()
    train, _, val = splits(rows)
    logf = open(d / "run.log", "a", encoding="utf-8")

    def log(msg):
        line = f"{now()} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log(f"baseline (untrained model, same prompt) on {len(val)} windows; power {ptxt}")
    proc, model = load(lora=False)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    m = run_validation(model, proc, val, d, "baseline", majority_type(train), log)
    m["peak_alloc_gb"] = torch.cuda.max_memory_allocated() / GB
    text = metrics_text(f"BASELINE: untrained {smoke.MODELS[vz.MODEL_KEY]} NF4 with the training prompt, fold 1 window A "
                        f"({now()}; power start {ptxt}, end {vz.power()[1]}; peak allocated {m['peak_alloc_gb']:.2f} GB)", m)
    (d / "report.txt").write_text(text + "\n", encoding="utf-8")
    log("\n" + text)


# ---------------------------------------------------------------- training

def save_ckpt(model, opt, sched, state, wdir, name, keep_opt=2):
    import torch

    d = wdir / name
    tmp = wdir / (name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(tmp)
    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "state": state,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, tmp / "trainer_state.pt")
    if d.exists():
        shutil.rmtree(d)
    os.replace(tmp, d)
    (wdir / "LATEST").write_text(name, encoding="utf-8")
    # keep the adapter of every checkpoint; optimizer states only for the newest keep_opt checkpoints
    ck = sorted((p for p in wdir.iterdir() if p.is_dir() and (p / "trainer_state.pt").exists()),
                key=lambda p: p.stat().st_mtime)
    for p in ck[:-keep_opt]:
        (p / "trainer_state.pt").unlink()


def train(smoke_run=False, max_steps=None):
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import get_cosine_schedule_with_warmup

    check_config()
    out = OUT / "smoke" if smoke_run else OUT
    wdir = WEIGHTS.with_name("qlora_train_smoke") if smoke_run else WEIGHTS
    out.mkdir(parents=True, exist_ok=True)
    wdir.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run.log", "a", encoding="utf-8")

    def log(msg):
        line = f"{now()} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    def power_ok(where):
        on, ptxt = vz.power()
        log(f"power {where}: {ptxt}")
        return on

    if not power_ok("at start"):
        log("PAUSED: on battery at start")
        sys.exit(4)
    rows = load_rows()
    train_rows, train_list, val = splits(rows)
    majority = majority_type(train_rows)
    spe = len(train_list)
    total = CFG["epochs"] * spe
    warm = round(CFG["warmup_frac"] * total)
    log(f"{'SMOKE ' if smoke_run else ''}training: {spe} examples per epoch, {total} steps, warmup {warm}; code sha256 "
        f"{vz.sha256(Path(__file__))}")
    proc, model = load(lora=True)
    device = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=CFG["lr"])
    sched = get_cosine_schedule_with_warmup(opt, warm, total)
    state = {"step": 0, "val_done": []}
    latest = wdir / "LATEST"
    if latest.exists():
        d = wdir / latest.read_text(encoding="utf-8").strip()
        set_peft_model_state_dict(model, load_file(str(d / "adapter_model.safetensors")))
        ts = torch.load(d / "trainer_state.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(ts["opt"])
        sched.load_state_dict(ts["sched"])
        state = ts["state"]
        torch.set_rng_state(ts["torch_rng"])
        torch.cuda.set_rng_state_all(ts["cuda_rng"])
        log(f"RESUMED from {d.name}: step {state['step']}, validations done {state['val_done']}")
    log(f"trainable parameters {sum(p.numel() for p in params):,}; LoRA modules "
        f"{sum(1 for n, _ in model.named_modules() if n.endswith('.lora_A'))}; dtype {params[0].dtype}")
    stop_at = min(total, max_steps) if max_steps else total
    steps_csv = out / "steps.csv"
    new = not steps_csv.exists()
    sf = open(steps_csv, "a", newline="", encoding="utf-8")
    sw = csv.writer(sf)
    if new:
        sw.writerow(["step", "epoch", "rgb_path", "window", "flip", "loss", "lr", "data_s", "gpu_s", "peak_alloc_gb"])
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    recent, t_last = [], time.perf_counter()

    def checkpoint(name):
        save_ckpt(model, opt, sched, state, wdir, name)
        log(f"checkpoint {name} saved (step {state['step']})")
        if not power_ok(f"at checkpoint {name}"):
            log(f"PAUSED (resumable): on battery after checkpoint {name}")
            sys.exit(4)

    def epoch_end(e):
        if e in state["val_done"] or smoke_run:
            return
        m = run_validation(model, proc, val, out, f"epoch{e}", majority, log)
        state["val_done"].append(e)
        state.setdefault("val", {})[str(e)] = {k: m[k] for k in ("type_accuracy", "S", "score_mean_acc_S")}
        save_ckpt(model, opt, sched, state, wdir, f"epoch{e}")
        text = metrics_text(f"EPOCH {e} VALIDATION (fold 1 window A; step {state['step']}; {now()})", m)
        (out / f"epoch{e}_report.txt").write_text(text + "\n", encoding="utf-8")
        log("\n" + text)

    # a run stopped after an epoch's checkpoint but before (the end of) its validation: finish it with these weights
    if not smoke_run and state["step"] and state["step"] % spe == 0:
        epoch_end(state["step"] // spe)
    plans = {}
    while state["step"] < stop_at:
        e, k = divmod(state["step"], spe)
        if e + 1 not in plans:
            plans = {e + 1: epoch_plan(train_list, e + 1)}
        r, flip = plans[e + 1][k]
        t0 = time.perf_counter()
        inputs, labels, _ = example(proc, r, flip)
        x = to_device(inputs, device)
        y = labels.to(device)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        loss = model(**x, labels=y).loss
        if not torch.isfinite(loss):
            log(f"STOPPED: loss not finite at step {state['step'] + 1} ({r['rgb_path']}, window {r['window']})")
            sys.exit(5)
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        state["step"] += 1
        lv = float(loss.detach())
        recent.append(lv)
        pk = torch.cuda.max_memory_allocated(device) / GB
        sw.writerow([state["step"], e + 1, r["rgb_path"], r["window"], flip, f"{lv:.5f}", f"{sched.get_last_lr()[0]:.3e}",
                     f"{t1 - t0:.3f}", f"{t2 - t1:.3f}", f"{pk:.3f}"])
        del x, y, loss
        if state["step"] % 25 == 0 or state["step"] == stop_at:
            sf.flush()
            dt = (time.perf_counter() - t_last) / len(recent)
            log(f"step {state['step']}/{total} (epoch {e + 1}) loss mean {statistics.mean(recent):.4f} lr "
                f"{sched.get_last_lr()[0]:.2e} {dt:.2f} s/step peak alloc {pk:.2f} GB reserved "
                f"{torch.cuda.max_memory_reserved(device) / GB:.2f} GB; ETA training {(total - state['step']) * dt / 3600:.1f} h")
            recent, t_last = [], time.perf_counter()
        if state["step"] % CFG["save_every_steps"] == 0 and state["step"] % spe != 0:
            checkpoint(f"step{state['step']:05d}")
        if state["step"] % spe == 0:
            checkpoint(f"epoch{state['step'] // spe}")  # saved before the validation, saved again after it
            epoch_end(state["step"] // spe)
    sf.close()
    if smoke_run:
        checkpoint(f"step{state['step']:05d}")
        log(f"SMOKE DONE at step {state['step']}; peak allocated {torch.cuda.max_memory_allocated(device) / GB:.2f} GB, "
            f"reserved {torch.cuda.max_memory_reserved(device) / GB:.2f} GB")
        return
    v = state["val"]
    best = max(v, key=lambda e: (v[e]["score_mean_acc_S"], -int(e)))
    lines = [f"CHECKPOINT CHOICE ({now()}): the epoch with the higher mean of type accuracy and S on fold 1 window A"]
    lines += [f"  epoch {e}: accuracy {v[e]['type_accuracy']:.4f}, S {v[e]['S']:.4f}, mean {v[e]['score_mean_acc_S']:.4f}"
              for e in sorted(v)]
    lines.append(f"  CHOSEN: epoch {best} -> {wdir / f'epoch{best}'} (a tie keeps the earlier epoch)")
    (out / "checkpoint_choice.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("\n".join(lines))
    power_ok("at the end")
    log("FINISHED")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    for m in ("write-config", "checks", "baseline", "smoke", "train"):
        g.add_argument(f"--{m}", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if args.write_config:
        write_config()
    elif args.checks:
        checks()
    elif args.baseline:
        baseline()
    elif args.smoke:
        if not args.max_steps:
            vz.fail("--smoke needs --max-steps")
        train(smoke_run=True, max_steps=args.max_steps)
    else:
        train()


if __name__ == "__main__":
    main()
