"""QLoRA memory and speed test: Qwen3-VL-4B-Instruct (NF4) on the local GPU. Synthetic clips only; no accuracy.

Measures whether QLoRA fine-tuning of the narrow call fits in GPU memory and how long a step takes. It does not
evaluate anything: the loss is logged only to show that the steps run.

Setup (fixed before the run):
  model   Qwen/Qwen3-VL-4B-Instruct, 4-bit NF4 (double quant, bf16 compute), loaded as in vlm_smoke_test.load_model
  LoRA    on the language model's attention (q/k/v/o_proj) and MLP (gate/up/down_proj) projections, alpha = 2 x rank,
          dropout 0.05, no bias; vision tower and all other weights frozen
  train   gradient checkpointing on (non-reentrant), use_cache off, batch 1, AdamW lr 1e-4, seed 0, N_STEPS steps
          cycling over N_CLIPS clips (clip k at steps k, k + 4, ...)
  input   the narrow-call layout: n frames evenly over t* +- 2 s (shifted inside the clip), full frame resized to
          448 x 256 pixel area, native Qwen pairing (2 frames per temporal token group); prompt = narrow v1 prompt
          (16 frames) or its 8-frame text (narrow v2 prompt) for 8 frames; target = the JSON answer
          {"collision_type", "x", "y"} from the synthetic labels (x, y rounded to 3 decimals); loss on the target only
  clips   the first development clip (vlm_dev_clips.csv order) of each of head-on, rear-end, sideswipe, t-bone
  attempts 1: 16 frames, rank 16; if it does not fit: 2: 8 frames, rank 16; if not: 3: 8 frames, rank 8
  fits    = all N_STEPS steps run without CUDA out-of-memory and the peak reserved memory is at most 0.97 x the GPU's
          total memory (the stage-test criterion; on Windows the driver can spill into shared system memory instead
          of failing, which this catches)
  timing  seconds per step = synchronised time of host-to-GPU copy + forward + backward + optimizer step (frames are
          decoded and processed on the CPU before the first step); step 1 is reported apart (warm-up), the mean and
          median over steps 2..N_STEPS are the step time
  power   mains (Win32_Battery BatteryStatus 2) checked at the start and before each attempt; on battery: stop

Each attempt runs in its own child process (clean GPU memory). Outputs in results/qlora_memory_test/.
Usage (from cctv-accident-ITS-system/):  .venv-vlm\\Scripts\\python.exe -u vlm_qlora_memory_test.py

--layout v2 (added 2026-11-21; the default layout v1 above is unchanged): the narrow-v2 input of narrow_prompt_v2.py
  instead: the n frames over t* +- 2 s each duplicated (one frame per temporal token group, its own timestamp; 16
  frames = 1,792 visual tokens), prompt PROMPT_V2 (for 8 frames the same text with "These 16 frames" -> "These 8
  frames"), target = a JSON answer in the v2 format from the synthetic labels: {"vehicles": placeholder text,
  "motion": the motion option of the true type, "collision_visible": true, "collision_type": the true type,
  "bbox_2d": the true box x1, y1, x2, y2 in 0-1000 (rounded)}; the vehicles text is a fixed placeholder (memory test
  only, no description labels exist). Attempts: 1: 16 frames, rank 16; if it does not fit: 2: 8 frames, rank 16.
  Outputs in results/qlora_memory_test_v2/.
    .venv-vlm\\Scripts\\python.exe -u vlm_qlora_memory_test.py --layout v2
"""

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
import vlm_zero_shot as vz  # noqa: E402  (frozen protocol code: imported, not changed)

smoke = vz.smoke
OUT_DIR = REPO_ROOT / "results" / "qlora_memory_test"
CLIP_TYPES = ("head-on", "rear-end", "sideswipe", "t-bone")
N_STEPS = 20
HALF_S = 2.0
LORA_TARGET = (r".*language_model.*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
               r"|mlp\.(gate_proj|up_proj|down_proj))")
LORA_DROPOUT = 0.05
LR = 1e-4
SEED = 0
ATTEMPTS = {1: dict(frames=16, rank=16), 2: dict(frames=8, rank=16), 3: dict(frames=8, rank=8)}
LAYOUTS = {"v1": (OUT_DIR, ATTEMPTS),
           "v2": (REPO_ROOT / "results" / "qlora_memory_test_v2", {1: dict(frames=16, rank=16), 2: dict(frames=8, rank=16)})}
MOTION_V2 = {"head-on": "opposite", "rear-end": "same direction", "sideswipe": "parallel", "t-bone": "crossing",
             "single": "one vehicle"}


def v2_target(c, box):
    """A narrow-v2 JSON answer from the synthetic labels (vehicles: fixed placeholder text)."""
    veh = ("one vehicle near the marked point" if c["type"] == "single"
           else "two vehicles meeting near the marked point")
    return json.dumps({"vehicles": f"{veh} (x {c['center_x']:.2f}, y {c['center_y']:.2f})",
                       "motion": MOTION_V2[c["type"]], "collision_visible": True, "collision_type": c["type"],
                       "bbox_2d": [round(1000 * float(box[k])) for k in ("x1", "y1", "x2", "y2")]})
CHILD_TIMEOUT_S = 3600
GB = 2 ** 30
RESULT_TAG = "RESULT_JSON "


def pick_clips():
    clips = vz.load_clips(vz.DEV_CSV)
    out = [next(c for c in clips if c["type"] == t) for t in CLIP_TYPES]
    if not all(c["rgb_path"].startswith("synthetic_videos/") for c in out):
        vz.fail("not a synthetic clip")
    return out


# ---------------------------------------------------------------- child: one attempt

def child(attempt, layout="v1"):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor

    a = LAYOUTS[layout][1][attempt]
    if layout == "v1":
        prompt = vz.NARROW_PROMPT if a["frames"] == 16 else vz.NARROW2_PROMPT
    else:
        import narrow_prompt_v2 as n2
        prompt = n2.PROMPT_V2.replace("These 16 frames", f"These {a['frames']} frames")
        meta = vz.synthetic_meta()
    if f"These {a['frames']} frames" not in prompt:
        vz.fail("prompt frame count does not match")
    res = {"attempt": attempt, "layout": layout, **a, "lora_alpha": 2 * a["rank"], "lora_target": LORA_TARGET,
           "steps": N_STEPS, "status": None, "oom": False, "prompt_sha256": vz.text_sha(prompt)}
    torch.manual_seed(SEED)
    processor = AutoProcessor.from_pretrained(smoke.MODELS[vz.MODEL_KEY])
    batches, res["clips"] = [], []
    for c in pick_clips():
        idx = vz.to_indices(c, vz.window_times(c, c["t_star"], HALF_S, a["frames"]))
        frames = vz.stack(vz.decode(c, idx), idx)
        if layout == "v1":
            target = json.dumps({"collision_type": c["type"], "x": round(c["center_x"], 3),
                                 "y": round(c["center_y"], 3)})
        else:  # each frame duplicated: one frame per temporal token group, with its own timestamp
            target = v2_target(c, meta[c["rgb_path"]])
            frames, idx = np.repeat(frames, 2, axis=0), [i for i in idx for _ in range(2)]
        inputs, labels, _ = smoke.build_inputs(processor, frames, idx, c["fps"], c, target, prompt_text_in=prompt)
        ids = inputs["input_ids"][0]
        res["clips"].append({"rgb_path": c["rgb_path"], "frames": idx, "t_first_last": [idx[0] / c["fps"], idx[-1] / c["fps"]],
                             "frame_size_in": [int(frames.shape[2]), int(frames.shape[1])],
                             "video_grid_thw": inputs["video_grid_thw"][0].tolist(),
                             "visual_tokens": int((ids == processor.video_token_id).sum()),
                             "input_tokens": int(ids.numel()), "target_tokens": int((labels != -100).sum()),
                             "target": target})
        batches.append((dict(inputs), labels))
    smoke.stamp(f"attempt {attempt}: inputs built for {len(batches)} clips")

    device = torch.device("cuda", 0)
    total = torch.cuda.get_device_properties(device).total_memory
    res["gpu"] = torch.cuda.get_device_name(0)
    nvml = smoke.NvmlPeak().start()
    model = opt = None
    steps = []
    try:
        t0 = time.perf_counter()
        model = smoke.load_model({"model": vz.MODEL_KEY, "precision": vz.PRECISION})
        torch.cuda.synchronize()
        res["load_s"] = time.perf_counter() - t0
        res["alloc_after_load_gb"] = torch.cuda.memory_allocated(device) / GB
        res["peak_alloc_during_load_gb"] = torch.cuda.max_memory_allocated(device) / GB
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=a["rank"], lora_alpha=2 * a["rank"], lora_dropout=LORA_DROPOUT,
                                                 bias="none", target_modules=LORA_TARGET))
        params = [p for p in model.parameters() if p.requires_grad]
        res["trainable_params"] = sum(p.numel() for p in params)
        res["total_params_stored"] = sum(p.numel() for p in model.parameters())
        res["lora_modules"] = sum(1 for n, _ in model.named_modules() if n.endswith(".lora_A"))
        opt = torch.optim.AdamW(params, lr=LR)
        model.train()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)  # training peak (weights stay allocated, so they are included)
        for step in range(1, N_STEPS + 1):
            k = (step - 1) % len(batches)
            inputs, labels = batches[k]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            x = {n: v.to(device) for n, v in inputs.items()}
            x["pixel_values_videos"] = x["pixel_values_videos"].to(torch.bfloat16)
            loss = model(**x, labels=labels.to(device)).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            s = time.perf_counter() - t0
            steps.append({"step": step, "clip": k, "s": s, "loss": float(loss.detach()),
                          "peak_alloc_gb_so_far": torch.cuda.max_memory_allocated(device) / GB,
                          "peak_reserved_gb_so_far": torch.cuda.max_memory_reserved(device) / GB})
            smoke.stamp(f"attempt {attempt} step {step}/{N_STEPS} clip {k} {s:.2f} s loss {float(loss):.4f} "
                        f"peak alloc {steps[-1]['peak_alloc_gb_so_far']:.2f} GB")
            del x, loss
        res["status"] = "OK"
    except torch.cuda.OutOfMemoryError as e:
        res.update(status="OOM", oom=True, error=str(e).splitlines()[0])
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        res.update(status="OOM", oom=True, error=str(e).splitlines()[0])
    finally:
        torch.cuda.synchronize()
        res["peak_alloc_train_gb"] = torch.cuda.max_memory_allocated(device) / GB
        res["peak_reserved_train_gb"] = torch.cuda.max_memory_reserved(device) / GB
        nvml.stop()
        res["nvml_baseline_gb"] = nvml.baseline / GB
        res["nvml_peak_gb"] = nvml.peak / GB
        res["total_gb"] = total / GB
        del model, opt
        gc.collect()
        torch.cuda.empty_cache()
    res["steps_done"] = len(steps)
    res["step_log"] = steps
    if steps:
        res["step1_s"] = steps[0]["s"]
    rest = [r["s"] for r in steps[1:]]
    if rest:
        res["step_s_mean_2_on"] = statistics.mean(rest)
        res["step_s_median_2_on"] = statistics.median(rest)
    res["fits"] = (res["status"] == "OK" and len(steps) == N_STEPS
                   and res["peak_reserved_train_gb"] <= smoke.OVER_VRAM_FRAC * res["total_gb"])
    print(RESULT_TAG + json.dumps(res), flush=True)


# ---------------------------------------------------------------- parent

def run_attempt(attempt, log, layout="v1"):
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--child", str(attempt),
                             "--layout", layout],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=str(REPO_ROOT),
                            text=True, encoding="utf-8", errors="replace")
    timed_out = threading.Event()

    def kill():
        timed_out.set()
        smoke.kill_tree(proc)

    timer = threading.Timer(CHILD_TIMEOUT_S, kill)
    timer.daemon = True
    timer.start()
    result = None
    try:
        for line in proc.stdout:
            if line.startswith(RESULT_TAG):
                result = json.loads(line[len(RESULT_TAG):])
                line = RESULT_TAG + "(saved to attempts.jsonl)\n"
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        proc.wait()
    except BaseException:
        smoke.kill_tree(proc)
        raise
    finally:
        timer.cancel()
    if result is None:
        result = {"attempt": attempt, **LAYOUTS[layout][1][attempt], "fits": False,
                  "status": "TIMEOUT" if timed_out.is_set() else f"CHILD FAILED (exit {proc.returncode})"}
    return result


def fmt(v, f="{:.2f}"):
    return "-" if v is None else f.format(v)


def report(results, power_lines, layout="v1"):
    L = ["QLoRA memory and speed test, Qwen/Qwen3-VL-4B-Instruct NF4; synthetic clips only; no accuracy measured."
         + ("" if layout == "v1" else f" Input layout {layout} (narrow v2: each frame duplicated, PROMPT_V2, v2 JSON target)."),
         "Setup: see the docstring of vlm_qlora_memory_test.py (copied in command.txt).", *power_lines, ""]
    for r in results:
        L.append(f"attempt {r['attempt']}: {r['frames']} frames, LoRA rank {r['rank']} (alpha {2 * r['rank']}) -> "
                 f"status {r['status']}, fits {r['fits']}")
        if "trainable_params" in r:
            L.append(f"  trainable parameters {r['trainable_params']:,} in {r['lora_modules']} LoRA modules "
                     f"(parameters stored, NF4 packed counted as stored: {r['total_params_stored']:,})")
        if "clips" in r:
            c0 = r["clips"][0]
            if layout != "v1":
                L.append(f"  target example ({Path(c0['rgb_path']).stem}): {c0['target']}")
            L.append(f"  input per clip: grid_thw {c0['video_grid_thw']}, visual tokens {c0['visual_tokens']}, "
                     f"input tokens {[c['input_tokens'] for c in r['clips']]}, target tokens "
                     f"{[c['target_tokens'] for c in r['clips']]}, frame size in {c0['frame_size_in']}")
        if "total_gb" in r:
            L.append(f"  GPU {r.get('gpu')} total {r['total_gb']:.2f} GB; after load allocated "
                     f"{fmt(r.get('alloc_after_load_gb'))} GB (peak during load {fmt(r.get('peak_alloc_during_load_gb'))})")
            L.append(f"  peak GPU memory during training: allocated {r['peak_alloc_train_gb']:.2f} GB, reserved "
                     f"{r['peak_reserved_train_gb']:.2f} GB; device-wide (NVML) peak {r['nvml_peak_gb']:.2f} GB "
                     f"(baseline before load {r['nvml_baseline_gb']:.2f} GB)")
        if r.get("steps_done"):
            L.append(f"  steps done {r['steps_done']}/{N_STEPS}; step 1 (warm-up) {r['step1_s']:.2f} s; steps 2-"
                     f"{r['steps_done']}: mean {fmt(r.get('step_s_mean_2_on'))} s, median "
                     f"{fmt(r.get('step_s_median_2_on'))} s per step; load {fmt(r.get('load_s'), '{:.1f}')} s")
            L.append("  loss per step (logged only to show the steps run; not an accuracy): "
                     + " ".join(f"{s['loss']:.3f}" for s in r["step_log"]))
        if r.get("error"):
            L.append(f"  error: {r['error']}")
        L.append("")
    if results:
        clips = results[0].get("clips", [])
        L.append("clips: " + ", ".join(Path(c["rgb_path"]).stem for c in clips))
    return "\n".join(L) + "\n"


def parent(layout="v1"):
    out_dir, attempts = LAYOUTS[layout]
    if (out_dir / "report.txt").exists():
        vz.fail(f"{out_dir / 'report.txt'} exists; refusing to overwrite a finished run")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "command.txt").write_text(
        f"{vz.now()}\n.venv-vlm\\Scripts\\python.exe -u vlm_qlora_memory_test.py --layout {layout} (HF_HUB_OFFLINE=1, "
        f"TRANSFORMERS_OFFLINE=1 in the child processes)\nscript sha256 {vz.sha256(Path(__file__))}\n\n{__doc__}",
        encoding="utf-8")
    results, power_lines = [], []
    with open(out_dir / "run.log", "a", encoding="utf-8") as log, \
            open(out_dir / "attempts.jsonl", "a", encoding="utf-8") as js:
        def note(msg):
            line = f"{vz.now()} {msg}"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            power_lines.append(line)

        for attempt in attempts:
            on, txt = vz.power()
            note(f"power before attempt {attempt}: {txt}")
            if not on:
                note("STOPPED: on battery")
                break
            r = run_attempt(attempt, log, layout)
            js.write(json.dumps(r) + "\n")
            js.flush()
            results.append(r)
            if r["fits"]:
                break
        on, txt = vz.power()
        note(f"power at end: {txt}")
    text = report(results, power_lines, layout)
    (out_dir / "report.txt").write_text(text, encoding="utf-8")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--child", type=int, choices=sorted(ATTEMPTS), default=None)
    ap.add_argument("--layout", choices=sorted(LAYOUTS), default="v1")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if args.child:
        child(args.child, args.layout)
    else:
        parent(args.layout)


if __name__ == "__main__":
    main()
