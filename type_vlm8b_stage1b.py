"""Type model, option 1, STAGE 1b: corrected model loading (results/type_vlm8b/AMENDMENT_1.txt), then the memory and
speed test on the same 5 clips as stage 1. SYNTHETIC ONLY. NO accuracy and NO comparison with the true types.

Reuses type_vlm8b.py (imported, not changed). Changes nothing in results/type_vlm8b/ except the new files below.
Run with the .venv-vlm interpreter.

  --speed   load with the corrected config, check the 4-bit module counts, test 5 clips (default_rng(1)); 448 x 448,
            or the pre-registered 336 x 336 fallback if 448 runs out of memory
  --report  stage1b_report.txt

Why the explicit BitsAndBytesConfig goes in through `config=` (transformers 5.22.1, read before running):
  - quantizers_utils.should_convert_module matches a skip name only from the start of the full module path or as its
    suffix, so the checkpoint's bare 'visual' does not match 'model.visual.blocks.0.attn.qkv' (stage 1 crash).
  - quantizers/auto.get_hf_quantizer: for a pre-quantised checkpoint, the quantization_config of the model config is
    used, and merge_quantization_configs takes only "loading attributes" from a quantization_config= argument, and
    only for config types in LOADING_ATTRIBUTES_CONFIG_TYPES, which does not include BitsAndBytesConfig. A
    BitsAndBytesConfig passed only as quantization_config= would therefore be ignored. So the explicit config is set
    as the model config's quantization_config and that config is passed to from_pretrained.
"""
import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import type_vlm8b as T

OUT = T.OUT
AMEND = OUT / "AMENDMENT_1.txt"
SPEED_JSON, ANSWERS, REPORT = OUT / "stage1b_speed.json", OUT / "stage1b_answers.jsonl", OUT / "stage1b_report.txt"
# full module paths in this transformers version (listed from the model built on the meta device, no weights)
FULL_PATHS = ["model.visual", "model.visual.merger", "model.visual.deepstack_merger_list", "lm_head"]
SIZES = [448, 336]  # 448 as planned; 336 only if 448 runs out of memory (AMENDMENT_1)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(p, man, meta, e1, size):
    """As type_vlm8b.prepare, with the output size as an argument (same crop)."""
    import vlm_zero_shot as vz
    from PIL import Image
    W, H = int(meta["width"]), int(meta["height"])
    idx = json.loads(man["frame_indices"])
    if len(idx) != 16:
        T.fail(f"{p}: {len(idx)} window-A frames")
    clip = {"video": str(T.DATASET / p), "rgb_path": p, "no_frames": int(meta["no_frames"]),
            "duration": float(meta["duration"])}
    got = vz.decode(clip, idx, budget=10 ** 12)
    left, top, side, info = T.crop_square(e1, W, H)
    out = []
    for i in idx:
        img = got[i]
        if img.size != (W, H):
            T.fail(f"{p}: decoded {img.size}, metadata {W}x{H}")
        out.append(np.asarray(img.crop((left, top, left + side, top + side)).resize((size, size), Image.BICUBIC)))
    info.update(left=left, top=top)
    return np.stack(out), idx, clip["no_frames"] / clip["duration"], clip, info


def speed():
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    import vlm_smoke_test as smoke
    import vlm_zero_shot as vz
    from transformers import AutoConfig, AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    T.check_plan()
    if not AMEND.is_file():
        T.fail("AMENDMENT_1.txt missing")
    T.refuse(SPEED_JSON, ANSWERS)
    on, ptxt = vz.power()
    if not on:
        T.fail("on battery")
    clips = T.load_clips()
    pick = sorted(np.random.default_rng(1).choice(len(clips), size=5, replace=False).tolist())
    res = {"amendment_sha256": sha(AMEND), "clips": [Path(clips[j][0]).stem for j in pick], "power_start": ptxt,
           "gpu": torch.cuda.get_device_name(0), "gpu_total_gb": torch.cuda.get_device_properties(0).total_memory / 2 ** 30,
           "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"), "runs": []}

    def save():
        SPEED_JSON.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")

    # ---- the explicit config: a copy of the checkpoint's own quantization_config, skip list extended
    cfg = AutoConfig.from_pretrained(T.REPO_ID, revision=T.REV)
    saved = cfg.quantization_config
    saved = copy.deepcopy(saved if isinstance(saved, dict) else saved.to_dict())
    qd = copy.deepcopy(saved)
    qd["llm_int8_skip_modules"] = list(saved["llm_int8_skip_modules"]) + FULL_PATHS
    bnb = BitsAndBytesConfig.from_dict(qd)
    got = bnb.to_dict()
    diff = sorted(k for k in saved if k != "llm_int8_skip_modules" and got.get(k) != saved[k])
    res["quantization_config_saved"] = saved
    res["quantization_config_explicit"] = got
    res["fields_differing_from_saved_except_skip_list"] = diff
    if diff:
        save()
        T.fail(f"explicit config differs from the checkpoint's in {diff}")
    # as a dict: get_hf_quantizer reads the config's quantization_config with .get() (an object fails, see
    # run_log_stage1b_speed.txt) and turns it back into this BitsAndBytesConfig (AutoQuantizationConfig.from_dict)
    cfg.quantization_config = got

    t0 = time.time()
    proc = AutoProcessor.from_pretrained(T.REPO_ID, revision=T.REV)
    model = Qwen3VLForConditionalGeneration.from_pretrained(T.REPO_ID, revision=T.REV, config=cfg, dtype=torch.bfloat16,
                                                            device_map={"": 0}, attn_implementation="sdpa").eval()
    torch.cuda.synchronize()
    res["load_s"] = time.time() - t0
    res["mem_after_load_alloc_gb"] = torch.cuda.memory_allocated() / 2 ** 30
    res["mem_after_load_reserved_gb"] = torch.cuda.memory_reserved() / 2 ** 30
    qc = model.config.quantization_config
    res["skip_modules_in_loaded_config"] = (qc.to_dict() if hasattr(qc, "to_dict") else qc)["llm_int8_skip_modules"]
    n4 = smoke.count_4bit(model)
    res["linear4bit_modules"] = n4
    vis_dtypes = sorted({str(p.dtype) for n, p in model.named_parameters() if n.startswith("model.visual.")})
    res["visual_param_dtypes"] = vis_dtypes
    print(f"loaded in {res['load_s']:.1f} s; allocated {res['mem_after_load_alloc_gb']:.2f} GB "
          f"(reserved {res['mem_after_load_reserved_gb']:.2f} GB); 4-bit modules {n4}; vision parameter dtypes {vis_dtypes}")
    if n4["visual"] != 0 or n4["language_model"] != 252:
        res["stopped"] = "4-bit module counts differ from 0 vision / 252 language"
        save()
        T.fail(res["stopped"])

    text_in = T.prompt_text("not given")
    with open(ANSWERS, "w", encoding="utf-8") as fa:
        for size in SIZES:
            run = {"size": size, "per_clip": [], "oom": None}
            res["runs"].append(run)
            for j in pick:
                p, man, meta, e1 = clips[j]
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                t1 = time.time()
                frames, idx, fps, clip, ci = prepare(p, man, meta, e1, size)
                prep = time.time() - t1
                inputs, _, _ = smoke.build_inputs(proc, frames, idx, fps, clip, prompt_text_in=text_in)
                x = {k: v.to("cuda") for k, v in inputs.items()}
                x["pixel_values_videos"] = x["pixel_values_videos"].to(torch.bfloat16)
                n_in = int(x["input_ids"].shape[1])
                grid = x["video_grid_thw"].tolist()
                try:
                    torch.cuda.synchronize()
                    t2 = time.perf_counter()
                    with torch.no_grad():
                        out = model.generate(**x, max_new_tokens=T.MAX_NEW, do_sample=False, use_cache=True,
                                             temperature=None, top_p=None, top_k=None)
                    torch.cuda.synchronize()
                    gen = time.perf_counter() - t2
                except torch.cuda.OutOfMemoryError as e:
                    run["oom"] = {"clip": Path(p).stem, "input_tokens": n_in, "video_grid_thw": grid,
                                  "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "error": str(e)[:300]}
                    print(f"{Path(p).stem} at {size} x {size}: OUT OF MEMORY ({n_in} input tokens)")
                    del x, inputs
                    torch.cuda.empty_cache()
                    break
                raw = proc.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]
                t, why = T.parse(raw)
                row = {"clip": Path(p).stem, "prep_s": prep, "gen_s": gen, "input_tokens": n_in, "video_grid_thw": grid,
                       "new_tokens": int(out.shape[1] - n_in), "parse_ok": t is not None, "parse_reason": why,
                       "crop_side": ci["side"], "crop_shift": ci["shift"],
                       "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                       "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2 ** 30}
                run["per_clip"].append(row)
                fa.write(json.dumps({"clip": Path(p).stem, "size": size, "raw_answer": raw, "parse_ok": t is not None}) + "\n")
                fa.flush()
                print(f"{row['clip']} at {size}: prep {prep:.2f} s, generate {gen:.2f} s, {n_in} input tokens, "
                      f"{row['new_tokens']} new, parse ok {row['parse_ok']}, peak {row['peak_alloc_gb']:.2f} GB")
                del out, x, inputs
            if run["oom"] is None:
                break
        else:
            res["stopped"] = "out of memory at 448 and at 336 (stopped as pre-registered)"
    res["power_end"] = vz.power()[1]
    save()
    if res.get("stopped"):
        T.fail(res["stopped"])


def report():
    T.check_plan()
    T.refuse(REPORT)
    S = json.loads(SPEED_JSON.read_text(encoding="utf-8"))
    L = ["TYPE MODEL, OPTION 1 (ZERO-SHOT QWEN3-VL-8B ON A CROP): STAGE 1b REPORT (corrected loading, AMENDMENT_1.txt)",
         "Synthetic only; no real file; no accuracy computed and no answer compared with the true types; PLAN.txt, "
         "PROMPT.txt and the stage 1 results unchanged; nothing committed.",
         f"PLAN.txt sha256 {T.PLAN_SHA}; PROMPT.txt sha256 {T.PROMPT_SHA}; AMENDMENT_1.txt sha256 {S['amendment_sha256']} "
         f"(now {sha(AMEND)}).", "",
         "LOADING (AMENDMENT_1)",
         f"  checkpoint {T.REPO_ID} @ {T.REV}, offline (HF_HUB_OFFLINE={S['hf_hub_offline']}).",
         f"  explicit BitsAndBytesConfig = the checkpoint's own quantization_config; fields that differ from it apart from "
         f"the skip list: {S['fields_differing_from_saved_except_skip_list'] or 'none'}.",
         f"  llm_int8_skip_modules: {S['quantization_config_explicit']['llm_int8_skip_modules']}",
         f"    (checkpoint names + full paths in transformers 5.22.1: {', '.join(FULL_PATHS)})",
         "  Passed as the model config's quantization_config (config=), because transformers 5.22.1 uses the "
         "config's quantization_config for a pre-quantised checkpoint and ignores a BitsAndBytesConfig given only as "
         "quantization_config= (merge_quantization_configs; BitsAndBytesConfig is not a loading-attributes config type). "
         "Other arguments as in PLAN.txt: dtype bfloat16, device_map {'': 0}, attn_implementation sdpa.",
         f"  skip list in the loaded model's config: {S['skip_modules_in_loaded_config']}",
         f"  CHECK after loading: 4-bit Linear modules {S['linear4bit_modules']} (required: visual 0, language_model 252) "
         f"-> {'PASS' if S['linear4bit_modules']['visual'] == 0 and S['linear4bit_modules']['language_model'] == 252 else 'FAIL'}; "
         f"vision-tower parameter dtypes {S['visual_param_dtypes']}",
         f"  model load {S['load_s']:.1f} s (not counted per clip); GPU memory after load: allocated "
         f"{S['mem_after_load_alloc_gb']:.2f} GB, reserved {S['mem_after_load_reserved_gb']:.2f} GB "
         f"({S['gpu']}, {S['gpu_total_gb']:.2f} GB)", "",
         "MEMORY AND SPEED (the same 5 clips as stage 1, numpy default_rng(1); answers parsed only)",
         f"  power at start: {S['power_start']}"]
    used = None
    for run in S["runs"]:
        L.append(f"  {run['size']} x {run['size']}:")
        for r in run["per_clip"]:
            L.append(f"    {r['clip']}: frames {r['prep_s']:.2f} s (CPU), generate {r['gen_s']:.2f} s (GPU), "
                     f"{r['input_tokens']} input tokens (video grid t,h,w {r['video_grid_thw'][0]}), {r['new_tokens']} "
                     f"new tokens, parses: {r['parse_ok']}{' (' + r['parse_reason'] + ')' if r['parse_reason'] else ''}; "
                     f"peak allocated {r['peak_alloc_gb']:.2f} GB, reserved {r['peak_reserved_gb']:.2f} GB")
        if run["oom"]:
            o = run["oom"]
            g = o["video_grid_thw"][0]
            L.append(f"    {o['clip']} (first clip): OUT OF MEMORY at generate ({o['input_tokens']} input tokens; video "
                     f"grid t,h,w {g} = {g[1] * 16} x {g[2] * 16} px per frame as given to the model; peak allocated "
                     f"{o['peak_alloc_gb']:.2f} GB); error: \"{o['error'].split(' If reserved')[0]}\"")
            if run["size"] != g[1] * 16:
                L.append(f"      note: the processor resized the {run['size']} x {run['size']} crops to {g[1] * 16} x "
                         f"{g[2] * 16} (a multiple of 32 = patch 16 x merge 2); this was not chosen by us.")
        else:
            used = run
    if S.get("stopped"):
        L += [f"  RESULT: STOPPED: {S['stopped']}. No clip finished, so there is no generate time, no per-clip "
              "frame time, no new-token count, no parsed answer and no estimate for the 755 clips; stage1b_answers.jsonl "
              "was created before the first clip and is empty (0 bytes).",
              f"  Memory: the loaded model alone takes {S['mem_after_load_alloc_gb']:.2f} GB of the "
              f"{S['gpu_total_gb']:.2f} GB card, so about {S['gpu_total_gb'] - S['mem_after_load_reserved_gb']:.2f} GB is "
              "left for the vision tower and the language model on 16 frames, and part of that is used outside PyTorch "
              "(both errors report 0 bytes free at about 7.05-7.08 GB allocated by PyTorch).",
              "  Fallback used: 336 x 336 (as pre-registered), which also ran out of memory -> stopped as pre-registered."]
    if used:
        ok = used["per_clip"]
        per = [r["prep_s"] + r["gen_s"] for r in ok]
        L += [f"  fallback used: {'no (16 frames at 448 x 448, as planned)' if used['size'] == 448 else 'YES: 16 frames at 336 x 336 (448 ran out of memory)'}",
              f"  mean per clip: frames {np.mean([r['prep_s'] for r in ok]):.2f} s + generate "
              f"{np.mean([r['gen_s'] for r in ok]):.2f} s = {np.mean(per):.2f} s (min {min(per):.2f}, max {max(per):.2f}); "
              f"max peak allocated {max(r['peak_alloc_gb'] for r in ok):.2f} GB (reserved "
              f"{max(r['peak_reserved_gb'] for r in ok):.2f} GB); answers that parse {sum(r['parse_ok'] for r in ok)}/{len(ok)}",
              f"  estimated time for the 755 clips: 755 x {np.mean(per):.2f} s = {755 * np.mean(per) / 3600:.2f} h, plus one "
              f"model load (~{S['load_s']:.0f} s); from 5 clips only, so a rough estimate."]
    L += [f"  power at end: {S.get('power_end')}", "",
          "Raw answers: stage1b_answers.jsonl (clip, size, raw answer, parses; no true types). Numbers: stage1b_speed.json. "
          "Logs: run_log_stage1b_speed.txt (first attempt: stopped inside from_pretrained before any weight was loaded, "
          "AttributeError: 'BitsAndBytesConfig' object has no attribute 'get', because the config's quantization_config "
          "must be a dict), run_log_stage1b_speed_retry1.txt (the run above; same explicit config passed in dict form), "
          "run_log_stage1b_report.txt. Script: type_vlm8b_stage1b.py."]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--speed", action="store_true")
    g.add_argument("--report", action="store_true")
    a = ap.parse_args()
    speed() if a.speed else report()


if __name__ == "__main__":
    main()
