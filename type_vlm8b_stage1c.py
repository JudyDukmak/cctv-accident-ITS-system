"""Type model, option 1, STAGE 1c: AMENDMENT_1 loading plus the embedding tables on the CPU (results/type_vlm8b/
AMENDMENT_2.txt), then the memory and speed test on the same 5 clips. SYNTHETIC ONLY. NO accuracy and NO comparison with
the true types. Imports type_vlm8b.py and type_vlm8b_stage1b.py (not changed). Run with .venv-vlm.

  --speed   AMENDMENT_2 steps in order, each tried on all 5 clips (numpy default_rng(1)); the next step only after an
            out-of-memory error:
              step 1: embed_tokens on the CPU, 448 x 448
              step 2: embed_tokens and lm_head on the CPU, 448 x 448
              step 3: embed_tokens and lm_head on the CPU, 336 x 336
              then stop
  --report  stage1c_report.txt

Library requirement (transformers 5.22.1, quantizers/quantizer_bnb_4bit.py, read before running): validate_environment
raises a ValueError for any device_map with a "cpu" entry unless quantization_config.llm_int8_enable_fp32_cpu_offload is
True. In the 4-bit quantizer that flag does only two things: it lets the device_map through, and it adds the CPU
modules to modules_to_not_convert (embed_tokens and lm_head are already in the skip list). It casts nothing to fp32, so
the flag is set and the CPU tables stay bfloat16 (checked after loading).
"""
import argparse
import copy
import gc
import json
import os
import time
from pathlib import Path

import numpy as np

import type_vlm8b as T
import type_vlm8b_stage1b as B

OUT = T.OUT
AMEND1, AMEND2 = OUT / "AMENDMENT_1.txt", OUT / "AMENDMENT_2.txt"
SPEED_JSON, ANSWERS, REPORT = OUT / "stage1c_speed.json", OUT / "stage1c_answers.jsonl", OUT / "stage1c_report.txt"
LM = "model.language_model"
STEPS = [{"step": 1, "cpu": ["embed_tokens"], "size": 448},
         {"step": 2, "cpu": ["embed_tokens", "lm_head"], "size": 448},
         {"step": 3, "cpu": ["embed_tokens", "lm_head"], "size": 336}]


def device_map(cpu):
    """Every module named explicitly: the listed tables on the CPU, everything else on GPU 0."""
    return {"model.visual": 0, f"{LM}.embed_tokens": "cpu" if "embed_tokens" in cpu else 0, f"{LM}.layers": 0,
            f"{LM}.norm": 0, f"{LM}.rotary_emb": 0, "lm_head": "cpu" if "lm_head" in cpu else 0}


def load(cpu, res_step):
    import torch
    import vlm_smoke_test as smoke
    from transformers import AutoConfig, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    cfg = AutoConfig.from_pretrained(T.REPO_ID, revision=T.REV)
    saved = copy.deepcopy(cfg.quantization_config if isinstance(cfg.quantization_config, dict)
                          else cfg.quantization_config.to_dict())
    qd = copy.deepcopy(saved)
    qd["llm_int8_skip_modules"] = list(saved["llm_int8_skip_modules"]) + B.FULL_PATHS   # AMENDMENT_1
    qd["llm_int8_enable_fp32_cpu_offload"] = True                                       # needed for any CPU entry
    got = BitsAndBytesConfig.from_dict(qd).to_dict()
    diff = sorted(k for k in saved if k != "llm_int8_skip_modules" and got.get(k) != saved[k])
    res_step["fields_differing_from_saved_except_skip_list"] = {k: [saved[k], got.get(k)] for k in diff}
    if diff != ["llm_int8_enable_fp32_cpu_offload"]:
        T.fail(f"explicit config differs from the checkpoint's in {diff}")
    cfg.quantization_config = got  # dict form (see type_vlm8b_stage1b.py)
    dm = device_map(cpu)
    res_step["device_map"] = dm
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(T.REPO_ID, revision=T.REV, config=cfg, dtype=torch.bfloat16,
                                                            device_map=dm, attn_implementation="sdpa").eval()
    torch.cuda.synchronize()
    res_step["load_s"] = time.time() - t0
    res_step["mem_after_load_alloc_gb"] = torch.cuda.memory_allocated() / 2 ** 30
    res_step["mem_after_load_reserved_gb"] = torch.cuda.memory_reserved() / 2 ** 30
    res_step["linear4bit_modules"] = smoke.count_4bit(model)
    on_cpu = {}
    for n, p in model.named_parameters():
        if p.device.type == "cpu":
            mod = n.rsplit(".", 1)[0]
            on_cpu.setdefault(mod, {"numel": 0, "dtypes": set()})
            on_cpu[mod]["numel"] += p.numel()
            on_cpu[mod]["dtypes"].add(str(p.dtype))
    res_step["modules_on_cpu"] = {k: {"params": v["numel"], "dtypes": sorted(v["dtypes"])} for k, v in on_cpu.items()}
    res_step["visual_param_dtypes"] = sorted({str(p.dtype) for n, p in model.named_parameters() if n.startswith("model.visual.")})
    res_step["skip_modules_in_loaded_config"] = model.config.quantization_config.to_dict()["llm_int8_skip_modules"] if hasattr(
        model.config.quantization_config, "to_dict") else model.config.quantization_config["llm_int8_skip_modules"]
    print(f"step {res_step['step']}: loaded in {res_step['load_s']:.1f} s; GPU allocated {res_step['mem_after_load_alloc_gb']:.2f} GB "
          f"(reserved {res_step['mem_after_load_reserved_gb']:.2f} GB); 4-bit modules {res_step['linear4bit_modules']}; "
          f"on CPU {res_step['modules_on_cpu']}")
    return model


def speed():
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    import vlm_smoke_test as smoke
    import vlm_zero_shot as vz
    from transformers import AutoProcessor
    T.check_plan()
    for a in (AMEND1, AMEND2):
        if not a.is_file():
            T.fail(f"{a.name} missing")
    T.refuse(SPEED_JSON, ANSWERS)
    on, ptxt = vz.power()
    if not on:
        T.fail("on battery")
    clips = T.load_clips()
    pick = sorted(np.random.default_rng(1).choice(len(clips), size=5, replace=False).tolist())
    res = {"amendment_1_sha256": B.sha(AMEND1), "amendment_2_sha256": B.sha(AMEND2),
           "clips": [Path(clips[j][0]).stem for j in pick], "power_start": ptxt, "gpu": torch.cuda.get_device_name(0),
           "gpu_total_gb": torch.cuda.get_device_properties(0).total_memory / 2 ** 30,
           "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"), "steps": [], "step_used": None}

    def save():
        SPEED_JSON.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")

    proc = AutoProcessor.from_pretrained(T.REPO_ID, revision=T.REV)
    text_in = T.prompt_text("not given")
    model, model_cpu = None, None
    with open(ANSWERS, "w", encoding="utf-8") as fa:
        for st in STEPS:
            rs = {**st, "per_clip": [], "oom": None}
            res["steps"].append(rs)
            if model is None or model_cpu != st["cpu"]:
                if model is not None:
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                model, model_cpu = load(st["cpu"], rs), st["cpu"]
                n4 = rs["linear4bit_modules"]
                if n4["visual"] != 0 or n4["language_model"] != 252:
                    res["stopped"] = f"step {st['step']}: 4-bit module counts {n4} differ from 0 vision / 252 language"
                    save()
                    T.fail(res["stopped"])
            else:
                rs["model"] = f"same loaded model as step {st['step'] - 1}"
            save()
            for j in pick:
                p, man, meta, e1 = clips[j]
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                t1 = time.time()
                frames, idx, fps, clip, ci = B.prepare(p, man, meta, e1, st["size"])
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
                    rs["oom"] = {"clip": Path(p).stem, "input_tokens": n_in, "video_grid_thw": grid,
                                 "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "error": str(e)[:300]}
                    print(f"step {st['step']}: {Path(p).stem} at {st['size']}: OUT OF MEMORY ({n_in} input tokens)")
                    del x, inputs
                    gc.collect()
                    torch.cuda.empty_cache()
                    break
                raw = proc.batch_decode(out[:, n_in:].cpu(), skip_special_tokens=True)[0]
                t, why = T.parse(raw)
                row = {"clip": Path(p).stem, "prep_s": prep, "gen_s": gen, "input_tokens": n_in, "video_grid_thw": grid,
                       "new_tokens": int(out.shape[1] - n_in), "parse_ok": t is not None, "parse_reason": why,
                       "crop_side": ci["side"], "crop_shift": ci["shift"],
                       "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                       "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2 ** 30}
                rs["per_clip"].append(row)
                fa.write(json.dumps({"clip": Path(p).stem, "step": st["step"], "size": st["size"], "raw_answer": raw,
                                     "parse_ok": t is not None}) + "\n")
                fa.flush()
                print(f"step {st['step']}: {row['clip']} at {st['size']}: prep {prep:.2f} s, generate {gen:.2f} s, {n_in} "
                      f"input tokens, {row['new_tokens']} new, parse ok {row['parse_ok']}, peak {row['peak_alloc_gb']:.2f} GB")
                del out, x, inputs
            save()
            if rs["oom"] is None:
                res["step_used"] = st["step"]
                break
        else:
            res["stopped"] = "out of memory at all three AMENDMENT_2 steps (stopped as pre-registered)"
    res["power_end"] = vz.power()[1]
    save()
    if res.get("stopped"):
        T.fail(res["stopped"])


def report():
    T.check_plan()
    T.refuse(REPORT)
    S = json.loads(SPEED_JSON.read_text(encoding="utf-8"))
    L = ["TYPE MODEL, OPTION 1 (ZERO-SHOT QWEN3-VL-8B ON A CROP): STAGE 1c REPORT (AMENDMENT_1 loading + AMENDMENT_2 "
         "embedding tables on the CPU)",
         "Synthetic only; no real file; no accuracy computed and no answer compared with the true types; PLAN.txt, "
         "PROMPT.txt, AMENDMENT_1.txt and all earlier results unchanged; nothing committed.",
         f"PLAN.txt sha256 {T.PLAN_SHA}; PROMPT.txt sha256 {T.PROMPT_SHA}; AMENDMENT_1.txt sha256 "
         f"{S['amendment_1_sha256']}; AMENDMENT_2.txt sha256 {S['amendment_2_sha256']}.",
         f"Checkpoint {T.REPO_ID} @ {T.REV}, offline (HF_HUB_OFFLINE={S['hf_hub_offline']}); {S['gpu']} "
         f"({S['gpu_total_gb']:.2f} GB); power at start: {S['power_start']}.",
         "Loading: as AMENDMENT_1 (explicit copy of the checkpoint's quantization_config, skip list extended with "
         f"{', '.join(B.FULL_PATHS)}, passed in dict form as the config's quantization_config), dtype bfloat16, sdpa, and "
         "the device_map below. One more config field had to change: llm_int8_enable_fp32_cpu_offload false -> true, "
         "because transformers 5.22.1 refuses any device_map with a 'cpu' entry for a bitsandbytes model without it. In the "
         "4-bit loader this flag only lets the device_map through and adds the CPU modules to the not-quantised list "
         "(they are on it already); it casts nothing to fp32 (the CPU tables' dtypes are checked below).",
         "Test: the same 5 clips as stages 1 and 1b (numpy default_rng(1)); answers parsed only.", ""]
    for rs in S["steps"]:
        L.append(f"AMENDMENT_2 STEP {rs['step']}: on the CPU {', '.join(rs['cpu'])}; 16 frames at {rs['size']} x {rs['size']}")
        if "load_s" in rs:
            n4 = rs["linear4bit_modules"]
            ok = n4["visual"] == 0 and n4["language_model"] == 252
            L += [f"  device_map: {rs['device_map']}",
                  f"  config fields differing from the checkpoint's (apart from the skip list): "
                  f"{rs['fields_differing_from_saved_except_skip_list']}",
                  f"  CHECK after loading: 4-bit Linear modules {n4} (required: visual 0, language_model 252) -> "
                  f"{'PASS' if ok else 'FAIL'}; vision-tower parameter dtypes {rs['visual_param_dtypes']}",
                  f"  modules on the CPU: {rs['modules_on_cpu']}",
                  f"  model load {rs['load_s']:.1f} s (not counted per clip); GPU memory after load: allocated "
                  f"{rs['mem_after_load_alloc_gb']:.2f} GB, reserved {rs['mem_after_load_reserved_gb']:.2f} GB"]
        else:
            L.append(f"  {rs.get('model', '')}")
        for r in rs["per_clip"]:
            L.append(f"  {r['clip']}: frames {r['prep_s']:.2f} s (CPU), generate {r['gen_s']:.2f} s, {r['input_tokens']} "
                     f"input tokens (video grid t,h,w {r['video_grid_thw'][0]}), {r['new_tokens']} new tokens, parses: "
                     f"{r['parse_ok']}{' (' + r['parse_reason'] + ')' if r['parse_reason'] else ''}; peak GPU allocated "
                     f"{r['peak_alloc_gb']:.2f} GB, reserved {r['peak_reserved_gb']:.2f} GB")
        if rs["oom"]:
            o = rs["oom"]
            g = o["video_grid_thw"][0]
            L.append(f"  {o['clip']}: OUT OF MEMORY at generate ({o['input_tokens']} input tokens; video grid t,h,w {g} = "
                     f"{g[1] * 16} x {g[2] * 16} px per frame as given to the model; peak allocated {o['peak_alloc_gb']:.2f} GB); "
                     f"error: \"{o['error'].split(' If reserved')[0]}\" -> next step")
            if rs["size"] != g[1] * 16:
                L.append(f"    note: the processor resized the {rs['size']} x {rs['size']} crops to {g[1] * 16} x {g[2] * 16} "
                         "(a multiple of 32); this was not chosen by us.")
        L.append("")
    chk = OUT / "stage1c_cpu_check.json"
    if chk.is_file():
        K = json.loads(chk.read_text(encoding="utf-8"))
        L += ["WHICH MODULES ARE ON THE CPU (separate load-only check, type_vlm8b_stage1c_cpucheck.py, no frames, no generate)",
              "  The 'modules on the CPU: {}' lines above (and 'on CPU {}' in run_log_stage1c_speed.txt) are EMPTY BY MISTAKE: "
              "accelerate offloads the device_map's CPU entries, so in the model their parameters are on the 'meta' device and "
              "the weights are held on the CPU by the module's hook; the speed script only looked for parameters on the CPU. "
              "The check reloaded the model with the step 1 and step 2 device maps and read the hooks:"]
        for rs in K["steps"]:
            L.append(f"  step {rs['step']}: hf_device_map {rs['hf_device_map']}; 4-bit modules {rs['linear4bit_modules']}")
            for m, o in rs["offloaded_modules"].items():
                for w, s in o["stored_weights"].items():
                    L.append(f"    {m}: {o['hook']}, offload {o['offload']}, execution device {o['execution_device']} "
                             f"(GPU), parameters in the model on {o['param_devices_in_model']}; stored weight {w} "
                             f"{s['shape']} {s['dtype']} on {s['device']} (held in CPU RAM: {s['held_in_cpu_state_dict']})")
        L += ["  So the tables are STORED in CPU RAM in bfloat16 and copied to GPU 0 each time the module runs (the "
              "computation itself is on the GPU); everything else is on GPU 0. The transient copy is included in the "
              "per-clip peak GPU memory above.",
              "  (The check's own GPU-memory-after-load for step 2, 5.57 GB, is inflated: a loop variable in the check script "
              "still held the step 1 model's GPU lm_head weight, 1.16 GiB, when step 2 was loaded. The speed run's figures "
              "above are the valid ones.)", ""]
    used = next((rs for rs in S["steps"] if rs["step"] == S["step_used"]), None)
    if used:
        ok = used["per_clip"]
        per = [r["prep_s"] + r["gen_s"] for r in ok]
        L += [f"RESULT: AMENDMENT_2 step used: {used['step']} (CPU: {', '.join(used['cpu'])}; {used['size']} x {used['size']}); "
              f"all {len(ok)} clips finished.",
              f"  mean per clip: frames {np.mean([r['prep_s'] for r in ok]):.2f} s + generate {np.mean([r['gen_s'] for r in ok]):.2f} s "
              f"= {np.mean(per):.2f} s (min {min(per):.2f}, max {max(per):.2f}); max peak GPU allocated "
              f"{max(r['peak_alloc_gb'] for r in ok):.2f} GB (reserved {max(r['peak_reserved_gb'] for r in ok):.2f} GB); "
              f"new tokens {[r['new_tokens'] for r in ok]}; answers that parse {sum(r['parse_ok'] for r in ok)}/{len(ok)}",
              f"  estimated time for the 755 clips: 755 x {np.mean(per):.2f} s = {755 * np.mean(per) / 3600:.2f} h, plus one "
              "model load; from 5 clips only, so a rough estimate."]
    else:
        L.append(f"RESULT: STOPPED: {S.get('stopped')}. No estimate for the 755 clips.")
    L += [f"Power at end: {S.get('power_end')}", "",
          "Raw answers: stage1c_answers.jsonl (clip, step, size, raw answer, parses; no true types). Numbers: "
          "stage1c_speed.json; stage1c_cpu_check.json. Logs: run_log_stage1c_speed.txt, run_log_stage1c_cpu_check.txt "
          "(first try of the check: KeyError from a wrong weights_map lookup in the check script, nothing written), "
          "run_log_stage1c_cpu_check_retry1.txt, run_log_stage1c_report.txt. Scripts: type_vlm8b_stage1c.py, "
          "type_vlm8b_stage1c_cpucheck.py."]
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
