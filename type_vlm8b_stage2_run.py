"""Type model, option 1, STAGE 2 RUN: zero-shot Qwen3-VL-8B on the 755 fold-1 window-A clips, as PLAN.txt with
AMENDMENT_1 (loading) and AMENDMENT_2 step 2 (embed_tokens and lm_head stored on the CPU; 16 frames at 448 x 448).
SYNTHETIC ONLY. Writes NO true type: scoring is a separate step (type_vlm8b_stage2_score.py).

Resumable: one JSON line per clip is appended (and flushed) to results/type_vlm8b/stage2_answers.jsonl; clips already in
the file are skipped. Each session appends one line to stage2_sessions.jsonl. Power is checked before the first clip
and every 50 clips; on battery the run pauses (checks again every 60 s) until mains power is back.
An out-of-memory error or any other error stops the run (rerun to resume). Run with .venv-vlm, HF_HUB_OFFLINE=1.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np

import type_vlm8b as T
import type_vlm8b_stage1b as B
import type_vlm8b_stage1c as C

OUT = T.OUT
ANSWERS, SESSIONS = OUT / "stage2_answers.jsonl", OUT / "stage2_sessions.jsonl"
NOTE = OUT / "STAGE2_NOTE.txt"
STEP2 = C.STEPS[1]  # {"step": 2, "cpu": ["embed_tokens", "lm_head"], "size": 448}
SCENE = "not given"
POWER_EVERY, PAUSE_S = 50, 60


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def wait_for_mains(vz, where):
    on, ptxt = vz.power()
    while not on:
        print(f"{now()} {where}: ON BATTERY ({ptxt}); pausing, next check in {PAUSE_S} s", flush=True)
        time.sleep(PAUSE_S)
        on, ptxt = vz.power()
    return ptxt


def read_done():
    done = {}
    if ANSWERS.is_file():
        for n, line in enumerate(ANSWERS.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                T.fail(f"{ANSWERS.name} line {n} is not valid JSON; fix by hand before resuming")
            if r["rgb_path"] in done:
                T.fail(f"{ANSWERS.name}: {r['rgb_path']} appears twice")
            done[r["rgb_path"]] = r
    return done


def main():
    import torch
    import vlm_smoke_test as smoke
    import vlm_zero_shot as vz
    from transformers import AutoProcessor
    T.check_plan()
    for a in (C.AMEND1, C.AMEND2, NOTE):
        if not a.is_file():
            T.fail(f"{a.name} missing")
    clips = T.load_clips()
    paths = [c[0] for c in clips]
    done = read_done()
    if not set(done) <= set(paths):
        T.fail("answers file has clips outside the 755")
    todo = [c for c in clips if c[0] not in done]
    sess = {"start": now(), "done_before": len(done), "todo": len(todo), "amendment_1_sha256": B.sha(C.AMEND1),
            "amendment_2_sha256": B.sha(C.AMEND2), "stage2_note_sha256": B.sha(NOTE),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"), "step": STEP2}
    print(f"{now()} {len(done)} clips already done, {len(todo)} to do", flush=True)
    if not todo:
        print("nothing to do")
        return
    sess["power_start"] = wait_for_mains(vz, "start")
    proc = AutoProcessor.from_pretrained(T.REPO_ID, revision=T.REV)
    rs = {"step": STEP2["step"]}
    model = C.load(STEP2["cpu"], rs)
    n4 = rs["linear4bit_modules"]
    sess.update({k: rs[k] for k in ("load_s", "mem_after_load_alloc_gb", "linear4bit_modules", "device_map",
                                    "fields_differing_from_saved_except_skip_list")})
    if n4["visual"] != 0 or n4["language_model"] != 252:
        T.fail(f"4-bit module counts {n4} differ from 0 vision / 252 language")
    text_in = T.prompt_text(SCENE)
    n_sess, t_sess = 0, time.time()
    try:
        with open(ANSWERS, "a", encoding="utf-8") as fa:
            for k, (p, man, meta, e1) in enumerate(todo):
                if k > 0 and k % POWER_EVERY == 0:
                    wait_for_mains(vz, f"after {k} clips this session")
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                t1 = time.time()
                frames, idx, fps, clip, ci = B.prepare(p, man, meta, e1, STEP2["size"])
                prep = time.time() - t1
                inputs, _, _ = smoke.build_inputs(proc, frames, idx, fps, clip, prompt_text_in=text_in)
                x = {kk: v.to("cuda") for kk, v in inputs.items()}
                x["pixel_values_videos"] = x["pixel_values_videos"].to(torch.bfloat16)
                n_in = int(x["input_ids"].shape[1])
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                with torch.no_grad():
                    out = model.generate(**x, max_new_tokens=T.MAX_NEW, do_sample=False, use_cache=True,
                                         temperature=None, top_p=None, top_k=None)
                torch.cuda.synchronize()
                gen = time.perf_counter() - t2
                raw = proc.batch_decode(out[:, n_in:].cpu(), skip_special_tokens=True)[0]
                parsed, why = T.parse(raw)
                final = T.apply_rule(parsed, SCENE) if parsed is not None else None
                row = {"rgb_path": p, "clip": Path(p).stem, "raw_answer": raw, "parsed_type": parsed, "final_type": final,
                       "rule_applied": parsed is not None and final != parsed, "parse_ok": parsed is not None,
                       "parse_reason": why, "scene_layout": SCENE, "size": STEP2["size"],
                       "video_grid_thw": x["video_grid_thw"].tolist()[0], "input_tokens": n_in,
                       "new_tokens": int(out.shape[1] - n_in), "prep_s": prep, "gen_s": gen,
                       "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                       "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2 ** 30,
                       "crop_side": ci["side"], "crop_left": ci["left"], "crop_top": ci["top"], "time": now()}
                fa.write(json.dumps(row) + "\n")
                fa.flush()
                os.fsync(fa.fileno())
                n_sess += 1
                del out, x, inputs
                if n_sess % 25 == 0:
                    el = time.time() - t_sess
                    print(f"{now()} {len(done) + n_sess}/755 done; this session {n_sess} in {el / 60:.1f} min "
                          f"({el / n_sess:.2f} s/clip); last peak {row['peak_alloc_gb']:.2f} GB", flush=True)
    finally:
        sess.update(end=now(), done_this_session=n_sess, seconds_this_session=time.time() - t_sess,
                    done_total=len(read_done()), power_end=vz.power()[1])
        with open(SESSIONS, "a", encoding="utf-8") as fs:
            fs.write(json.dumps(sess, default=str) + "\n")
        print(f"{now()} session end: {n_sess} clips this session; {sess['done_total']}/755 in the answers file", flush=True)


if __name__ == "__main__":
    main()
