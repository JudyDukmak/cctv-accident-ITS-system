"""VLM smoke test: Qwen3-VL 2B / 4B memory, speed and output format on ONE synthetic clip.

SYNTHETIC DATA ONLY. Reads synthetic_split.csv, ACCIDENT/dataset/metadata-synthetic.csv and one
synthetic video. It never opens a real clip or the frozen test set (real_test_set.csv).

Clip: CLIP (fold-0 validation, Town06 head-on). Frames are sampled evenly from t* - 2 s to t* + 2 s
(the narrow-path window, endpoints included) from the original video and resized so the longer
side is 448 px (PIL bicubic). The Qwen3-VL processor then rounds each side to a multiple of 32
(patch 16 x merge 2); its default total-pixel cap is raised so it does not shrink the frames.
Frame indices and fps are passed as video_metadata so the frame timestamps in the prompt are real.

Tests (each in a fresh process, so peak memory never mixes):
  A  2B bf16        8 frames  generate
  B  2B bf16       16 frames  generate
  C  2B 4-bit NF4  16 frames  generate
  D  4B 4-bit NF4   8 frames  generate
  E  4B 4-bit NF4  16 frames  generate
  F  2B 4-bit NF4   8 frames  QLoRA step
  G  4B 4-bit NF4   8 frames  QLoRA step
Generate: greedy, max MAX_NEW_TOKENS new tokens. QLoRA step: LoRA r 16 on the language-model
attention projections (q/k/v/o), batch 1, gradient checkpointing on, one forward + backward +
AdamW step on prompt + target JSON (loss on the target tokens only). Nothing is saved; the
target uses the clip's own labels and the step only measures memory and time.
4-bit: bitsandbytes NF4, double quantization, bf16 compute; transformers' default choice of
which Linear layers to quantize (counts are recorded). The non-quantized weights stay bf16
(no fp32 upcast from prepare_model_for_kbit_training).

Status: OOM (CUDA out of memory, the test stops there), OVER_VRAM (peak reserved > 97% of GPU
memory: Windows may silently spill into shared system memory), ERROR, else OK.
Memory: torch.cuda.max_memory_allocated / max_memory_reserved over the whole test (load included);
NVML peak = device-wide used memory polled every NVML_POLL_S (includes other processes; the
baseline before the test is recorded).

Outputs: gpu_check/vlm_smoke_test.json, .txt, and .log (full terminal output).

--stage-test (planning of the zero-shot VLM stage, weeks 6-7; memory, speed and output format only, NO accuracy):
  two synthetic clips (STAGE_CLIPS: the clip above, 17.8 s, and the longest synthetic clip, 32.2 s), configs
  2B bf16, 2B NF4, 4B NF4 (4B bf16 not run: its weights alone exceed the GPU memory), inputs
    narrow16  16 frames evenly over t_hat +- 2 s, t_hat = the saved out-of-fold C_cctv_seed0 frozen prediction
              (STAGE_PRED), prompt PROMPT (type and point);
    full1fps / full2fps  the whole clip sampled at 1 / 2 fps (t_k = k / f, frame round(t_k * video fps)), the frame
              number k drawn in the top-left corner, prompt PROMPT_FULL_DRAFT (a draft, not frozen).
  One fresh process per (config, input); each clip is run STAGE_REPEATS times (call 1 includes CUDA warm-up; call 2
  is the warm time). Peak allocated memory per call (reset before each call), peak reserved over the process
  (load included), NVML device peak. Outputs results/vlm_stage_test/vlm_stage_test.{txt,json,log} and
  vlm_stage_test_frame_example.png (one labelled full-scan frame, to check the number is legible).

Usage:
    .venv-vlm\\Scripts\\python.exe gpu_check\\vlm_smoke_test.py               # all tests
    .venv-vlm\\Scripts\\python.exe gpu_check\\vlm_smoke_test.py --only A F    # a subset
    .venv-vlm\\Scripts\\python.exe gpu_check\\vlm_smoke_test.py --stage-test  # VLM-stage planning test
"""

import argparse
import csv
import gc
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUT_DIR.parent
DATASET = REPO_ROOT / "ACCIDENT" / "dataset"
SPLIT_CSV = REPO_ROOT / "synthetic_split.csv"
CLIP = "synthetic_videos/videos/head-on/Town06_head-on_clear_01.mp4"
FOLD = 0
HALF_WINDOW_S = 2.0
LONG_SIDE = 448
MAX_NEW_TOKENS = 64
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGET = r".*language_model.*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
LR = 1e-4
OVER_VRAM_FRAC = 0.97
NVML_POLL_S = 0.01
CHILD_TIMEOUT_S = 3600
GB = 2 ** 30
MODELS = {"2B": "Qwen/Qwen3-VL-2B-Instruct", "4B": "Qwen/Qwen3-VL-4B-Instruct"}
TESTS = {
    "A": dict(model="2B", precision="bf16", frames=8, mode="generate"),
    "B": dict(model="2B", precision="bf16", frames=16, mode="generate"),
    "C": dict(model="2B", precision="nf4", frames=16, mode="generate"),
    "D": dict(model="4B", precision="nf4", frames=8, mode="generate"),
    "E": dict(model="4B", precision="nf4", frames=16, mode="generate"),
    "F": dict(model="2B", precision="nf4", frames=8, mode="qlora"),
    "G": dict(model="4B", precision="nf4", frames=8, mode="qlora"),
}
TYPES = ["head-on", "rear-end", "sideswipe", "single", "t-bone"]
REQUIRED_KEYS = ["collision_type", "x", "y"]
PROMPT = (
    "These frames come from a fixed traffic camera and show a collision. "
    "Answer with one JSON object and nothing else, in this exact form: "
    '{"collision_type": "<one of head-on, rear-end, sideswipe, single, t-bone>", "x": <number 0-1>, "y": <number 0-1>}. '
    "collision_type is the type of collision (single means a single-vehicle crash). "
    "x and y give the collision point in the image as fractions of the image width (x, from the left) "
    "and height (y, from the top)."
)
STAGE_CLIPS = (CLIP, "synthetic_videos/videos/rear-end/Town04_rear-end_night_32.mp4")  # ~median; longest synthetic
STAGE_OUT_DIR = REPO_ROOT / "results" / "vlm_stage_test"  # stage-test outputs; gpu_check/ holds the script only
STAGE_PRED = REPO_ROOT / "results" / "diagnostics" / "C_cctv_seeds012" / "seed0_frozen" / "tta_clip_predictions_flip.csv"
STAGE_CONFIGS = {"2B-bf16": ("2B", "bf16"), "2B-nf4": ("2B", "nf4"), "4B-nf4": ("4B", "nf4")}
STAGE_NOT_RUN = {"4B-bf16": "not run: the bf16 weights are 8.88 GB (total_size in model.safetensors.index.json), "
                            "more than the 7.96 GB of GPU memory"}
STAGE_INPUTS = {"narrow16": None, "full1fps": 1.0, "full2fps": 2.0}
NARROW_FRAMES = 16
STAGE_REPEATS = 2
LOAD_LIMIT_S, CALL_LIMIT_S = 600, 300  # per combination: model load, and each call; exceeded -> child killed, FAILED
PROGRESS = "PROGRESS"  # timestamped child lines "<time> PROGRESS load|call start|done ..." drive the parent's watchdog
LABEL_FONT_PX = 28
PROMPT_FULL_DRAFT = (
    "These {n} frames were sampled in time order at {fps:g} frames per second from a fixed traffic camera. "
    "Each frame shows its frame number in the top-left corner (0 is the first frame). "
    "A collision happens in this video. Answer with one JSON object and nothing else, in this exact form: "
    '{{"frame": <the frame number at which the collision first happens>}}.'
)
VERSION_PACKAGES = ["torch", "torchvision", "transformers", "accelerate", "bitsandbytes", "peft",
                    "qwen-vl-utils", "av", "pillow", "numpy", "nvidia-ml-py", "huggingface-hub"]
RESULT_TAG = "RESULT_JSON "


def fail(msg):
    sys.exit(f"ERROR: {msg}")


# ---------------------------------------------------------------- data (synthetic only)

def load_clip_row(clip_path=CLIP, fold=FOLD):
    """A synthetic clip's row from metadata-synthetic.csv, after checking it is a fold-`fold` validation clip
    (fold None: any synthetic clip of the split; its validation fold is recorded)."""
    if not clip_path.startswith("synthetic_videos/"):
        fail(f"{clip_path} is not a synthetic clip")
    with open(SPLIT_CSV, newline="", encoding="utf-8") as f:
        split = [r for r in csv.DictReader(f) if r["rgb_path"].replace("\\", "/") == clip_path]
    if len(split) != 1:
        fail(f"{clip_path} found {len(split)} times in {SPLIT_CSV.name}")
    if fold is not None and (int(split[0]["val_fold"]) != fold or split[0]["split_dev"] != "val"):
        fail(f"{clip_path} is not a fold-{fold} validation clip: {split[0]}")
    meta_csv = DATASET / "metadata-synthetic.csv"
    with open(meta_csv, newline="", encoding="utf-8") as f:
        meta = [r for r in csv.DictReader(f) if r["rgb_path"].replace("\\", "/") == clip_path]
    if len(meta) != 1:
        fail(f"{clip_path} found {len(meta)} times in {meta_csv}")
    row = meta[0]
    video = DATASET / clip_path
    if not video.is_file():
        fail(f"video not found: {video}")
    return {
        "rgb_path": clip_path, "video": str(video), "scenario_id": split[0]["scenario_id"],
        "val_fold": int(split[0]["val_fold"]),
        "type": row["type"], "t_star": float(row["accident_time"]),
        "accident_frame": int(row["accident_frame"]), "no_frames": int(row["no_frames"]),
        "duration": float(row["duration"]), "width": int(row["width"]), "height": int(row["height"]),
        "center_x": float(row["center_x"]), "center_y": float(row["center_y"]),
    }


def frame_indices(clip, n, centre=None):
    """n frames evenly from centre - HALF_WINDOW_S to centre + HALF_WINDOW_S (centre default t*), endpoints included."""
    fps = clip["no_frames"] / clip["duration"]
    c = clip["t_star"] if centre is None else centre
    times = np.linspace(c - HALF_WINDOW_S, c + HALF_WINDOW_S, n)
    idx = np.clip(np.round(times * fps).astype(int), 0, clip["no_frames"] - 1)
    return idx.tolist(), fps


def scan_indices(clip, sample_fps):
    """Whole-clip sampling at sample_fps: t_k = k / sample_fps for every t_k inside the clip; frame round(t_k * fps)."""
    fps = clip["no_frames"] / clip["duration"]
    n = int(np.floor((clip["no_frames"] - 1) / fps * sample_fps + 1e-9)) + 1
    idx = np.clip(np.round(np.arange(n) / sample_fps * fps).astype(int), 0, clip["no_frames"] - 1)
    return idx.tolist(), fps


def draw_label(img, text):
    """Frame number in the top-left corner: white text on a black box."""
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=LABEL_FONT_PX)
    x0, y0, x1, y1 = draw.textbbox((6, 4), text, font=font)
    draw.rectangle((0, 0, x1 + 6, y1 + 4), fill=(0, 0, 0))
    draw.text((6, 4), text, fill=(255, 255, 255), font=font)
    return img


def read_frames(clip, indices, labels=None):
    """Decode the video once and keep the wanted frames, resized so the longer side is LONG_SIDE; with labels,
    labels[j] is drawn on the j-th frame (after the resize)."""
    import av
    from PIL import Image

    wanted = set(indices)
    got = {}
    with av.open(clip["video"]) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                img = frame.to_image()
                scale = LONG_SIDE / max(img.size)
                img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BICUBIC)
                got[i] = img.convert("RGB")
            if i >= max(indices):
                break
    missing = [i for i in indices if i not in got]
    if missing:
        fail(f"could not decode frames {missing} of {clip['video']}")
    out = []
    for j, i in enumerate(indices):
        img = got[i].copy()
        if labels is not None:
            img = draw_label(img, labels[j])
        out.append(np.asarray(img))
    return np.stack(out)  # (T, H, W, 3) uint8


# ---------------------------------------------------------------- output parsing

def parse_output(text):
    """Parse the model output as JSON (a ``` fence around it is allowed) and check the keys and values."""
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    res = {"json_ok": False, "keys_ok": False, "values_ok": False, "parsed": None}
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return res
    res["json_ok"] = True
    res["parsed"] = obj
    if not isinstance(obj, dict) or not all(k in obj for k in REQUIRED_KEYS):
        return res
    res["keys_ok"] = True
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1
    res["values_ok"] = obj["collision_type"] in TYPES and num(obj["x"]) and num(obj["y"])
    return res


def parse_full_output(text, n_frames):
    """Full-scan answer: JSON (a ``` fence allowed) with an integer "frame" in [0, n_frames - 1]."""
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    res = {"json_ok": False, "keys_ok": False, "values_ok": False, "parsed": None}
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return res
    res["json_ok"] = True
    res["parsed"] = obj
    if not isinstance(obj, dict) or "frame" not in obj:
        return res
    res["keys_ok"] = True
    v = obj["frame"]
    res["values_ok"] = isinstance(v, int) and not isinstance(v, bool) and 0 <= v < n_frames
    return res


# ---------------------------------------------------------------- child: one test

class NvmlPeak:
    """Background thread that records the peak device-wide used memory (NVML)."""

    def __init__(self):
        import pynvml
        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.baseline = self.used()
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def used(self):
        return self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.used())
            time.sleep(NVML_POLL_S)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.used())


def build_inputs(processor, frames, indices, fps, clip, target=None, prompt_text_in=PROMPT):
    """Processor inputs for the chat prompt; with target, also the labels (loss on the target only)."""
    from transformers.video_utils import VideoMetadata

    messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt_text_in}]}]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # The chat template wraps the video as <|vision_start|><|video_pad|><|vision_end|>, but this processor
    # only replaces <|video_pad|> and adds its own start/end pair around every frame. Keep the bare pad so
    # the result matches the reference format: <t seconds><|vision_start|>pads<|vision_end|> per frame.
    wrapped = processor.vision_start_token + processor.video_token + processor.vision_end_token
    if prompt_text.count(wrapped) != 1:
        fail(f"expected one {wrapped} in the chat prompt")
    prompt_text = prompt_text.replace(wrapped, processor.video_token)
    t, h, w = frames.shape[:3]
    vp = processor.video_processor
    kwargs = dict(
        videos=[frames], do_sample_frames=False, cap_pixels_per_frame=False, return_tensors="pt",
        size={"shortest_edge": vp.size["shortest_edge"], "longest_edge": t * 2 * h * w},
        video_metadata=[VideoMetadata(total_num_frames=clip["no_frames"], fps=fps, width=w, height=h,
                                      duration=clip["duration"], frames_indices=list(indices))],
    )
    prompt = processor(text=[prompt_text], **kwargs)
    if target is None:
        return prompt, None, prompt_text
    full = processor(text=[prompt_text + target + "<|im_end|>\n"], **kwargs)
    n = prompt["input_ids"].shape[1]
    if not bool((full["input_ids"][0, :n] == prompt["input_ids"][0]).all()):
        fail("prompt tokens are not a prefix of prompt + target tokens")
    labels = full["input_ids"].clone()
    labels[:, :n] = -100
    return full, labels, prompt_text


def load_model(spec):
    import torch
    from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    quant = None
    if spec["precision"] == "nf4":
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    return Qwen3VLForConditionalGeneration.from_pretrained(
        MODELS[spec["model"]], dtype=torch.bfloat16, device_map={"": 0},
        quantization_config=quant, attn_implementation="sdpa")


def count_4bit(model):
    import bitsandbytes as bnb
    names = [n for n, m in model.named_modules() if isinstance(m, bnb.nn.Linear4bit)]
    return {"visual": sum(".visual." in f".{n}." for n in names),
            "language_model": sum(".language_model." in f".{n}." for n in names),
            "other": sum(".visual." not in f".{n}." and ".language_model." not in f".{n}." for n in names)}


def run_child(test_id):
    import torch
    import transformers
    from transformers import AutoProcessor

    spec = TESTS[test_id]
    res = {"test": test_id, **spec, "model_id": MODELS[spec["model"]], "status": None, "oom": False}
    clip = load_clip_row()
    indices, fps = frame_indices(clip, spec["frames"])
    frames = read_frames(clip, indices)
    res.update(frame_indices=indices, frame_times_s=[round(i / fps, 3) for i in indices],
               frame_size_in=[int(frames.shape[2]), int(frames.shape[1])])

    if not torch.cuda.is_available():
        fail("CUDA is not available")
    device = torch.device("cuda", 0)
    nvml = NvmlPeak().start()
    total = torch.cuda.get_device_properties(device).total_memory
    model = opt = inputs = out = None
    try:
        torch.manual_seed(0)
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        processor = AutoProcessor.from_pretrained(MODELS[spec["model"]])
        model = load_model(spec)
        torch.cuda.synchronize()
        res["load_s"] = time.perf_counter() - t0
        res["alloc_after_load_gb"] = torch.cuda.memory_allocated(device) / GB
        res["attn_implementation"] = getattr(model.config, "_attn_implementation", None)
        if spec["precision"] == "nf4":
            res["linear4bit_modules"] = count_4bit(model)

        target = None
        if spec["mode"] == "qlora":
            target = json.dumps({"collision_type": clip["type"], "x": round(clip["center_x"], 3),
                                 "y": round(clip["center_y"], 3)})
            res["target"] = target
        inputs, labels, prompt_text = build_inputs(processor, frames, indices, fps, clip, target)
        ids = inputs["input_ids"][0]
        grid = inputs["video_grid_thw"][0].tolist()
        res.update(
            video_grid_thw=grid,
            frame_size_model=[grid[2] * processor.video_processor.patch_size,
                              grid[1] * processor.video_processor.patch_size],
            visual_tokens=int((ids == processor.video_token_id).sum()),
            vision_start_tokens=int((ids == processor.vision_start_token_id).sum()),
            input_tokens=int(ids.numel()),
        )
        inputs = inputs.to(device)
        inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(torch.bfloat16)

        if spec["mode"] == "generate":
            model.eval()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                     temperature=None, top_p=None, top_k=None)
            torch.cuda.synchronize()
            res["gen_s"] = time.perf_counter() - t0
            new = out[0, ids.numel():]
            res["new_tokens"] = int(new.numel())
            res["output"] = processor.batch_decode(new[None], skip_special_tokens=True)[0]
            res.update(parse_output(res["output"]))
        else:
            from peft import LoraConfig, get_peft_model

            model.config.use_cache = False
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
            model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                                     bias="none", target_modules=LORA_TARGET))
            params = [p for p in model.parameters() if p.requires_grad]
            res["trainable_params"] = sum(p.numel() for p in params)
            res["total_params"] = sum(p.numel() for p in model.parameters())
            res["lora_modules"] = sum(1 for n, _ in model.named_modules() if n.endswith(".lora_A"))
            res["target_tokens"] = int((labels != -100).sum())
            labels = labels.to(device)
            opt = torch.optim.AdamW(params, lr=LR)
            model.train()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss = model(**inputs, labels=labels).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            res["step_s"] = time.perf_counter() - t0
            res["loss"] = float(loss.detach())
        res["status"] = "OK"
    except torch.cuda.OutOfMemoryError as e:
        res.update(status="OOM", oom=True, error=str(e).splitlines()[0])
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        res.update(status="OOM", oom=True, error=str(e).splitlines()[0])
    finally:
        torch.cuda.synchronize()
        res["max_alloc_gb"] = torch.cuda.max_memory_allocated(device) / GB
        res["max_reserved_gb"] = torch.cuda.max_memory_reserved(device) / GB
        nvml.stop()
        res["nvml_baseline_gb"] = nvml.baseline / GB
        res["nvml_peak_gb"] = nvml.peak / GB
        res["total_gb"] = total / GB
        del model, opt, inputs, out
        gc.collect()
        torch.cuda.empty_cache()
    if res["status"] == "OK" and res["max_reserved_gb"] > OVER_VRAM_FRAC * res["total_gb"]:
        res["status"] = "OVER_VRAM"
    res["torch_cuda"] = torch.version.cuda
    res["transformers"] = transformers.__version__
    print(RESULT_TAG + json.dumps(res), flush=True)


# ---------------------------------------------------------------- parent: run all tests

def versions():
    out = {"python": platform.python_version()}
    for p in VERSION_PACKAGES:
        try:
            out[p] = importlib_metadata.version(p)
        except importlib_metadata.PackageNotFoundError:
            out[p] = None
    return out


def gpu_info():
    import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    name, driver = pynvml.nvmlDeviceGetName(h), pynvml.nvmlSystemGetDriverVersion()
    info = {"gpu": name.decode() if isinstance(name, bytes) else name,
            "driver": driver.decode() if isinstance(driver, bytes) else driver,
            "total_gb": pynvml.nvmlDeviceGetMemoryInfo(h).total / GB}
    pynvml.nvmlShutdown()
    return info


def check_weights_cached(only):
    from huggingface_hub import snapshot_download
    for mid in sorted({MODELS[TESTS[t]["model"]] for t in only}):
        try:
            snapshot_download(mid, local_files_only=True)
        except Exception as e:  # noqa: BLE001 - any failure means the weights are not usable offline
            fail(f"{mid} is not fully in the Hugging Face cache ({type(e).__name__}); download it first")


def stamp(msg):
    """Timestamped progress line, flushed at once (children run with python -u)."""
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}", flush=True)


def kill_tree(proc):
    """Kill a child and its descendants: on Windows the venv python.exe is a launcher that starts the real
    interpreter as its own child, so proc.kill() alone would leave the real process (and its GPU memory) running."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        proc.kill()


def run_test_process(test_id, log, child_args=None, meta=None):
    """One test in a fresh process; child_args / meta (default: the TESTS entry) for the --stage-test children.
    Offline (HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE), unbuffered (python -u). Watchdog: after a child's
    "PROGRESS load start" line it must print "PROGRESS load done" within LOAD_LIMIT_S, after "PROGRESS call start"
    "PROGRESS call done" within CALL_LIMIT_S; otherwise the child tree is killed and the test is FAILED."""
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_SYMLINKS_WARNING="1",
               TOKENIZERS_PARALLELISM="false", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), *(child_args or ["--child", test_id])]
    result = None
    tail = []
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                            text=True, encoding="utf-8", errors="replace")
    state = {"deadline": time.monotonic() + CHILD_TIMEOUT_S, "what": f"whole test > {CHILD_TIMEOUT_S} s",
             "killed": None}
    lock = threading.Lock()
    done = threading.Event()

    def watchdog():
        while not done.wait(1.0):
            with lock:
                if time.monotonic() > state["deadline"] and state["killed"] is None:
                    state["killed"] = state["what"]
                    kill_tree(proc)

    wd = threading.Thread(target=watchdog, daemon=True)
    wd.start()
    try:
        for line in proc.stdout:
            if line.startswith(RESULT_TAG):
                result = json.loads(line[len(RESULT_TAG):])
                line = f"[{test_id}] result received\n"
            else:
                m = re.search(rf"{PROGRESS} (load|call) (start|done)(.*)", line)
                if m:
                    with lock:
                        if m.group(2) == "start":
                            limit = LOAD_LIMIT_S if m.group(1) == "load" else CALL_LIMIT_S
                            state["deadline"] = time.monotonic() + limit
                            state["what"] = f"{m.group(1)}{m.group(3).rstrip()} exceeded {limit} s"
                        else:
                            state["deadline"] = time.monotonic() + CHILD_TIMEOUT_S
                            state["what"] = f"whole test > {CHILD_TIMEOUT_S} s"
            tail = (tail + [line.rstrip()])[-5:]
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
    except BaseException:
        # never leave a child blocked on a pipe nobody reads (the 2026-11-15 hang: the parent died on a
        # UnicodeEncodeError and the child froze at "Loading weights 0/625" writing its progress bar)
        done.set()
        kill_tree(proc)
        raise
    proc.wait()
    done.set()
    wd.join()
    if state["killed"] is not None:
        msg = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} WATCHDOG: {test_id}: {state['killed']}; " \
              "child process tree killed\n"
        print(msg, end="", flush=True)
        log.write(msg)
    if result is None:
        info = meta if meta is not None else {**TESTS[test_id], "model_id": MODELS[TESTS[test_id]["model"]]}
        if state["killed"] is not None:
            result = {"test": test_id, **info, "status": f"FAILED: {state['killed']}", "oom": False,
                      "error": f"killed by the watchdog: {state['killed']}; last lines: {' / '.join(tail)}"}
        else:
            result = {"test": test_id, **info, "status": "ERROR", "oom": False,
                      "error": f"child exited with code {proc.returncode}, no result: {' / '.join(tail)}"}
    result["wall_s"] = time.perf_counter() - t0
    result["exit_code"] = proc.returncode
    return result


def f2(v, fmt="{:.2f}"):
    return "-" if v is None else fmt.format(v)


def table(results):
    head = ["Test", "Model", "Prec", "Frames", "Mode", "VisTok", "Load s", "Gen/step s", "MaxAlloc GB",
            "MaxRes GB", "NVML pk GB", "JSON+keys", "OOM", "Status", "Trainable"]
    rows = []
    for r in results:
        timing = r.get("gen_s") if r["mode"] == "generate" else r.get("step_s")
        json_col = ("yes" if r.get("keys_ok") else "no") if r["mode"] == "generate" and "output" in r else "-"
        rows.append([r["test"], r["model"], r["precision"], str(r["frames"]), r["mode"],
                     str(r.get("visual_tokens", "-")), f2(r.get("load_s"), "{:.1f}"), f2(timing),
                     f2(r.get("max_alloc_gb")), f2(r.get("max_reserved_gb")), f2(r.get("nvml_peak_gb")),
                     json_col, "OOM" if r.get("oom") else "no", r["status"],
                     f"{r['trainable_params']:,}" if "trainable_params" in r else "-"])
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(head)]
    line = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
    return [line(head), "|-" + "-|-".join("-" * w for w in widths) + "-|"] + [line(r) for r in rows]


def run_parent(only):
    clip = load_clip_row()
    check_weights_cached(only)
    gpu, vers = gpu_info(), versions()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results = []
    with open(OUT_DIR / "vlm_smoke_test.log", "w", encoding="utf-8") as log:
        log.write(f"command: {' '.join([sys.executable, *sys.argv])}\nstarted: {started}\n")
        for t in only:
            msg = f"\n===== test {t}: {TESTS[t]} =====\n"
            print(msg, end="", flush=True)
            log.write(msg)
            results.append(run_test_process(t, log))
            log.flush()

    L = []
    L.append("Qwen3-VL smoke test (one synthetic clip; no real clip, no frozen test set)")
    L.append(f"started: {started}")
    L.append(f"GPU: {gpu['gpu']}, driver {gpu['driver']}, total memory {gpu['total_gb']:.2f} GB; "
             f"OVER_VRAM threshold {OVER_VRAM_FRAC * gpu['total_gb']:.2f} GB")
    L.append("versions: " + ", ".join(f"{k} {v}" for k, v in vers.items()))
    L.append(f"CUDA (torch build): {next((r['torch_cuda'] for r in results if 'torch_cuda' in r), '-')}")
    L.append(f"clip: {clip['rgb_path']} (fold {FOLD} val, {clip['scenario_id']}); {clip['width']}x{clip['height']}, "
             f"{clip['no_frames']} frames, {clip['duration']} s; t* {clip['t_star']} s")
    L.append(f"ground truth: type {clip['type']}, center ({clip['center_x']:.3f}, {clip['center_y']:.3f})")
    L.append(f"window: t* +/- {HALF_WINDOW_S} s, frames resized to longer side {LONG_SIDE} px (PIL bicubic)")
    L.append(f"generate: greedy, max_new_tokens {MAX_NEW_TOKENS}; QLoRA: r {LORA_R}, alpha {LORA_ALPHA}, "
             f"dropout {LORA_DROPOUT}, targets {LORA_TARGET}, AdamW lr {LR}, batch 1, grad ckpt on")
    L.append("4-bit: NF4, double quant, bf16 compute. Memory columns are peaks over the whole test (load included).")
    L.append("Prompt video format: <t seconds><|vision_start|><|video_pad|>*n<|vision_end|> per 2-frame group "
             "(the template's outer start/end pair is removed, as in the reference Qwen3-VL processor).")
    L.append("JSON+keys: output parses as JSON (a ``` fence is allowed) and has collision_type, x, y.")
    L.append("")
    L.extend(table(results))
    L.append("")
    L.append("Per test:")
    for r in results:
        L.append(f" {r['test']}: status {r['status']}, frames {r.get('frame_indices')} "
                 f"(model size {r.get('frame_size_model')}, grid_thw {r.get('video_grid_thw')}), "
                 f"input tokens {r.get('input_tokens', '-')}, NVML baseline {f2(r.get('nvml_baseline_gb'))} GB, "
                 f"alloc after load {f2(r.get('alloc_after_load_gb'))} GB")
        if "linear4bit_modules" in r:
            L.append(f"    Linear4bit modules: {r['linear4bit_modules']}")
        if r["mode"] == "generate" and "output" in r:
            L.append(f"    new tokens {r['new_tokens']}; json_ok {r['json_ok']}, keys_ok {r['keys_ok']}, "
                     f"values_ok {r['values_ok']}")
            L.append(f"    raw output: {r['output']!r}")
        if "trainable_params" in r:
            L.append(f"    trainable {r['trainable_params']:,} of {r['total_params']:,} params "
                     f"({r['lora_modules']} LoRA modules); target tokens {r['target_tokens']}; "
                     f"loss {r['loss']:.4f}; step {r['step_s']:.2f} s (first step, includes warm-up)")
        if "error" in r:
            L.append(f"    error: {r['error']}")

    (OUT_DIR / "vlm_smoke_test.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    payload = {"started": started, **gpu, "versions": vers, "clip": clip, "half_window_s": HALF_WINDOW_S,
               "long_side": LONG_SIDE, "prompt": PROMPT, "max_new_tokens": MAX_NEW_TOKENS,
               "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "target": LORA_TARGET, "lr": LR},
               "over_vram_frac": OVER_VRAM_FRAC, "results": results}
    (OUT_DIR / "vlm_smoke_test.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\n" + "\n".join(table(results)))
    print(f"\nsaved {OUT_DIR / 'vlm_smoke_test.txt'}, .json and .log")


# ---------------------------------------------------------------- --stage-test (VLM-stage planning, weeks 6-7)

def stage_prepare():
    """The STAGE_CLIPS rows with their saved out-of-fold t_hat (C_cctv_seed0, frozen post-processing)."""
    with open(STAGE_PRED, newline="", encoding="utf-8") as f:
        pred = {r["rgb_path"].replace("\\", "/"): r for r in csv.DictReader(f)}
    clips = []
    for p in STAGE_CLIPS:
        c = load_clip_row(p, None)
        if p not in pred:
            fail(f"{p} has no prediction in {STAGE_PRED}")
        c["t_hat"] = float(pred[p]["t_hat"])
        if int(pred[p]["fold"]) != c["val_fold"]:
            fail(f"{p}: the saved prediction is not out-of-fold")
        clips.append(c)
    return clips


def stage_input(clip, inp):
    """(frame indices, video fps, frames, prompt) of one input mode."""
    if inp == "narrow16":
        idx, fps = frame_indices(clip, NARROW_FRAMES, centre=clip["t_hat"])
        return idx, fps, read_frames(clip, idx), PROMPT
    f = STAGE_INPUTS[inp]
    idx, fps = scan_indices(clip, f)
    return idx, fps, read_frames(clip, idx, [str(k) for k in range(len(idx))]), PROMPT_FULL_DRAFT.format(n=len(idx), fps=f)


def run_stage_child(cfg, inp):
    import torch
    import transformers
    from transformers import AutoProcessor

    model_key, precision = STAGE_CONFIGS[cfg]
    spec = {"model": model_key, "precision": precision}
    res = {"test": f"{cfg} {inp}", "config": cfg, "input": inp, **spec, "model_id": MODELS[model_key],
           "status": None, "oom": False, "calls": []}
    stamp(f"{cfg} {inp}: decoding frames of {len(STAGE_CLIPS)} clips (CPU)")
    prepared = [(c, *stage_input(c, inp)) for c in stage_prepare()]
    stamp(f"{cfg} {inp}: frames ready ({', '.join(str(len(p[1])) for p in prepared)} frames)")
    if not torch.cuda.is_available():
        fail("CUDA is not available")
    device = torch.device("cuda", 0)
    nvml = NvmlPeak().start()
    total = torch.cuda.get_device_properties(device).total_memory
    model = inputs = out = None
    proc_alloc = proc_res = 0.0
    current = None
    try:
        torch.manual_seed(0)
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        current = "model load"
        stamp(f"{PROGRESS} load start {cfg} ({MODELS[model_key]}, {precision})")
        processor = AutoProcessor.from_pretrained(MODELS[model_key])
        stamp(f"{cfg}: processor loaded, loading weights")
        model = load_model(spec)
        model.eval()
        torch.cuda.synchronize()
        res["load_s"] = time.perf_counter() - t0
        res["alloc_after_load_gb"] = torch.cuda.memory_allocated(device) / GB
        stamp(f"{PROGRESS} load done {cfg}: {res['load_s']:.1f} s, allocated {res['alloc_after_load_gb']:.2f} GB")
        proc_alloc = torch.cuda.max_memory_allocated(device) / GB
        proc_res = torch.cuda.max_memory_reserved(device) / GB
        for clip, idx, fps, frames, prompt in prepared:
            for rep in range(1, STAGE_REPEATS + 1):
                current = f"{clip['rgb_path']} call {rep}"
                stamp(f"{PROGRESS} call start {Path(clip['rgb_path']).stem} call {rep} ({len(idx)} frames)")
                call = {"clip": clip["rgb_path"], "repeat": rep, "n_frames": len(idx), "frame_indices": idx,
                        "frame_size_in": [int(frames.shape[2]), int(frames.shape[1])], "prompt": prompt}
                inputs, _, _ = build_inputs(processor, frames, idx, fps, clip, prompt_text_in=prompt)
                ids = inputs["input_ids"][0]
                call.update(video_grid_thw=inputs["video_grid_thw"][0].tolist(),
                            visual_tokens=int((ids == processor.video_token_id).sum()), input_tokens=int(ids.numel()))
                inputs = inputs.to(device)
                inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(torch.bfloat16)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
                t0 = time.perf_counter()
                with torch.inference_mode():
                    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                         temperature=None, top_p=None, top_k=None)
                torch.cuda.synchronize()
                call["gen_s"] = time.perf_counter() - t0
                call["peak_alloc_gb"] = torch.cuda.max_memory_allocated(device) / GB
                call["peak_reserved_gb"] = torch.cuda.max_memory_reserved(device) / GB
                proc_alloc, proc_res = max(proc_alloc, call["peak_alloc_gb"]), max(proc_res, call["peak_reserved_gb"])
                new = out[0, ids.numel():]
                call["new_tokens"] = int(new.numel())
                call["output"] = processor.batch_decode(new[None], skip_special_tokens=True)[0]
                call.update(parse_output(call["output"]) if inp == "narrow16"
                            else parse_full_output(call["output"], len(idx)))
                res["calls"].append(call)
                stamp(f"{PROGRESS} call done {Path(clip['rgb_path']).stem} call {rep}: {call['gen_s']:.2f} s, "
                      f"peak allocated {call['peak_alloc_gb']:.2f} GB, valid {call['values_ok']}")
                inputs = out = None
        res["status"] = "OK"
    except torch.cuda.OutOfMemoryError as e:
        res.update(status="OOM", oom=True, error=f"{current}: {str(e).splitlines()[0]}")
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        res.update(status="OOM", oom=True, error=f"{current}: {str(e).splitlines()[0]}")
    finally:
        torch.cuda.synchronize()
        res["max_alloc_gb"] = max(proc_alloc, torch.cuda.max_memory_allocated(device) / GB)
        res["max_reserved_gb"] = max(proc_res, torch.cuda.max_memory_reserved(device) / GB)
        nvml.stop()
        res["nvml_baseline_gb"] = nvml.baseline / GB
        res["nvml_peak_gb"] = nvml.peak / GB
        res["total_gb"] = total / GB
        del model, inputs, out
        gc.collect()
        torch.cuda.empty_cache()
    if res["status"] == "OK" and res["max_reserved_gb"] > OVER_VRAM_FRAC * res["total_gb"]:
        res["status"] = "OVER_VRAM"
    res["transformers"] = transformers.__version__
    print(RESULT_TAG + json.dumps(res), flush=True)


def stage_table(results):
    head = ["Config", "Input", "Clip", "Frames", "VisTok", "InTok", "Load s", "Call1 s", "Call2 s", "PkAlloc GB",
            "MaxRes GB", "NVML pk GB", "JSON", "Valid", "Status"]
    rows = []
    for r in results:
        if not r.get("calls"):
            rows.append([r["config"], r["input"], "-", "-", "-", "-", f2(r.get("load_s"), "{:.1f}"), "-", "-", "-",
                         f2(r.get("max_reserved_gb")), f2(r.get("nvml_peak_gb")), "-", "-", r["status"]])
            continue
        for clip in dict.fromkeys(c["clip"] for c in r["calls"]):
            cs = [c for c in r["calls"] if c["clip"] == clip]
            t = {c["repeat"]: c["gen_s"] for c in cs}
            rows.append([r["config"], r["input"], Path(clip).stem, str(cs[0]["n_frames"]), str(cs[0]["visual_tokens"]),
                         str(cs[0]["input_tokens"]), f2(r.get("load_s"), "{:.1f}"), f2(t.get(1)), f2(t.get(2)),
                         f2(max(c["peak_alloc_gb"] for c in cs)), f2(r.get("max_reserved_gb")), f2(r.get("nvml_peak_gb")),
                         "/".join("yes" if c["json_ok"] else "no" for c in cs),
                         "/".join("yes" if c["values_ok"] else "no" for c in cs), r["status"]])
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(head)]
    line = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"  # noqa: E731
    return [line(head), "|-" + "-|-".join("-" * w for w in widths) + "-|"] + [line(r) for r in rows]


def run_stage_parent():
    from huggingface_hub import snapshot_download
    from PIL import Image

    clips = stage_prepare()
    STAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    gpu, vers = gpu_info(), versions()
    not_run = {}
    for cfg, (mk, _) in STAGE_CONFIGS.items():
        try:
            snapshot_download(MODELS[mk], local_files_only=True)
        except Exception as e:  # noqa: BLE001 - any failure means the weights are not usable offline
            not_run[cfg] = f"not run: {MODELS[mk]} weights not fully in the Hugging Face cache ({type(e).__name__})"
    idx, _ = scan_indices(clips[1], 1.0)
    j = len(idx) // 2
    Image.fromarray(read_frames(clips[1], [idx[j]], [str(j)])[0]).save(STAGE_OUT_DIR / "vlm_stage_test_frame_example.png")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results = []
    with open(STAGE_OUT_DIR / "vlm_stage_test.log", "w", encoding="utf-8") as log:
        log.write(f"command: {' '.join([sys.executable, *sys.argv])}\nstarted: {started}\n")
        for cfg in STAGE_CONFIGS:
            for inp in STAGE_INPUTS:
                tid = f"{cfg} {inp}"
                if cfg in not_run:
                    results.append({"test": tid, "config": cfg, "input": inp, "status": not_run[cfg], "calls": []})
                    continue
                msg = f"\n===== {datetime.now(timezone.utc).isoformat(timespec='seconds')} stage test {tid} =====\n"
                print(msg, end="", flush=True)
                log.write(msg)
                results.append(run_test_process(tid, log, ["--stage-child", cfg, inp],
                                                meta={"config": cfg, "input": inp, "calls": []}))
                log.flush()

    L = ["Zero-shot VLM stage planning test (memory, speed, output format; NO accuracy; 2 synthetic clips; no real clip "
         "or real label opened)",
         f"started: {started}",
         f"GPU: {gpu['gpu']}, driver {gpu['driver']}, total memory {gpu['total_gb']:.2f} GB; OVER_VRAM threshold "
         f"{OVER_VRAM_FRAC * gpu['total_gb']:.2f} GB",
         "versions: " + ", ".join(f"{k} {v}" for k, v in vers.items())]
    for c in clips:
        L.append(f"clip: {c['rgb_path']} (val fold {c['val_fold']}); {c['width']}x{c['height']}, {c['no_frames']} frames, "
                 f"{c['duration']} s; narrow window centre t_hat {c['t_hat']:.3f} s (saved out-of-fold C_cctv_seed0 "
                 "frozen prediction)")
    L += [f"frames resized to longer side {LONG_SIDE} px (PIL bicubic); generate greedy, max_new_tokens {MAX_NEW_TOKENS}; "
          "4-bit = NF4, double quant, bf16 compute",
          f"narrow16: {NARROW_FRAMES} frames evenly over t_hat +- {HALF_WINDOW_S} s; prompt: {PROMPT}",
          f"full1fps / full2fps: whole clip at 1 / 2 fps, frame number k (0 = first) drawn top-left ({LABEL_FONT_PX} px, "
          f"white on black); prompt (DRAFT, not frozen): {PROMPT_FULL_DRAFT}",
          "Qwen3-VL merges 2 consecutive frames into one temporal token group (timestamp = mean of the two); an odd "
          "frame count repeats the last frame.",
          f"Offline (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1), children run with python -u. Limits per combination: "
          f"model load {LOAD_LIMIT_S} s, each call {CALL_LIMIT_S} s; a combination over a limit is killed and FAILED.",
          f"Each clip is run {STAGE_REPEATS} times per process: Call1 includes CUDA warm-up, Call2 is the warm time. "
          "PkAlloc = peak allocated during the calls of that clip; MaxRes = peak reserved over the process (load "
          "included); NVML pk = device-wide peak. JSON / Valid per call (narrow: collision_type in the 5 types and x, y "
          "in [0, 1]; full: integer frame in [0, n-1]). No answer is compared with a label.",
          ""] + stage_table(results) + ["", "Not run:"]
    L += [f" {k}: {v}" for k, v in {**STAGE_NOT_RUN, **{f'{c} (all inputs)': v for c, v in not_run.items()}}.items()]
    L += ["", "Raw outputs (call 2; call 1 is the same input):"]
    for r in results:
        for c in r.get("calls", []):
            if c["repeat"] == STAGE_REPEATS:
                L.append(f" {r['test']} | {Path(c['clip']).stem}: {c['output']!r}")
        if "error" in r:
            L.append(f" {r['test']}: error: {r['error']}")
    (STAGE_OUT_DIR / "vlm_stage_test.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    payload = {"started": started, **gpu, "versions": vers, "clips": clips, "prompt_narrow": PROMPT,
               "prompt_full_draft": PROMPT_FULL_DRAFT, "long_side": LONG_SIDE, "max_new_tokens": MAX_NEW_TOKENS,
               "not_run": {**STAGE_NOT_RUN, **not_run}, "results": results}
    (STAGE_OUT_DIR / "vlm_stage_test.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\n" + "\n".join(L))
    print(f"\nsaved {STAGE_OUT_DIR / 'vlm_stage_test.txt'}, .json, .log and vlm_stage_test_frame_example.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", choices=list(TESTS), default=list(TESTS))
    ap.add_argument("--child", choices=list(TESTS), help=argparse.SUPPRESS)
    ap.add_argument("--stage-test", action="store_true", help="VLM-stage planning test (see module doc)")
    ap.add_argument("--stage-child", nargs=2, metavar=("CONFIG", "INPUT"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    # Console output redirected to a file uses the Windows code page (cp1252), which cannot encode the progress-bar
    # blocks of the weight loader; UTF-8 with replacement keeps the parent from crashing on a child's line.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    if args.stage_child:
        cfg, inp = args.stage_child
        if cfg not in STAGE_CONFIGS or inp not in STAGE_INPUTS:
            fail(f"unknown stage child {cfg} {inp}")
        run_stage_child(cfg, inp)
    elif args.stage_test:
        run_stage_parent()
    elif args.child:
        run_child(args.child)
    else:
        run_parent(args.only)


if __name__ == "__main__":
    main()
