"""GPU memory / speed test for VideoMAE-base on 16 x 224 x 224 clips (random inputs only).

Reads NO dataset file. Inputs are random tensors (torch.manual_seed(0)).

Model: MCG-NJU/videomae-base-finetuned-kinetics with the classifier replaced by
nn.Linear(hidden_size, 1). bf16 autocast on CUDA.

1. Training step, grad_ckpt False / True, batch 8 -> 4 -> 2 -> 1 (stop at the first FITS):
   AdamW(lr=3e-5, wd=0.05), BCEWithLogitsLoss, 2 warm-up + 5 timed steps of
   forward + backward + clip_grad_norm(1.0) + step + zero_grad.
2. Inference (eval, no_grad, bf16), batch 32 -> 16 -> 8 (stop at the first FITS).
Status: OOM (torch.cuda.OutOfMemoryError), OVER_VRAM (peak reserved > 97% of total GPU
memory: Windows may silently spill into shared system memory), else FITS.

Outputs: gpu_check/videomae_memory_test.txt and .json

Usage:
    .venv-ml\\Scripts\\python.exe gpu_memory_test_videomae.py
"""

import gc
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import transformers
from transformers import VideoMAEForVideoClassification

MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"
FRAMES, SIZE = 16, 224
TRAIN_BATCHES = [8, 4, 2, 1]
INFER_BATCHES = [32, 16, 8]
WARMUP, TIMED = 2, 5
INFER_TIMED = 5
OVER_VRAM_FRAC = 0.97
LR, WD, CLIP = 3e-5, 0.05, 1.0
TRAIN_WINDOWS = {0: 19817, 1: 19903, 2: 21114}  # labelled training windows per fold
VAL_WINDOWS = {0: 25988, 1: 25041, 2: 19701}  # all validation windows per fold
OUT_DIR = Path(__file__).resolve().parent / "gpu_check"
GB = 2 ** 30


class Report:
    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text, flush=True)
        self.lines.append(str(text))


def free():
    gc.collect()
    torch.cuda.empty_cache()


def build_model(device):
    model = VideoMAEForVideoClassification.from_pretrained(MODEL_ID)
    model.classifier = nn.Linear(model.config.hidden_size, 1)
    return model.to(device)


def status_of(peak_reserved, total):
    return "OVER_VRAM" if peak_reserved > OVER_VRAM_FRAC * total else "FITS"


def train_setting(batch, grad_ckpt, device, total):
    model = opt = x = y = loss = logits = None
    res = {"phase": "train", "grad_ckpt": grad_ckpt, "batch": batch}
    try:
        free()
        torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(0)
        model = build_model(device)
        if grad_ckpt:
            model.gradient_checkpointing_enable()
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        crit = nn.BCEWithLogitsLoss()
        x = torch.randn(batch, FRAMES, 3, SIZE, SIZE, device=device)
        y = torch.randint(0, 2, (batch,), device=device).float()

        def step():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(pixel_values=x).logits.squeeze(-1)
            l = crit(out.float(), y)
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            opt.zero_grad(set_to_none=True)

        for _ in range(WARMUP):
            step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(TIMED):
            step()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / TIMED
        res.update(
            peak_alloc_gb=torch.cuda.max_memory_allocated(device) / GB,
            peak_reserved_gb=torch.cuda.max_memory_reserved(device) / GB,
            sec_per_step=dt,
            windows_per_sec=batch / dt,
        )
        res["status"] = status_of(torch.cuda.max_memory_reserved(device), total)
    except torch.cuda.OutOfMemoryError:
        res.update(status="OOM", peak_alloc_gb=torch.cuda.max_memory_allocated(device) / GB,
                   peak_reserved_gb=torch.cuda.max_memory_reserved(device) / GB)
    finally:
        del model, opt, x, y, loss, logits
        free()
    return res


def infer_setting(batch, device, total):
    model = x = None
    res = {"phase": "infer", "grad_ckpt": False, "batch": batch}
    try:
        free()
        torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(0)
        model = build_model(device).eval()
        x = torch.randn(batch, FRAMES, 3, SIZE, SIZE, device=device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(WARMUP):
                model(pixel_values=x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(INFER_TIMED):
                model(pixel_values=x)
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / INFER_TIMED
        res.update(
            peak_alloc_gb=torch.cuda.max_memory_allocated(device) / GB,
            peak_reserved_gb=torch.cuda.max_memory_reserved(device) / GB,
            sec_per_step=dt,
            windows_per_sec=batch / dt,
        )
        res["status"] = status_of(torch.cuda.max_memory_reserved(device), total)
    except torch.cuda.OutOfMemoryError:
        res.update(status="OOM", peak_alloc_gb=torch.cuda.max_memory_allocated(device) / GB,
                   peak_reserved_gb=torch.cuda.max_memory_reserved(device) / GB)
    finally:
        del model, x
        free()
    return res


def fmt(res):
    s = f"  batch {res['batch']:>2}  {res['status']:<9}  peak alloc {res['peak_alloc_gb']:5.2f} GB  " \
        f"peak reserved {res['peak_reserved_gb']:5.2f} GB"
    if "sec_per_step" in res:
        s += f"  {res['sec_per_step']:.3f} s/step  {res['windows_per_sec']:6.2f} windows/s"
    return s


def main():
    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA is not available")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    total = props.total_memory
    torch.manual_seed(0)

    r = Report()
    r("VideoMAE GPU memory test (random inputs, no dataset)")
    r(f"time: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    r(f"GPU: {props.name}, total memory {total / GB:.2f} GB; OVER_VRAM threshold {OVER_VRAM_FRAC * total / GB:.2f} GB")
    r(f"torch {torch.__version__}, CUDA {torch.version.cuda}, transformers {transformers.__version__}, "
      f"python {platform.python_version()}")
    r(f"model: {MODEL_ID}, classifier -> Linear(hidden, 1); input (B, {FRAMES}, 3, {SIZE}, {SIZE}); bf16 autocast")

    probe = build_model(torch.device("cpu"))
    n_params = sum(p.numel() for p in probe.parameters())
    n_train = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    r(f"parameters: {n_params:,} (trainable {n_train:,}); hidden_size {probe.config.hidden_size}")
    del probe
    free()

    results = []
    largest = {}
    r("")
    r(f"1. Training step (AdamW lr={LR} wd={WD}, BCEWithLogits, clip {CLIP}; {WARMUP} warm-up + {TIMED} timed steps)")
    for ckpt in (False, True):
        r(f" grad_ckpt = {ckpt}")
        largest[ckpt] = None
        for b in TRAIN_BATCHES:
            res = train_setting(b, ckpt, device, total)
            results.append(res)
            r(fmt(res))
            if res["status"] == "FITS":
                largest[ckpt] = res
                break

    r("")
    r(f"2. Inference (eval, no_grad, bf16; {WARMUP} warm-up + {INFER_TIMED} timed passes)")
    infer = None
    for b in INFER_BATCHES:
        res = infer_setting(b, device, total)
        results.append(res)
        r(fmt(res))
        if res["status"] == "FITS":
            infer = res
            break

    r("")
    r("3. Largest fitting training batch")
    for ckpt in (False, True):
        lr_ = largest[ckpt]
        r(f"   grad_ckpt {ckpt!s:<5}: " + ("none" if lr_ is None else
                                          f"batch {lr_['batch']} ({lr_['windows_per_sec']:.2f} windows/s, "
                                          f"peak reserved {lr_['peak_reserved_gb']:.2f} GB)"))
    suggested = largest[False] or largest[True]
    if suggested is None:
        r("   suggested: none - no training setting fits")
    else:
        r(f"   suggested: batch {suggested['batch']}, grad_ckpt {suggested['grad_ckpt']}")

    estimates = {}
    r("")
    r("4. Time estimates (data loading assumed not to be the bottleneck)")
    if suggested is None or infer is None:
        r("   not available (no fitting training or inference setting)")
    else:
        r(f"   training {suggested['windows_per_sec']:.2f} windows/s (batch {suggested['batch']}, "
          f"grad_ckpt {suggested['grad_ckpt']}); inference {infer['windows_per_sec']:.2f} windows/s (batch {infer['batch']})")
        r(f"   {'fold':<5} {'train windows':>13} {'min/train epoch':>16} {'val windows':>12} {'min/val pass':>13}")
        for k in TRAIN_WINDOWS:
            tm = TRAIN_WINDOWS[k] / suggested["windows_per_sec"] / 60
            vm = VAL_WINDOWS[k] / infer["windows_per_sec"] / 60
            estimates[k] = {"train_windows": TRAIN_WINDOWS[k], "min_per_train_epoch": tm,
                            "val_windows": VAL_WINDOWS[k], "min_per_val_pass": vm}
            r(f"   {k:<5} {TRAIN_WINDOWS[k]:>13} {tm:>16.1f} {VAL_WINDOWS[k]:>12} {vm:>13.1f}")

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "videomae_memory_test.txt").write_text("\n".join(r.lines) + "\n", encoding="ascii")
    payload = {
        "gpu": props.name, "total_memory_gb": total / GB, "over_vram_frac": OVER_VRAM_FRAC,
        "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
        "model": MODEL_ID, "n_params": n_params, "input": [FRAMES, 3, SIZE, SIZE], "amp": "bf16",
        "results": results,
        "largest_fitting": {str(k): (v["batch"] if v else None) for k, v in largest.items()},
        "suggested": None if suggested is None else {"batch": suggested["batch"], "grad_ckpt": suggested["grad_ckpt"]},
        "inference": None if infer is None else {"batch": infer["batch"], "windows_per_sec": infer["windows_per_sec"]},
        "estimates": estimates,
    }
    (OUT_DIR / "videomae_memory_test.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    print(f"\nsaved {OUT_DIR / 'videomae_memory_test.txt'} and .json")


if __name__ == "__main__":
    main()
