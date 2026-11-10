"""Train a window-level accident classifier (VideoMAE-base) on synthetic data, one fold.

SYNTHETIC DATA ONLY: never opens real videos, metadata-real.csv or real_test_set.csv.

Data:   synthetic_window_labels.csv (hash-checked against split_manifest.json) and the
        frame cache cache/frames224_v1/<clip>.npz (build_frame_cache.py). Window frames =
        grid positions 8*t_start .. 8*t_start+15, normalised with the VideoMAE processor
        mean/std (frames are already 224x224 full-frame squashes; no further resize/crop).
Fold k: validation = ALL windows with val_fold == k. Training windows (val_fold != k) depend on
        the config's label scheme:
          B: windows with label_B 0/1, target label_B (label_B = -1 is not used);
          C: every window, soft target from column weight_C_text (the evaluation document's text
             rule: label_B 0 windows 0, windows containing t* 1, Gaussian decay sigma 0.5 s).
Model:  VideoMAE-base (Kinetics fine-tuned), classifier -> Linear(hidden, 1), bf16 autocast.
        Loss per window = type weight * BCEWithLogits(logit, target, pos_weight), mean over the
        micro-batch; pos_weight = sum(1 - target) / sum(target) (= n0/n1 for B) and collision-type
        weights (sqrt inverse frequency, clipped, mean 1) from the windows that enter the loss.
        val_loss is BCE on the label_B 0/1 validation windows for both schemes. AdamW with layer-wise lr decay; linear warmup,
        then cosine to 0. Augmentation profile "standard" (hflip) or "cctv" (hflip + frame-rate
        drop, blur, downscale, colour, JPEG; optional night_guard: windows darker than dark_luma are
        never darkened by the colour shift) or "cctv_v2" (each degradation p 0.3, at most 2 per window,
        1 for dark windows, milder ranges; fixed in CCTV_V2); validation is never augmented.
Val:    per clip, sort by t_centre, 3-window moving average, t_hat = t_centre at the peak,
        confidence = peak - max smoothed score outside +-2 windows (evaluation document,
        Section 6.4); T at sigma_t = 0.5 / 1 / 2 s with accs_scorer.temporal.
Validation checks: once per epoch (v1, "patience" in epochs) or every validate_every_updates
updates and at the end of training ("patience_checks"); one metrics.csv row per check. Early
stopping on val T@1 s. Checkpoints: last.pt (every epoch), best.pt (weights of the best check),
step_latest.pt (every step_ckpt_every updates, after each check every N updates, on Ctrl+C, at
--max-hours and when an out-of-memory error cannot be recovered). All hyperparameters come from
the config file.

Usage (training environment):
    .venv-ml\\Scripts\\python.exe train_classifier.py --plan
    .venv-ml\\Scripts\\python.exe train_classifier.py --config configs\\videomae_v1_cctv.json --fold 0 --preview-aug 6
    .venv-ml\\Scripts\\python.exe train_classifier.py --fold 0 --smoke-test
    .venv-ml\\Scripts\\python.exe train_classifier.py --fold 0 [--run-name NAME] [--seed S] [--max-hours H] [--resume]
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import OrderedDict  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable, Iterator, NoReturn  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset, Sampler  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from accs_scorer import temporal, temporal_per_clip  # noqa: E402
from build_frame_cache import read_cached_frame  # noqa: E402

LABELS_CSV = REPO_ROOT / "synthetic_window_labels.csv"
SPLIT_CSV = REPO_ROOT / "synthetic_split.csv"
MANIFEST = REPO_ROOT / "split_manifest.json"
CACHE_DIR = REPO_ROOT / "cache" / "frames224_v1"
CACHE_MANIFEST = CACHE_DIR / "cache_manifest.json"
RUNS_DIR = REPO_ROOT / "runs"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "videomae_v1.json"
SPLIT_VERSION = "v2_rebalanced"
SIGMAS_T = (0.5, 1.0, 2.0)
SMOKE = {"train_clips": 20, "val_clips": 10, "max_epochs": 2, "step_ckpt_every": 4}
OOM_CHUNKS = (4, 2)
TYPES = ("head-on", "rear-end", "sideswipe", "single", "t-bone")  # collision type = folder in rgb_path
AUG_PREVIEW_DIR = RUNS_DIR / "aug_preview"
PREVIEW_FRAMES = (0, 5, 10, 15)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def fail(msg: str) -> NoReturn:
    sys.exit(f"ERROR: {msg}")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def stem_of(p: str) -> str:
    return Path(str(p).replace("\\", "/")).stem


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_info() -> tuple[str, bool | None]:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True,
                                    text=True, check=True).stdout.strip())
        return head, dirty
    except Exception as e:  # git missing or not a repo
        return f"unavailable ({type(e).__name__})", None


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
"""All hyperparameters are read from a JSON config file; the CLI only selects the run."""

REQUIRED_KEYS = (
    "model_id", "frames", "model_fps", "micro_batch", "grad_accum", "val_batch", "num_workers",
    "lru_size", "amp", "optimizer", "betas", "weight_decay", "base_lr", "lr_scaling", "layer_decay",
    "warmup_fraction", "schedule", "max_epochs", "grad_clip", "drop_path", "pos_weight", "type_weight",
    "augmentation", "hflip_p", "cctv_aug", "monitor", "min_delta", "step_ckpt_every", "log_every",
    "smooth_window", "confidence_exclude", "gpu_pause_temp_c", "gpu_resume_temp_c",
)
# label: "label" (v1, must be "label_B") or "label_scheme" ("B" | "C");
# validation: "patience" (epochs, validate once per epoch; v1) or "validate_every_updates" + "patience_checks"
CHOICE_KEYS = ("label", "label_scheme", "patience", "validate_every_updates", "patience_checks")
# optional; absent = previous behaviour (so the canonical hash of configs without it is unchanged)
#   night_guard {"dark_luma": Y}: cctv colour shift draws the brightness factor from [NIGHT_GUARD_MIN_BRIGHTNESS,
#   brightness max] instead of the full range when the window's mean luma before the colour shift is below Y
OPTIONAL_KEYS = ("night_guard",)
NIGHT_GUARD_MIN_BRIGHTNESS = 1.0  # never darken a window that is already dark
CCTV_AUG_KEYS = ("p", "fps_drop", "blur_sigma", "downscale", "brightness", "contrast", "saturation", "hue",
                 "grayscale_p", "jpeg_quality")
# values this script implements; anything else in the config would be silently ignored
FIXED_VALUES = {"amp": "bf16", "optimizer": "AdamW", "monitor": "val_T_1",
                "schedule": "cosine to 0", "lr_scaling": "base_lr * effective_batch / 256"}
TRAIN_RATE, VAL_RATE = 15, 51  # windows/s used for the --plan time estimates


def load_config(path: Path) -> tuple[dict, str]:
    """Read and check the config; returns (config, sha256 of its canonical JSON)."""
    if not path.is_file():
        fail(f"config file not found: {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    unknown = [k for k in cfg if k not in REQUIRED_KEYS + CHOICE_KEYS + OPTIONAL_KEYS]
    if missing or unknown:
        fail(f"{path.name}: missing keys {missing}, unknown keys {unknown}")
    for k, v in FIXED_VALUES.items():
        if cfg[k] != v:
            fail(f"{path.name}: {k} must be {v!r} (got {cfg[k]!r})")
    if ("label" in cfg) == ("label_scheme" in cfg):
        fail(f"{path.name}: give exactly one of \"label\" (v1, \"label_B\") and \"label_scheme\" (\"B\" | \"C\")")
    if cfg.get("label", "label_B") != "label_B" or cfg.get("label_scheme", "B") not in ("B", "C"):
        fail(f"{path.name}: label must be \"label_B\" / label_scheme must be \"B\" or \"C\"")
    v = cfg.get("validate_every_updates")
    if v is None:
        if "patience" not in cfg or cfg.get("patience_checks") is not None:
            fail(f"{path.name}: validating once per epoch needs \"patience\" (epochs) and no patience_checks")
    elif not (isinstance(v, int) and v > 0 and isinstance(cfg.get("patience_checks"), int) and "patience" not in cfg):
        fail(f"{path.name}: validate_every_updates must be a positive integer, with patience_checks (an integer) "
             "instead of patience")
    if cfg["gpu_resume_temp_c"] >= cfg["gpu_pause_temp_c"]:
        fail(f"{path.name}: gpu_resume_temp_c must be below gpu_pause_temp_c")
    if not (cfg["pos_weight"] == "auto" or isinstance(cfg["pos_weight"], (int, float))):
        fail(f"{path.name}: pos_weight must be \"auto\" or a number")
    tw = cfg["type_weight"]
    if set(tw) != {"scheme", "clip", "normalise_mean"} or tw["scheme"] != "sqrt_inverse_frequency":
        fail(f"{path.name}: type_weight must have scheme sqrt_inverse_frequency, clip and normalise_mean")
    if cfg["augmentation"] not in ("standard", "cctv", "cctv_v2"):
        fail(f"{path.name}: augmentation must be \"standard\", \"cctv\" or \"cctv_v2\"")
    if cfg["augmentation"] == "cctv_v2" and "night_guard" not in cfg:
        fail(f"{path.name}: augmentation \"cctv_v2\" needs night_guard {{\"dark_luma\": Y}} (its dark-window rules)")
    if set(cfg["cctv_aug"]) != set(CCTV_AUG_KEYS):
        fail(f"{path.name}: cctv_aug must have exactly the keys {CCTV_AUG_KEYS}")
    if "night_guard" in cfg:
        ng = cfg["night_guard"]
        if cfg["augmentation"] not in ("cctv", "cctv_v2"):
            fail(f"{path.name}: night_guard only applies to augmentation \"cctv\" or \"cctv_v2\"")
        if not (isinstance(ng, dict) and set(ng) == {"dark_luma"} and isinstance(ng["dark_luma"], (int, float))
                and 0 < ng["dark_luma"] <= 255):
            fail(f"{path.name}: night_guard must be {{\"dark_luma\": Y}} with 0 < Y <= 255")
        if cfg["augmentation"] == "cctv" and cfg["cctv_aug"]["brightness"][1] <= NIGHT_GUARD_MIN_BRIGHTNESS:
            fail(f"{path.name}: night_guard needs a brightness maximum above {NIGHT_GUARD_MIN_BRIGHTNESS}")
    return cfg, canonical_hash(cfg)


def dark_luma_of(cfg: dict) -> float | None:
    """night_guard dark_luma, or None without the guard (previous behaviour)."""
    ng = cfg.get("night_guard")
    return None if ng is None else float(ng["dark_luma"])


def label_scheme(cfg: dict) -> str:
    return cfg.get("label_scheme", "B")


def check_schedule(cfg: dict) -> tuple[int | None, int]:
    """(validate_every_updates or None for once per epoch, patience counted in checks)."""
    v = cfg.get("validate_every_updates")
    return (None, cfg["patience"]) if v is None else (v, cfg["patience_checks"])


def n_checks(total_updates: int, max_epochs: int, every: int | None) -> int:
    """Validation checks of a full run: every `every` updates plus one at the end (or one per epoch)."""
    return max_epochs if every is None else math.ceil(total_updates / every)


def scaled_lr(cfg: dict) -> float:
    """base_lr * effective_batch / 256 (the lr of the head; lower layers get layer decay)."""
    return cfg["base_lr"] * cfg["micro_batch"] * cfg["grad_accum"] / 256


def schedule_sizes(n_train_windows: int, cfg: dict, max_epochs: int) -> dict:
    """Micro-batches and optimizer updates per epoch, total updates and warmup updates."""
    n_batches = math.ceil(n_train_windows / cfg["micro_batch"])
    updates = math.ceil(n_batches / cfg["grad_accum"])
    total = updates * max_epochs
    return {"micro_batches_per_epoch": n_batches, "updates_per_epoch": updates, "total_updates": total,
            "warmup_updates": round(cfg["warmup_fraction"] * total)}


def loss_weights(train_rows: pd.DataFrame, cfg: dict) -> dict:
    """pos_weight = sum(1 - target) / sum(target) over the training windows (= n_label0 / n_label1 for
    scheme B) and collision-type weights w_c = sqrt(N / (5 n_c)), clipped, then rescaled so the mean weight
    over all training windows is normalise_mean (types absent: no weight)."""
    lab = train_rows["label_B"]
    n0, n1 = int((lab == 0).sum()), int((lab == 1).sum())
    tsum = float(train_rows["target"].sum())
    nsum = float((1.0 - train_rows["target"]).sum())
    if tsum == 0:
        fail("no positive training targets")
    pos_weight = nsum / tsum if cfg["pos_weight"] == "auto" else float(cfg["pos_weight"])
    tw = cfg["type_weight"]
    lo, hi = tw["clip"]
    n = len(train_rows)
    counts = {t: int((train_rows["ctype"] == t).sum()) for t in TYPES}
    clipped = {t: min(max(math.sqrt(n / (len(TYPES) * c)), lo), hi) for t, c in counts.items() if c}
    mean_w = sum(counts[t] * w for t, w in clipped.items()) / n
    weights = {t: w * tw["normalise_mean"] / mean_w for t, w in clipped.items()}
    return {"n_label0": n0, "n_label1": n1, "target_sum_neg": nsum, "target_sum_pos": tsum,
            "pos_weight": pos_weight, "type_counts": counts, "type_weights_clipped": clipped, "type_weights": weights}


def format_type_weights(lw: dict) -> str:
    return ", ".join(f"{t} {lw['type_weights'][t]:.3f} (n={lw['type_counts'][t]})" if t in lw["type_weights"]
                     else f"{t} n/a (n=0)" for t in TYPES)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
"""Window labels, fold split, frame-cache checks, the window dataset and its sampler."""


def load_labels(model_fps: int) -> tuple[pd.DataFrame, dict, dict]:
    """Read the window labels after checking them (and the split file) against split_manifest.json.
    Returns (data, manifest, {window_labels_sha256, split_sha256})."""
    manifest = json.loads(MANIFEST.read_text(encoding="ascii"))
    labels_sha = sha256_file(LABELS_CSV)
    if labels_sha != manifest.get("window_labels_sha256"):
        fail(f"{LABELS_CSV.name} SHA-256 {labels_sha} != window_labels_sha256 {manifest.get('window_labels_sha256')}")
    data = pd.read_csv(LABELS_CSV)
    if not (data["split_version"] == SPLIT_VERSION).all():
        fail(f"split_version is not {SPLIT_VERSION} in every row: {sorted(data['split_version'].unique())}")
    data["stem"] = data["rgb_path"].map(stem_of)
    data["k0"] = np.round(data["t_start"] * model_fps).astype(int)
    if not np.allclose(data["k0"] / model_fps, data["t_start"]):
        fail("t_start values are not multiples of 1/model_fps")
    data["ctype"] = data["rgb_path"].map(lambda p: Path(str(p).replace("\\", "/")).parent.name)
    unknown = sorted(set(data["ctype"]) - set(TYPES))
    if unknown:
        fail(f"unknown collision types (rgb_path folders): {unknown}")
    split_sha = sha256_file(SPLIT_CSV)
    if split_sha != manifest.get("split_sha256"):
        fail("synthetic_split.csv SHA-256 differs from split_sha256 in split_manifest.json")
    return data, manifest, {"window_labels_sha256": labels_sha, "split_sha256": split_sha}


def split_fold(data: pd.DataFrame, fold: int, smoke: bool,
               scheme: str = "B") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validation = all windows of the fold. Train (outside the fold), with a "target" column:
      B: windows with label_B 0/1, target = label_B (label_B -1 is not used);
      C: all windows, soft target = weight_C_text (the evaluation document's text rule: label_B 0 windows 0,
         windows containing t* 1, Gaussian decay with sigma 0.5 s before and after).
    The smoke test keeps the first clips in sorted rgb_path order."""
    outside = data["val_fold"] != fold
    if scheme == "B":
        train_all = data[outside & data["label_B"].isin([0, 1])].assign(target=lambda d: d["label_B"].astype(float))
    else:
        train_all = data[outside].assign(target=lambda d: d["weight_C_text"].astype(float))
        if train_all["target"].isna().any() or not train_all["target"].between(0, 1).all():
            fail("weight_C_text must be in [0, 1] for every training window")
    val_all = data[data["val_fold"] == fold]
    tr_clips = sorted(train_all["rgb_path"].unique())
    va_clips = sorted(val_all["rgb_path"].unique())
    if smoke:
        tr_clips, va_clips = tr_clips[:SMOKE["train_clips"]], va_clips[:SMOKE["val_clips"]]
    train_rows = train_all[train_all["rgb_path"].isin(tr_clips)]
    val_rows = val_all[val_all["rgb_path"].isin(va_clips)]
    train_rows = train_rows.sort_values(["rgb_path", "window_idx"]).reset_index(drop=True)
    val_rows = val_rows.sort_values(["rgb_path", "window_idx"]).reset_index(drop=True)
    if train_rows.empty or val_rows.empty:
        fail("empty training or validation set")
    return train_rows, val_rows


def check_cache(train_rows: pd.DataFrame, val_rows: pd.DataFrame, manifest: dict, partial: bool) -> str:
    """Check the frame cache (complete unless partial, files present, windows hash); returns its manifest hash."""
    cache_manifest = json.loads(CACHE_MANIFEST.read_text(encoding="ascii")) if CACHE_MANIFEST.is_file() else {}
    if not partial and cache_manifest.get("complete") is not True:
        fail(f"{CACHE_MANIFEST} does not say complete: true")
    needed = sorted(set(train_rows["stem"]) | set(val_rows["stem"]))
    missing = [s for s in needed if not (CACHE_DIR / f"{s}.npz").is_file()]
    if missing:
        fail(f"{len(missing)} cache files missing, e.g. {missing[:3]}")
    if cache_manifest.get("windows_sha256") != manifest.get("windows_sha256"):
        fail("cache_manifest windows_sha256 differs from split_manifest windows_sha256")
    return sha256_file(CACHE_MANIFEST)


def fps_drop_indices(k0: int, frames: int, f: int, model_fps: int) -> list[int]:
    """Frame i at t_i = t_start + i/model_fps shows the grid frame at floor(t_i * f) / f (an f fps camera)."""
    return [max(0, math.floor(model_fps * math.floor((k0 + i) / model_fps * f) / f + 1e-9)) for i in range(frames)]


def gray_of(f: np.ndarray) -> np.ndarray:
    return f[..., 0] * 0.299 + f[..., 1] * 0.587 + f[..., 2] * 0.114


def window_luma(x: np.ndarray) -> float:
    """Mean luma Y = 0.299 R + 0.587 G + 0.114 B over all frames and pixels of (T, H, W, 3) uint8 RGB, 0-255."""
    return float(gray_of(x.astype(np.float32)).mean())


def colour_jitter(x: np.ndarray, rng: np.random.Generator, c: dict, applied: list[str],
                  dark_luma: float | None = None) -> np.ndarray:
    """Brightness, contrast, saturation, hue, then grayscale (p grayscale_p); one draw per window.
    Night guard (dark_luma not None): if the window's mean luma before the colour shift is below dark_luma, the
    brightness factor comes from [NIGHT_GUARD_MIN_BRIGHTNESS, max] instead of [min, max]. It is the same single
    uniform draw mapped to the narrower range, so the random-draw sequence is unchanged."""
    lo, hi = c["brightness"]
    dark = dark_luma is not None and window_luma(x) < dark_luma
    if dark:
        lo = NIGHT_GUARD_MIN_BRIGHTNESS
    b = rng.uniform(lo, hi)
    ct, s = (rng.uniform(*c[k]) for k in ("contrast", "saturation"))
    h = rng.uniform(-c["hue"], c["hue"])
    gray = rng.random() < c["grayscale_p"]
    applied.append(f"colour b{b:.2f} c{ct:.2f} s{s:.2f} h{h:+.3f}{' dark' if dark else ''}{' gray' if gray else ''}")
    return apply_colour(x, b, ct, s, h, gray)


def apply_colour(x: np.ndarray, b: float, ct: float, s: float, h: float, gray: bool) -> np.ndarray:
    """Brightness x b, contrast x ct around each frame's mean grey, saturation x s, hue + h (fraction of the circle),
    then grayscale if gray; the same parameters for every frame."""
    f = np.clip(x.astype(np.float32) / 255.0 * b, 0, 1)
    m = gray_of(f).mean(axis=(1, 2)).reshape(-1, 1, 1, 1)
    f = np.clip((f - m) * ct + m, 0, 1)
    g = gray_of(f)[..., None]
    f = np.clip(g + (f - g) * s, 0, 1)
    for t in range(len(f)):
        hsv = cv2.cvtColor(f[t], cv2.COLOR_RGB2HSV)  # float32: H in [0, 360)
        hsv[..., 0] = (hsv[..., 0] + h * 360.0) % 360.0
        f[t] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    f = np.clip(f, 0, 1)
    if gray:
        f = np.repeat(gray_of(f)[..., None], 3, axis=-1)
    return (f * 255.0 + 0.5).astype(np.uint8)


def apply_blur(x: np.ndarray, sigma: float) -> np.ndarray:
    return np.stack([cv2.GaussianBlur(fr, (0, 0), sigma) for fr in x])


def apply_downscale(x: np.ndarray, s: float) -> np.ndarray:
    """Downscale every frame to round(s * size) (area), then back up (bilinear)."""
    h, w = x.shape[1:3]
    small = (max(1, round(w * s)), max(1, round(h * s)))
    return np.stack([cv2.resize(cv2.resize(fr, small, interpolation=cv2.INTER_AREA), (w, h),
                                interpolation=cv2.INTER_LINEAR) for fr in x])


def apply_jpeg(x: np.ndarray, q: int) -> np.ndarray:
    enc = [cv2.imencode(".jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])[1] for fr in x]
    return np.stack([cv2.cvtColor(cv2.imdecode(e, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for e in enc])


def cctv_after_load(x: np.ndarray, rng: np.random.Generator, c: dict, applied: list[str],
                    dark_luma: float | None = None) -> np.ndarray:
    """cctv steps b-e on (T, H, W, 3) uint8 RGB frames, each with probability p, one draw per window
    (dark_luma: night guard of the colour shift, None = off)."""
    if rng.random() < c["p"]:
        sigma = rng.uniform(*c["blur_sigma"])
        x = apply_blur(x, sigma)
        applied.append(f"blur {sigma:.2f}px")
    if rng.random() < c["p"]:
        s = rng.uniform(*c["downscale"])
        x = apply_downscale(x, s)
        applied.append(f"downscale {s:.2f}")
    if rng.random() < c["p"]:
        x = colour_jitter(x, rng, c, applied, dark_luma)
    if rng.random() < c["p"]:
        q = int(rng.integers(c["jpeg_quality"][0], c["jpeg_quality"][1] + 1))
        x = apply_jpeg(x, q)
        applied.append(f"jpeg q{q}")
    return x


# Ablation 1, profile v2 (augmentation "cctv_v2"). Fixed in code: the config only selects the profile, so a cctv_v2
# config differs from its standard counterpart only in the augmentation key and night_guard; cctv_aug is not used.
# Label-preserving: each degradation p 0.3, at most 2 per window (1 for protected windows), milder ranges than "cctv".
# Protected ("dark") windows: mean luma of the unaugmented window < night_guard dark_luma; they are never darkened,
# never lose contrast and get milder blur, downscale and JPEG (the *_dark ranges). A *_dark range is drawn with the same
# single draw as the normal one (uniform: lo + (hi - lo) u; integers: one 32-bit draw mapped by multiply-high), so the
# draw sequence does not depend on which range is used. fps drop, saturation, hue and grayscale are the same for all.
CCTV_V2 = {"p": 0.3, "max_degradations": 2, "max_degradations_dark": 1, "fps_drop": [4, 5, 6],
           "blur_sigma": [0.3, 1.0], "downscale": [0.5, 0.9], "brightness": [0.7, 1.3], "contrast": [0.8, 1.2],
           "saturation": [0.7, 1.3], "hue": 0.02, "grayscale_p": 0.1, "jpeg_quality": [25, 50],
           "brightness_dark": [1.0, 1.3], "contrast_dark": [1.0, 1.2], "blur_sigma_dark": [0.3, 0.7],
           "downscale_dark": [0.7, 0.9], "jpeg_quality_dark": [35, 50]}
CCTV_V2_OPS = ("fps", "blur", "downscale", "colour", "jpeg")  # application order (hflip after fps, as in "cctv")


def cctv_v2_plan(rng: np.random.Generator, dark: bool) -> dict:
    """All random choices of one window: a gate per degradation (p), then, if more are drawn than allowed, a random
    subset of the allowed size (order kept), then the parameters of the kept degradations in application order."""
    v = CCTV_V2
    drawn = [op for op in CCTV_V2_OPS if rng.random() < v["p"]]
    cap = v["max_degradations_dark"] if dark else v["max_degradations"]
    kept = drawn
    if len(drawn) > cap:
        keep = {int(i) for i in rng.choice(len(drawn), size=cap, replace=False)}
        kept = [op for i, op in enumerate(drawn) if i in keep]
    plan: dict = {"drawn": drawn, "kept": kept, "dark": dark}
    if "fps" in kept:
        plan["fps"] = int(rng.choice(v["fps_drop"]))
    sfx = "_dark" if dark else ""
    if "blur" in kept:
        plan["blur"] = float(rng.uniform(*v["blur_sigma" + sfx]))
    if "downscale" in kept:
        plan["downscale"] = float(rng.uniform(*v["downscale" + sfx]))
    if "colour" in kept:
        b = float(rng.uniform(*v["brightness_dark" if dark else "brightness"]))
        ct = float(rng.uniform(*v["contrast_dark" if dark else "contrast"]))
        s = float(rng.uniform(*v["saturation"]))
        h = float(rng.uniform(-v["hue"], v["hue"]))
        plan["colour"] = (b, ct, s, h, bool(rng.random() < v["grayscale_p"]))
    if "jpeg" in kept:
        lo, hi = v["jpeg_quality" + sfx]
        plan["jpeg"] = int(rng.integers(lo, hi + 1))
    return plan


class WindowDataset(Dataset):
    """Index i encodes (row, flip) as i = 2*row + flip, so flips are decided in the main process.
    The cctv profiles draw their other augmentations from the worker's seeded generator (self.rng).
    Items: (frames, label, weight, row, guard). cctv / standard: guard 0 = no colour shift, 1 = colour shift (night
    guard off or window not dark), 2 = colour shift with the night guard's brightness range (window darker than
    dark_luma). cctv_v2: guard 1 = window not dark, 2 = dark window (dark-window rules applied)."""

    def __init__(self, stems: pd.Series, k0: pd.Series, labels: pd.Series, weights: pd.Series | None,
                 mean: list[float], std: list[float], frames: int, model_fps: int, lru_size: int,
                 aug: str = "none", aug_cfg: dict | None = None, dark_luma: float | None = None) -> None:
        self.stems, self.k0, self.labels = list(stems), np.asarray(k0, int), np.asarray(labels, np.float32)
        self.weights = np.ones(len(self.stems), np.float32) if weights is None else np.asarray(weights, np.float32)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.frames, self.model_fps, self.lru_size = frames, model_fps, lru_size
        self.aug, self.aug_cfg, self.dark_luma = aug, aug_cfg, dark_luma
        self.rng: np.random.Generator | None = None  # set per worker by worker_init
        self._lru: OrderedDict = OrderedDict()

    def __len__(self) -> int:
        return len(self.stems)

    def _clip(self, stem: str) -> dict:
        if stem in self._lru:
            self._lru.move_to_end(stem)
            return self._lru[stem]
        with np.load(CACHE_DIR / f"{stem}.npz") as z:
            item = {"offsets": z["offsets"], "data": z["data"]}
        self._lru[stem] = item
        if len(self._lru) > self.lru_size:
            self._lru.popitem(last=False)
        return item

    def load_v2(self, row: int, flip: bool, rng: np.random.Generator | None,
                make_plan: Callable[[np.random.Generator | None, bool], dict] = cctv_v2_plan
                ) -> tuple[np.ndarray, list[str]]:
        """cctv_v2: the window's unaugmented frames decide dark (mean luma < dark_luma), then the plan is drawn and
        applied: frame-rate drop, hflip, blur, downscale, colour, JPEG. applied[0] describes the draw."""
        z = self._clip(self.stems[row])
        k0 = int(self.k0[row])
        x = np.stack([read_cached_frame(z, k) for k in range(k0, k0 + self.frames)])
        luma = window_luma(x)
        dark = self.dark_luma is not None and luma < self.dark_luma
        p = make_plan(rng, dark)
        applied = [f"v2 drawn {'+'.join(p['drawn']) or '-'} kept {'+'.join(p['kept']) or '-'} luma {luma:.1f}"
                   + (" dark" if p["dark"] else "")]
        if "fps" in p["kept"]:
            x = np.stack([read_cached_frame(z, k) for k in fps_drop_indices(k0, self.frames, p["fps"], self.model_fps)])
            applied.append(f"fps {p['fps']}")
        if flip:
            x = np.ascontiguousarray(x[:, :, ::-1, :])
            applied.append("hflip")
        if "blur" in p["kept"]:
            x = apply_blur(x, p["blur"])
            applied.append(f"blur {p['blur']:.2f}px")
        if "downscale" in p["kept"]:
            x = apply_downscale(x, p["downscale"])
            applied.append(f"downscale {p['downscale']:.2f}")
        if "colour" in p["kept"]:
            b, ct, s, h, gray = p["colour"]
            x = apply_colour(x, b, ct, s, h, gray)
            applied.append(f"colour b{b:.2f} c{ct:.2f} s{s:.2f} h{h:+.3f}{' dark' if p['dark'] else ''}"
                           f"{' gray' if gray else ''}")
        if "jpeg" in p["kept"]:
            x = apply_jpeg(x, p["jpeg"])
            applied.append(f"jpeg q{p['jpeg']}")
        return x, applied

    def load(self, row: int, flip: bool, aug: str, rng: np.random.Generator | None) -> tuple[np.ndarray, list[str]]:
        """(T, H, W, 3) uint8 RGB frames of one window and the list of augmentations applied."""
        if aug == "cctv_v2":
            return self.load_v2(row, flip, rng)
        z = self._clip(self.stems[row])
        k0 = int(self.k0[row])
        idx, applied = list(range(k0, k0 + self.frames)), []
        c = self.aug_cfg
        if aug == "cctv" and rng.random() < c["p"]:  # a) frame-rate drop decides which grid frames are read
            f = int(rng.choice(c["fps_drop"]))
            idx = fps_drop_indices(k0, self.frames, f, self.model_fps)
            applied.append(f"fps {f}")
        x = np.stack([read_cached_frame(z, k) for k in idx])
        if flip:
            x = np.ascontiguousarray(x[:, :, ::-1, :])
            applied.insert(0, "hflip")
        if aug == "cctv":
            x = cctv_after_load(x, rng, c, applied, self.dark_luma)
        return x, applied

    def __getitem__(self, code: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        row, flip = divmod(int(code), 2)
        if self.aug in ("cctv", "cctv_v2") and self.rng is None:  # main process (num_workers 0)
            self.rng = np.random.default_rng(torch.initial_seed() % 2 ** 32)
        x, applied = self.load(row, bool(flip), self.aug, self.rng)
        if self.aug == "cctv_v2":  # every window: 1 = not dark, 2 = dark (dark-window rules applied)
            guard = 2 if applied[0].endswith(" dark") else 1
        else:
            colour = [a for a in applied if a.startswith("colour")]
            guard = 0 if not colour else 2 if " dark" in colour[0] else 1
        x = torch.from_numpy(x).permute(0, 3, 1, 2).float().div_(255.0)
        x = (x - self.mean) / self.std
        return x, torch.tensor(self.labels[row]), torch.tensor(self.weights[row]), row, guard


def worker_init(worker_id: int) -> None:
    """Give each DataLoader worker its own generator, seeded from the worker's torch seed."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % 2 ** 32)


class ListSampler(Sampler):
    """Yields a list set by the main process before each epoch (works with persistent workers)."""

    def __init__(self) -> None:
        self.items: list[int] = []

    def set(self, items) -> None:
        self.items = list(items)

    def __iter__(self) -> Iterator[int]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)


def epoch_codes(n: int, seed: int, epoch: int, hflip_p: float) -> list[int]:
    """Seeded shuffle of the rows and horizontal flips for one epoch, as dataset codes 2*row + flip."""
    g = torch.Generator()
    g.manual_seed(seed * 1000 + epoch)
    perm = torch.randperm(n, generator=g)
    flips = torch.rand(n, generator=g) < hflip_p
    return (perm * 2 + flips[perm].long()).tolist()


def make_loader(ds: Dataset, batch: int, workers: int, sampler: Sampler) -> DataLoader:
    kw = dict(batch_size=batch, sampler=sampler, num_workers=workers, pin_memory=True, drop_last=False)
    if workers > 0:
        kw.update(persistent_workers=True, worker_init_fn=worker_init)
    return DataLoader(ds, **kw)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
"""VideoMAE with a one-logit head; drop path is applied only if transformers supports it."""


def drop_path_field() -> str | None:
    """Name of the VideoMAEConfig field for drop path / stochastic depth, or None if it has none."""
    from transformers import VideoMAEConfig
    fields = VideoMAEConfig().to_dict()
    return next((k for k in ("drop_path_rate", "drop_path", "stochastic_depth_rate") if k in fields), None)


def build_model(cfg: dict, device: torch.device, dp_field: str | None) -> nn.Module:
    from transformers import VideoMAEForVideoClassification
    kw = {dp_field: cfg["drop_path"]} if dp_field else {}
    model = VideoMAEForVideoClassification.from_pretrained(cfg["model_id"], **kw)
    model.classifier = nn.Linear(model.config.hidden_size, 1)
    return model.to(device)


# --------------------------------------------------------------------------- #
# Optimizer and schedule
# --------------------------------------------------------------------------- #
"""AdamW with layer-wise lr decay (BEiT/VideoMAE style) and warmup + cosine schedule."""

HEAD_PREFIXES = ("fc_norm.", "classifier.", "videomae.layernorm.")


def layer_id(name: str, n_layers: int) -> int:
    """Embeddings 0, encoder block i -> i + 1, final norm / pooler / head n_layers + 1."""
    if name.startswith("videomae.embeddings."):
        return 0
    m = re.match(r"videomae\.encoder\.layer\.(\d+)\.", name)
    if m:
        return int(m.group(1)) + 1
    if name.startswith(HEAD_PREFIXES):
        return n_layers + 1
    fail(f"no layer id rule for parameter {name}")


def param_groups(model: nn.Module, cfg: dict, lr: float) -> list[dict]:
    """One group per (layer id, decay / no decay); lr = lr * layer_decay ** (top id - layer id).
    No weight decay on biases and LayerNorm weights (1-D parameters)."""
    n_layers = model.config.num_hidden_layers
    top = n_layers + 1
    groups: dict[tuple[int, bool], list] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        decay = not (p.ndim == 1 or name.endswith(".bias"))
        groups.setdefault((layer_id(name, n_layers), decay), []).append(p)
    out = []
    for (lid, decay), params in sorted(groups.items()):
        out.append({"params": params, "lr": lr * cfg["layer_decay"] ** (top - lid),
                    "weight_decay": cfg["weight_decay"] if decay else 0.0, "layer_id": lid,
                    "name": f"layer{lid:02d}_{'decay' if decay else 'no_decay'}"})
    return out


def lr_by_layer(optimizer: torch.optim.Optimizer) -> dict[int, float]:
    return {g["layer_id"]: g["lr"] for g in optimizer.param_groups}


def make_lr_lambda(warmup: int, total: int) -> Callable[[int], float]:
    """Linear warmup over `warmup` updates, then cosine to 0 at `total` updates."""
    def f(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        span = max(1, total - warmup)
        progress = min(1.0, (step - warmup) / span)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return f


# --------------------------------------------------------------------------- #
# Train step
# --------------------------------------------------------------------------- #
"""Forward/backward of one micro-batch, with an out-of-memory fallback that recomputes the
current accumulation group in smaller chunks so the optimizer update stays the same."""


class OOMAbort(RuntimeError):
    """Out of GPU memory even with the smallest chunk size."""


def fwd_bwd(model: nn.Module, criterion: nn.Module, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor,
            device: torch.device, weight: float, chunk: int) -> float:
    """Forward + backward of (x, y) in chunks. Window loss = w * BCE(pos_weight) (criterion has
    reduction "none"); micro-batch loss = mean over its windows. Each chunk's mean is weighted by
    len(chunk) / len(y) * weight, so the accumulated gradient equals the full micro-batch one.
    Returns the micro-batch mean loss."""
    n, total = len(y), 0.0
    for s in range(0, n, chunk):
        xc = x[s:s + chunk].to(device, non_blocking=True)
        yc = y[s:s + chunk].to(device, non_blocking=True)
        wc = w[s:s + chunk].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(pixel_values=xc).logits.squeeze(-1)
        loss = (criterion(logits.float(), yc) * wc).mean()
        (loss * (len(yc) / n * weight)).backward()
        total += loss.item() * len(yc)
    return total / n


def recompute_group(model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer,
                    group: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], device: torch.device,
                    weight: float, log: "Logger", where: str) -> list[float]:
    """After an OOM: drop the group's gradients (including partial ones) and recompute every micro-batch
    of the group so far in chunks of 4, then 2. Raises OOMAbort if chunks of 2 also fail."""
    for chunk in OOM_CHUNKS:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        losses = None
        try:
            losses = [fwd_bwd(model, criterion, x, y, w, device, weight, chunk) for x, y, w in group]
        except torch.cuda.OutOfMemoryError:
            pass
        if losses is not None:
            log(f"OOM fallback ({where}): recomputed {len(group)} micro-batch(es) of this update in chunks of {chunk}")
            return losses
        log(f"OOM fallback ({where}): chunks of {chunk} also ran out of memory")
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    raise OOMAbort(f"out of GPU memory at {where}, even in chunks of {OOM_CHUNKS[-1]}")


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
"""Window scores -> per-clip t_hat (3-window moving average, centre of the max) -> T at 0.5/1/2 s."""


def clip_predictions(val_rows: pd.DataFrame, scores: np.ndarray, smooth_window: int,
                     exclude: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = val_rows[["rgb_path", "window_idx", "t_centre", "t_star", "label_B"]].copy()
    df["score"] = scores
    out = []
    kernel = np.ones(smooth_window) / smooth_window
    for path, g in df.groupby("rgb_path", sort=True):
        g = g.sort_values("t_centre")
        s = np.convolve(g["score"].to_numpy(float), kernel, mode="same")
        p = int(np.argmax(s))
        idx = np.arange(len(s))
        outside = s[np.abs(idx - p) > exclude]
        conf = float(s[p] - outside.max()) if len(outside) else 0.0
        out.append({"rgb_path": path, "t_star": float(g["t_star"].iloc[0]),
                    "t_hat": float(g["t_centre"].iloc[p]), "confidence": conf})
    pred = pd.DataFrame(out)
    pred["t_hat_minus_t_star"] = pred["t_hat"] - pred["t_star"]  # diagnostic only, positive = late
    pred["within_2s"] = pred["t_hat_minus_t_star"].abs() <= 2.0  # diagnostic only
    for s_ in SIGMAS_T:
        pred[f"T_i_{s_:g}"] = temporal_per_clip(pred["t_hat"], pred["t_star"], s_)
    return df, pred


def val_logits(model: nn.Module, x: torch.Tensor, device: torch.device, chunk: int) -> np.ndarray:
    outs = []
    for s in range(0, len(x), chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(pixel_values=x[s:s + chunk].to(device, non_blocking=True)).logits.squeeze(-1)
        outs.append(out.float().cpu().numpy())
    return np.concatenate(outs)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, sampler: ListSampler, val_rows: pd.DataFrame, cfg: dict,
             device: torch.device, log: "Logger") -> tuple[float, dict, pd.DataFrame, pd.DataFrame, int]:
    """Returns (val loss on labelled windows, T by sigma, window scores, clip predictions, OOM fallbacks)."""
    model.eval()
    sampler.set(range(0, 2 * len(val_rows), 2))  # rows in order, no flip
    logits = np.empty(len(val_rows), dtype=np.float32)
    fallbacks = 0
    for x, _, _, rows, _ in loader:
        out = None
        try:
            out = val_logits(model, x, device, len(x))
        except torch.cuda.OutOfMemoryError:
            pass
        if out is None:  # recover outside the except block, so the failed attempt's memory is released
            fallbacks += 1
            for chunk in OOM_CHUNKS:
                torch.cuda.empty_cache()
                try:
                    out = val_logits(model, x, device, chunk)
                except torch.cuda.OutOfMemoryError:
                    log(f"OOM fallback (validation): chunks of {chunk} also ran out of memory")
                    continue
                log(f"OOM fallback (validation): batch of {len(x)} recomputed in chunks of {chunk}")
                break
            if out is None:
                torch.cuda.empty_cache()
                raise OOMAbort(f"out of GPU memory in validation, even in chunks of {OOM_CHUNKS[-1]}")
        logits[rows.numpy()] = out
    scores = 1.0 / (1.0 + np.exp(-logits))
    lab = val_rows["label_B"].to_numpy()
    m = lab >= 0
    val_loss = float(nn.functional.binary_cross_entropy_with_logits(
        torch.from_numpy(logits[m]), torch.from_numpy(lab[m].astype(np.float32)))) if m.any() else float("nan")
    win_df, pred = clip_predictions(val_rows, scores, cfg["smooth_window"], cfg["confidence_exclude"])
    T = {s: temporal(pred["t_hat"], pred["t_star"], s) for s in SIGMAS_T}
    model.train()
    return val_loss, T, win_df, pred, fallbacks


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
"""File-first logging, Ctrl+C held back during critical sections, atomic saves, RNG state."""


class Logger:
    def __init__(self, path: Path) -> None:
        self.fh = open(path, "a", encoding="utf-8")

    def __call__(self, text: str = "") -> None:
        # file first, so the record survives when the console pipe (e.g. Tee-Object) is gone
        self.fh.write(str(text) + "\n")
        self.fh.flush()
        try:
            print(text, flush=True)
        except OSError:
            pass

    def close(self) -> None:
        self.fh.flush()
        os.fsync(self.fh.fileno())
        self.fh.close()


class InterruptGuard:
    """Ctrl+C raises KeyboardInterrupt, except inside critical() blocks, where it is
    deferred to the end of the block (so an optimizer step is never half recorded)."""

    def __init__(self) -> None:
        self.depth, self.pending = 0, False
        signal.signal(signal.SIGINT, self._handler)

    def _handler(self, signum, frame) -> None:
        if self.depth:
            self.pending = True
        else:
            raise KeyboardInterrupt

    @contextmanager
    def critical(self) -> Iterator[None]:
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1
        if self.depth == 0 and self.pending:
            self.pending = False
            raise KeyboardInterrupt


def atomic_save(obj: dict, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def rng_states() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def set_rng_states(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    torch.cuda.set_rng_state_all(s["cuda"])


def latest_checkpoint(ckpt_dir: Path) -> tuple[str, dict]:
    """The newer (by global step) of last.pt and step_latest.pt."""
    cands = []
    for name in ("last.pt", "step_latest.pt"):
        p = ckpt_dir / name
        if p.is_file():
            c = torch.load(p, map_location="cpu", weights_only=False)
            cands.append((c["state"]["global_step"], name, c))
    if not cands:
        fail(f"--resume: no checkpoint in {ckpt_dir}")
    _, name, ck = max(cands, key=lambda t: t[0])
    return name, ck


# --------------------------------------------------------------------------- #
# GPU guard
# --------------------------------------------------------------------------- #
"""GPU temperature and SM clock via NVML (nvidia-ml-py); pauses training while the GPU is too hot."""


class GpuGuard:
    CHECK_S = 30

    def __init__(self, pause_c: float, resume_c: float, log: Logger) -> None:
        self.pause_c, self.resume_c, self.log = pause_c, resume_c, log
        self.nvml = self.handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.nvml = pynvml
        except Exception as e:  # package missing or NVML unavailable
            log(f"WARNING: GPU guard disabled ({type(e).__name__}: {e}); training continues without "
                "temperature monitoring or pauses")
        self.reset_epoch()

    def read(self) -> tuple[float, float]:
        """(temperature C, SM clock MHz), NaN if unavailable."""
        if self.nvml is None:
            return math.nan, math.nan
        try:
            return (float(self.nvml.nvmlDeviceGetTemperature(self.handle, self.nvml.NVML_TEMPERATURE_GPU)),
                    float(self.nvml.nvmlDeviceGetClockInfo(self.handle, self.nvml.NVML_CLOCK_SM)))
        except Exception as e:
            self.log(f"WARNING: GPU guard disabled after an NVML error ({type(e).__name__}: {e})")
            self.nvml = None
            return math.nan, math.nan

    def reset_epoch(self) -> None:
        self.temp_max = self.clock_min = math.nan

    def sample(self) -> tuple[float, float]:
        """Read and track the epoch's max temperature and min SM clock."""
        t, c = self.read()
        if not math.isnan(t):
            self.temp_max = t if math.isnan(self.temp_max) else max(self.temp_max, t)
        if not math.isnan(c):
            self.clock_min = c if math.isnan(self.clock_min) else min(self.clock_min, c)
        return t, c

    def maybe_pause(self, temp: float, where: str) -> float:
        """If temp >= pause threshold, sleep in 30 s checks until <= resume threshold; returns seconds paused."""
        if math.isnan(temp) or temp < self.pause_c:
            return 0.0
        start = time.perf_counter()
        self.log(f"GPU guard: {temp:.0f} C >= {self.pause_c} C after {where}; pausing, checking every "
                 f"{self.CHECK_S} s until <= {self.resume_c} C")
        while not math.isnan(temp) and temp > self.resume_c:
            time.sleep(self.CHECK_S)
            temp, _ = self.read()
        paused = time.perf_counter() - start
        self.log(f"GPU guard: resuming at {temp:.0f} C after a pause of {paused / 60:.1f} min")
        return paused


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
"""--plan (counts and schedule for all folds, no model, no GPU) and the training loop."""


def plan(cfg: dict, cfg_path: Path, cfg_hash: str) -> None:
    data, _, _ = load_labels(cfg["model_fps"])
    lr = scaled_lr(cfg)
    eff = cfg["micro_batch"] * cfg["grad_accum"]
    scheme = label_scheme(cfg)
    every, patience = check_schedule(cfg)
    print(f"config {cfg_path} (sha256 {cfg_hash[:12]}...)")
    print(f"label scheme {scheme}; effective batch {cfg['micro_batch']} x {cfg['grad_accum']} = {eff}; scaled lr = "
          f"{cfg['base_lr']:g} * {eff} / 256 = {lr:.4g}; max_epochs {cfg['max_epochs']}; warmup_fraction "
          f"{cfg['warmup_fraction']}; validation "
          + ("once per epoch" if every is None else f"every {every} updates and at the end")
          + f"; early stopping after {patience} checks without improvement > {cfg['min_delta']}")
    rows = []
    for fold in (0, 1, 2):
        tr, va = split_fold(data, fold, smoke=False, scheme=scheme)
        s = schedule_sizes(len(tr), cfg, cfg["max_epochs"])
        lw = loss_weights(tr, cfg)
        checks = n_checks(s["total_updates"], cfg["max_epochs"], every)
        hours = (len(tr) * cfg["max_epochs"] / TRAIN_RATE + checks * len(va) / VAL_RATE) / 3600
        rows.append({"fold": fold, "train_clips": tr["rgb_path"].nunique(), "train_windows": len(tr),
                     "target_sum": round(lw["target_sum_pos"], 1), "val_clips": va["rgb_path"].nunique(),
                     "val_windows": len(va), "updates/epoch": s["updates_per_epoch"],
                     "total_updates": s["total_updates"], "warmup": s["warmup_updates"], "checks": checks,
                     "hours": f"{hours:.2f}", "pos_weight": f"{lw['pos_weight']:.3f}",
                     **{f"w_{t}": f"{lw['type_weights'][t]:.3f}" if t in lw["type_weights"] else "n/a"
                        for t in TYPES}})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"target_sum = sum of training targets (= positives for scheme B); pos_weight = sum(1 - target) / "
          f"sum(target); type weight sqrt(N / (5 n_c)) clipped to {cfg['type_weight']['clip']}, then mean "
          f"{cfg['type_weight']['normalise_mean']}, over the windows that enter the loss")
    print(f"hours = training {cfg['max_epochs']} epochs at {TRAIN_RATE} windows/s + {'' if every else 'one '}"
          f"validation{' per check' if every else ' per epoch'} at {VAL_RATE} windows/s, for the full "
          f"{cfg['max_epochs']} epochs (early stopping can end sooner); GPU-guard pauses not included")


def preview_aug(args: argparse.Namespace, cfg: dict, cfg_path: Path) -> None:
    """Save N training windows of the fold as a PNG grid: per window one row of original frames
    0/5/10/15 and one row with the config's augmentation profile. No model, no GPU, no training."""
    data, manifest, _ = load_labels(cfg["model_fps"])
    train_rows, val_rows = split_fold(data, args.fold, smoke=False, scheme=label_scheme(cfg))
    codes = epoch_codes(len(train_rows), args.seed, 0, cfg["hflip_p"])[:args.preview_aug]
    check_cache(train_rows.iloc[[c // 2 for c in codes]], val_rows.iloc[:0], manifest, partial=True)
    ds = WindowDataset(train_rows["stem"], train_rows["k0"], train_rows["label_B"], None, [0, 0, 0], [1, 1, 1],
                       cfg["frames"], cfg["model_fps"], cfg["lru_size"], cfg["augmentation"], cfg["cctv_aug"],
                       dark_luma_of(cfg))
    rng = np.random.default_rng(args.seed)
    tiles, lines = [], []
    for i, code in enumerate(codes):
        row, flip = divmod(code, 2)
        orig, _ = ds.load(row, False, "none", None)
        aug, applied = ds.load(row, bool(flip), cfg["augmentation"], rng)
        r = train_rows.iloc[row]
        name = f"{i}: {r['stem']} t_start {r['t_start']:.1f} s label_B {int(r['label_B'])}"
        lines.append(f"{name}; augmented: {', '.join(applied) if applied else 'none'}")
        for label, x in (("original", orig), ("augmented", aug)):
            strip = np.concatenate([x[k] for k in PREVIEW_FRAMES], axis=1).copy()
            cv2.putText(strip, f"{i} {label}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            tiles.append(strip)
    AUG_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out = AUG_PREVIEW_DIR / f"{cfg_path.stem}_fold{args.fold}_n{len(codes)}.png"
    cv2.imwrite(str(out), cv2.cvtColor(np.concatenate(tiles, axis=0), cv2.COLOR_RGB2BGR))
    out.with_suffix(".txt").write_text(
        f"config {cfg_path}; augmentation {cfg['augmentation']}; fold {args.fold}; seed {args.seed}; "
        f"frames {PREVIEW_FRAMES}\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"augmentation preview ({cfg['augmentation']}): {out}")
    print("\n".join(lines))


def train(args: argparse.Namespace, cfg: dict, cfg_path: Path, cfg_hash: str) -> None:
    smoke = args.smoke_test
    max_epochs = min(cfg["max_epochs"], SMOKE["max_epochs"]) if smoke else cfg["max_epochs"]
    step_ckpt_every = SMOKE["step_ckpt_every"] if smoke else cfg["step_ckpt_every"]
    run_name = (args.run_name or f"{cfg_path.stem}_seed{args.seed}") + ("_smoke" if smoke else "")
    run_hash = canonical_hash({"config_hash": cfg_hash, "fold": args.fold, "seed": args.seed, "smoke_test": smoke})
    mb, ga, log_every = cfg["micro_batch"], cfg["grad_accum"], cfg["log_every"]

    out_dir = RUNS_DIR / run_name / f"fold_{args.fold}"
    ckpt_dir = out_dir / "checkpoints"
    if out_dir.exists() and not args.resume and any(out_dir.iterdir()):
        fail(f"{out_dir} already exists; use --resume or a different --run-name")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "train_log.txt")

    # ---------------- data and environment checks ----------------
    scheme = label_scheme(cfg)
    every, patience = check_schedule(cfg)
    data, manifest, data_hashes = load_labels(cfg["model_fps"])
    train_rows, val_rows = split_fold(data, args.fold, smoke, scheme)
    data_hashes["cache_manifest_sha256"] = check_cache(train_rows, val_rows, manifest, partial=smoke)
    if not torch.cuda.is_available():
        fail("CUDA is not available")
    device = torch.device("cuda")
    import transformers
    from transformers import VideoMAEImageProcessor
    transformers.utils.logging.disable_progress_bar()
    proc = VideoMAEImageProcessor.from_pretrained(cfg["model_id"])
    dp_field = drop_path_field()
    head, dirty = git_info()
    seed_everything(args.seed)
    sizes = schedule_sizes(len(train_rows), cfg, max_epochs)
    n_batches, updates_per_epoch = sizes["micro_batches_per_epoch"], sizes["updates_per_epoch"]
    lr0 = scaled_lr(cfg)
    lw = loss_weights(train_rows, cfg)
    train_rows["w_type"] = train_rows["ctype"].map(lw["type_weights"])

    run_info = {
        "config_file": str(cfg_path), "config": cfg, "config_hash": cfg_hash,
        "fold": args.fold, "seed": args.seed, "smoke_test": smoke, "run_hash": run_hash,
        "effective": {"max_epochs": max_epochs, "step_ckpt_every": step_ckpt_every, "scaled_lr": lr0, **sizes,
                      "label_scheme": scheme, "validate_every_updates": every, "patience_checks": patience,
                      "checks_full_run": n_checks(sizes["total_updates"], max_epochs, every)},
        "drop_path_applied": dp_field is not None,
        "drop_path_note": (f"set {dp_field} = {cfg['drop_path']}" if dp_field else
                           f"transformers {transformers.__version__} VideoMAE has no drop path / stochastic depth "
                           "option; the model was not patched"),
        "loss_weights": lw,
        "data_hashes": data_hashes,
        "normalisation": {"mean": list(proc.image_mean), "std": list(proc.image_std),
                          "note": "frames are cached 224x224 full-frame squashes; processor resize/crop not used"},
        "n_train_windows": len(train_rows), "n_train_clips": int(train_rows["rgb_path"].nunique()),
        "train_label_counts": {str(k): int(v) for k, v in train_rows["label_B"].value_counts().sort_index().items()},
        "n_val_windows": len(val_rows), "n_val_clips": int(val_rows["rgb_path"].nunique()),
        "git_commit": head, "git_dirty": dirty, "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": transformers.__version__, "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0), "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if cfg["augmentation"] == "cctv_v2":  # the profile's ranges live in the code, not in the config
        run_info["cctv_v2_profile"] = CCTV_V2
    info_path = out_dir / "config.json"
    if args.resume and info_path.is_file():
        old = json.loads(info_path.read_text(encoding="utf-8"))
        if old.get("run_hash") != run_hash or old.get("data_hashes") != data_hashes:
            fail("config, fold, seed or data hashes differ from the run being resumed")
    else:
        info_path.write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")

    resume_ck = None
    if args.resume:
        # a finished run (all updates applied and the final check made, or early stopping triggered) is not
        # trained, validated or written again
        resume_ck = latest_checkpoint(ckpt_dir)
        s = resume_ck[1]["state"]
        if s["no_improve"] >= patience:
            reason = f"early stopping triggered ({s['no_improve']} checks without improvement)"
        elif s["global_step"] >= sizes["total_updates"] and (
                s["epoch"] >= max_epochs if every is None else s.get("last_check_step") == s["global_step"]):
            reason = f"all {sizes['total_updates']} updates done"
        else:
            reason = None
        if reason:
            m = pd.read_csv(out_dir / "metrics.csv")
            b = m[m["best"].astype(str) == "True"].iloc[-1]
            k = int(b["check_idx"]) if "check_idx" in m and not pd.isna(b["check_idx"]) else int(b["epoch"])
            print(f"run already complete: {reason}, best val T@1 {s['best']:.4f} at check {k} "
                  f"(update {int(b['global_step'])}); {resume_ck[0]} not resumed, nothing written")
            sys.exit(0)

    log(f"=== train_classifier.py  fold {args.fold}  run {run_name}  {'SMOKE TEST' if smoke else ''}")
    log(f"time {run_info['created_at']}; GPU {run_info['gpu']}; torch {torch.__version__}; "
        f"transformers {transformers.__version__}; git {head}{' (dirty)' if dirty else ''}")
    log(f"config {cfg_path} (sha256 {cfg_hash[:12]}...); seed {args.seed}; run hash {run_hash[:12]}...; "
        f"output {out_dir}")
    log(f"data: window labels {data_hashes['window_labels_sha256'][:12]}..., split "
        f"{data_hashes['split_sha256'][:12]}..., cache manifest {data_hashes['cache_manifest_sha256'][:12]}...; "
        f"split_version {SPLIT_VERSION}")
    log(f"train: {len(train_rows)} windows from {run_info['n_train_clips']} clips, label_B counts "
        f"{run_info['train_label_counts']}; val: {len(val_rows)} windows from {run_info['n_val_clips']} clips")
    log(f"drop path: {'applied, ' if dp_field else 'NOT applied: '}{run_info['drop_path_note']} "
        f"(drop_path_applied: {str(dp_field is not None).lower()})")
    if scheme == "B":
        log(f"loss: pos_weight = n_label0 / n_label1 = {lw['n_label0']} / {lw['n_label1']} = {lw['pos_weight']:.4f}; "
            f"type weights: {format_type_weights(lw)}")
    else:
        log(f"label scheme C: all {len(train_rows)} training windows, soft target weight_C_text (sum "
            f"{lw['target_sum_pos']:.1f}; label_B counts {run_info['train_label_counts']})")
        log(f"loss: pos_weight = sum(1 - target) / sum(target) = {lw['target_sum_neg']:.1f} / "
            f"{lw['target_sum_pos']:.1f} = {lw['pos_weight']:.4f}; type weights: {format_type_weights(lw)}")
    log("validation: " + (f"once per epoch; early stopping after {patience} epochs" if every is None else
                          f"every {every} updates and at the end of training ({run_info['effective']['checks_full_run']}"
                          f" checks in a full run); early stopping after {patience} checks")
        + f" without improvement > {cfg['min_delta']} in val T@1")
    dark_luma = dark_luma_of(cfg)
    v2 = cfg["augmentation"] == "cctv_v2"
    log(f"augmentation: {cfg['augmentation']} (hflip p {cfg['hflip_p']}"
        + (f"; then frame-rate drop, blur, downscale, colour, JPEG, each p {cfg['cctv_aug']['p']})"
           if cfg["augmentation"] == "cctv" else
           f"; then frame-rate drop, blur, downscale, colour, JPEG, each p {CCTV_V2['p']}, at most "
           f"{CCTV_V2['max_degradations']} per window; profile in code: {json.dumps(CCTV_V2)})" if v2 else ")")
        + "; validation not augmented")
    if v2:
        log(f"cctv_v2 protected windows (unaugmented mean luma < {dark_luma:g}): at most "
            f"{CCTV_V2['max_degradations_dark']} degradation, brightness {CCTV_V2['brightness_dark']}, contrast "
            f"{CCTV_V2['contrast_dark']}, blur {CCTV_V2['blur_sigma_dark']}, downscale {CCTV_V2['downscale_dark']}, "
            f"JPEG {CCTV_V2['jpeg_quality_dark']}; cctv_aug in the config is not used; counts logged at every "
            "validation check")
    if dark_luma is not None and not v2:
        log(f"night guard: windows with mean luma < {dark_luma:g} before the colour shift draw brightness from "
            f"[{NIGHT_GUARD_MIN_BRIGHTNESS:g}, {cfg['cctv_aug']['brightness'][1]:g}] instead of "
            f"{cfg['cctv_aug']['brightness']}; counts logged at every validation check")
    if dark_luma is not None:  # weather of every training row, for the night guard / cctv_v2 counts (both profiles)
        weather_of = pd.read_csv(SPLIT_CSV, usecols=["rgb_path", "weather"]).assign(
            stem=lambda d: d["rgb_path"].map(stem_of)).set_index("stem")["weather"]
        row_weather = train_rows["stem"].map(weather_of).to_numpy()
        if pd.isna(row_weather).any():
            fail("night guard: weather missing for some training clips in synthetic_split.csv")
    guard_counts: dict[str, list[int]] = {}  # weather -> [colour-shifted windows, guard fired], since the last check
    if args.max_hours is not None:
        log(f"--max-hours {args.max_hours}: stops after the first update past that time, saving step_latest.pt")

    # ---------------- data loaders ----------------
    mean, std = list(proc.image_mean), list(proc.image_std)
    train_ds = WindowDataset(train_rows["stem"], train_rows["k0"], train_rows["target"], train_rows["w_type"],
                             mean, std, cfg["frames"], cfg["model_fps"], cfg["lru_size"], cfg["augmentation"],
                             cfg["cctv_aug"], dark_luma)
    val_ds = WindowDataset(val_rows["stem"], val_rows["k0"], val_rows["label_B"].clip(lower=0), None,
                           mean, std, cfg["frames"], cfg["model_fps"], cfg["lru_size"])
    train_sampler, val_sampler = ListSampler(), ListSampler()
    train_loader = make_loader(train_ds, mb, cfg["num_workers"], train_sampler)
    val_loader = make_loader(val_ds, cfg["val_batch"], cfg["num_workers"], val_sampler)

    # ---------------- model, optimizer, schedule ----------------
    model = build_model(cfg, device, dp_field)
    model.train()
    n_layers = model.config.num_hidden_layers
    optimizer = torch.optim.AdamW(param_groups(model, cfg, lr0), betas=tuple(cfg["betas"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(sizes["warmup_updates"], sizes["total_updates"]))
    criterion = nn.BCEWithLogitsLoss(reduction="none",
                                     pos_weight=torch.tensor(lw["pos_weight"], dtype=torch.float32, device=device))
    peak_lr = {g["layer_id"]: g["initial_lr"] for g in optimizer.param_groups}
    log(f"model {cfg['model_id']}: {sum(p.numel() for p in model.parameters()):,} parameters; "
        f"{n_batches} micro-batches = {updates_per_epoch} updates per epoch x {max_epochs} epochs = "
        f"{sizes['total_updates']} updates; warmup {sizes['warmup_updates']} updates "
        f"({cfg['warmup_fraction']:g} of total), then cosine to 0")
    log(f"lr: base_lr {cfg['base_lr']:g} x effective batch {mb * ga} / 256 = {lr0:.4g}; layer decay "
        f"{cfg['layer_decay']} over layer ids 0..{n_layers + 1}; {len(optimizer.param_groups)} param groups; "
        f"AdamW betas {tuple(cfg['betas'])}, weight decay {cfg['weight_decay']} (none on biases and LayerNorm)")
    log(f"lr (peak, after warmup): head (layer {n_layers + 1}) {peak_lr[n_layers + 1]:.4g}, top block "
        f"(layer {n_layers}) {peak_lr[n_layers]:.4g}, embeddings (layer 0) {peak_lr[0]:.4g}")

    # loss_sum .. oom_fallbacks count the updates since the last validation check
    state = {"epoch": 0, "batches_done": 0, "global_step": 0, "best": -math.inf, "no_improve": 0,
             "loss_sum": 0.0, "loss_n": 0, "wait": 0.0, "train_time": 0.0, "windows": 0, "oom_fallbacks": 0,
             "lr_top_used": None, "lr_head_used": None,  # lr of the last optimizer step
             "checks_done": 0, "last_check_step": -1}

    def save_ckpt(path: Path) -> None:
        atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "state": dict(state), "rng": rng_states(),
                     "run_hash": run_hash, "config_hash": cfg_hash, "data_hashes": data_hashes}, path)

    if args.resume:
        name, ck = resume_ck
        resume_ck = None
        if ck.get("run_hash") != run_hash or ck["data_hashes"] != data_hashes:
            fail(f"--resume: {name} was made with a different config, fold, seed or data")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        state.update(ck["state"])
        set_rng_states(ck["rng"])
        log(f"resumed from {ckpt_dir / name}: epoch {state['epoch']}, batches_done {state['batches_done']}, "
            f"global step {state['global_step']}, best {state['best']:.4f}; skipping {state['batches_done']} "
            f"of {n_batches} batches already applied in epoch {state['epoch']}; validation checks done "
            f"{state['checks_done']} (last at update {state['last_check_step']})")
        del ck

    metrics_path = out_dir / "metrics.csv"
    if args.resume and metrics_path.is_file():  # drop rows of checks made after the checkpoint (they are redone)
        m = pd.read_csv(metrics_path)
        if "check_idx" in m and (m["check_idx"] >= state["checks_done"]).any():
            m[m["check_idx"] < state["checks_done"]].to_csv(metrics_path, index=False)
            log(f"metrics.csv: removed {int((m['check_idx'] >= state['checks_done']).sum())} row(s) of checks after "
                "the checkpoint; they are validated again")
    step_losses: list[float] = []  # unscaled mean loss per optimizer step
    resume_argv = [a for a in sys.argv[1:]]
    if "--resume" not in resume_argv:
        resume_argv.append("--resume")
    resume_cmd = subprocess.list2cmdline([".venv-ml\\Scripts\\python.exe", Path(sys.argv[0]).name] + resume_argv)
    guard = InterruptGuard()
    gpu = GpuGuard(cfg["gpu_pause_temp_c"], cfg["gpu_resume_temp_c"], log)
    t, c = gpu.read()
    if not math.isnan(t):
        log(f"GPU guard: pause at >= {cfg['gpu_pause_temp_c']} C, resume at <= {cfg['gpu_resume_temp_c']} C; "
            f"now {t:.0f} C, SM clock {c:.0f} MHz")

    def save_and_exit(reason: str, code: int) -> NoReturn:
        """Save step_latest.pt at the last optimizer-step boundary, print the resume command and exit."""
        optimizer.zero_grad(set_to_none=True)  # drop gradients of an unfinished accumulation group
        log(f"\n{reason}: epoch {state['epoch']}, batches_done {state['batches_done']}/{n_batches}, "
            f"global step {state['global_step']}; saving {ckpt_dir / 'step_latest.pt'} ...")
        save_ckpt(ckpt_dir / "step_latest.pt")
        log(f"saved {ckpt_dir / 'step_latest.pt'} (epoch {state['epoch']}, batches_done "
            f"{state['batches_done']}, global step {state['global_step']})")
        log(f"resume with: {resume_cmd}")
        log.close()
        sys.exit(code)

    def validate_check(since: float, epoch_end: bool) -> bool:
        """One validation check; returns True if early stopping triggers. Once-per-epoch checks (v1) are
        recorded together with the end of the epoch and last.pt; checks every N updates together with
        step_latest.pt, so a resumed run continues the check schedule exactly."""
        peak_train = torch.cuda.max_memory_reserved(device) / 2 ** 30
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        val_loss, T, win_df, pred, val_ooms = validate(model, val_loader, val_sampler, val_rows, cfg, device, log)
        peak_val = torch.cuda.max_memory_reserved(device) / 2 ** 30
        torch.cuda.empty_cache()
        gpu.sample()
        state["oom_fallbacks"] += val_ooms
        minutes = (time.perf_counter() - since) / 60
        with guard.critical():  # metrics row, best.pt, check counters and the checkpoint are recorded together
            stop_early = record_check(val_loss, T, win_df, pred, minutes, peak_train, peak_val)
            if dark_luma is not None:
                log_guard()
            if epoch_end:
                end_epoch()
            else:
                save_ckpt(ckpt_dir / "step_latest.pt")
        torch.cuda.reset_peak_memory_stats(device)
        gpu.reset_epoch()
        return stop_early

    def log_guard() -> None:
        """Night guard counts of the windows loaded since the last check (not checkpointed: after a resume they
        start again from 0)."""
        col = sum(v[0] for v in guard_counts.values())
        fired = sum(v[1] for v in guard_counts.values())
        by_w = ", ".join(f"{w} {v[1]}/{v[0]}" for w, v in sorted(guard_counts.items()))
        if v2:
            log(f"  cctv_v2 since the last check: {fired} of {col} training windows protected "
                f"({100 * fired / max(1, col):.1f}%); by weather (protected/windows): {by_w or 'none'}")
        else:
            log(f"  night guard since the last check: fired {fired} of {col} colour-shifted windows "
                f"({100 * fired / max(1, col):.1f}%); by weather (fired/colour-shifted): {by_w or 'none'}")
        guard_counts.clear()

    def end_epoch() -> None:
        state.update(epoch=state["epoch"] + 1, batches_done=0)
        save_ckpt(ckpt_dir / "last.pt")

    def record_check(val_loss: float, T: dict, win_df: pd.DataFrame, pred: pd.DataFrame, minutes: float,
                     peak_train: float, peak_val: float) -> bool:
        monitor = T[1.0]
        improved = monitor > state["best"] + cfg["min_delta"]
        if improved:
            state["best"], state["no_improve"] = monitor, 0
        else:
            state["no_improve"] += 1
        row = {
            "epoch": state["epoch"], "global_step": state["global_step"],
            "train_loss": state["loss_sum"] / max(1, state["loss_n"]), "val_loss": val_loss,
            "val_T_0.5": T[0.5], "val_T_1": T[1.0], "val_T_2": T[2.0],
            "val_signed_err_mean": float(pred["t_hat_minus_t_star"].mean()),  # reported only
            "val_signed_err_median": float(pred["t_hat_minus_t_star"].median()),
            "lr_backbone": state["lr_top_used"], "lr_head": state["lr_head_used"], "minutes": minutes,
            "windows_per_s": state["windows"] / max(1e-9, state["wait"] + state["train_time"]),
            "peak_gpu_mem_gb": max(peak_train, peak_val),
            "peak_train_gb": peak_train, "peak_val_gb": peak_val,
            "data_wait_share": state["wait"] / max(1e-9, state["wait"] + state["train_time"]),
            "best": improved,
            "oom_fallbacks": state["oom_fallbacks"], "gpu_temp_max": gpu.temp_max, "gpu_clock_min": gpu.clock_min,
            "val_within_2s": float(pred["within_2s"].mean()),  # reported only
            "check_idx": state["checks_done"], "epoch_fraction": state["global_step"] / updates_per_epoch,
        }
        pd.DataFrame([row]).to_csv(metrics_path, mode="a", header=not metrics_path.exists(), index=False)
        head_ = (f"epoch {state['epoch']} done" if every is None else
                 f"check {state['checks_done']} (update {state['global_step']}, epoch {row['epoch_fraction']:.2f})")
        log(f"{head_}: train loss {row['train_loss']:.4f}, val loss {val_loss:.4f}, "
            f"val T@0.5/1/2 {T[0.5]:.4f} / {T[1.0]:.4f} / {T[2.0]:.4f}, "
            f"signed err mean/median {row['val_signed_err_mean']:+.3f} / {row['val_signed_err_median']:+.3f} s, "
            f"within 2 s {row['val_within_2s']:.3f}, {minutes:.2f} min, "
            f"{row['windows_per_s']:.1f} windows/s, data wait {100 * row['data_wait_share']:.0f}%, "
            f"peak {row['peak_gpu_mem_gb']:.2f} GB (train {peak_train:.2f}, val {peak_val:.2f}), "
            f"GPU max {gpu.temp_max:.0f} C / min {gpu.clock_min:.0f} MHz, OOM fallbacks {state['oom_fallbacks']}"
            f"{'  [best]' if improved else ''}")
        if improved:
            win_df[["rgb_path", "window_idx", "t_centre", "label_B", "score"]].to_csv(
                out_dir / "val_window_scores_best.csv", index=False)
            pred.to_csv(out_dir / "val_clip_predictions_best.csv", index=False)
            atomic_save({"model": model.state_dict(), "epoch": state["epoch"], "val_T_1": monitor,
                         "check_idx": state["checks_done"], "global_step": state["global_step"],
                         "run_hash": run_hash, "config_hash": cfg_hash, "data_hashes": data_hashes},
                        ckpt_dir / "best.pt")
        # the next check interval starts fresh
        state.update(loss_sum=0.0, loss_n=0, wait=0.0, train_time=0.0, windows=0, oom_fallbacks=0,
                     checks_done=state["checks_done"] + 1, last_check_step=state["global_step"])
        stop_early = state["no_improve"] >= patience
        if stop_early:
            log(f"early stopping: no improvement > {cfg['min_delta']} in val T@1 for {patience} "
                f"{'epochs' if every is None else 'checks'}")
        return stop_early

    # ---------------- training loop ----------------
    t_run = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    gpu.reset_epoch()
    t_check = time.perf_counter()  # start of the current check interval
    try:
        stop = False
        while not stop:
            epoch = state["epoch"]
            if epoch >= max_epochs:  # never start an epoch past the schedule
                fail(f"internal error: about to start epoch {epoch} of {max_epochs}")
            if every is not None and state["global_step"] >= (state["checks_done"] + 1) * every:
                # resumed from a checkpoint saved while this check was validating: redo it first
                stop = validate_check(t_check, epoch_end=False)
                t_check = time.perf_counter()
                if stop:
                    break
            codes = epoch_codes(len(train_rows), args.seed, epoch, cfg["hflip_p"])
            start_b = state["batches_done"]
            train_sampler.set(codes[start_b * mb:])
            t_last = time.perf_counter()
            group: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []  # CPU copies, for an OOM recompute
            group_losses: list[float] = []
            for i, (x, y, w, rows_b, guard_b) in enumerate(train_loader):
                b = start_b + i
                t_got = time.perf_counter()
                state["wait"] += t_got - t_last
                if dark_luma is not None:
                    for r, gf in zip(rows_b.tolist(), guard_b.tolist()):
                        if gf:
                            cnt = guard_counts.setdefault(row_weather[r], [0, 0])
                            cnt[0] += 1
                            cnt[1] += gf == 2
                paused = 0.0
                group_start = (b // ga) * ga
                group_size = min(ga, n_batches - group_start)
                group.append((x, y, w))
                loss_value = None
                try:
                    loss_value = fwd_bwd(model, criterion, x, y, w, device, 1.0 / group_size, len(y))
                except torch.cuda.OutOfMemoryError:
                    pass
                if loss_value is None:  # recover outside the except block, so the failed attempt's memory is released
                    state["oom_fallbacks"] += 1
                    group_losses = recompute_group(model, criterion, optimizer, group, device, 1.0 / group_size,
                                                   log, f"epoch {epoch} batch {b + 1}/{n_batches}")
                else:
                    group_losses.append(loss_value)
                if b + 1 - group_start == group_size:  # end of an accumulation group
                    with guard.critical():  # Ctrl+C waits until this step is fully recorded
                        if state["global_step"] >= sizes["total_updates"]:  # never update past the schedule
                            fail(f"internal error: update {state['global_step'] + 1} > {sizes['total_updates']}")
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                        lr_now = lr_by_layer(optimizer)
                        state["lr_top_used"], state["lr_head_used"] = lr_now[n_layers], lr_now[n_layers + 1]
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        n_group = sum(len(yy) for _, yy, _ in group)
                        state["global_step"] += 1
                        state["batches_done"] = b + 1
                        state["loss_sum"] += sum(lv * len(yy) for lv, (_, yy, _) in zip(group_losses, group))
                        state["loss_n"] += n_group
                        state["windows"] += n_group
                        step_losses.append(float(np.mean(group_losses)))
                        group, group_losses = [], []
                        gs = state["global_step"]
                        temp, clock = gpu.sample()
                        if gs % log_every == 0:
                            log(f"  step {gs:>6}  epoch {epoch}  batch {b + 1}/{n_batches}  "
                                f"loss {np.mean(step_losses[-log_every:]):.4f} (mean of last "
                                f"{min(log_every, len(step_losses))} steps)  "
                                f"lr {state['lr_top_used']:.2e}  GPU {temp:.0f} C {clock:.0f} MHz")
                        if gs % step_ckpt_every == 0:
                            save_ckpt(ckpt_dir / "step_latest.pt")
                    paused = gpu.maybe_pause(temp, f"update {gs}")
                    if args.max_hours is not None and time.monotonic() - t_run >= args.max_hours * 3600:
                        with guard.critical():
                            save_and_exit(f"--max-hours {args.max_hours} reached", 0)
                    if every is not None and gs % every == 0:
                        state["train_time"] += time.perf_counter() - t_got - paused  # this interval's share
                        stop = validate_check(t_check, epoch_end=False)
                        t_check = t_got = time.perf_counter()
                        paused = 0.0
                        if stop:
                            break
                t_last = time.perf_counter()
                state["train_time"] += t_last - t_got - paused
            if stop:
                break
            torch.cuda.synchronize()
            if every is None:  # v1: validate once per epoch
                stop = validate_check(t_check, epoch_end=True) or state["epoch"] >= max_epochs
                t_check = time.perf_counter()
            else:
                if state["epoch"] == max_epochs - 1 and state["last_check_step"] != state["global_step"]:
                    stop = validate_check(t_check, epoch_end=True)  # the check at the end of training
                    t_check = time.perf_counter()
                else:
                    with guard.critical():
                        end_epoch()
                stop = stop or state["epoch"] >= max_epochs
    except KeyboardInterrupt:
        # state only changes at optimizer-step boundaries, so it already counts fully applied batches
        save_and_exit("interrupted", 130)
    except OOMAbort as e:
        save_and_exit(f"ERROR: {e}; reduce micro_batch / val_batch in the config", 1)

    log(f"training finished: best val T@1 {state['best']:.4f}; outputs in {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--fold", type=int, choices=[0, 1, 2])
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke-test", action="store_true",
                    help=f"first {SMOKE['train_clips']} train / {SMOKE['val_clips']} val clips, max "
                         f"{SMOKE['max_epochs']} epochs, step checkpoint every {SMOKE['step_ckpt_every']} updates, "
                         "run name suffixed _smoke")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="after H hours, finish the current update, save step_latest.pt and exit 0")
    ap.add_argument("--plan", action="store_true", help="print counts and schedule for all folds; no model, no GPU")
    ap.add_argument("--preview-aug", type=int, default=None, metavar="N",
                    help="save N training windows of the fold, original vs augmented, as a PNG in runs/aug_preview; "
                         "no training")
    args = ap.parse_args()

    cfg_path = args.config.resolve()
    cfg, cfg_hash = load_config(cfg_path)
    if args.plan:
        plan(cfg, cfg_path, cfg_hash)
        return
    if args.fold is None:
        ap.error("--fold is required (except with --plan)")
    if args.preview_aug is not None:
        preview_aug(args, cfg, cfg_path)
        return
    train(args, cfg, cfg_path, cfg_hash)


if __name__ == "__main__":
    main()
