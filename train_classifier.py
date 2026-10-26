"""Train a window-level accident classifier (VideoMAE-base) on synthetic data, one fold.

SYNTHETIC DATA ONLY: never opens real videos, metadata-real.csv or real_test_set.csv.

Data:   synthetic_window_labels.csv (hash-checked against split_manifest.json) and the
        frame cache cache/frames224_v1/<clip>.npz (build_frame_cache.py). Window frames =
        grid positions 8*t_start .. 8*t_start+15, normalised with the VideoMAE processor
        mean/std (frames are already 224x224 full-frame squashes; no further resize/crop).
Fold k: train = label_B in {0, 1} and val_fold != k; validation = ALL windows with val_fold == k.
Model:  MCG-NJU/videomae-base-finetuned-kinetics, classifier -> Linear(hidden, 1),
        BCEWithLogitsLoss on label_B, bf16 autocast.
Val:    per clip, sort by t_centre, 3-window moving average, t_hat = t_centre at the peak,
        confidence = peak - max smoothed score outside +-2 windows (evaluation document,
        Section 6.4); T at sigma_t = 0.5 / 1 / 2 s with accs_scorer.temporal.
Early stopping on val T@1 s (patience 2, min delta 0.002). Checkpoints: last.pt (every
epoch), step_latest.pt (every 500 optimizer steps and on Ctrl+C), best.pt (weights).

--overfit: train and validate on the first 2 training clips of the fold, no augmentation,
constant lr 1e-4, 100 optimizer steps, then PASS if the mean training loss of the last
10 optimizer steps is < 0.05. Outputs go to runs/overfit_test/. To keep the test fast,
checkpoints are written only at the end of an overfit run.

Usage (training environment):
    .venv-ml\\Scripts\\python.exe train_classifier.py --fold 0 [--run-name NAME] [--resume] ...
    .venv-ml\\Scripts\\python.exe train_classifier.py --fold 0 --overfit
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import OrderedDict  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

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
SPLIT_VERSION = "v2_rebalanced"
MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"
SIGMAS_T = (0.5, 1.0, 2.0)

DEFAULTS = {
    "model_id": MODEL_ID,
    "frames": 16,
    "model_fps": 8,
    "batch_size": 8,
    "grad_accum": 2,
    "val_batch_size": 32,
    "num_workers": 4,
    "lru_size": 8,
    "amp": "bf16",
    "grad_ckpt": False,
    "grad_clip": 1.0,
    "lr_backbone": 3e-5,
    "lr_head": 3e-4,
    "weight_decay": 0.05,
    "warmup_steps": 500,
    "schedule": "linear warmup, then cosine to 0 at max_epochs * steps_per_epoch",
    "max_epochs": 8,
    "seed": 0,
    "aug_hflip": True,
    "hflip_p": 0.5,
    "monitor": "val_T_1",
    "patience": 2,
    "min_delta": 0.002,
    "step_ckpt_every": 500,
    "log_every": 50,
    "smooth_window": 3,
    "confidence_exclude": 2,
    "label": "label_B",
}
OVERFIT = {"clips": 2, "lr": 1e-4, "steps": 100, "log_every": 10, "pass_loss": 0.05, "loss_window": 10}


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
class Logger:
    def __init__(self, path):
        self.fh = open(path, "a", encoding="utf-8")

    def __call__(self, text=""):
        print(text, flush=True)
        self.fh.write(str(text) + "\n")
        self.fh.flush()


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stem_of(p):
    return Path(str(p).replace("\\", "/")).stem


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_states():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def set_rng_states(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    torch.cuda.set_rng_state_all(s["cuda"])


def atomic_save(obj, path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def git_info():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True,
                                    text=True, check=True).stdout.strip())
        return head, dirty
    except Exception as e:  # git missing or not a repo
        return f"unavailable ({type(e).__name__})", None


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class WindowDataset(Dataset):
    """Index i encodes (row, flip) as i = 2*row + flip, so flips are decided in the main process."""

    def __init__(self, stems, k0, labels, mean, std, frames, lru_size):
        self.stems, self.k0, self.labels = list(stems), np.asarray(k0, int), np.asarray(labels, np.float32)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.frames, self.lru_size = frames, lru_size
        self._lru = OrderedDict()

    def __len__(self):
        return len(self.stems)

    def _clip(self, stem):
        if stem in self._lru:
            self._lru.move_to_end(stem)
            return self._lru[stem]
        with np.load(CACHE_DIR / f"{stem}.npz") as z:
            item = {"offsets": z["offsets"], "data": z["data"]}
        self._lru[stem] = item
        if len(self._lru) > self.lru_size:
            self._lru.popitem(last=False)
        return item

    def __getitem__(self, code):
        row, flip = divmod(int(code), 2)
        z = self._clip(self.stems[row])
        k0 = self.k0[row]
        x = np.stack([read_cached_frame(z, k0 + j) for j in range(self.frames)])  # (T, H, W, 3) uint8
        if flip:
            x = x[:, :, ::-1, :]
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float().div_(255.0)
        x = (x - self.mean) / self.std
        return x, torch.tensor(self.labels[row]), row


class ListSampler(Sampler):
    """Yields a list set by the main process before each epoch (works with persistent workers)."""

    def __init__(self):
        self.items = []

    def set(self, items):
        self.items = list(items)

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)


def epoch_codes(n, seed, epoch, aug, p):
    g = torch.Generator()
    g.manual_seed(seed * 1000 + epoch)
    perm = torch.randperm(n, generator=g)
    flips = (torch.rand(n, generator=g) < p) if aug else torch.zeros(n, dtype=torch.bool)
    return (perm * 2 + flips[perm].long()).tolist()


def make_loader(ds, batch, workers, sampler):
    kw = dict(batch_size=batch, sampler=sampler, num_workers=workers, pin_memory=True, drop_last=False)
    if workers > 0:
        kw.update(persistent_workers=True)
    return DataLoader(ds, **kw)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def clip_predictions(val_rows, scores, smooth_window, exclude):
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
    for s_ in SIGMAS_T:
        pred[f"T_i_{s_:g}"] = temporal_per_clip(pred["t_hat"], pred["t_star"], s_)
    return df, pred


@torch.no_grad()
def validate(model, loader, sampler, val_rows, cfg, device):
    model.eval()
    sampler.set(range(0, 2 * len(val_rows), 2))  # rows in order, no flip
    logits = np.empty(len(val_rows), dtype=np.float32)
    for x, _, rows in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(pixel_values=x.to(device, non_blocking=True)).logits.squeeze(-1)
        logits[rows.numpy()] = out.float().cpu().numpy()
    scores = 1.0 / (1.0 + np.exp(-logits))
    lab = val_rows["label_B"].to_numpy()
    m = lab >= 0
    val_loss = float(nn.functional.binary_cross_entropy_with_logits(
        torch.from_numpy(logits[m]), torch.from_numpy(lab[m].astype(np.float32)))) if m.any() else float("nan")
    win_df, pred = clip_predictions(val_rows, scores, cfg["smooth_window"], cfg["confidence_exclude"])
    T = {s: temporal(pred["t_hat"], pred["t_star"], s) for s in SIGMAS_T}
    model.train()
    return val_loss, T, win_df, pred


# --------------------------------------------------------------------------- #
# Model, optimiser, schedule
# --------------------------------------------------------------------------- #
def build_model(cfg, device):
    from transformers import VideoMAEForVideoClassification
    model = VideoMAEForVideoClassification.from_pretrained(cfg["model_id"])
    model.classifier = nn.Linear(model.config.hidden_size, 1)
    if cfg["grad_ckpt"]:
        model.gradient_checkpointing_enable()
    return model.to(device)


def param_groups(model, cfg, overfit):
    groups = {(h, d): [] for h in (False, True) for d in (False, True)}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        head = name.startswith("classifier.")
        no_decay = p.ndim == 1 or name.endswith(".bias")  # biases and LayerNorm weights
        groups[(head, not no_decay)].append(p)
    out = []
    for (head, decay), params in groups.items():
        if params:
            lr = OVERFIT["lr"] if overfit else (cfg["lr_head"] if head else cfg["lr_backbone"])
            out.append({"params": params, "lr": lr, "weight_decay": cfg["weight_decay"] if decay else 0.0,
                        "name": f"{'head' if head else 'backbone'}_{'decay' if decay else 'no_decay'}"})
    return out


def make_lr_lambda(warmup, total, overfit):
    def f(step):
        if overfit:
            return 1.0
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        span = max(1, total - warmup)
        progress = min(1.0, (step - warmup) / span)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return f


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fold", type=int, required=True, choices=[0, 1, 2])
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many optimizer steps")
    ap.add_argument("--limit-train-clips", type=int, default=None)
    ap.add_argument("--limit-val-clips", type=int, default=None)
    ap.add_argument("--overfit", action="store_true")
    args = ap.parse_args()

    cfg = dict(DEFAULTS)
    cfg["fold"] = args.fold
    if args.max_epochs is not None:
        cfg["max_epochs"] = args.max_epochs
    if args.seed is not None:
        cfg["seed"] = args.seed
    cfg["aug_hflip"] = not (args.no_aug or args.overfit)
    cfg["max_steps"] = OVERFIT["steps"] if args.overfit else args.max_steps
    cfg["limit_train_clips"] = args.limit_train_clips
    cfg["limit_val_clips"] = args.limit_val_clips
    cfg["overfit"] = args.overfit
    if args.overfit:
        cfg.update(lr_backbone=OVERFIT["lr"], lr_head=OVERFIT["lr"], warmup_steps=0, max_epochs=10 ** 6,
                   log_every=OVERFIT["log_every"], schedule="constant lr 1e-4 (overfit test)",
                   early_stopping=False)
    run_name = "overfit_test" if args.overfit else (args.run_name or f"videomae_labelB_seed{cfg['seed']}")
    cfg["run_name"] = run_name
    hash_keys = sorted(k for k in cfg if k != "run_name")
    cfg_hash = hashlib.sha256(json.dumps({k: cfg[k] for k in hash_keys}, sort_keys=True).encode()).hexdigest()

    out_dir = RUNS_DIR / run_name / f"fold_{args.fold}"
    ckpt_dir = out_dir / "checkpoints"
    if out_dir.exists() and not args.resume and not args.overfit and any(out_dir.iterdir()):
        fail(f"{out_dir} already exists; use --resume or a different --run-name")
    if args.overfit and out_dir.exists():
        for f in ("train_log.txt", "metrics.csv"):  # the overfit test is always a fresh run
            if (out_dir / f).exists():
                (out_dir / f).unlink()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "train_log.txt")

    # ---------------- input checks ----------------
    manifest = json.loads(MANIFEST.read_text(encoding="ascii"))
    labels_sha = sha256_file(LABELS_CSV)
    if labels_sha != manifest.get("window_labels_sha256"):
        fail(f"{LABELS_CSV.name} SHA-256 {labels_sha} != window_labels_sha256 {manifest.get('window_labels_sha256')}")
    data = pd.read_csv(LABELS_CSV)
    if not (data["split_version"] == SPLIT_VERSION).all():
        fail(f"split_version is not {SPLIT_VERSION} in every row: {sorted(data['split_version'].unique())}")
    data["stem"] = data["rgb_path"].map(stem_of)
    data["k0"] = np.round(data["t_start"] * cfg["model_fps"]).astype(int)
    if not np.allclose(data["k0"] / cfg["model_fps"], data["t_start"]):
        fail("t_start values are not multiples of 1/model_fps")

    train_all = data[(data["val_fold"] != args.fold) & data["label_B"].isin([0, 1])]
    val_all = data[data["val_fold"] == args.fold]
    if args.overfit:
        clips = sorted(train_all["rgb_path"].unique())[:OVERFIT["clips"]]
        train_rows = train_all[train_all["rgb_path"].isin(clips)]
        val_rows = data[data["rgb_path"].isin(clips)]
    else:
        tr_clips = sorted(train_all["rgb_path"].unique())
        va_clips = sorted(val_all["rgb_path"].unique())
        if args.limit_train_clips:
            tr_clips = tr_clips[:args.limit_train_clips]
        if args.limit_val_clips:
            va_clips = va_clips[:args.limit_val_clips]
        train_rows = train_all[train_all["rgb_path"].isin(tr_clips)]
        val_rows = val_all[val_all["rgb_path"].isin(va_clips)]
    train_rows = train_rows.sort_values(["rgb_path", "window_idx"]).reset_index(drop=True)
    val_rows = val_rows.sort_values(["rgb_path", "window_idx"]).reset_index(drop=True)
    if train_rows.empty or val_rows.empty:
        fail("empty training or validation set")

    cache_manifest = json.loads(CACHE_MANIFEST.read_text(encoding="ascii")) if CACHE_MANIFEST.is_file() else {}
    partial = args.overfit or args.limit_train_clips or args.limit_val_clips
    if not partial and cache_manifest.get("complete") is not True:
        fail(f"{CACHE_MANIFEST} does not say complete: true")
    needed = sorted(set(train_rows["stem"]) | set(val_rows["stem"]))
    missing = [s for s in needed if not (CACHE_DIR / f"{s}.npz").is_file()]
    if missing:
        fail(f"{len(missing)} cache files missing, e.g. {missing[:3]}")
    if cache_manifest.get("windows_sha256") != manifest.get("windows_sha256"):
        fail("cache_manifest windows_sha256 differs from split_manifest windows_sha256")

    data_hashes = {"window_labels_sha256": labels_sha, "split_sha256": sha256_file(SPLIT_CSV),
                   "cache_manifest_sha256": sha256_file(CACHE_MANIFEST)}
    if data_hashes["split_sha256"] != manifest.get("split_sha256"):
        fail("synthetic_split.csv SHA-256 differs from split_sha256 in split_manifest.json")

    # ---------------- environment ----------------
    if not torch.cuda.is_available():
        fail("CUDA is not available")
    device = torch.device("cuda")
    import transformers
    from transformers import VideoMAEImageProcessor
    transformers.utils.logging.disable_progress_bar()
    proc = VideoMAEImageProcessor.from_pretrained(cfg["model_id"])
    head, dirty = git_info()
    seed_everything(cfg["seed"])

    run_info = {
        "config": cfg, "config_hash": cfg_hash, "data_hashes": data_hashes,
        "normalisation": {"mean": list(proc.image_mean), "std": list(proc.image_std),
                          "note": "frames are cached 224x224 full-frame squashes; processor resize/crop not used"},
        "n_train_windows": len(train_rows), "n_train_clips": int(train_rows["rgb_path"].nunique()),
        "train_label_counts": {str(k): int(v) for k, v in train_rows["label_B"].value_counts().sort_index().items()},
        "n_val_windows": len(val_rows), "n_val_clips": int(val_rows["rgb_path"].nunique()),
        "git_commit": head, "git_dirty": dirty, "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": transformers.__version__, "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0), "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    cfg_path = out_dir / "config.json"
    if args.resume and cfg_path.is_file():
        old = json.loads(cfg_path.read_text(encoding="utf-8"))
        if old.get("config_hash") != cfg_hash or old.get("data_hashes") != data_hashes:
            fail("config or data hashes differ from the run being resumed")
    else:
        cfg_path.write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")

    log(f"=== train_classifier.py  fold {args.fold}  run {run_name}  {'OVERFIT TEST' if args.overfit else ''}")
    log(f"time {run_info['created_at']}; GPU {run_info['gpu']}; torch {torch.__version__}; "
        f"transformers {transformers.__version__}; git {head}{' (dirty)' if dirty else ''}")
    log(f"data: window labels {labels_sha[:12]}..., split {data_hashes['split_sha256'][:12]}..., "
        f"cache manifest {data_hashes['cache_manifest_sha256'][:12]}...; split_version {SPLIT_VERSION}")
    log(f"train: {len(train_rows)} windows from {run_info['n_train_clips']} clips, label_B counts "
        f"{run_info['train_label_counts']}; val: {len(val_rows)} windows from {run_info['n_val_clips']} clips")
    if args.overfit:
        log(f"overfit clips: {', '.join(stem_of(c) for c in sorted(train_rows['rgb_path'].unique()))}")
    log(f"config hash {cfg_hash[:12]}...; output {out_dir}")

    # ---------------- data loaders ----------------
    mean, std = list(proc.image_mean), list(proc.image_std)
    train_ds = WindowDataset(train_rows["stem"], train_rows["k0"], train_rows["label_B"], mean, std,
                             cfg["frames"], cfg["lru_size"])
    val_ds = WindowDataset(val_rows["stem"], val_rows["k0"], val_rows["label_B"].clip(lower=0), mean, std,
                           cfg["frames"], cfg["lru_size"])
    train_sampler, val_sampler = ListSampler(), ListSampler()
    train_loader = make_loader(train_ds, cfg["batch_size"], cfg["num_workers"], train_sampler)
    val_loader = make_loader(val_ds, cfg["val_batch_size"], cfg["num_workers"], val_sampler)

    n_batches = math.ceil(len(train_rows) / cfg["batch_size"])
    steps_per_epoch = math.ceil(n_batches / cfg["grad_accum"])
    total_steps = cfg["max_epochs"] * steps_per_epoch

    # ---------------- model ----------------
    model = build_model(cfg, device)
    model.train()
    log(f"model {cfg['model_id']}: {sum(p.numel() for p in model.parameters()):,} parameters; "
        f"{n_batches} batches = {steps_per_epoch} optimizer steps per epoch"
        + ("" if args.overfit else f"; {total_steps} scheduled steps"))
    optimizer = torch.optim.AdamW(param_groups(model, cfg, args.overfit))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(cfg["warmup_steps"], total_steps, args.overfit))
    criterion = nn.BCEWithLogitsLoss()

    state = {"epoch": 0, "batches_done": 0, "global_step": 0, "best": -math.inf, "no_improve": 0,
             "loss_sum": 0.0, "loss_n": 0, "wait": 0.0, "train_time": 0.0, "windows": 0}

    def save_ckpt(path):
        atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "state": dict(state), "rng": rng_states(),
                     "config_hash": cfg_hash, "data_hashes": data_hashes}, path)

    if args.resume:
        cands = []
        for name in ("last.pt", "step_latest.pt"):
            p = ckpt_dir / name
            if p.is_file():
                c = torch.load(p, map_location="cpu", weights_only=False)
                cands.append((c["state"]["global_step"], name, c))
        if not cands:
            fail(f"--resume: no checkpoint in {ckpt_dir}")
        _, name, ck = max(cands, key=lambda t: t[0])
        if ck["config_hash"] != cfg_hash or ck["data_hashes"] != data_hashes:
            fail(f"--resume: {name} was made with a different config or data")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        state.update(ck["state"])
        set_rng_states(ck["rng"])
        log(f"resumed from {name}: epoch {state['epoch']}, batch {state['batches_done']}, "
            f"global step {state['global_step']}, best {state['best']:.4f}")
        del ck

    metrics_path = out_dir / "metrics.csv"
    step_losses = []  # unscaled mean loss per optimizer step
    last_step_batch = state["batches_done"]

    def finish_epoch(epoch_minutes_start):
        """Validate, write metrics, checkpoints; returns True if training should stop."""
        val_loss, T, win_df, pred = validate(model, val_loader, val_sampler, val_rows, cfg, device)
        minutes = (time.perf_counter() - epoch_minutes_start) / 60
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
            "lr_backbone": optimizer.param_groups[0]["lr"], "minutes": minutes,
            "windows_per_s": state["windows"] / max(1e-9, state["train_time"]),
            "peak_gpu_mem_gb": torch.cuda.max_memory_reserved(device) / 2 ** 30,
            "data_wait_share": state["wait"] / max(1e-9, state["train_time"]),
            "best": improved,
        }
        pd.DataFrame([row]).to_csv(metrics_path, mode="a", header=not metrics_path.exists(), index=False)
        log(f"epoch {state['epoch']} done: train loss {row['train_loss']:.4f}, val loss {val_loss:.4f}, "
            f"val T@0.5/1/2 {T[0.5]:.4f} / {T[1.0]:.4f} / {T[2.0]:.4f}, {minutes:.2f} min, "
            f"{row['windows_per_s']:.1f} windows/s, data wait {100 * row['data_wait_share']:.0f}%, "
            f"peak {row['peak_gpu_mem_gb']:.2f} GB{'  [best]' if improved else ''}")
        if improved:
            win_df[["rgb_path", "window_idx", "t_centre", "label_B", "score"]].to_csv(
                out_dir / "val_window_scores_best.csv", index=False)
            pred.to_csv(out_dir / "val_clip_predictions_best.csv", index=False)
            if not args.overfit:
                atomic_save({"model": model.state_dict(), "epoch": state["epoch"], "val_T_1": monitor,
                             "config_hash": cfg_hash, "data_hashes": data_hashes}, ckpt_dir / "best.pt")
        # next epoch starts fresh
        state.update(epoch=state["epoch"] + 1, batches_done=0, loss_sum=0.0, loss_n=0, wait=0.0,
                     train_time=0.0, windows=0)
        if not args.overfit:
            save_ckpt(ckpt_dir / "last.pt")
        stop_steps = cfg["max_steps"] is not None and state["global_step"] >= cfg["max_steps"]
        stop_early = not args.overfit and state["no_improve"] >= cfg["patience"]
        if stop_early:
            log(f"early stopping: no improvement > {cfg['min_delta']} in val T@1 for {cfg['patience']} epochs")
        return stop_steps or stop_early or state["epoch"] >= cfg["max_epochs"], val_loss, T

    # ---------------- training loop ----------------
    val_loss = T = None
    try:
        stop = False
        while not stop:
            epoch = state["epoch"]
            codes = epoch_codes(len(train_rows), cfg["seed"], epoch, cfg["aug_hflip"], cfg["hflip_p"])
            start_b = state["batches_done"]
            train_sampler.set(codes[start_b * cfg["batch_size"]:])
            torch.cuda.reset_peak_memory_stats(device)
            t_epoch = time.perf_counter()
            t_last = time.perf_counter()
            micro_losses = []
            last_step_batch = start_b
            for i, (x, y, _) in enumerate(train_loader):
                b = start_b + i
                t_got = time.perf_counter()
                state["wait"] += t_got - t_last
                group_start = (b // cfg["grad_accum"]) * cfg["grad_accum"]
                group_size = min(cfg["grad_accum"], n_batches - group_start)
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(pixel_values=x).logits.squeeze(-1)
                loss = criterion(logits.float(), y)
                (loss / group_size).backward()
                micro_losses.append(float(loss))
                state["loss_sum"] += float(loss) * len(y)
                state["loss_n"] += len(y)
                state["windows"] += len(y)
                if b + 1 - group_start == group_size:  # end of an accumulation group
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    state["global_step"] += 1
                    state["batches_done"] = b + 1
                    last_step_batch = b + 1
                    step_losses.append(float(np.mean(micro_losses)))
                    micro_losses = []
                    gs = state["global_step"]
                    if gs % cfg["log_every"] == 0:
                        log(f"  step {gs:>6}  epoch {epoch}  batch {b + 1}/{n_batches}  "
                            f"loss {np.mean(step_losses[-cfg['log_every']:]):.4f} (mean of last "
                            f"{min(cfg['log_every'], len(step_losses))} steps)  lr {optimizer.param_groups[0]['lr']:.2e}")
                    if not args.overfit and gs % cfg["step_ckpt_every"] == 0:
                        save_ckpt(ckpt_dir / "step_latest.pt")
                    if cfg["max_steps"] is not None and gs >= cfg["max_steps"]:
                        state["train_time"] += time.perf_counter() - t_got
                        break
                t_last = time.perf_counter()
                state["train_time"] += t_last - t_got
            torch.cuda.synchronize()
            stop, val_loss, T = finish_epoch(t_epoch)
    except KeyboardInterrupt:
        state["batches_done"] = last_step_batch
        save_ckpt(ckpt_dir / "step_latest.pt")
        log(f"\ninterrupted: saved {ckpt_dir / 'step_latest.pt'} (epoch {state['epoch']}, batch "
            f"{state['batches_done']}, global step {state['global_step']})")
        log(f"resume with: .venv-ml\\Scripts\\python.exe train_classifier.py --fold {args.fold} "
            f"--run-name {run_name} --resume" + (" --no-aug" if args.no_aug else ""))
        sys.exit(130)

    if args.overfit:
        save_ckpt(ckpt_dir / "last.pt")
        k = OVERFIT["loss_window"]
        final = float(np.mean(step_losses[-k:]))
        verdict = "PASS" if final < OVERFIT["pass_loss"] else "FAIL"
        log("")
        log(f"OVERFIT RESULT: {verdict}  mean training loss of the last {k} optimizer steps = {final:.4f} "
            f"(threshold {OVERFIT['pass_loss']}); last step loss {step_losses[-1]:.4f}; "
            f"{state['global_step']} optimizer steps")
        log(f"  eval-mode BCE on the same labelled windows (validation pass): {val_loss:.4f}; "
            f"val T@0.5/1/2 on the 2 clips: {T[0.5]:.4f} / {T[1.0]:.4f} / {T[2.0]:.4f}")
        sys.exit(0 if verdict == "PASS" else 1)
    log(f"training finished: best val T@1 {state['best']:.4f}; outputs in {out_dir}")


if __name__ == "__main__":
    main()
