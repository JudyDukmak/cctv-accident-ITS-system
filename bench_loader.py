"""Training data-loader throughput, standard vs cctv augmentation. No model, no GPU (CUDA hidden before torch import).

Uses train_classifier's own WindowDataset, ListSampler, epoch_codes and make_loader with the config's micro_batch and
num_workers (persistent workers, worker_init, default prefetch_factor 2, pin_memory=True as in training; pinning is
skipped without a visible GPU, torch warns about it). The same first B*micro_batch codes of epoch 0 of the fold are
loaded in every setting; the clips' .npz files are read once beforehand so every setting sees a warm OS file cache.
Reported: windows/s from loader creation (includes worker start) and steady state (after the first batch).

Usage: .venv-ml\\Scripts\\python.exe bench_loader.py --config configs\\videomae_v2_C_cctv.json [--fold 0] [--batches 200]
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # no GPU: set before torch is imported

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
import train_classifier as tc  # noqa: E402


def run(ds, codes, batch, workers) -> tuple[float, float, int]:
    sampler = tc.ListSampler()
    sampler.set(codes)
    t0 = time.perf_counter()
    loader = tc.make_loader(ds, batch, workers, sampler)
    n, t1 = 0, None
    for x, *_ in loader:
        if t1 is None:
            t1, n1 = time.perf_counter(), len(x)
        n += len(x)
    t2 = time.perf_counter()
    del loader
    return n / (t2 - t0), (n - n1) / (t2 - t1), n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "videomae_v2_C_cctv.json")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None, help="comma-separated setting names, e.g. standard,cctv_v2")
    ap.add_argument("--config-workers-only", action="store_true", help="skip the 1-worker settings")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", message=".*pin_memory.*")
    cfg, cfg_hash = tc.load_config(args.config.resolve())
    from transformers import VideoMAEImageProcessor
    proc = VideoMAEImageProcessor.from_pretrained(cfg["model_id"])
    data, manifest, _ = tc.load_labels(cfg["model_fps"])
    train_rows, val_rows = tc.split_fold(data, args.fold, smoke=False, scheme=tc.label_scheme(cfg))
    mb, nw = cfg["micro_batch"], cfg["num_workers"]
    codes = tc.epoch_codes(len(train_rows), args.seed, 0, cfg["hflip_p"])[:args.batches * mb]
    stems = sorted({train_rows["stem"].iloc[c // 2] for c in codes})
    tc.check_cache(train_rows[train_rows["stem"].isin(stems)], val_rows.iloc[:0], manifest, partial=True)
    for s in stems:  # warm the OS file cache
        (tc.CACHE_DIR / f"{s}.npz").read_bytes()
    print(f"config {args.config} (sha256 {cfg_hash[:12]}...); fold {args.fold}; {args.batches} batches x {mb} = "
          f"{len(codes)} windows from {len(stems)} clips; num_workers {nw}; prefetch_factor default (2); "
          f"pin_memory True (inactive: no visible GPU); persistent_workers True; cpu_count {os.cpu_count()}")
    mean, std = list(proc.image_mean), list(proc.image_std)
    dl0 = tc.dark_luma_of(cfg)
    settings = [("standard", "standard", None, nw), ("cctv", "cctv", None, nw),
                ("cctv + night_guard", "cctv", dl0, nw), ("cctv_v2", "cctv_v2", dl0, nw),
                ("standard", "standard", None, 1), ("cctv + night_guard", "cctv", dl0, 1), ("cctv_v2", "cctv_v2", dl0, 1)]
    if args.only:
        settings = [s for s in settings if s[0] in args.only.split(",")]
    if args.config_workers_only:
        settings = [s for s in settings if s[3] == nw]
    res = []
    for name, aug, dl, w in settings:
        if dl is None and name != "standard" and name != "cctv":
            tc.fail(f"{name} needs night_guard dark_luma in the config")
        ds = tc.WindowDataset(train_rows["stem"], train_rows["k0"], train_rows["target"], None, mean, std,
                              cfg["frames"], cfg["model_fps"], cfg["lru_size"], aug, cfg["cctv_aug"], dl)
        torch_seed = args.seed
        import torch
        torch.manual_seed(torch_seed)  # worker seeds derive from the base seed, as in training
        total, steady, n = run(ds, codes, mb, w)
        res.append((name, w, total, steady))
        print(f"{name:<20} workers {w}: {n} windows; {total:6.1f} windows/s incl. worker start, {steady:6.1f} "
              f"windows/s steady", flush=True)
    base = {w: s for n_, w, _, s in res if n_ == "standard"}
    print("\nratio to standard (steady): " + "; ".join(f"{n_} workers {w}: {s / base[w]:.3f}"
                                                     for n_, w, _, s in res if n_ != "standard" and w in base))


if __name__ == "__main__":
    main()
