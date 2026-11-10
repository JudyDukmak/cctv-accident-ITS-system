"""CPU unit test of the "cctv" augmentation profile in train_classifier.py against the specification in
docs/Training_Configuration_Classifier.docx, Section 6 (Ablation 1). No model, no GPU, no training.

Checks, on N training windows of one fold drawn through the real pipeline (epoch_codes -> WindowDataset.load)
with a fixed seed:
  1. config cctv_aug == specification values;
  2. empirical probability of each degradation (|p_hat - 0.5| <= TOL; grayscale among colour-shifted windows
     |p_hat - 0.1| <= TOL) and of the horizontal flip;
  3. every drawn parameter lies inside its specified range (exact values, recorded from the generator);
  4. all 16 frames of a window got the same parameters (per-frame cv2 calls recorded; plus a black-box test:
     16 identical input frames must give 16 identical output frames with every degradation forced on);
  5. frame-rate drop: frame indices equal an independent exact-fraction "last frame a camera at f fps shows";
  6. order of the degradations in the code, and whether applying hflip 2nd (code) instead of last (spec)
     changes any pixel;
  7. validation datasets are not augmented (default aug "none", validation codes have flip 0).

Usage: .venv-ml\\Scripts\\python.exe test_cctv_aug.py [--config configs\\videomae_v2_C.json] [--fold 0] [--n 2000]
"""

import argparse
import inspect
import json
import math
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
import train_classifier as tc  # noqa: E402

# Section 6 of Training_Configuration_Classifier.docx (one draw per window, same for all 16 frames)
SPEC = {"p": 0.5, "fps_drop": [4, 5, 6], "blur_sigma": [0.3, 1.5], "downscale": [0.30, 0.70],
        "brightness": [0.5, 1.3], "contrast": [0.7, 1.3], "saturation": [0.6, 1.3], "hue": 0.02,
        "grayscale_p": 0.1, "jpeg_quality": [15, 40]}
SPEC_ORDER = ["fps", "blur", "downscale", "colour", "jpeg", "hflip"]
HFLIP_P = 0.5
TOL = 0.03


class RecRng:
    """Delegates to a numpy Generator and records (method, args, result) of every draw."""

    def __init__(self, seed: int) -> None:
        self.g = np.random.default_rng(seed)
        self.log: list[tuple] = []

    def _rec(self, name, args, out):
        self.log.append((name, args, out))
        return out

    def random(self):
        return self._rec("random", (), self.g.random())

    def uniform(self, lo, hi):
        return self._rec("uniform", (float(lo), float(hi)), self.g.uniform(lo, hi))

    def integers(self, lo, hi):
        return self._rec("integers", (int(lo), int(hi)), self.g.integers(lo, hi))

    def choice(self, a, size=None, replace=True):
        if size is None:
            return self._rec("choice", (tuple(a),) if not isinstance(a, int) else (a,), self.g.choice(a))
        return self._rec("choice", (a, size, replace), self.g.choice(a, size=size, replace=replace))


class Hooks:
    """Records per-frame cv2 calls and cache reads of the current window."""

    def __init__(self) -> None:
        self.reset()
        self._orig = {"GaussianBlur": cv2.GaussianBlur, "resize": cv2.resize, "imencode": cv2.imencode,
                      "read": tc.read_cached_frame}

    def reset(self) -> None:
        self.blur, self.resize, self.jpeg, self.read = [], [], [], []

    def install(self) -> None:
        o = self._orig

        def blur(src, ksize, sigmaX, *a, **k):
            self.blur.append((tuple(ksize), float(sigmaX), a, k))
            return o["GaussianBlur"](src, ksize, sigmaX, *a, **k)

        def resize(src, dsize, *a, **k):
            self.resize.append((tuple(dsize), k.get("interpolation")))
            return o["resize"](src, dsize, *a, **k)

        def imencode(ext, img, params=None):
            if ext == ".jpg":
                self.jpeg.append(tuple(params))
            return o["imencode"](ext, img, params) if params is not None else o["imencode"](ext, img)

        def read(z, k):
            self.read.append(int(k))
            return o["read"](z, k)

        cv2.GaussianBlur, cv2.resize, cv2.imencode, tc.read_cached_frame = blur, resize, imencode, read

    def uninstall(self) -> None:
        o = self._orig
        cv2.GaussianBlur, cv2.resize, cv2.imencode, tc.read_cached_frame = (o["GaussianBlur"], o["resize"],
                                                                          o["imencode"], o["read"])


def expected_fps_indices(k0: int, frames: int, f: int, model_fps: int) -> list[int]:
    """Exact fractions: frame i at t = (k0 + i) / model_fps shows the camera frame at floor(t f) / f, i.e. the
    last frame a camera running at f fps has produced by t; the cache holds the model_fps grid, so the grid frame
    at or before that camera time is read."""
    out = []
    for i in range(frames):
        t = Fraction(k0 + i, model_fps)
        t_cam = Fraction(math.floor(t * f), f)
        out.append(math.floor(t_cam * model_fps))
    return out


def op_names(applied: list[str]) -> list[str]:
    return [a.split()[0] for a in applied]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "videomae_v2_C.json")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0, help="seed of the window draw and of the augmentation generator")
    ap.add_argument("--n-flip", type=int, default=300, help="windows for the hflip-order comparison")
    ap.add_argument("--n-same", type=int, default=200, help="windows for the identical-frames black-box test")
    ap.add_argument("--dark-luma", type=float, default=None,
                    help="night guard checks with this dark_luma (default: the config's night_guard, else skipped)")
    ap.add_argument("--hashes-before", type=Path,
                    default=REPO_ROOT / "runs" / "aug_preview" / "C_v2_cctv" / "config_hashes_before.txt",
                    help="config hashes recorded before the night guard code change")
    ap.add_argument("--golden", type=Path, default=GOLDEN_DEFAULT,
                    help="output hashes of the standard / cctv / cctv+guard profiles recorded before the cctv_v2 change")
    ap.add_argument("--write-golden", action="store_true", help="record --golden with the current code and exit")
    ap.add_argument("--v2-only", action="store_true", help="only the cctv_v2, golden and config-hash checks")
    ap.add_argument("--golden-v2", type=Path, default=GOLDEN_V2_DEFAULT,
                    help="per-window cctv_v2 outputs recorded before the night50 change (dark_luma 17.9)")
    ap.add_argument("--write-golden-v2", action="store_true", help="record --golden-v2 with the current code and exit")
    args = ap.parse_args()

    cfg, cfg_hash = tc.load_config(args.config.resolve())
    c = cfg["cctv_aug"]
    if args.dark_luma is None:
        args.dark_luma = tc.dark_luma_of(cfg)
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    print(f"config {args.config} (sha256 {cfg_hash[:12]}...); fold {args.fold}; n {args.n}; seed {args.seed}")

    # 1. config vs specification
    diff = {k: (c.get(k), v) for k, v in SPEC.items() if c.get(k) != v}
    check("config cctv_aug == spec", not diff and set(c) == set(SPEC), f"differences {diff}" if diff else "identical")

    # windows: the first n codes of epoch 0 of the real training draw (seeded shuffle + flips)
    data, manifest, _ = tc.load_labels(cfg["model_fps"])
    train_rows, val_rows = tc.split_fold(data, args.fold, smoke=False, scheme=tc.label_scheme(cfg))
    codes = tc.epoch_codes(len(train_rows), args.seed, 0, cfg["hflip_p"])[:args.n]
    tc.check_cache(train_rows.iloc[[cd // 2 for cd in codes]], val_rows.iloc[:0], manifest, partial=True)
    ds = tc.WindowDataset(train_rows["stem"], train_rows["k0"], train_rows["target"], None, [0, 0, 0], [1, 1, 1],
                          cfg["frames"], cfg["model_fps"], cfg["lru_size"], "cctv", c)
    T, W = cfg["frames"], 224
    if args.write_golden_v2:
        if args.golden_v2.exists():
            tc.fail(f"{args.golden_v2} exists; not overwritten")
        args.golden_v2.parent.mkdir(parents=True, exist_ok=True)
        g = v2_window_outputs(train_rows, codes, cfg, args.seed, GOLDEN_V2_DARK_LUMA)
        args.golden_v2.write_text(json.dumps({"dark_luma": GOLDEN_V2_DARK_LUMA, "CCTV_V2": tc.CCTV_V2,
                                              "windows": g}) + "\n", encoding="utf-8")
        print(f"wrote {len(g)} cctv_v2 windows (dark_luma {GOLDEN_V2_DARK_LUMA}) to {args.golden_v2}")
        return
    if args.write_golden:
        if args.golden.exists():
            tc.fail(f"{args.golden} exists; not overwritten")
        args.golden.parent.mkdir(parents=True, exist_ok=True)
        g = golden_outputs(train_rows, codes, cfg, args.seed)
        args.golden.write_text(json.dumps(g, indent=0) + "\n", encoding="utf-8")
        print(f"wrote {len(g)} output hashes to {args.golden}")
        return
    if args.v2_only:
        v2_checks(ds, codes, args, cfg, check)
        night50_checks(train_rows, codes, cfg, args, check)
        golden_check(train_rows, codes, cfg, args, check)
        hash_checks(args, check)
        val_check(check)
        finish(results, args, cfg_hash)
        return

    hooks = Hooks()
    hooks.install()
    rng = RecRng(args.seed)
    fired = Counter()
    n_degr = Counter()  # degradations (excluding hflip) per window
    n_colour = n_gray = 0
    values = defaultdict(list)
    per_frame_bad = Counter()
    order_bad = 0
    fps_bad = 0
    try:
        for j, code in enumerate(codes):
            row, flip = divmod(code, 2)
            hooks.reset()
            start = len(rng.log)
            _, applied = ds.load(row, bool(flip), "cctv", rng)
            draws = rng.log[start:]
            ops = op_names(applied)
            for o in set(ops):
                fired[o] += 1
            n_degr[len([o for o in ops if o != "hflip"])] += 1
            # order of the labels in `applied`: load() inserts "hflip" at position 0, the others are appended in
            # execution order (execution: frame selection (fps) -> flip -> blur -> downscale -> colour -> jpeg)
            label_order = ["hflip", "fps", "blur", "downscale", "colour", "jpeg"]
            pos = [label_order.index(o) for o in ops]
            order_bad += pos != sorted(pos)
            # exact parameters from the generator
            for name, a, out in draws:
                if name == "uniform":
                    key = {tuple(SPEC["blur_sigma"]): "blur_sigma", tuple(SPEC["downscale"]): "downscale",
                           tuple(SPEC["brightness"]): "brightness", tuple(SPEC["contrast"]): "contrast",
                           tuple(SPEC["saturation"]): "saturation", (-SPEC["hue"], SPEC["hue"]): "hue"}.get(a)
                    values[key or f"uniform{a}"].append(float(out))
                elif name == "integers":
                    values["jpeg_quality" if a == (SPEC["jpeg_quality"][0], SPEC["jpeg_quality"][1] + 1)
                           else f"integers{a}"].append(int(out))
                elif name == "choice":
                    values["fps_drop" if list(a[0]) == SPEC["fps_drop"] else f"choice{a}"].append(int(out))
            if "colour" in ops:
                n_colour += 1
                n_gray += any(a.endswith(" gray") for a in applied)
            # per-frame parameters
            if "blur" in ops:
                sig = {s for _, s, _, _ in hooks.blur}
                ks = {k for k, _, _, _ in hooks.blur}
                per_frame_bad["blur"] += not (len(hooks.blur) == T and len(sig) == 1 and ks == {(0, 0)}
                                              and math.isclose(sig.pop(), values["blur_sigma"][-1]))
            else:
                per_frame_bad["blur"] += len(hooks.blur) != 0
            if "downscale" in ops:
                down = [d for d, it in hooks.resize if it == cv2.INTER_AREA]
                up = [d for d, it in hooks.resize if it == cv2.INTER_LINEAR]
                s = values["downscale"][-1]
                exp_small = (max(1, round(W * s)), max(1, round(W * s)))
                per_frame_bad["downscale"] += not (len(down) == T and len(up) == T and set(down) == {exp_small}
                                                   and set(up) == {(W, W)})
                values["downscale_px"].append(exp_small[0])
            else:
                per_frame_bad["downscale"] += len(hooks.resize) != 0
            if "jpeg" in ops:
                per_frame_bad["jpeg"] += not (len(hooks.jpeg) == T and set(hooks.jpeg) == {
                    (cv2.IMWRITE_JPEG_QUALITY, values["jpeg_quality"][-1])})
            else:
                per_frame_bad["jpeg"] += len(hooks.jpeg) != 0
            k0 = int(ds.k0[row])
            if "fps" in ops:
                f = values["fps_drop"][-1]
                fps_bad += hooks.read != expected_fps_indices(k0, T, f, cfg["model_fps"])
            else:
                fps_bad += hooks.read != list(range(k0, k0 + T))
            if (j + 1) % 500 == 0:
                print(f"  {j + 1}/{len(codes)} windows", flush=True)
    finally:
        hooks.uninstall()

    n = len(codes)
    print(f"\n{n} windows; colour-shifted {n_colour}; grayscale {n_gray}")
    print("  cctv (old profile) degradations per window: " + ", ".join(
        f"{k}: {n_degr[k]} ({n_degr[k] / n:.3f})" for k in sorted(n_degr))
        + f"; 3 or more: {sum(v for k, v in n_degr.items() if k >= 3) / n:.3f}")
    # 2. probabilities
    for o in ("fps", "blur", "downscale", "colour", "jpeg", "hflip"):
        p_hat = fired[o] / n
        target = HFLIP_P if o == "hflip" else SPEC["p"]
        check(f"P({o})", abs(p_hat - target) <= TOL, f"{fired[o]}/{n} = {p_hat:.4f} (spec {target}, tol {TOL})")
    p_gray = n_gray / max(1, n_colour)
    check("P(grayscale | colour)", abs(p_gray - SPEC["grayscale_p"]) <= TOL,
          f"{n_gray}/{n_colour} = {p_gray:.4f} (spec {SPEC['grayscale_p']}, tol {TOL})")
    fps_counts = Counter(values["fps_drop"])
    check("fps drawn from {4,5,6}", set(fps_counts) <= set(SPEC["fps_drop"]),
          f"counts {dict(sorted(fps_counts.items()))}")
    # 3. ranges
    for k in ("blur_sigma", "downscale", "brightness", "contrast", "saturation", "jpeg_quality"):
        v = values[k]
        lo, hi = SPEC[k]
        check(f"range {k}", bool(v) and min(v) >= lo and max(v) <= hi,
              f"n {len(v)}, observed [{min(v):.4g}, {max(v):.4g}] within spec [{lo}, {hi}]")
    v = values["hue"]
    check("range hue", bool(v) and min(v) >= -SPEC["hue"] and max(v) <= SPEC["hue"],
          f"n {len(v)}, observed [{min(v):+.4f}, {max(v):+.4f}] within spec +-{SPEC['hue']}")
    jq = Counter(values["jpeg_quality"])
    check("jpeg quality covers integers 15..40 inclusive", set(jq) == set(range(15, 41)),
          f"{len(jq)} distinct values, min {min(jq)}, max {max(jq)}")
    v = values["downscale_px"]
    check("downscaled size in px", min(v) >= round(0.3 * W) and max(v) <= round(0.7 * W),
          f"observed [{min(v)}, {max(v)}] px of {W} (spec 30-70% = {0.3 * W:.1f}-{0.7 * W:.1f})")
    unknown = [k for k in values if k not in SPEC and k != "downscale_px"]
    check("no unexpected random draws", not unknown, f"unexpected {unknown}" if unknown else "none")
    # 4. per-frame parameters
    for o in ("blur", "downscale", "jpeg"):
        check(f"{o}: same parameter for all {T} frames (per-frame cv2 calls)", per_frame_bad[o] == 0,
              f"{per_frame_bad[o]} windows with mismatching calls")
    check("fps drop: frame indices == exact last-camera-frame rule", fps_bad == 0, f"{fps_bad} mismatching windows")
    check("degradations applied in a fixed order", order_bad == 0,
          f"{order_bad} windows out of order; execution order (load + cctv_after_load): fps (frame selection), "
          "hflip, blur, downscale, colour, jpeg; spec order: " + ", ".join(SPEC_ORDER))

    # black box: 16 identical frames, every degradation forced on (p = 1, grayscale as specified)
    forced = dict(c, p=1.0)
    g = np.random.default_rng(args.seed + 1)
    bad_same = 0
    for j, code in enumerate(codes[:args.n_same]):
        row = code // 2
        x, _ = ds.load(row, False, "none", None)
        x16 = np.repeat(x[T // 2][None], T, axis=0)
        out = tc.cctv_after_load(x16.copy(), g, forced, [])
        bad_same += not all(np.array_equal(out[0], out[t]) for t in range(1, T))
    check("identical input frames -> identical output frames (blur, downscale, colour incl. hue/gray, jpeg on)",
          bad_same == 0, f"{bad_same}/{args.n_same} windows with differing frames")

    # 6. hflip position: code flips before blur/downscale/colour/jpeg; spec lists hflip last
    diffs, n_diff, ops_diff, max_abs = [], 0, Counter(), 0
    for j, code in enumerate(codes[:args.n_flip]):
        row = code // 2
        seed = 10_000 + j
        a, applied = ds.load(row, True, "cctv", np.random.default_rng(seed))       # code order: flip 2nd
        b, applied_b = ds.load(row, False, "cctv", np.random.default_rng(seed))    # spec order: flip last
        b = np.ascontiguousarray(b[:, :, ::-1, :])
        d = np.abs(a.astype(np.int16) - b.astype(np.int16))
        if d.any():
            n_diff += 1
            max_abs = max(max_abs, int(d.max()))
            diffs.append((float(d.mean()), float((d > 0).mean()), int(d.max())))
            ops_diff[",".join(o for o in op_names(applied_b))] += 1
    detail = f"{n_diff}/{args.n_flip} windows differ"
    if n_diff:
        md = np.array(diffs)
        detail += (f"; in those: mean |diff| {md[:, 0].mean():.3f} grey levels, share of differing values "
                   f"{100 * md[:, 1].mean():.2f}%, max |diff| {max_abs}; op sets of differing windows "
                   f"{dict(ops_diff.most_common(6))}")
    # accepted deviation (decision B, runs/aug_preview/C_v2_cctv/ORDER_NOTE.txt): reported, not a pass/fail check
    print(f"[INFO] hflip 2nd (code, accepted) vs hflip last (spec table), same draws: {detail}")
    # which single op breaks flip equivariance
    single = {}
    for op in ("blur", "downscale", "colour", "jpeg"):
        nd = 0
        mx = 0
        for j, code in enumerate(codes[:100]):
            row = code // 2
            x, _ = ds.load(row, False, "none", None)
            cfg1 = dict(c, p=1.0)
            gens = [np.random.default_rng(20_000 + j) for _ in range(2)]

            def one(v, gg):
                # same draws as cctv_after_load for a single op: replicate by forcing other ops off
                return tc.cctv_after_load(v, _OnlyOp(gg, op), cfg1, [])
            a = one(np.ascontiguousarray(x[:, :, ::-1, :]), gens[0])
            b = np.ascontiguousarray(one(x, gens[1])[:, :, ::-1, :])
            d = np.abs(a.astype(np.int16) - b.astype(np.int16))
            nd += bool(d.any())
            mx = max(mx, int(d.max()))
        single[op] = (nd, mx)
    print("  flip equivariance by single op (100 windows, op forced on): "
          + "; ".join(f"{k}: {v[0]}/100 windows differ, max |diff| {v[1]}" for k, v in single.items()))

    # 8. night guard
    if args.dark_luma is not None:
        guard_checks(ds, codes, args, check)
    hash_checks(args, check)
    # 9. cctv_v2 and unchanged old profiles
    v2_checks(ds, codes, args, cfg, check)
    night50_checks(train_rows, codes, cfg, args, check)
    golden_check(train_rows, codes, cfg, args, check)
    # 7. validation not augmented
    val_check(check)
    finish(results, args, cfg_hash)


def val_check(check) -> None:
    sig = inspect.signature(tc.WindowDataset.__init__)
    src = inspect.getsource(tc.train)
    val_line = next(line.strip() for line in src.splitlines() if "val_ds = WindowDataset(" in line)
    vsrc = inspect.getsource(tc.validate)
    check("validation not augmented", sig.parameters["aug"].default == "none" and "cfg[\"augmentation\"]" not in
          src.split("val_ds = WindowDataset(")[1].split("train_sampler")[0] and
          "sampler.set(range(0, 2 * len(val_rows), 2))" in vsrc,
          f"WindowDataset aug default {sig.parameters['aug'].default!r}; train(): {val_line} ...(no aug argument); "
          "validate(): codes 2*row (flip 0)")


GOLDEN_DEFAULT = REPO_ROOT / "runs" / "aug_preview" / "C_v2_cctv_v2" / "golden_before_v2.json"
GOLDEN_N = 300
GOLDEN_DARK_LUMA = 17.9


def golden_outputs(train_rows, codes, cfg, seed) -> dict:
    """sha256 of load() and __getitem__ outputs (frames, applied labels, label, weight, row, guard flag) for the first
    GOLDEN_N codes under the standard, cctv and cctv + night guard (dark_luma GOLDEN_DARK_LUMA) profiles; seeded
    generators per window."""
    import hashlib

    def h(*parts) -> str:
        m = hashlib.sha256()
        for p in parts:
            m.update(p.tobytes() if hasattr(p, "tobytes") else str(p).encode())
        return m.hexdigest()

    out = {}
    for tag, aug, dl in (("standard", "standard", None), ("cctv", "cctv", None),
                         ("cctv_guard", "cctv", GOLDEN_DARK_LUMA)):
        ds = tc.WindowDataset(train_rows["stem"], train_rows["k0"], train_rows["target"], None, [0.45] * 3,
                              [0.225] * 3, cfg["frames"], cfg["model_fps"], cfg["lru_size"], aug, cfg["cctv_aug"], dl)
        for j, code in enumerate(codes[:GOLDEN_N]):
            row, flip = divmod(code, 2)
            rng = np.random.default_rng([seed, 9, j]) if aug == "cctv" else None
            x, applied = ds.load(row, bool(flip), aug, rng)
            out[f"load/{tag}/{j}"] = h(x, "|".join(applied))
            ds.rng = np.random.default_rng([seed, 10, j])
            item = ds[code]
            out[f"item/{tag}/{j}"] = h(item[0].numpy(), item[1].numpy(), item[2].numpy(), *item[3:])
    return out


GOLDEN_V2_DEFAULT = REPO_ROOT / "runs" / "aug_preview" / "C_v2_cctv_v2_night50" / "golden_cctv_v2_before.json"
GOLDEN_V2_DARK_LUMA = 17.9  # the cctv_v2 rule as it was before the night50 change


def v2_window_outputs(train_rows, codes, cfg, seed, dark_luma) -> list[dict]:
    """Per window (own generator seeded with (seed, 21, j), flip from the code): luma, plan and output hash."""
    import hashlib
    ds = tc.WindowDataset(train_rows["stem"], train_rows["k0"], train_rows["target"], None, [0, 0, 0], [1, 1, 1],
                          cfg["frames"], cfg["model_fps"], cfg["lru_size"], "cctv_v2", None, dark_luma)
    out = []
    for j, code in enumerate(codes):
        row, flip = divmod(code, 2)
        plans = []

        def rec(rng, dark):
            plans.append(tc.cctv_v2_plan(rng, dark))
            return plans[-1]
        x, applied = ds.load_v2(row, bool(flip), np.random.default_rng([seed, 21, j]), rec)
        out.append({"j": j, "luma": float(applied[0].split()[6]), "plan": plans[0], "applied": applied,
                    "hash": hashlib.sha256(x.tobytes()).hexdigest()})
    return out


def night50_checks(train_rows, codes, cfg, args, check) -> None:
    """Current cctv_v2 (dark_luma args.dark_luma) vs the per-window outputs recorded before the night50 change
    (dark_luma 17.9, same per-window generator seeds)."""
    p = args.golden_v2
    if not p.is_file():
        check("cctv_v2 night50 vs before", False, f"{p} not found")
        return
    before = json.loads(p.read_text(encoding="utf-8"))
    old = {w["j"]: w for w in before["windows"]}
    dl, dl_old = args.dark_luma, before["dark_luma"]
    now = json.loads(json.dumps(v2_window_outputs(train_rows, codes[:len(old)], cfg, args.seed, dl)))  # as stored
    v = V2_SPEC
    n_np = n_both = n_new = 0
    bad = Counter()
    cnt_new = Counter()
    for w in now:
        o = old[w["j"]]
        bad["luma differs"] += abs(w["luma"] - o["luma"]) > 1e-9
        pn, po = w["plan"], o["plan"]
        bad["gates differ"] += pn["drawn"] != po["drawn"]
        if w["luma"] >= dl:  # not protected: must be identical to before
            n_np += 1
            bad["non-protected: output differs"] += w["hash"] != o["hash"] or w["applied"] != o["applied"]
            bad["non-protected: plan differs"] += pn != po
            continue
        bad["protected: more than 1 degradation"] += len(pn["kept"]) > 1
        bad["protected: kept not in drawn"] += any(k not in pn["drawn"] for k in pn["kept"])
        for k, key in (("blur", "blur_sigma_dark"), ("downscale", "downscale_dark")):
            if k in pn:
                bad[f"protected: {k} range"] += not (v[key][0] <= pn[k] <= v[key][1])
        if "jpeg" in pn:
            bad["protected: jpeg range"] += not (v["jpeg_quality_dark"][0] <= pn["jpeg"] <= v["jpeg_quality_dark"][1])
        if "colour" in pn:
            b, ct, s, h, g = pn["colour"]
            bad["protected: colour range"] += not (1.0 <= b <= 1.3 and 1.0 <= ct <= 1.2 and 0.7 <= s <= 1.3
                                                    and abs(h) <= 0.02)
        if o["luma"] < dl_old:  # protected before and now: same draws, narrower blur/downscale/jpeg ranges
            n_both += 1
            bad["dark before and now: kept differs"] += pn["kept"] != po["kept"]
            for k in ("fps", "colour"):
                if k in po:
                    bad[f"dark before and now: {k} differs"] += pn.get(k) != po[k]
            if "blur" in po:  # old dark blur was [0.3, 1.0]: same u mapped to [0.3, 0.7]
                u = (po["blur"] - 0.3) / 0.7
                bad["same draw mapped: blur"] += not math.isclose(pn["blur"], 0.3 + 0.4 * u, abs_tol=1e-9)
            if "downscale" in po:
                u = (po["downscale"] - 0.5) / 0.4
                bad["same draw mapped: downscale"] += not math.isclose(pn["downscale"], 0.7 + 0.2 * u, abs_tol=1e-9)
            if "jpeg" in po:  # one 32-bit draw u: q_old = 25 + floor(26 u / 2^32), q_new = 35 + floor(16 u / 2^32)
                lo_n = math.floor((po["jpeg"] - 25) * 16 / 26)
                hi_n = math.ceil((po["jpeg"] - 24) * 16 / 26) - 1
                bad["same draw mapped: jpeg"] += not (35 + lo_n <= pn["jpeg"] <= 35 + hi_n)
        else:
            n_new += 1
            cnt_new[len(pn["kept"])] += 1
    # integers: the same generator state after a draw from [25, 50] and from [35, 50]
    st_bad = 0
    for s in range(100_000):
        g1, g2 = np.random.default_rng([7, s]), np.random.default_rng([7, s])
        g1.integers(25, 51)
        g2.integers(35, 51)
        st_bad += g1.bit_generator.state != g2.bit_generator.state
    hard = {k: n for k, n in bad.items() if n}
    check(f"cctv_v2 night50: non-protected windows (luma >= {dl:g}) identical to before (output, labels, plan)",
          bad["non-protected: output differs"] == 0 and bad["non-protected: plan differs"] == 0 and n_np > 0,
          f"{n_np} windows, {bad['non-protected: output differs']} outputs and {bad['non-protected: plan differs']} "
          "plans differ")
    check(f"cctv_v2 night50: protected windows (luma < {dl:g}) within ranges and cap; same draws mapped",
          not hard, f"{len(now) - n_np} protected ({n_both} protected before too, {n_new} newly protected; newly "
          f"protected kept 0/1: {cnt_new[0]}/{cnt_new[1]}); violations {hard or 'none'}")
    check("integers(35, 51) leaves the generator in the same state as integers(25, 51) (no extra draws)",
          st_bad == 0, f"{st_bad}/100000 seeds differ")


def golden_check(train_rows, codes, cfg, args, check) -> None:
    if not args.golden.is_file():
        check("old profiles unchanged (golden outputs)", False, f"{args.golden} not found")
        return
    before = json.loads(args.golden.read_text(encoding="utf-8"))
    now = golden_outputs(train_rows, codes, cfg, args.seed)
    bad = Counter(k.rsplit("/", 1)[0] for k in before if now.get(k) != before[k])
    check("standard / cctv / cctv+night_guard outputs unchanged vs before the cctv_v2 change (load + __getitem__)",
          not bad and len(now) == len(before),
          f"{len(before)} hashes ({GOLDEN_N} windows x 3 profiles x 2 paths), mismatches {dict(bad) or 'none'}")


def finish(results, args, cfg_hash) -> None:
    n_fail = sum(not ok for _, ok, _ in results)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    print(json.dumps({"summary": f"{len(results) - n_fail}/{len(results)} passed"}))
    if n_fail:
        sys.exit(1)


V2_SPEC = {"p": 0.3, "max": 2, "max_dark": 1, "fps_drop": [4, 5, 6], "blur_sigma": (0.3, 1.0),
           "downscale": (0.5, 0.9), "brightness": (0.7, 1.3), "contrast": (0.8, 1.2), "saturation": (0.7, 1.3),
           "hue": 0.02, "grayscale_p": 0.1, "jpeg_quality": (25, 50), "brightness_dark": (1.0, 1.3),
           "contrast_dark": (1.0, 1.2), "blur_sigma_dark": (0.3, 0.7), "downscale_dark": (0.7, 0.9),
           "jpeg_quality_dark": (35, 50)}  # the task specifications (v2 + night50), independent of the code's CCTV_V2


def v2_expected_counts(p: float, cap: int, n_ops: int = 5) -> list[float]:
    """P(0..cap degradations kept) when each of n_ops is drawn with p and the excess is cut to cap."""
    pk = [math.comb(n_ops, k) * p ** k * (1 - p) ** (n_ops - k) for k in range(n_ops + 1)]
    return pk[:cap] + [sum(pk[cap:])]


def v2_checks(ds_old, codes, args, cfg, check) -> None:
    """cctv_v2 on the same windows as the other checks, one sequential generator (as in a worker)."""
    dl = args.dark_luma if args.dark_luma is not None else GOLDEN_DARK_LUMA
    T = cfg["frames"]
    code_v2 = {k: (tuple(v) if isinstance(v, list) and k != "fps_drop" else v) for k, v in tc.CCTV_V2.items()}
    spec_map = {"max_degradations": "max", "max_degradations_dark": "max_dark"}
    mism = {k: (code_v2[k], V2_SPEC[spec_map.get(k, k)]) for k in code_v2 if code_v2[k] != V2_SPEC[spec_map.get(k, k)]}
    check("cctv_v2: CCTV_V2 in the code == task specification", not mism and len(code_v2) == len(V2_SPEC),
          f"differences {mism}" if mism else "identical")
    ds = tc.WindowDataset(ds_old.stems, ds_old.k0, ds_old.labels, None, [0, 0, 0], [1, 1, 1], T, cfg["model_fps"],
                          cfg["lru_size"], "cctv_v2", None, dl)
    hooks = Hooks()
    hooks.install()
    rng = RecRng(args.seed)
    drawn_n, kept_n = Counter(), Counter()
    dist = {False: Counter(), True: Counter()}
    capped_drawn, capped_kept = Counter(), Counter()
    bad = Counter()
    vals = defaultdict(list)
    dark_colour_ratio = []
    n_dark = 0
    try:
        for j, code in enumerate(codes):
            row, flip = divmod(code, 2)
            hooks.reset()
            start = len(rng.log)
            x, applied = ds.load(row, bool(flip), "cctv_v2", rng)
            draws = rng.log[start:]
            tok = applied[0].split()
            drawn = [] if tok[2] == "-" else tok[2].split("+")
            kept = [] if tok[4] == "-" else tok[4].split("+")
            luma, dark = float(tok[6]), tok[-1] == "dark"
            k0 = int(ds.k0[row])
            n_dark += dark
            bad["dark flag != (luma < dark_luma)"] += dark != (luma < dl)
            bad["base frames not k0..k0+15"] += hooks.read[:T] != list(range(k0, k0 + T))
            for o in drawn:
                drawn_n[o] += 1
            for o in kept:
                kept_n[o] += 1
            cap = V2_SPEC["max_dark"] if dark else V2_SPEC["max"]
            dist[dark][len(kept)] += 1
            bad["more than cap kept"] += len(kept) > cap
            bad["kept not an ordered subset of drawn"] += [o for o in drawn if o in kept] != kept
            bad["cut although within cap"] += len(drawn) <= cap and kept != drawn
            if len(drawn) > cap:
                for o in drawn:
                    capped_drawn[o] += 1
                    capped_kept[o] += o in kept
            # applied labels: v2, [fps], [hflip], blur, downscale, colour, jpeg (execution order)
            ops = [a.split()[0] for a in applied[1:]]
            bad["applied ops != kept (+ hflip after fps)"] += ops != (
                [o for o in kept if o == "fps"] + (["hflip"] if flip else []) + [o for o in kept if o != "fps"])
            # parameters from the generator
            gates = [d for d in draws if d[0] == "random"][:5]
            bad["not 5 gate draws first"] += len(gates) != 5 or draws[:5] != gates
            uni = [d for d in draws if d[0] == "uniform"]
            geo = {V2_SPEC[k]: k for k in ("blur_sigma", "downscale", "blur_sigma_dark", "downscale_dark")}
            col = [d for d in uni if d[1] not in geo]
            for name, a, v in uni:
                if a in geo:
                    vals[geo[a]].append(v)
                    bad["blur/downscale range does not match dark flag"] += geo[a].endswith("_dark") != dark
            if "colour" in kept:
                if len(col) != 4:
                    bad["colour: not 4 uniform draws"] += 1
                else:
                    (_, ab, b), (_, act, ct), (_, as_, s), (_, ah, h) = col
                    kb, kc = ("brightness_dark", "contrast_dark") if dark else ("brightness", "contrast")
                    bad["colour: draw ranges"] += (ab, act, as_, ah) != (V2_SPEC[kb], V2_SPEC[kc], V2_SPEC["saturation"],
                                                                       (-V2_SPEC["hue"], V2_SPEC["hue"]))
                    vals[kb].append(b)
                    vals[kc].append(ct)
                    vals["saturation"].append(s)
                    vals["hue"].append(h)
                    if dark:
                        bad["dark window darkened or lower contrast"] += b < 1.0 or ct < 1.0
                        dark_colour_ratio.append(tc.window_luma(x) / max(luma, 1e-6))
                gate_gray = [d for d in draws if d[0] == "random"][5:]
                vals["gray"].append(bool(gate_gray) and gate_gray[-1][2] < V2_SPEC["grayscale_p"])
            for name, a, v in draws:
                if name == "integers":
                    kq = "jpeg_quality_dark" if dark else "jpeg_quality"
                    bad["jpeg draw range"] += a != (V2_SPEC[kq][0], V2_SPEC[kq][1] + 1)
                    vals[kq].append(int(v))
                elif name == "choice" and isinstance(a[0], (list, tuple)):
                    bad["fps choice set"] += list(a[0]) != V2_SPEC["fps_drop"]
                    vals["fps_drop"].append(int(v))
            # per-frame parameters
            if "blur" in kept:
                bad["blur per frame"] += not (len(hooks.blur) == T and len({s for _, s, _, _ in hooks.blur}) == 1)
            else:
                bad["blur without draw"] += len(hooks.blur) != 0
            if "downscale" in kept:
                down = {d for d, it in hooks.resize if it == cv2.INTER_AREA}
                bad["downscale per frame"] += not (len(hooks.resize) == 2 * T and len(down) == 1)
                vals["downscale_px_dark" if dark else "downscale_px"].append(next(iter(down))[0])
            else:
                bad["downscale without draw"] += len(hooks.resize) != 0
            if "jpeg" in kept:
                bad["jpeg per frame"] += not (len(hooks.jpeg) == T and len(set(hooks.jpeg)) == 1)
            else:
                bad["jpeg without draw"] += len(hooks.jpeg) != 0
            if "fps" in kept:
                bad["fps indices"] += hooks.read[T:] != expected_fps_indices(k0, T, vals["fps_drop"][-1],
                                                                             cfg["model_fps"])
            else:
                bad["frames read beyond the window"] += len(hooks.read) != T
            if j < 200:  # independent luma of the unaugmented window
                xo, _ = ds_old.load(row, False, "none", None)
                bad["luma label != unaugmented window luma"] += abs(tc.window_luma(xo) - luma) > 0.051
    finally:
        hooks.uninstall()
    n = len(codes)
    print(f"\ncctv_v2: {n} windows, dark_luma {dl:g}, dark windows {n_dark}")
    for o in tc.CCTV_V2_OPS:
        check(f"cctv_v2 P({o} drawn, before the cap)", abs(drawn_n[o] / n - V2_SPEC["p"]) <= TOL,
              f"{drawn_n[o]}/{n} = {drawn_n[o] / n:.4f} (spec {V2_SPEC['p']}, tol {TOL}); kept after the cap "
              f"{kept_n[o]}/{n} = {kept_n[o] / n:.4f}")
    for dark, cap in ((False, V2_SPEC["max"]), (True, V2_SPEC["max_dark"])):
        m = sum(dist[dark].values())
        exp = v2_expected_counts(V2_SPEC["p"], cap)
        obs = [dist[dark][k] / max(1, m) for k in range(cap + 1)]
        tol = TOL if m >= 1000 else 3 * math.sqrt(0.25 / max(1, m))
        check(f"cctv_v2 {'dark' if dark else 'non-dark'} windows: share with " + " / ".join(map(str, range(cap + 1)))
              + " degradations", all(abs(o - e) <= tol for o, e in zip(obs, exp)) and max(dist[dark], default=0) <= cap,
              f"n {m}: " + " / ".join(f"{dist[dark][k]} ({o:.3f}, expected {e:.3f})"
                                      for k, (o, e) in enumerate(zip(obs, exp))) + f"; tol {tol:.3f}")
    allc = dist[False] + dist[True]
    print("  all windows: " + ", ".join(f"{k} degradations {allc[k]} ({allc[k] / n:.3f})" for k in sorted(allc)))
    print("  cap applied: P(kept | drawn) per op " + ", ".join(
        f"{o} {capped_kept[o]}/{capped_drawn[o]} = {capped_kept[o] / max(1, capped_drawn[o]):.3f}"
        for o in tc.CCTV_V2_OPS))
    for k in ("blur_sigma", "downscale", "brightness", "contrast", "saturation", "jpeg_quality", "brightness_dark",
              "contrast_dark", "blur_sigma_dark", "downscale_dark", "jpeg_quality_dark"):
        v = vals[k]
        lo, hi = V2_SPEC[k]
        check(f"cctv_v2 range {k}", (not v and k.endswith("_dark")) or (bool(v) and min(v) >= lo and max(v) <= hi),
              f"n {len(v)}" + (f", observed [{min(v):.4g}, {max(v):.4g}] within [{lo}, {hi}]" if v else ""))
    v = vals["downscale_px_dark"]
    if v:
        check("cctv_v2 protected downscaled size in px", min(v) >= round(0.7 * 224) and max(v) <= round(0.9 * 224),
              f"observed [{min(v)}, {max(v)}] px of 224 (spec 70-90% = 156.8-201.6)")
    v = vals["hue"]
    check("cctv_v2 range hue", bool(v) and max(abs(min(v)), abs(max(v))) <= V2_SPEC["hue"],
          f"n {len(v)}, observed [{min(v):+.4f}, {max(v):+.4f}]")
    v = vals["downscale_px"]
    check("cctv_v2 downscaled size in px", bool(v) and min(v) >= round(0.5 * 224) and max(v) <= round(0.9 * 224),
          f"observed [{min(v)}, {max(v)}] px of 224 (spec 50-90% = 112-201.6)")
    fc = Counter(vals["fps_drop"])
    check("cctv_v2 fps drawn from {4,5,6}", set(fc) <= set(V2_SPEC["fps_drop"]), f"counts {dict(sorted(fc.items()))}")
    g = vals["gray"]
    check("cctv_v2 P(grayscale | colour)", abs(sum(g) / max(1, len(g)) - V2_SPEC["grayscale_p"]) <= 0.05,
          f"{sum(g)}/{len(g)} = {sum(g) / max(1, len(g)):.4f} (spec {V2_SPEC['grayscale_p']}, tol 0.05)")
    r = dark_colour_ratio
    check("cctv_v2 dark windows never darkened, never lower contrast (draws b >= 1, c >= 1)",
          bad["dark window darkened or lower contrast"] == 0,
          f"{len(r)} dark windows with a colour shift; output/input mean luma min {min(r):.3f}, "
          f"median {np.median(r):.3f}" if r else "no dark colour-shifted window")
    hard = {k: v for k, v in bad.items() if v}
    check("cctv_v2 structure (cap, ordered subset, labels, 5 gates first, per-frame parameters, fps indices, luma)",
          not hard, f"violations {hard}" if hard else "0 violations in all " + str(len(bad)) + " structural checks")

    # black box: every frame read is the same frame -> all 16 output frames must be identical
    orig_read = tc.read_cached_frame
    tc.read_cached_frame = lambda z, k: orig_read(z, 8)
    try:
        g2 = np.random.default_rng(args.seed + 5)
        n_bad = n_ops = 0
        for j, code in enumerate(codes[:args.n_same]):
            row, flip = divmod(code, 2)
            out, applied = ds.load(row, bool(flip), "cctv_v2", g2)
            n_ops += len(applied) > 1
            n_bad += not all(np.array_equal(out[0], out[t]) for t in range(1, T))
    finally:
        tc.read_cached_frame = orig_read
    check("cctv_v2 identical input frames -> identical output frames", n_bad == 0,
          f"{n_bad}/{args.n_same} windows differ ({n_ops} had at least one augmentation)")
    # config validation for cctv_v2
    base = json.loads((REPO_ROOT / "configs" / "videomae_v2_C.json").read_text(encoding="utf-8"))
    res = []
    for name, cf, ok in (("cctv_v2 without night_guard", dict(base, augmentation="cctv_v2"), False),
                         ("cctv_v2 + night_guard", dict(base, augmentation="cctv_v2", night_guard={"dark_luma": 17.9}),
                          True)):
        p = GOLDEN_DEFAULT.with_name("_tmp_config.json")
        p.write_text(json.dumps(cf), encoding="utf-8")
        try:
            tc.load_config(p)
            res.append((name, True, ok))
        except SystemExit:
            res.append((name, False, ok))
        finally:
            p.unlink()
    check("load_config: cctv_v2 needs night_guard", all(a == b for _, a, b in res),
          "; ".join(f"{n_}: {'accepted' if a else 'rejected'}" for n_, a, _ in res))


def guard_checks(ds, codes, args, check) -> None:
    """Same windows and per-window generator seeds without and with the night guard (dark_luma)."""
    dl = args.dark_luma
    lo, hi = SPEC["brightness"]
    lumas: list[float] = []
    orig_luma = tc.window_luma

    def rec_luma(x):
        v = orig_luma(x)
        lumas.append(v)
        return v

    tc.window_luma = rec_luma
    n_dark = n_bright = bad_dark = bad_bright = bad_map = bad_seq = 0
    min_b_dark = math.inf
    try:
        for j, code in enumerate(codes):
            row, flip = divmod(code, 2)
            outs = {}
            for tag, d in (("off", None), ("on", dl)):
                ds.dark_luma = d
                lumas.clear()
                rng = RecRng([args.seed, 7, j])
                x, applied = ds.load(row, bool(flip), "cctv", rng)
                outs[tag] = (x, applied, rng.log, list(lumas))
            ds.dark_luma = None
            (x0, a0, l0, _), (x1, a1, l1, lum1) = outs["off"], outs["on"]
            # draw sequence: same number and kind of draws, same values except the brightness one
            kinds0 = [(n, a) for n, a, _ in l0]
            kinds1 = [(n, a) for n, a, _ in l1]
            if len(l0) != len(l1) or [k for k, _ in kinds0] != [k for k, _ in kinds1]:
                bad_seq += 1
                continue
            colour = any(a.startswith("colour") for a in a1)
            if not colour:
                bad_bright += not (np.array_equal(x0, x1) and not lum1)  # guard off: luma not even computed
                continue
            dark = lum1[0] < dl
            ib = next(i for i, (n, a, _) in enumerate(l1) if n == "uniform" and a[1] == hi and a[0] in (lo, 1.0))
            b_off, b_on = l0[ib][2], l1[ib][2]
            others_same = all(v0 == v1 for i, ((_, _, v0), (_, _, v1)) in enumerate(zip(l0, l1)) if i != ib)
            if dark:
                n_dark += 1
                min_b_dark = min(min_b_dark, b_on)
                bad_dark += not (b_on >= 1.0 and l1[ib][1] == (1.0, hi) and others_same)
                # the same uniform u: b_off = lo + (hi - lo) u, b_on = 1 + (hi - 1) u
                bad_map += not math.isclose(b_on, 1.0 + (hi - 1.0) * (b_off - lo) / (hi - lo), abs_tol=1e-9)
            else:
                n_bright += 1
                bad_bright += not (np.array_equal(x0, x1) and b_on == b_off and others_same
                                   and a0 == a1)
    finally:
        tc.window_luma = orig_luma
    check(f"night guard (dark_luma {dl:g}): dark windows get brightness >= 1.0 from [1.0, {hi}]", bad_dark == 0
          and n_dark > 0, f"{n_dark} colour-shifted windows below dark_luma, {bad_dark} bad; lowest factor "
          f"{min_b_dark:.4f}")
    check("night guard: same uniform draw mapped to [1.0, 1.3] (b_on = 1 + 0.3 (b_off - 0.5) / 0.8)", bad_map == 0,
          f"{bad_map} mismatches in {n_dark} dark windows")
    check("night guard: random-draw sequence unchanged (same number and kind of draws)", bad_seq == 0,
          f"{bad_seq} of {len(codes)} windows differ")
    check("night guard: windows not below dark_luma (or no colour shift) are pixel-identical to no guard",
          bad_bright == 0, f"{n_bright} colour-shifted windows above dark_luma + {len(codes) - n_dark - n_bright} "
          f"without colour shift; {bad_bright} differ")
    # config validation
    base = json.loads((REPO_ROOT / "configs" / "videomae_v2_C.json").read_text(encoding="utf-8"))
    rejected = []
    for name, bad in (("standard + night_guard", dict(base, night_guard={"dark_luma": 17.9})),
                      ("extra key", dict(base, augmentation="cctv", night_guard={"dark_luma": 17.9, "x": 1})),
                      ("dark_luma 0", dict(base, augmentation="cctv", night_guard={"dark_luma": 0}))):
        p = Path(args.hashes_before).with_name("_tmp_bad_config.json")
        p.write_text(json.dumps(bad), encoding="utf-8")
        try:
            tc.load_config(p)
            rejected.append(f"{name}: accepted")
        except SystemExit:
            rejected.append(f"{name}: rejected")
        finally:
            p.unlink()
    check("load_config rejects invalid night_guard", all(r.endswith("rejected") for r in rejected), "; ".join(rejected))


def hash_checks(args, check) -> None:
    """Canonical and file sha256 of every config recorded before the change must be unchanged."""
    p = Path(args.hashes_before)
    if not p.is_file():
        check("existing config hashes unchanged", False, f"{p} not found")
        return
    bad, n = [], 0
    for line in p.read_text(encoding="utf-8-sig").splitlines():  # PowerShell Tee-Object writes a BOM
        name, _, canon, _, fsha = line.split()
        f = REPO_ROOT / "configs" / name
        _, h = tc.load_config(f)
        n += 1
        if h != canon or tc.sha256_file(f) != fsha:
            bad.append(name)
    check("existing config hashes unchanged (canonical and file sha256, recorded before the change)", not bad,
          f"{n} configs checked, changed: {bad or 'none'}")


class _OnlyOp:
    """Generator wrapper for cctv_after_load: the gate draw (random() < p) of one op returns 0 (on), the others
    return 1 (off); parameter draws pass through."""
    GATES = ("blur", "downscale", "colour", "jpeg")

    def __init__(self, g, op):
        self.g, self.op, self.gate_i, self.in_colour = g, op, 0, False

    def random(self):
        if self.in_colour:  # grayscale draw inside colour_jitter
            self.in_colour = False
            return self.g.random()
        if self.gate_i >= len(self.GATES):
            return self.g.random()
        name = self.GATES[self.gate_i]
        self.gate_i += 1
        on = name == self.op
        if on and name == "colour":
            self.in_colour = True
        return 0.0 if on else 1.0

    def uniform(self, lo, hi):
        return self.g.uniform(lo, hi)

    def integers(self, lo, hi):
        return self.g.integers(lo, hi)


if __name__ == "__main__":
    main()
